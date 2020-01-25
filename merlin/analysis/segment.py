import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
from skimage import morphology
from skimage import feature
from skimage import filters
import rtree
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree
from scipy.ndimage.morphology import binary_fill_holes

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions()))])


class WatershedSegmentNucleiCV2(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm
    implemented in CV2.

    A tutorial explaining the general scheme of the method can be
    found in  https://opencv-python-tutroals.readthedocs.io/en/latest/
    py_tutorials/py_imgproc/py_watershed/py_watershed.html.

    The watershed segmentation is performed in each z-position
    independently and     combined into 3D objects in a later step

    Since each field of view is analyzed individually, the segmentation
    results should be cleaned in order to merge cells that cross the
    field of view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'WGA'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'DAPI'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])

        # Prepare masks for cv2 watershed
        membraneMask = self._get_membrane_mask(fragmentIndex, watershedIndex)
        nucleiMask = self._get_nuclei_mask(fragmentIndex, watershedIndex)
        watershedMarkers = self._get_watershed_markers(nucleiMask,
                                                       membraneMask)

        # perform watershed in individual z positions
        watershedOutput = self._apply_watershed(fragmentIndex, watershedIndex,
                                                watershedMarkers)

        # combine all z positions in watershed
        watershedCombinedOutput = self._combine_watershed_z_positions(
                                                watershedOutput)

        """
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask =
        watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)
        """

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _get_membrane_mask(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        imageStack = np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                               for z in range(len(self.dataSet.
                                                  get_z_positions()))])
        # generate mask based on edge detection
        """
        edgeMask = np.zeros(imageStack.shape)
        for z in range(len(self.dataSet.get_z_positions())):
            edgeMask[:, :, z] = canny(
                white_tophat(imageStack[:, :, z], selem.disk(10)),
                             sigma=2, use_quantiles=True,
                             low_threshold=0.5, high_threshold=0.8)
            edgeMask[:, :, z] = binary_closing(edgeMask[:, :, z],selem.disk(5))
            edgeMask[:, :, z] = remove_small_objects(
                                edgeMask[:, :, z].astype('bool'),
                                min_size=100, connectivity=1)
            edgeMask[:, :, z] = skeletonize(edgeMask[:, :, z])
        """

        # generate mask based on thresholding
        tresholdingMask = np.zeros(imageStack.shape)
        fineBlockSize = 61
        for z in range(len(self.dataSet.get_z_positions())):
            tresholdingMask[:, :, z] = imageStack[:, :, z] >
            threshold_local(imageStack[:, :, z], fineBlockSize, offset=0)
            
            tresholdingMask[:, :, z] = remove_small_objects(
                imageStack[:, :, z].astype('bool'), min_size=100,
                connectivity=1)
            
            tresholdingMask[:, :, z] = binary_closing(imageStack[:, :, z],
                selem.disk(5))
            
            tresholdingMask[:, :, z] = skeletonize(imageStack[:, :, z])

        # combine masks
        # return edgeMask + thresholdingMask
        return thresholdingMask

    def _get_nuclei_mask(self, fov: int, channelIndex: int) -> np.ndarray:

        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        imageStack = np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                               for z in range(len(self.dataSet.
                                                  get_z_positions()))])

        # generate nuclei mask based on thresholding
        thresholdingMask = np.zeros(imageStack.shape)
        coarseBlockSize = 241
        fineBlockSize = 61
        for z in range(len(self.dataSet.get_z_positions())):
            coarseThresholdingMask = imageStack[:, :, z]
                                        >threshold_local(imageStack[:, :, z],
                                                        coarseBlockSize,
                                                        offset=0)
            fineThresholdingMask = imageStack[:, :, z]
                                        > threshold_local(imageStack[:, :, z],
                                                        fineBlockSize,
                                                        offset=0)
            thresholdingMask[:, :, z] = coarseThresholdingMask*
                                        fineThresholdingMask
            thresholdingMask[:, :, z] = binary_fill_holes(
                                        thresholdingMask[:, :, z])

        # generate border mask, necessary to avoid making a single
        # connected component when using binary_fill_holes below
        borderMask = np.zeros((2048, 2048))
        borderMask[25:2023, 25:2023] = 1

        # TODO - use the image size variable for borderMask

        # generate nuclei mask from hessian, fine
        fineHessianMask = np.zeros(imageStack.shape)
        for z in range(len(self.dataSet.get_z_positions())):
            fineHessian = hessian(imageStack[:, :, z])
            fineHessianMask[:, :, z] = fineHessian == fineHessian.max()
            fineHessianMask[:, :, z] = binary_closing(fineHessianMask[:, :, z],
                                                    selem.disk(5))
            fineHessianMask[:, :, z] = fineHessianMask[:, :, z]*borderMask
            fineHessianMask[:, :, z] = binary_fill_holes(fineHessianMask[:, :, z])

        # generate dapi mask from hessian, coarse
        coarseHessianMask = np.zeros(imageStack.shape)
        for z in range(len(self.dataSet.get_z_positions())):
            coarseHessian = hessian(imageStack[:, :, z] -
                                    white_tophat(imageStack[:, :, z],
                                                 selem.disk(20)))
            coarseHessianMask[:, :, z] = coarseHessian == coarseHessian.max()
            coarseHessianMask[:, :, z] = binary_closing(coarseHessianMask[:, :, z],
                                                      selem.disk(5))
            coarseHessianMask[:, :, z] = coarseHessianMask[:, :, z]*borderMask
            coarseHessianMask[:, :, z] = binary_fill_holes(
                                            coarseHessianMask[:, :, z])

        # combine masks
        nucleiMask = thresholdingMask + fineHessianMask + coarseHessianMask
        return binary_fill_holes(nucleiMask)

    def _get_watershed_markers(self, nucleiMask: np.ndarray,
                                membraneMask: np.ndarray) -> np.ndarray:
        watershedMarker = np.zeros(nucleiMask.shape)

        for z in range(len(self.dataSet.get_z_positions())):

            # generate areas of sure bg and fg, as well as the area of
            # unknown classification
            background = sm.dilation(nucleiMask[:, :, z], sm.selem.disk(15))
            membraneDilated  = sm.dilation(membraneMask[:, :, z].astype('bool'),
                                    sm.selem.disk(10))
            foreground = sm.erosion(nucleiMask[:, :, z]*~membraneDilated,
                            sm.selem.disk(5))
            unknown = background*~foreground

            background = np.uint8(background)*255
            foreground = np.uint8(foreground)*255
            unknown    = np.uint8(unknown)*255

            # Marker labelling
            ret, markers = cv2.connectedComponents(foreground)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+100

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            watershedMarker[:, :, z] = markers

        return watershedMarker

    def _convert_grayscale_to_rgb(self, uint16Image: np.ndarray) -> np.ndarray:
        # cv2 only works in 3D images of 8bit. Make a 3D grayscale by
        # using the same grayscale image in each of the rgb channels
        # code below based on https://stackoverflow.com/questions/
        # 25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv

        # invert image
        uint16Image = 2**16 - uint16Image

        # convert to uint8
        ratio = np.amax(uint16Image) / 256
        uint8Image = (uint16Image / ratio).astype('uint8')

        rgbImage = np.zeros((2048, 2048, 3))
        rgbImage[:, :, 0] = uint8Image
        rgbImage[:, :, 1] = uint8Image
        rgbImage[:, :, 2] = uint8Image
        rgbImage = rgbImage.astype('uint8')

        return rgbImage

    def _apply_watershed(self, fov: int, channelIndex: int,
                                watershedMarkers: np.ndarray) -> np.ndarray:
         warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        imageStack = np.array([warpTask.get_aligned_image(fov, channelIndex, z)
            for z in range(len(self.dataSet.get_z_positions()))])

        watershedOutput = np.zeros(watershedMarkers.shape)
        for z in range(len(self.dataSet.get_z_positions())):
            rgbImage = _convert_grayscale_to_rgb(dapiStack[:, :, z])
            watershedOutput[:, :, z] = cv2.watershed(rgbImage,
                                                    watershedMarkers[:, :, z].
                                                    astype('int32'))
        return watershedOutput

    def _get_overlapping_nuclei(self,watershedZ0: np.ndarray, 
                                    watershedZ1: np.ndarray, n0: int):
        z1NucleiIndexes = np.unique(watershedZ1[watershedZ0 == n0])
        z1NucleiIndexes = z1NucleiIndexes[z1NucleiIndexes>100]

        if z1NucleiIndexes.shape[0] > 0:

            # calculate overlap fraction        
            n0Area = np.count_nonzero(watershedZ0 == n0)
            n1Area = np.zeros(len(z1NucleiIndexes))
            overlapArea = np.zeros(len(z1NucleiIndexes))

            for ii in range(len(z1NucleiIndexes)):
                n1 = z1NucleiIndexes[ii]
                n1Area[ii] = np.count_nonzero(watershedZ1 == n1)
                overlapArea[ii] = np.count_nonzero((watershedZ0 == n0)
                                                    *(watershedZ1 == n1))

            n0OverlapFraction = np.asarray(overlapArea/n0Area)
            n1OverlapFraction = np.asarray(overlapArea/n1Area)
            index = list(range(len(n0OverlapFraction)))

            # select the nuclei that has the highest fraction in n0 and n1
            r1, r2, indexSorted = zip(*sorted(zip(n0OverlapFraction,
                                                  n1OverlapFraction,
                                                  index),
                                      reverse=True))

            if n0OverlapFraction[indexSorted[0]] > 0.2 and
                    n1OverlapFraction[indexSorted[0]] > 0.5:
                return m1NucleiIndexes[indexSorted[0]],
                        n0OverlapFraction[indexSorted[0]],
                        n1OverlapFraction[indexSorted[0]]
            else:
                return False, False, False
        else:
            return False, False, False


    def _combine_watershed_z_positions(self, watershedOutput: np.ndarray)
                                                                -> np.ndarray:
        # TO DO: this implementation is very rough, needs to be improved.
        # good just for testing purposes

        # Initialize empty array with size as watershedOutput array
        watershedCombinedZ = np.zeros(watershedOutput.shape)

        # copy the mask of the section farthest to the coverslip
        watershedCombinedZ[:, :, -1] = watershedOutput[:, :, -1]

        # starting far from coverslip
        for z in range(len(self.dataSet.get_z_positions())-1, 0, -1):
            zNucleiIndex = np.unique(watershedOutput[:, :, z])[
                                    np.unique(watershedOutput[:, :, z])>100]

        for n0 in zNucleiIndex: # for each nuclei N(Z) in Z
            n1,f0,f1 = _get_overlapping_nuclei(watershedCombinedZ[:, :, z],
                                                watershedOutput[:, :, z-1],n0)
            if n1:
                watershedCombinedZ[:, :, z-1][watershedOutput[:, :, z-1] == n1]
                                                                        = n0
        return watershedCombinedZ

class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _write_graph(self, analysisResult, resultName: str,
                     analysisName: str, resultIndex: int = None,
                     subdirectory: str = None) -> None:

        savePath = self.dataSet._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, '.gpickle')
        nx.readwrite.gpickle.write_gpickle(analysisResult, savePath)

    def return_exported_data(self, fragmentIndex) -> nx.Graph:

        savePath = self.dataSet._analysis_result_save_path(
            'cleaned_cells', self.analysisName, fragmentIndex, None, '.gpickle')

        loadedG = nx.readwrite.gpickle.read_gpickle(savePath)

        return loadedG

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, fragmentIndex,
                                               allFOVs, fovBoxes)

        self._write_graph(graph, 'cleaned_cells',
                          self.analysisName, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)
