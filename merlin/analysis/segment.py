import cv2
import numpy as np
from skimage import measure
from skimage import segmentation as skiseg
from skimage import morphology
from skimage import feature
from skimage import filters
import rtree
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import segmentation
import pandas
import networkx as nx
import time

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
        seeds = segmentation.separate_merged_seeds(
            segmentation.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = segmentation\
            .prepare_watershed_images(watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = skiseg.watershed(
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
    independently and combined into 3D objects in a later step

    The class can be used to segment either nuclear or cytoplasmic
    compartments. If both the compartment and membrane channels are the
    same, the membrane channel is calculated from the edge transform of
    the provided channel.

    Since each field of view is analyzed individually, the segmentation
    results should be cleaned in order to merge cells that cross the
    field of view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'membrane_channel_name' not in self.parameters:
            self.parameters['membrane_channel_name'] = 'DAPI'
        if 'compartment_channel_name' not in self.parameters:
            self.parameters['compartment_channel_name'] = 'DAPI'

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
        startTime = time.time()

        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # read membrane and compartment  indexes
        membraneIndex = self.dataSet \
                            .get_data_organization() \
                            .get_data_channel_index(
                                self.parameters['membrane_channel_name'])
        compartmentIndex = self.dataSet \
                               .get_data_organization() \
                               .get_data_channel_index(
                                self.parameters['compartment_channel_name'])

        # read membrane and compartment images
        membraneImages = self._read_image_stack(fragmentIndex, membraneIndex)
        compartmentImages = self._read_image_stack(fragmentIndex,
                                                   compartmentIndex)

        # Prepare masks for cv2 watershed
        watershedMarkers = segmentation.get_cv2_watershed_markers(
                            compartmentImages,
                            membraneImages,
                            self.parameters['compartment_channel_name'],
                            self.parameters['membrane_channel_name'])

        # perform watershed in individual z positions
        watershedOutput = segmentation.apply_cv2_watershed(compartmentImages,
                                                           watershedMarkers)

        # combine all z positions in watershed
        watershedCombinedOutput = segmentation \
            .combine_2d_segmentation_masks_into_3d(watershedOutput)

        # get features from mask. This is the slowestart (6 min for the
        # previous part, 15+ for the rest, for a 7 frame Image.
        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedCombinedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedCombinedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                         for z in range(len(self.dataSet.get_z_positions()))])


class MachineLearningSegment(FeatureSavingAnalysisTask):
    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a the specified machine learning
    method. The available method is cellpose (https://github.com/MouseLand/
    cellpose).

    TODO: implement unets / Ilastik
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'method' not in self.parameters:
            self.parameters['method'] = 'cellpose'
        if 'diameter' not in self.parameters:
            self.parameters['diameter'] = 50
        if 'compartment_channel_name' not in self.parameters:
            self.parameters['compartment_channel_name'] = 'DAPI'
        if 'flow_threshold' not in self.parameters:
            self.parameters['flow_threshold'] = 0.5
        if 'cellprob_threshold' not in self.parameters:
            self.parameters['cellprob_threshold'] = 1

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

        # read membrane and compartment indexes
        compartmentIndex = self.dataSet \
                               .get_data_organization() \
                               .get_data_channel_index(
                                self.parameters['compartment_channel_name'])

        # Read images and perform segmentation
        compartmentImages = self._read_image_stack(fragmentIndex,
                                                   compartmentIndex)

        if self.parameters['method'] == 'cellpose':
            segParameters = dict({
                'method': 'cellpose',
                'diameter': self.parameters['diameter'],
                'channel': self.parameters['compartment_channel_name'], 
                'flow_threshold': self.parameters['flow_threshold'],
                'cellprob_threshold': self.parameters['cellprob_threshold']
            })

        segmentationOutput = segmentation.apply_machine_learning_segmentation(
                                compartmentImages, segParameters)

        # combine all z positions in watershed
        watershedCombinedOutput = segmentation \
            .combine_2d_segmentation_masks_into_3d(segmentationOutput)

        # get features from mask. This is the slowestart (6 min for the
        # previous part, 15+ for the rest, for a 7 frame Image.
        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedCombinedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedCombinedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                         for z in range(len(self.dataSet.get_z_positions()))])


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

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle(
            'cleaned_cells', self, fragmentIndex)

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

        self.dataSet.save_graph_as_gpickle(
            graph, 'cleaned_cells', self, fragmentIndex)


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
