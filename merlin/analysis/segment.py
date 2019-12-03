import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import rtree
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx
import os

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

class CleanCellBoundaries(analysistask.AnalysisTask):
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

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _get_fov_boxes(self):
        allFOVs = self.dataSet.get_fovs()
        coords = [self.alignTask.fov_global_extent(f) for f in allFOVs]
        coordsDF = pandas.DataFrame(coords,
                                    columns=['minx', 'miny', 'maxx', 'maxy'],
                                    index=allFOVs)
        coordsDF['centerX'] = (coordsDF['minx'] + coordsDF['maxx']) / 2
        coordsDF['centerY'] = (coordsDF['miny'] + coordsDF['maxy']) / 2

        boxes = [geometry.box(x[0], x[1], x[2], x[3]) for x in
                 coordsDF.loc[:, ['minx', 'miny', 'maxx', 'maxy']].values]

        return coordsDF, boxes

    def _construct_fov_tree(self, tiledPositions: pandas.DataFrame,
                            fovIntersections: List):
        return cKDTree(data=tiledPositions.loc[fovIntersections,
                                               ['centerX', 'centerY']].values)

    def _intial_clean(self, currentFOV: int):
        currentCells = self.segmentTask.get_feature_database()\
            .read_features(currentFOV)
        return [cell for cell in currentCells
                if len(cell.get_bounding_box()) == 4 and cell.get_volume() > 0]

    def _append_cells_to_spatial_tree(self, tree: rtree.index.Index,
                                      cells: List, idToNum: Dict):
        for element in cells:
            tree.insert(idToNum[element.get_feature_id()],
                        element.get_bounding_box(), obj=element)

    def _return_overlapping_cells(self, currentCell, cells: List):
        areas = [currentCell.intersection(x) for x in cells]
        overlapping = [cells[i] for i,x in enumerate(areas) if x > 0]
        return overlapping

    def _construct_graph(self):
        G = nx.Graph()
        spatialIndex = rtree.index.Index()
        allFOVs = self.dataSet.get_fovs()
        tiledPositions, allFOVBoxes = self._get_fov_boxes()
        numToID = dict()
        idToNum = dict()
        currentID = 0
        for currentFOV in allFOVs:
            currentUnassigned = self._intial_clean(currentFOV)
            for i in range(len(currentUnassigned)):
                numToID[currentID] = currentUnassigned[i].get_feature_id()
                idToNum[currentUnassigned[i].get_feature_id()] = currentID
                currentID += 1
            self._append_cells_to_spatial_tree(
                spatialIndex, currentUnassigned, idToNum)

        for currentFOV in allFOVs:
            fovIntersections = sorted([i for i, x in enumerate(allFOVBoxes) if
                                       allFOVBoxes[currentFOV].intersects(x)])
            fovTree = self._construct_fov_tree(tiledPositions, fovIntersections)
            currentCells = self._intial_clean(currentFOV)
            for cell in currentCells:
                overlappingCells = spatialIndex.intersection(
                    cell.get_bounding_box(), objects=True)
                toCheck = [x.object for x in overlappingCells]
                toCheck = [x for x in toCheck if x.get_feature_id()
                           not in droppedCells]
                cellsToConsider = self._return_overlapping_cells(cell, toCheck)

                if len(cellsToConsider) == 0:
                    raise Exception(('Missing {} from spatial tree. Spatial ' +
                                     'tree must be malformed.').format(
                        cell.get_feature_id()))

                else:
                    for cellToConsider in cellsToConsider:
                        xmin, ymin, xmax, ymax =\
                            cellToConsider.get_bounding_box()
                        xCenter = (xmin + xmax) / 2
                        yCenter = (ymin + ymax) / 2
                        [d, i] = fovTree.query(np.array([xCenter, yCenter]))
                        assignedFOV = tiledPositions \
                            .loc[fovIntersections, :] \
                            .index.values.tolist()[i]
                        if cellsToConsider[0].get_feature_id() not in G.nodes:
                            G.add_node(cellsToConsider[0].get_feature_id(),
                                       originalFOV=currentFOV,
                                       assignedFOV=assignedFOV)
                    if len(cellsToConsider) > 1:
                        for cellToConsider1 in cellsToConsider:
                            for cellToConsider2 in cellsToConsider:
                                if cellToConsider1.get_feature_id() !=\
                                    cellToConsider2.get_feature_id():
                                    G.add_edge(cellToConsider1.get_feature_id(),
                                               cellToConsider2.get_feature_id())
        return G

    def _remove_overlapping_cells(self, graph):
        connectedComponents = list(nx.connected_components(graph))
        cleanedCells = []
        for component in connectedComponents:
            if len(component) == 1:
                originalFOV = graph.nodes[component[0]]['originalFOV']
                assignedFOV = graph.nodes[component[0]]['assignedFOV']
                cleanedCells.append([component[0], originalFOV, assignedFOV])
            if len(component) > 1:
                compList = []
                for c in component:
                    cellID = c
                    originalFOV = graph.nodes[c]['originalFOV']
                    assignedFOV = graph.nodes[c]['assignedFOV']
                    compList.append([cellID, originalFOV, assignedFOV])
                compDF = pandas.DataFrame(data=compList,
                                          columns=['cell_ID', 'originalFOV',
                                                   'assignedFOV'])
                matchingAssignment = compDF[compDF['originalFOV'] ==
                                            compDF['assignedFOV']]
                if len(matchingAssignment) > 0:
                    selected = matchingAssignment.sample(n=1)
                else:
                    selected = compDF.sample(n=1)
                cleanedCells.append(selected.loc[:,'cell_ID'],
                                    selected.loc[:,'originalFOV'],
                                    selected.loc[:,'assignedFOV'])
        cleanedCellsDF = pandas.DataFrame(cleanedCells,
                                          columns = ['cell_id', 'originalFOV',
                                                     'assignedFOV'])
        return cleanedCellsDF

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'cleanedcells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self) -> None:
        G = self._construct_graph()
        cleanedCells = self._remove_overlapping_cells(G)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'cleanedcells',
                                           analysisTask=self)

class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    self.segmentTask = self.dataSet.load_analysis_task(
        self.parameters['segment_task'])
    self.cleaningTask = self.dataSet.load_analysis_task(
        self.parameters['cleaning_task'])

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
                self.parameters['cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k,g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           x.get_feature_id().isin(cellsToConsider)]
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
