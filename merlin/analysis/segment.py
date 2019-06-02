import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import networkx
import rtree
from shapely import geometry
from shapely.ops import unary_union
from typing import List

from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed


class WatershedSegment(analysistask.ParallelAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentations
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
        featureDB = spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
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

        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex))
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)
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


class CleanCellSegmentation(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.boundaryList = None
        self.cellPolygons = None
        self.regionIndex = None

    def get_estimated_memory(self):
        return 2048 

    def get_estimated_time(self):
        return 30 

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def get_cell_boundaries(self):
        if self.boundaryList is None:
            self.boundaryList = self.dataSet.load_numpy_analysis_result(
                        'cell_boundaries', self.get_analysis_name())

        return self.boundaryList

    def get_cell_polygons(self):
        if self.cellPolygons is None:
            self.cellPolygons = [geometry.Polygon(x).buffer(0) \
                    for x in self.get_cell_boundaries()]

        return self.cellPolygons

    def _prepare_region_tree(self):
        if self.regionIndex is None:
            self.regionIndex = rtree.index.Index()

            boundaryPolygons = self.get_cell_polygons()
            for i, cell in enumerate(boundaryPolygons):
                self.regionIndex.insert(i, cell.bounds)

    def get_cell_containing_position(self, x, y):
        self._prepare_region_tree()
        polygons = self.get_cell_polygons()
        intersections = self.regionIndex.intersection([x-1, y-1, x+1, y+1])

        point = geometry.Point(x, y)
        for i in intersections:
            if polygons[i].contains(point):
                return i

        return -1

    def _get_intersection_graph(self, polygonList, areaThreshold=250):
        polygonIndex = rtree.index.Index()
        intersectGraphEdges = [[i,i] for i in range(len(polygonList))]
        for i,cell in enumerate(polygonList):
            putativeIntersects = list(polygonIndex.intersection(cell.bounds))

            if len(putativeIntersects) > 0:
                intersectGraphEdges += \
                        [[i, j] for j in putativeIntersects \
                        if cell.intersection(
                            polygonList[j]).area>areaThreshold] 
                intersectGraphEdges += \
                        [[i, j] for j in putativeIntersects \
                        if cell.within(polygonList[j])] 
                intersectGraphEdges += \
                        [[i, j] for j in putativeIntersects \
                        if polygonList[j].within(cell)] 

            polygonIndex.insert(i, cell.bounds)

        intersectionGraph = networkx.Graph()
        intersectionGraph.add_edges_from(intersectGraphEdges)

        return intersectionGraph

    def _subtract_region(self, region1, region2):
        """Substracts region 2 from region 1 and returns the modified
        region 1.
        """
        return region1.symmetric_difference(region2).difference(region2)

    def _clean_polygon(self, inputPolygon):
        """Cleans the polygon if polygon manipulations resulted in
        a multipolygon or an empty shape. 

        Returns:
            The cleaned polygon. If the multipolygon is passed, this will 
            return the largest polygon within the multipolygon. If a 
            non-polygon shape is passed, this function will return None.
        """
        if inputPolygon.geom_type == 'Polygon':
            return inputPolygon
        elif inputPolygon.geom_type == 'MultiPolygon':
            return inputPolygon[np.argmax([x.area for x in inputPolygon])]
        else:
            return None

    def _run_analysis(self):
        segmentTask = self.dataSet.load_analysis_task(
                self.parameters['segment_task'])

        # The boundaries are filtered to avoid empty boundaries and to
        # prevent intersection in the boundary path.
        rawBoundaries = [geometry.Polygon(x).buffer(0) \
                for x in segmentTask.get_cell_boundaries()]
        rawBoundaries = [x for x in rawBoundaries if x.area>0]

        overlapComponents = list(networkx.connected_components(
            self._get_intersection_graph(rawBoundaries)))
        mergedCells = [unary_union([rawBoundaries[i] for i in c]).buffer(1) \
                for c in overlapComponents]
    
        refinedComponents = self._get_intersection_graph(
                mergedCells, areaThreshold=0)

        cleanedCells = []
        for i, currentCell in enumerate(mergedCells):
            edgesFromCell = refinedComponents.edges(i)

            if currentCell.geom_type == 'Polygon': 
                cleanedCell = geometry.Polygon(currentCell)
                for edge in edgesFromCell:
                    if edge[0] != edge[1] and cleanedCell is not None:
                        if not mergedCells[edge[1]].within(cleanedCell):
                            cleanedCell = self._clean_polygon(
                                    self._subtract_region(
                                    cleanedCell, mergedCells[edge[1]]))
                if cleanedCell is not None:
                    cleanedCells.append(cleanedCell)

        refinedBoundaries = np.array(
                [np.array(x.exterior.coords) for x in cleanedCells])

        self.dataSet.save_numpy_analysis_result(
                refinedBoundaries, 'cell_boundaries',
                self.get_analysis_name())



