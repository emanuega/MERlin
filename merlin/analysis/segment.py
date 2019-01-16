import cv2
import numpy as np
from scipy.ndimage import morphology
import networkx
import rtree
from shapely import geometry
from shapely.ops import unary_union
from starfish.image._segmentation import watershed

from merlin.core import analysistask


class SegmentCells(analysistask.ParallelAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view. 
    
    Since each field of view is analyzed individually, the segmentations
    should be cleaned in order to merge cells that cross the field of 
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'nucleus_threshold' not in self.parameters:
            self.parameters['nucleus_threshold'] = 0.41
        if 'cell_threshold' not in self.parameters:
            self.parameters['cell_threshold'] = 0.08
        if 'nucleus_index' not in self.parameters:
            self.parameters['nucleus_index'] = 17
        if 'cell_index' not in self.parameters:
            self.parameters['cell_index'] = 16
        if 'z_index' not in self.parameters:
            self.parameters['z_index'] = 0

        self.nucleusThreshold = self.parameters['nucleus_threshold']
        self.cellThreshold = self.parameters['cell_threshold']
        self.nucleusIndex = self.parameters['nucleus_index']
        self.cellIndex = self.parameters['cell_index']
        self.zIndex = self.parameters['z_index']

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

    @staticmethod
    def _label_to_regions(inputImage: np.ndarray) -> np.ndarray:
        uniqueLabels = sorted(np.unique(inputImage))[1:]

        def extract_contours(labelImage: np.ndarray, label: int) -> np.ndarray:
            filledImage = morphology.binary_fill_holes(
                    labelImage==label)
            im2, contours, hierarchy = cv2.findContours(
                    filledImage.astype(np.uint8),
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_TC89_KCOS)
            return contours

        return np.array(
                [np.array([x[0] for x in extract_contours(inputImage, i)[0]])
                 for i in uniqueLabels])

    @staticmethod
    def _transform_contours(
            contours: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transforms the coordinates in the contours based on the
        provided transformation.

        Args:
            contours: a n x 2 numpy array specifying the coordinates of the n
                points in the contour
            transform: a 3 x 3 numpy array specifying the transformation
                matrix
        Returns:
            a n x 2 numpy array containing the transformed coordinates
        """
        reshapedContours = np.reshape(
            contours, (1, contours.shape[0], 2)).astype(np.float)
        transformedContours = cv2.transform(
                reshapedContours, transform)[0, :, :2]

        return transformedContours

    def get_cell_boundaries(self):
        boundaryList = []
        for f in self.dataSet.get_fovs():
            currentBoundaries = self.dataSet.load_analysis_result(
                    'cell_boundaries', self.get_analysis_name(), resultIndex=f)
            boundaryList += [x for x in currentBoundaries]

        return boundaryList

    def run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # TODO - extend to 3D
        # TODO - this does not do well with image boundaries. Cell
        # boundaries are not traced past the edge of the field of
        # view
        nucleusImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.nucleusIndex, self.zIndex),
                (int(35), int(35)), 8)
        cellImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.cellIndex, self.zIndex),
                (int(35), int(35)), 8)

        w = watershed._WatershedSegmenter(nucleusImage, cellImage)
        labels = w.segment(
                self.nucleusThreshold, self.cellThreshold, [10, 100000])

        cellContours = self._label_to_regions(labels)
        transformation = globalTask.fov_to_global_transform(fragmentIndex)
        transformedContours = np.array(
                [self._transform_contours(x, transformation)
                 for x in cellContours])

        self.dataSet.save_analysis_result(
                transformedContours, 'cell_boundaries',
                self.get_analysis_name(), resultIndex=fragmentIndex)


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
            self.boundaryList = self.dataSet.load_analysis_result(
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

    def run_analysis(self):
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

        self.dataSet.save_analysis_result(
                refinedBoundaries, 'cell_boundaries',
                self.get_analysis_name())



