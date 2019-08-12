import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import networkx
import rtree
from shapely import geometry
from shapely.ops import unary_union
from typing import List, Dict
from scipy.spatial import cKDTree
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas


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

    def get_feature_database(self):
        return(spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self))


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
            self.cellPolygons = [geometry.Polygon(x).buffer(0)
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
        intersectGraphEdges = [[i, i] for i in range(len(polygonList))]
        for i, cell in enumerate(polygonList):
            putativeIntersects = list(polygonIndex.intersection(cell.bounds))

            if len(putativeIntersects) > 0:
                intersectGraphEdges += \
                    [[i, j] for j in putativeIntersects
                     if cell.intersection(
                        polygonList[j]).area > areaThreshold]
                intersectGraphEdges += \
                    [[i, j] for j in putativeIntersects
                     if cell.within(polygonList[j])]
                intersectGraphEdges += \
                    [[i, j] for j in putativeIntersects
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
        rawBoundaries = [geometry.Polygon(x).buffer(0)
                         for x in segmentTask.get_cell_boundaries()]
        rawBoundaries = [x for x in rawBoundaries if x.area > 0]

        overlapComponents = list(networkx.connected_components(
            self._get_intersection_graph(rawBoundaries)))
        mergedCells = [unary_union([rawBoundaries[i] for i in c]).buffer(1)
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


class AssignCellFOV(analysistask.AnalysisTask):

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
        coords = []
        for fov in allFOVs:
            coords.append(self.alignTask.fov_global_extent(fov))

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
        cleanCells = [len(cell.get_bounding_box()) == 4
                      for cell in currentCells]
        currentCells = np.take(currentCells, np.where(cleanCells))[0].tolist()
        return currentCells

    def _secondary_assignments(self, currentFOV: int,
                               secondaryAssignmentDict: Dict):
        currentCells = self._intial_clean(currentFOV)

        secondaryAssignments = [
            [x, secondaryAssignmentDict[currentFOV][x.get_feature_id()]]
            for x in currentCells
            if x.get_feature_id()
            in secondaryAssignmentDict[currentFOV]]
        return secondaryAssignments

    def _construct_spatial_tree(self, tree: rtree.index.Index,
                                cells: List, idToNum: Dict):
        for element in cells:
            tree.insert(idToNum[element.get_feature_id()],
                        element.get_bounding_box(), obj=element.get_fov())

    def get_feature_database(self):
        return(spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self))

    def _run_analysis(self):

        featureDB = spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)

        idx = rtree.index.Index()
        allFOVs = self.dataSet.get_fovs()
        tiledPositions, allFOVBoxes = self._get_fov_boxes()
        numToID = dict()
        idToNum = dict()
        idTracking = 0
        for currentFOV in allFOVs:
            currentUnassigned = self._intial_clean(currentFOV)
            for i in range(len(currentUnassigned)):
                numToID[i+idTracking] = currentUnassigned[i].get_feature_id()
                idToNum[currentUnassigned[i].get_feature_id()] = i + idTracking
            idTracking += len(currentUnassigned)
            self._construct_spatial_tree(idx, currentUnassigned, idToNum)

        secondaryAssignments = dict()
        for currentFOV in allFOVs:
            secondaryAssignments[currentFOV] = dict()

        for currentFOV in allFOVs:
            FOVintersections = sorted([x for x in range(len(allFOVBoxes))
                                       if allFOVBoxes[currentFOV].intersects(
                                                            allFOVBoxes[x])])
            fovTree = self._construct_fov_tree(tiledPositions, FOVintersections)
            currentCells = self._intial_clean(currentFOV)
            for cell in currentCells:
                overlappingCells = idx.intersection(cell.get_bounding_box(),
                                                    objects=True)
                toCheck = []
                for c in overlappingCells:
                    xmin, ymin, xmax, ymax = c.bbox
                    toCheck.append([numToID[c.id],
                                    c.object,
                                    xmin, ymin, xmax, ymax])
                if len(toCheck) == 0:
                    # I dont know if this case will come up but want to guard
                    # against this, it implies
                    # there was an error in the original tree construction
                    print('missing {} from tree'.format(cell.get_feature_id()))
                else:
                    # If a cell does not overlap another cell,
                    # keep it and assign it to an fov based on the fov centroid
                    # it's closest to
                    if len(toCheck) == 1 and \
                            cell.get_feature_id() == toCheck[0][0]:
                        xCenter = (toCheck[0][2] + toCheck[0][4]) / 2
                        yCenter = (toCheck[0][3] + toCheck[0][5]) / 2
                        [d, i] = fovTree.query(np.array([xCenter, yCenter]))
                        assignedFOV = tiledPositions \
                            .loc[FOVintersections, :] \
                            .index.values.tolist()[i]
                        secondaryAssignments[currentFOV][toCheck[0][0]] = \
                            assignedFOV
                    # I dont know if this case will come up but want to guard
                    # against this, it implies there was an error
                    # in the original tree construction
                    elif len(toCheck) == 1 and not \
                            cell.get_feature_id() == toCheck[0][0]:
                        print('encountered unexpected overlap between {} and {}'
                              .format(cell.get_feature_id(), toCheck[0][0]))
                    # If a cell overlaps at least one other cell first check if
                    # all cells are closest to a particular FOV centroid,
                    # then if a cell boundary was determined from that fov
                    # keep it, else choose a cell at random. If overlapping
                    # cells are closest to different fov centroids,
                    # again keep one at random
                    elif len(toCheck) >= 2:
                        toCheckDF = pandas.DataFrame(toCheck,
                                                     columns=['cell ID',
                                                              'initial FOV',
                                                              'xmin', 'ymin',
                                                              'xmax', 'ymax'])
                        toCheckDF['centerX'] = (toCheckDF.loc[:, 'xmin'] +
                                                toCheckDF.loc[:, 'xmax']) / 2
                        toCheckDF['centerY'] = (toCheckDF.loc[:, 'ymin'] +
                                                toCheckDF.loc[:, 'ymax']) / 2
                        [d, i] = fovTree.query(toCheckDF.loc[:, ['centerX',
                                                                 'centerY']],
                                               k=1)
                        assignedFOV = np.array(tiledPositions
                                               .loc[FOVintersections, :]
                                               .index.values.tolist())[i]
                        toCheckDF['assigned FOV'] = assignedFOV
                        if len(np.unique(assignedFOV)) == 1:
                            if len(toCheckDF[toCheckDF['initial FOV'] ==
                                             np.unique(assignedFOV)[0]]) > 0:
                                selected = toCheckDF[
                                    toCheckDF['initial FOV'] == np.unique(
                                        assignedFOV)[0]].sample(n=1)
                            else:
                                selected = toCheckDF.sample(n=1)
                        else:
                            matching = toCheckDF[toCheckDF['initial FOV'] ==
                                                 toCheckDF['assigned FOV']]
                            if len(matching) == 0:
                                selected = toCheckDF.sample(n=1)
                            else:
                                selected = matching.sample(n=1)
                        selectedFOV = selected.loc[:, 'assigned FOV'] \
                            .values.tolist()[0]
                        selectedCellID = selected.loc[:, 'cell ID'] \
                            .values.tolist()[0]
                        secondaryAssignments[
                            currentFOV][selectedCellID] = selectedFOV

        for currentFOV in allFOVs:
            secondaryCellAssignments = \
                self._secondary_assignments(currentFOV, secondaryAssignments)
            allFOV = list(set([x[1] for x in secondaryCellAssignments]))
            for f in allFOV:
                cellsInFOV = [x[0] for x in secondaryCellAssignments
                              if x[1] == f]
                for c in cellsInFOV:
                    c.set_fov(f)
                featureDB.write_features(cellsInFOV, f)







