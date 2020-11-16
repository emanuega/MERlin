from abc import abstractmethod
import numpy as np
import uuid
import cv2
from skimage import measure
from typing import List
from typing import Tuple
from typing import Dict
from shapely import geometry
import h5py
import merlin
import pandas
import networkx as nx
import rtree
from scipy.spatial import cKDTree


from merlin.core import dataset
from merlin.core import analysistask


class SpatialFeature(object):

    """
    A spatial feature is a collection of contiguous voxels.
    """

    def __init__(self, boundaryList: List[List[geometry.Polygon]], fov: int,
                 zCoordinates: np.array = None, uniqueID: int = None,
                 label: int = -1) -> None:
        """Create a new feature specified by a list of pixels

        Args:
            boundaryList: a list of boundaries that define this feature.
                The first index of the list corresponds with the z index.
                The second index corresponds with the index of the shape since
                some regions might split in some z indexes.
            fov: the index of the field of view that this feature belongs to.
                The pixel list specifies pixel in the local fov reference
                frame.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
            uniqueID: the uuid of this feature. If no uuid is specified,
                a new uuid is randomly generated.
            label: unused
        """
        self._boundaryList = boundaryList
        self._fov = fov

        if uniqueID is None:
            self._uniqueID = uuid.uuid4().int
        else:
            self._uniqueID = uniqueID

        if zCoordinates is not None:
            self._zCoordinates = zCoordinates
        else:
            self._zCoordinates = np.arange(len(boundaryList))

    @staticmethod
    def feature_from_label_matrix(labelMatrix: np.ndarray, fov: int,
                                  transformationMatrix: np.ndarray = None,
                                  zCoordinates: np.ndarray = None,
                                  label: int = -1):
        """Generate a new feature from the specified label matrix.

        Args:
            labelMatrix: a 3d matrix indicating the z, x, y position
                of voxels that contain the feature. Voxels corresponding
                to the feature have a value of True while voxels outside of the
                feature should have a value of False.
            fov: the index of the field of view corresponding to the
                label matrix.
            transformationMatrix: a 3x3 numpy array specifying the
                transformation from fov to global coordinates. If None,
                the feature coordinates are not transformed.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
        Returns: the new feature
        """

        boundaries = [SpatialFeature._extract_boundaries(x)
                      for x in labelMatrix]

        if transformationMatrix is not None:
            boundaries = [SpatialFeature._transform_boundaries(
                x, transformationMatrix) for x in boundaries]

        return SpatialFeature([SpatialFeature._remove_invalid_boundaries(
            SpatialFeature._remove_interior_boundaries(
                [geometry.Polygon(x) for x in b if len(x) > 2]))
                               for b in boundaries], fov, zCoordinates)

    @staticmethod
    def _extract_boundaries(labelMatrix: np.ndarray) -> List[np.ndarray]:
        """Determine the boundaries of the feature indicated in the
        label matrix.

        Args:
            labelMatrix: a 2 dimensional numpy array indicating the x, y
                position of pixels that contain the feature.
        Returns: a list of n x 2 numpy arrays indicating the x, y coordinates
            of the boundaries where n is the number of boundary coordinates
        """
        boundaries = measure.find_contours(np.transpose(labelMatrix), 0.9,
                                           fully_connected='high')
        return boundaries

    @staticmethod
    def _transform_boundaries(
            boundaries: List[np.ndarray],
            transformationMatrix: np.ndarray) -> List[np.ndarray]:

        transformedList = []
        for b in boundaries:
            reshapedBoundaries = np.reshape(
                b, (1, b.shape[0], 2)).astype(np.float)
            transformedBoundaries = cv2.transform(
                reshapedBoundaries, transformationMatrix)[0, :, :2]
            transformedList.append(transformedBoundaries)

        return transformedList

    @staticmethod
    def _remove_interior_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        goodPolygons = []

        for p in inPolygons:
            if not any([pTest.contains(p)
                        for pTest in inPolygons if p != pTest]):
                goodPolygons.append(p)

        return goodPolygons

    @staticmethod
    def _remove_invalid_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        return [p for p in inPolygons if p.is_valid]

    def set_fov(self, newFOV: int) -> None:
        """Update the FOV for this spatial feature.

        Args:
            nowFOV: the new FOV index
        """
        self._fov = newFOV

    def get_fov(self) -> int:
        return self._fov

    def get_boundaries(self) -> List[List[geometry.Polygon]]:
        return self._boundaryList

    def get_feature_id(self) -> int:
        return self._uniqueID

    def get_z_coordinates(self) -> np.ndarray:
        return self._zCoordinates

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the 2d box that contains all boundaries in all z plans of this
        feature.

        Returns:
            a tuple containing (x1, y1, x2, y2) coordinates of the bounding box
        """
        boundarySet = []
        for f in self.get_boundaries():
            for b in f:
                boundarySet.append(b)

        multiPolygon = geometry.MultiPolygon(boundarySet)
        return multiPolygon.bounds

    def get_volume(self) -> float:
        """Get the volume enclosed by this feature.

        Returns:
            the volume represented in global coordinates. If only one z
            slice is present for the feature, the z height is taken as 1.
        """
        boundaries = self.get_boundaries()

        zPos = np.array(self._zCoordinates)
        if len(zPos) > 1:
            zDiff = np.diff(zPos)
            zNum = np.array([[x, x + 1] for x in range(len(zPos) - 1)])
            areas = np.array([np.sum([y.area for y in x]) if len(x) > 0
                              else 0 for x in boundaries])
            totalVolume = np.sum([np.mean(areas[zNum[x]]) * zDiff[x]
                                  for x in range(zNum.shape[0])])
        else:
            totalVolume = np.sum([y.area for x in boundaries for y in x])

        return totalVolume

    def intersection(self, intersectFeature) -> float:

        intersectArea = 0
        for p1Set, p2Set in zip(self.get_boundaries(),
                                intersectFeature.get_boundaries()):
            for p1 in p1Set:
                for p2 in p2Set:
                    intersectArea += p1.intersection(p2).area

        return intersectArea

    def is_contained_within_boundary(self, inFeature) -> bool:
        """Determine if any part of this feature is contained within the
        boundary of the specified feature.

        Args:
            inFeature: the feature whose boundary should be checked whether
                it contains this feature
        Returns:
            True if inFeature contains pixels that are within inFeature,
                otherwise False. This returns false if inFeature only shares
                a boundary with this feature.
        """
        if all([b1.disjoint(b2) for b1List, b2List in zip(
                    self.get_boundaries(), inFeature.get_boundaries())
                for b1 in b1List for b2 in b2List]):
            return False

        for b1List, b2List in zip(
                self.get_boundaries(), inFeature.get_boundaries()):
            for b1 in b1List:
                for b2 in b2List:
                    x, y = b1.exterior.coords.xy
                    for p in zip(x, y):
                        if geometry.Point(p).within(b2):
                            return True

        return False

    def equals(self, testFeature) -> bool:
        """Determine if this feature is equivalent to testFeature

        Args:
            testFeature: the feature to test equivalency
        Returns:
            True if this feature and testFeature are equivalent, otherwise
                false
        """
        if self.get_fov() != testFeature.get_fov():
            return False
        if self.get_feature_id() != testFeature.get_feature_id():
            return False
        if not np.array_equal(self.get_z_coordinates(),
                              testFeature.get_z_coordinates()):
            return False

        if len(self.get_boundaries()) != len(testFeature.get_boundaries()):
            return False
        for b, bIn in zip(self.get_boundaries(), testFeature.get_boundaries()):
            if len(b) != len(bIn):
                return False
            for x, y in zip(b, bIn):
                if not x.equals(y):
                    return False

        return True

    def contains_point(self, point: geometry.Point, zIndex: int) -> bool:
        """Determine if this spatial feature contains the specified point.

        Args:
            point: the point to check
            zIndex: the z-index that the point corresponds to
        Returns:
            True if the boundaries of this spatial feature in the zIndex plane
                contain the given point.
        """
        for boundaryElement in self.get_boundaries()[zIndex]:
            if boundaryElement.contains(point):
                return True

        return False

    def contains_positions(self, positionList: np.ndarray) -> np.ndarray:
        """Determine if this spatial feature contains the specified positions

        Args:
            positionList: a N x 3 numpy array containing the (x, y, z)
                positions for N points where x and y are spatial coordinates
                and z is the z index. If z is not an integer it is rounded
                to the nearest integer.
        Returns:
            a numpy array of booleans containing true in the i'th index if
                the i'th point provided is in this spatial feature.
        """
        boundaries = self.get_boundaries()
        positionList[:, 2] = np.round(positionList[:, 2])

        containmentList = np.zeros(positionList.shape[0], dtype=np.bool)

        for zIndex in range(len(boundaries)):
            currentIndexes = np.where(positionList[:, 2] == zIndex)[0]
            currentContainment = [self.contains_point(
                geometry.Point(x[0], x[1]), zIndex)
                for x in positionList[currentIndexes]]
            containmentList[currentIndexes] = currentContainment

        return containmentList

    def get_overlapping_features(self, featuresToCheck: List['SpatialFeature']
                                 ) -> List['SpatialFeature']:
        """ Determine which features within the provided list overlap with this
        feature.

        Args:
            featuresToCheck: the list of features to check for overlap with
                this feature.
        Returns: the features that overlap with this feature
        """
        areas = [self.intersection(x) for x in featuresToCheck]
        overlapping = [featuresToCheck[i] for i, x in enumerate(areas) if x > 0]
        benchmark = self.intersection(self)
        contained = [x for x in overlapping if
                     x.intersection(self) == benchmark]
        if len(contained) > 1:
            overlapping = []
        else:
            toReturn = []
            for c in overlapping:
                if c.get_feature_id() == self.get_feature_id():
                    toReturn.append(c)
                else:
                    if c.intersection(self) != c.intersection(c):
                        toReturn.append(c)
            overlapping = toReturn

        return overlapping

    def to_json_dict(self) -> Dict:
        return {
            'fov': self._fov,
            'id': self._uniqueID,
            'z_coordinates': self._zCoordinates.tolist(),
            'boundaries': [[geometry.mapping(y) for y in x]
                           for x in self.get_boundaries()]
        }

    @staticmethod
    def from_json_dict(jsonIn: Dict):
        boundaries = [[geometry.shape(y) for y in x]
                      for x in jsonIn['boundaries']]

        return SpatialFeature(boundaries,
                              jsonIn['fov'],
                              np.array(jsonIn['z_coordinates']),
                              jsonIn['id'])


class SpatialFeatureDB(object):

    """A database for storing spatial features."""

    def __init__(self, dataSet, analysisTask):
        self._dataSet = dataSet
        self._analysisTask = analysisTask

    @abstractmethod
    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        """Write the features into this database.

        If features already exist in the database with feature IDs equal to
        those in the provided list, an exception is raised.

        Args:
            features: a list of features
            fov: the fov of the features if all feature correspond to the same
                fov. If the features correspond to different fovs, fov
                should be None
        """
        pass

    @abstractmethod
    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        """Read the features in this database

        Args:
            fov: if not None, only the features associated with the specified
                fov are returned
        """
        pass

    @abstractmethod
    def empty_database(self, fov: int = None) -> None:
        """Remove all features from this database.

        Args:
            fov: index of the field of view. If specified, only features
                corresponding to the specified fov will be removed.
                Otherwise all barcodes will be removed.
        """
        pass


class HDF5SpatialFeatureDB(SpatialFeatureDB):

    """
    A data store for spatial features that uses a HDF5 file to store the feature
    information.
    """

    def __init__(self, dataSet: dataset.DataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    @staticmethod
    def _save_geometry_to_hdf5_group(h5Group: h5py.Group,
                                     polygon: geometry.Polygon) -> None:
        geometryDict = geometry.mapping(polygon)
        h5Group.attrs['type'] = np.string_(geometryDict['type'])
        h5Group['coordinates'] = np.array(geometryDict['coordinates'])

    @staticmethod
    def _save_feature_to_hdf5_group(h5Group: h5py.Group,
                                    feature: SpatialFeature,
                                    fov: int) -> None:
        featureKey = str(feature.get_feature_id())
        featureGroup = h5Group.create_group(featureKey)
        featureGroup.attrs['id'] = np.string_(feature.get_feature_id())
        featureGroup.attrs['fov'] = fov
        featureGroup.attrs['bounding_box'] = \
            np.array(feature.get_bounding_box())
        featureGroup.attrs['volume'] = feature.get_volume()
        featureGroup['z_coordinates'] = feature.get_z_coordinates()

        for i, bSet in enumerate(feature.get_boundaries()):
            zBoundaryGroup = featureGroup.create_group('zIndex_' + str(i))
            for j, b in enumerate(bSet):
                geometryGroup = zBoundaryGroup.create_group('p_' + str(j))
                HDF5SpatialFeatureDB._save_geometry_to_hdf5_group(
                    geometryGroup, b)

    @staticmethod
    def _load_geometry_from_hdf5_group(h5Group: h5py.Group):
        geometryDict = {'type': h5Group.attrs['type'].decode(),
                        'coordinates': np.array(h5Group['coordinates'])}

        return geometry.shape(geometryDict)

    @staticmethod
    def _load_feature_from_hdf5_group(h5Group):
        zCount = len([x for x in h5Group.keys() if x.startswith('zIndex_')])
        boundaryList = []
        for z in range(zCount):
            zBoundaryList = []
            zGroup = h5Group['zIndex_' + str(z)]
            pCount = len([x for x in zGroup.keys() if x[:2] == 'p_'])
            for p in range(pCount):
                zBoundaryList.append(
                    HDF5SpatialFeatureDB._load_geometry_from_hdf5_group(
                        zGroup['p_' + str(p)]))
            boundaryList.append(zBoundaryList)

        loadedFeature = SpatialFeature(
            boundaryList,
            h5Group.attrs['fov'],
            np.array(h5Group['z_coordinates']),
            int(h5Group.attrs['id']))

        return loadedFeature

    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        if fov is None:
            uniqueFOVs = np.unique([f.get_fov() for f in features])
            for currentFOV in uniqueFOVs:
                currentFeatures = [f for f in features
                                   if f.get_fov() == currentFOV]
                self.write_features(currentFeatures, currentFOV)

        else:
            with self._dataSet.open_hdf5_file(
                    'a', 'feature_data', self._analysisTask, fov, 'features') \
                    as f:
                featureGroup = f.require_group('featuredata')
                featureGroup.attrs['version'] = merlin.version()
                for currentFeature in features:
                    self._save_feature_to_hdf5_group(featureGroup,
                                                     currentFeature,
                                                     fov)

    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        if fov is None:
            featureList = [f for x in self._dataSet.get_fovs()
                           for f in self.read_features(x)]
            return featureList

        featureList = []
        try:
            with self._dataSet.open_hdf5_file('r', 'feature_data',
                                              self._analysisTask, fov,
                                              'features') as f:
                featureGroup = f.require_group('featuredata')
                for k in featureGroup.keys():
                    featureList.append(
                        self._load_feature_from_hdf5_group(featureGroup[k]))
        except FileNotFoundError:
            pass

        return featureList

    def empty_database(self, fov: int = None) -> None:
        if fov is None:
            for f in self._dataSet.get_fovs():
                self.empty_database(f)

        self._dataSet.delete_hdf5_file('feature_data', self._analysisTask,
                                       fov, 'features')

    def read_feature_metadata(self, fov: int = None) -> pandas.DataFrame:
        """ Get the metadata for the features stored within this feature
        database.

        Args:
            fov: an index of a fov to only get the features within the
                specified field of view. If not specified features
                within all fields of view are returned.
        Returns: a data frame containing the metadata, including:
            fov, volume, center_x, center_y, min_x, min_y, max_x, max_y.
            Coordinates are in microns.
        """
        if fov is None:
            finalDF = pandas.concat([self.read_feature_metadata(x)
                                     for x in self._dataSet.get_fovs()], 0)

        else:
            try:
                with self._dataSet.open_hdf5_file('r', 'feature_data',
                                                  self._analysisTask, fov,
                                                  'features') as f:
                    allAttrKeys = []
                    allAttrValues = []
                    for key in f['featuredata'].keys():
                        attrNames = list(f['featuredata'][key].attrs.keys())
                        attrValues = list(f['featuredata'][key].attrs.values())
                        allAttrKeys.append(attrNames)
                        allAttrValues.append(attrValues)

                    columns = list(np.unique(allAttrKeys))
                    df = pandas.DataFrame(data=allAttrValues, columns=columns)
                    finalDF = df.loc[:, ['fov', 'volume']].copy(deep=True)
                    finalDF.index = df['id'].str.decode(encoding='utf-8'
                                                        ).values.tolist()
                    boundingBoxDF = pandas.DataFrame(
                        df['bounding_box'].values.tolist(),
                        index=finalDF.index)
                    finalDF['center_x'] = \
                        (boundingBoxDF[0] + boundingBoxDF[2]) / 2
                    finalDF['center_y'] = \
                        (boundingBoxDF[1] + boundingBoxDF[3]) / 2
                    finalDF['min_x'] = boundingBoxDF[0]
                    finalDF['max_x'] = boundingBoxDF[2]
                    finalDF['min_y'] = boundingBoxDF[1]
                    finalDF['max_y'] = boundingBoxDF[3]
            except FileNotFoundError:
                return pandas.DataFrame()

        return finalDF


class JSONSpatialFeatureDB(SpatialFeatureDB):

    """
    A database for storing spatial features with json serialization.
    """

    def __init__(self, dataSet: dataset.DataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        if fov is None:
            raise NotImplementedError

        try:
            existingFeatures = [SpatialFeature.from_json_dict(x)
                                for x
                                in self._dataSet.load_json_analysis_result(
                    'feature_data', self._analysisTask, fov, 'features')]

            existingIDs = set([x.get_feature_id() for x in existingFeatures])

            for f in features:
                if f.get_feature_id() not in existingIDs:
                    existingFeatures.append(f)

            featuresAsJSON = [f.to_json_dict() for f in existingFeatures]

        except FileNotFoundError:
            featuresAsJSON = [f.to_json_dict() for f in features]

        self._dataSet.save_json_analysis_result(
            featuresAsJSON, 'feature_data', self._analysisTask,
            fov, 'features')

    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        if fov is None:
            raise NotImplementedError

        features = [SpatialFeature.from_json_dict(x)
                    for x in self._dataSet.load_json_analysis_result(
                'feature_metadata', self._analysisTask, fov, 'features')]

        return features

    def empty_database(self, fov: int = None) -> None:
        pass

    @staticmethod
    def _extract_feature_metadata(feature: SpatialFeature) -> Dict:
        boundingBox = feature.get_bounding_box()
        return {'fov': feature.get_fov(),
                'featureID': feature.get_feature_id(),
                'bounds_x1': boundingBox[0],
                'bounds_y1': boundingBox[1],
                'bounds_x2': boundingBox[2],
                'bounds_y2': boundingBox[3],
                'volume': feature.get_volume()}


def simple_clean_cells(cells: List) -> List:
    """
    Removes cells that lack a bounding box or have a volume equal to 0

    Args:
        cells: List of spatial features

    Returns:
        List of spatial features

    """
    return [cell for cell in cells
            if len(cell.get_bounding_box()) == 4 and cell.get_volume() > 0]


def append_cells_to_spatial_tree(tree: rtree.index.Index,
                                 cells: List, idToNum: Dict):
    for element in cells:
        tree.insert(idToNum[element.get_feature_id()],
                    element.get_bounding_box(), obj=element)


def construct_tree(cells: List,
                   spatialIndex: rtree.index.Index = rtree.index.Index(),
                   count: int = 0, idToNum: Dict = dict()):
    """
    Builds or adds to an rtree with a list of cells

    Args:
        cells: list of spatial features
        spatialIndex: an existing rtree to append to
        count: number of existing entries in existing rtree
        idToNum: dict containing feature ID as key, and number in rtree as value

    Returns:
        spatialIndex: an rtree updated with the input cells
        count: number of entries in rtree
        idToNum: dict containing feature ID as key, and number in rtree as value
    """

    for i in range(len(cells)):
        idToNum[cells[i].get_feature_id()] = count
        count += 1
    append_cells_to_spatial_tree(spatialIndex, cells, idToNum)

    return spatialIndex, count, idToNum


def return_overlapping_cells(currentCell, cells: List):
    """
    Determines if there is overlap between a cell of interest and a list of
    other cells. In the event that the cell of interest is entirely contained
    within one of the cells in the cells it is being compared to, an empty
    list is returned. Otherwise, the cell of interest and any overlapping
    cells are returned.
    Args:
        currentCell: A spatial feature of interest
        cells: A list of spatial features to compare to, the spatial feature
               of interest is expected to be in this list

    Returns:
        A list of spatial features including the cell of interest and all
        overlapping cells, or an empty list if the cell of intereset is
        entirely contained within one of the cells it is compared to
    """
    areas = [currentCell.intersection(x) for x in cells]
    overlapping = [cells[i] for i, x in enumerate(areas) if x > 0]
    benchmark = currentCell.intersection(currentCell)
    contained = [x for x in overlapping if
                 x.intersection(currentCell) == benchmark]
    if len(contained) > 1:
        overlapping = []
    else:
        toReturn = []
        for c in overlapping:
            if c.get_feature_id() == currentCell.get_feature_id():
                toReturn.append(c)
            else:
                if c.intersection(currentCell) != c.intersection(c):
                    toReturn.append(c)
        overlapping = toReturn

    return overlapping


def construct_graph(graph, cells, spatialTree, currentFOV, allFOVs, fovBoxes):
    """
    Adds the cells from the current fov to a graph where each node is a cell
    and edges connect overlapping cells.

    Args:
        graph: An undirected graph, either empty of already containing cells
        cells: A list of spatial features to potentially add to graph
        spatialTree: an rtree index containing each cell in the dataset
        currentFOV: the fov currently being added to the graph
        allFOVs: a list of all fovs in the dataset
        fovBoxes: a list of shapely polygons containing the bounds of each fov

    Returns:
        A graph updated to include cells from the current fov
    """

    fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                               fovBoxes[currentFOV].intersects(x)])

    coords = [x.centroid.coords.xy for x in fovBoxes]
    xcoords = [x[0][0] for x in coords]
    ycoords = [x[1][0] for x in coords]
    coordsDF = pandas.DataFrame(data=np.array(list(zip(xcoords, ycoords))),
                                index=allFOVs,
                                columns=['centerX', 'centerY'])
    fovTree = cKDTree(data=coordsDF.loc[fovIntersections,
                                        ['centerX', 'centerY']].values)
    for cell in cells:
        overlappingCells = spatialTree.intersection(
            cell.get_bounding_box(), objects=True)
        toCheck = [x.object for x in overlappingCells]
        cellsToConsider = return_overlapping_cells(
            cell, toCheck)
        if len(cellsToConsider) == 0:
            pass
        else:
            for cellToConsider in cellsToConsider:
                xmin, ymin, xmax, ymax =\
                    cellToConsider.get_bounding_box()
                xCenter = (xmin + xmax) / 2
                yCenter = (ymin + ymax) / 2
                [d, i] = fovTree.query(np.array([xCenter, yCenter]))
                assignedFOV = coordsDF.loc[fovIntersections, :]\
                    .index.values.tolist()[i]
                if cellToConsider.get_feature_id() not in graph.nodes:
                    graph.add_node(cellToConsider.get_feature_id(),
                                   originalFOV=cellToConsider.get_fov(),
                                   assignedFOV=assignedFOV)
            if len(cellsToConsider) > 1:
                for cellToConsider1 in cellsToConsider:
                    if cellToConsider1.get_feature_id() !=\
                            cell.get_feature_id():
                        graph.add_edge(cell.get_feature_id(),
                                       cellToConsider1.get_feature_id())
    return graph


def remove_overlapping_cells(graph):
    """
    Takes in a graph in which each node is a cell and edges connect cells that
    overlap eachother in space. Removes overlapping cells, preferentially
    eliminating the cell that overlaps the most cells (i.e. if cell A overlaps
    cells B, C, and D, whereas cell B only overlaps cell A, cell C only overlaps
    cell A, and cell D only overlaps cell A, then cell A will be removed,
    leaving cells B, C, and D remaining because there is no more overlap
    within this group of cells).
    Args:
        graph: An undirected graph, in which each node is a cell and each
               edge connects overlapping cells. nodes are expected to have
               the following attributes: originalFOV, assignedFOV
    Returns:
        A pandas dataframe containing the feature ID of all cells after removing
        all instances of overlap. There are columns for cell_id, originalFOV,
        and assignedFOV
    """
    connectedComponents = list(nx.connected_components(graph))
    cleanedCells = []
    connectedComponents = [list(x) for x in connectedComponents]
    for component in connectedComponents:
        if len(component) == 1:
            originalFOV = graph.nodes[component[0]]['originalFOV']
            assignedFOV = graph.nodes[component[0]]['assignedFOV']
            cleanedCells.append([component[0], originalFOV, assignedFOV])
        if len(component) > 1:
            sg = nx.subgraph(graph, component)
            verts = list(nx.articulation_points(sg))
            if len(verts) > 0:
                sg = nx.subgraph(graph,
                                 [x for x in component if x not in verts])
            allEdges = [[k, v] for k, v in nx.degree(sg)]
            sortedEdges = sorted(allEdges, key=lambda x: x[1], reverse=True)
            maxEdges = sortedEdges[0][1]
            while maxEdges > 0:
                sg = nx.subgraph(graph, [x[0] for x in sortedEdges[1:]])
                allEdges = [[k, v] for k, v in nx.degree(sg)]
                sortedEdges = sorted(allEdges, key=lambda x: x[1],
                                     reverse=True)
                maxEdges = sortedEdges[0][1]
            keptComponents = list(sg.nodes())
            cellIDs = []
            originalFOVs = []
            assignedFOVs = []
            for c in keptComponents:
                cellIDs.append(c)
                originalFOVs.append(graph.nodes[c]['originalFOV'])
                assignedFOVs.append(graph.nodes[c]['assignedFOV'])
            listOfLists = list(zip(cellIDs, originalFOVs, assignedFOVs))
            listOfLists = [list(x) for x in listOfLists]
            cleanedCells = cleanedCells + listOfLists
    cleanedCellsDF = pandas.DataFrame(cleanedCells,
                                      columns=['cell_id', 'originalFOV',
                                               'assignedFOV'])
    return cleanedCellsDF
