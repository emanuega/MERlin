import pytest
import numpy as np
import json
import rtree
import networkx as nx
from shapely import geometry

from merlin.util import spatialfeature


testCoords1 = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
testCoords2 = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]

feature1 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)]], 0)
feature2 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords2)]], 0)
feature3 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)],
                                          [geometry.Polygon(testCoords1)]],
                                         0, zCoordinates=np.array([0, 0.5]))
feature4 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)],
                                          [geometry.Polygon(testCoords2)]],
                                         0, zCoordinates=np.array([0, 0.5]))

p1 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]], 0)
p2 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0.5), (0, 1), (1, 1), (1, 0.5)])]], 0)
p3 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0.5), (0, 1.5), (1, 1.5), (1, 0.5)])]], 0)
p4 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 1), (0, 2), (1, 2), (1, 1)])]], 0)
p5 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(4, 4), (4, 5), (5, 5), (5, 4)])]], 0)
allCells = [p1, p2, p3, p4, p5]


def test_feature_from_label_matrix():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(geometry.Polygon(
        list(zip([2.1, 2.1, 2.0, 1.0, 0.9, 0.9, 1.0, 2.0, 2.1],
                 [2.0, 1.0, 0.9, 0.9, 1.0, 2.0, 2.1, 2.1, 2.0]))))


def test_feature_from_label_matrix_transform():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1
    transformMatrix = np.array([[2.0, 0, 3],
                                [0, 2.5, 1],
                                [0, 0, 1]])

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0, transformMatrix)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(
        geometry.Polygon(list(zip(
            [7.2, 7.2, 7.0, 5.0, 4.8, 4.8, 5.0, 7.0, 7.2],
            [6.0, 3.5, 3.25, 3.25, 3.5, 6.0, 6.25, 6.25, 6.0]))))


@pytest.mark.parametrize('feature, volume',
                         [(feature1, 1), (feature2, 4), (feature3, 0.5)])
def test_feature_get_volume(feature, volume):
    assert feature.get_volume() == volume


@pytest.mark.parametrize('feature', [feature1, feature2, feature3])
def test_feature_equals(feature):
    assert feature.equals(feature)


@pytest.mark.parametrize('feature', [feature1, feature2, feature3])
def test_feature_serialization_to_json(feature):
    featureIn = spatialfeature.SpatialFeature.from_json_dict(
        json.loads(json.dumps(feature.to_json_dict())))

    assert featureIn.equals(feature)


def test_feature_hdf5_db_read_write_delete_one_fov(
        single_task, simple_merfish_data):
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature1, feature2], fov=0)
    readFeatures = featureDB.read_features(fov=0)
    featureDB.empty_database(0)
    readFeatures2 = featureDB.read_features(fov=0)

    assert len(readFeatures) == 2
    if readFeatures[0].get_feature_id() == feature1.get_feature_id():
        f1Index = 0
        f2Index = 1
    else:
        f1Index = 1
        f2Index = 0
    assert readFeatures[f1Index].equals(feature1)
    assert readFeatures[f2Index].equals(feature2)

    assert len(readFeatures2) == 0


def test_feature_hdf5_db_read_write_delete_multiple_fov(
        single_task, simple_merfish_data):
    tempFeature2 = spatialfeature.SpatialFeature(
        [[geometry.Polygon(testCoords2)]], 1)
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature1, tempFeature2])
    readFeatures = featureDB.read_features()
    readFeatures0 = featureDB.read_features(0)
    readFeatures1 = featureDB.read_features(1)
    featureMetadata = featureDB.read_feature_metadata()
    featureMetadata0 = featureDB.read_feature_metadata(0)
    featureMetadata1 = featureDB.read_feature_metadata(1)
    featureDB.empty_database()
    readFeaturesEmpty = featureDB.read_features()
    metaDataEmpty = featureDB.read_feature_metadata()

    assert len(readFeatures0) == 1
    assert len(featureMetadata0) == 1
    assert readFeatures0[0].equals(feature1)
    assert int(featureMetadata0.index[0]) == feature1.get_feature_id()
    assert np.allclose(featureMetadata0.iloc[0][
                           ['fov', 'volume', 'center_x', 'center_y',
                            'min_x', 'min_y', 'max_x', 'max_y']].values,
                       np.array([0, 1, 1.5, 1.5, 1, 1, 2, 2]))

    assert len(readFeatures1) == 1
    assert len(featureMetadata1) == 1
    assert readFeatures1[0].equals(tempFeature2)
    assert int(featureMetadata1.index[0]) == tempFeature2.get_feature_id()
    assert np.allclose(featureMetadata1.iloc[0][
        ['fov', 'volume', 'center_x', 'center_y',
         'min_x', 'min_y', 'max_x', 'max_y']].values,
           np.array([1, 4, 1, 1, 0, 0, 2, 2]))

    assert len(readFeatures) == 2
    assert len(featureMetadata) == 2
    if readFeatures[0].get_feature_id() == feature1.get_feature_id():
        f1Index = 0
        f2Index = 1
    else:
        f1Index = 1
        f2Index = 0
    assert readFeatures[f1Index].equals(feature1)
    assert readFeatures[f2Index].equals(tempFeature2)

    assert len(readFeaturesEmpty) == 0
    assert len(metaDataEmpty) == 0


def test_feature_contained_within_boundary():
    interiorLabels = np.zeros((1, 8, 8))
    interiorLabels[0, 2:6, 2:6] = 1
    interiorFeature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        interiorLabels, 0)

    exteriorLabels = np.zeros((1, 8, 8))
    exteriorLabels[0, 1:7, 1:7] = 1
    exteriorFeature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        exteriorLabels, 0)

    overlappingLabels = np.zeros((1, 8, 8))
    overlappingLabels[0, 0:5, 0:5] = 1
    overlappingFeature = spatialfeature.SpatialFeature\
        .feature_from_label_matrix(overlappingLabels, 0)

    assert interiorFeature.is_contained_within_boundary(exteriorFeature)
    assert not exteriorFeature.is_contained_within_boundary(interiorFeature)

    assert interiorFeature.is_contained_within_boundary(overlappingFeature)
    assert overlappingFeature.is_contained_within_boundary(interiorFeature)

    assert exteriorFeature.is_contained_within_boundary(overlappingFeature)
    assert overlappingFeature.is_contained_within_boundary(exteriorFeature)


def test_feature_contains_point():
    point1 = geometry.Point(-0.1, -0.1)
    point2 = geometry.Point(0.9, 0.9)
    point3 = geometry.Point(1.5, 1.5)

    assert not feature1.contains_point(point1, 0)
    assert not feature1.contains_point(point2, 0)
    assert feature1.contains_point(point3, 0)
    assert not feature4.contains_point(point1, 0)
    assert not feature4.contains_point(point2, 0)
    assert feature4.contains_point(point3, 0)
    assert not feature4.contains_point(point1, 1)
    assert feature4.contains_point(point2, 1)
    assert feature4.contains_point(point3, 1)


def test_feature_contains_positions():
    positions1 = np.array([[0, 0, 0], [1.5, 1.5, 0]])
    positions2 = np.array([[-0.1, -0.1, 0], [0.9, 0.9, 0], [1.5, 1.5, 0],
                           [-0.1, -0.1, 1], [0.9, 0.9, 1], [1.5, 1.5, 1]])
    assert all([a == b for a, b in zip(feature1.contains_positions(positions1),
                                       [False, True])])
    assert all([a == b for a, b in zip(feature4.contains_positions(positions1),
                                       [False, True])])
    assert all([a == b for a, b in zip(feature4.contains_positions(positions2),
                                       [False, False, True,
                                        False, True, True])])


def test_find_overlapping_cells():
    t1 = p1.get_overlapping_features(allCells)
    t2 = p2.get_overlapping_features(allCells)
    t3 = p3.get_overlapping_features(allCells)
    t4 = p4.get_overlapping_features(allCells)
    t5 = p5.get_overlapping_features(allCells)

    assert ((p1 in t1) and (p3 in t1)
            and (p2 not in t1) and (p5 not in t1) and (p5 not in t1))
    assert len(t2) == 0
    assert ((p3 in t3) and (p1 in t3) and (p4 in t3)
            and (p2 not in t3) and (p5 not in t3))
    assert ((p4 in t4) and (p3 in t4) and
            (p1 not in t4) and (p2 not in t4) and (p5 not in t4))
    assert ((p5 in t5) and (p1 not in t5)
            and (p2 not in t5) and (p3 not in t5) and (p4 not in t5))

<<<<<<< HEAD
def test_remove_overlapping_cells():

    allFOVs = [0]
    fovBoxes = [geometry.box(-1, -1, 10, 10)]
    currentFOV = 0

    spatialIndex = rtree.index.Index()
    numToID = dict()
    idToNum = dict()
    currentID = 0
    for currentFOV in allFOVs:
        for i in range(len(allCells)):
            numToID[currentID] = allCells[i].get_feature_id()
            idToNum[allCells[i].get_feature_id()] = currentID
            currentID += 1
    for cell in allCells:
        spatialIndex.insert(idToNum[cell.get_feature_id()],
                            cell.get_bounding_box(), obj=cell)

    G = nx.Graph()
    G = spatialfeature.construct_graph(G, allCells, spatialIndex,
                                       currentFOV, allFOVs, fovBoxes)
=======

def test_remove_overlapping_cells():

    def construct_graph(cells):
        G = nx.Graph()
        spatialIndex = rtree.index.Index()
        numToID = dict()
        idToNum = dict()
        currentID = 0
        currentUnassigned = cells
        for i in range(len(currentUnassigned)):
            numToID[currentID] = currentUnassigned[i].get_feature_id()
            idToNum[currentUnassigned[i].get_feature_id()] = currentID
            currentID += 1
        spatialfeature.append_cells_to_spatial_tree(
            spatialIndex, currentUnassigned, idToNum)

        currentCells = cells
        for cell in currentCells:
            overlappingCells = spatialIndex.intersection(
                cell.get_bounding_box(), objects=True)
            toCheck = [x.object for x in overlappingCells]
            cellsToConsider = cell.get_overlapping_features(toCheck)
            if len(cellsToConsider) == 0:
                pass

            else:
                for cellToConsider in cellsToConsider:
                    assignedFOV = 0
                    if cellToConsider.get_feature_id() not in G.nodes:
                        G.add_node(cellToConsider.get_feature_id(),
                                   originalFOV=cellToConsider.get_fov(),
                                   assignedFOV=assignedFOV)
                if len(cellsToConsider) > 1:
                    for cellToConsider1 in cellsToConsider:
                        if cellToConsider1.get_feature_id() !=\
                                cell.get_feature_id():
                            G.add_edge(cell.get_feature_id(),
                                       cellToConsider1.get_feature_id())
        return G

    G = construct_graph(allCells)
>>>>>>> b46e01fbcf2daf9f5898a282726b02ed717a930c

    cleanedCellsDF = spatialfeature.remove_overlapping_cells(G)
    keptCells = cleanedCellsDF['cell_id'].values.tolist()

    assert p1.get_feature_id() in keptCells
    assert p4.get_feature_id() in keptCells
    assert p5.get_feature_id() in keptCells
    assert p2.get_feature_id() not in keptCells
    assert p3.get_feature_id() not in keptCells