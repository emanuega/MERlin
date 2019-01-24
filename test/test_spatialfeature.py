import pytest
import numpy as np
import json
from shapely import geometry

from merlin.util import spatialfeature


testLabels1 = np.zeros((1, 4, 4))
testLabels1[0, 1:3, 1:3] = 1
feature1 = spatialfeature.SpatialFeature.feature_from_label_matrix(
    testLabels1, 0)

testLabels2 = np.zeros((1, 4, 4))
testLabels2[0, 0:3, 0:3] = 1
feature2 = spatialfeature.SpatialFeature.feature_from_label_matrix(
    testLabels2, 0)

testLabels3 = np.zeros((2, 4, 4))
testLabels3[:, 1:3, 1:3] = 1
feature3 = spatialfeature.SpatialFeature.feature_from_label_matrix(
    testLabels3, 0, zCoordinates=np.array([0, 0.5]))


def test_feature_from_label_matrix():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(geometry.Polygon(
        [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]
    ))


def test_feature_from_label_matrix_transform():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1
    transformMatrix = np.array([[2.0, 0, 3],
                                [0, 2.5, 1],
                                [0, 0, 1]])

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0, transformMatrix)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(geometry.Polygon(
        [[5, 3.5], [5, 6], [7, 6], [7, 3.5], [5, 3.5]]
    ))


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


def test_feature_json_db_read_write_one_fov(single_task, simple_merfish_data):
    featureDB = spatialfeature.JSONSpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature1, feature2], fov=0)
    readFeatures = featureDB.get_features(fov=0)

    assert len(readFeatures) == 2
    if readFeatures[0].get_feature_id() == feature1.get_feature_id():
        f1Index = 0
        f2Index = 1
    else:
        f1Index = 1
        f2Index = 0
    assert readFeatures[f1Index].equals(feature1)
    assert readFeatures[f2Index].equals(feature2)


