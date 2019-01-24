import numpy as np
import json
from shapely import geometry

from merlin.util import spatialfeature


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


def test_feature_get_volume_2d():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0)

    assert feature.get_volume() == 1


def test_feature_get_volume_3d():
    testLabels = np.zeros((2, 4, 4))
    testLabels[:, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0, zCoordinates=np.array([0, 0.5]))

    assert feature.get_volume() == 0.5


def test_feature_serialization_to_json():
    testLabels = np.zeros((2, 4, 4))
    testLabels[:, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0, zCoordinates=np.array([0, 0.5]))

    featureIn = spatialfeature.SpatialFeature.from_json_dict(
        json.loads(json.dumps(feature.to_json_dict())))

    assert featureIn.get_fov() == feature.get_fov()
    assert featureIn.get_feature_id() == feature.get_feature_id()
    assert np.array_equal(featureIn.get_z_coordinates(),
                          feature.get_z_coordinates())
    assert len(feature.get_boundaries()) == len(featureIn.get_boundaries())
    for b, bIn in zip(feature.get_boundaries(), featureIn.get_boundaries()):
        assert(len(b) == len(bIn))
        for x, y in zip(b, bIn):
            assert x.equals(y)