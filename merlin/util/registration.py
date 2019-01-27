import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from skimage import transform


def extract_control_points(
        referencePoints: np.ndarray, movingPoints: np.ndarray,
        gridSpacing: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.arange(-30, 30, gridSpacing)

    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(referencePoints)
    distances, indexes = neighbors.kneighbors(
        movingPoints, return_distance=True)
    differences = [[movingPoints[i] - referencePoints[x]
                    for x in indexes[i]]
                   for i in range(len(movingPoints))]
    counts, xedges, yedges = np.histogram2d(
        [x[0] for y in differences for x in y],
        [x[1] for y in differences for x in y],
        bins=edges)
    maxIndex = np.unravel_index(counts.argmax(), counts.shape)
    offset = (xedges[maxIndex[0]], yedges[maxIndex[1]])

    distancesShifted, indexesShifted = neighbors.kneighbors(
        movingPoints - np.tile(offset, (movingPoints.shape[0], 1)),
        return_distance=True)

    controlIndexes = [x[0] < gridSpacing for x in distancesShifted]
    referenceControls = np.array([referencePoints[x[0]]
                                  for x in indexesShifted[controlIndexes]])
    movingControls = movingPoints[controlIndexes, :]

    return referenceControls, movingControls


def estimate_transform_from_points(
        referencePoints: np.ndarray, movingPoints: np.ndarray) \
        -> transform.EuclideanTransform:
    # TODO when there are too few points, this should return unit transformation
    tform = transform.SimilarityTransform()
    tform.estimate(referencePoints, movingPoints)
    return tform