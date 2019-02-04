import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from skimage import transform


def extract_control_points(
        referencePoints: np.ndarray, movingPoints: np.ndarray,
        gridSpacing: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    If fewer than 10 points are provided for either the reference or the moving
    list, this returns no points.

    Args:
        referencePoints: a n x 2 numpy array containing the reference points.
        movingPoints: a m x 2 numpy array containing the moving points.
        gridSpacing: the spacing of the grid for the 2d histogram for
            estimating the course transformation
    Returns: two np arrays (select reference points, select moving points)
        both of which are p x 2. The i'th point in the reference list
        has been matched to the i'th point in the moving list.
    """
    if len(referencePoints) < 10 or len(movingPoints) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))

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
    """

    If fewer than two points are provided, this will return the identity
    transform.

    Args:
        referencePoints: a n x 2 numpy array containing the reference points
        movingPoints: a n x 2 numpy array containing the moving points, where
            the i'th point of moving points corresponds with the i'th point
            of reference points.
    Returns: a similarity transform estimated from the paired points.

    """
    tform = transform.SimilarityTransform()
    if len(referencePoints) < 2 or len(movingPoints) < 2:
        return tform
    tform.estimate(referencePoints, movingPoints)
    return tform