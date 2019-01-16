import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage import transform


def extract_control_points(referencePoints, movingPoints, gridSpacing=0.5):
    edges = np.arange(-30, 30, gridSpacing)

    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(referencePoints)
    distances, indexes = neighbors.kneighbors(
        movingPoints, return_distance=True)
    differences = [[referencePoints[x] - movingPoints[i]
                    for x in indexes[i]]
                   for i in range(len(movingPoints))]
    counts, xedges, yedges = np.histogram2d(
        [x[0] for y in differences for x in y],
        [x[1] for y in differences for x in y],
        bins=edges)
    maxIndex = np.unravel_index(counts.argmax(), counts.shape)
    offset = (-xedges[maxIndex[0]], -yedges[maxIndex[1]])

    distancesShifted, indexesShifted = neighbors.kneighbors(
        movingPoints - np.tile(offset, (movingPoints.shape[0], 1)),
        return_distance=True)

    controlIndexes = [x[0] < gridSpacing for x in distancesShifted]
    referenceControls = np.array([referencePoints[x[0]]
                                  for x in indexesShifted[controlIndexes]])
    movingControls = movingPoints[controlIndexes, :]

    return referenceControls, movingControls


def estimate_affine_transform(referenceControls, movingControls):
    tform = transform.AffineTransform()
    tform.estimate(referenceControls, movingControls)
    return tform