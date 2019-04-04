import numpy as np
from types import Tuple


"""
This module contains Matlab functions that do not have equivalents in
python libraries.
"""


def imregionalmax(inputData: np.ndarray, mask: np.ndarray=None) -> np.ndarray:
    """Compute the regional maximum of inputData.

    This is algorithmically equivalent to Matlab's imregionalmax for 3d
    data with full connectivity. It currently runs much slower than
    the Matlab implementation.

    Args:
        inputData: a 3 dimensional numpy array to find the regional maximums
        mask: a mask the same size as inputData that indicates the position
            of pixels to exclude from consideration as possible maximums.
            If None, then all pixels are considered as possible maximums.
    Returns:
        a boolean numpy array indicating the positions of the regional maximums
            with True
    """
    result = np.ones(inputData.shape).astype(bool)
    if mask is not None:
        result[mask] = False

    def neighborhood_offsets(z, x, y):
        zMin = -1
        zMax = 2
        if z == 0:
            zMin = 0
        if z == inputData.shape[0] - 1:
            zMax = 1

        xMin = -1
        xMax = 2
        if x == 0:
            xMin = 0
        if x == inputData.shape[1] - 1:
            xMax = 1

        yMin = -1
        yMax = 2
        if y == 0:
            yMin = 0
        if y == inputData.shape[2] - 1:
            yMax = 1

        offsets = set(
            [(k, i, j) for k in range(zMin, zMax) for i in range(xMin, xMax)
             for j in range(yMin, yMax)])
        offsets.remove((0, 0, 0))
        return offsets

    def check_pixel(z, x, y):
        pixelValue = inputData[z, x, y]
        for no in neighborhood_offsets(z, x, y):
            if inputData[z + no[0], x + no[1], y + no[2]] > pixelValue:
                return False
            if inputData[z + no[0], x + no[1], y + no[2]] == pixelValue and not\
                    result[z + no[0], x + no[1], y + no[2]]:
                return False
        return True

    done = False
    while not done:
        tempResult = result.copy()
        for k in range(inputData.shape[0]):
            for i in range(inputData.shape[1]):
                for j in range(inputData.shape[2]):
                    if result[k, i, j]:
                        tempResult[k, i, j] = check_pixel(k, i, j)

        done = np.array_equal(result, tempResult)
        result = tempResult

    return result


def matlab_gauss2D(shape: Tuple[int, int]=(3, 3), sigma: float=0.5
                   ) -> np.array:
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
