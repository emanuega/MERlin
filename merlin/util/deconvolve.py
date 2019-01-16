import cv2
import numpy as np


"""
This module containts utility functions for performing deconvolution on
images.
"""


def deconvolve_lucyrichardson(image: np.ndarray, windowSize: int, sigmaG: float,
                              iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function.

    Ported from Matlab deconvlucy.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    eps = np.finfo(float).eps
    Y = np.copy(image)
    J1 = np.copy(image)
    J2 = np.copy(image)
    wI = np.copy(image)
    imR = np.copy(image)
    reblurred = np.copy(image)
    tmpMat1 = np.zeros(image.shape, dtype=float)
    tmpMat2 = np.zeros(image.shape, dtype=float)
    T1 = np.zeros(image.shape, dtype=float)
    T2 = np.zeros(image.shape, dtype=float)
    l = 0
    for i in range(iterationCount):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1)/(np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        cv2.GaussianBlur(Y, (windowSize, windowSize), sigmaG, reblurred)
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        cv2.GaussianBlur(imR, (windowSize, windowSize), sigmaG, imR)
        J2 = J1
        np.multiply(Y, imR, out=J1)
        T2 = T1
        np.subtract(J1, Y, out=T1)
    return J1
