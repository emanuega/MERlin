import cv2
import numpy as np
from scipy import ndimage

from merlin.util import matlab

"""
This module containts utility functions for performing deconvolution on
images.
"""


def calculate_projectors(windowSize: int, sigmaG: float) -> list:
    """Calculate forward and backward projectors as described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function

    Returns:
        A list containing the forward and backward projectors to use for
        Lucy-Richardson deconvolution.
    """
    pf = matlab.matlab_gauss2D(shape=(windowSize, windowSize),
                               sigma=sigmaG)
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    #
    # These values are from Guo et al.
    alpha = 0.001
    beta = 0.001
    n = 8

    # This is the cut-off frequency.
    kc = 1.0/(0.5 * 2.355 * sigmaG)

    # FFT frequencies
    kv = np.fft.fftfreq(pfFFT.shape[0])

    kx = np.zeros((kv.size, kv.size))
    for i in range(kv.size):
        kx[i, :] = np.copy(kv)

    ky = np.transpose(kx)
    kk = np.sqrt(kx*kx + ky*ky)

    # Wiener filter
    bWiener = pfFFT/(np.abs(pfFFT) * np.abs(pfFFT) + alpha)

    # Buttersworth filter
    eps = np.sqrt(1.0/(beta*beta) - 1)

    kkSqr = kk*kk/(kc*kc)
    bBWorth = 1.0/np.sqrt(1.0 + eps * eps * np.power(kkSqr, n))

    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth

    # back projector.
    pb = np.real(np.fft.ifft2(pbFFT))

    return [pf, pb]


def deconvolve_lucyrichardson(image: np.ndarray,
                              windowSize: int,
                              sigmaG: float,
                              iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function.

    Ported from Matlab deconvlucy.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
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

    if windowSize % 2 != 1:
        gaussianFilter = matlab.matlab_gauss2D(shape=(windowSize, windowSize),
                                               sigma=sigmaG)

    for i in range(iterationCount):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1) / (np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        if windowSize % 2 == 1:
            cv2.GaussianBlur(Y, (windowSize, windowSize), sigmaG, reblurred,
                             borderType=cv2.BORDER_REPLICATE)
        else:
            reblurred = ndimage.convolve(Y, gaussianFilter, mode='constant')
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        if windowSize % 2 == 1:
            cv2.GaussianBlur(imR, (windowSize, windowSize), sigmaG, imR,
                             borderType=cv2.BORDER_REPLICATE)
        else:
            imR = ndimage.convolve(imR, gaussianFilter, mode='constant')
            imR[imR > 2 ** 16] = 0
        np.copyto(J2, J1)
        np.multiply(Y, imR, out=J1)
        np.copyto(T2, T1)
        np.subtract(J1, Y, out=T1)
    return J1


def deconvolve_lucyrichardson_guo(image: np.ndarray,
                                  windowSize: int,
                                  sigmaG: float,
                                  iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    [pf, pb] = calculate_projectors(windowSize, sigmaG)

    eps = 1.0e-6
    i_max = 2**16-1

    ek = np.copy(image)
    np.clip(ek, eps, None, ek)

    for i in range(iterationCount):
        ekf = cv2.filter2D(ek, -1, pf,
                           borderType=cv2.BORDER_REPLICATE)
        np.clip(ekf, eps, i_max, ekf)

        ek = ek*cv2.filter2D(image/ekf, -1, pb,
                             borderType=cv2.BORDER_REPLICATE)
        np.clip(ek, eps, i_max, ek)

    return ek
