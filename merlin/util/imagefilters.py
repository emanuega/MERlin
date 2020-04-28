import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def est_significance(foreground: np.ndarray,
                     background: np.ndarray) -> np.ndarray:
    """
    Args:
        foreground: the image foreground estimate.
        background: the image background estimate.

    Returns:
        an estimate of the significance of each pixel in units of
        sigma (or standard deviation).
    """
    snbfIm = foreground/np.sqrt(background)

    # Rescale snbfIM so that it's histogram is Gaussian with
    # zero offset and sigma = 1.0.
    #

    # Why subtract 1.0 in order to make offset = 0.0? This is
    # emperical, not entirely sure I understand why the offset
    # is 1.0.
    #
    snbfIm -= 1.0

    sx = min(snbfIm.shape[0], 512)
    sy = min(snbfIm.shape[1], 512)

    [hist, edges] = np.histogram(snbfIm[0:sx, 0:sy], bins=100, range=(0, 6))
    centers = 0.5*(edges[1:] + edges[:-1])

    sigma = np.sum(hist*centers)/np.sum(hist)

    return snbfIm/sigma


def high_low_filter(image: np.ndarray,
                    windowSize: int,
                    sigma: float,
                    reps: int) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.
        reps: number of repetitions of foreground/background estimation.

    Returns:
        the high pass and low pass filtered images.
    """
    lowpass = np.copy(image)
    for i in range(reps):
        lowpass = cv2.GaussianBlur(lowpass,
                                   (windowSize, windowSize),
                                   sigma,
                                   borderType=cv2.BORDER_REPLICATE)
        highpass = image - lowpass
        highpass[lowpass > image] = 0
        lowpass = image - highpass

    return [highpass, lowpass]


def high_pass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    lowpass = cv2.GaussianBlur(image,
                               (windowSize, windowSize),
                               sigma,
                               borderType=cv2.BORDER_REPLICATE)
    highpass = image - lowpass
    highpass[lowpass > image] = 0
    return highpass
