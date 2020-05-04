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
    # Clip so that we don't try and take the sqrt of 0.0 or
    # a negative number.
    #
    background = np.clip(background, 1.0, None)
    return foreground/np.sqrt(background)


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
