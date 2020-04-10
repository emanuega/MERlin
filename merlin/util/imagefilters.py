import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def high_pass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image image
    """
    img = image.astype(np.int16)
    lowpass = cv2.GaussianBlur(img,
                               (windowSize, windowSize),
                               sigma,
                               borderType=cv2.BORDER_REPLICATE)
    gauss_highpass = img - lowpass
    gauss_highpass[gauss_highpass < 0] = 0
    return gauss_highpass
