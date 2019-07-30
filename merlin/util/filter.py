from scipy import ndimage
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def high_pass_filter(image: np.ndarray, lowPassKernel: int) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        lowPassKernel: the size of the gaussian kernel to use for low pass.

    Returns:
        the high pass filtered image image
    """
    img = image.astype(np.int16)
    lowpass = ndimage.gaussian_filter(img, lowPassKernel)
    gauss_highpass = img - lowpass
    gauss_highpass[gauss_highpass < 0] = 0
    return gauss_highpass
