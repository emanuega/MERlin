from scipy import ndimage
import numpy as np    

"""
This module contains code for performing filtering operations on images
"""

def high_pass_filter(img: np.ndarray, lowPassKrernel: int) -> np.ndarray:
	"""
	 Args:
		image: the input image to be filtered
		lowPassKrernel: the size of the gaussian kernel to use for low pass.

	Returns:
		the high pass filtered image image

	"""
	img = img.astype(np.int16)
	lowpass = ndimage.gaussian_filter(img, lowPassKrernel)
	gauss_highpass = img - lowpass
	gauss_highpass[gauss_highpass<0] = 0
	return gauss_highpass

