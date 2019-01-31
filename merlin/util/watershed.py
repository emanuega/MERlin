import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
from skimage import filters
from skimage import measure
from pyclustering.cluster import kmedoids
from typing import Tuple

from merlin.util import matlab

"""
This module contains utility functions for preparing imagmes for 
watershed segmentation.
"""

# To match Matlab's strel('disk', 20)
diskStruct = morphology.diamond(28)[9:48, 9:48]


def extract_seeds(seedImageStackIn: np.ndarray) -> np.ndarray:
    """Determine seed positions from the input images.

    The initial seeds are determined by finding the regional intensity maximums
    after erosion and filtering with an adaptive threshold. These initial
    seeds are then expanded by dilation.

    Args:
        seedImageStackIn: a 3 dimensional numpy array arranged as (z,x,y)
    Returns: a boolean numpy array with the same dimensions as seedImageStackIn
        where a given (z,x,y) coordinate is True if it corresponds to a seed
        position and false otherwise.
    """
    seedImages = seedImageStackIn.copy()

    seedImages = ndimage.grey_erosion(
        seedImages,
        footprint=ndimage.morphology.generate_binary_structure(3, 1))
    seedImages = np.array([cv2.erode(x, diskStruct,
                                     borderType=cv2.BORDER_REFLECT)
                           for x in seedImages])

    thresholdFilterSize = int(2 * np.floor(seedImages.shape[1] / 16) + 1)
    seedMask = np.array([x < 1.1 * filters.threshold_local(
        x, thresholdFilterSize, method='mean', mode='nearest')
                         for x in seedImages])

    seedImages[seedMask] = 0

    seeds = matlab.imregionalmax(seedImages, seedMask)

    seeds = ndimage.morphology.binary_dilation(
        seeds, structure=ndimage.morphology.generate_binary_structure(3, 1))
    seeds = np.array([ndimage.morphology.binary_dilation(
        x, structure=morphology.diamond(28)[9:48, 9:48]) for x in seeds])

    return seeds


def separate_merged_seeds(seedsIn: np.ndarray) -> np.ndarray:
    """Separate seeds that are merged in 3 dimensions but are separated
    in some 2 dimensional slices.

    Args:
        seedsIn: a 3 dimensional binary numpy array arranged as (z,x,y) where
            True indicates the pixel corresponds with a seed.
    Returns: a 3 dimensional binary numpy array of the same size as seedsIn
        indicating the positions of seeds after processing.
    """

    def create_region_image(shape, c):
        region = np.zeros(shape)
        for x in c.coords:
            region[x[0], x[1], x[2]] = 1
        return region

    components = measure.regionprops(measure.label(seedsIn))
    seeds = np.zeros(seedsIn.shape)
    for c in components:
        seedImage = create_region_image(seeds.shape, c)
        localProps = [measure.regionprops(measure.label(x)) for x in seedImage]
        seedCounts = [len(x) for x in localProps]

        if all([x < 2 for x in seedCounts]):
            goodFrames = [i for i, x in enumerate(seedCounts) if x == 1]
            goodProperties = [y for x in goodFrames for y in localProps[x]]
            seedPositions = np.round([np.median(
                [x.centroid for x in goodProperties], axis=0)]).astype(int)
        else:
            goodFrames = [i for i, x in enumerate(seedCounts) if x > 1]
            goodProperties = [y for x in goodFrames for y in localProps[x]]
            goodCentroids = [x.centroid for x in goodProperties]
            km = kmedoids.kmedoids(
                goodCentroids,
                np.random.choice(np.arange(len(goodCentroids)),
                                 size=np.max(seedCounts)))
            km.process()
            seedPositions = np.round(
                [goodCentroids[x] for x in km.get_medoids()]).astype(int)

        for s in seedPositions:
            for f in goodFrames:
                seeds[f, s[0], s[1]] = 1

    seeds = ndimage.morphology.binary_dilation(
        seeds, structure=ndimage.morphology.generate_binary_structure(3, 1))
    seeds = np.array([ndimage.morphology.binary_dilation(
        x, structure=diskStruct) for x in seeds])

    return seeds


def prepare_watershed_images(watershedImageStack: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the given images as the input image for watershedding.

    A watershed mask is determined using an adaptive threshold and the watershed
    images are inverted so the largest values in the watershed images become
    minima and then the image stack is normalized to have values between 0
    and 1.

    Args:
        watershedImageStack: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
    Returns: a tuple containing the normalized watershed images and the
        calculated watershed mask
    """
    filterSize = int(2 * np.floor(watershedImageStack.shape[1] / 16) + 1)

    watershedMask = np.array([ndimage.morphology.binary_fill_holes(
        x > 1.1 * filters.threshold_local(x, filterSize, method='mean',
                                          mode='nearest'))
        for x in watershedImageStack])

    normalizedWatershed = 1 - (watershedImageStack
                               - np.min(watershedImageStack)) / \
                          (np.max(watershedImageStack)
                           - np.min(watershedImageStack))
    normalizedWatershed[np.invert(watershedMask)] = 1

    return normalizedWatershed, watershedMask
