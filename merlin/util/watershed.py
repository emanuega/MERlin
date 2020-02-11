import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
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

    seeds = morphology.local_maxima(seedImages, allow_borders=True)

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

def get_membrane_mask(self, membraneImages: np.ndarray) -> np.ndarray:
    """Calculate binary mask with 1's in membrane pixels and 0 otherwise.
    The images expected are some type of membrane label (WGA, ConA, 
    Lamin, Cadherins)

    Args:
        membraneImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
    Returns: 
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """
    mask = np.zeros(membraneImages.shape)
    fineBlockSize = 61
    for z in range(len(self.dataSet.get_z_positions())):
        mask[z, :, :] = (membraneImages[z, :, :] >
                         filters.threshold_local(membraneImages[z, :, :],
                                                 fineBlockSize,
                                                 offset=0))
        mask[z, :, :] = morphology.remove_small_objects(
                                mask[z, :, :].astype('bool'),
                                min_size=100,
                                connectivity=1)
        mask[z, :, :] = morphology.binary_closing(mask[z, :, :],
                                                  morphology.selem.disk(5))
        mask[z, :, :] = morphology.skeletonize(mask[z, :, :])

    # combine masks
    return mask

def get_nuclei_mask(self, nucleiImages: np.ndarray) -> np.ndarray:
    """Calculate binary mask with 1's in membrane pixels and 0 otherwise.
    The images expected are some type of Nuclei label (e.g. DAPI)

    Args:
        membraneImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
    Returns: 
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """

    # generate nuclei mask based on thresholding
    thresholdingMask = np.zeros(nucleiImages.shape)
    coarseBlockSize = 241
    fineBlockSize = 61
    for z in range(len(self.dataSet.get_z_positions())):
        coarseThresholdingMask = (nucleiImages[z, :, :] >
                                  filters.threshold_local(
                                    nucleiImages[z, :, :],
                                    coarseBlockSize,
                                    offset=0))
        fineThresholdingMask = (nucleiImages[z, :, :] >
                                filters.threshold_local(
                                    nucleiImages[z, :, :],
                                    fineBlockSize,
                                    offset=0))
        thresholdingMask[z, :, :] = (coarseThresholdingMask *
                                     fineThresholdingMask)
        thresholdingMask[z, :, :] = binary_fill_holes(
                                    thresholdingMask[z, :, :])

    # generate border mask, necessary to avoid making a single
    # connected component when using binary_fill_holes below
    borderMask = np.zeros((2048, 2048))
    borderMask[25:2023, 25:2023] = 1

    # TODO - use the image size variable for borderMask

    # generate nuclei mask from hessian, fine
    fineHessianMask = np.zeros(nucleiImages.shape)
    for z in range(len(self.dataSet.get_z_positions())):
        fineHessian = filters.hessian(nucleiImages[z, :, :])
        fineHessianMask[z, :, :] = fineHessian == fineHessian.max()
        fineHessianMask[z, :, :] = morphology.binary_closing(
                                                fineHessianMask[z, :, :],
                                                morphology.selem.disk(5))
        fineHessianMask[z, :, :] = fineHessianMask[z, :, :] * borderMask
        fineHessianMask[z, :, :] = binary_fill_holes(
                                    fineHessianMask[z, :, :])

    # generate dapi mask from hessian, coarse
    coarseHessianMask = np.zeros(nucleiImages.shape)
    for z in range(len(self.dataSet.get_z_positions())):
        coarseHessian = filters.hessian(nucleiImages[z, :, :] -
                                        morphology.white_tophat(
                                            nucleiImages[z, :, :],
                                            morphology.selem.disk(20)))
        coarseHessianMask[z, :, :] = coarseHessian == coarseHessian.max()
        coarseHessianMask[z, :, :] = morphology.binary_closing(
            coarseHessianMask[z, :, :], morphology.selem.disk(5))
        coarseHessianMask[z, :, :] = (coarseHessianMask[z, :, :] *
                                      borderMask)
        coarseHessianMask[z, :, :] = binary_fill_holes(
                                        coarseHessianMask[z, :, :])

    # combine masks
    nucleiMask = thresholdingMask + fineHessianMask + coarseHessianMask
    return binary_fill_holes(nucleiMask)

def get_cv2_watershed_markers(self, nucleiImages: np.ndarray,
                              membraneImages: np.ndarray) -> np.ndarray:
    """Combine membrane and nuclei markers into a single multilabel mask
    for CV2 watershed

    Args:
        nucleiImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        membraneImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
    Returns: 
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            cv2-compatible watershed markers
    """

    nucleiMask = self.get_nuclei_mask(nucleiImages)
    membraneMask = self.get_membrane_mask(membraneImages)

    watershedMarker = np.zeros(nucleiMask.shape)

    for z in range(len(self.dataSet.get_z_positions())):

        # generate areas of sure bg and fg, as well as the area of
        # unknown classification
        background = morphology.dilation(nucleiMask[z, :, :],
                                         morphology.selem.disk(15))
        membraneDilated = morphology.dilation(
            membraneMask[z, :, :].astype('bool'),
            morphology.selem.disk(10))
        foreground = morphology.erosion(nucleiMask[z, :, :] * ~
                                        membraneDilated,
                                        morphology.selem.disk(5))
        unknown = background * ~ foreground

        background = np.uint8(background) * 255
        foreground = np.uint8(foreground) * 255
        unknown = np.uint8(unknown) * 255

        # Marker labelling
        ret, markers = cv2.connectedComponents(foreground)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 100

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        watershedMarker[z, :, :] = markers

    return watershedMarker

def convert_grayscale_to_rgb(self, uint16Image: np.ndarray) -> np.ndarray:
    """Convert a 16 bit 2D grayscale image into a 3D 8-bit RGB image. 
    cv2 only works in 8-bit. Based on https://stackoverflow.com/questions/
    25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv3D 

    Args:
        uint16Image: a 2 dimensional numpy array containing the 16-bit 
            image
    Returns: 
        ndarray containing a 3 dimensional 8-bit image stack 
    """

    # invert image
    uint16Image = 2**16 - uint16Image

    # convert to uint8
    ratio = np.amax(uint16Image) / 256
    uint8Image = (uint16Image / ratio).astype('uint8')

    rgbImage = np.zeros((2048, 2048, 3))
    rgbImage[:, :, 0] = uint8Image
    rgbImage[:, :, 1] = uint8Image
    rgbImage[:, :, 2] = uint8Image
    rgbImage = rgbImage.astype('uint8')

    return rgbImage

def apply_cv2_watershed(self, nucleiImages: np.ndarray,
                    watershedMarkers: np.ndarray) -> np.ndarray:
    """Perform watershed using cv2

    Args:
        nucleiImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        watershedMarkers: a 3 dimensional numpy array containing the cv2
            markers arranged as (z, x, y).
    Returns: 
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            segmented cells. masks in different z positions are 
            independent
    """

    watershedOutput = np.zeros(watershedMarkers.shape)
    for z in range(len(self.dataSet.get_z_positions())):
        rgbImage = self.convert_grayscale_to_rgb(nucleiImages[z, :, :])
        watershedOutput[z, :, :] = cv2.watershed(rgbImage,
                                                 watershedMarkers[z, :, :].
                                                 astype('int32'))
        watershedOutput[z, :, :][watershedOutput[z, :, :] <= 100] = 0

    return watershedOutput

def get_overlapping_nuclei(self, watershedZ0: np.ndarray,
                           watershedZ1: np.ndarray, n0: int):
    """Perform watershed using cv2

    Args:
        watershedZ0: a 2 dimensional numpy array containing a 
            segmentation mask
        watershedZ1: a 2 dimensional numpy array containing a
            segmentation mask adjacent to watershedZ1
        n0: an integer with the index of the cell/nuclei to be compared
            between the provided watershed segmentation masks
    Returns:
        a tuple (n1, f0, f1) containing the label of the cell in Z1 
        overlapping n0 (n1), the fraction of n0 overlaping n1 (f0) and 
        the fraction of n1 overlapping n0 (f1)
       
    """

    z1NucleiIndexes = np.unique(watershedZ1[watershedZ0 == n0])
    z1NucleiIndexes = z1NucleiIndexes[z1NucleiIndexes > 100]

    if z1NucleiIndexes.shape[0] > 0:

        # calculate overlap fraction
        n0Area = np.count_nonzero(watershedZ0 == n0)
        n1Area = np.zeros(len(z1NucleiIndexes))
        overlapArea = np.zeros(len(z1NucleiIndexes))

        for ii in range(len(z1NucleiIndexes)):
            n1 = z1NucleiIndexes[ii]
            n1Area[ii] = np.count_nonzero(watershedZ1 == n1)
            overlapArea[ii] = np.count_nonzero((watershedZ0 == n0) *
                                               (watershedZ1 == n1))

        n0OverlapFraction = np.asarray(overlapArea / n0Area)
        n1OverlapFraction = np.asarray(overlapArea / n1Area)
        index = list(range(len(n0OverlapFraction)))

        # select the nuclei that has the highest fraction in n0 and n1
        r1, r2, indexSorted = zip(*sorted(zip(n0OverlapFraction,
                                              n1OverlapFraction,
                                              index),
                                  reverse=True))

        if (n0OverlapFraction[indexSorted[0]] > 0.2 and
                n1OverlapFraction[indexSorted[0]] > 0.5):
            return z1NucleiIndexes[indexSorted[0]],
            n0OverlapFraction[indexSorted[0]],
            n1OverlapFraction[indexSorted[0]]
        else:
            return False, False, False
    else:
        return False, False, False

def combine_2d_segmentation_masks_into_3d(self,
                                          watershedOutput:
                                          np.ndarray) -> np.ndarray:
    """Take a 3 dimensional watershed masks and relabel them so that 
    nuclei in adjacent sections have the same label if the area their 
    overlap surpases certain threshold 

    Args:
        watershedOutput: a 3 dimensional numpy array containing the 
            segmentation masks arranged as (z, x, y).
    Returns: 
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            relabeled segmented cells
    """

    # Initialize empty array with size as watershedOutput array
    watershedCombinedZ = np.zeros(watershedOutput.shape)

    # copy the mask of the section farthest to the coverslip
    watershedCombinedZ[-1, :, :] = watershedOutput[-1, :, :]

    # starting far from coverslip
    for z in range(len(self.dataSet.get_z_positions())-1, 0, -1):
        zNucleiIndex = np.unique(watershedOutput[z, :, :])[
                                np.unique(watershedOutput[z, :, :]) > 100]

    for n0 in zNucleiIndex:
        n1, f0, f1 = self.get_overlapping_nuclei(
                                            watershedCombinedZ[z, :, :],
                                            watershedOutput[z-1, :, :],
                                            n0)
        if n1:
            watershedCombinedZ[z-1, :, :][(watershedOutput[z-1, :, :] ==
                                           n1)] = n0
    return watershedCombinedZ
