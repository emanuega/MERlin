import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage import filters
from skimage import measure
from skimage import feature
from pyclustering.cluster import kmedoids
from typing import Tuple

from merlin.util import matlab

from cellpose import models


"""
This module contains utility functions for preparing images for
watershed segmentation, as well as functions to perform segmentation
using machine learning approaches
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


def get_membrane_mask(membraneImages: np.ndarray,
                      compartmentChannelName: str,
                      membraneChannelName: str) -> np.ndarray:
    """Calculate binary mask with 1's in membrane pixels and 0 otherwise.
    The images expected are some type of membrane label (WGA, ConA,
    Lamin, Cadherins) or compartment images (DAPI, CD45, polyT)

    Args:
        membraneImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        membraneChannelName: A string with the name of a membrane channel.
        compartmentChannelName: A string with the name of the compartment
            channel
    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """
    mask = np.zeros(membraneImages.shape)
    if membraneChannelName != compartmentChannelName:
        fineBlockSize = 61
        for z in range(membraneImages.shape[0]):
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
    else:
        filterSigma2 = 5
        filterSize2 = int(2*np.ceil(2*filterSigma2)+1)
        edgeSigma = 2#  1 #2
        lowThresh = 0.1#  0.5 #0.2
        hiThresh = 0.5#  0.7 #0.6
        for z in range(membraneImages.shape[0]):
            blurredImage = cv2.GaussianBlur(membraneImages[z, :, :],
                                            (filterSize2, filterSize2),
                                            filterSigma2)
            edge0 = feature.canny(membraneImages[z, :, :],
                                  sigma=edgeSigma,
                                  use_quantiles=True,
                                  low_threshold=lowThresh,
                                  high_threshold=hiThresh)
            edge0 = morphology.dilation(edge0, morphology.selem.disk(10))

            edge1 = feature.canny(blurredImage,
                                  sigma=edgeSigma,
                                  use_quantiles=True,
                                  low_threshold=lowThresh,
                                  high_threshold=hiThresh)
            edge1 = morphology.dilation(edge1, morphology.selem.disk(10))

            mask[z, :, :] = edge0 + edge1

            mask[z, :, :] = morphology.skeletonize(mask[z, :, :])

    return mask


def get_compartment_mask(compartmentImages: np.ndarray) -> np.ndarray:
    """Calculate binary mask with 1's in compartment (nuclei or cytoplasm)
    pixels and 0 otherwise. The images expected are some type of compartment
    label (e.g. Nuclei: DAPI, Cytoplasm: PolyT, CD45, etc)

    Args:
        compartmentImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """

    # generate compartment mask based on thresholding
    thresholdingMask = np.zeros(compartmentImages.shape)
    coarseBlockSize = 241
    fineBlockSize = 61
    for z in range(compartmentImages.shape[0]):
        coarseThresholdingMask = (compartmentImages[z, :, :] >
                                  filters.threshold_local(
                                    compartmentImages[z, :, :],
                                    coarseBlockSize,
                                    offset=0))
        fineThresholdingMask = (compartmentImages[z, :, :] >
                                filters.threshold_local(
                                    compartmentImages[z, :, :],
                                    fineBlockSize,
                                    offset=0))
        thresholdingMask[z, :, :] = (coarseThresholdingMask *
                                     fineThresholdingMask)
        thresholdingMask[z, :, :] = binary_fill_holes(
                                    thresholdingMask[z, :, :])

    # generate border mask, necessary to avoid making a single
    # connected component when using binary_fill_holes below
    borderMask = np.zeros((compartmentImages.shape[1],
                           compartmentImages.shape[2]))
    borderMask[25:(compartmentImages.shape[1]-25),
               25:(compartmentImages.shape[2]-25)] = 1

    # generate compartment mask from hessian, fine
    fineHessianMask = np.zeros(compartmentImages.shape)
    for z in range(compartmentImages.shape[0]):
        fineHessian = filters.hessian(compartmentImages[z, :, :])
        fineHessianMask[z, :, :] = fineHessian == fineHessian.max()
        fineHessianMask[z, :, :] = morphology.binary_closing(
                                                fineHessianMask[z, :, :],
                                                morphology.selem.disk(5))
        fineHessianMask[z, :, :] = fineHessianMask[z, :, :] * borderMask
        fineHessianMask[z, :, :] = binary_fill_holes(
                                    fineHessianMask[z, :, :])

    # generate compartment mask from hessian, coarse
    coarseHessianMask = np.zeros(compartmentImages.shape)
    for z in range(compartmentImages.shape[0]):
        coarseHessian = filters.hessian(compartmentImages[z, :, :] -
                                        morphology.white_tophat(
                                            compartmentImages[z, :, :],
                                            morphology.selem.disk(20)))
        coarseHessianMask[z, :, :] = coarseHessian == coarseHessian.max()
        coarseHessianMask[z, :, :] = morphology.binary_closing(
            coarseHessianMask[z, :, :], morphology.selem.disk(5))
        coarseHessianMask[z, :, :] = (coarseHessianMask[z, :, :] *
                                      borderMask)
        coarseHessianMask[z, :, :] = binary_fill_holes(
                                        coarseHessianMask[z, :, :])

    # combine masks
    compartmentMask = thresholdingMask + fineHessianMask + coarseHessianMask
    return binary_fill_holes(compartmentMask)


def get_cv2_watershed_markers(compartmentImages: np.ndarray,
                              membraneImages: np.ndarray,
                              compartmentChannelName: str, 
                              membraneChannelName: str) -> np.ndarray:
    """Combine membrane and compartment markers into a single multilabel mask
    for CV2 watershed

    Args:
        compartmentImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        membraneImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        compartmentChannelName: str with the name of the compartment channel 
            to use
        membraneChannelName: str with the name of the membrane channel 
            to use

    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            cv2-compatible watershed markers
    """

    compartmentMask = get_compartment_mask(compartmentImages)
    membraneMask = get_membrane_mask(membraneImages,
                                     compartmentChannelName,
                                     membraneChannelName)

    watershedMarker = np.zeros(compartmentMask.shape)

    for z in range(compartmentImages.shape[0]):

        # generate areas of sure bg and fg, as well as the area of
        # unknown classification
        background = morphology.dilation(compartmentMask[z, :, :],
                                         morphology.selem.disk(15))
        membraneDilated = morphology.dilation(
            membraneMask[z, :, :].astype('bool'),
            morphology.selem.disk(10))
        foreground = morphology.erosion(compartmentMask[z, :, :] * ~
                                        membraneDilated,
                                        morphology.selem.disk(5))
        unknown = background * ~ foreground

        background = np.uint8(background) * 255
        foreground = np.uint8(foreground) * 255
        unknown = np.uint8(unknown) * 255

        # Marker labelling
        ret, markers = cv2.connectedComponents(foreground)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        watershedMarker[z, :, :] = markers

    return watershedMarker


def convert_grayscale_to_rgb(uint16Image: np.ndarray) -> np.ndarray:
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

    print('size = [' + str(uint16Image.shape[0])  +  ', ' +  str(uint16Image.shape[1]) + ']')

    rgbImage = np.zeros((uint16Image.shape[0], uint16Image.shape[1], 3))
    rgbImage[:, :, 0] = uint8Image
    rgbImage[:, :, 1] = uint8Image
    rgbImage[:, :, 2] = uint8Image
    rgbImage = rgbImage.astype('uint8')

    return rgbImage


def apply_cv2_watershed(compartmentImages: np.ndarray,
                        watershedMarkers: np.ndarray) -> np.ndarray:
    """Perform watershed using cv2

    Args:
        compartmentImages: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        watershedMarkers: a 3 dimensional numpy array containing the cv2
            markers arranged as (z, x, y).
    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            segmented cells. masks in different z positions are
            independent
    """

    watershedOutput = np.zeros(watershedMarkers.shape)
    for z in range(compartmentImages.shape[0]):
        rgbImage = convert_grayscale_to_rgb(compartmentImages[z, :, :])
        watershedOutput[z, :, :] = cv2.watershed(rgbImage,
                                                 watershedMarkers[z, :, :].
                                                 astype('int32'))
        watershedOutput[z, :, :][watershedOutput[z, :, :] <= 1] = 0

    return watershedOutput


def get_overlapping_objects(segmentationZ0: np.ndarray,
                            segmentationZ1: np.ndarray,
                            n0: int) -> Tuple[np.float64, 
                                              np.float64, np.float64]:
    """compare cell labels in adjacent image masks

    Args:
        segmentationZ0: a 2 dimensional numpy array containing a
            segmentation mask in position Z
        segmentationZ1: a 2 dimensional numpy array containing a
            segmentation mask adjacent tosegmentationZ0
        n0: an integer with the index of the object (cell/nuclei)
            to be compared between the provided segmentation masks

    Returns:
        a tuple (n1, f0, f1) containing the label of the cell in Z1
        overlapping n0 (n1), the fraction of n0 overlaping n1 (f0) and
        the fraction of n1 overlapping n0 (f1)
    """

    z1Indexes = np.unique(segmentationZ1[segmentationZ0 == n0])

    z1Indexes = z1Indexes[z1Indexes > 0]

    if z1Indexes.shape[0] > 0:

        # calculate overlap fraction
        n0Area = np.count_nonzero(segmentationZ0 == n0)
        n1Area = np.zeros(len(z1Indexes))
        overlapArea = np.zeros(len(z1Indexes))

        for ii in range(len(z1Indexes)):
            n1 = z1Indexes[ii]
            n1Area[ii] = np.count_nonzero(segmentationZ1 == n1)
            overlapArea[ii] = np.count_nonzero((segmentationZ0 == n0) *
                                               (segmentationZ1 == n1))

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
            return (z1Indexes[indexSorted[0]],
                    n0OverlapFraction[indexSorted[0]],
                    n1OverlapFraction[indexSorted[0]])
        else:
            return (False, False, False)
    else:
        return (False, False, False)


def combine_2d_segmentation_masks_into_3d(segmentationOutput:
                                          np.ndarray) -> np.ndarray:
    """Take a 3 dimensional segmentation masks and relabel them so that
    nuclei in adjacent sections have the same label if the area their
    overlap surpases certain threshold

    Args:
        segmentationOutput: a 3 dimensional numpy array containing the
            segmentation masks arranged as (z, x, y).
    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            relabeled segmented cells
    """

    # Initialize empty array with size as segmentationOutput array
    segmentationCombinedZ = np.zeros(segmentationOutput.shape)

    # copy the mask of the section farthest to the coverslip to start
    segmentationCombinedZ[-1, :, :] = segmentationOutput[-1, :, :]

    # starting far from coverslip
    for z in range(segmentationOutput.shape[0]-1, 0, -1):

        # get non-background cell indexes
        zIndex = np.unique(segmentationOutput[z, :, :])[
                                np.unique(segmentationOutput[z, :, :]) > 0]

        # compare each cell in z0
        for n0 in zIndex:
            n1, f0, f1 = get_overlapping_objects(segmentationCombinedZ[z, :, :],
                                                 segmentationOutput[z-1, :, :],
                                                 n0)
            if n1:
                segmentationCombinedZ[z-1, :, :][
                    (segmentationOutput[z-1, :, :] == n1)] = n0

    return segmentationCombinedZ


def segment_using_ilastik(imageStackIn: np.ndarray) -> np.ndarray:
    return None


def segment_using_unet(imageStackIn: np.ndarray) -> np.ndarray:
    return None


def segment_using_cellpose(imageStackIn: np.ndarray,
                           params: dict) -> np.ndarray:
    """Perform segmentation using cellpose. Code adapted from
    https://nbviewer.jupyter.org/github/MouseLand/cellpose/blob/
    master/notebooks/run_cellpose.ipynb
    Args:
        imageStackIn: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        params: a dictionary with the parameters for segmentation. 
                Available parameters:
                    channel
                    diameter
                    flow_threshold
                    cellprob_threshold

    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """
    channelName = params['channel'].lower()

    # Define cellpose model
    if any([channelName == 'dapi',
            channelName == 'lamin']):
        model = models.Cellpose(gpu=False, model_type='nuclei')
    if any([channelName == 'polyt',
            channelName == 'polya',
            channelName == 'ecadherin',
            channelName == 'cd45',
            channelName == 'wga',
            channelName == 'cona']):
        model = models.Cellpose(gpu=False, model_type='cyto')

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    channels = [0, 0]  # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    # channels = [[0,0],[0,0]]

    # if diameter is set to None, the size of the cells is estimated on a per
    # image basis you can set the average cell `diameter` in pixels yourself
    # (recommended) diameter can be a list or a single number for all images

    # put list of images in cellpose format
    imageList = np.split(imageStackIn, imageStackIn.shape[0])

    masks, flows, styles, diams = model.eval(imageList,
                                             diameter=params['diameter'],
                                             channels=channels,
                                             flow_threshold=
                                                params['flow_threshold'],
                                             cellprob_threshold=
                                                params['cellprob_threshold'])
    # combine masks into array
    masksArray = np.stack(masks)

    return masksArray


def apply_machine_learning_segmentation(imageStackIn: np.ndarray,
                                        params: dict) -> np.ndarray:
    """Select segmentation algorithm to use
    Args:
        imageStackIn: a 3 dimensional numpy array containing the images
            arranged as (z, x, y).
        params: dictionary with key:value pairs with parameters to be passed
            to the segmentation code. Keys used are 'method', 'diameter',
            'channel'

    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y)
    """
    if params['method'] == 'ilastik':
        segmentOutput = segment_using_ilastik(imageStackIn, params)
    elif params['method'] == 'cellpose':
        segmentOutput = segment_using_cellpose(imageStackIn, params)
    elif params['method'] == 'unet':
        segmentOutput = segment_using_unet(imageStackIn, params)

    return segmentOutput
