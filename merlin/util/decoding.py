import numpy as np
import pandas
import cv2
from typing import Tuple
from typing import Dict
from skimage import measure
from sklearn.neighbors import NearestNeighbors

from merlin.util import binary
from merlin.data import codebook as mcodebook

"""
Utility functions for pixel based decoding.
"""


def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x/norm
    else:
        return x


class PixelBasedDecoder(object):

    def __init__(self, codebook: mcodebook.Codebook,
                 scaleFactors: np.ndarray=None):
        self._codebook = codebook
        self._decodingMatrix = self._calculate_normalized_barcodes()
        self._barcodeCount = self._decodingMatrix.shape[0]
        self._bitCount = self._decodingMatrix.shape[1]

        if scaleFactors is None:
            self._scaleFactors = np.ones(self._decodingMatrix.shape[1])
        else:
            self._scaleFactors = scaleFactors

    def decode_pixels(self, imageData: np.ndarray,
                      scaleFactors: np.ndarray=None,
                      distanceThreshold: float=0.5176):
        """Assign barcodes to the pixels in the provided image stock.

        Each pixel is assigned to the nearest barcode from the codebook if
        the distance between the normalized pixel trace and the barcode is
        less than the distance threshold.

        Args:
            imageData: input image stack. The first dimension indexes the bit
                number while the second and third dimensions contain the
                corresponding image.
            scaleFactors: factors to rescale each bit prior to normalization.
                The length of scaleFactors must be equal to the number of bits.
            distanceThreshold: the maximum distance between an assigned pixel
                and the nearest barcode. Pixels for which the nearest barcode
                is greater than distanceThreshold are left unassigned.
        Returns:
            Four results are returned as a tuple (decodedImage, pixelMagnitudes,
                normalizedPixelTraces, distances). decodedImage is an image
                indicating the barcode index assigned to each pixel.
                pixelMagnitudes is an image where each pixel is the norm of
                the pixel trace after scaling by the provided scaleFactors.
                normalizedPixelTraces is an image stack containing the
                normalized intensities for each pixel. distances is an
                image containing the distance for each pixel to the assigned
                barcode.
        """
        if scaleFactors is None:
            scaleFactors = self._scaleFactors

        filteredImages = np.array([cv2.GaussianBlur(x, (5, 5), 1)
                                   for x in imageData])

        pixelTraces = np.reshape(
                filteredImages, 
                (filteredImages.shape[0], np.prod(filteredImages.shape[1:])))
        scaledPixelTraces = np.transpose(
                np.array([p/s for p, s in zip(pixelTraces, scaleFactors)]))
        pixelMagnitudes = np.array(
                [np.linalg.norm(x) for x in scaledPixelTraces])
        pixelMagnitudes[pixelMagnitudes == 0] = 1
        normalizedPixelTraces = scaledPixelTraces/pixelMagnitudes[:, None]
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(self._decodingMatrix)
        distances, indexes = neighbors.kneighbors(
                normalizedPixelTraces, return_distance=True)
        decodedImage = np.reshape(
            np.array([i[0] if d[0] <= distanceThreshold else -1
                      for i, d in zip(indexes, distances)]),
            filteredImages.shape[1:])

        pixelMagnitudes = np.reshape(pixelMagnitudes, filteredImages.shape[1:])
        normalizedPixelTraces = np.moveaxis(normalizedPixelTraces, 1, 0)
        normalizedPixelTraces = np.reshape(
                normalizedPixelTraces, filteredImages.shape)
        distances = np.reshape(distances, filteredImages.shape[1:])
        return decodedImage, pixelMagnitudes, normalizedPixelTraces, distances

    # TODO  barcodes here has two different meanings. One of these should be
    # renamed.
    def extract_barcodes_with_index(
            self, barcodeIndex: int, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zPosition: float,
            cropWidth: float, globalAligner,
            segmenter=None) -> pandas.DataFrame:
        """Extract the barcode information from the decoded image for barcodes
        that were decoded to the specified barcode index.

        Args:
            barcodeIndex: the index of the barcode to extract the corresponding
                barcodes
            decodedImage: the image indicating the barcode index assigned to
                each pixel
            pixelMagnitudes: an image containing norm of the intensities for
                each pixel across all bits after scaling by the scale factors
            pixelTraces: an image stack containing the normalized pixel
                intensity traces
            distances: an image indicating the distance between the normalized
                pixel trace and the assigned barcode for each pixel
            fov: the index of the field of view
            zPosition: the z position
            cropWidth: the number of pixels around the edge of each image within
                which barcodes are excluded from the output list.
            globalAligner: the aligner used for converted to local x,y
                coordinates to global x,y coordinates
            segmenter: the cell segmenter for assigning a cell for each of the
                identified barcodes
        Returns:
            a pandas dataframe containing all the barcodes decoded with the
                specified barcode index
        """
        properties = measure.regionprops(
                measure.label(decodedImage == barcodeIndex),
                intensity_image=pixelMagnitudes)
        dList = [self._bc_properties_to_dict(p, barcodeIndex, fov, zPosition,
                                             distances, pixelTraces,
                                             globalAligner, segmenter)
                 for p in properties if self._position_within_crop(
                    p.centroid, cropWidth, decodedImage.shape)]
        barcodeInformation = pandas.DataFrame(dList)

        return barcodeInformation

    @staticmethod
    def _position_within_crop(position: np.ndarray, cropWidth: float,
                              imageSize: Tuple[int]) -> bool:
        return cropWidth < position[0] < imageSize[0] - cropWidth \
                and cropWidth < position[1] < imageSize[1] - cropWidth

    def _bc_properties_to_dict(self, properties, bcIndex: int, fov: int,
                               zPosition: float, distances: np.ndarray,
                               pixelTraces: np.ndarray, globalAligner,
                               segmenter=None) -> Dict:
        # centroid is reversed since skimage regionprops returns the centroid
        # as (r,c)
        centroid = properties.weighted_centroid[::-1]
        globalCentroid = globalAligner.fov_coordinates_to_global(
                fov, centroid)
        d = [distances[x[0], x[1]] for x in properties.coords]
        outputDict = {'barcode': binary.bit_list_to_int(
                            self._codebook.get_barcode(bcIndex)),
                      'barcode_id': bcIndex,
                      'fov': fov,
                      'mean_intensity': properties.mean_intensity,
                      'max_intensity': properties.max_intensity,
                      'area': properties.area,
                      'mean_distance': np.mean(d),
                      'min_distance': np.min(d),
                      'x': centroid[0],
                      'y': centroid[1],
                      'z': zPosition,
                      'global_x': globalCentroid[0],
                      'global_y': globalCentroid[1],
                      'global_z': zPosition,
                      'cell_index': -1}

        for i in range(len(pixelTraces)):
            outputDict['intensity_' + str(i)] = \
                np.mean([pixelTraces[i, x[0], x[1]]
                        for x in properties.coords])

        if segmenter is not None:
            outputDict['cell_index'] = segmenter \
                    .get_cell_containing_position(
                            globalCentroid[0], globalCentroid[1])

        return outputDict

    def _calculate_normalized_barcodes(
            self, ignoreBlanks=False, includeErrors=False):
        """Normalize the barcodes present in the provided codebook so that
        their L2 norm is 1.

        Args:
            ignoreBlanks: Flag to set if the barcodes corresponding to blanks
                should be ignored. If True, barcodes corresponding to a name
                that contains 'Blank' are ignored.
            includeErrors: Flag to set if barcodes corresponding to single bit 
                errors should be added.
        Returns:
            A 2d numpy array where each row is a normalized barcode and each
                column is the corresponding normalized bit value.
        """
        
        barcodeSet = self._codebook.get_barcodes(ignoreBlanks=ignoreBlanks)
        magnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
       
        if not includeErrors:
            weightedBarcodes = np.array(
                [normalize(x) for x, m in zip(barcodeSet, magnitudes)])
            return weightedBarcodes

        else:
            barcodesWithSingleErrors = []
            for b in barcodeSet:
                barcodeSet = np.array([b]
                                      + [binary.flip_bit(b, i)
                                         for i in range(len(b))])
                bcMagnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
                weightedBC = np.array(
                    [x/m for x, m in zip(barcodeSet, bcMagnitudes)])
                barcodesWithSingleErrors.append(weightedBC)
            return np.array(barcodesWithSingleErrors)
            
    def extract_refactors(
            self, imageSet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the scale factors that would result in the mean
        on bit intensity for each bit to be equal to one.

        If the scale factors for this decoder are not set to 1, then the
        calculated scale factors are dependent on the input scale factors
        used for the decoding.

        Args:
            imageSet: the image stack to decode in order to determine the
                scale factors

        Returns:
            a tupble containing an a array of the scale factors where the i'th
                entry is the scale factor for bit i and an array indicating
                the abundance of each barcode determined during the decoding
        """
        di, pm, npt, d = self.decode_pixels(imageSet)

        sumPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        barcodesSeen = np.zeros(self._barcodeCount)
        for b in range(self._barcodeCount):
            barcodeRegions = [x for x in measure.regionprops(
                        measure.label((di == b).astype(np.int))) if x.area >= 4]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                meanPixelTrace = \
                    np.mean([npt[:, y[0], y[1]]*pm[y[0], y[1]]
                             for y in br.coords], axis=0)
                normPixelTrace = meanPixelTrace/np.linalg.norm(meanPixelTrace)
                sumPixelTraces[b, :] += normPixelTrace/barcodesSeen[b]

        sumPixelTraces[self._decodingMatrix == 0] = np.nan
        onBitIntensity = np.nanmean(sumPixelTraces, axis=0)
        refactors = onBitIntensity/np.mean(onBitIntensity)

        return refactors, barcodesSeen