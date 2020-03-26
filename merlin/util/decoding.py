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
                 scaleFactors: np.ndarray=None, backgrounds: np.ndarray=None):
        self._codebook = codebook
        self._decodingMatrix = self._calculate_normalized_barcodes()
        self._barcodeCount = self._decodingMatrix.shape[0]
        self._bitCount = self._decodingMatrix.shape[1]

        if scaleFactors is None:
            self._scaleFactors = np.ones(self._decodingMatrix.shape[1])
        else:
            self._scaleFactors = scaleFactors.copy()

        if backgrounds is None:
            self._backgrounds = np.zeros(self._decodingMatrix.shape[1])
        else:
            self._backgrounds = backgrounds.copy()

        self.refactorAreaThreshold = 4

    def decode_pixels(self, imageData: np.ndarray,
                      scaleFactors: np.ndarray=None,
                      backgrounds: np.ndarray=None,
                      distanceThreshold: float=0.5176,
                      magnitudeThreshold: float=1,
                      lowPassSigma: float=1):
        """Assign barcodes to the pixels in the provided image stock.

        Each pixel is assigned to the nearest barcode from the codebook if
        the distance between the normalized pixel trace and the barcode is
        less than the distance threshold.

        Args:
            imageData: input image stack. The first dimension indexes the bit
                number and the second and third dimensions contain the
                corresponding image.
            scaleFactors: factors to rescale each bit prior to normalization.
                The length of scaleFactors must be equal to the number of bits.
            backgrounds: background to subtract from each bit prior to applying
                the scale factors and prior to normalization. The length of
                backgrounds must be equal to the number of bits.
            distanceThreshold: the maximum distance between an assigned pixel
                and the nearest barcode. Pixels for which the nearest barcode
                is greater than distanceThreshold are left unassigned.
            magnitudeThreshold: the minimum pixel magnitude for which a
                barcode can be assigned that pixel. All pixels that fall
                below the magnitude threshold are not assigned a barcode
                in the decoded image.
            lowPassSigma: standard deviation for the low pass filter that is
                applied to the images prior to decoding.
        Returns:
            Four results are returned as a tuple (decodedImage, pixelMagnitudes,
                normalizedPixelTraces, distances). decodedImage is an image
                indicating the barcode index assigned to each pixel. Pixels
                for which a barcode is not assigned have a value of -1.
                pixelMagnitudes is an image where each pixel is the norm of
                the pixel trace after scaling by the provided scaleFactors.
                normalizedPixelTraces is an image stack containing the
                normalized intensities for each pixel. distances is an
                image containing the distance for each pixel to the assigned
                barcode.
        """
        if scaleFactors is None:
            scaleFactors = self._scaleFactors
        if backgrounds is None:
            backgrounds = self._backgrounds

        filteredImages = np.zeros(imageData.shape, dtype=np.float32)
        filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
        for i in range(imageData.shape[0]):
            filteredImages[i, :, :] = cv2.GaussianBlur(
                imageData[i, :, :], (filterSize, filterSize), lowPassSigma)

        pixelTraces = np.reshape(
                filteredImages, 
                (filteredImages.shape[0], np.prod(filteredImages.shape[1:])))
        scaledPixelTraces = np.transpose(
                np.array([(p-b)/s for p, s, b in zip(pixelTraces, scaleFactors,
                                                   backgrounds)]))

        pixelMagnitudes = np.array(
            [np.linalg.norm(x) for x in scaledPixelTraces], dtype=np.float32)
        pixelMagnitudes[pixelMagnitudes == 0] = 1

        normalizedPixelTraces = scaledPixelTraces/pixelMagnitudes[:, None]

        neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        neighbors.fit(self._decodingMatrix)

        distances, indexes = neighbors.kneighbors(
                normalizedPixelTraces, return_distance=True)

        decodedImage = np.reshape(
            np.array([i[0] if d[0] <= distanceThreshold else -1
                      for i, d in zip(indexes, distances)], dtype=np.int16),
            filteredImages.shape[1:])

        pixelMagnitudes = np.reshape(pixelMagnitudes, filteredImages.shape[1:])
        normalizedPixelTraces = np.moveaxis(normalizedPixelTraces, 1, 0)
        normalizedPixelTraces = np.reshape(
                normalizedPixelTraces, filteredImages.shape)
        distances = np.reshape(distances, filteredImages.shape[1:])

        decodedImage[pixelMagnitudes < magnitudeThreshold] = -1

        return decodedImage, pixelMagnitudes, normalizedPixelTraces, distances

    def extract_barcodes_with_index_inbatch(
            self, barcodeIndex: int, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, cropWidth: int, zIndex: int = None,
            globalAligner=None, segmenter=None, minimumArea: int = 0
    ) -> pandas.DataFrame:
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
            cropWidth: the number of pixels around the edge of each image within
                which barcodes are excluded from the output list.
            zIndex: the index of the z position
            globalAligner: the aligner used for converted to local x,y
                coordinates to global x,y coordinates
            segmenter: the cell segmenter for assigning a cell for each of the
                identified barcodes
            minimumArea: the minimum area of barcodes to identify. Barcodes
                less than the specified minimum area are ignored.
        Returns:
            a pandas dataframe containing all the barcodes decoded with the
                specified barcode index
        """
        properties = measure.regionprops(
            measure.label(decodedImage == barcodeIndex),
            intensity_image=pixelMagnitudes,
            cache=False)
        columnNames = ['barcode_id', 'fov', 'mean_intensity', 'max_intensity',
         'area', 'mean_distance', 'min_distance',
         'x', 'y', 'z', 'global_x', 'global_y', 'global_z',
         'cell_index']
        if len(pixelTraces.shape) == 3:
            intensityColumns = ['intensity_{}'.format(i) for i in
                                range(pixelTraces.shape[0])]
        else:
            intensityColumns = ['intensity_{}'.format(i) for i in
                                range(pixelTraces.shape[1])]
        if len(properties) == 0:
            return pandas.DataFrame(columns=columnNames + intensityColumns)

        centroidCoords = np.array(
            [prop.weighted_centroid for prop in properties])
        print(centroidCoords)
        if centroidCoords.shape[1] == 2:
            centroids = np.zeros((centroidCoords.shape[0], 3))
            centroids[:, 0] = zIndex
            centroids[:, [1, 2]] = centroidCoords[:, [1, 0]]
        else:
            centroids = centroidCoords[:, [0, 2, 1]]

        if globalAligner is not None:
            tForm = globalAligner.fov_to_global_transform(fov)
            toGlobal = np.ones(centroids.shape)
            toGlobal[:, [0, 1]] = centroids[:, [1, 2]]
            globalCentroids = np.matmul(tForm, toGlobal.T).T[:, [2, 0, 1]]
            globalCentroids[:, 0] = centroids[:, 0]
        else:
            globalCentroids = centroids

        allCoords = [list(p.coords) for p in properties]
        if len(distances.shape) == 2:
            d = [[distances[y[0], y[1]] for y in x] for x in allCoords]
        else:
            d = [[distances[y[0], y[1], y[2]] for y in x] for x in allCoords]

        df = pandas.DataFrame(np.zeros((len(properties), 14)), columns=columnNames)
        df['barcode_id'] = barcodeIndex
        df['fov'] = fov
        df.loc[:, ['mean_intensity', 'max_intensity', 'area']] = np.array(
            [[x.mean_intensity, x.max_intensity, x.area] for x in properties])
        df.loc[:, ['mean_distance', 'min_distance']] = np.array(
            [[np.mean(x), np.min(x)] if len(x) > 1 else [x[0], x[0]] for x in
             d])
        df.loc[:, ['x', 'y', 'z']] = centroids[:, [1, 2, 0]]
        df.loc[:, ['global_x', 'global_y', 'global_z']] = centroids[:,
                                                          [1, 2, 0]]
        df['cell_index'] = -1
        if len(pixelTraces.shape) == 3:
            intensities = [[pixelTraces[:, y[0], y[1]] for y in x] for
                           x in allCoords]
            intensities = pandas.DataFrame(
                [np.mean(x, 0) if len(x) > 1 else x[0] for x in intensities],
                columns=intensityColumns)
        else:
            intensities = [
                [pixelTraces[y[0], :, y[1], y[2]] for y in x] for x in
                allCoords]
            intensities = pandas.DataFrame(
                [np.mean(x, 0) if len(x) > 1 else x[0] for x in intensities],
                columns=intensityColumns)

        fullDF = pandas.concat([df, intensities], 1)
        fullDF = fullDF[(fullDF['x'].between(cropWidth,
                                             decodedImage.shape[0] - cropWidth,
                                             inclusive=False)) &
                        (fullDF['y'].between(cropWidth,
                                             decodedImage.shape[1] - cropWidth,
                                             inclusive=False)) &
                        (fullDF['area'] >= minimumArea)]

        if segmenter is not None:
            print('no support for segmenter in extract barcodes in batch')

        return fullDF

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
            self, decodedImage, pixelMagnitudes, normalizedPixelTraces,
            extractBackgrounds = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the scale factors that would result in the mean
        on bit intensity for each bit to be equal.

        This code follows the legacy matlab decoder.

        If the scale factors for this decoder are not set to 1, then the
        calculated scale factors are dependent on the input scale factors
        used for the decoding.

        Args:
            imageSet: the image stack to decode in order to determine the
                scale factors
        Returns:
             a tuple containing an array of the scale factors, an array
                of the backgrounds, and an array of the abundance of each
                barcode determined during the decoding. For the scale factors
                and the backgrounds, the i'th entry is the scale factor
                for bit i. If extractBackgrounds is false, the returned
                background array is all zeros.
        """

        if extractBackgrounds:
            backgroundRefactors = self._extract_backgrounds(
                decodedImage, pixelMagnitudes, normalizedPixelTraces)
        else:
            backgroundRefactors = np.zeros(self._bitCount)

        sumPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        barcodesSeen = np.zeros(self._barcodeCount)
        for b in range(self._barcodeCount):
            barcodeRegions = [x for x in measure.regionprops(
                        measure.label((decodedImage == b).astype(np.int)))
                              if x.area >= self.refactorAreaThreshold]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                meanPixelTrace = \
                    np.mean([normalizedPixelTraces[:, y[0],
                             y[1]]*pixelMagnitudes[y[0], y[1]]
                             for y in br.coords], axis=0) - backgroundRefactors
                normPixelTrace = meanPixelTrace/np.linalg.norm(meanPixelTrace)
                sumPixelTraces[b, :] += normPixelTrace/barcodesSeen[b]

        sumPixelTraces[self._decodingMatrix == 0] = np.nan
        onBitIntensity = np.nanmean(sumPixelTraces, axis=0)
        refactors = onBitIntensity/np.mean(onBitIntensity)

        return refactors, backgroundRefactors, barcodesSeen

    def _extract_backgrounds(
            self, decodedImage, pixelMagnitudes, normalizedPixelTraces
    ) -> np.ndarray:
        """Calculate the backgrounds to be subtracted for the the mean off
        bit intensity for each bit to be equal to zero.

        Args:
            imageSet: the image stack to decode in order to determine the
                scale factors
        Returns:
            an array of the backgrounds where the i'th entry is the scale factor
                for bit i.
        """
        sumMinPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        barcodesSeen = np.zeros(self._barcodeCount)
        # TODO this core functionality is very similar to that above. They
        # can be abstracted
        for b in range(self._barcodeCount):
            barcodeRegions = [x for x in measure.regionprops(
                measure.label((decodedImage == b).astype(np.int)))
                              if x.area >= 5]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                minPixelTrace = \
                    np.min([normalizedPixelTraces[:, y[0],
                            y[1]] * pixelMagnitudes[y[0], y[1]]
                            for y in br.coords], axis=0)
                sumMinPixelTraces[b, :] += minPixelTrace

        offPixelTraces = sumMinPixelTraces.copy()
        offPixelTraces[self._decodingMatrix > 0] = np.nan
        offBitIntensity = np.nansum(offPixelTraces, axis=0) / np.sum(
            (self._decodingMatrix == 0) * barcodesSeen[:, np.newaxis], axis=0)
        backgroundRefactors = offBitIntensity

        return backgroundRefactors
