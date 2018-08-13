import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
Utility functions for pixel based decoding.
'''

def flip_bit(barcode, bitIndex):
    '''Flip the specified bit in the barcode.

    Args:
        barcode: A binary array where the i'th entry corresponds with the 
            value of the i'th bit
        bitIndex: The index of the bit to reverse
    '''
    bcCopy = np.copy(barcode)
    bcCopy[bitIndex] = not bcCopy[bitIndex]
    return bcCopy

def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x/norm
    else:
        return x

def extract_barcodes(codebook, ignoreBlanks=False):
    '''Extract the barcodes from a codebook.'''

    if ignoreBlanks:
        barcodeSet = np.array([x['barcode'] for i,x \
                in codebook.iterrows() if 'Blank' not in x['name']])
    else:
        barcodeSet = np.array([x['barcode'] for i,x in codebook.iterrows()])

    return barcodeSet


class PixelBasedDecoder(object):

    def __init__(self, codebook, scaleFactors=None):
        self.codebook = codebook
        self.decodingMatrix = self._calculate_normalized_barcodes()
        if scaleFactors is None:
            self.scaleFactors = np.ones(self.decodingMatrix.shape[1])
        else:
            self.scaleFactors = scaleFactors

    def decode_pixels(
            self, imageData, scaleFactors=None, distanceThreshold=0.5176):
        '''Match the pixels in imagaDate to the barcodes present in the 
        codebook. 
        '''
        if scaleFactors is None:
            scaleFactors = self.scaleFactors

        filteredImages = np.array([cv2.GaussianBlur(x, (5, 5), 1) \
                    for x in imageData])
        #For no z data
        pixelTraces = np.reshape(
                filteredImages, 
                (filteredImages.shape[0], np.prod(filteredImages.shape[1:])))
        pixelCount = pixelTraces.shape[1]
        scaledPixelTraces = np.transpose(
                np.array([p/s for p,s in zip(pixelTraces, scaleFactors)]))
        pixelMagnitudes = np.array(
                [np.linalg.norm(x) for x in scaledPixelTraces])
        pixelMagnitudes[pixelMagnitudes==0] = 1
        normalizedPixelTraces = scaledPixelTraces/pixelMagnitudes[:,None]
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(self.decodingMatrix)
        distances, indexes = neighbors.kneighbors(
                normalizedPixelTraces, return_distance=True)
        decodedImage = np.reshape(
            np.array([i[0] if d[0]<=distanceThreshold else -1 \
                    for i,d in zip(indexes, distances) ]), 
            filteredImages.shape[1:])

        pixelMagnitudes = np.reshape(pixelMagnitudes, filteredImages.shape[1:])
        normalizedPixelTraces = np.moveaxis(normalizedPixelTraces, 1, 0)
        normalizedPixelTraces = np.reshape(
                normalizedPixelTraces, filteredImages.shape)
        return decodedImage, pixelMagnitudes, normalizedPixelTraces, distances

    def _calculate_normalized_barcodes(
            self, ignoreBlanks=False, includeErrors=False):
        '''Normalize the barcodes present in the provided codebook so that 
        their L2 norm is 1.

        Args:
            codebook: The codebook that contains the set of barcodes 
            ignoreBlanks: Flag to set if the barcodes corresponding to blanks
                should be ignored. If True, barcodes corresponding to a name
                that contains 'Blank' are ignored.
            includeErrors: Flag to set if barcodes corresponding to single bit 
                errors should be added.
        Returns:
            A 2d numpy array where each row is a normalized barcode and each
            column is the corresponding normalized bit value.
        '''
        
        barcodeSet = extract_barcodes(self.codebook, ignoreBlanks=ignoreBlanks)
        magnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
       
        if not includeErrors:
            weightedBarcodes = np.array(
                [normalize(x) for x,m in zip(barcodeSet, magnitudes)])
            return weightedBarcodes

        else:
            barcodesWithSingleErrors = []
            for b in barcodeSet:
                barcodeSet = np.array([b] \
                        + [flip_bit(b, i) for i in range(len(b))])
                bcMagnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
                weightedBC = np.array(
                    [x/m for x,m in zip(barcodeSet, bcMagnitudes)])
                barcodesWithSingleErrors.append(weightedBC)
            return np.array(barcodesWithSingleErrors)
            
