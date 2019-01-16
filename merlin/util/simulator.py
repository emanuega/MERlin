import os
import numpy as np
import PIL
import cv2
import tifffile
from scipy.signal import convolve2d

import merlin
from merlin.core import dataset
from merlin.data import codebook as cb

class MERFISHDataFactory(object):

    """
    A class for simulating MERFISH data sets.
    """

    def __init__(self):
        self.codebookPath = 'L26E1.csv'
        self.psfSigma = 1.2
        self.imageSize = np.array([1024, 1024])
        self.upsampleFactor = 10
        self.fluorophoreBrightness = 1000
        self.fiducialBrightness = 10000
        self.background = 100
        self.bitOrganization = [[0, 1], [0, 0], [1, 0], [1, 1],
                [2, 1], [2, 0], [3, 1], [3, 0], [4, 0], [4, 1],
                [5, 1], [5, 0], [6, 1], [6, 0], [7, 0], [7, 1]]

    def simulate_image(self, spotPositions: np.ndarray=None,
                       addNoise: bool=False) -> np.ndarray:
        """Simulate a single image consisting of point sources with a Gaussian
        point spread function

        Args:
            spotPositions: a n x 2 numpy array containing the positions to
                simulate the point sources. If not specified, 1000 random
                positions are selected.
            addNoise: flag indicating whether poisson noise should be added
                to the simulated image.
        Returns:
            the simulated image
        """
        if spotPositions is None:
            spotPositions = np.random.uniform(size=(1000, 2))
            spotPositions[:, 0] *= self.imageSize[0]
            spotPositions[:, 1] *= self.imageSize[1]

        upsampledImage = np.zeros(self.upsampleFactor*self.imageSize)
        for p in spotPositions:
            upsampledImage[int(np.floor(p[0]*self.upsampleFactor)),
                            int(np.floor(p[1]*self.upsampleFactor))] += 1000

        return self._downsample_image_stack([upsampledImage],
                                            addNoise=addNoise)[0]

    def simulate_dataset(self, datasetName, abundanceScale=1, 
            fluorophoreCount=5, fovCount=10):
        """Simulate a full MERFISH dataset"""
        dataDir = os.sep.join([merlin.DATA_HOME, datasetName])
        if not os.path.exists(dataDir):
            os.mkdir(dataDir)

        simDataset = dataset.DataSet(datasetName)
        codebook = cb.Codebook(simDataset, self.codebookPath)

        barcodeNumber = codebook.get_barcode_count()
        barcodeAbundances = abundanceScale*np.array(
                [10**np.random.uniform(3) for i in range(barcodeNumber)])
        barcodeAbundances[:10] = 0

        for i in range(fovCount):
            merfishImages, rnaPositions  = self._simulate_single_fov(
                    codebook, barcodeAbundances, fluorophoreCount)
            fiducialImage = self._simulate_fiducial_image()
            tifffile.imsave(
                    os.sep.join([dataDir, 'full_stack_' + str(i) + '.tiff']), 
                        merfishImages.astype(np.uint16))

            imageCount = np.max([x[0] for x in self.bitOrganization]) + 1
            for j in range(imageCount):
                fileName = 'Conventional_750_650_561_488_405_' + str(i) + \
                        '_' + str(j) + '.tiff'
                filePath = os.sep.join([dataDir, fileName])

                imageData = np.zeros(
                        shape=(5, *self.imageSize), dtype=np.uint16)
                firstBitIndex = [i for i,x in enumerate(self.bitOrganization) \
                        if x[0] == j and x[1] == 0][0]
                secondBitIndex = [i for i,x in enumerate(self.bitOrganization) \
                        if x[0] == j and x[1] == 1][0]

                imageData[0,:,:] = merfishImages[firstBitIndex]
                imageData[1,:,:] = merfishImages[secondBitIndex]
                imageData[2,:,:] = fiducialImage

                tifffile.imsave(filePath, imageData)

            np.save(os.sep.join(
                [dataDir, 'true_positions_' + str(i) + '.npy']), rnaPositions)

    def _simulate_fiducial_image(self):
        fiducialPositions = np.random.uniform(size=(1000,2))
        upsampledFiducials = self.fiducialBrightness*np.histogram2d(
                fiducialPositions[:,0]*self.imageSize[0],
                fiducialPositions[:,1]*self.imageSize[1],
                bins=self.upsampleFactor*self.imageSize)[0]

        return self._downsample_image_stack([upsampledFiducials])[0]

    def _simulate_single_fov(self, codebook, barcodeAbundances, 
            fluorophoreCount):
        barcodeCount = len(barcodeAbundances)
        bitNumber = codebook.get_bit_count()
        imageSize = self.imageSize

        rnaCounts = np.random.poisson(barcodeAbundances)
        rnaPositions = [np.random.uniform(size=(c, 2)) for c in rnaCounts]
        for b in range(barcodeCount):
            rnaPositions[b][:, 0] *= imageSize[0]
            rnaPositions[b][:, 1] *= imageSize[1]

        upsampledStack = np.zeros((bitNumber, *self.upsampleFactor*imageSize)) 

        for b in range(barcodeCount):
            self._add_spots_for_barcode(
                    codebook.get_barcode(b), rnaPositions[b], fluorophoreCount,
                    upsampledStack)

        imageStack = self._downsample_image_stack(upsampledStack)

        return imageStack, rnaPositions

    def _add_spots_for_barcode(self, barcode, positions, fluorophoreCount,
            upsampledStack):
        upsampledImage = np.zeros(self.upsampleFactor*self.imageSize)
        for p in positions:
            upsampledImage[int(np.floor(p[0]*self.upsampleFactor)), \
                            int(np.floor(p[1]*self.upsampleFactor))] += 1
        upsampledImage = self.fluorophoreBrightness*np.random.poisson(
                upsampledImage*fluorophoreCount)

        for i in np.where(barcode)[0]:
            np.add(upsampledStack[i], upsampledImage, out=upsampledStack[i])

    def _downsample_image_stack(self, upsampledStack, addNoise=True):
        imageStack = np.zeros((len(upsampledStack), *self.imageSize))

        for i in range(len(imageStack)):
            blurredImage = cv2.GaussianBlur(upsampledStack[i].astype(float), 
                    ksize=(51, 51), sigmaX=self.upsampleFactor*self.psfSigma)
            downsampledImage = np.array(PIL.Image.fromarray(
                    convolve2d(blurredImage, 
                        np.ones((self.upsampleFactor, self.upsampleFactor))))\
                                .resize(self.imageSize, PIL.Image.BILINEAR))
            if addNoise:
                imageStack[i] = np.random.poisson(
                        downsampledImage + self.background)
            else:
                imageStack[i] = downsampledImage + self.background

        return imageStack

