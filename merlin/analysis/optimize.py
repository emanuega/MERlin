import random
import os

import numpy as np

from skimage import morphology
from skimage import feature
from skimage import measure
from sklearn import preprocessing
import cv2

from merlin.core import analysistask
from merlin.util import decoding


class Optimize(analysistask.AnalysisTask):

    '''
    An analysis task for optimizing the parameters used for assigning barcodes
    to the image data.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.iterationCount = parameters.get('iteration_count', 20)
        self.fovPerIteration = parameters.get('fov_per_iteration', 10) 

        self.bitCount = len(self.dataSet.get_bit_names())
        self.barcodeCount = self.dataSet.codebook.shape[0]

        self.decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)

        self.preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])

    def get_estimated_memory(self):
        return 2000

    def get_estimated_time(self):
        return 120

    def get_dependencies(self):
        return [self.parameters['preprocess_task']]

    def run_analysis(self):
        initialScaleFactors = np.ones(self.bitCount)
        scaleFactors = np.ones((self.iterationCount, self.bitCount))
        barcodeCounts = np.ones((self.iterationCount, self.barcodeCount))
        for i in range(1,self.iterationCount):
            fovIndexes = random.sample(
                    list(self.dataSet.get_fovs()), self.fovPerIteration)
            r = [self.extract_refactors_for_fov(f, scaleFactors[i-1,:]) \
                    for f in fovIndexes]
            scaleFactors[i,:] = scaleFactors[i-1,:]\
                    *np.mean([x[0] for x in r], axis=0)
            barcodeCounts[i,:] = np.mean([x[1] for x in r],axis=0)

        self.dataSet.save_analysis_result(scaleFactors, 'scale_factors',
                self.analysisName)
        self.dataSet.save_analysis_result(barcodeCounts, 'barcode_counts',
                self.analysisName)

    def get_scale_factors(self):
        '''Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the 
            scale factor corresponding to the i'th bit.
        '''
        return self.dataSet.load_analysis_result('scale_factors',
                self.analysisName)[-1,:]

    def get_scale_factor_history(self):
        '''Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the 
            scale factor corresponding to the i'th bit in the j'th 
            iteration.
        '''
        return self.dataSet.load_analysis_result('scale_factors',
                self.analysisName)

    def get_barcode_count_history(self):
        ''' Get the set of barcode counts for each iteration of the 
        optimization.
        '''
        return self.dataSet.load_analysis_result('barcode_counts',
                self.analysisName)

    def extract_refactors_for_fov(self, fov, scaleFactors=None):
        imageSet = np.array(self.preprocessTask.get_processed_image_set(fov))
        di, pm, npt, d = self.decoder.decode_pixels(
                imageSet, scaleFactors=scaleFactors)

        sumPixelTraces = np.zeros((self.barcodeCount, self.bitCount))
        barcodesSeen = np.zeros(self.barcodeCount)
        for b in range(self.barcodeCount):
            barcodeRegions = [x \
                    for x in measure.regionprops(
                        measure.label((di==b).astype(np.int))) if x.area >= 4]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                meanPixelTrace = \
                    np.mean([npt[:, y[0], y[1]]*pm[y[0],y[1]] \
                    for y in br.coords], axis=0)
                normPixelTrace = meanPixelTrace/np.linalg.norm(meanPixelTrace)
                sumPixelTraces[b,:] += normPixelTrace/barcodesSeen[b]

        sumPixelTraces[self.decoder.decodingMatrix == 0] = np.nan
        onBitIntensity = np.nanmean(sumPixelTraces, axis=0)
        refactors = onBitIntensity/np.mean(onBitIntensity)
        return refactors, barcodesSeen

                    
