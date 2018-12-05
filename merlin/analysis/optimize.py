import random
import os
import numpy as np
import multiprocessing

from skimage import morphology
from skimage import feature

from sklearn import preprocessing
import cv2

from merlin.core import analysistask
from merlin.util import decoding


class Optimize(analysistask.InternallyParallelAnalysisTask):

    '''
    An analysis task for optimizing the parameters used for assigning barcodes
    to the image data.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'iteration_count' not in self.parameters:
            parameters['iteration_count'] = 20
        if 'fov_per_iteration' not in self.parameters:
            parameters['fov_per_iteration'] = 10

        self.iterationCount = parameters['iteration_count']
        self.fovPerIteration = parameters['fov_per_iteration'] 

    def get_estimated_memory(self):
        return 4000*self.coreCount

    def get_estimated_time(self):
        return 60 

    def get_dependencies(self):
        return [self.parameters['preprocess_task']]

    def run_analysis(self):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])

        bitCount = len(self.dataSet.get_bit_names())
        barcodeCount = self.dataSet.codebook.shape[0]
        decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)

        initialScaleFactors = np.ones(bitCount)
        scaleFactors = np.ones((self.iterationCount, bitCount))
        barcodeCounts = np.ones((self.iterationCount, barcodeCount))
        pool = multiprocessing.Pool(processes=self.coreCount)
        for i in range(1,self.iterationCount):
            fovIndexes = random.sample(
                    list(self.dataSet.get_fovs()), self.fovPerIteration)
            zIndexes = np.random.choice(
                    list(range(len(self.dataSet.get_z_positions()))),
                    self.fovPerIteration)
            self.decoder.scaleFactors = scaleFactors[i-1,:]
            r = pool.starmap(self.decoder.extract_refactors, 
                    ([preprocessTask.get_processed_image_set(f, zIndex=z)]
                        for f,z in zip(fovIndexes, zIndexes))) 
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
        '''Get the set of barcode counts for each iteration of the 
        optimization.
        '''
        return self.dataSet.load_analysis_result('barcode_counts',
                self.analysisName)



                    
