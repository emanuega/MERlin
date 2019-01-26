import random
import numpy as np
import multiprocessing

from merlin.core import analysistask
from merlin.util import decoding


class Optimize(analysistask.InternallyParallelAnalysisTask):

    """
    An analysis task for optimizing the parameters used for assigning barcodes
    to the image data.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'iteration_count' not in self.parameters:
            self.parameters['iteration_count'] = 20
        if 'fov_per_iteration' not in self.parameters:
            self.parameters['fov_per_iteration'] = 10
        if 'estimate_initial_scale_factors_from_cdf' not in self.parameters:
            self.parameters['estimate_initial_scale_factors_from_cdf'] = False

        self.iterationCount = self.parameters['iteration_count']
        self.fovPerIteration = self.parameters['fov_per_iteration']

    def get_estimated_memory(self):
        return 4000*self.coreCount

    def get_estimated_time(self):
        return 60 

    def get_dependencies(self):
        return [self.parameters['preprocess_task']]

    def run_analysis(self):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])

        codebook = self.dataSet.get_codebook()
        bitCount = codebook.get_bit_count()
        barcodeCount = codebook.get_barcode_count()
        decoder = decoding.PixelBasedDecoder(codebook)

        scaleFactors = np.ones((self.iterationCount, bitCount))
        if self.parameters['estimate_initial_scale_factors_from_cdf']:
            scaleFactors[0, :] = self._calculate_initial_scale_factors()
        barcodeCounts = np.ones((self.iterationCount, barcodeCount))
        pool = multiprocessing.Pool(processes=self.coreCount)
        for i in range(1, self.iterationCount):
            fovIndexes = random.sample(
                    list(self.dataSet.get_fovs()), self.fovPerIteration)
            zIndexes = np.random.choice(
                    list(range(len(self.dataSet.get_z_positions()))),
                    self.fovPerIteration)
            decoder._scaleFactors = scaleFactors[i - 1, :]
            r = pool.starmap(decoder.extract_refactors,
                             ([preprocessTask.get_processed_image_set(
                                 f, zIndex=z)]
                                 for f, z in zip(fovIndexes, zIndexes)))
            scaleFactors[i, :] = scaleFactors[i-1, :] \
                                 * np.mean([x[0] for x in r], axis=0)
            barcodeCounts[i, :] = np.mean([x[1] for x in r], axis=0)

        self.dataSet.save_numpy_analysis_result(scaleFactors, 'scale_factors',
                                                self.analysisName)
        self.dataSet.save_numpy_analysis_result(barcodeCounts, 'barcode_counts',
                                                self.analysisName)

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])

        bitCount = self.dataSet.get_codebook().get_bit_count()
        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = preprocessTask.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
            # Add two to match matlab code.
            initialScaleFactors[i] = \
                np.argmin(np.abs(cumulativeHistogram-0.9)) + 2

        return initialScaleFactors

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the 
            scale factor corresponding to the i'th bit.
        """
        return self.dataSet.load_numpy_analysis_result(
            'scale_factors', self.analysisName)[-1, :]

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the 
            scale factor corresponding to the i'th bit in the j'th 
            iteration.
        """
        return self.dataSet.load_numpy_analysis_result('scale_factors',
                                                       self.analysisName)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        return self.dataSet.load_numpy_analysis_result('barcode_counts',
                                                       self.analysisName)
