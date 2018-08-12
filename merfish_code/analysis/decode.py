import numpy as np
import cv2

from merfish_code.core import analysistask
from merfish_code.util import decoding


class Decode(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.areaThreshold = 4

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])


        decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)

        imageSet = np.array([cv2.GaussianBlur(x, (5, 5), 1) \
                        for x in preprocessTask.get_images(fragmentIndex)])
        scaleFactors = optimizeTask.get_scale_factors()
        di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors)

        return di, pm, npt, d

    def _extract_barcodes(self, decodedImage, pixelMagnitudes, 
            singleErrorBarcodes, pixelTraces, 
