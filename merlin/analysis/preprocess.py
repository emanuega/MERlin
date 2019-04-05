import os

import cv2
import numpy as np

from merlin.core import analysistask
from merlin.util import deconvolve


class Preprocess(analysistask.ParallelAnalysisTask):

    """
    An abstract class for preparing data for barcode calling. 
    """

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])
    
    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result(
                'pixel_histogram', self.analysisName, fov, 'histograms')
        
        pixelHistogram = np.zeros(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(
            histogram, 'pixel_histogram', self.analysisName, fov, 'histograms')


class DeconvolutionPreprocess(Preprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        if 'decon_sigma' not in self.parameters:
            self.parameters['decon_sigma'] = 2
        if 'decon_filter_size' not in self.parameters:
            self.parameters['decon_filter_size'] = \
                int(2 * np.ceil(2 * self.parameters['decon_sigma']) + 1)
        if 'decon_iterations' not in self.parameters:
            self.parameters['decon_iterations'] = 20

        self._highPassSigma = self.parameters['highpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']

    def fragment_count(self):
        return len(self.dataSet.get_fovs())
    
    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_processed_image_set(self, fov, zIndex=None) -> np.ndarray:
        if zIndex is None:
            return self.dataSet.get_analysis_image_set(
                    self, 'processed_image', fov)
        else:
            return np.array([self.get_processed_image(fov,
                self.dataSet.get_data_organization()
                        .get_data_channel_for_bit(b), zIndex)
                    for b in self.dataSet.get_codebook().get_bit_names()])

    def get_processed_image(self, fov, dataChannel, zIndex):
        return self.dataSet.get_analysis_image(
                self, 'processed_image', fov, 
                len(self.dataSet.get_z_positions()), dataChannel, zIndex)

    def _run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

        imageDescription = self.dataSet.analysis_tiff_description(
                len(self.dataSet.get_z_positions()),
                len(self.dataSet.get_codebook().get_bit_names()))

        histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
        pixelHistogram = np.zeros(
                (self.dataSet.get_codebook().get_bit_count(),
                    len(histogramBins)-1))
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        deconFilterSize = self.parameters['decon_filter_size']

        with self.dataSet.writer_for_analysis_images(
                self, 'processed_image', fragmentIndex) as outputTif:
            for bi, b in enumerate(self.dataSet.get_codebook().get_bit_names()):
                dataChannel = self.dataSet.get_data_organization()\
                        .get_data_channel_for_bit(b)
                for i in range(len(self.dataSet.get_z_positions())):
                    inputImage = warpTask.get_aligned_image(
                            fragmentIndex, dataChannel, i)
                    filteredImage = inputImage.astype(float) - cv2.GaussianBlur(
                        inputImage, (highPassFilterSize, highPassFilterSize),
                        self._highPassSigma, borderType=cv2.BORDER_REPLICATE)
                    filteredImage[filteredImage < 0] = 0
                    deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
                        filteredImage, deconFilterSize, self._deconSigma,
                        self._deconIterations).astype(np.uint16)
                    
                    outputTif.save(
                            deconvolvedImage, photometric='MINISBLACK',
                            metadata=imageDescription)

                    pixelHistogram[bi, :] += np.histogram(
                            deconvolvedImage, bins=histogramBins)[0]

        self._save_pixel_histogram(pixelHistogram, fragmentIndex)
