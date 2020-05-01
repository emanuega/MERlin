import os
import cv2
import numpy as np

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.data import codebook


class Preprocess(analysistask.ParallelAnalysisTask):

    """
    An abstract class for preparing data for barcode calling.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(
            histogram, 'pixel_histogram', self.analysisName, fov, 'histograms')

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result(
                'pixel_histogram', self.analysisName, fov, 'histograms')

        pixelHistogram = np.zeros(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                            for zIndex in range(len(self.dataSet.get_z_positions()))]
                            for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                             for b in self.get_codebook().get_bit_names()])

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                     chromaticCorrector)
        return self._preprocess_image(inputImage)


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
        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0

        self._highPassSigma = self.parameters['highpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']

    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                self._highPassSigma)
        return hpImage.astype(np.float)

    def _run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

        histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
        pixelHistogram = np.zeros(
                (self.get_codebook().get_bit_count(), len(histogramBins)-1))

        # this currently only is to calculate the pixel histograms in order
        # to estimate the initial scale factors. This is likely unnecessary
        for bi, b in enumerate(self.get_codebook().get_bit_names()):
            dataChannel = self.dataSet.get_data_organization()\
                    .get_data_channel_for_bit(b)
            for i in range(len(self.dataSet.get_z_positions())):
                inputImage = warpTask.get_aligned_image(
                        fragmentIndex, dataChannel, i)
                deconvolvedImage = self._preprocess_image(inputImage)

                pixelHistogram[bi, :] += np.histogram(
                        deconvolvedImage, bins=histogramBins)[0]

        self._save_pixel_histogram(pixelHistogram, fragmentIndex)

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
            filteredImage, deconFilterSize, self._deconSigma,
            self._deconIterations).astype(np.uint16)
        return deconvolvedImage


class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        if 'decon_iterations' not in parameters:
            self.parameters['decon_iterations'] = 2

        self._deconIterations = self.parameters['decon_iterations']

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson_guo(
            filteredImage, deconFilterSize, self._deconSigma,
            self._deconIterations).astype(np.uint16)
        return deconvolvedImage


class EstimatePixelSignificance(Preprocess):
    """
    Estimates pixel significance in units of sigma.

    In order for this to work correctly you must provide the correct
    values for the camera gain and the camera offset. You can
    verify that this is true by loading the histograms and checking
    that their shape is approximately that of a Gaussian with 0
    mean and 1 sigma.
    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'calculate_histograms' not in self.parameters:
            self.parameters['calculate_histograms'] = False
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        if 'filter_iterations' not in self.parameters:
            self.parameters['filter_iterations'] = 5

        self._cameraGain = self.parameters['camera_gain']
        self._cameraOffset = self.parameters['camera_offset']
        self._filterIterations = self.parameters['filter_iterations']
        self._highPassSigma = self.parameters['highpass_sigma']
        self._highPassFilterSize =\
            int(2 * np.ceil(2 * self._highPassSigma) + 1)

    def _run_analysis(self, fragmentIndex):
        if not self.parameters['calculate_histograms']:
            return

        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        histogramBins = np.arange(0, 1000, 0.1)
        pixelHistogram = np.zeros(
                (self.get_codebook().get_bit_count(), len(histogramBins)-1))

        # This is only used to calculate the pixel sigma value histograms,
        # which might not be that useful.
        for bi, b in enumerate(self.get_codebook().get_bit_names()):
            dataChannel = self.dataSet.get_data_organization()\
                    .get_data_channel_for_bit(b)
            for i in range(len(self.dataSet.get_z_positions())):
                inputImage = warpTask.get_aligned_image(
                        fragmentIndex, dataChannel, i)
                significanceImage = self._preprocess_image(inputImage)

                pixelHistogram[bi, :] += np.histogram(
                        significanceImage, bins=histogramBins)[0]

        self._save_pixel_histogram(pixelHistogram, fragmentIndex)

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        [fg, bg] = imagefilters.high_low_filter(inputImage,
                                                self._highPassFilterSize,
                                                self._highPassSigma,
                                                self._filterIterations)
        return imagefilters.est_significance(fg, bg)
