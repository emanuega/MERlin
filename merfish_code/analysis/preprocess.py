import os

import cv2
import numpy as np

from merfish_code.core import analysistask

class Preprocess(analysistask.ParallelAnalysisTask):


    '''
    An abstract class for preparing data for barcode calling. 
    '''

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])
    
    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_analysis_result('pixel_histogram',
                    self.analysisName, fov, 'histograms')
        
        pixelHistogram = np.array(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f).astype(np.float64)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_analysis_result(histogram, 'pixel_histogram',
                self.analysisName, fov, 'histograms')


class DeconvolutionPreprocess(Preprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.highPassSigma = 3
        self.deconSigma = 2
        #TODO -  this should be based on a convergence measure
        self.deconIterations = 20

    def fragment_count(self):
        return len(self.dataSet.get_fovs())
    
    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_processed_image_set(self, fov):
        return self.dataSet.get_analysis_image_set(
                self, 'processed_image', fov)

    def get_processed_image(self, fov, dataChannel, zIndex):
        return self.dataSet.get_analysis_image(
                self, 'processed_image', fov, 
                len(self.dataSet.get_z_positions()), dataChannel, zIndex)

    def run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

        imageDescription = self.dataSet._analysis_tiff_description(
                len(self.dataSet.get_bit_names()),
                len(self.dataSet.get_data_channels()))

        histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
        pixelHistogram = np.zeros(
                (len(self.dataSet.get_bit_names()), len(histogramBins)-1))

        with self.dataSet._writer_for_analysis_images(
                self, 'processed_image', fragmentIndex) as outputTif:
            for bi,b in enumerate(self.dataSet.get_bit_names()):
                dataChannel = self.dataSet.get_data_channel_for_bit(b)
                for i in range(len(self.dataSet.get_z_positions())):
                    inputImage = warpTask.get_aligned_image(
                            fragmentIndex, dataChannel, i)
                    filteredImage = inputImage.astype(float) \
                            - cv2.GaussianBlur(inputImage,
                                (int(5*self.highPassSigma), \
                                        int(5*self.highPassSigma)),
                                self.highPassSigma).astype(float)
                    filteredImage[filteredImage < 0] = 0
                    deconvolvedImage = deconvolve_lr(
                            filteredImage, self.deconSigma*4+1,
                            self.deconSigma, self.deconIterations)\
                                    .astype(np.uint16)
                    
                    outputTif.save(
                            deconvolvedImage, photometric='MINISBLACK',
                            metadata=imageDescription)

                    pixelHistogram[bi,:] += np.histogram(
                            deconvolvedImage, bins=histogramBins)[0]

        self._save_pixel_histogram(pixelHistogram, fragmentIndex)

#TODO - move the utility functions below to their own file and clean
#up the above into utility functions

def deconvolve_lr(image, windowSize, sigmaG, iterationCount):
    '''Ported from Matlab deconvlucy'''
    eps = np.finfo(float).eps
    Y = np.copy(image)
    J1 = np.copy(image)
    J2 = np.copy(image)
    wI = np.copy(image)
    imR = np.copy(image)
    reblurred = np.copy(image)
    tmpMat1 = np.zeros(image.shape, dtype=float)
    tmpMat2 = np.zeros(image.shape, dtype=float)
    T1 = np.zeros(image.shape, dtype=float)
    T2 = np.zeros(image.shape, dtype=float)
    l = 0
    for i in range(iterationCount):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1)/(np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        cv2.GaussianBlur(Y, (windowSize, windowSize), sigmaG, reblurred) 
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        cv2.GaussianBlur(imR, (windowSize, windowSize), sigmaG, imR)
        J2 = J1
        np.multiply(Y, imR, out=J1)
        T2 = T1
        np.subtract(J1, Y, out=T1)
    return J1


                
        
