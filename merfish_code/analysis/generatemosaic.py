import PIL 
import numpy as np
import cv2

from merfish_code.core import analysistask

class GenerateMosaic(analysistask.AnalysisTask):

    '''An analysis task that generates mosaic images by compiling different
    field of views.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.mosaicMicronsPerPixel = 3 

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30
    
    def _micron_to_mosaic_pixel(self, micronCoordinates):
        '''Calculates the mosaic coordinates in pixels from the specified
        global coordinates.

        Prior to calling this function, self.micronExtents must be set.
        '''
        return tuple([int((c-e)/self.mosaicMicronsPerPixel) \
                for c,e in zip(micronCoordinates, self.micronExtents[:2])])


    def _micron_to_mosaic_transform(self):
        s = 1/self.mosaicMicronsPerPixel
        return s*np.float32(
                [[1, 0, -self.micronExtents[0]], \
                [0, 1, -self.micronExtents[0]], \
                [0, 0, 1/s]])

    def _transform_image_to_mosaic(self, inputImage, fov):
        imageOffset = self._micron_to_mosaic_pixel(
                self.alignTask.fov_coordinates_to_global(fov, (0,0)))

        transform = \
                np.matmul(self._micron_to_mosaic_transform(), 
                    self.alignTask.fov_to_global_transform(fov))
        return cv2.warpAffine(
                inputImage, transform[:2,:], self.mosaicDimensions)

    def run_analysis(self):
        self.alignTask = self.dataSet.load_analysis_task(
                self.parameters['align_task'])
        self.micronExtents = self.alignTask.get_global_extent()
        self.mosaicDimensions = tuple(self._micron_to_mosaic_pixel(
                self.micronExtents[-2:]))

        mosaic = np.zeros(self.mosaicDimensions, dtype=np.uint16)

        for f in self.dataSet.get_fovs():
            inputImage = self.dataSet.get_raw_image(0, f, 0)
            transformedImage = self._transform_image_to_mosaic(inputImage, f)
            divisionMask = np.bitwise_and(transformedImage>0, mosaic>0)
            cv2.add(mosaic, transformedImage, dst=mosaic,
                    mask=np.array(transformedImage>0).astype(np.uint8))
            dividedMosaic = cv2.divide(mosaic, 2)
            mosaic[divisionMask] = dividedMosaic[divisionMask]

        return mosaic

