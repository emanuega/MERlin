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
    
    def get_dependencies(self):
        return [self.parameters['global_align_task'], \
                self.parameters['warp_task']]

    def _micron_to_mosaic_pixel(self, micronCoordinates):
        '''Calculates the mosaic coordinates in pixels from the specified
        global coordinates.

        Prior to calling this function, self.micronExtents must be set.
        '''
        return tuple([int((c-e)/self.mosaicMicronsPerPixel) \
                for c,e in zip(micronCoordinates, self.micronExtents[:2])])


    def _micron_to_mosaic_transform(self):
        s = 1/self.mosaicMicronsPerPixel
        return np.float32(
                [[s*1, 0, -s*self.micronExtents[0]], \
                [0, s*1, -s*self.micronExtents[0]], \
                [0, 0, 1]])

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
                self.parameters['global_align_task'])
        self.warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
        self.micronExtents = self.alignTask.get_global_extent()
        self.mosaicDimensions = tuple(self._micron_to_mosaic_pixel(
                self.micronExtents[-2:]))

        imageDescription = self.dataSet._analysis_tiff_description(
                len(self.dataSet.get_z_positions()),
                len(self.dataSet.get_data_channels()))

        with self.dataSet._writer_for_analysis_images(
                self, 'mosaic') as outputTif:
            for d in self.dataSet.get_data_channels():
                for z in range(len(self.dataSet.get_z_positions())):
                    mosaic = np.zeros(self.mosaicDimensions, dtype=np.uint16)
                    for f in self.dataSet.get_fovs():
                        inputImage = self.warpTask.get_aligned_image(f, d, z)
                        transformedImage = self._transform_image_to_mosaic(
                                inputImage, f)

                        divisionMask = np.bitwise_and(
                                transformedImage>0, mosaic>0)
                        cv2.add(mosaic, transformedImage, dst=mosaic,
                                mask=np.array(
                                    transformedImage>0).astype(np.uint8))
                        dividedMosaic = cv2.divide(mosaic, 2)
                        mosaic[divisionMask] = dividedMosaic[divisionMask]
                    outputTif.save(mosaic, photometric='MINISBLACK',
                            metadata=imageDescription)


