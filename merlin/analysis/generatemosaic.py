import numpy as np
import cv2
from typing import Tuple

from merlin.core import analysistask


ExtentTuple = Tuple[float, float, float, float]


class GenerateMosaic(analysistask.AnalysisTask):

    """
    An analysis task that generates mosaic images by compiling different
    field of views.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'microns_per_pixel' not in self.parameters:
            self.parameters['microns_per_pixel'] = 3

        self.mosaicMicronsPerPixel = self.parameters['microns_per_pixel'] 

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30
    
    def get_dependencies(self):
        return [self.parameters['global_align_task'],
                self.parameters['warp_task']]

    def _micron_to_mosaic_pixel(self, micronCoordinates,
                                micronExtents) -> Tuple[int, int]:
        """Calculates the mosaic coordinates in pixels from the specified
        global coordinates.
        """
        return tuple([int((c-e)/self.mosaicMicronsPerPixel)
                      for c, e in zip(micronCoordinates, micronExtents[:2])])

    def _micron_to_mosaic_transform(self, micronExtents: ExtentTuple) \
            -> np.ndarray:
        s = 1/self.mosaicMicronsPerPixel
        return np.float32(
                [[s*1, 0, -s*micronExtents[0]],
                 [0, s*1, -s*micronExtents[0]],
                 [0, 0, 1]])

    def _transform_image_to_mosaic(
            self, inputImage: np.ndarray, fov: int, alignTask,
            micronExtents: ExtentTuple, mosaicDimensions: Tuple[int, int])\
            -> np.ndarray:
        transform = \
                np.matmul(self._micron_to_mosaic_transform(micronExtents),
                          alignTask.fov_to_global_transform(fov))
        return cv2.warpAffine(
                inputImage, transform[:2, :], mosaicDimensions)

    def run_analysis(self):
        alignTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])
        warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
        micronExtents = alignTask.get_global_extent()
        mosaicDimensions = tuple(self._micron_to_mosaic_pixel(
                micronExtents[-2:], micronExtents))

        imageDescription = self.dataSet._analysis_tiff_description(
                len(self.dataSet.get_z_positions()),
                len(self.dataSet.get_data_organization().get_data_channels()))

        with self.dataSet._writer_for_analysis_images(
                self, 'mosaic') as outputTif:
            for d in self.dataSet.get_data_organization().get_data_channels():
                for z in range(len(self.dataSet.get_z_positions())):
                    mosaic = np.zeros(
                            np.flip(
                                mosaicDimensions, axis=0), dtype=np.uint16)
                    for f in self.dataSet.get_fovs():
                        inputImage = warpTask.get_aligned_image(f, d, z)
                        transformedImage = self._transform_image_to_mosaic(
                            inputImage, f, alignTask, micronExtents,
                            mosaicDimensions)

                        divisionMask = np.bitwise_and(
                                transformedImage > 0, mosaic > 0)
                        cv2.add(mosaic, transformedImage, dst=mosaic,
                                mask=np.array(
                                    transformedImage > 0).astype(np.uint8))
                        dividedMosaic = cv2.divide(mosaic, 2)
                        mosaic[divisionMask] = dividedMosaic[divisionMask]
                    outputTif.save(mosaic, photometric='MINISBLACK',
                                   metadata=imageDescription)
