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
        if 'fov_crop_width' not in self.parameters:
            self.parameters['fov_crop_width'] = 0
        if 'separate_files' not in self.parameters:
            self.parameters['separate_files'] = False

        if self.parameters['microns_per_pixel'] == 'full_resolution':
            self.mosaicMicronsPerPixel = self.dataSet.get_microns_per_pixel()
        else:
            self.mosaicMicronsPerPixel = self.parameters['microns_per_pixel']

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30
    
    def get_dependencies(self):
        return [self.parameters['global_align_task'],
                self.parameters['warp_task']]

    def get_mosaic(self) -> np.ndarray:
        """Get the mosaic generated by this analysis task.

        Returns:
            a 5-dimensional array containing the mosaic. The images are arranged
            as [channel, zIndex, 1, x, y]. The order of the channels is as
            specified in the provided parameters file or in the data
            organization if no data channels are specified.
        """
        return self.dataSet.get_analysis_image_set(self, 'mosaic')

    def _micron_to_mosaic_pixel(self, micronCoordinates,
                                micronExtents) -> np.ndarray:
        """Calculates the mosaic coordinates in pixels from the specified
        global coordinates.
        """
        return np.matmul(self._micron_to_mosaic_transform(micronExtents),
                         np.append(micronCoordinates, 1)).astype(np.int32)[:2]

    def _micron_to_mosaic_transform(self, micronExtents: ExtentTuple) \
            -> np.ndarray:
        s = 1/self.mosaicMicronsPerPixel
        return np.float32(
                [[s*1, 0, -s*micronExtents[0]],
                 [0, s*1, -s*micronExtents[1]],
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

    def _run_analysis(self):
        alignTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])
        micronExtents = alignTask.get_global_extent()
        self.dataSet.save_numpy_txt_analysis_result(
            self._micron_to_mosaic_transform(micronExtents),
            'micron_to_mosaic_pixel_transform', self)

        dataOrganization = self.dataSet.get_data_organization()
        if 'data_channels' in self.parameters:
            if isinstance(self.parameters['data_channels'], str):
                dataChannels = [dataOrganization.get_data_channel_index(
                    self.parameters['data_channels'])]
            elif isinstance(self.parameters['data_channels'], int):
                dataChannels = [self.parameters['data_channels']]
            else:
                dataChannels = [dataOrganization.get_data_channel_index(x)
                                if isinstance(x, str) else x
                                for x in self.parameters['data_channels']]
        else:
            dataChannels = dataOrganization.get_data_channels()

        maximumProjection = False
        if 'z_index' in self.parameters:
            if self.parameters['z_index'] != 'maximum_projection':
                zIndexes = [self.parameters['z_index']]
            else:
                maximumProjection = True
                zIndexes = [0]
        else:
            zIndexes = range(len(self.dataSet.get_z_positions()))

        if not self.parameters['separate_files']:
            imageDescription = self.dataSet.analysis_tiff_description(
                len(zIndexes), len(dataChannels))
            with self.dataSet.writer_for_analysis_images(
                    self, 'mosaic') as outputTif:
                for d in dataChannels:
                    for z in zIndexes:
                        mosaic = self._prepare_mosaic_slice(
                            z, d, micronExtents, alignTask, maximumProjection)
                        outputTif.save(mosaic, photometric='MINISBLACK',
                                       metadata=imageDescription)
        else:
            imageDescription = self.dataSet.analysis_tiff_description(1, 1)
            for d in dataChannels:
                for z in zIndexes:
                    with self.dataSet.writer_for_analysis_images(
                        self, 'mosaic_%s_%i'
                              % (dataOrganization.get_data_channel_name(d), z))\
                            as outputTif:
                        mosaic = self._prepare_mosaic_slice(
                            z, d, micronExtents, alignTask, maximumProjection)
                        outputTif.save(mosaic, photometric='MINISBLACK',
                                       metadata=imageDescription)

    def _prepare_mosaic_slice(self, zIndex, dataChannel, micronExtents,
                              alignTask, maximumProjection):
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        chromaticCorrector = None
        if 'optimize_task' in self.parameters:
            chromaticCorrector = self.dataSet.load_analysis_task(
                self.parameters['optimize_task']).get_chromatic_corrector()

        cropWidth = self.parameters['fov_crop_width']
        mosaicDimensions = tuple(self._micron_to_mosaic_pixel(
                micronExtents[-2:], micronExtents))

        mosaic = np.zeros(np.flip(mosaicDimensions, axis=0), dtype=np.uint16)

        for f in self.dataSet.get_fovs():
            if maximumProjection:
                inputImage = np.max([warpTask.get_aligned_image(
                    f, dataChannel, z, chromaticCorrector)
                    for z in range(len(self.dataSet.get_z_positions()))],
                    axis=0)
            else:
                inputImage = warpTask.get_aligned_image(
                    f, dataChannel, zIndex, chromaticCorrector)

            if cropWidth > 0:
                inputImage[:cropWidth, :] = 0
                inputImage[inputImage.shape[0] - cropWidth:, :] = 0
                inputImage[:, :cropWidth] = 0
                inputImage[:, inputImage.shape[0] - cropWidth:] = 0

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

        return mosaic
