import numpy as np
from typing import List

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb


class Decode(analysistask.ParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    def __init__(self, dataSet: dataset.ImageDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decode_3d' not in self.parameters:
            self.parameters['decode_3d'] = False

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]

        if 'segment_task' in self.parameters:
            dependencies += [self.parameters['segment_task']]

        return dependencies

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        decode3d = self.parameters['decode_3d']

        lowPassSigma = self.parameters['lowpass_sigma']

        decoder = decoding.PixelBasedDecoder(self.dataSet.get_codebook())
        scaleFactors = optimizeTask.get_scale_factors()
        chromaticCorrector = optimizeTask.get_chromatic_corrector()

        zPositionCount = len(self.dataSet.get_z_positions())
        bitCount = self.dataSet.get_codebook().get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape),
                                   dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), dtype=np.float32)

        if not decode3d:
            for zIndex in range(zPositionCount):
                imageSet = preprocessTask.get_processed_image_set(
                        fragmentIndex, zIndex, chromaticCorrector)
                imageSet = imageSet.reshape(
                    (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

                di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors,
                                                       lowPassSigma=lowPassSigma)
                self._extract_and_save_barcodes(
                        decoder, di, pm, npt, d, fragmentIndex, zIndex)

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

        else:
            normalizedPixelTraces = np.zeros(
                (zPositionCount, bitCount, *imageShape), dtype=np.float32)

            for zIndex in range(zPositionCount):
                imageSet = preprocessTask.get_processed_image_set(
                    fragmentIndex, zIndex, chromaticCorrector)
                imageSet = imageSet.reshape(
                    (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

                di, pm, npt, d = decoder.decode_pixels(
                    imageSet, scaleFactors, lowPassSigma=lowPassSigma)

                normalizedPixelTraces[zIndex, :, :, :] = npt
                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = distances

            self._extract_and_save_barcodes(
                decoder, decodedImages, magnitudeImages, normalizedPixelTraces,
                distances, fragmentIndex)

            decodedImages = np.zeros(())

        if self.parameters['write_decoded_images']:
            self._save_decoded_images(
                fragmentIndex, zPositionCount, decodedImages, magnitudeImages,
                distances)

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, 3)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(magnitudeImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(distanceImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zIndex: int=None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        segmentTask = None
        if 'segment_task' in self.parameters:
            segmentTask = self.dataSet.load_analysis_task(
                self.parameters['segment_task'])

        minimumArea = self.parameters['minimum_area']

        for i in range(self.dataSet.get_codebook().get_barcode_count()):
            self.get_barcode_database().write_barcodes(
                    decoder.extract_barcodes_with_index(
                        i, decodedImage, pixelMagnitudes, pixelTraces,
                        distances, fov, self.cropWidth,
                        self.dataSet.z_index_to_position(zIndex),
                        globalTask, segmentTask, minimumArea
                    ), fov=fov)
