import numpy as np
import pandas
import os
import tempfile

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.data.codebook import Codebook
from merlin.util import barcodefilters


class BarcodeSavingParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves barcodes into a barcode database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_barcode_database().empty_database(fragmentIndex)

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """ Get the barcode database this analysis task saves barcodes into.

        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)


class Decode(BarcodeSavingParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    def __init__(self, dataSet: dataset.MERFISHDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5167
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decode_3d' not in self.parameters:
            self.parameters['decode_3d'] = False
        if 'memory_map' not in self.parameters:
            self.parameters['memory_map'] = False
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_zPlane_threshold' not in self.parameters:
                self.parameters['z_duplicate_zPlane_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)

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

        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

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

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)
        scaleFactors = optimizeTask.get_scale_factors()
        backgrounds = optimizeTask.get_backgrounds()
        chromaticCorrector = optimizeTask.get_chromatic_corrector()

        zPositionCount = len(self.dataSet.get_z_positions())
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape),
                                   dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), dtype=np.float32)

        if not decode3d:
            for zIndex in range(zPositionCount):
                di, pm, d = self._process_independent_z_slice(
                    fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                    backgrounds, preprocessTask, decoder
                )

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

        else:
            with tempfile.TemporaryDirectory() as tempDirectory:
                if self.parameters['memory_map']:
                    normalizedPixelTraces = np.memmap(
                        os.path.join(tempDirectory, 'pixel_traces.dat'),
                        mode='w+', dtype=np.float32,
                        shape=(zPositionCount, bitCount, *imageShape))
                else:
                    normalizedPixelTraces = np.zeros(
                        (zPositionCount, bitCount, *imageShape),
                        dtype=np.float32)

                for zIndex in range(zPositionCount):
                    imageSet = preprocessTask.get_processed_image_set(
                        fragmentIndex, zIndex, chromaticCorrector)
                    imageSet = imageSet.reshape(
                        (imageSet.shape[0], imageSet.shape[-2],
                         imageSet.shape[-1]))

                    di, pm, npt, d = decoder.decode_pixels(
                        imageSet, scaleFactors, backgrounds,
                        lowPassSigma=lowPassSigma,
                        distanceThreshold=self.parameters['distance_threshold'])

                    normalizedPixelTraces[zIndex, :, :, :] = npt
                    decodedImages[zIndex, :, :] = di
                    magnitudeImages[zIndex, :, :] = pm
                    distances[zIndex, :, :] = d

                self._extract_and_save_barcodes(
                    decoder, decodedImages, magnitudeImages,
                    normalizedPixelTraces,
                    distances, fragmentIndex)

                del normalizedPixelTraces

        if self.parameters['write_decoded_images']:
            self._save_decoded_images(
                fragmentIndex, zPositionCount, decodedImages, magnitudeImages,
                distances)

        if self.parameters['remove_z_duplicated_barcodes']:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(
                bcDB.get_barcodes(fov=fragmentIndex))
            bcDB.empty_database(fragmentIndex)
            bcDB.write_barcodes(bc, fov=fragmentIndex)


    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder):

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

        di, pm, npt, d = decoder.decode_pixels(
            imageSet, scaleFactors, backgrounds,
            lowPassSigma=self.parameters['lowpass_sigma'],
            distanceThreshold=self.parameters['distance_threshold'])
        self._extract_and_save_barcodes(
            decoder, di, pm, npt, d, fov, zIndex)

        return di, pm, d

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

        minimumArea = self.parameters['minimum_area']

        self.get_barcode_database().write_barcodes(
            pandas.concat([decoder.extract_barcodes_with_index(
                i, decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
                self.cropWidth, zIndex, globalTask, minimumArea)
                for i in range(self.get_codebook().get_barcode_count())]),
            fov=fov)

    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_zPlane_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc
