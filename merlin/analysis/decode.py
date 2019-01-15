import numpy as np
import pandas
from skimage import measure

from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import binary
from merlin.util import barcodedb


class Decode(analysistask.ParallelAnalysisTask):

    '''
    An analysis task that extracts barcodes from images.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.imageDimensions

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

    def run_analysis(self, fragmentIndex):
        '''This function decodes the barcodes in a fov and saves them to the
        barcode database.
        '''
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        self.globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        self.segmentTask = None
        if 'segment_task' in self.parameters:
            self.segmentTask = self.dataSet.load_analysis_task(
                    self.parameters['segment_task'])

        decoder = decoding.PixelBasedDecoder(self.dataSet.get_codebook())
        scaleFactors = optimizeTask.get_scale_factors()

        decodedImages = []
        magnitudeImages = []
        zPositionCount = len(self.dataSet.get_z_positions())
        for zIndex in range(zPositionCount):
            imageSet = preprocessTask.get_processed_image_set(
                    fragmentIndex, zIndex)
            imageSet = imageSet.reshape(
                (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))
            di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors)
            self._extract_and_save_barcodes(
                    di, pm, npt, d, fragmentIndex, zIndex)
            decodedImages.append(di)
            magnitudeImages.append(pm)

        if self.parameters['write_decoded_images']:

            imageDescription = self.dataSet._analysis_tiff_description(
                zPositionCount, 2)
            with self.dataSet._writer_for_analysis_images(
                    self, 'decoded', fragmentIndex) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    # rescale image so that it can be viewed with the decoded
                    # image without adjusting contrast
                    outputTif.save((magnitudeImages[i]/np.max(
                        magnitudeImages[i])*256).astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def get_barcode_database(self):
        return barcodedb.SQLiteBarcodeDB(self.dataSet, self)

    def _initialize_barcode_dataframe(self):
        columnInformation = self._get_bc_column_types()
        df = pandas.DataFrame(columns=columnInformation.keys())

        return df

    def _bc_properties_to_dict(
            self, properties, bcIndex, fov, zIndex, distances, pixelTraces):
        #centroid is reversed since skimage regionprops returns the centroid
        #as (r,c)
        centroid = properties.weighted_centroid[::-1]
        globalCentroid = self.globalTask.fov_coordinates_to_global(
                fov, centroid)
        d = [distances[x[0], x[1]] for x in properties.coords]
        zPosition = self.dataSet.z_index_to_position(zIndex)
        outputDict = {'barcode': binary.bit_list_to_int(
                            self.dataSet.get_codebook().get_barcode(bcIndex)), \
                    'barcode_id': bcIndex, \
                    'fov': fov, \
                    'mean_intensity': properties.mean_intensity, \
                    'max_intensity': properties.max_intensity, \
                    'area': properties.area, \
                    'mean_distance': np.mean(d), \
                    'min_distance': np.min(d), \
                    'x': centroid[0], \
                    'y': centroid[1], \
                    'z': zPosition, \
                    'global_x': globalCentroid[0], \
                    'global_y': globalCentroid[1], \
                    'global_z': zPosition, \
                    'cell_index': -1}

        for i in range(len(pixelTraces)):
            outputDict['intensity_' + str(i)] = \
                np.mean([pixelTraces[i,x[0],x[1]] \
                        for x in properties.coords])

        if self.segmentTask is not None:
            outputDict['cell_index'] = self.segmentTask \
                    .get_cell_containing_position(
                            globalCentroid[0], globalCentroid[1])

        return outputDict

    def _extract_and_save_barcodes(self, decodedImage, pixelMagnitudes, 
            pixelTraces, distances, fov, zIndex):

        for i in range(self.dataSet.get_codebook().get_barcode_count()):
            self.get_barcode_database().write_barcodes(
                    self._extract_barcodes_with_index(
                        i, decodedImage, pixelMagnitudes, pixelTraces, 
                        distances, fov, zIndex), fov=fov)

    def _position_within_crop(self, position):
        return position[0] > self.cropWidth \
                and position[1] > self.cropWidth \
                and position[0] < self.imageSize[0] - self.cropWidth \
                and position[1] < self.imageSize[1] - self.cropWidth

    def _extract_barcodes_with_index(
            self, barcodeIndex, decodedImage, pixelMagnitudes, 
            pixelTraces, distances, fov, zIndex):

        properties = measure.regionprops(
                measure.label(decodedImage == barcodeIndex),
                intensity_image=pixelMagnitudes)
        dList = [self._bc_properties_to_dict(p, barcodeIndex, fov, zIndex, 
            distances, pixelTraces) \
                for p in properties if self._position_within_crop(p.centroid)]
        barcodeInformation = pandas.DataFrame(dList)

        return barcodeInformation
