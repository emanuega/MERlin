import numpy as np
import cv2
import pandas
from skimage import measure

from merfish_code.core import analysistask
from merfish_code.util import decoding
from merfish_code.util import binary
from merfish_code.util import barcodedb


class Decode(analysistask.ParallelAnalysisTask):

    '''An analysis task that extracts barcodes information from 
    images.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)


    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependences(self):
        return [self.parameters['preprocess_task'], \
                self.parameters['optimize_task'], \
                self.parameters['global_align_task']]

    def run_analysis(self, fragmentIndex):
        '''This function generates the barcodes for a fov and saves them to the 
        barcode database.
        '''
        self.preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        self.optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        self.globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)
        imageSet = np.array(
                self.preprocessTask.get_processed_image_set(fragmentIndex))
        scaleFactors = self.optimizeTask.get_scale_factors()
        di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors)
        self._extract_and_save_barcodes(di, pm, npt, d, fragmentIndex)

    def _initialize_barcode_dataframe(self):
        columnInformation = self._get_bc_column_types()
        df = pandas.DataFrame(columns=columnInformation.keys())

        return df

    def get_barcode_database(self):
        #TODO - this should belong to the class so that I don't have to 
        #create a new one 
        return barcodedb.BarcodeDB(self.dataSet, self)

    def _bc_properties_to_dict(
            self, properties, bcIndex, fov, distances):
        #TODO update for 3D
        centroid = properties.centroid
        globalCentroid = self.globalTask.fov_coordinates_to_global(
                fov, centroid)
        d = [distances[x[0], x[1]] for x in properties.coords]
        outputDict = {'barcode': binary.bit_array_to_int(
                            self.dataSet.codebook.loc[bcIndex, 'barcode']), \
                    'barcode_id': bcIndex, \
                    'fov': fov, \
                    'mean_intensity': properties.mean_intensity, \
                    'max_intensity': properties.max_intensity, \
                    'area': properties.area, \
                    'mean_distance': np.mean(d), \
                    'min_distance': np.min(d), \
                    'x': centroid[0], \
                    'y': centroid[1], \
                    'z': 0.0, \
                    'global_x': globalCentroid[0], \
                    'global_y': globalCentroid[1], \
                    'global_z': 0.0}

        return outputDict

    def _extract_and_save_barcodes(self, decodedImage, pixelMagnitudes, 
            pixelTraces, distances, fov):

        for i in range(len(self.dataSet.codebook)):
            self.get_barcode_database().write_barcodes(
                    self._extract_barcodes_with_index(
                        i, decodedImage, pixelMagnitudes, pixelTraces, 
                        distances, fov))

    def _extract_barcodes_with_index(
            self, barcodeIndex, decodedImage, pixelMagnitudes, 
            pixelTraces, distances, fov):

        properties = measure.regionprops(
                measure.label(decodedImage == barcodeIndex),
                intensity_image=pixelMagnitudes)
        dList = [self._bc_properties_to_dict(p, barcodeIndex, fov, distances) \
                for p in properties]
        barcodeInformation = pandas.DataFrame(dList)

        return barcodeInformation
