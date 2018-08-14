import numpy as np
import cv2
import pandas
import sqlalchemy
from sqlalchemy import types
from skimage import measure

from merfish_code.core import analysistask
from merfish_code.util import decoding
from merfish_code.util import binary


class Decode(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.barcodeDB = dataSet.get_database_engine(self)
        self.areaThreshold = 4

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def run_analysis(self, fragmentIndex):
        '''This function generates the barcodes for a fov and saves them to the 
        barcode database.
        '''
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])

        decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)

        imageSet = np.array(
                preprocessTask.get_processed_image_set(fragmentIndex))
        scaleFactors = optimizeTask.get_scale_factors()
        di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors)


        self._extract_and_save_barcodes(di, pm, npt, d, fragmentIndex)

    def _initialize_db(self):
        #TODO - maybe I can initialize the database with an autoincrementing
        #column
        '''
        bcTable = sqlalchemy.Table(
                'barcode_information', sqlalchemy.MetaData(),
                Column('id', types.Integer, primary_key=True),
        '''
        pass

    def _dataframe_empty(cls, columns, dtypes, index=None):
        df = pandas.DataFrame(index=index)
        for c,d in zip(columns, dtypes):
            df[c] = pandas.Series(dtype=d)
        return df

    def _initialize_barcode_dataframe(self):
        '''
        barcode - the error corrected binary word corresponding to the barcode
        barcode_id - the index of the barcode in the codebook
        fov - the field of view where the barcode was identified
        magnitude - the sum of the fluorescence intensities in the pixels 
            corresponding to this  barcode
        area - the number of pixels covered by the barcode
        mean_distance - the distance between the barcode and the measured
            pixel traces averaged for all pixels corresponding to the barcode
        min_distance - the distance between the barcode and the measured
            pixel traces averaged for all pixels corresponding to the barcode
        x,y,z - the average x,y,z position of all pixels covered by the barcode
        weighted_x, weighted_y, weighted_z - the average x,y,z position of 
            of all pixels covered by the barcode weighted by the magnitude
            of eachc pixel
        global_x, global_y, global_z - the global x,y,z position of the barcode 

        Removed: (I am not convinced this is a useful way to quantify the errors
            in pixel-based decoding)
        measured_barcode - the measureed, uncorrected binary word corresponding
            to the barcode
        is_exact - flag indicating if non errors were detected while reading
            out the barcode
        error_bit - the index of the bit where an error occured if the barcode
            is not exact
        error_direction - the direction of othe error. True corresponds to
            a 0 to 1 error and false corresponds to a 1 to 0 error.
        '''
        
        columnInformation = self._get_bc_column_types()
        df = pandas.DataFrame(columns=columnInformation.keys())

        return df
        
    def _get_bc_column_types(self):
        columnInformation={'barcode': types.BigInteger(), \
                            'barcode_id': types.SmallInteger(), \
                            'fov': types.SmallInteger(), \
                            'mean_intensity': types.Float(precision=32), \
                            'max_intensity': types.Float(precision=32), \
                            'area': types.SmallInteger(), \
                            'mean_distance': types.Float(precision=32), \
                            'min_distance': types.Float(precision=32), \
                            'x': types.Float(precision=32), \
                            'y': types.Float(precision=32), \
                            'z': types.Float(precision=32), \
                            'global_x': types.Float(precision=32), \
                            'global_y': types.Float(precision=32), \
                            'global_z': types.Float(precision=32)}
        return columnInformation

    def _write_barcodes_to_db(self, barcodeInformation):
        columnInformation = self._get_bc_column_types()
    
        #TODO - the database needs to create a unique ID for each barcode
        barcodeInformation.to_sql(
                'barcode_information', self.barcodeDB, chunksize=50,
                dtype=columnInformation, index=False, if_exists='append')

    def get_barcode_information(self, sqlQuery=None):
        if sqlQuery is None:
            return pandas.read_sql_table('barcode_information', self.barcodeDB)
    
    def _bc_properties_to_dict(
            self, properties, bcIndex, fov, distances):
        #TODO update for 3D
        centroid = properties.centroid
        globalCentroid = self.dataSet.calculate_global_position(fov, centroid)
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
            self._write_barcodes_to_db(
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
