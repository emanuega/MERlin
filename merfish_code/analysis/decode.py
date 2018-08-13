import numpy as np
import cv2
import pandas
import sqlalchemy
from sqlalchemy import types

from merfish_code.core import analysistask
from merfish_code.util import decoding


class Decode(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        #super().__init__(dataSet, parameters, analysisName)

        #TODO - this should be a path created from the dataset
        #TODO - the database needs to create a unique ID for each barcode
        self.barcodeDB = sqlalchemy.create_engine('sqlite:///bc.db')
        self.areaThreshold = 4

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])


        decoder = decoding.PixelBasedDecoder(self.dataSet.codebook)

        imageSet = np.array([cv2.GaussianBlur(x, (5, 5), 1) \
                        for x in preprocessTask.get_images(fragmentIndex)])
        scaleFactors = optimizeTask.get_scale_factors()
        di, pm, npt, d = decoder.decode_pixels(imageSet, scaleFactors)

        return di, pm, npt, d

    def _initialize_db(self):
        bcTable = sqlalchemy.Table(
                'barcode_information', sqlalchemy.MetaData(),
                Column('id', types.Integer, primary_key=True),

    def _dataframe_empty(cls, columns, dtypes, index=None):
        df = pandas.DataFrame(index=index)
        for c,d in zip(columns, dtypes):
            df[c] = pandas.Series(dtype=d)
        return df

    def _initialize_barcode_dataframe(self):
        '''
        barcode - the error corrected binary word corresponding to the barcode
        measured_barcode - the measureed, uncorrected binary word corresponding
            to the barcode
        barcode_id - the index of the barcode in the codebook
        fov - the field of view where the barcode was identified
        magnitude - the sum of the fluorescence intensities in the pixels 
            corresponding to this  barcode
        area - the number of pixels covered by the barcode
        average_distance - the distance between the barcode and the measured
            pixel traces averaged for all pixels corresponding to the barcode
        x,y,z - the average x,y,z position of all pixels covered by the barcode
        weighted_x, weighted_y, weighted_z - the average x,y,z position of 
            of all pixels covered by the barcode weighted by the magnitude
            of eachc pixel
        global_x, global_y, global_z - the global x,y,z position of the barcode 
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
                            'measured_barcode': types.BigInteger(), \
                            'barcode_id': types.SmallInteger(), \
                            'fov': types.SmallInteger(), \
                            'magnitude': types.Float(precision=32), \
                            'area': types.SmallInteger(), \
                            'average_distance': types.Float(precision=32), \
                            'x': types.Float(precision=32), \
                            'y': types.Float(precision=32), \
                            'z': types.Float(precision=32), \
                            'weighted_x': types.Float(precision=32), \
                            'weighted_y': types.Float(precision=32), \
                            'weighted_z': types.Float(precision=32), \
                            'global_x': types.Float(precision=32), \
                            'global_y': types.Float(precision=32), \
                            'global_z': types.Float(precision=32), \
                            'is_exact': types.Boolean(),  \
                            'error_bit': types.SmallInteger(), \
                            'error_direction': types.Boolean()}
        return columnInformation

    def _write_barcodes_to_db(self, barcodeInformation):
        columnInformation = self._get_bc_column_types()
    
        barcodeInformation.to_sql(
                'barcode_information', self.barcodeDB, chunksize=500,
                dtype=columnInformation, index=False, if_exists='append')

    def get_barcode_information(self, sqlQuery=None):
        if sqlQuery is None:
            return pandas.read_sql_table('barcode_information', self.barcodeDB)
    
    def _test_barcode_data(self):
        barcodeInformation={'barcode': np.uint64(2128), \
                            'measured_barcode': np.uint64(2128), \
                            'barcode_id': np.uint16(5), \
                            'fov': np.uint16(27), \
                            'magnitude': np.float32(15.8), \
                            'area': np.uint16(12), \
                            'average_distance': np.float32(12.8), \
                            'x': np.float32(1.0), \
                            'y': np.float32(1.0), \
                            'z': np.float32(1.0), \
                            'weighted_x': np.float32(1.2), \
                            'weighted_y': np.float32(1.2), \
                            'weighted_z': np.float32(1.1), \
                            'global_x': np.float32(728.9), \
                            'global_y': np.float32(182.8), \
                            'global_z': np.float32(128.7), \
                            'is_exact': True,  \
                            'error_bit': np.uint8(0), \
                            'error_direction': False}
        return barcodeInformation

    def _extract_barcodes(self, decodedImage, pixelMagnitudes, 
            singleErrorBarcodes, pixelTraces):
        pass
