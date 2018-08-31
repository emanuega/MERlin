import pandas
from sqlalchemy import types

class BarcodeDB():
    
    '''A class for storing and retrieving barcode information.
    '''

    #TODO - the functions in this class can be consolidated
    #TODO - the read functions take a lot of memory, much more than the final
    #dataframe. This can be reduced by reading in smaller batches.

    def __init__(self, dataSet, analysisTask):
        self.dataSet = dataSet
        self.analysisTask = analysisTask

    def _get_barcodeDB(self):
        return self.dataSet.get_database_engine(self.analysisTask)
    
    def _initialize_db(self):
        #TODO - maybe I can initialize the database with an autoincrementing
        #column
        '''
        bcTable = sqlalchemy.Table(
                'barcode_information', sqlalchemy.MetaData(),
                Column('id', types.Integer, primary_key=True),
        '''
        pass    

    def _get_bc_column_types(self):
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
        cell_index - the cell that contains this barcode
        '''
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
                            'global_z': types.Float(precision=32), \
                            'cell_index': types.Integer()}
        return columnInformation

    def get_barcodes(self, columnList=None):
        if columnList is None:
            return pandas.read_sql_table(
                    'barcode_information', self._get_barcodeDB())
        else:
            return pandas.read_sql_table('barcode_information', 
                    self._get_barcodeDB(), 
                    columns=columnList)

    def get_filtered_barcodes(
            self, areaThreshold, intensityThreshold, chunksize=None):
        return pandas.read_sql_query(
                'select * from barcode_information ' \
                        + 'where area>=' + str(areaThreshold) \
                        + ' and mean_intensity>=' + str(intensityThreshold),
                        self._get_barcodeDB(), chunksize=chunksize)

    def get_barcode_intensities(self):
        return self.get_barcodes(
                ['mean_intensity'])['mean_intensity']

    def get_intensities_for_barcodes_with_area(self, area):
        '''Gets the barcode intensities for barcodes that have the specified
        area.
        '''
        return pandas.read_sql_query(
                'select mean_intensity from ' \
                + 'barcode_information where area=' + str(area), 
                self._get_barcodeDB())['mean_intensity'] 

    def get_barcode_areas(self):
        return self.get_barcodes(['area'])['area']

    def get_barcode_distances(self):
        return self.get_barcodes(['mean_distance'])['mean_distance']

    def write_barcodes(self, barcodeInformation):
        columnInformation = self._get_bc_column_types()
    
        #TODO - the database needs to create a unique ID for each barcode
        barcodeInformation.to_sql(
                'barcode_information', self._get_barcodeDB(), chunksize=50,
                dtype=columnInformation, index=False, if_exists='append')


