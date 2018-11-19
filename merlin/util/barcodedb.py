from abc import ABC, abstractmethod
import itertools
import pandas
import sqlalchemy
from sqlalchemy import types

class BarcodeDB():

    '''A class for storing and retrieving barcode information.
    '''

    def __init__(self, dataSet, analysisTask):
        self._dataSet = dataSet
        self._analysisTask = analysisTask

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

    @abstractmethod
    def get_barcodes(self, fov=None, columnList=None, chunksize=None):
        pass

    @abstractmethod
    def get_filtered_barcodes(
            self, areaThreshold, intensityThreshold, fov=None, chunksize=None):
        pass

    @abstractmethod
    def get_intensities_for_barcodes_with_area(self, area):
        '''Gets the barcode intensities for barcodes that have the specified
        area.
        '''
        pass

    @abstractmethod
    def write_barcodes(self, barcodeInformation, fov=None):
        '''Writes the barcodes specified in barcodeInformation into the 
        barcode database. If all the barcodes correspond to the same fov,
        then fov can be specified to improve performance.
        '''
        pass

    def get_barcode_intensities(self):
        return self.get_barcodes(
                columnList=['mean_intensity'])['mean_intensity']

    def get_barcode_areas(self):
        return self.get_barcodes(columnList=['area'])['area']

    def get_barcode_distances(self):
        return self.get_barcodes(columnList=['mean_distance'])['mean_distance']


class SQLiteBarcodeDB(BarcodeDB):
    
    '''A class for storing and retrieving barcode information using a
    SQLite back end. 
    
    This barcode database stores the barcodes corresponding to each FOV
    into its own SQLite file so that multiple FOVS can be processed 
    in parallel efficiently.
    '''

    #TODO - the functions in this class can be consolidated
    #TODO - the read functions take a lot of memory, much more than the final
    #dataframe. This can be reduced by reading in smaller batches.

    def __init__(self, dataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def _get_barcodeDB(self, fov):
        return self._dataSet.get_database_engine(
                self._analysisTask, index=fov)
    
    def _initialize_db(self):
        #TODO - maybe I can initialize the database with an autoincrementing
        #column
        '''
        bcTable = sqlalchemy.Table(
                'barcode_information', sqlalchemy.MetaData(),
                Column('id', types.Integer, primary_key=True),
        '''
        pass    

    def _aggregate_barcodes_from_iterator(self, barcodeIterator):
        barcodeDF = pandas.DataFrame([])
        for currentBarcodes in barcodeIterator:
            barcodeDF = barcodeDF.append(currentBarcodes)

        return barcodeDF

    def get_barcodes(self, fov=None, columnList=None, chunksize=None):
        returnIterator = chunksize is not None
        chunksize = chunksize or 100000
        
        if fov is None:
            barcodeIterator = itertools.chain.from_iterable(
                    (self.get_barcodes(fov=x, columnList=columnList, 
                        chunksize=chunksize) \
                                for x in self._dataSet.get_fovs()))
        else:
            #In order to prevent memory spikes that happen when pandas
            #reads the whole database in at once, here it is read in 
            #in smaller increments and aggregated.
            if columnList is None:
                barcodeIterator = pandas.read_sql_table(
                        'barcode_information', self._get_barcodeDB(fov), 
                        chunksize=chunksize)
            else:
                barcodeIterator = pandas.read_sql_table('barcode_information', 
                        self._get_barcodeDB(fov), 
                        columns=columnList, chunksize=chunksize)

        if returnIterator:
            return barcodeIterator
        else:
            return self._aggregate_barcodes_from_iterator(barcodeIterator)

    def get_filtered_barcodes(
            self, areaThreshold, intensityThreshold, fov=None, chunksize=None):
        returnIterator = chunksize is not None
        chunksize = chunksize or 100000


        queryString = 'select * from barcode_information ' \
                + 'where area>=' + str(areaThreshold) \
                + ' and mean_intensity>=' + str(intensityThreshold)
        barcodeIterator = self._iterator_for_query(queryString, fov, chunksize)

        if returnIterator:
            return barcodeIterator
        else:
            return self._aggregate_barcodes_from_iterator(barcodeIterator)

    def get_intensities_for_barcodes_with_area(self, area):
        queryString = 'select mean_intensity from ' \
                + 'barcode_information where area=' + str(area)
        barcodeIterator = self._iterator_for_query(queryString)
        return self._aggregate_barcodes_from_iterator(barcodeIterator)\
                ['mean_intensity']

    def _iterator_for_query(self, queryString, fov=None, chunksize=100000):
        if fov is None:
            barcodeIterator = itertools.chain.from_iterable(
                    (self._iterator_for_query(queryString, fov=x, 
                        chunksize=chunksize)) \
                                for x in self._dataSet.get_fovs())
        else:
            barcodeIterator = pandas.read_sql_query(queryString,
                    self._get_barcodeDB(fov), chunksize=chunksize)

        return barcodeIterator


    def write_barcodes(self, barcodeInformation, fov=None):
        if len(barcodeInformation) <= 0:
            return
        if fov is None:
            raise NotImplementedError

        columnInformation = self._get_bc_column_types()
    
        #TODO - the database needs to create a unique ID for each barcode
        written = False 
        attempts = 0
        while not written and attempts < 100:
            try: 
                barcodeInformation.to_sql(
                        'barcode_information', self._get_barcodeDB(fov), 
                        chunksize=50, dtype=columnInformation, index=False, 
                        if_exists='append')
                written = True
            except sqlalchemy.exc.OperationalError:
                attempts += 1

        if not written:
            raise sqlalchemy.exc.OperationalError('Failed to write barcodes',
                    None, self)
