from abc import abstractmethod
from typing import List
import itertools
import pandas
import sqlalchemy
from sqlalchemy import types
import tables


class BarcodeDB:

    """
    An abstract class for storing and retrieving barcode information.

    For each barcode in the database, the following information is stored:
        barcode - the error corrected binary word corresponding to the barcode
        barcode_id - the index of the barcode in the codebook
        fov - the field of view where the barcode was identified
        mean_intensity - the mean intensity in the pixels corresponding to
            this  barcode
        max_intensity - the max intensity in the pixels corresponding to
            this barcode
        area - the number of pixels covered by the barcode
        mean_distance - the distance between the barcode and the measured
            pixel traces averaged for all pixels corresponding to the barcode
        min_distance - the minimum distance between the barcode and the measured
            pixel traces of all pixels corresponding to the barcode
        x,y,z - the average x,y,z position of all pixels covered by the barcode
        weighted_x, weighted_y, weighted_z - the average x,y,z position of
            of all pixels covered by the barcode weighted by the magnitude
            of each pixel
        global_x, global_y, global_z - the global x,y,z position of the barcode
        cell_index - the cell that contains this barcode
        intensity_i - the mean intensity across corresponding pixels for
            bit i where i is an integer from 0 to the number of bits-1.
    """

    def __init__(self, dataSet, analysisTask):
        self._dataSet = dataSet
        self._analysisTask = analysisTask

    def _get_bc_column_types(self):
        columnInformation = {'barcode': types.BigInteger(),
                             'barcode_id': types.SmallInteger(),
                             'fov': types.SmallInteger(),
                             'mean_intensity': types.Float(precision=32),
                             'max_intensity': types.Float(precision=32),
                             'area': types.SmallInteger(),
                             'mean_distance': types.Float(precision=32),
                             'min_distance': types.Float(precision=32),
                             'x': types.Float(precision=32),
                             'y': types.Float(precision=32),
                             'z': types.Float(precision=32),
                             'global_x': types.Float(precision=32),
                             'global_y': types.Float(precision=32),
                             'global_z': types.Float(precision=32),
                             'cell_index': types.Integer()}

        for i in range(self._dataSet.get_codebook().get_bit_count()):
            columnInformation['intensity_'+str(i)] = types.Float(precision=32)

        return columnInformation

    @abstractmethod
    def empty_database(self, fov: int=None) -> None:
        """Remove all barcodes from this database.

        Args:
            fov: index of the field of view. If specified, only barcodes
                corresponding to the specified fov will be removed. Otherwise,
                all barcodes will be removed.
        """
        pass

    @abstractmethod
    def get_barcodes(self, fov: int=None, columnList: List[str]=None,
                     chunksize: int=None):
        """Get barcodes stored in this database.

        Args:
            fov: index of the field view. If None, barcodes for all fovs
                are returned.
            columnList: list of columns to extract. If not specified, all
                columns are returned.
            chunksize: the size of chunks to iterate. If not specified, a
                pandas dataframe is returned otherwise an iterator over the
                barcodes is returned.
        Returns:
            if chunksize is not set, a pandas dataframe containing all the
                requested barcodes is returned. Otherwise an iterator is
                returned that iterates over the requested barcodes.
        """
        pass

    @abstractmethod
    def get_filtered_barcodes(
            self, areaThreshold: int, intensityThreshold: float,
            fov: int=None, chunksize: int=None):
        """Get barcodes from this barcode database that pass the area and
        intensity thresholds.

        Args:
            areaThreshold: the minimum area threshold. Barcodes that
                have an area equal to the specified threshold are included
                in the output.
            intensityThreshold: the minimum value for mean_intenity for
                the select barcodes
            fov: index of the field view. If None, barcodes for all fovs
                are returned.
            chunksize: the size of chunks to iterate. If not specified, a
                pandas dataframe is returned otherwise an iterator over the
                barcodes is returned.
        Returns:
            if chunksize is not set, a pandas dataframe containing all the
                requested barcodes is returned. Otherwise an iterator is
                returned that iterates over the requested barcodes.
        """
        pass

    @abstractmethod
    def get_intensities_for_barcodes_with_area(
            self, area: int) -> pandas.Series:
        """Gets the barcode intensities for barcodes that have the specified
        area.
        """
        pass

    @abstractmethod
    def write_barcodes(self, barcodeInformation: pandas.DataFrame,
                       fov: int=None) -> None:
        """Writes the specified barcodes into the barcode database.

        If all the barcodes correspond to the same fov, then fov can be
        specified to improve performance. This function does not verify that
        fov is specified incorrectly. If fov is specified but the provided
        barcodes are not all from the same fov, the barcode database may become
        corrupted.

        Args:
            barcodeInformation: barcodes to write to the database. The
                dataframe must have the columns specified for a barcode
                database.
            fov: the fov of the barcodes if they all correspond to the same
                fov. If barcodeInformation contains barcodes from different
                fovs, then fov should be set to None.
        """
        pass

    def get_barcode_intensities(self) -> pandas.Series:
        """Get mean intensities for all barcodes in this database.

        Returns:
            series containing mean intensity for all barcodes
        """
        return self.get_barcodes(
                columnList=['mean_intensity'])['mean_intensity']

    def get_barcode_areas(self) -> pandas.Series:
        """Get areas for all barcodes in this database.

        Returns:
            series containing areas for all barcodes
        """
        return self.get_barcodes(columnList=['area'])['area']

    def get_barcode_distances(self) -> pandas.Series:
        """Get distances for all barcodes in this database

        Returns:
            series containing distances for all barcodes
        """
        return self.get_barcodes(columnList=['mean_distance'])['mean_distance']


class PyTablesBarcodeDB(BarcodeDB):

    def __init__(self, dataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def empty_database(self, fov: int=None) -> None:
        if fov is None:
            for f in self._dataSet.get_fovs():
                self.empty_database(f)

        self._dataSet.delete_table('barcode_data', self._analysisTask, fov,
                                   'barcodes')

    def get_barcodes(self, fov=None, columnList=None, chunkSize=None)\
            -> pandas.DataFrame:

        if fov is None:
            barcodes = pandas.concat(
                [self.get_barcodes(fov=x, columnList=columnList)
                 for x in self._dataSet.get_fovs()])
        else:
            try:
                with self._dataSet.open_table(
                        'r', 'barcode_data', self._analysisTask,
                        fov, 'barcodes') as tableFile:

                    if '/barcodes' not in tableFile:
                        return pandas.DataFrame()
                    barcodes = pandas.DataFrame.from_records(
                        tableFile.root.barcodes[:])

                    if columnList is not None:
                        barcodes = barcodes[columnList]

            except OSError:
                return pandas.DataFrame()

        print(barcodes)
        return barcodes

    def get_filtered_barcodes(
            self, areaThreshold: int, intensityThreshold: float,
            fov: int=None, chunksize: int=None):
        allBarcodes = self.get_barcodes(fov)
        filteredBarcodes = allBarcodes[
            (allBarcodes['area'] >= areaThreshold)
            & (allBarcodes['mean_intensity'] >= intensityThreshold)
        ]

        return filteredBarcodes

    def get_intensities_for_barcodes_with_area(
            self, area: int) -> pandas.Series:
        allBarcodes = self.get_barcodes()
        return allBarcodes[allBarcodes['area'] == area]['mean_intensity']

    def write_barcodes(self, barcodeInformation: pandas.DataFrame,
                       fov: int=None) -> None:
        if len(barcodeInformation) <= 0:
            return

        if fov is None:
            for f in barcodeInformation.fov.unique():
                self.write_barcodes(
                        barcodeInformation.loc[barcodeInformation['fov'] == f],
                        fov=f)

        with self._dataSet.open_table('a', 'barcode_data', self._analysisTask,
                                      fov, 'barcodes') as tableFile:

            if '/barcodes' not in tableFile:
                tablesTypeConversion = {
                    type(types.BigInteger()): tables.UInt64Col(),
                    type(types.SmallInteger()): tables.UInt16Col(),
                    type(types.Float(precision=32)): tables.Float32Col(),
                    type(types.Integer()): tables.Int32Col()
                }
                tablesType = {k: tablesTypeConversion[type(v)] for k, v in
                              self._get_bc_column_types().items()}
                barcodeTable = tableFile.create_table(
                    '/', 'barcodes', tablesType)

            else:
                barcodeTable = tableFile.root.barcodes

            barcode = barcodeTable.row
            for i, currentBarcode in barcodeInformation.iterrows():
                for k, x in currentBarcode.iteritems():
                    barcode[k] = x
                barcode.append()

            barcodeTable.flush()


class SQLiteBarcodeDB(BarcodeDB):
    
    """A class for storing and retrieving barcode information using a
    SQLite back end. 
    
    This barcode database stores the barcodes corresponding to each FOV
    into its own SQLite file so that multiple FOVS can be processed 
    in parallel efficiently.
    """

    def __init__(self, dataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def _get_barcodeDB(self, fov):
        return self._dataSet.get_database_engine(
                self._analysisTask, index=fov)
    
    def _initialize_db(self):
        pass    

    @staticmethod
    def _aggregate_barcodes_from_iterator(barcodeIterator):
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
                                       chunksize=chunksize)
                     for x in self._dataSet.get_fovs()))
        else:
            dbEngine = self._get_barcodeDB(fov)
            if not dbEngine.dialect.has_table(dbEngine, 'barcode_information'):
                return []

            # In order to prevent large memory spikes that happen when pandas
            # reads the whole database in at once, here it is read in
            # in smaller increments and aggregated.
            if columnList is None:
                barcodeIterator = pandas.read_sql_table(
                    'barcode_information', dbEngine, chunksize=chunksize)
            else:
                barcodeIterator = pandas.read_sql_table(
                    'barcode_information', dbEngine, columns=columnList,
                    chunksize=chunksize)

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
        barcodes = self._aggregate_barcodes_from_iterator(barcodeIterator)
        if len(barcodes) > 0:
            return barcodes['mean_intensity'].tolist()
        else:
            return []

    def _iterator_for_query(self, queryString, fov=None, chunksize=100000):
        if fov is None:
            barcodeIterator = itertools.chain.from_iterable(
                    (self._iterator_for_query(queryString, fov=x,
                                              chunksize=chunksize))
                    for x in self._dataSet.get_fovs())
        else:
            dbEngine = self._get_barcodeDB(fov)
            if not dbEngine.dialect.has_table(dbEngine, 'barcode_information'):
                return []

            barcodeIterator = pandas.read_sql_query(queryString,
                                                    dbEngine,
                                                    chunksize=chunksize)

        return barcodeIterator

    def write_barcodes(self, barcodeInformation, fov=None):
        if len(barcodeInformation) <= 0:
            return

        if fov is None:
            for f in barcodeInformation.fov.unique():
                self.write_barcodes(
                        barcodeInformation.loc[barcodeInformation['fov'] == f],
                        fov=f)

        columnInformation = self._get_bc_column_types()
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

    def empty_database(self, fov=None):
        if fov is None:
            for f in self._dataSet.get_fovs():
                self.empty_database(fov=f)

        self._get_barcodeDB(fov).execute(
                'DROP TABLE IF EXISTS barcode_information;')
