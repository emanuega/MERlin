from abc import abstractmethod
from typing import List
import pandas
import numpy as np

from merlin.core import dataset


class BarcodeDB:

    """
    An abstract class for storing and retrieving barcode information.

    For each barcode in the database, the following information is stored:
        barcode - the error corrected binary word corresponding to the barcode
        assigned_barcode -
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
        try:
            self._codebook = self._analysisTask.get_codebook()
        except AttributeError:
            self._codebook = self._dataSet.get_codebook()
            
    def _get_bc_column_types(self):
        columnInformation = {'barcode_id': np.uint16,
                             'fov': np.uint16,
                             'mean_intensity': np.float32,
                             'max_intensity': np.float32,
                             'area': np.uint16,
                             'mean_distance': np.float32,
                             'min_distance': np.float32,
                             'x': np.float32,
                             'y': np.float32,
                             'z': np.float32,
                             'global_x': np.float32,
                             'global_y': np.float32,
                             'global_z': np.float32,
                             'cell_index': np.int32}

        for i in range(self._codebook.get_bit_count()):
            columnInformation['intensity_'+str(i)] = np.float32

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
            distanceThreshold: float=None, fov: int=None, chunksize: int=None):
        """Get barcodes from this barcode database that pass the area and
        intensity thresholds.

        Args:
            areaThreshold: the minimum area threshold. Barcodes that
                have an area equal to the specified threshold are included
                in the output.
            intensityThreshold: the minimum value for mean_intenity for
                the select barcodes
            distanceThreshold: the maximum value for min_distance for
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

    def __init__(self, dataSet: dataset.DataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def empty_database(self, fov: int=None) -> None:
        if fov is None:
            for f in self._dataSet.get_fovs():
                self.empty_database(f)

        self._dataSet.delete_pandas_hdfstore(
            'barcode_data', self._analysisTask, fov, 'barcodes')

    def get_barcodes(self, fov=None, columnList=None, chunkSize=None)\
            -> pandas.DataFrame:

        if fov is None:
            barcodes = pandas.concat(
                [self.get_barcodes(fov=x, columnList=columnList)
                 for x in self._dataSet.get_fovs()], sort=True)
        else:
            try:
                with self._dataSet.open_pandas_hdfstore(
                        'r', 'barcode_data', self._analysisTask,
                        fov, 'barcodes') as pandasHDF:

                    if 'barcodes' not in pandasHDF:
                        return pandas.DataFrame()

                    if columnList is None:
                        barcodes = pandasHDF['barcodes']
                    else:
                        barcodes = pandas.read_hdf(pandasHDF, key='barcodes',
                                                   columns=columnList)

            except OSError:
                barcodes = pandas.DataFrame()

        # if no barcodes are present make sure the dataframe still has the
        # correct columns
        if len(barcodes) == 0:
            if columnList:
                barcodes = pandas.DataFrame(columns=columnList)
            else:
                barcodes = pandas.DataFrame(
                    columns=self._get_bc_column_types().keys())

        return barcodes

    def get_filtered_barcodes(
            self, areaThreshold: int, intensityThreshold: float,
            distanceThreshold: float=None, fov: int=None, chunksize: int=None):
        allBarcodes = self.get_barcodes(fov)
        if distanceThreshold is None:
            filteredBarcodes = allBarcodes[
                (allBarcodes['area'] >= areaThreshold)
                & (allBarcodes['mean_intensity'] >= intensityThreshold)]
        else:
            filteredBarcodes = allBarcodes[
                (allBarcodes['area'] >= areaThreshold)
                & (allBarcodes['mean_intensity'] >= intensityThreshold)
                & (allBarcodes['min_distance'] <= distanceThreshold)]

        return filteredBarcodes

    def get_intensities_for_barcodes_with_area(
            self, area: int) -> pandas.Series:
        allBarcodes = self.get_barcodes(columnList=['area', 'mean_intensity'])
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

        with self._dataSet.open_pandas_hdfstore(
                'a', 'barcode_data', self._analysisTask, fov, 'barcodes'
        ) as pandasHDF:
            tablesType = self._get_bc_column_types()
            pandasHDF.append('barcodes', barcodeInformation.astype(tablesType),
                             format='table')
