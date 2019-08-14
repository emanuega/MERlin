import os
import csv
import numpy as np
import pandas
from typing import List
from typing import Union

import merlin


def _parse_barcode_from_string(inputString):
    return np.array([int(x) for x in inputString if x is not ' '])


class Codebook(object):

    """
    A Codebook stores the association of barcodes to genes.
    """

    def __init__(self, dataSet, filePath):
        """
        Create a new Codebook for the data in the specified data set.

        If filePath is not specified, a previously stored Codebook
        is loaded from the dataSet if it exists. If filePath is specified,
        the Codebook at the specified filePath is loaded and
        stored in the dataSet, overwriting any previously stored 
        Codebook.
        """
        self._dataSet = dataSet

        allAnalysisFiles = os.listdir(self._dataSet.analysisPath)
        existingCodebooks = [x for x in allAnalysisFiles if 'codebook' in x]

        currentCodebookNum = len(existingCodebooks)

        if not os.path.exists(filePath):
            filePath = os.sep.join(
                    [merlin.CODEBOOK_HOME, filePath])

        newVersion = True
        with open(filePath, 'r') as f:
            if 'version' in f.readline():
                newVersion = False

        if newVersion:
            self._data = pandas.read_csv(filePath)

        else:
            headerLength = 3
            barcodeData = pandas.read_csv(
                filePath, header=headerLength, skipinitialspace=True,
                usecols=['name', 'id', 'barcode'],
                converters={'barcode': _parse_barcode_from_string})
            with open(filePath, 'r') as inFile:
                csvReader = csv.reader(inFile, delimiter=',')
                header = [row for i, row in enumerate(csvReader)
                          if i < headerLength]

            bitNames = [x.strip() for x in header[2][1:]]

            self._data = self._generate_codebook_dataframe(
                    barcodeData, bitNames)
        name = os.path.splitext(os.path.basename(filePath))[0]

        if not os.path.isfile('{}/codebook_{}_{}.csv'.format(
                self._dataSet.analysisPath, currentCodebookNum, name)):
            self._dataSet.save_dataframe_to_csv(self._data,
                                                'codebook_{}_{}'.format(
                                                    currentCodebookNum, name),
                                                index=False)
        self.codebook_name = name


    @staticmethod
    def _generate_codebook_dataframe(barcodeData, bitNames):
        dfData = np.array([[currentRow['name'], currentRow['id']]
                           + currentRow['barcode'].tolist()
                           for i, currentRow in barcodeData.iterrows()])
        df = pandas.DataFrame(dfData, columns=['name', 'id'] + bitNames)
        df[bitNames] = df[bitNames].astype('uint8')
        return df

    def get_barcode(self, index: int) -> List[bool]:
        """Get the barcode with the specified index.

        Args:
            index: the index of the barcode in the barcode list
        Returns:
            A list of 0's and 1's denoting the barcode
        """
        return [self._data.loc[index][n] for n in self.get_bit_names()]

    def get_barcode_count(self) -> int: 
        """
        Get the number of barcodes in this codebook.

        Returns:
            The number of barcodes, counting barcodes for blanks and genes
        """
        return len(self._data)

    def get_bit_count(self) -> int:
        """
        Get the number of bits used for MERFISH barcodes in this codebook.
        """
        return len(self.get_bit_names())

    def get_bit_names(self) -> List[str]:
        """Get the names of the bits for this MERFISH data set.

        Returns:
            A list of the names of the bits in order from the lowest to highest
        """
        return [s for s in self._data.columns if s not in ['name', 'id']]

    def get_barcodes(self, ignoreBlanks=False) -> np.array:
        """Get the barcodes present in this codebook.
        
        Args:
            ignoreBlanks: flag indicating whether barcodes corresponding 
                    to blanks should be included.
        Returns:
            A list of the barcodes represented as lists of bits.
        """
        bitNames = self.get_bit_names()
        if ignoreBlanks:
            return np.array([[x[n] for n in bitNames] for i, x
                             in self._data.iterrows()
                             if 'BLANK' not in x['name'].upper()])
        else:
            return np.array([[x[n] for n in bitNames]
                             for i, x in self._data.iterrows()])

    def get_coding_indexes(self) -> List[int]:
        """Get the barcode indexes that correspond with genes.

        Returns:
            A list of barcode indexes that correspond with genes and not 
                    blanks
        """
        return self._data[
                ~self._data['name'].str.contains('Blank', case=False)].index
    
    def get_blank_indexes(self) -> List[int]:
        """Get the barcode indexes that do not correspond with genes.

        Returns:
            A list of barcode indexes that correspond with blanks
        """
        return self._data[
                self._data['name'].str.contains('Blank', case=False)].index

    def get_gene_names(self) -> List[str]:
        """"Get the names of the genes represented in this codebook.

        Returns:
            A list of the gene names. The list does not contain the names of
            the blanks.
        """
        return self._data.loc[self.get_coding_indexes()]['name'].tolist()

    def get_name_for_barcode_index(self, index: int) -> str:
        """Get the gene name for the barcode with the specified index.

        Returns:
            The gene name
        """
        return self._data.loc[index]['name']

    def get_barcode_index_for_name(self, name: str) -> Union[int, None]:
        """Get the barcode index for the barcode with the specified name.

        Returns:
            The barcode index. If name appears more than once, the index of
            the first appearance is returned. If name is not in this codebook
            then None is returned.
        """
        matches = self._data[self._data['name'].str.match('^' + name + '$')]
        if len(matches) == 0:
            return None
        return matches.index[0]

    def get_codebook_name(self):
        """
        Gets the original file name used to generate a codebook saved in the
        analysis directory

        Returns:
            Original file name of codebook
        """
        return self.codebook_name

