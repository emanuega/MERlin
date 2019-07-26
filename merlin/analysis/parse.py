import os
import merlin
import pandas
from merlin.util import spatialfeature
from shapely import geometry
from scipy import ndimage
from merlin.core import dataset

from merlin.analysis import segment

from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt


class Parse(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.filterTask = self.dataSet.load_analysis_task(self.parameters['filter_barcodes_task'])
        self.assignmentTask = self.dataSet.load_analysis_task(self.parameters['assignment_task'])
        self.alignTask = self.dataSet.load_analysis_task(self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['filter_barcodes_task'],
                self.parameters['assignment_task'],
                self.parameters['global_align_task']]

    def get_parsed_barcodes(self, fov: int=None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_parsed_barcodes(fov) for fov in self.dataSet.get_fovs()]
            ).reset_index(drop=True)

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov)



    def _run_analysis(self, fragmentIndex):

        bcDB = self.filterTask.get_barcode_database()
        sDB = self.assignmentTask.get_feature_database()
        currentFOVBarcodes = bcDB.get_barcodes(fragmentIndex)
        currentCells = sDB.read_features(fragmentIndex)
        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        countsDF = pandas.DataFrame(data=np.zeros((len(currentCells), self.dataSet.get_codebook().get_barcode_count())),
                                    columns=range(self.dataSet.get_codebook().get_barcode_count()),
                                    index=[x.get_feature_id() for x in currentCells])
        for cell in currentCells:
            cellMinX, cellMinY, cellMaxX, cellMaxY = cell.get_bounding_box()
            cellCount = [0] * data.get_codebook().get_barcode_count()
            allBoundaries = cell.get_boundaries()
            barcodesToConsider = currentFOVBarcodes[(currentFOVBarcodes['global_x'] >= cellMinX) &
                                                    (currentFOVBarcodes['global_x'] <= cellMaxX) &
                                                    (currentFOVBarcodes['global_y'] >= cellMinY) &
                                                    (currentFOVBarcodes['global_y'] <= cellMaxY)]
            for zPos in list(range(len(allBoundaries))):
                if len(allBoundaries[zPos]) > 0:
                    for elem in allBoundaries[zPos]:
                        currentZBarcodes = barcodesToConsider[barcodesToConsider['z'] == zPos].loc[:,
                                           ['global_x', 'global_y', 'barcode_id']]
                        if len(currentZBarcodes) > 0:
                            points = [geometry.Point(x[0], x[1]) for x in
                                      currentZBarcodes.loc[:, ['global_x', 'global_y']].values.tolist()]
                            within = [elem.contains(point) for point in points]
                            hits = np.where(within)
                            hits = np.take(currentZBarcodes['barcode_id'].values.tolist(), hits)[0]
                            uniqueHits = list(set(hits))
                            for hit in uniqueHits:
                                cellCount[hit] += list(hits).count(hit)
            countsDF.loc[cell.get_feature_id(), :] = cellCount

        names = [data.get_codebook().get_name_for_barcode_index(x) for x in countsDF.columns.values.tolist()]
        countsDF.columns = names

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)

class ExportParsedBarcodes(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['parse_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['parse_task'])
        parsedBarcodes = pTask.get_parsed_barcodes()

        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'parsed_barcodes_combined',
                    self.get_analysis_name())
