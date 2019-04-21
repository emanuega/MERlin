import numpy as np
import pandas
from merlin.core import analysistask
from merlin.util import barcodedb


class FilterBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_barcode_database(self):
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        barcodeDB = self.get_barcode_database()
        currentBC = decodeTask.get_barcode_database() \
            .get_filtered_barcodes(areaThreshold, intensityThreshold,
                                   fov=fragmentIndex)
        barcodeDB.write_barcodes(currentBC, fov=fragmentIndex)


class AdaptiveFilterBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'blank_fraction' not in self.parameters:
            self.parameters['blank_fraction'] = 0.1

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_barcode_database(self):
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def get_adaptive_thresholds(self):
        return self.dataSet.load_numpy_analysis_result('adaptive_thresholds',
                                                       self)

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = self.dataSet.get_codebook()

        try:
            thresholds = self.get_adaptive_thresholds()

        except IOError:
            allBarcodes = decodeTask.get_barcode_database().get_barcodes()
            blankBarcodes = allBarcodes[allBarcodes['barcode_id'].isin(
                codebook.get_blank_indexes())]
            blankFraction = len(codebook.get_blank_indexes()) / (
                    len(codebook.get_blank_indexes())
                    + len(codebook.get_coding_indexes()))

            maxIntensity = np.log10(np.max(allBarcodes['mean_intensity']))
            minIntensity = np.log10(np.min(allBarcodes['mean_intensity']))
            intensityStep = (maxIntensity-minIntensity)/20
            histBins = np.arange(minIntensity, maxIntensity, intensityStep)

            nonzeroBlankCounts = np.array([np.histogram(np.log10(
                blankBarcodes[blankBarcodes['area'] == a]['mean_intensity']),
                                                        bins=histBins)[0]
                       for a in np.arange(1, np.max(allBarcodes['area'])+1)])

            nonzeroCounts = np.array([np.histogram(np.log10(
                allBarcodes[allBarcodes['area'] == a]['mean_intensity']),
                bins=histBins)[0] for a
                in np.arange(1, np.max(allBarcodes['area'])+1)])

            blankFractionMatrix = nonzeroBlankCounts / nonzeroCounts
            blankFractionMatrix[np.isnan(blankFractionMatrix)] = 0

            thresholdedBlankFraction = \
                blankFractionMatrix \
                > blankFraction*self.parameters['blank_fraction']

            thresholds = np.array([histBins[np.where(a)[0][-1] + 1]
                                   if True in a else histBins[0]
                                   for a in thresholdedBlankFraction])

            self.dataSet.save_numpy_analysis_result(
                thresholds, 'adaptive_thresholds', self)

        fovBarcodes = decodeTask.get_barcode_database().get_barcodes(
            fragmentIndex)

        filteredBarcodes = pandas.concat(
            [allBarcodes[(fovBarcodes['area'] == a) &
                         (np.log10(fovBarcodes['mean_intensity']) > thresholds[
                             a - 1])]
             for a in np.unique(fovBarcodes['area'])])

        self.get_barcode_database().write_barcodes(
            filteredBarcodes, fov=fragmentIndex)
