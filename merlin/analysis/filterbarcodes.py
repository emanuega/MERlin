import numpy as np
from scipy import optimize
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
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 1e6

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
        distanceThreshold = self.parameters['distance_threshold']
        barcodeDB = self.get_barcode_database()
        currentBC = decodeTask.get_barcode_database() \
            .get_filtered_barcodes(areaThreshold, intensityThreshold,
                                   distanceThreshold=distanceThreshold,
                                   fov=fragmentIndex)
        barcodeDB.write_barcodes(currentBC, fov=fragmentIndex)


class AdaptiveFilterBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

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

    @staticmethod
    def _extract_barcodes_with_threshold(blankThreshold, barcodeSet,
                                         blankFractionHistogram, histogramBins):
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        selectData[:, 0] = np.log10(selectData[:, 0])

        selectData[selectData[:, 2] >= 34, 2] = 33

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], histogramBins[0], right=True),
             np.digitize(selectData[:, 1], histogramBins[1], right=True),
             np.digitize(selectData[:, 2], histogramBins[2]))) - 1
        raveledIndexes = np.ravel_multi_index(barcodeBins[:, :],
                                              blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    def _calculate_error_rate(self, barcodeSet):
        codebook = self.dataSet.get_codebook(idx=self.codebookNum)
        blankFraction = len(
            codebook.get_blank_indexes()) / codebook.get_barcode_count()
        return np.sum(barcodeSet['barcode_id'].isin(
            codebook.get_blank_indexes())) / (len(barcodeSet) * blankFraction)

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = self.dataSet.get_codebook(idx=self.codebookNum)

        allBarcodes = decodeTask.get_barcode_database().get_barcodes(
            columnList=['barcode_id', 'mean_intensity', 'min_distance', 'area'])
        allBarcodes = allBarcodes[(allBarcodes['area'] >= 2)
                                  & (allBarcodes['min_distance'] < 0.61)]

        blankBarcodes = allBarcodes[allBarcodes['barcode_id'].isin(
            codebook.get_blank_indexes())]
        maxIntensity = np.max(np.log10(allBarcodes['mean_intensity']))
        maxDistance = np.max(allBarcodes['min_distance'])

        blankData = blankBarcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        blankData[:, 0] = np.log10(blankData[:, 0])
        intensityBins = np.arange(0, 1.021 * maxIntensity, maxIntensity / 50)
        distanceBins = np.arange(0, maxDistance+0.02, 0.01)
        blankHistogram = np.histogramdd(
            blankData, bins=(intensityBins, distanceBins, np.arange(35)))

        allData = allBarcodes[['mean_intensity', 'min_distance', 'area']].values
        allData[:, 0] = np.log10(allData[:, 0])
        allHistogram = np.histogramdd(
            allData, bins=(intensityBins, distanceBins, np.arange(35)))

        blankFraction = blankHistogram[0] / allHistogram[0]
        blankFraction[allHistogram[0] == 0] = 0
        blankFraction = blankFraction / (len(codebook.get_blank_indexes()) / (
                    len(codebook.get_blank_indexes()) + len(
                codebook.get_coding_indexes())))

        def misidentification_rate_error_for_threshold(x, targetError):
            result = self._calculate_error_rate(
                self._extract_barcodes_with_threshold(
                    x, allBarcodes, blankFraction, allHistogram[1])) \
                     - targetError
            return result

        threshold = optimize.newton(misidentification_rate_error_for_threshold,
                                    0.2, args=[0.05], tol=0.001, x1=0.3)

        bcDatabase = self.get_barcode_database()
        for f in self.dataSet.get_fovs():
            currentBarcodes = decodeTask.get_barcode_database().get_barcodes(f)
            currentBarcodes = currentBarcodes[(currentBarcodes['area'] >= 2) & (
                        currentBarcodes['min_distance'] < 0.61)]
            bcDatabase.write_barcodes(self._extract_barcodes_with_threshold(
                    threshold, currentBarcodes), fov=f)
