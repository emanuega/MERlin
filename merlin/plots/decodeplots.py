from matplotlib import pyplot as plt
import numpy as np
import pandas

from merlin.plots._base import AbstractPlot, PlotMetadata
from merlin.analysis import filterbarcodes


class MinimumDistanceDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeMetadata = inputMetadata[
            'decodeplots/DecodedBarcodesMetadata']

        distanceX = decodeMetadata.distanceBins[:-1]
        shift = (distanceX[0] + distanceX[1]) / 2
        distanceX = [x + shift for x in distanceX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(distanceX, decodeMetadata.distanceCounts)
        plt.xlabel('Barcode distance')
        plt.ylabel('Count')
        plt.title('Distance distribution for all barcodes')

        return fig


class AreaDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeMetadata = inputMetadata[
            'decodeplots/DecodedBarcodesMetadata']
        areaX = decodeMetadata.areaBins[:-1]
        shift = (areaX[0] + areaX[1]) / 2
        areaX = [x + shift for x in areaX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(areaX, decodeMetadata.areaCounts,
                width=2*shift)
        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Count')
        plt.title('Area distribution for all barcodes')

        return fig


class MeanIntensityDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeMetadata = inputMetadata[
            'decodeplots/DecodedBarcodesMetadata']
        intensityX = decodeMetadata.intensityBins[:-1]
        shift = (intensityX[0] + intensityX[1]) / 2
        intensityX = [x + shift for x in intensityX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(intensityX, decodeMetadata.intensityCounts,
                width=2*shift)
        plt.xlabel('Mean intensity ($log_{10}$)')
        plt.ylabel('Count')
        plt.title('Intensity distribution for all barcodes')

        return fig


class DecodedBarcodeAbundancePlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeTask = inputTasks['decode_task']
        codebook = decodeTask.get_codebook()
        decodeMetadata = inputMetadata[
            'decodeplots/DecodedBarcodesMetadata']

        barcodeCounts = decodeMetadata.barcodeCounts
        countDF = pandas.DataFrame(decodeMetadata.barcodeCounts,
                                   index=np.arange(len(barcodeCounts)),
                                   columns=['counts'])

        codingDF = countDF[countDF.index.isin(codebook.get_coding_indexes())]\
            .sort_values(by='counts', ascending=False)
        blankDF = countDF[countDF.index.isin(codebook.get_blank_indexes())]\
            .sort_values(by='counts', ascending=False)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(codingDF)), np.log10(codingDF['counts']),
                 'b.')
        plt.plot(np.arange(len(codingDF), len(countDF)),
                 np.log10(blankDF['counts']), 'r.')
        plt.xlabel('Sorted barcode index')
        plt.ylabel('Count (log10)')
        plt.title('Barcode abundances')
        plt.legend(['Coding', 'Blank'])

        return fig


class AreaIntensityViolinPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeTask = inputTasks['decode_task']
        decodeMetadata = inputMetadata[
            'decodeplots/DecodedBarcodesMetadata']

        bins = decodeMetadata.intensityBins[:-1]
        def distribution_dict(countList):
            if np.max(countList) > 0:
                return {'coords': bins,
                   'vals': countList,
                   'mean': np.average(bins, weights=countList),
                   'median': np.average(bins, weights=countList),
                   'min': bins[np.nonzero(countList)[0]],
                   'max': bins[np.nonzero(countList)[-1]]}
            else:
                return {'coords': bins,
                   'vals': countList,
                   'mean': 0,
                   'median': 0,
                   'min': 0,
                   'max': 0}
        vpstats = [distribution_dict(x) for x in 
                   decodeMetadata.intensityCountsByArea]

        fig = plt.figure(figsize=(15, 5))
        ax = plt.subplot(1,1,1)
        ax.violin(vpstats, positions=decodeMetadata.areaBins[:-1], 
                  showmeans=True, showextrema=False)

        if 'filter_task' in inputTasks and isinstance(
                inputTasks['filter_task'], 
                filterbarcodes.FilterBarcodes):
            plt.axvline(
                x=inputTasks['filter_task'].parameters['area_threshold'] - 0.5,
                color='green', linestyle=':')
            plt.axhline(y=np.log10(
                inputTasks['filter_task'].parameters['intensity_threshold']),
                color='green', linestyle=':')

        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Mean intensity ($log_{10}$)')
        plt.title('Intensity distribution by barcode area')
        plt.xlim([0, 17])

        return fig


class DecodedBarcodesMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.decodeTask = self._taskDict['decode_task']
        codebook = self.decodeTask.get_codebook()

        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*self.decodeTask.fragment_count())
        self.areaBins = np.arange(25)
        self.areaCounts = self._load_numpy_metadata(
            'area_counts', np.zeros(len(self.areaBins)-1))

        self.barcodeCounts = self._load_numpy_metadata(
            'barcode_counts', np.zeros(codebook.get_barcode_count()))
        if np.sum(self.completeFragments) >= min(
                20, self.decodeTask.fragment_count()):
            self.intensityBins = self._load_numpy_metadata('intensity_bins')
            self.intensityCounts = self._load_numpy_metadata(
                'intensity_counts', np.zeros(len(self.intensityBins)-1))
            self.distanceBins = self._load_numpy_metadata('distance_bins')
            self.distanceCounts = self._load_numpy_metadata(
                'distance_counts', np.zeros(len(self.distanceBins)-1))
            self.intensityCountsByArea = self._load_numpy_metadata(
                'intensity_by_area',
                np.zeros((len(self.areaBins)-1, len(self.intensityBins)-1)))
            self.binsDetermined = True

        self.binsDetermined = False
        self.queuedBarcodeData = []

    def _determine_bins(self):
        aggregateDF = pandas.concat(self.queuedBarcodeData)
        minIntensity = np.log10(aggregateDF['mean_intensity'].min())
        maxIntensity = np.log10(aggregateDF['mean_intensity'].max())
        self.intensityBins = np.linspace(minIntensity, maxIntensity, 100)

        minDistance = aggregateDF['min_distance'].min()
        maxDistance = aggregateDF['min_distance'].max()
        self.distanceBins = np.linspace(minDistance, maxDistance, 100)

        self.intensityCountsByArea = \
            np.zeros((len(self.areaBins)-1, len(self.intensityBins)-1))
        self.distanceCounts = np.zeros(len(self.distanceBins)-1)
        self.intensityCounts = np.zeros(len(self.intensityBins)-1)

        self._save_numpy_metadata(self.intensityBins, 'intensity_bins')
        self._save_numpy_metadata(self.distanceBins, 'distance_bins')

        self.binsDetermined = True

    def _extract_from_barcodes(self, barcodes):
        self.barcodeCounts += np.histogram(
            barcodes['barcode_id'], 
            bins=np.arange(len(self.barcodeCounts)+1))[0]

        self.areaCounts += np.histogram(barcodes['area'],
                                        bins=self.areaBins)[0]
        self.intensityCounts += np.histogram(
            np.log10(barcodes['mean_intensity']), bins=self.intensityBins)[0]
        self.distanceCounts += np.histogram(
            np.log10(barcodes['min_distance']), bins=self.distanceBins)[0]

        for i, currentArea in enumerate(self.areaBins[:-1]):
            self.intensityCountsByArea[i, :] += np.histogram(
                np.log10(barcodes[barcodes['area']==currentArea][
                    'mean_intensity']),
                bins=self.intensityBins)[0]

    def update(self) -> None:
        updated = False
        decodeTask = self._taskDict['decode_task']

        for i in range(decodeTask.fragment_count()):
            if not self.completeFragments[i] and decodeTask.is_complete(i):
                self.completeFragments[i] = True

                self.queuedBarcodeData.append(
                    decodeTask.get_barcode_database().get_barcodes(
                        i,
                        columnList=['barcode_id', 'area', 'mean_intensity',
                                    'min_distance']))

                if np.sum(self.completeFragments) \
                        >= min(20, decodeTask.fragment_count()):
                    if not self.binsDetermined:
                        self._determine_bins()

                    for bcData in self.queuedBarcodeData:
                        self._extract_from_barcodes(bcData)
                        updated = True
                    self.queuedBarcodeData = []

        if updated:
            self._save_numpy_metadata(self.completeFragments,
                                      'complete_fragments')
            self._save_numpy_metadata(self.barcodeCounts, 'barcode_counts')
            self._save_numpy_metadata(self.areaCounts, 'area_counts')
            self._save_numpy_metadata(self.intensityCounts, 'intensity_counts')
            self._save_numpy_metadata(self.distanceCounts, 'distance_counts')
            self._save_numpy_metadata(self.intensityCountsByArea,
                                      'intensity_by_area')

    def is_complete(self) -> bool:
        return all(self.completeFragments)
