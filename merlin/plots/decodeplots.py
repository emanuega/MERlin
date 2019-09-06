from matplotlib import pyplot as plt
import numpy as np
import pandas

from merlin.plots._base import AbstractPlot, PlotMetadata


class MinimumDistanceDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'decode_task': 'all'}

    def get_required_metadata(self):
        return [DecodedBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        decodeMetadata = inputMetadata[
            'decodeplots.DecodedBarcodesMetadata']

        distanceX = decodeMetadata.distanceBins[:-1]
        shift = (distanceX[0] + distanceX[1]) / 2
        distanceX = [x + shift for x in distanceX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(distanceX, decodeMetadata.distanceCounts)
        plt.xlabel('Barcode distance')
        plt.ylabel('Count')
        plt.title('Distance distribution for all barcodes')
        plt.tight_layout(pad=0.2)

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
            'decodeplots.DecodedBarcodesMetadata']
        areaX = decodeMetadata.areaBins[:-1]
        shift = (areaX[0] + areaX[1]) / 2
        areaX = [x + shift for x in areaX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(areaX, decodeMetadata.areaCounts)
        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Count')
        plt.title('Area distribution for all barcodes')
        plt.tight_layout(pad=0.2)

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
            'decodeplots.DecodedBarcodesMetadata']
        intensityX = decodeMetadata.intensityBins
        shift = (intensityX[0] + intensityX[1]) / 2
        intensityX = [x + shift for x in intensityX]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(intensityX, decodeMetadata.intensityCounts)
        plt.xlabel('Mean intensity ($log_{10}$)')
        plt.ylabel('Count')
        plt.title('Intensity distribution for all barcodes')
        plt.tight_layout(pad=0.2)

        return fig


class DecodedBarcodesMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.decodeTask = self._taskDict['decode_task']

        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*self.decodeTask.fragment_count())
        self.areaBins = np.arange(25)
        self.areaCounts = self._load_numpy_metadata(
            'area_counts', np.zeros(len(self.areaBins)))

        if np.sum(self.completeFragments >= min(
                20, self.decodeTask.fragment_count())):
            self.intensityBins = self._load_numpy_metadata('intensity_bins')
            self.intensityCounts = self._load_numpy_metadata(
                'intensity_counts', np.zeros(len(self.intensityBins)))
            self.distanceBins = self._load_numpy_metadata('distance_bins')
            self.distanceCounts = self._load_numpy_metadata(
                'distance_counts', np.zeros(len(self.distanceBins)))
            self.intensityCountsByArea = self._load_numpy_metadata(
                'intensity_by_area',
                np.zeros(len(self.areaBins), len(self.intensityBins)))
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
            np.zeros(len(self.areaBins), len(self.intensityBins))
        self.distanceCounts = np.zeros(len(self.distanceBins))
        self.intensityCounts = np.zeros(len(self.intensityBins))

        self._save_numpy_metadata(self.intensityBins, 'intensity_bins')
        self._save_numpy_metadata(self.distanceBins, 'distance_bins')

        self.binsDetermined = True

    def _extract_from_barcodes(self, barcodes):
        self.areaCounts += np.histogram(barcodes['area'],
                                        bins=self.areaBins)[0]
        self.intensityCounts += np.histogram(
            np.log10(barcodes['mean_intensity']), bins=self.intensityBins)[0]
        self.distanceCounts += np.histogram(
            np.log10(barcodes['min_distance']), bins=self.distanceBins)[0]

        for i, currentArea in enumerate(self.areaBins):
            self.intensityCountsByArea[i, :] += np.histogram(
                np.log10(barcodes['mean_intensity']),
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
                        columnList=['area', 'mean_intensity', 'min_distance']))

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
            self._save_numpy_metadata(self.areaCounts, 'area_counts')
            self._save_numpy_metadata(self.intensityCounts, 'intensity_counts')
            self._save_numpy_metadata(self.distanceCounts, 'distance_counts')
            self._save_numpy_metadata(self.intensityCountsByArea,
                                      'intensity_by_area')

    def is_complete(self) -> bool:
        return all(self.completeFragments)
