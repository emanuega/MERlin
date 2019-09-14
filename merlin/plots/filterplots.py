from matplotlib import pyplot as plt
import numpy as np
import pandas
from typing import List

from merlin.plots._base import AbstractPlot, PlotMetadata
from merlin.analysis import filterbarcodes


class CodingBarcodeSpatialDistribution(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all',
                'global_align_task': 'all'}

    def get_required_metadata(self):
        return [GlobalSpatialDistributionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        spatialMetadata = inputMetadata[
            'filterplots/GlobalSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.spatialCodingCounts,
                   extent=spatialMetadata.get_spatial_extents(),
                   cmap=plt.get_cmap('Greys'))
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of coding barcodes')
        cbar = plt.colorbar(ax=ax)
        cbar.set_label('Barcode count', rotation=270)

        return fig


class BlankBarcodeSpatialDistribution(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all',
                'global_align_task': 'all'}

    def get_required_metadata(self):
        return [GlobalSpatialDistributionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        spatialMetadata = inputMetadata[
            'filterplots/GlobalSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.spatialBlankCounts,
                   extent=spatialMetadata.get_spatial_extents(),
                   cmap=plt.get_cmap('Greys'))
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of blank barcodes')
        cbar = plt.colorbar(ax=ax)
        cbar.set_label('Barcode count', rotation=270)

        return fig


class BarcodeRadialDensityPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return [FOVSpatialDistributionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(7, 7))

        spatialMetadata = inputMetadata[
            'filterplots/FOVSpatialDistributionMetadata']
        singleColorCounts = spatialMetadata.singleColorCounts
        plt.plot(spatialMetadata.radialBins[:-1],
                 singleColorCounts/np.sum(singleColorCounts))
        multiColorCounts = spatialMetadata.multiColorCounts
        plt.plot(spatialMetadata.radialBins[:-1],
                 multiColorCounts/np.sum(multiColorCounts))
        plt.legend(['Single color barcodes', 'Multi color barcodes'])
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Normalized radial barcode density')

        return fig


class CodingBarcodeFOVDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return [FOVSpatialDistributionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        spatialMetadata = inputMetadata[
            'filterplots/FOVSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.spatialCodingCounts,
                   extent=spatialMetadata.get_spatial_extents(),
                   cmap=plt.get_cmap('Greys'))
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of coding barcodes within FOV')
        cbar = plt.colorbar(ax=ax)
        cbar.set_label('Barcode count', rotation=270)

        return fig


class BlankBarcodeFOVDistributionPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return [FOVSpatialDistributionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(7, 7))

        spatialMetadata = inputMetadata[
            'filterplots/FOVSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.spatialBlankCounts,
                   extent=spatialMetadata.get_spatial_extents())
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of blank barcodes within FOV')

        return fig


class FilteredBarcodeAbundancePlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return [FilteredBarcodesMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        filterTask = inputTasks['filter_task']
        codebook = filterTask.get_codebook()
        decodeMetadata = inputMetadata[
            'filterplots/FilteredBarcodesMetadata']

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
        plt.tight_layout(pad=0.2)

        return fig


class AdaptiveFilterBarcodeDistributionPlots(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': filterbarcodes.AdaptiveFilterBarcodes}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        filterTask = inputTasks['filter_task']
        adaptiveTask = inputTasks['filter_task'].get_adaptive_thresholds()
        blankHistogram = adaptiveTask.get_blank_count_histogram()
        codingHistogram = adaptiveTask.get_coding_count_histogram()
        blankFraction = adaptiveTask.get_blank_fraction_histogram()
        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            filterTask.parameters['misidentification_rate'])
        areaBins = adaptiveTask.get_area_bins()
        intensityBins = adaptiveTask.get_intensity_bins()
        distanceBins = adaptiveTask.get_distance_bins()
        plotExtent = (distanceBins[0], distanceBins[-1],
                      intensityBins[0], intensityBins[-1])

        fig = plt.figure(figsize=(20, 30))
        for i in range(min(len(areaBins), 6)):
            plt.subplot(6, 4, 4*i+1)
            plt.imshow(blankHistogram[:, :, i].T + codingHistogram[:, :, i].T,
                       extent=plotExtent, origin='lower', aspect='auto', 
                       cmap='OrRd')
            cbar = plt.colorbar()
            cbar.set_label('Barcode count', rotation=270, labelpad=8)
            plt.ylabel('Area=%i\nMean intensity (log10)' % areaBins[i])
            plt.xlabel('Minimum distance')
            if i==0:
                plt.title('Distribution of all barcodes')

            plt.subplot(6, 4, 4*i+2)
            plt.imshow(blankHistogram[:, :, i].T, extent=plotExtent, 
                       origin='lower', aspect='auto', cmap='OrRd')
            cbar = plt.colorbar()
            cbar.set_label('Blank count', rotation=270, labelpad=8)
            plt.ylabel('Mean intensity (log10)')
            plt.xlabel('Minimum distance')
            if i==0:
                plt.title('Distribution of blank barcodes')

            plt.subplot(6, 4, 4*i+3)
            plt.imshow(blankFraction[:, :, i].T, extent=plotExtent, 
                       origin='lower', aspect='auto', cmap='OrRd',
                       vmax = 1.0)
            cbar = plt.colorbar()
            cbar.set_label('Blank fraction', rotation=270, labelpad=8)
            plt.ylabel('Mean intensity (log10)')
            plt.xlabel('Minimum distance')
            if i==0:
                plt.title('Distribution of normalized blank fraction')

            plt.subplot(6, 4, 4*i+4)
            plt.imshow(blankFraction[:, :, i].T < threshold, extent=plotExtent, 
                       origin='lower', aspect='auto', cmap='OrRd')
            plt.ylabel('Mean intensity (log10)')
            plt.xlabel('Minimum distance')
            if i==0:
                plt.title('Accepted pixels')

        return fig


class FOVSpatialDistributionMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.filterTask = self._taskDict['filter_task']

        dataSet = self._analysisTask.dataSet
        self._width = dataSet.get_image_dimensions()[0]
        self._height = dataSet.get_image_dimensions()[1]
        imageSize = max(self._height, self._width)
        self.radialBins = self._load_numpy_metadata(
            'radial_bins', np.arange(0, 0.5*imageSize, (0.5*imageSize)/200))
        self.spatialXBins = self._load_numpy_metadata(
            'spatial_x_bins', np.arange(0, self._width, 0.01*self._width))
        self.spatialYBins = self._load_numpy_metadata(
            'spatial_y_bins', np.arange(0, self._height, 0.01*self._height))
        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*self.filterTask.fragment_count())
        self.multiColorCounts = self._load_numpy_metadata(
            'multi_color_radial_counts', np.zeros(len(self.radialBins)-1))
        self.singleColorCounts = self._load_numpy_metadata(
            'single_color_radial_counts', np.zeros(len(self.radialBins)-1))
        self.spatialCodingCounts = self._load_numpy_metadata(
            'spatial_coding_counts',
            np.zeros((len(self.spatialXBins)-1, len(self.spatialYBins)-1)))
        self.spatialBlankCounts = self._load_numpy_metadata(
            'spatial_blank_counts',
            np.zeros((len(self.spatialXBins)-1, len(self.spatialYBins)-1)))

        bitColors = dataSet.get_data_organization().data['color']
        bcSet = self.filterTask.get_codebook().get_barcodes()
        self.singleColorBarcodes = [i for i, b in enumerate(bcSet) if
                                    bitColors[np.where(b)[0]].nunique() == 1]
        self.multiColorBarcodes = [i for i, b in enumerate(bcSet) if
                                   bitColors[np.where(b)[0]].nunique() > 1]

    def get_spatial_extents(self) -> List[float]:
        return [self.spatialXBins[0], self.spatialXBins[-1],
                self.spatialYBins[0], self.spatialYBins[-1]]

    def _radial_distance(self, x: float, y: float) -> float:
        return np.sqrt((x - 0.5 * self._width) ** 2
                       + (y - 0.5 * self._height) ** 2)

    def _radial_distribution(self, inputBarcodes, barcodeIDs):
        selectBarcodes = inputBarcodes[
                inputBarcodes['barcode_id'].isin(barcodeIDs)]
        radialDistances = [self._radial_distance(r['x'], r['y'])
                           for i, r in selectBarcodes.iterrows()]
        return np.histogram(radialDistances, bins=self.radialBins)[0]

    def _spatial_distribution(self, inputBarcodes, barcodeIDs):
        selectBarcodes = inputBarcodes[
                inputBarcodes['barcode_id'].isin(barcodeIDs)]
        if len(selectBarcodes) > 1:
            return np.histogram2d(
                    selectBarcodes['x'], selectBarcodes['y'],
                    bins=(self.spatialXBins, self.spatialYBins))[0]
        else:
            return 0

    def update(self) -> None:
        updated = False
        filterTask = self._taskDict['filter_task']
        codebook = filterTask.get_codebook()

        for i in range(filterTask.fragment_count()):
            if not self.completeFragments[i] and filterTask.is_complete(i):
                fovBarcodes = filterTask.get_barcode_database().get_barcodes(
                    i, columnList=['barcode_id', 'x', 'y'])

                if len(fovBarcodes) > 0:
                    self.spatialCodingCounts += self._spatial_distribution(
                        fovBarcodes, codebook.get_coding_indexes())
                    self.spatialBlankCounts += self._spatial_distribution(
                        fovBarcodes, codebook.get_blank_indexes())
                    self.singleColorCounts += self._radial_distribution(
                        fovBarcodes, self.singleColorBarcodes)
                    self.multiColorCounts += self._radial_distribution(
                        fovBarcodes, self.multiColorBarcodes)
                    updated = True

                self.completeFragments[i] = True

        if updated:
            self._save_numpy_metadata(self.completeFragments,
                                      'complete_fragments')
            self._save_numpy_metadata(self.multiColorCounts,
                                      'multi_color_radial_counts')
            self._save_numpy_metadata(self.singleColorCounts,
                                      'single_color_radial_counts')
            self._save_numpy_metadata(self.spatialCodingCounts,
                                      'spatial_coding_counts')
            self._save_numpy_metadata(self.spatialBlankCounts,
                                      'spatial_blank_counts')

    def is_complete(self) -> bool:
        return all(self.completeFragments)


class FilteredBarcodesMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        filterTask = self._taskDict['filter_task']
        codebook = filterTask.get_codebook()

        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*filterTask.fragment_count())
        self.barcodeCounts = self._load_numpy_metadata(
            'barcode_counts', np.zeros(codebook.get_barcode_count()))

    def update(self) -> None:
        updated = False
        filterTask = self._taskDict['filter_task']

        for i in range(filterTask.fragment_count()):
            if not self.completeFragments[i] and filterTask.is_complete(i):
                self.completeFragments[i] = True

                barcodes = filterTask.get_barcode_database().get_barcodes(
                    i, columnList=['barcode_id'])

                self.barcodeCounts += np.histogram(
                    barcodes['barcode_id'],
                    bins=np.arange(len(self.barcodeCounts)+1))[0]

                updated = True

        if updated:
            self._save_numpy_metadata(self.completeFragments,
                                      'complete_fragments')
            self._save_numpy_metadata(self.barcodeCounts, 'barcode_counts')

    def is_complete(self) -> bool:
        return all(self.completeFragments)


class GlobalSpatialDistributionMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        filterTask = self._taskDict['filter_task']
        globalTask = self._taskDict['global_align_task']
        minX, minY, maxX, maxY = globalTask.get_global_extent()
        xStep = (maxX - minX)/1000
        yStep = (maxX - minX)/1000
        codebook = filterTask.get_codebook()

        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*filterTask.fragment_count())
        self.barcodeCounts = self._load_numpy_metadata(
            'barcode_counts', np.zeros(codebook.get_barcode_count()))
        self.spatialXBins = self._load_numpy_metadata(
            'spatial_x_bins', np.arange(minX, maxX, xStep))
        self.spatialYBins = self._load_numpy_metadata(
            'spatial_y_bins', np.arange(minY, maxY, yStep))
        self.spatialCodingCounts = self._load_numpy_metadata(
            'spatial_coding_counts',
            np.zeros((len(self.spatialXBins)-1, len(self.spatialYBins)-1)))
        self.spatialBlankCounts = self._load_numpy_metadata(
            'spatial_blank_counts',
            np.zeros((len(self.spatialXBins)-1, len(self.spatialYBins)-1)))

    def _spatial_distribution(self, inputBarcodes, barcodeIDs):
        selectBarcodes = inputBarcodes[
                inputBarcodes['barcode_id'].isin(barcodeIDs)]
        if len(selectBarcodes) > 1:
            return np.histogram2d(
                    selectBarcodes['global_x'], selectBarcodes['global_y'],
                    bins=(self.spatialXBins, self.spatialYBins))[0]
        else:
            return 0

    def get_spatial_extents(self) -> List[float]:
        globalTask = self._taskDict['global_align_task']
        minX, minY, maxX, maxY = globalTask.get_global_extent()
        return [minX, maxX, minY, maxY]

    def update(self) -> None:
        updated = False
        filterTask = self._taskDict['filter_task']
        codebook = filterTask.get_codebook()

        for i in range(filterTask.fragment_count()):
            if not self.completeFragments[i] and filterTask.is_complete(i):
                self.completeFragments[i] = True

                barcodes = filterTask.get_barcode_database().get_barcodes(
                    i, columnList=['barcode_id', 'global_x', 'global_y'])

                self.spatialCodingCounts += self._spatial_distribution(
                    barcodes, codebook.get_coding_indexes())
                self.spatialBlankCounts += self._spatial_distribution(
                    barcodes, codebook.get_blank_indexes())
                updated = True

        if updated:
            self._save_numpy_metadata(self.completeFragments,
                                      'complete_fragments')
            self._save_numpy_metadata(self.spatialCodingCounts,
                                      'spatial_coding_counts')
            self._save_numpy_metadata(self.spatialBlankCounts,
                                      'spatial_blank_counts')

    def is_complete(self) -> bool:
        return all(self.completeFragments)
