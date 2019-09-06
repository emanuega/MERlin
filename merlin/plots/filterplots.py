from matplotlib import pyplot as plt
import numpy as np
from typing import List

from merlin.plots._base import AbstractPlot, PlotMetadata


class CodingBarcodeSpatialDistribution(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        codebook = inputTasks['filter_task'].get_codebook()
        bc = inputTasks['filter_task']\
            .get_barcode_database().get_barcodes(
                columnList=['barcode_id', 'global_x', 'global_y'])
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        codingIDs = codebook.get_coding_indexes()
        codingBC = bc[bc['barcode_id'].isin(codingIDs)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(codingBC['global_x'], codingBC['global_y'],
                      bins=(
                      np.ceil(maxX - minX) / 5, np.ceil(maxY - minY) / 5),
                      cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of identified barcodes')

        return fig


class BlankBarcodeSpatialDistribution(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'filter_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        codebook = self.inputTasks['filter_task'].get_codebook()
        bc = self.inputTasks['filter_task']\
            .get_barcode_database().get_barcodes(
                columnList=['barcode_id', 'global_x', 'global_y'])
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        blankIDs = codebook.get_coding_indexes()
        codingBC = bc[bc['barcode_id'].isin(blankIDs)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(codingBC['global_x'], codingBC['global_y'],
                      bins=(
                      np.ceil(maxX - minX) / 5, np.ceil(maxY - minY) / 5),
                      cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of identified barcodes')

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
            'filterplots.FOVSpatialDistributionMetadata']
        singleColorCounts = spatialMetadata.singleColorCounts
        plt.plot(spatialMetadata.radialBins,
                 singleColorCounts/np.sum(singleColorCounts))
        multiColorCounts = spatialMetadata.multiColorCounts
        plt.plot(spatialMetadata.radialBins,
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

        spatialMetadata = inputMetadata[
            'filterplots.FOVSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.spatialCodingCounts,
                   extent=spatialMetadata.get_spatial_extents())
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of coding barcodes within FOV')

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
            'filterplots.FOVSpatialDistributionMetadata']
        plt.imshow(spatialMetadata.blankCodingCounts,
                   extent=spatialMetadata.get_spatial_extents())
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Spatial distribution of blank barcodes within FOV')

        return fig


class FOVSpatialDistributionMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.filterTask = self._taskDict['filter_task']

        dataSet = self._analysisTask.dataSet
        self._width = dataSet.get_image_dimesions()[0]
        self._height = dataSet.get_image_dimensions()[1]
        imageSize = np.sqrt(self._height**2 + self._width**2)
        self.radialBins = self._load_numpy_metadata(
            'radial_bins', np.arange(0, 0.5*imageSize, (0.5*imageSize)/200))
        self.spatialXBins = self._load_numpy_metadata(
            'spatial_x_bins', np.arange(0, self._width, 0.01*self._width))
        self.spatialYBins = self._load_numpy_metadata(
            'spatial_y_bins', np.arange(0, self._height, 0.01*self._height))
        self.completeFragments = self._load_numpy_metadata(
            'complete_fragments', [False]*self.filterTask.fragment_count())
        self.multiColorCounts = self._load_numpy_metadata(
            'multi_color_radial_counts', np.zeros(len(self.radialBins)))
        self.singleColorCounts = self._load_numpy_metadata(
            'single_color_radial_counts', np.zeros(len(self.radialBins)))
        self.spatialCodingCounts = self._load_numpy_metadata(
            'spatial_coding_counts',
            np.zeros((len(self.spatialXBins), len(self.spatialYBins))))
        self.spatialBlankCounts = self._load_numpy_metadata(
            'spatial_blank_counts',
            np.zeros((len(self.spatialXBins), len(self.spatialYBins))))

        bitColors = dataSet.get_data_organization().data['color']
        bcSet = self.filterTask.get_codebook().get_barcodes()
        self.singleColorBarcodes = [i for i, b in enumerate(bcSet) if
                                    bitColors[np.where(b)[0]].nunique() == 1]
        self.multiColorBarcodes = [i for i, b in enumerate(bcSet) if
                                   bitColors[np.where(b)[0]].nunique() > 1]

    def get_spatial_extents(self) -> List[float]:
        return [self.spatialXBins[0], self.spatialXBins[-1],
                self.spatialYBins[0], self.spatialYBins[1]]

    def _radial_distance(self, x: float, y: float) -> float:
        return np.sqrt((x - 0.5 * self._width) ** 2
                       + (y - 0.5 * self._height) ** 2)

    def _radial_distribution(self, inputBarcodes, barcodeIDs):
        selectBarcodes = inputBarcodes['barcode_id'].isin(barcodeIDs)
        radialDistances = [self._radial_distance(r['x'], r['y'])
                           for i, r in selectBarcodes.iterrows()]
        return np.histogram(radialDistances, bins=self.radialBins)[0]

    def _spatial_distribution(self, inputBarcodes, barcodeIDs):
        selectBarcodes = inputBarcodes['barcode_id'].isin(barcodeIDs)
        return np.histogram2d(selectBarcodes['x'], selectBarcodes['y'],
                              bins=(self.spatialXBins, self.spatialYBins))[0]

    def update(self) -> None:
        updated = False
        filterTask = self._taskDict['filter_task']
        codebook = filterTask.get_codebook()

        for i in range(filterTask.fragment_count()):
            if not self.completeFragments[i] and filterTask.is_complete(i):
                fovBarcodes = filterTask.get_barcode_database().get_barcocdes(
                    i, columnList=['barcode_id', 'x', 'y'])

                self.spatialCodingCounts += self._spatial_distribution(
                    fovBarcodes, codebook.get_coding_indexes())
                self.spatialBlankCounts += self._spatial_distribution(
                    fovBarcodes, codebook.get_blank_indexes())
                self.singleColorCounts += self._radial_distribution(
                    fovBarcodes, self.singleColorBarcodes)
                self.multiColorCounts += self._radial_distribution(
                    fovBarcodes, self.multiColorBarcodes)

                self.completeFragments[i] = True
                updated = True

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
