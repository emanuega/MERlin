from matplotlib import pyplot as plt
import numpy as np

from merlin.plots._base import AbstractPlot


class CodingBarcodeSpatialDistribution(AbstractPlot):

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

