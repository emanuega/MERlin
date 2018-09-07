from matplotlib import pyplot as plt
import merlin
import os
plt.style.use(
        os.sep.join([os.path.dirname(merlin.__file__), 
            'ext', 'default.mplstyle']))
import numpy as np

from merlin.core import analysistask

class PlotPerformance(analysistask.AnalysisTask):

    '''An analysis task that generates plots depicting metrics of the MERFISH
    decoding.

    '''
    #TODO - I expect the plots of the barcode information can be 
    #made much more memory efficient.

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        #TODO - move this definition to run_analysis()
        self.decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        self.filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])
        self.segmentTask = self.dataSet.load_analysis_task(
                self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task'], self.parameters['filter_task']]

    #TODO - the functions in this class have too much repeated code
    #TODO - for the following 4 plots, I can add a line indicating the
    #barcode selection thresholds.
    def _plot_barcode_intensity_distribution(self):
        bcIntensities = self.decodeTask.get_barcode_database() \
                .get_barcode_intensities()
        fig = plt.figure(figsize=(4,4))
        plt.hist(np.log10(bcIntensities), bins=500)
        plt.xlabel('Mean intensity ($log_{10}$)')
        plt.ylabel('Count')
        plt.title('Intensity distribution for all barcodes')
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_intensity_distribution')

    def _plot_barcode_area_distribution(self):
        bcAreas = self.decodeTask.get_barcode_database() \
                .get_barcode_areas()
        fig = plt.figure(figsize=(4,4))
        plt.hist(bcAreas, bins=np.arange(15))
        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Count')
        plt.title('Area distribution for all barcodes')
        plt.xticks(np.arange(15))
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_area_distribution')

    def _plot_barcode_distance_distribution(self):
        bcDistances = self.decodeTask.get_barcode_database() \
                .get_barcode_distances()
        fig = plt.figure(figsize=(4,4))
        plt.hist(bcDistances, bins=500)
        plt.xlabel('Barcode distance')
        plt.ylabel('Count')
        plt.title('Distance distribution for all barcodes')
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_distance_distribution')

    def _plot_barcode_intensity_area_violin(self):
        barcodeDB = self.decodeTask.get_barcode_database()
        intensityData = [np.log10(
            barcodeDB.get_intensities_for_barcodes_with_area(x)) \
                    for x in range(1,15)]
        fig = plt.figure(figsize=(8,4))
        plt.violinplot(intensityData, showextrema=False, showmedians=True)
        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Mean intensity ($log_{10}$)')
        plt.title('Intensity distribution by barcode area')
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_intensity_area_violin')

    def _plot_blank_distribution(self):
        codebook = self.dataSet.get_codebook()
        bc = self.filterTask.get_barcode_database().get_barcodes()
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        blankIDs = codebook[codebook['name'].str.contains('Blank')]
        blankBC = bc[bc['barcode_id'].isin(blankIDs.index)]
        codingIDs = codebook[~codebook['name'].str.contains('Blank')]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(blankBC['global_x'], blankBC['global_y'],
            bins=(np.ceil(maxX-minX)/5, np.ceil(maxY-minY)/5),
            cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of blank barcodes')
        self.dataSet.save_figure(self, fig, 'blank_spatial_distribution')

    def _plot_matched_barcode_distribution(self):
        codebook = self.dataSet.get_codebook()
        bc = self.filterTask.get_barcode_database().get_barcodes()
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        codingIDs = codebook[~codebook['name'].str.contains('Blank')]
        codingBC = bc[bc['barcode_id'].isin(codingIDs.index)]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(codingBC['global_x'], codingBC['global_y'],
            bins=(np.ceil(maxX-minX)/5, np.ceil(maxY-minY)/5),
            cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of identified barcodes')
        self.dataSet.save_figure(self, fig, 'barcode_spatial_distribution')

    def _plot_cell_segmentation(self):
        cellBoundaries = self.segmentTask.get_cell_boundaries()

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        def plot_cell_boundary(boundary):
            ax.plot([x[0] for x in boundary], [x[1] for x in boundary])
        cellPlots = [plot_cell_boundary(b) for b in cellBoundaries]

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Cell boundaries')
        self.dataSet.save_figure(self, fig, 'cell_boundaries')

    def _plot_barcode_abundances(self):
        pass
    
    def run_analysis(self):
        self._plot_barcode_intensity_distribution()
        self._plot_barcode_area_distribution()
        self._plot_barcode_distance_distribution()
        self._plot_barcode_intensity_area_violin()
        self._plot_blank_distribution()
        self._plot_matched_barcode_distribution()
        self._plot_cell_segmentation()
        #TODO - barcode correlation plots
        #TODO - alignment error plots - need to save transformation information
        # first
        #TODO - barcode size spatial distribution
        #TODO - barcode distance spatial distribution
        #TODO - barcode intensity spatial distribution
        #TODO - abundance per barcode with blanks
        #TODO - confidence ratio per barcode with blanks
        #TODO - optimization convergence
        #TODO - good barcodes/blanks per cell