from matplotlib import pyplot as plt
plt.style.use('./ext/default.mplstyle')
import numpy as np

from merfish_code.core import analysistask

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

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

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
        barcodeDB = self.decodeTask.get_baroced_database()
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

    def run_analysis(self):
        self._plot_barcode_intensity_distribution()
        self._plot_barcode_area_distribution()
        self._plot_barcode_distance_distribution()
        self._plot_barcode_intensity_area_violin()
        #TODO - barcode correlation plots
        #TODO - alignment error plots - need to save transformation information
        # first
        #TODO - good barcode and blank spatial distributions
        #TODO - barcode size spatial distribution
        #TODO - barcode distance spatial distribution
        #TODO - barcode intensity spatial distribution
        #TODO - abundance per barcode with blanks
        #TODO - confidence ratio per barcode with blanks
        #TODO - optimization convergence
        #TODO - good barcodes/blanks per cell
        pass
