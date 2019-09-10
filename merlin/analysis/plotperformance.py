import os
from matplotlib import pyplot as plt
import pandas
import merlin
import seaborn
import numpy as np
from typing import List
from merlin.core import analysistask
from merlin.analysis import filterbarcodes
from random import sample
import time

from merlin import plots
plt.style.use(
        os.sep.join([os.path.dirname(merlin.__file__),
                     'ext', 'default.mplstyle']))


class PlotPerformance(analysistask.AnalysisTask):

    """
    An analysis task that generates plots depicting metrics of the MERFISH
    decoding.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'exclude_plots' in self.parameters:
            self.parameters['exclude_plots'] = []

        self.taskTypes = ['decode_task', 'filter_task', 'optimize_task',
                          'segment_task', 'sum_task', 'partition_task',
                          'global_align_task']

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 180

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        taskDict = {t: self.dataSet.load_analysis_task(self.parameters[t])
                    for t in self.taskTypes if t in self.parameters}
        print('Creating plot engine')
        plotEngine = plots.PlotEngine(self, taskDict)
        print('Plot engine created')
        print(plotEngine.take_step())
        while not plotEngine.take_step():
            pass


class OldPlotPerformance(analysistask.AnalysisTask):

    """
    An analysis task that generates plots depicting metrics of the MERFISH
    decoding.
    """

    # TODO all the plotting should be refactored. I do not like the way
    # this class is structured as a long list of plotting functions. It would
    # be more convenient if each plot could track it's dependent tasks and
    # be executed once those tasks are complete.

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # TODO - move this definition to run_analysis()
        self.optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        self.decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        self.filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])
        if 'segment_task' in self.parameters:
            self.segmentTask = self.dataSet.load_analysis_task(
                    self.parameters['segment_task'])
        else:
            self.segmentTask = None

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 180

    def get_dependencies(self):
        return [self.parameters['decode_task'], self.parameters['filter_task']]

    def _plot_fpkm_correlation(self):
        fpkmPath = os.sep.join([merlin.FPKM_HOME, self.parameters['fpkm_file']])
        fpkm = pandas.read_csv(fpkmPath, index_col='name')
        barcodes = self.filterTask.get_barcode_database().get_barcodes()
        codebook = self.filterTask.get_codebook()

        barcodeIndexes = codebook.get_coding_indexes()
        barcodeCounts = np.array(
            [np.sum(barcodes['barcode_id'] == i) for i in barcodeIndexes])
        fpkmCounts = np.array(
            [fpkm.loc[codebook.get_name_for_barcode_index(i)]['FPKM'] for
             i in barcodeIndexes])

        fig = plt.figure(figsize=(4, 4))
        plt.loglog(fpkmCounts, barcodeCounts, '.', alpha=0.5)
        plt.ylabel('Detected counts')
        plt.xlabel('FPKM')
        correlation = np.corrcoef(
            np.log(fpkmCounts + 1), np.log(barcodeCounts + 1))
        plt.title('%s (r=%0.2f)' % (self.parameters['fpkm_file'],
                                    correlation[0, 1]))
        self.dataSet.save_figure(self, fig, 'fpkm_correlation')

    def _plot_barcode_intensity_area_violin(self):
        barcodeDB = self.decodeTask.get_barcode_database()
        intensityData = [np.log10(
            barcodeDB.get_intensities_for_barcodes_with_area(x).tolist())
                    for x in range(1, 15)]
        nonzeroIntensities = [x for x in intensityData if len(x) > 0]
        nonzeroIndexes = [i+1 for i, x in enumerate(intensityData)
                          if len(x) > 0]
        fig = plt.figure(figsize=(8, 4))
        plt.violinplot(nonzeroIntensities, nonzeroIndexes, showextrema=False,
                       showmedians=True)
        if not isinstance(
                self.filterTask, filterbarcodes.AdaptiveFilterBarcodes):
            plt.axvline(x=self.filterTask.parameters['area_threshold']-0.5,
                        color='green', linestyle=':')
            plt.axhline(y=np.log10(
                self.filterTask.parameters['intensity_threshold']),
                    color='green', linestyle=':')
        else:
            adaptiveThresholds = [a for a in
                                  self.filterTask.get_adaptive_thresholds()
                                  for _ in (0, 1)]
            adaptiveXCoords = [0.5] + [x for x in np.arange(
                1.5, len(adaptiveThresholds)/2) for _ in (0, 1)] \
                + [len(adaptiveThresholds)/2+0.5]
            plt.plot(adaptiveXCoords, adaptiveThresholds)

        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Mean intensity ($log_{10}$)')
        plt.title('Intensity distribution by barcode area')
        plt.xlim([0, 15])
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_intensity_area_violin')

    def _plot_bitwise_intensity_violin(self):
        bcDF = pandas.DataFrame(self.filterTask.get_codebook().get_barcodes())

        bc = self.filterTask.get_barcode_database().get_barcodes()
        bitCount = self.filterTask.get_codebook().get_bit_count()
        onIntensities = [bc[bc['barcode_id'].isin(bcDF[bcDF[i] == 1].index)]
                         ['intensity_%i' % i].tolist() for i in range(bitCount)]
        offIntensities = [bc[bc['barcode_id'].isin(bcDF[bcDF[i] == 0].index)]
                          ['intensity_%i' % i].tolist() for i in
                          range(bitCount)]
        fig = plt.figure(figsize=(bitCount / 2, 5))
        onViolin = plt.violinplot(onIntensities,
                                  np.arange(1, bitCount + 1) - 0.25,
                                  showextrema=False, showmedians=True,
                                  widths=0.35)
        offViolin = plt.violinplot(offIntensities,
                                   np.arange(1, bitCount + 1) + 0.25,
                                   showextrema=False, showmedians=True,
                                   widths=0.35)
        plt.xticks(np.arange(1, bitCount + 1))
        plt.xlabel('Bit')
        plt.ylabel('Normalized intensity')
        plt.title('Bitwise intensity distributions')
        plt.legend([onViolin['bodies'][0], offViolin['bodies'][0]], ['1', '0'])

        self.dataSet.save_figure(self, fig, 'barcode_bitwise_intensity_violin')

    def _run_analysis(self):
        if 'fpkm_file' in self.parameters:
            self._plot_fpkm_correlation()
        self._plot_bitwise_intensity_violin()
        self._plot_barcode_intensity_area_violin()
        # TODO _ analysis run times
        # TODO - barcode correlation plots
        # TODO - alignment error plots - need to save transformation information
        # first
        # TODO - barcode size spatial distribution global and FOV average
        # TODO - barcode distance spatial distribution global and FOV average
        # TODO - barcode intensity spatial distribution global and FOV average
        # TODO - good barcodes/blanks per cell
