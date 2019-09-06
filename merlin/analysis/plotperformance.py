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


class StreamingPlotPerformance(analysistask.AnalysisTask):

    """
    An analysis task that is meant to run in the background during decoding
    to make QC plots as the data becomes available. At present this doesnt
    seem like it needs to be as parallel as possible, though the different
    tasks could be broken off into separate jobs if desired. Would potentially
    have a little more overhead if done this way, and currently the tasks
    necessarily finish at different times, so probably not worth it unless
    something changes.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 2880

    def get_dependencies(self):
        return [self.parameters['decode_task'],
                self.parameters['filter_task'],
                self.parameters['optimize_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _return_finished_fovs(self, analysisTask, remainingFOVs: List):
        finishedBool = [self.dataSet.check_analysis_done(analysisTask,
                                                         fragmentIndex=x)
                        for x in remainingFOVs]
        if True in finishedBool:
            out = [remainingFOVs[x] for x in range(len(remainingFOVs))
                   if finishedBool[x] is True]
            return out
        else:
            return []

    def _sample_barcodeDB(self, fovs: List, barcodeDB):
        for fov in fovs:
            decodeData = barcodeDB.get_barcodes(fov=fov,
                                                columnList=['area',
                                                            'mean_intensity',
                                                            'min_distance'])
            if fov == fovs[0]:
                sampleData = decodeData.copy(deep=True)
            else:
                sampleData = pandas.concat([sampleData, decodeData], 0)

        areaBins = np.array(range(1, sampleData['area'].max() * 2))
        intensityBins = np.array([x / 100 for x in range(
            int(round(np.log10(sampleData['mean_intensity'].values).max() * 2,
                      2) * 100))])
        distBins = np.array([x / 100 for x in range(
            0, int(round(sampleData['min_distance'].max() * 1.5, 2) * 100))])

        toWrite = ['area_bins', 'intensity_bins', 'dist_bins']
        toWriteData = [areaBins, intensityBins, distBins]
        for i in range(len(toWrite)):
            self.dataSet.save_numpy_analysis_result(toWriteData[i], toWrite[i],
                                                    self.analysisName,
                                                    subdirectory='hist_bins')

    def _retrive_bins(self, resultName):
        return self.dataSet.load_numpy_analysis_result(resultName,
                                                       self.analysisName,
                                                       subdirectory='hist_bins')

    def _barcodeDB_data_metadata(self, fov: int,
                                 decodeTask: merlin.analysis.decode):
        decodeDB = decodeTask.get_barcode_database()

        areaBins = self._retrive_bins('area_bins')
        intBins = self._retrive_bins('intensity_bins')
        distBins = self._retrive_bins('dist_bins')

        decodeData = decodeDB.get_barcodes(fov=fov,
                                           columnList=['area',
                                                       'mean_intensity',
                                                       'min_distance',
                                                       'barcode_id'])
        areaHist = np.histogram(decodeData['area'],
                                bins=areaBins)[0]
        intHist = np.histogram(np.log10(decodeData['mean_intensity']),
                               bins=intBins)[0]
        distHist = np.histogram(decodeData['min_distance'],
                                bins=distBins)[0]
        bcCounts = decodeData.groupby('barcode_id').size()
        barcodeDF = pandas.DataFrame(data=bcCounts.values,
                                     index=bcCounts.index.values.tolist(),
                                     columns=['barcode_counts'])
        areaGroups = decodeData.groupby('area')
        current = []
        maxArea = 16
        for g in areaGroups:
            if g[0] < maxArea:
                k, df = g[0], g[1]
                temp = np.histogram(np.log10(df['mean_intensity']),
                                    bins=intBins)[0]
                current.append(pandas.DataFrame(data=temp,
                                                index=intBins[:-1],
                                                columns=[k]))
            areaIntensityDF = pandas.concat(current, 1)
            areaIntensityDF.reindex(range(1, maxArea), axis='columns')

        return [areaHist, intHist, distHist, barcodeDF, areaIntensityDF]

    def _decode_plots(self, areaHist,
                      intHist, distHist,
                      barcodeDF, areaIntensityDF):

        # Area vs intensity plot
        allIntByArea = []
        for i in areaIntensityDF.columns.values.tolist():
            currentIntensities = []
            [currentIntensities.extend([x] * y) for x, y in list(zip(
                areaIntensityDF[i].index, areaIntensityDF[i].values))]
            allIntByArea.append(currentIntensities)
        f, axs = plt.subplots(1, 1, figsize=(8, 4))
        plt.violinplot(allIntByArea, areaIntensityDF.columns.values.tolist(),
                       showextrema=False, showmedians=True)
        if not isinstance(
                self.filterTask, filterbarcodes.AdaptiveFilterBarcodes):
            plt.axvline(
                x=self.filterTask.parameters['area_threshold'] - 0.5,
                color='green', linestyle=':')
            plt.axhline(y=np.log10(
                self.filterTask.parameters['intensity_threshold']),
                color='green', linestyle=':')

        else:
            adaptiveThresholds = [a for a in
                                  self.filterTask.get_adaptive_thresholds()
                                  for _ in (0, 1)]
            adaptiveXCoords = [0.5] + [x for x in np.arange(
                1.5, len(adaptiveThresholds) / 2) for _ in (0, 1)
                                       ] + [len(adaptiveThresholds) / 2 + 0.5]
            plt.plot(adaptiveXCoords, adaptiveThresholds)

        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Mean intensity ($log_{10}$)')
        plt.title('Intensity distribution by barcode area')
        plt.xlim([0, 17])
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, f, 'barcode_intensity_area_violin')

    def _sample_filter(self, fovs: List, filterDB):
        for fov in fovs:
            filterData = filterDB.get_barcodes(fov=fov)
            if fov == fovs[0]:
                sampleData = filterData.copy(deep=True)
            else:
                sampleData = pandas.concat([sampleData, filterData], 0)
        bcDF = pandas.DataFrame(self.dataSet.get_codebook().get_barcodes())
        bitCount = self.dataSet.get_codebook().get_bit_count()
        onIntensities = [
            sampleData[sampleData['barcode_id'].isin(bcDF[bcDF[i] == 1].index)]
            ['intensity_%i' % i].tolist() for i in range(bitCount)]
        allOn = []
        maxOn = max([allOn.extend(x) for x in onIntensities])
        bitBins = np.array([x/100 for x in
                            range(0, int(round(maxOn * 2, 2) * 100))])

        toWrite = ['bit_bins']
        toWriteData = [bitBins]
        for i in range(len(toWrite)):
            self.dataSet.save_numpy_analysis_result(toWriteData[i],
                                                    toWrite[i],
                                                    self.analysisName,
                                                    subdirectory='hist_bins')

    def _barcode_position_hist(self, bcData):
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])
        minX, minY, maxX, maxY = alignTask.get_global_extent()
        xBins = list(range(minX, maxX, 5))
        yBins = list(range(minY, maxY, 5))
        histOut = np.histogram2d(bcData['global_x'], bcData['global_y'],
                                 bins=[xBins, yBins])
        histDF = pandas.DataFrame(data=histOut[0],
                                  index=histOut[1][:-1],
                                  columns=histOut[2][:-1])
        return histDF

    def _filter_barcodeDB_metadata(self,
                                   filterTask: merlin.analysis.filterbarcodes,
                                   fov: int):
        filterDB = filterTask.get_barcode_database()
        filterData = filterDB.get_barcodes(fov=fov)
        bcCounts = filterData.groupby('barcode_id').size()
        barcodeDF = pandas.DataFrame(data=bcCounts.values,
                                     index=bcCounts.index.values.tolist(),
                                     columns=['barcode_counts'])

        bcDF = pandas.DataFrame(self.dataSet.get_codebook().get_barcodes())
        bitCount = self.dataSet.get_codebook().get_bit_count()
        onIntensities = [
            filterData[filterData['barcode_id'].isin(bcDF[bcDF[i] == 1].index)]
            ['intensity_%i' % i].tolist() for i in
            range(bitCount)]
        offIntensities = [
            filterData[filterData['barcode_id'].isin(bcDF[bcDF[i] == 0].index)]
            ['intensity_%i' % i].tolist() for i in
            range(bitCount)]
        bitBins = self._retrive_bins('bit_bins')
        processedOnIntensities = [np.histogram(x, bins=bitBins)[0]
                                  for x in onIntensities]
        processedOffIntensities = [np.histogram(x, bins=bitBins)[0]
                                   for x in offIntensities]
        onIntensityDF = pandas.DataFrame(data=processedOnIntensities,
                                         index=bitBins[:-1],
                                         columns=list(range(bitCount)))
        offIntensityDF = pandas.DataFrame(data=processedOffIntensities,
                                          index=bitBins[:-1],
                                          columns=list(range(bitCount)))

        blankIDs = self.dataset.get_codebook().get_blank_indexes()
        blankBC = filterData[filterData['barcode_id'].isin(blankIDs)]
        matchedBC = filterData[~(filterData['barcode_id'].isin(blankIDs))]
        blankPositions = self._barcode_position_hist(blankBC)
        matchedPositions = self._barcode_position_hist(matchedBC)

        bitColors = self.dataSet.get_data_organization().data['color']
        bcSet = self.dataSet.get_codebook().get_barcodes()
        singleColorBarcodes = [i for i, b in enumerate(bcSet) if
                               bitColors[np.where(b)[0]].nunique() == 1]
        multiColorBarcodes = [i for i, b in enumerate(bcSet) if
                              bitColors[np.where(b)[0]].nunique() > 1]

        sBC = filterData[filterData['barcode_id'].isin(singleColorBarcodes)]
        mBC = filterData[filterData['barcode_id'].isin(multiColorBarcodes)]

        imageSize = self.dataSet.get_image_dimensions()
        width = imageSize[0]
        height = imageSize[1]

        def radial_distance(x, y):
            return np.sqrt((x - 0.5 * width) ** 2 + (y - 0.5 * height) ** 2)

        maxDist = radial_distance(width, height)
        bins = range(0, int(round(maxDist)), 5)
        sRD = [radial_distance(x, y) for x, y in zip(sBC['x'], sBC['y'])]
        mRD = [radial_distance(x, y) for x, y in zip(mBC['x'], mBC['y'])]
        countsS, binsS = np.histogram(sRD, bins=bins)
        countsM, binsM = np.histogram(mRD, bins=bins)

        return(barcodeDF, onIntensityDF, offIntensityDF,
               blankPositions, matchedPositions, countsS, countsM)

    def _filter_plots(self, filterBarcodeDF, filterOnIntensity,
                      filterOffIntensity, filterBlankPos, filterMatchedPos,
                      countsS, countsM):
        # barcode and blank count frequencies
        self._plot_barcode_abundances(filterBarcodeDF,
                                      'filtered_barcode_abundances')

        # bitwise intensity plot
        bitCount = len(filterOnIntensity.columns.values.tolist())
        f, axs = plt.subplots(1, 1, figsize=(bitCount / 2.5))
        onViolin = plt.violinplot(filterOnIntensity,
                                  np.arange(1, bitCount + 1) - 0.25,
                                  showextrema=False, showmedians=True,
                                  widths=0.35)
        offViolin = plt.violinplot(filterOffIntensity,
                                   np.arange(1, bitCount + 1) + 0.25,
                                   showextrema=False, showmedians=True,
                                   widths=0.35)
        plt.xticks(np.arange(1, bitCount + 1))
        plt.xlabel('Bit')
        plt.ylabel('Normalized intensity')
        plt.title('Bitwise intensity distributions')
        plt.legend([onViolin['bodies'][0], offViolin['bodies'][0]],
                   ['1', '0'])
        self.dataSet.save_figure(self, f, 'barcode_bitwise_intensity_violin')

        # blank barcode positions
        f, axs = plt.subplots(1, 1, figsize=(10, 10))
        toPlot = filterBlankPos.unstack().reset_index()
        toPlot = toPlot[toPlot[0] > 0]
        plt.scatter(toPlot['level_1'], toPlot['level_0'], c=toPlot[0],
                    cmap='Greys', vmax=toPlot[0].quantile(q=0.95))
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of blank barcodes')
        self.dataSet.save_figure(self, f, 'blank_spatial_distribution')

        # matched barcode positions
        f, axs = plt.subplots(1, 1, figsize=(10, 10))
        toPlot = filterMatchedPos.unstack().reset_index()
        toPlot = toPlot[toPlot[0] > 0]
        plt.scatter(toPlot['level_1'], toPlot['level_0'], c=toPlot[0],
                    cmap='Greys', vmax=toPlot[0].quantile(q=0.95))
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of barcodes')
        self.dataSet.save_figure(self, f, 'barcode_spatial_distribution')

        # FPKM plot
        if 'fpkm_file' in self.parameters:
            fpkmPath = os.sep.join([merlin.FPKM_HOME,
                                    self.parameters['fpkm_file']])
            fpkm = pandas.read_csv(fpkmPath, index_col='name')
            bcIDs = filterBarcodeDF.index.values.tolist()
            cb = self.dataSet.get_codebook()
            genes = [cb.get_name_for_barcode_index(x) for x in bcIDs]
            bcDF = filterBarcodeDF.copy(deep=True)
            bcDF.index = genes
            merged = bcDF.merge(fpkm.loc[:, ['FPKM']])
            f, axs = plt.subplots(1, 1, figsize=(4, 4))
            plt.loglog(merged['FPKM'], merged['barcode_counts'], '.', alpha=0.5)
            plt.ylabel('Detected counts')
            plt.xlabel('FPKM')
            correlation = np.corrcoef(
                np.log(merged['FPKM'] + 1),
                np.log(merged['barcode_counts'] + 1))
            plt.title('%s (r=%0.2f)' % (self.parameters['fpkm_file'],
                                        correlation[0, 1]))
            self.dataSet.save_figure(self, f, 'fpkm_correlation')

    def _optimization_plots(self, optimizeTask):
        # scale factors
        f, axs = plt.subplots(1, 1, figsize=(5, 5))
        seaborn.heatmap(optimizeTask.get_scale_factor_history())
        plt.xlabel('Bit index')
        plt.ylabel('Iteration number')
        plt.title('Scale factor optimization history')
        self.dataSet.save_figure(self, f, 'optimization_scale_factors')

        # optimization barcode counts
        f, axs = plt.subplots(1, 1, figsize=(5, 5))
        seaborn.heatmap(optimizeTask.get_barcode_count_history())
        plt.xlabel('Barcode index')
        plt.ylabel('Iteration number')
        plt.title('Barcode counts optimization history')
        self.dataSet.save_figure(self, f, 'optimization_barcode_counts')

    def _run_analysis(self):
        doneDict = {'decode_task': False, 'filter_task': False,
                    'optimize_task': False, 'segment_task': False}
        remainingDecodeFOVs = self.dataSet.get_fovs()
        remainingFilterFOVs = self.dataSet.get_fovs()
        remainingSegmentFOVs = self.dataSet.get_fovs()
        availableDecodeFOVs, availableFilterFOVs, availableSegment = [], [], []
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        optimizeTask = self.dataSet.load_analysis_task(
            self.parameters['optimize_task'])
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        decodeSamplingDone = False
        filterSamplingDone = False
        decodeDB = decodeTask.get_barcode_database()
        filterDB = filterTask.get_barcode_database()
        while False in list(doneDict.values()):
            if doneDict['decode_task'] is False:
                availableFOVs = self._return_finished_fovs(decodeTask,
                                                           remainingDecodeFOVs)
                if len(availableFOVs) > 0:
                    availableDecodeFOVs.extend(availableFOVs)
                if len(availableDecodeFOVs) > 20:
                    if decodeSamplingDone is False:
                        sampledFOVs = sample(availableDecodeFOVs, 5)
                        self._sample_barcodeDB(sampledFOVs, decodeDB)
                        decodeSamplingDone = True
                    if decodeSamplingDone is True:
                        for fov in availableDecodeFOVs:
                            data = self._barcodeDB_data_metadata(fov,
                                                                 decodeTask)
                            if fov == availableDecodeFOVs[0]:
                                decodeAreaHist = data[0].copy()
                                decodeIntHist = data[1].copy()
                                decodeDistHist = data[2].copy()
                                decodeBarcodeDF = data[3].copy(deep=True)
                                decodeAIDF = data[4].copy(deep=True)
                            else:
                                decodeAreaHist = decodeAreaHist + data[0]
                                decodeIntHist = decodeIntHist + data[1]
                                decodeDistHist = decodeDistHist + data[2]
                                decodeBarcodeDF = decodeBarcodeDF + data[3]
                                decodeAIDF = decodeAIDF + data[4]
                            remainingDecodeFOVs.remove(fov)
                if len(remainingDecodeFOVs) == 0:
                    self._decode_plots(decodeAreaHist, decodeIntHist,
                                       decodeDistHist, decodeBarcodeDF,
                                       decodeAIDF)
                    doneDict['decode_task'] = True
            if doneDict['filter_task'] is False:
                availableFOVs = self._return_finished_fovs(filterTask,
                                                           remainingFilterFOVs)
                if len(availableFOVs) > 0:
                    availableFilterFOVs.extend(availableFOVs)
                if len(availableFilterFOVs) > 20:
                    if filterSamplingDone is False:
                        sampledFOVs = sample(availableFilterFOVs, 5)
                        self._sample_filter(sampledFOVs, filterDB)
                        filterSamplingDone = True
                    if filterSamplingDone is True:
                        for fov in availableFilterFOVs:
                            data = self._filter_barcodeDB_metadata(fov,
                                                                   filterTask)
                            if fov == availableFilterFOVs[0]:
                                filterBarcodeDF = data[0].copy(deep=True)
                                filterOnIntensity = data[1].copy(deep=True)
                                filterOffIntensity = data[2].copy(deep=True)
                                filterBlankPos = data[3].copy(deep=True)
                                filterMatchedPos = data[4].copy(deep=True)
                                filterSingleColorCnts = data[5].copy()
                                filterMultiColorCnts = data[6].copy()
                            else:
                                filterBarcodeDF = filterBarcodeDF + data[0]
                                filterOnIntensity = filterOnIntensity + data[1]
                                filterOffIntensity = filterOffIntensity +\
                                                     data[2]
                                filterBlankPos = filterBlankPos + data[3]
                                filterMatchedPos = filterMatchedPos + data[4]
                                filterSingleColorCnts = filterSingleColorCnts +\
                                                        data[5]
                                filterMultiColorCnts = filterMultiColorCnts +\
                                                       data[6]
                            remainingFilterFOVs.remove(fov)
                if len(remainingFilterFOVs) == 0:
                    self._filter_plots(filterBarcodeDF, filterOnIntensity,
                                       filterOffIntensity, filterBlankPos,
                                       filterMatchedPos, filterSingleColorCnts,
                                       filterMultiColorCnts)
                    doneDict['filter_task'] = True
            if doneDict['optimize_task'] is False:
                if optimizeTask.is_complete():
                    self._optimization_plots()
                    doneDict['optimize_task'] = True
            if doneDict['segment_task'] is False:
                if segmentTask.is_complete():
                    f, axs = plt.subplots(1, 1, figsize=(20,  20))
                    for fov in remainingSegmentFOVs:
                        self._segmentation_plots(fov, segmentTask)
                    plt.xlabel('X position (microns)')
                    plt.ylabel('Y position (microns)')
                    plt.title('Cell boundaries')
                    self.dataSet.save_figure(self, f, 'cell_boundaries')
                    doneDict['segment_task'] = True
            time.sleep(300)


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
        plotEngine = plots.PlotEngine(self, taskDict)
        while not plotEngine.take_step():
            time.sleep(60)


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
