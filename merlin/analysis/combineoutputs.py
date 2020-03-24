import numpy as np
import pandas as pd
from merlin.core import dataset
from merlin.util import spatialfeature
from merlin.core import analysistask
from merlin.analysis.partition import ExportPartitionedBarcodes
from merlin.analysis.segment import ExportCellMetadata
from merlin.analysis.sequential import ExportSumSignals
from merlin.analysis.globalalign import GlobalAlignment

class CombineOutputs(analysistask.AnalysisTask):
    """
    An analysis task to combine the outputs of various export tasks into
    a single file, using the output of the segment export task to align all
    outputs in final file
    """

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [v['analysisName'] for k, v in self.parameters.items() if 'task' in k]

    def _flag_ignored_task_parameters(self, allowedParameters, taskParameters):
        ignoredParams = [x for x in list(taskParameters.keys()) if x not in allowedParameters]
        print('The following parameters were ignored for {}:\n {}'.format(taskParameters['analysisName'],
                                                                          '\n'.join(ignoredParams)))
        print('The allowed parameters for this task are:\n {}'.format('\n'.join(allowedParameters)))

    def _process_segmentation_data(self, tasks, taskParameters):
        allowedParameters = ['analysisName', 'minimum_volume', 'maximum_volume']
        minVolume = 0
        maxVolume = np.inf
        cellExportTasks = self._select_requested_task_type(tasks, ExportCellMetadata)
        cellData = []
        for cellExport in cellExportTasks:
            tempData = cellExport.return_exported_data()
            currentTaskParams = taskParameters[cellExport.analysisName]
            self._flag_ignored_task_parameters(allowedParameters, currentTaskParams)

            alignmentTask = self._select_requested_task_type(tasks, GlobalAlignment)[0]

            groups = tempData.groupby('fov')
            pos = []
            for k, v in groups:
                pos.extend(alignmentTask.global_coordinates_to_fov(k, list(
                    zip(v['center_x'].values.tolist(), v['center_y'].values.tolist()))))
            tempData['fov_x'] = np.array(pos)[:, 0]
            tempData['fov_y'] = np.array(pos)[:, 1]
            tempData = tempData.loc[:, ['fov', 'volume', 'fov_x', 'fov_y', 'center_x', 'center_y']].copy(deep=True)
            tempData.columns = ['fov', 'volume', 'fov_x', 'fov_y', 'global_x', 'global_y']
            if 'minimum_volume' in currentTaskParams:
                minVolume = float(currentTaskParams['minimum_volume'])
            if 'maximum_volume' in currentTaskParams:
                maxVolume = float(currentTaskParams['maximum_volume'])
            cellData.append(tempData[(tempData['volume'] > minVolume) &
                                     (tempData['volume'] < maxVolume)])
        cellData = pd.concat(cellData, axis = 0)
        return cellData

    def _return_barcodes_and_blanks(self):
        codebooks = self.dataSet.get_codebooks()
        genes = []
        blanks = []
        for codebook in codebooks:
            allEntries = codebook.get_data()['name'].values.tolist()
            codingEntries = codebook.get_gene_names()
            blankEntries = [x for x in allEntries if x not in codingEntries]
            genesToAdd = [x for x in codingEntries if x not in genes]
            blanksToAdd = [x for x in blankEntries if x not in blanks]
            genes.extend(genesToAdd)
            blanks.extend(blanksToAdd)
        return genes, blanks

    def _process_barcode_data(self, tasks, taskParameters, cellData):
        """

        Args:
            tasks:
            taskParameters:
            cellData:
        Returns:
            barcodeData:
            totalCount:
            medianTotalDensity:
            targetDensity:
        """
        allowedParameters = ['analysisName','report_counts','log_x_plus_1','volume_normalize','target_median_density']
        barcodeExportTasks = self._select_requested_task_type(tasks, ExportPartitionedBarcodes)
        genes, blanks = self._return_barcodes_and_blanks()

        barcodeData = dict()
        countHolding = []
        getCount = False
        normalizeDensity = False
        logNorm = False
        totalCount = None
        medianTotalDensity = None
        targetDensity = None
        for barcodeExport in partitionedBarcodeExportTasks:
            tempData = barcodeExport.return_exported_data()
            currentTaskParams = taskParameters[barcodeExport.analysisName]
            self._flag_ignored_task_parameters(allowedParameters, currentTaskParams)
            includedGenes = [x for x in tempData.columns.values.tolist() if x in genes]
            if 'report_counts' in currentTaskParams:
                getCount = True
                geneCounts = pd.DataFrame(tempData.loc[:,includedGenes].sum(1))
                countHolding.append(geneCounts.copy(deep=True))
            if 'volume_normalize' in currentTaskParams:
                commonCells = tempData.index.intersection(cellData.index).values.tolist()
                tempData = tempData.loc[commonCells, includedGenes].div(cellData.loc[commonCells,'volume'], 0)
            if 'target_median_density' in currentTaskParams:
                normalizeDensity = True
            if str(currentTaskParams['log_x_plus_1']).upper() == 'TRUE':
                logNorm = True
            barcodeData[barcodeExport.analysisName] = tempData.copy(deep=True)
        if getCount:
            if len(countHolding) > 1:
                currentCount = countHolding[0]
                for count in countHolding[1:]:
                    currentCount.merge(count, left_index = True, right_index = True)
                totalCount = currentCount.sum(1)
            elif len(countHolding) == 1:
                totalCount = currentCount[0]
        if normalizeDensity:
            commonCells = totalCount.index.intersection(cellData.index).values.tolist()
            medianTotalDensity = totalCount.loc[commonCells,:].median()
            medianNorm = dict()
            noMedianNorm = dict()
            for k,v in taskParameters.items():
                if 'target_median_density' in v:
                    targetDensity = float(v['target_median_density'])
                    medianNorm[k] = barcodeData[k]
                else:
                    noMedianNorm[k] = barcodeData[k]
            scalingFactor = targetDensity/medianTotalDensity
            postMedianDict = dict()
            if len(medianNorm) > 0:
                for k,v in medianNorm.items():
                    postMedianDict[k] = v * scalingFactor
            if len(noMedianNorm) > 0:
                for k,v in noMedianNorm.items():
                    postMedianDict[k] = v
            barcodeData = postMedianDict
        if logNorm:
            logNorm = dict()
            noLogNorm = dict()
            for k, v in taskParameters.items():
                if str(v['log_x_plus_1']).upper() == 'TRUE':
                    logNorm[k] = barcodeData[k].apply(lambda x: np.log10(x + 1))
                else:
                    noLogNorm[k] = barcodeData[k]
        return barcodeData, totalCount, medianTotalDensity, targetDensity

    def _regression_normalize(self):



    def _subtraction_normalize(self):



    def _process_sequential_data(self, tasks):

    def _load_requested_tasks(self):
        tasks = []
        taskParameters = dict()
        dependenciesAndParams = [v for k, v in self.parameters.items() if 'task' in k]
        for dependency in dependenciesAndParams:
            tasks.append(self.dataSet.load_analysis_task(dependency['analysisName']))
            taskParameters[dependency['analysisName']] = dependency
        return tasks, taskParameters

    def _select_requested_task_type(self, tasks: List, taskType: analysistask.AnalysisTask) -> List:
        selectedTasks = []
        for t in tasks:
            if isintance(t, taskType):
                selectedTasks.append(t)
        return selectedTasks

    def _run_analysis(self):
        tasks, taskParameters = self._load_requested_tasks()
        cellData = self._process_segmentation_data(tasks)





















    # TODO would this be easier if volume normalize, calculate counts, and
    # log_x_plus_1 were parameters specific to each task? could set this up
    # in the parameters up front with task: {name: x, param1: ...}

    """
    An analysis task to combine the outputs of various export tasks into
    a single file, using the output of the segment export task to align all
    outputs in final file
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # ensure segment_export_task is specified
        segmentExportTask = self.parameters['segment_export_task']

        if 'volume_normalize' not in self.parameters:
            self.parameters['volume_normalize'] = False
        if 'calculate_counts' not in self.parameters:
            self.parameters['calculate_counts'] = False
        if 'log_x_plus_1' not in self.parameters:
            self.parameters['log_x_plus_1'] = False

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [v for k, v in self.parameters.items() if 'task' in k]

    def _add_fov_pos(self, cellData: pd.DataFrame):
        groups = cellData.groupby('fov')
        pos = []
        for k, v in groups:
            pos.extend(alignmentTask.global_coordinates_to_fov(k, list(
                zip(v['center_x'].values.tolist(),
                    v['center_y'].values.tolist()))))
        cellData['fov_x'] = np.array(pos)[:, 0]
        cellData['fov_y'] = np.array(pos)[:, 1]
        return cellData

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'combined_output', analysisTask=self.analysisName, resultIndex=None,
            subdirectory=None, **kwargs)

    def _run_analysis(self):
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_export_task'])
        cellData = segmentTask.return_exported_data()

        remainingTasks = self.get_dependencies()
        remainingTasks = [x for x in remainingTasks if
                          x != self.parameters['segment_export_task']]
        allData = []
        for t in remainingTasks:
            loadedTask = self.dataSet.load_analysis_task(t)
            currentData = loadedTask.return_exported_data()
            allData.append(currentData)
        cbs = self.dataSet.get_codebooks()
        allGenes = []
        for cb in cbs:
            allGenes.extend(cb.get_gene_names())

        if 'slice_info' in self.parameters:
            sliceDict = dict()
            for i in range(len(slices)):
                sliceNum = slices.loc[i, 'Slice']
                fovStart = slices.loc[i, 'FOV start']
                fovStop = slices.loc[i, 'FOV stop']
                for j in range(fovStart - 1, fovStop):
                    sliceDict[j] = sliceNum

            cellData['slice_id'] = cellData['fov'].map(sliceDict)
        cellDataColumns = cellData.columns.values.tolist()

        arrangedColumns = []
        for d in allData:
            cellData = cellData.merge(d, left_index=True, right_index=True)
            arrangedColumns.extend(d.columns.values.tolist())

        if self.parameters['calculate_counts']:
            counts = cellData.loc[:, allGenes].sum(1)

        if self.parameters['volume_normalize']:
            cellData.loc[:, allGenes] = \
                cellData.loc[:, allGenes].div(cellData['volume'], 0)

        if self.parameters['log_x_plus_1']:
            cellData.loc[:, arrangedColumns] = cellData.loc[
                                              :, arrangedColumns].apply(
                lambda x: np.log10(x+1))
        cellData['counts'] = counts

        orderedCol = cellDataColumns + ['counts'] + arrangedColumns
        cellData = cellData.loc[:, orderedCol]
        self.dataSet.save_dataframe_to_csv(cellData, 'combined_output',
                                           self.get_analysis_name())
