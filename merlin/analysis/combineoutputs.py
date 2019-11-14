import numpy as np
import pandas as pd
from merlin.core import dataset
from merlin.util import spatialfeature
from merlin.core import analysistask


class CombineOutputs(analysistask.AnalysisTask):
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
            counts = cellData.loc[:,allGenes].sum(1)

        if self.parameters['volume_normalize']:
            cellData.loc[:, allGenes] = \
                cellData.loc[:, allGenes].div(cellData['volume'], 0)

        if self.parameters['log_x_plus_1']:
            cellData.loc[:,arrangedColumns] = cellData.loc[
                                              :,arrangedColumns].apply(
                lambda x: np.log10(x+1))
        cellData['counts'] = counts


        orderedCol = cellDataColumns + ['counts'] + arrangedColumns
        cellData = cellData.loc[:,orderedCol]
        self.dataSet.save_dataframe_to_csv(cellData, 'combined_output',
                                           self.get_analysis_name())
