from merlin.core import analysistask
import pandas
from typing import Dict

class AggregateData(analysistask.AnalysisTask):
    """
    A metaanalysis task that aggregates data from multiple datasets
    """
    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        if 'overwrite' not in self.parameters:
            self.parameters['overwrite'] = False

        self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        dep = [v for k, v in self.parameters.items() if 'task' in k]
        if len(dep) > 1:
            print('This task can only aggregate one analysis task')
        else:
            return dep

    def load_aggregated_data(self, **kwargs):
        return self.metaDataSet.load_dataframe_from_csv('aggregated_data', self, **kwargs)

    def _run_analysis(self):
        allAnalyses = []
        dep = self.get_dependencies()
        dsDict = self.metaDataSet.load_datasets()
        if None in dsDict.values():
            raise FileNotFoundError
        else:
            for ds in self.metaDataSet.dataSetNames:
                tempData = dsDict[ds].load_analysis_task(dep).return_exported_data()
                tempData['dataset'] = d.dataSet.dataSetName
                allAnalyses.append(tempData)
        if len(allAnalyses) == len(self.metaDataSet.dataSets):
            combinedAnalysis = pandas.concat(allAnalyses, 0)
            self.metaDataSet.save_dataframe_to_csv(combinedAnalysis, 'aggregated_data', self)
