import os
import pytest
import shutil
from merlin.core import analysistask
from merlin.core import dataset
import merlin

merlin.DATA_HOME = os.sep.join(['.', 'test_data'])
merlin.ANALYSIS_HOME = os.sep.join(['.', 'test_analysis'])
dataDirectory = os.sep.join([merlin.DATA_HOME, 'test'])


class SimpleAnalysisTask(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def run_analysis(self):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

class SimpleParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def run_analysis(self, fragmentIndex):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def fragment_count(self):
        return 5

class SimpleInternallyParallelAnalysisTask(
        analysistask.InternallyParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def run_analysis(self):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []


@pytest.fixture(scope='session')
def simple_data():
    if os.path.exists(merlin.DATA_HOME):
        shutil.rmtree(merlin.DATA_HOME)
    if os.path.exists(merlin.ANALYSIS_HOME):
        shutil.rmtree(merlin.ANALYSIS_HOME)
    os.mkdir(merlin.DATA_HOME)
    os.mkdir(merlin.ANALYSIS_HOME)
    os.mkdir(dataDirectory)
    
    testData = dataset.DataSet('test')
    yield testData

    shutil.rmtree(merlin.DATA_HOME)
    shutil.rmtree(merlin.ANALYSIS_HOME)

@pytest.fixture(scope='session', params=[SimpleAnalysisTask, \
        SimpleParallelAnalysisTask, \
        SimpleInternallyParallelAnalysisTask])
def simple_task(simple_data, request):
    task = request.param(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)
