import os
import pytest
import shutil
import glob
from merlin.core import analysistask
from merlin.core import dataset
import merlin

merlin.DATA_HOME = os.sep.join(['.', 'test_data'])
merlin.ANALYSIS_HOME = os.sep.join(['.', 'test_analysis'])
merlin.ANALYSIS_PARAMETERS_HOME = os.sep.join(
        ['.', 'test_analysis_parameters'])
merlin.CODEBOOK_HOME = os.sep.join(['.', 'test_codebooks'])
merlin.DATA_ORGANIZATION_HOME = os.sep.join(['.', 'test_dataorganization'])
merlin.POSITION_HOME = os.sep.join(['.', 'test_poitions'])
merlin.MICROSCOPE_PARAMETERS_HOME = os.sep.join(
        ['.', 'test_microcope_parameters'])


dataDirectory = os.sep.join([merlin.DATA_HOME, 'test'])
merfishDataDirectory = os.sep.join([merlin.DATA_HOME, 'merfish_test'])

class SimpleAnalysisTask(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _run_analysis(self):
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

    def _run_analysis(self, fragmentIndex):
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

    def _run_analysis(self):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

@pytest.fixture(scope='session')
def base_files():
    folderList = [merlin.DATA_HOME, merlin.ANALYSIS_HOME, \
            merlin.ANALYSIS_PARAMETERS_HOME, merlin.CODEBOOK_HOME, \
            merlin.DATA_ORGANIZATION_HOME, merlin.POSITION_HOME, \
            merlin.MICROSCOPE_PARAMETERS_HOME]
    for folder in folderList:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    shutil.copyfile(
        os.sep.join(
            ['.', 'auxiliary_files', 'test_data_organization.csv']),
        os.sep.join(
            [merlin.DATA_ORGANIZATION_HOME, 'test_data_organization.csv']))
    shutil.copyfile(
        os.sep.join(
            ['.', 'auxiliary_files', 'test_codebook.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook.csv']))
    shutil.copyfile(
        os.sep.join(
            ['.', 'auxiliary_files', 'test_positions.csv']),
        os.sep.join(
            [merlin.POSITION_HOME, 'test_positions.csv']))
    shutil.copyfile(
        os.sep.join(
            ['.', 'auxiliary_files', 'test_analysis_parameters.json']),
        os.sep.join(
            [merlin.ANALYSIS_PARAMETERS_HOME, 'test_analysis_parameters.json']))

    yield

    for folder in folderList:
        shutil.rmtree(folder)


@pytest.fixture(scope='session')
def simple_data(base_files):
    os.mkdir(dataDirectory)
    testData = dataset.DataSet('test')

    yield testData

    shutil.rmtree(dataDirectory)


@pytest.fixture(scope='session')
def simple_merfish_data(base_files):
    os.mkdir(merfishDataDirectory)

    for imageFile in glob.iglob(
            os.sep.join(['.', 'auxiliary_files', '*.tif'])):
        if os.path.isfile(imageFile):
            shutil.copy(imageFile, merfishDataDirectory)

    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test', 
            dataOrganizationName='test_data_organization.csv',
            codebookName='test_codebook.csv',
            positionFileName='test_positions.csv')
    yield testMERFISHData

    shutil.rmtree(merfishDataDirectory)


@pytest.fixture(scope='function')
def single_task(simple_data):
    task = SimpleAnalysisTask(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(scope='function', params=[SimpleAnalysisTask,
                                          SimpleParallelAnalysisTask,
                                          SimpleInternallyParallelAnalysisTask])
def simple_task(simple_data, request):
    task = request.param(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)



