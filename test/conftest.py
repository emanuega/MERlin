import os
import pytest
import shutil
import glob
from merlin.core import dataset
from merlin.analysis import testtask
import merlin
import numpy as np
import pandas as pd


root = os.path.join(os.path.dirname(merlin.__file__), '..', 'test')
merlin.DATA_HOME = os.path.abspath('test_data')
merlin.ANALYSIS_HOME = os.path.abspath('test_analysis')
merlin.ANALYSIS_PARAMETERS_HOME = os.path.abspath('test_analysis_parameters')
merlin.CODEBOOK_HOME = os.path.abspath('test_codebooks')
merlin.DATA_ORGANIZATION_HOME = os.path.abspath('test_dataorganization')
merlin.POSITION_HOME = os.path.abspath('test_positions')
merlin.MICROSCOPE_PARAMETERS_HOME = os.path.abspath('test_microcope_parameters')


dataDirectory = os.sep.join([merlin.DATA_HOME, 'test'])
merfishDataDirectory = os.sep.join([merlin.DATA_HOME, 'merfish_test'])


@pytest.fixture(scope='session')
def base_files():
    folderList = [merlin.DATA_HOME, merlin.ANALYSIS_HOME,
                  merlin.ANALYSIS_PARAMETERS_HOME, merlin.CODEBOOK_HOME,
                  merlin.DATA_ORGANIZATION_HOME, merlin.POSITION_HOME,
                  merlin.MICROSCOPE_PARAMETERS_HOME]
    for folder in folderList:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_data_organization.csv']),
        os.sep.join(
            [merlin.DATA_ORGANIZATION_HOME, 'test_data_organization.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_codebook.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_codebook2.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook2.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_positions.csv']),
        os.sep.join(
            [merlin.POSITION_HOME, 'test_positions.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_analysis_parameters.json']),
        os.sep.join(
            [merlin.ANALYSIS_PARAMETERS_HOME, 'test_analysis_parameters.json']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_microscope_parameters.json']),
        os.sep.join(
            [merlin.MICROSCOPE_PARAMETERS_HOME,
             'test_microscope_parameters.json']))

    yield

    for folder in folderList:
        shutil.rmtree(folder)


@pytest.fixture(scope='session')
def merfish_files(base_files):
    os.mkdir(merfishDataDirectory)

    for imageFile in glob.iglob(
            os.sep.join([root, 'auxiliary_files', '*.tif'])):
        if os.path.isfile(imageFile):
            shutil.copy(imageFile, merfishDataDirectory)

    yield

    shutil.rmtree(merfishDataDirectory)


@pytest.fixture(scope='session')
def simple_data(base_files):
    os.mkdir(dataDirectory)
    testData = dataset.DataSet('test')

    yield testData

    shutil.rmtree(dataDirectory)


@pytest.fixture(scope='session')
def simple_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            microscopeParametersName='test_microscope_parameters.json')
    yield testMERFISHData


@pytest.fixture(scope='session')
def simple_metamerfish_data(simple_merfish_data):

    testMetaMERFISHDataSet = dataset.MetaMERFISHDataSet(
        'metamerfish_test', ['merfish_test'])

    yield testMetaMERFISHDataSet


@pytest.fixture(scope='session')
def two_codebook_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook2.csv', 'test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=os.path.join(merlin.ANALYSIS_HOME, '..',
                                      'test_analysis_two_codebook'),
            microscopeParametersName='test_microscope_parameters.json')
    yield testMERFISHData

    shutil.rmtree('test_analysis_two_codebook')


@pytest.fixture(scope='function')
def single_task(simple_data):
    task = testtask.SimpleAnalysisTask(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(scope='function', params=[
    testtask.SimpleAnalysisTask, testtask.SimpleParallelAnalysisTask,
    testtask.SimpleInternallyParallelAnalysisTask])
def simple_task(simple_data, request):
    task = request.param(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(scope='function', params=[
    testtask.SimpleAnalysisTask, testtask.SimpleParallelAnalysisTask,
    testtask.SimpleInternallyParallelAnalysisTask])
def simple_merfish_task(simple_merfish_data, request):
    task = request.param(
        simple_merfish_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_merfish_data.delete_analysis(task)
