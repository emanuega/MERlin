import dotenv
import os
import glob
import json
import importlib
from typing import List

from merlin.core import dataset

envPath = os.path.join(os.path.expanduser('~'), '.merlinenv')

if os.path.exists(envPath):
    dotenv.load_dotenv(envPath)

    try:
        DATA_HOME = os.path.expanduser(os.environ.get('DATA_HOME'))
        ANALYSIS_HOME = os.path.expanduser(os.environ.get('ANALYSIS_HOME'))
        PARAMETERS_HOME = os.path.expanduser(os.environ.get('PARAMETERS_HOME'))
        ANALYSIS_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'analysis'])
        CODEBOOK_HOME = os.sep.join(
                [PARAMETERS_HOME, 'codebooks'])
        DATA_ORGANIZATION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'dataorganization'])
        POSITION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'positions'])
        MICROSCOPE_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'microscope'])
        FPKM_HOME = os.sep.join([PARAMETERS_HOME, 'fpkm'])
        SNAKEMAKE_PARAMETERS_HOME = os.sep.join(
            [PARAMETERS_HOME, 'snakemake'])

    except TypeError:
        print('MERlin environment appears corrupt. Please run ' +
              '\'merlin --configure .\' in order to configure the environment.')
else:
    print(('Unable to find MERlin environment file at %s. Please run ' +
          '\'merlin --configure .\' in order to configure the environment.')
          % envPath)


def store_env(dataHome, analysisHome, parametersHome):
    with open(envPath, 'w') as f:
        f.write('DATA_HOME=%s\n' % dataHome)
        f.write('ANALYSIS_HOME=%s\n' % analysisHome)
        f.write('PARAMETERS_HOME=%s\n' % parametersHome)


class IncompatibleVersionException(Exception):
    pass


def version():
    import pkg_resources
    return pkg_resources.require("merlin")[0].version


def is_compatible(testVersion: str, baseVersion: str = None) -> bool:
    """ Determine if testVersion is compatible with baseVersion

    Args:
        testVersion: the version identifier to test, as the string 'x.y.z'
            where x is the major version, y is the minor version,
            and z is the patch.
        baseVersion: the version to check testVersion's compatibility. If  not
            specified then the current MERlin version is used as baseVersion.
    Returns: True if testVersion are compatible, otherwise false.
    """
    if baseVersion is None:
        baseVersion = version()
    return testVersion.split('.')[0] == baseVersion.split('.')[0]


def get_analysis_datasets(maxDepth=2) -> List[dataset.DataSet]:
    """ Get a list of all datasets currently stored in analysis home.

    Args:
        maxDepth: the directory depth to search for datasets.
    Returns: A list of the dataset objects currently within analysis home.
    """
    metadataFiles = []
    for d in range(1, maxDepth+1):
        metadataFiles += glob.glob(os.path.join(
            ANALYSIS_HOME, *['*']*d, 'dataset.json'))

    def load_dataset(jsonPath) -> dataset.DataSet:
        with open(jsonPath, 'r') as f:
            metadata = json.load(f)
            analysisModule = importlib.import_module(metadata['module'])
            analysisTask = getattr(analysisModule, metadata['class'])
            return analysisTask(metadata['dataset_name'])

    return [load_dataset(m) for m in metadataFiles]
