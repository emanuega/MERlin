import dotenv
import os

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


def version():
    import pkg_resources
    return pkg_resources.require("merlin")[0].version
