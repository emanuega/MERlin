import dotenv
import os

dotenv.load_dotenv(os.path.join(os.path.expanduser('~'), '.env'))

DATA_HOME = os.path.expanduser(os.environ.get('DATA_HOME'))
ANALYSIS_HOME = os.path.expanduser(os.environ.get('ANALYSIS_HOME'))
PARAMETERS_HOME = os.path.expanduser(os.environ.get('PARAMETERS_HOME'))
ANALYSIS_PARAMETERS_HOME = os.sep.join(
        [PARAMETERS_HOME, 'analysis_parameters'])
CODEBOOK_HOME = os.sep.join(
        [PARAMETERS_HOME, 'codebooks'])
DATA_ORGANIZATION_HOME = os.sep.join(
        [PARAMETERS_HOME, 'dataorganization'])
POSITION_HOME = os.sep.join(
        [PARAMETERS_HOME, 'positions'])
MICROSCOPE_PARAMETERS_HOME = os.sep.join(
        [PARAMETERS_HOME, 'microscope_parameters'])
FPKM_HOME = os.sep.join([PARAMETERS_HOME, 'fpkm'])
SNAKEMAKE_PARAMETERS_HOME = os.sep.join(
    [PARAMETERS_HOME, 'snakemake_parameters'])


def version():
    import pkg_resources
    return pkg_resources.require("merlin")[0].version
