import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())

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
