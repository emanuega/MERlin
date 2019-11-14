import os


import os
import pytest

import merlin
from merlin import merlin as m


@pytest.mark.fullrun
@pytest.mark.slowtest
def test_metamerfish_full_local(simple_metamerfish_data):
    with open(os.sep.join([merlin.ANALYSIS_PARAMETERS_HOME,
                           'test_metaanalysis_parameters.json']), 'r') as f:
        snakefilePath = m.generate_analysis_tasks_and_snakefile(
            simple_metamerfish_data, f)
        m.run_with_snakemake(simple_metamerfish_data, snakefilePath, 5)
