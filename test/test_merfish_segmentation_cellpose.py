import os
import pytest

import merlin
from merlin import merlin as m


@pytest.mark.fullrun
@pytest.mark.slowtest
def test_cellpose_2d_local(simple_merfish_data):
    with open(os.sep.join([merlin.ANALYSIS_PARAMETERS_HOME,
                           'test_analysis_parameters_segmentation_cellpose.json']), 'r') as f:
        snakefilePath = m.generate_analysis_tasks_and_snakefile(
            simple_merfish_data, f)
        m.run_with_snakemake(simple_merfish_data, snakefilePath, 5)
