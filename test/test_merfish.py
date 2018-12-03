import os
import json
import pytest

from merlin.core import scheduler
from merlin.core import executor
import merlin


@pytest.mark.slowtest
def test_merfish_2d_full_local(simple_merfish_data):
    e = executor.LocalExecutor(coreCount=2)
    with open(os.sep.join([merlin.ANALYSIS_PARAMETERS_HOME, 
            'test_analysis_parameters.json']), 'r') as f:
        s = scheduler.Scheduler(simple_merfish_data, e, json.load(f))
    
    s.start()

