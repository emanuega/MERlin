import os
import shutil
import pytest

from merlin.core import dataset
from merlin.core import executor
from merlin.core import analysistask
import merlin

    
def test_task_delete(simple_data, simple_task):
    simple_data.save_analysis_task(simple_task)
    assert simple_data.analysis_exists(simple_task)
    simple_data.delete_analysis(simple_task)
    assert not simple_data.analysis_exists(simple_task)

def test_task_save(simple_data, simple_task):
    task1 = simple_task
    simple_data.save_analysis_task(task1)
    loadedTask = simple_data.load_analysis_task(task1.analysisName)
    unsharedKeys = [k for k in task1.parameters \
            if k not in loadedTask.parameters \
            or task1.parameters[k] != loadedTask.parameters[k]]
    assert len(unsharedKeys)==0

def test_task_run(simple_task):
    task1 = simple_task
    assert not task1.is_complete()
    assert not task1.is_running()
    e = executor.LocalExecutor()
    e.run(task1, join=True)
    assert not task1.is_running()
    assert task1.is_complete()
    print(task1.is_error())
    assert not task1.is_error()




