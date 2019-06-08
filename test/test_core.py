import pytest
import os

from merlin.core import executor
from merlin.core import analysistask

    
def test_task_delete(simple_data, simple_task):
    simple_data.save_analysis_task(simple_task)
    assert simple_data.analysis_exists(simple_task)
    simple_data.delete_analysis(simple_task)
    assert not simple_data.analysis_exists(simple_task)


def test_task_save(simple_data, simple_task):
    task1 = simple_task
    simple_data.save_analysis_task(task1)
    loadedTask = simple_data.load_analysis_task(task1.analysisName)
    unsharedKeys1 = [k for k in task1.parameters
                     if k not in loadedTask.parameters
                     or task1.parameters[k] != loadedTask.parameters[k]]
    assert len(unsharedKeys1) == 0
    unsharedKeys2 = [k for k in loadedTask.parameters
                     if k not in task1.parameters
                     or loadedTask.parameters[k] != task1.parameters[k]]
    assert len(unsharedKeys2) == 0
    assert loadedTask.analysisName == task1.analysisName


def test_task_run(simple_task):
    task1 = simple_task
    assert not task1.is_complete()
    assert not task1.is_started()
    assert not task1.is_running()
    assert not task1.is_error()
    task1.run()
    assert task1.is_started()
    assert not task1.is_running()
    assert not task1.is_error()
    assert task1.is_complete()


def test_save_environment(simple_task):
    task1 = simple_task
    task1.run()
    environment = dict(os.environ)
    if isinstance(simple_task, analysistask.ParallelAnalysisTask):
        taskEnvironment = simple_task.dataSet.get_analysis_environment(
            simple_task, 0)
    else:
        taskEnvironment = simple_task.dataSet.get_analysis_environment(
            simple_task)

    assert environment == taskEnvironment


@pytest.mark.slowtest
def test_task_run_with_executor(simple_task):
    task1 = simple_task
    assert not task1.is_complete()
    assert not task1.is_started()
    assert not task1.is_running()
    assert not task1.is_error()
    e = executor.LocalExecutor()
    e.run(task1, join=True)
    assert task1.is_started()
    assert not task1.is_running()
    assert not task1.is_error()
    assert task1.is_complete()


def test_task_reset(simple_task):
    simple_task.run(overwrite=False)
    assert simple_task.is_complete()
    with pytest.raises(analysistask.AnalysisAlreadyStartedException):
        simple_task.run(overwrite=False)
    simple_task.run(overwrite=True)
    assert simple_task.is_complete()


def test_task_overwrite(simple_task):
    simple_task.save()
    simple_task.parameters['new_parameter'] = 0
    with pytest.raises(analysistask.AnalysisAlreadyExistsException):
        simple_task.save()
