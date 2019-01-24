import pytest

from merlin.core import executor

    
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
    assert not task1.is_running()
    assert not task1.is_idle()
    assert not task1.is_error()
    task1.run()
    assert not task1.is_running()
    assert not task1.is_idle()
    assert not task1.is_error()
    assert task1.is_complete()


@pytest.mark.slowtest
def test_task_run_with_executor(simple_task):
    task1 = simple_task
    assert not task1.is_complete()
    assert not task1.is_running()
    assert not task1.is_idle()
    assert not task1.is_error()
    e = executor.LocalExecutor()
    e.run(task1, join=True)
    assert not task1.is_running()
    assert not task1.is_idle()
    assert not task1.is_error()
    assert task1.is_complete()
