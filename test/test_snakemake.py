import snakemake
import os
import shutil

from merlin.util import snakewriter


def test_run_single_task(simple_merfish_task):
    simple_merfish_task.save()
    assert not simple_merfish_task.is_complete()
    snakeRule = snakewriter.SnakemakeRule(simple_merfish_task)
    with open('temp.Snakefile', 'w') as outFile:
        outFile.write('rule all: \n\tinput: '
                      + snakeRule.full_output() + '\n\n')
        outFile.write(snakeRule.as_string())

    snakemake.snakemake('temp.Snakefile')
    os.remove('temp.Snakefile')
    shutil.rmtree('.snakemake')

    assert simple_merfish_task.is_complete()


def test_snakemake_generator_one_task(simple_merfish_data):
    taskDict = {'analysis_tasks': [
        {'task': 'SimpleAnalysisTask',
         'module': 'testtask',
         'parameters': {}}
    ]}

    generator = snakewriter.SnakemakeGenerator(taskDict, simple_merfish_data)
    workflow = generator.generate_workflow()
    outputTask = simple_merfish_data.load_analysis_task('SimpleAnalysisTask')
    assert not outputTask.is_complete()
    snakemake.snakemake(workflow)
    assert outputTask.is_complete()

    shutil.rmtree('.snakemake')


def test_snakemake_generator_task_chain(simple_merfish_data):
    taskDict = {'analysis_tasks': [
        {'task': 'SimpleAnalysisTask',
         'module': 'testtask',
         'analysis_name': 'Task1',
         'parameters': {}},
        {'task': 'SimpleParallelAnalysisTask',
         'module': 'testtask',
         'analysis_name': 'Task2',
         'parameters': {'dependencies': ['Task1']}},
        {'task': 'SimpleParallelAnalysisTask',
         'module': 'testtask',
         'analysis_name': 'Task3',
         'parameters': {'dependencies': ['Task2']}}
    ]}

    generator = snakewriter.SnakemakeGenerator(taskDict, simple_merfish_data)
    workflow = generator.generate_workflow()
    outputTask1 = simple_merfish_data.load_analysis_task('Task1')
    outputTask2 = simple_merfish_data.load_analysis_task('Task2')
    outputTask3 = simple_merfish_data.load_analysis_task('Task3')
    assert not outputTask1.is_complete()
    assert not outputTask2.is_complete()
    assert not outputTask3.is_complete()
    snakemake.snakemake(workflow)
    assert outputTask1.is_complete()
    assert outputTask2.is_complete()
    assert outputTask3.is_complete()

    shutil.rmtree('.snakemake')
