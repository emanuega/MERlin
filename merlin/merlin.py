import argparse
import cProfile
import os
import json
import sys
import snakemake
import time
import requests
from typing import TextIO
from typing import Dict

import merlin as m
from merlin.core import dataset
from merlin.core import executor
from merlin.util import snakewriter


def build_parser():
    parser = argparse.ArgumentParser(description='Decode MERFISH data.')

    parser.add_argument('--profile', action='store_true',
                        help='enable profiling')
    parser.add_argument('--generate-only', action='store_true',
                        help='only generate the directory structure and ' +
                        'do not run any analysis.')
    parser.add_argument('--configure', action='store_true',
                        help='configure MERlin environment by specifying ' +
                        ' data, analysis, and parameters directories.')
    parser.add_argument('dataset',
                        help='directory where the raw data is stored')
    parser.add_argument('-a', '--analysis-parameters',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-o', '--data-organization',
                        help='name of the data organization file to use')
    parser.add_argument('-c', '--codebook', nargs='+',
                        help='name of the codebook to use')
    parser.add_argument('-m', '--microscope-parameters',
                        help='name of the microscope parameters to use')
    parser.add_argument('-p', '--positions',
                        help='name of the position file to use')
    parser.add_argument('-n', '--core-count', type=int,
                        help='number of cores to use for the analysis')
    parser.add_argument('--check-done', action='store_true',
                        help='flag to only check if the analysis task is ' +
                        'done')
    parser.add_argument(
        '-t', '--analysis-task',
        help='the name of the analysis task to execute. If no '
             + 'analysis task is provided, all tasks are executed.')
    parser.add_argument(
        '-i', '--fragment-index', type=int,
        help='the index of the fragment of the analysis task to execute')
    parser.add_argument('-e', '--data-home',
                        help='the data home directory')
    parser.add_argument('-s', '--analysis-home',
                        help='the analysis home directory')
    parser.add_argument('-k', '--snakemake-parameters',
                        help='the name of the snakemake parameters file')
    parser.add_argument('--no_report',
                        help='flag indicating that the snakemake stats ' +
                        'should not be shared to improve MERlin')

    return parser


def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    return stringIn.strip('\'').strip('\"')


def _get_input_path(prompt):
    while True:
        pathString = str(input(prompt))
        if not pathString.startswith('s3://') \
                and not os.path.exists(os.path.expanduser(pathString)):
            print('Directory %s does not exist. Please enter a valid path.'
                  % pathString)
        else:
            return pathString


def configure_environment():
    dataHome = _get_input_path('DATA_HOME=')
    analysisHome = _get_input_path('ANALYSIS_HOME=')
    parametersHome = _get_input_path('PARAMETERS_HOME=')
    m.store_env(dataHome, analysisHome, parametersHome)


def merlin():
    print('MERlin - the MERFISH decoding pipeline')
    parser = build_parser()
    args, argv = parser.parse_known_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.configure:
        print('Configuring MERlin environment')
        configure_environment()
        return

    dataSet = dataset.MERFISHDataSet(
        args.dataset,
        dataOrganizationName=_clean_string_arg(args.data_organization),
        codebookNames=args.codebook,
        microscopeParametersName=_clean_string_arg(args.microscope_parameters),
        positionFileName=_clean_string_arg(args.positions),
        dataHome=_clean_string_arg(args.data_home),
        analysisHome=_clean_string_arg(args.analysis_home)
    )

    parametersHome = m.ANALYSIS_PARAMETERS_HOME
    e = executor.LocalExecutor(coreCount=args.core_count)
    snakefilePath = None
    if args.analysis_parameters:
        # This is run in all cases that analysis parameters are provided
        # so that new analysis tasks are generated to match the new parameters
        with open(os.sep.join(
                [parametersHome, args.analysis_parameters]), 'r') as f:
            snakefilePath = generate_analysis_tasks_and_snakefile(
                dataSet, f)

    if not args.generate_only:
        if args.analysis_task:
            task = dataSet.load_analysis_task(args.analysis_task)
            if args.check_done:
                # checking completion creates the .done file for parallel tasks
                # where completion has not yet been checked
                if task.is_complete():
                    print('Task %s is complete' % args.analysis_task)
                else:
                    print('Task %s is not complete' % args.analysis_task)

            else:
                print('Running %s' % args.analysis_task)
                e.run(task, index=args.fragment_index)
        elif snakefilePath:
            snakemakeParameters = {}
            if args.snakemake_parameters:
                with open(os.sep.join([m.SNAKEMAKE_PARAMETERS_HOME,
                                      args.snakemake_parameters])) as f:
                    snakemakeParameters = json.load(f)

            run_with_snakemake(dataSet, snakefilePath, args.core_count,
                               snakemakeParameters, not args.no_report)


def generate_analysis_tasks_and_snakefile(dataSet: dataset.MERFISHDataSet,
                                          parametersFile: TextIO) -> str:
    print('Generating analysis tasks from %s' % parametersFile.name)
    analysisParameters = json.load(parametersFile)
    snakeGenerator = snakewriter.SnakefileGenerator(
        analysisParameters, dataSet, sys.executable)
    snakefilePath = snakeGenerator.generate_workflow()
    print('Snakefile generated at %s' % snakefilePath)
    return snakefilePath


def run_with_snakemake(
        dataSet: dataset.MERFISHDataSet, snakefilePath: str, coreCount: int,
        snakemakeParameters: Dict = {}, report: bool = True):
    print('Running MERlin pipeline through snakemake')
    snakemake.snakemake(snakefilePath, cores=coreCount,
                        workdir=dataSet.get_snakemake_path(),
                        stats=snakefilePath + '.stats', lock=False,
                        **snakemakeParameters)

    if report:
        reportTime = int(time.time())
        try:
            with open(snakefilePath + '.stats', 'r') as f:
                requests.post('http://merlin.georgeemanuel.com/post',
                              files={
                                  'file': (
                                      '.'.join(
                                          ['snakestats',
                                           dataSet.dataSetName,
                                           str(reportTime)]) + '.csv',
                                      f)},
                              timeout=10)
        except requests.exceptions.RequestException:
            pass

        analysisParameters = {
            t: dataSet.load_analysis_task(t).get_parameters()
            for t in dataSet.get_analysis_tasks()}
        datasetMeta = {
            'image_width': dataSet.get_image_dimensions()[0],
            'image_height': dataSet.get_image_dimensions()[1],
            'barcode_length': dataSet.get_codebook().get_bit_count(),
            'barcode_count': dataSet.get_codebook().get_barcode_count(),
            'fov_count': len(dataSet.get_fovs()),
            'z_count': len(dataSet.get_z_positions()),
            'sequential_count': len(dataSet.get_data_organization()
                                    .get_sequential_rounds()),
            'dataset_name': dataSet.dataSetName,
            'report_time': reportTime,
            'analysis_parameters': analysisParameters
        }
        try:
            requests.post('http://merlin.georgeemanuel.com/post',
                          files={'file': ('.'.join(
                              [dataSet.dataSetName,
                               str(reportTime)])
                                          + '.json',
                                          json.dumps(datasetMeta))},
                          timeout=10)
        except requests.exceptions.RequestException:
            pass
