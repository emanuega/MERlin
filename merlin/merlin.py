import argparse
import cProfile
import dotenv
import os
import json
import sys
import snakemake

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
    parser.add_argument('dataset',
                        help='directory where the raw data is stored')
    parser.add_argument('-a', '--analysis-parameters',
                        help='name of the analysis parameters file to use')
    parser.add_argument('-o', '--data-organization',
                        help='name of the data organization file to use')
    parser.add_argument('-c', '--codebook',
                        help='name of the codebook to use')
    parser.add_argument('-m', '--microscope-parameters',
                        help='name of the microscope parameters to use')
    parser.add_argument('-p', '--positions',
                        help='name of the position file to use')
    parser.add_argument('-n', '--core-count', type=int,
                        help='number of cores to use for the analysis')
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

    return parser


def merlin():
    print('MERlin - the MERFISH decoding pipeline')
    parser = build_parser()
    args, argv = parser.parse_known_args()
    dotenv.load_dotenv(dotenv.find_dotenv())

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    dataSet = dataset.MERFISHDataSet(
        args.dataset,
        dataOrganizationName=args.data_organization,
        codebookName=args.codebook,
        microscopeParametersName=args.microscope_parameters,
        positionFileName=args.positions,
        dataHome=args.data_home,
        analysisHome=args.analysis_home
    )

    parametersHome = m.ANALYSIS_PARAMETERS_HOME
    e = executor.LocalExecutor(coreCount=args.core_count)
    snakefilePath = None
    if args.analysis_parameters:
        # This is run in all cases that analysis parameters are provided
        # so that new analysis tasks are generated to match the new parameters
        with open(os.sep.join(
                [parametersHome, args.analysis_parameters]), 'r') as f:
            print('Generating specified analysis tasks')
            analysisParameters = json.load(f)
            snakeGenerator = snakewriter.SnakemakeGenerator(
                analysisParameters, dataSet, sys.executable)
            snakefilePath = snakeGenerator.generate_workflow()
            print('Snakefile generated at %s' % snakefilePath)

    if not args.generate_only:
        if args.analysis_task:
            print('Running %s' % args.analysis_task)
            e.run(dataSet.load_analysis_task(args.analysis_task),
                  index=args.fragment_index, join=True)
        elif snakefilePath:
            print('Running pipeline with snakemake')
            snakemake.snakemake(snakefilePath, cores=args.core_count)
