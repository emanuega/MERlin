import argparse
import cProfile
import dotenv
import os
import json

from merlin.core import dataset
from merlin.core import scheduler
from merlin.core import executor

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--profile', action='store_true', 
            help='enable profiling')

    parser.add_argument('-d', '--data-set', required=True)
    parser.add_argument('-a', '--analysis-parameters', required=True)
    parser.add_argument('-o', '--data-organization')
    parser.add_argument('-c', '--codebook')
    parser.add_argument('-n', '--core-count', type=int)

    return parser


def merlin():
    print('MERlin - MERFISH decoding software')
    parser = build_parser()
    args, argv = parser.parse_known_args()
    dotenv.load_dotenv(dotenv.find_dotenv())

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    dataSet = dataset.MERFISHDataSet(args.data_set, 
            dataOrganizationName=args.data_organization,
            codebookName=args.codebook)

    parametersHome = os.environ.get('PARAMETERS_HOME')

    e = executor.LocalExecutor(coreCount=args.core_count)
    with open(os.sep.join(
            [parametersHome, args.analysis_parameters]), 'r') as f:
        s = scheduler.Scheduler(dataSet, e, json.load(f))

    s.start()

