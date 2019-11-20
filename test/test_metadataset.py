import os
import pytest

import merlin
from merlin import merlin as m
from merlin.analysis import combineoutputs

import numpy as np
import pandas as pd

@pytest.mark.fullrun
@pytest.mark.slowtest

def test_metamerfish_full_local(simple_metamerfish_data, simple_merfish_data):
    parameters = {"segment_export_task": "ExportCellMetadata",
                  "partitioned_bc_export_task": "ExportPartitionedBarcodes",
                  "sum_signal_export_task": "ExportSumSignals",
                  "volume_normalize": False, "calculate_counts": False,
                  "log_x_plus_1": False}

    combineTask = combineoutputs.CombineOutputs(simple_merfish_data,
                                                parameters = parameters)
    combineTask.save(overwrite=True)
    cbs = simple_merfish_data.codebooks
    geneList = []
    for cb in cbs:
        geneList.extend(cb._data['name'].values.tolist())
    geneList.extend(
        simple_merfish_data.dataOrganization.get_sequential_rounds()[1])
    cellData = pd.DataFrame(data = np.random.rand(1000, len(geneList)),
                            columns = geneList)
    combineTask.dataSet.save_dataframe_to_csv(cellData,
                                              'combined_output',
                                              combineTask.get_analysis_name())
    combineTask.dataSet.record_analysis_complete(combineTask)


    with open(os.sep.join([merlin.ANALYSIS_PARAMETERS_HOME,
                           'test_metaanalysis_parameters.json']), 'r') as f:
        snakefilePath = m.generate_analysis_tasks_and_snakefile(
            simple_metamerfish_data, f)
        m.run_with_snakemake(simple_metamerfish_data, snakefilePath,
                             5, report = False)
