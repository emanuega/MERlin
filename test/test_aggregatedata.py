import os
from merlin.metaanalysis import aggregatedata


def test_aggregatedata(simple_metamerfish_data):
    aDataTask = aggregatedata.AggregateData(simple_metamerfish_data, parameters={
        'combine_outputs_task': 'CombineOutputs'})

    simple_metamerfish_data.save_analysis_task(aDataTask)
    aDataTask._run_analysis()
    assert os.path.isfile(os.sep.join([simple_metamerfish_data.analysisPath,
                                       aDataTask.analysisName,
                                       'CombineOutputs.csv'])) == 1
