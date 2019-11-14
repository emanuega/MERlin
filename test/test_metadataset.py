import os
from merlin.metaanalysis import aggregatedata
from merlin.metaanalysis import cluster


def test_construct_metadataset(simple_metamerfish_data):
    assert ('merfish_test' in simple_metamerfish_data.dataSets) == 1

def test_aggregatedata(simple_metamerfish_data):
    aDataTask = aggregatedata.AggregateData(simple_metamerfish_data, parameters={
        'combine_outputs_task': 'CombineOutputs'})

    simple_metamerfish_data.save_analysis_task(aDataTask)
    aDataTask._run_analysis()
    assert os.path.isfile(os.sep.join([simple_metamerfish_data.analysisPath,
                                       aDataTask.analysisName,
                                       'CombineOutputs.csv'])) == 1

def test_cluster_pipeline(simple_metamerfish_data):
    aDataTask = cluster.CreateAnnData(simple_metamerfish_data, parameters={
        'combined_analysis': 'CombineOutputs',
        'aggregate_task':'AggregateData'})

    simple_metamerfish_data.save_analysis_task(aDataTask)
    aDataTask._run_analysis()
    assert os.path.isfile(os.sep.join([simple_metamerfish_data.analysisPath,
                                       aDataTask.analysisName,
                                       'data.h5ad'])) == 1

    params = {'filecreationtask': 'CreateAnnData',
              'log_x_plus_1_performed': True,
              'volume_normalization_performed': True,
              'clustering_algorithm': 'louvain'}

    clTask = cluster.Clustering(simple_metamerfish_data, parameters=params)
    clTask._run_analysis(0)

    kValue = clTask.parameters['k_value'][0]
    resolution = int(clTask.parameters['resolution'][0])
    cellType = clTask.parameters['cell_type']

    clPath = simple_metamerfish_data._analysis_result_save_path(
        'kValue_{}_resolution_{}_type_{}'.format(kValue, resolution, cellType),
        clTask,
        subdirectory='final_clusters',
        fileExtension='.csv')

    assert os.path.isfile(clPath) == 1

    bTask = cluster.BootstrapClustering(simple_metamerfish_data,
                                        parameters=params)
    bTask._run_analysis(0)

    bPath = simple_metamerfish_data._analysis_result_save_path(
        'kValue_{}_resolution_{}_type_{}_bootstrap_{}'.format(
            kValue, resolution, cellType, 0),
        bTask,
        subdirectory='final_clusters',
        fileExtension='.csv')

    assert os.path.isfile(bPath) == 1
