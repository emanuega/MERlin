import numpy as np

from merlin.metaanalysis import cluster


def test_cluster_pipeline(simple_metamerfish_data):
    aDataTask = cluster.CreateAnnData(simple_metamerfish_data, parameters={
        'combine_outputs_task': 'CombineOutputs'})

    simple_metamerfish_data.save_analysis_task(aDataTask)
    aDataTask._run_analysis()
    assert os.path.isfile(os.sep.join([simple_metamerfish_data.analysisPath,
                                       aDataTask.analysisName,
                                       'data.h5ad']) == 1

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

    clResult = clTask.return_clustering_result(kValue, resolution, cellType)
    lowExpectedSize = (clResult.shape[0] / 3) * 0.9
    highExpectedSize = (clResult.shape[0] / 3) * 1.1
    clSizes = clResult.groupby(
        'kValue_{}_resolution_{}'.format(kValue, resolution)).size()
    assert (lowExpectedSize < clSizes[0] and clSizes[0] < highExpectedSize) == 1
    assert (lowExpectedSize < clSizes[1] and clSizes[1] < highExpectedSize) == 1
    assert (lowExpectedSize < clSizes[2] and clSizes[2] < highExpectedSize) == 1

    groups = clResult.groupby(
        'kValue_{}_resolution_{}'.format(kValue, resolution))

    for k, v in groups:
        cellList = [int(x) for x in v['index'].values.tolist()]
        if k < 3:
            r2 = (k + 1) * 1000
            r1 = r2 - 1000
            correctAssignments = [x for x in cellList
                                  if x in list(range(r1, r2)))]
            assert len(correctAssignments) >= 0.9 * len(cellList)


    bTask = cluster.BootstrapClustering(simple_metamerfish_data,
                                        parameters=params)
    bTask._run_analysis(0)

    bPath = simple_metamerfish_data._analysis_result_save_path(
        'kValue_{}_resolution_{}_type_{}__bootstrap_{}'.format(
            kValue, resolution, cellType, 0),
        bTask,
        subdirectory='final_clusters',
        fileExtension='.csv')

    assert os.path.isfile(bPath) == 1
