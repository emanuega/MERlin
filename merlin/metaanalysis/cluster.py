import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA
from merlin.core import analysistask
from merlin.util import scanpy_helpers


class CreateAnnData(analysistask.AnalysisTask):
    """
    A metaanalysis task that creates an h5ad file compatible with scanpy,
    useful to save time with file reading in the clustering tasks
    """
    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

    self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        dep = [v for k, v in self.parameters.items() if 'task' in k]
        if len(dep) > 0:
            print('Cannot combine different file types, please combine'
                  'them ahead of time with the combineoutputs analysis task')
        return dep

    def _load_data(self):
        requestedTask = self.get_dependencies()[0]
        return self.metaDataSet.load_or_aggregate_data(requestedTask)

    def _save_h5ad(self, aData):
        path = os.sep.join([self.analysisPath, self.analysisName, 'data.h5ad'])
        aData.write(path)

    def _run_analysis(self) -> None:
        data = self._load_data()
        allGenes = self.metaDataSet.identify_multiplex_and_sequential_genes()
        scData = sc.AnnData(X=data.loc[:, allGenes].values)
        scData.obs.index = data.index.values.tolist()
        scData.var.index = allGenes
        observationNames = [x for x in data.columns.values.tolist()
                            if x not in allGenes]
        scData.obs = data.loc[:, observationNames]
        self._save_h5ad(scData)

    def load_data(self):
        path = os.sep.join([self.analysisPath, self.analysisName, 'data.h5ad'])
        data = sc.read_h5ad(path)
        return data


class Clustering(analysistask.ParallelAnalysisTask):

    """
    A metaanalysis task that determines clusters of cells based on the
    underlying merfish data
    """

    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        if 'cell_type' not in self.parameters:
            self.parameters['cell_type'] = 'All'
        if 'k_value' not in self.parameters:
            self.parameters['k_value'] = [12]
        if 'resolution' not in self.parameters:
            self.parameters['resolution'] = [1.0]
        if 'log_x_plus_1_performed' not in self.parameters:
            self.parameters['log_x_plus_1_performed'] = False
        if 'volume_normalization_performed' not in self.parameters:
            self.parameters['volume_normalization_performed'] = False
        if 'use_PCs' not in self.parameters:
            self.parameters['use_PCs'] = True
        if 'cluster_min_size' not in self.parameters:
            self.parameters['cluster_min_size'] = 10
        if 'clustering_algorithm' not in self.parameters:
            self.parameters['clustering_algorithm'] = 'leiden'

        self.metaDataSet = metaDataSet

    def fragment_count(self):
        return len(self.parameters['k_value']) *\
               len(self.parameters['resolution'])

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        return self.parameters['filecreationtask']

    def _load_data(self):
        requestedTask = self.get_dependencies()
        data = self.metaDataSet.load_analysis_task(requestedTask).load_data()
        data.var_names_make_unique()
        return data

    def _expand_k_and_resolution(self):
        kValues = self.parameters['k_value']
        resolutions = self.parameters['resolution']
        allPairs = []
        for k in kValues:
            for r in resolutions:
                allPairs.append([k, r])
        return allPairs

    def _save_h5ad(self):
        path = os.sep.join([self.analysisPath, self.analysisName, 'data.h5ad'])
        aData.write(path)

    def load_data(self):
        path = os.sep.join([self.analysisPath, self.analysisName, 'data.h5ad'])
        data = sc.read_h5ad(path)
        return data

    def _cut_to_cell_list(self, aData, pathToCells: str):
        """
        cuts the anndata object to the subset of cells found in the csv file
        present at pathToCells.

        Inputs:
            pathToCells: path to csv file containing each cell to be used in
                clustering as its own line. Assumes file has a header and does
                not have an index.
        Returns:
            anndata object cut to requested cells
        """
        cellsToKeep = pd.read_csv(pathToCells).iloc[:, 0].values.tolist()
        return aData[cellsToKeep, :]

    def _add_cell_obs(self, aData):
        """
        Given an anndata object, computes the number of non-zero variables for
        each observation and adds it to the object as n_genes, also computes
        n_counts as the sum of raw values for each observation
        """
        if 'n_genes' not in aData.obs:
            mask = pd.DataFrame(data=np.where(aData.X > 0, 1, 0),
                                index=aData.obs.index, columns=aData.var.index)
            aData.obs['n_genes'] = mask.sum(1)

        if 'n_counts' not in aData.obs:
            temp = pd.DataFrame(data=aData.X, index=aData.obs.index,
                                columns=aData.var.index)
            if self.parameters['log_x_plus_1_performed']:
                temp = temp.apply(lambda x: ((10 ** x) - 1))
            if self.parameters['volume_normalization_performed']:
                if 'volume' in aData.obs:
                    temp = temp.mul(aData.obs['volume'], 0)
                else:
                    print('Rename volume column in anndata obs to \'volume\'')
            aData.obs['n_counts'] = temp.sum(1)
        return aData

    def _filter_by_obs(self, aData, column, minPercentile, maxPercentile):
        print('original dataset contains {} cells'.format(aData.X.shape[0]))
        currentData = aData.copy()
        if column in currentData.obs:
            filterColumn = currentData.obs.loc[:, column]
            minVal = filterColumn.quantile(q=minPercentile)
            maxVal = filterColumn.quantile(q=maxPercentile)
            keptObs = filterColumn[(filterColumn > minVal) &
                                   (filterColumn < maxVal)]
            currentData = currentData[keptObs, :].copy()
        else:
            print('column \'{}\' was not found in data, skipping'
                  .format(column))
        print('filtered by {}, dataset contains {} cells'.format(
            column, currentData.X.shape[0]))
        return currentData

    def _filter_by_var(self, aData, minPercentile, maxPercentile):
        print('original dataset contains {} genes'.format(aData.X.shape[1]))
        currentData = aData.copy()
        mask = pd.DataFrame(np.where(currentData.X > 0, 1, 0),
                            columns=currentData.var.index.values.tolist())
        counts = mask.sum()
        minVal = counts.quantile(q=minPercentile)
        maxVal = counts.quantile(q=maxPercentile)
        keptVar = counts[(counts > minVal) & (counts < maxVal)]
        currentData = currentData[:, keptVar].copy()
        print('filtered dataset contains {} genes'.format(
            currentData.X.shape[1]))
        return currentData

    def _select_significant_PCs(self, aData):
        maxPCs = int(np.min(aData.X.shape)) - 1
        if maxPCs < 100:
            pcsToCalc = maxPCs
        else:
            pcsToCalc = 100
        sc.tl.pca(aData, svd_solver='arpack', n_comps=pcsToCalc)
        randomPCs = PCA(n_components=1, svd_solver='arpack')
        randomVariance = [randomPCs.fit(
            scanpy_helpers.shuffler(aData.X)).explained_variance_[0]
                          for _ in range(10)]

        # Use only PCs that explain more variance than the random dataframe
        pcsToUse = len(aData.uns['pca']['variance']
                       [aData.uns['pca']['variance'] >
                        np.median(randomVariance)])
        self.pcsToUse = pcsToUse
        print('Using {} PCs'.format(pcsToUse))
        return aData

    def _compute_neighbors(self, aData, kValue):
        self.kValue = kValue

        if self.parameters['use_PCs']:
            sc.pp.neighbors(aData, n_neighbors=int(kValue),
                            n_pcs=self.pcsToUse)
        else:
            sc.pp.neighbors(aData, n_neighbors=int(kValue), n_pcs=0)

        aData.uns['neighbors']['connectivities'] =\
            scanpy_helpers.neighbor_graph(
                scanpy_helpers.jaccard_kernel,
                aData.uns['neighbors']['connectivities'])
        return aData

    def _cluster(self, aData, resolution, clusterMin=10,
                 clusteringAlgorithm='louvain', i=None):
        self.resolution = resolution

        adjacency = aData.uns['neighbors']['connectivities']
        g = sc.utils.get_igraph_from_adjacency(adjacency, directed=False)

        if clusteringAlgorithm == 'louvain':
            import louvain as clAlgo
            print('using louvain algorithm')
        elif clusteringAlgorithm == 'leiden':
            import leidenalg as clAlgo
            print('using leiden algorithm')

        optimiser = clAlgo.Optimiser()
        tracking = []
        partition = clAlgo.RBConfigurationVertexPartition(
            g, weights='weight', resolution_parameter=resolution)
        partition_agg = partition.aggregate_partition()
        print(partition.summary())

        diff = optimiser.move_nodes(partition_agg)
        while diff > 0.0:
            partition.from_coarse_partition(partition_agg)
            partition_agg = partition_agg.aggregate_partition()
            tracking.append(partition.membership)
            print(partition_agg.summary())
            diff = optimiser.move_nodes(partition_agg)

        df = pd.DataFrame(tracking, columns=self.dataset.obs.index).T

        self._save_clustering_result(df, 'iterations')

        clustering = scanpy_helpers.minimum_cluster_size(
            df.iloc[:, [-1]].copy(deep=True), min_size=clusterMin)

        clustering.columns = ['kValue_{}_resolution_{}'.format(
            self.kValue, int(self.resolution))]

        print('Clustering yields {} clusters with at least {} cells'.format(
            clustering['kValue_{}_resolution_{}'.format(self.kValue, int(
                    self.resolution))].unique().astype(int).max(), clusterMin))

        self._save_clustering_result(clustering, 'final_clusters')

    def _save_clustering_result(self, df, subdir):
        self.metaDataSet.save_dataframe_to_csv(
            df, 'kValue_{}_resolution_{}_type_{}'.format(
                self.kValue, int(self.resolution),
                self.parameters['cell_type']), analysisTask=self.analysisName,
            subdirectory=subdir)

    def return_clustering_result(self, kValue, resolution, cellType):
        data = self.metaDataSet.load_dataframe_from_csv(
            'kValue_{}_resolution_{}_type_{}'.format(kValue, int(resolution),
                                                     cellType),
            analysisTask=self.analysisName, subdirectory='final_clusters')
        return data

    def _run_analysis(self, fragmentIndex):
        aData = self._load_data()
        kValue, resolution = self._expand_k_and_resolution()[fragmentIndex]

        if self.parameters['cell_type'] != 'All':
            aData = self._cut_to_cell_list(aData,
                                           self.parameters['path_to_cells'])
        if 'filter_obs' in self.parameters:
            if ((('n_counts' in self.parameters['filter_obs']) and
                 ('n_counts' not in aData.obs))
                    or (('n_genes' in self.parameters['filter_obs']) and
                        ('n_genes' not in aData.obs))):
                aData = self._add_cell_obs(aData)
            for k, v in self.parameters['filter_obs'].items():
                column = k
                minPercentile = v['min_pct']
                maxPercentile = v['max_pct']
                aData = self._filter_by_obs(aData, column,
                                            minPercentile, maxPercentile)
        if 'filter_var' in self.parameters:
            minPercentile = self.parameters['filter_var']['min_pct']
            maxPercentile = self.parameters['filter_var']['max_pct']
            aData = self._filter_by_var(aData, minPercentile, maxPercentile)

        if self.parameters['use_PCs']:
            aData = self._select_significant_PCs(aData)

        aData = self._compute_neighbors(aData, kValue)

        clusterMin = self.parameters['cluster_min_size']
        clusteringAlgorithm = self.parameters['clustering_algorithm']
        self._cluster(aData, resolution, clusterMin=clusterMin,
                      clusteringAlgorithm=clusteringAlgorithm)


class BootstrapClustering(Clustering):

    """
    A metaanalysis task that determines clusters of cells based on the
    underlying merfish data
    """

    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters=None, analysisName=None)

        if 'bootstrap_fraction' not in self.parameters:
            self.parameters['bootstrap_fraction'] = 0.8
        if 'bootstrap_iterations' not in self.parameters:
            self.parameters['bootstrap_iterations'] = 20

        self.metaDataSet = metaDataSet

    def fragment_count(self):
        return len(self.parameters['k_value']) *\
               len(self.parameters['resolution']) *\
               self.parameters['bootstrap_iterations']

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        return self.parameters['filecreationtask']

    def _expand_k_and_resolution(self):
        kValues = self.parameters['k_value']
        resolutions = self.parameters['resolution']
        allPairs = []
        for i in range(self.parameters['bootstrap_iterations']):
            for k in kValues:
                for r in resolutions:
                    allPairs.append([k, r, i])
        return allPairs

    def _bootstrapCells(self, aData, bootstrapFrac):
        sampleDF = pd.DataFrame(aData.X, index=aData.obs.index,
                                columns=aData.var.index)
        downSample = sampleDF.sample(frac=bootstrapFrac)
        downSampleAD = sc.AnnData(downSample.values)
        downSampleAD.obs.index = downSample.index.values.tolist()
        downSampleAD.var.index = downSample.columns.values.tolist()
        downSampleAD.obs = self.dataset.obs.loc[downSampleAD.obs.index, :]
        return downSampleAD

    def _save_clustering_result(self, df, subdir):
        self.metaDataSet.save_dataframe_to_csv(
            df, 'kValue_{}_resolution_{}_type_{}_bootstrap_{}'.format(
                self.kValue, int(self.resolution),
                self.parameters['cell_type'], self.i),
            analysisTask=self.analysisName, subdirectory=subdir)

    def return_clustering_result(self, kValue, resolution, cellType, i):
        data = self.metaDataSet.load_dataframe_from_csv(
            'kValue_{}_resolution_{}_type_{}_bootstrap_{}'.format(
                kValue, int(resolution), cellType, i),
            analysisTask=self.analysisName, subdirectory='final_clusters')
        return data

    def _run_analysis(self, fragmentIndex):
        aData = self._load_data()
        aData = self._bootstrapCells(aData,
                                     self.parameters['bootstrap_fraction'])
        kValue, resolution, i = self._expand_k_and_resolution()[fragmentIndex]
        self.i = i
        if self.parameters['cell_type'] != 'All':
            aData = self._cut_to_cell_list(aData,
                                           self.parameters['path_to_cells'])
        if 'filter_obs' in self.parameters:
            if ((('n_counts' in self.parameters['filter_obs']) and
                 ('n_counts' not in aData.obs))
                    or (('n_genes' in self.parameters['filter_obs']) and
                        ('n_genes' not in aData.obs))):
                aData = self._add_cell_obs(aData)
            for k, v in self.parameters['filter_obs'].items():
                column = k
                minPercentile = v['min_pct']
                maxPercentile = v['max_pct']
                aData = self._filter_by_obs(aData, column,
                                            minPercentile, maxPercentile)
        if 'filter_var' in self.parameters:
            minPercentile = self.parameters['filter_var']['min_pct']
            maxPercentile = self.parameters['filter_var']['max_pct']
            aData = self._filter_by_var(aData, minPercentile, maxPercentile)

        if self.parameters['use_PCs']:
            aData = self._select_significant_PCs(aData)

        aData = self._compute_neighbors(aData, kValue)

        clusterMin = self.parameters['cluster_min_size']
        clusteringAlgorithm = self.parameters['clustering_algorithm']
        self._cluster(aData, resolution, clusterMin=clusterMin,
                      clusteringAlgorithm=clusteringAlgorithm, i=i)


class ClusterStabilityAnalysis(analysistask.AnalysisTask):
    """
    A metaanalysis task that determines the stability of clusters based on
    the proportion of cells originally assigned to a given cluster that
    remain clustered when a random subest of the data is reclustered
    """

    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        return [self.parameters['cluster_task'],
                self.parameters['bootstrap_cluster_task']]

    def _get_cluster_and_bootstrap_params(self):
        clTask = self.metaDataSet.load_analysis_task(
            self.parameters['cluster_task'])
        bootTask = self.metaDataSet.load_analysis_task(
            self.parameters['bootstrap_cluster_task'])
        kValues = sorted(clTask.parameters['k_value'])
        resolutions = sorted(clTask.parameters['resolution'])
        cellType = clTask.parameters['cell_type']
        bootstrapIterations = bootTask.parameters['bootstrap_iterations']

        return (kValues, resolutions, cellType, bootstrapIterations)

    def _gather_data(self, kValue, resolution, cellType, bootstrapIterations):
        clTask = self.metaDataSet.load_analysis_task(
            self.parameters['cluster_task'])
        bootTask = self.metaDataSet.load_analysis_task(
            self.parameters['bootstrap_cluster_task'])

        fullClustering = clTask.return_clustering_result(kValue,
                                                         resolution,
                                                         cellType)
        for result in range(bootstrapIterations):
            bootClustering = bootTask.return_clustering_result(kValue,
                                                               resolution,
                                                               cellType, result)
            if result == 0:
                fullBoot = bootClustering.copy(deep=True)
            else:
                fullBoot = pd.concat([fullBoot, bootClustering], axis=1)

        return fullClustering, fullBoot

    def _determine_stability(self, fullClustering, fullBoot):
        for boot in range(fullBoot.shape[1]):
            tempMerge = fullClustering.merge(fullBoot, left_index=True,
                                             right_index=True)
            tempMerge.columns = ['Full', 'Boot']
            tempMerge = tempMerge[tempMerge['Full'] != -1]
            recovery = tempMerge.groupby(['Full', 'Boot']).size().unstack().\
                max(1).div(tempMerge.groupby('Full').size())
            if boot == 0:
                recoveryDF = pd.DataFrame(recovery)
            else:
                recoveryDF = pd.concat([recoveryDF, pd.DataFrame(recovery)],
                                       axis=1)
        stableClusters = recoveryDF[recoveryDF.median(1) > 0.5]\
            .index.values.tolist()
        colName = fullClustering.columns.values.tolist()[0]

        totalCells = fullClustering.shape[0]
        recoveredCells = len(fullClustering[(fullClustering[colName].isin(
            stableClusters)) & (fullClustering[colName] != -1)].index.
                             values.tolist())

        return (stableClusters, recoveryDF,
                recoveredCells, totalCells)

    def _run_analysis(self) -> None:
        kValues, resolutions, cellType, bootstrapIterations =\
            self._get_cluster_and_bootstrap_params()

        toDF = []
        for kValue in kValues:
            for resolution in resolutions:
                fullClustering, fullBoot = self._gather_data(
                    kValue, resolution, cellType, bootstrapIterations)
                stableClusters, recoveryDF, recoveredCells, totalCells =\
                    self._determine_stability(fullClustering, fullBoot)
                toDF.append([kValue, resolution, len(stableClusters),
                             len(recoveryDF), recoveredCells, totalCells])
        df = pd.DataFrame(toDF, columns=['kValue', 'resolution',
                                         'stable clusters', 'total clusters',
                                         'stable cells', 'total cells'])
        df['fraction stable clusters'] = df['stable clusters'] /\
                                         df['total clusters']

        df['fraction stable cells'] = df['stable cells'] /\
                                      df['total cells']

        selectedKandRes = df[df['fraction stable cells'] >= 0.9].sort_values(
            by='stable clusters', ascending=False).iloc[0, :]



