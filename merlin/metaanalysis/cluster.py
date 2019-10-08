import scanpy as sc
from merlin.core import analysistask



class CreateAnnData(analysistask.AnalysisTask):
    """
    A metaanalysis task that creates an h5ad file compatible with scanpy
    """
    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

    self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        dep = [v for k,v in self.parameters.items() if 'task' in k]
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
        scData = sc.AnnData(X = data.loc[:,allGenes].values)
        scData.obs.index = data.index.values.tolist()
        scData.var.index = allGenes
        observationNames = [x for x in data.columns.values.tolist()
                            if x not in allGenes]
        scData.obs = data.loc[:,observationNames]
        self._save_h5ad(scData)

    def load_data(self):
        path = os.sep.join([self.analysisPath, self.analysisName, 'data.h5ad'])
        data = sc.read_h5ad(path)
        return data

class DetermineClusters(analysistask.ParallelAnalysisTask):

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
        return self.metaDataSet.load_analysis_task(requestedTask).load_data()

    