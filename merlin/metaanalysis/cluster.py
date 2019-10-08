



from merlin.core import analysistask






class DetermineClusters(analysistask.AnalysisTask):

    """
    A metaanalysis task that determines clusters of cells based on the
    underlying merfish data
    """

    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        if 'cell_type' not in self.parameters:
            self.parameters['cell_type'] = 'All'

        self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        dep = [v for k,v in self.parameters.items() if 'task' in k]
        if len(dep) > 0:
            print('Clustering cannot be run on multiple distinct file types,'
                  'please combine them ahead of time and reference')
        return dep

    def _load_data(self):
        requestedTask = self.get_dependencies()[0]
        return self.metaDataSet.load_or_aggregate_data(requestedTask)



