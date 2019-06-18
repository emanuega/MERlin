from merlin.core import analysistask

'''This module contains dummy analysis tasks for running tests'''


class SimpleAnalysisTask(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _run_analysis(self):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        if 'dependencies' in self.parameters:
            return self.parameters['dependencies']
        else:
            return []


class SimpleParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _run_analysis(self, fragmentIndex):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        if 'dependencies' in self.parameters:
            return self.parameters['dependencies']
        else:
            return []

    def fragment_count(self):
        return 5


class SimpleInternallyParallelAnalysisTask(
        analysistask.InternallyParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _run_analysis(self):
        pass

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        if 'dependencies' in self.parameters:
            return self.parameters['dependencies']
        else:
            return []
