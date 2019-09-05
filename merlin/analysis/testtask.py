import numpy as np

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


class RandomNumberParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """A test analysis task that generates random numbers."""

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_random_result(self, fragmentIndex):
        return self.dataSet.load_numpy_analysis_result('random_numbers',
                                                       self, fragmentIndex)

    def _run_analysis(self, fragmentIndex):
        self.dataSet.save_numpy_analysis_result(
            fragmentIndex*np.random.rand(100), 'random_numbers', self,
            fragmentIndex)

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
        return 10


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
