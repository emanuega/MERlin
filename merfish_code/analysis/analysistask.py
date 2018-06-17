import os
from abc import ABC, abstractmethod
import time

class AnalysisTask(object):

    def __init__(self, dataSet, analysisName=None):
        self.dataSet = dataSet

        if analysisName is None:
            self.analysisName = type(self).__name__
        else:
            self.analysisName = analysisName

    def run(self):
        self.dataSet.record_analysis_complete(self)

    def is_complete(self):
        return self.dataSet.check_analysis_done(self)

    def get_analysis_name(self):
        return self.analysisName

    def get_savepath(self, fileName=None):
        if fileName is None:
            return self.dataSet.get_analysis_subdirectory(self.analysisName)
        else:
            return os.sep.join([self.get_savepath(), fileName])


class ParallelAnalysisTask(AnalysisTask):

    def __init__(self, dataSet, analysisName=None):
        super().__init__(dataSet, analysisName)

    @abstractmethod
    def fragment_count(self):
        pass

    def run(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self.run(i)
        
        else:
            self.run_for_fragment(fragmentIndex)
            self.dataSet.record_analysis_complete(self, fragmentIndex) 

    @abstractmethod
    def run_for_fragment(self, fragmentIndex):
        pass

    def is_complete(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if not self.is_complete(i):
                    return False

            return True

        else:
            return self.dataSet.check_analysis_done(self, fragmentIndex)

