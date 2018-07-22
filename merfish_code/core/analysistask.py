import os
from abc import ABC, abstractmethod
import time

class AnalysisTask(ABC):

    '''
    An abstract class for performing analysis on a DataSet. Subclasses
    should implement the analysis to perform in the run_analysis() function.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        '''Creates an AnalysisTask object that performs analysis on the
        specified DataSet.

        Args:
            dataSet: the DataSet to run analysis on.
            parameters: a dictionary containing parameters used to run the
                analysis.
            analysisName: specifies a unique identifier for this
                AnalysisTask. If analysisName is not set, the analysis name
                will default to the name of the class.
        '''
        self.dataSet = dataSet
        self.parameters = parameters
        if analysisName is None:
            self.analysisName = type(self).__name__
        else:
            self.analysisName = analysisName

    def run(self):
        '''Run this AnalysisTask.
        
        Upon completion of the analysis, this function informs the DataSet
        that analysis is complete.
        '''
        self.run_analysis()
        self.dataSet.record_analysis_complete(self)

    @abstractmethod
    def run_analysis(self):
        '''Perform the analysis for this AnalysisTask.

        This function should be implemented in all subclasses with the
        logic to complete the analysis.
        '''
        pass

    @abstractmethod
    def get_estimated_memory(self):
        '''Get an estimate of how much memory is required for this
        AnalysisTask.

        Returns:
            a memory estimate in megabytes.
        '''
        pass

    @abstractmethod
    def get_estimated_time(self):
        '''Get an estimate for the amount of time required to complete
        this AnalysisTask.

        Returns:
            a time estimate in minutes.
        '''
        pass

    def is_complete(self):
        '''Determines if this analysis has completed successfully
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_done(self)

    def get_analysis_name(self):
        '''Get the name for this AnalysisTask.

        Returns:
            the name of this AnalysisTask
        '''
        return self.analysisName


class ParallelAnalysisTask(AnalysisTask):

    '''
    An abstract class for analysis that can be run in multiple parts 
    independently. Subclasses should implement the analysis to perform in 
    the run_analysis() function
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    @abstractmethod
    def fragment_count(self):
        pass

    def run(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self.run(i)
        else:
            self.run_analysis(fragmentIndex)
            self.dataSet.record_analysis_complete(self, fragmentIndex) 

    @abstractmethod
    def run_analysis(self, fragmentIndex):
        pass

    def is_complete(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if not self.is_complete(i):
                    return False

            return True

        else:
            return self.dataSet.check_analysis_done(self, fragmentIndex)

