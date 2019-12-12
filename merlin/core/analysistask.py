import copy
from abc import ABC, abstractmethod
import threading
import multiprocessing
from typing import List

import merlin


class AnalysisAlreadyStartedException(Exception):
    pass


class AnalysisAlreadyExistsException(Exception):
    pass


class InvalidParameterException(Exception):
    pass


class AnalysisTask(ABC):

    """
    An abstract class for performing analysis on a DataSet. Subclasses
    should implement the analysis to perform in the run_analysis() function.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        """Creates an AnalysisTask object that performs analysis on the
        specified DataSet.

        Args:
            dataSet: the DataSet to run analysis on.
            parameters: a dictionary containing parameters used to run the
                analysis.
            analysisName: specifies a unique identifier for this
                AnalysisTask. If analysisName is not set, the analysis name
                will default to the name of the class.
        """
        self.dataSet = dataSet
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = copy.deepcopy(parameters)

        if analysisName is None:
            self.analysisName = type(self).__name__
        else:
            self.analysisName = analysisName

        if 'merlin_version' not in self.parameters:
            self.parameters['merlin_version'] = merlin.version()
        else:
            if not merlin.is_compatible(self.parameters['merlin_version']):
                raise merlin.IncompatibleVersionException(
                    ('Analysis task %s has already been created by MERlin ' +
                     'version %s, which is incompatible with the current ' +
                     'MERlin version, %s')
                    % (self.analysisName, self.parameters['merlin_version'],
                       merlin.version()))

        self.parameters['module'] = type(self).__module__
        self.parameters['class'] = type(self).__name__

        if 'codebookNum' in self.parameters:
            self.codebookNum = self.parameters['codebookNum']

    def save(self, overwrite=False) -> None:
        """Save a copy of this AnalysisTask into the data set.

        Args:
            overwrite: flag indicating if an existing analysis task with the
                same name as this analysis task should be overwritten even
                if the specified parameters are different.
        Raises:
            AnalysisAlreadyExistsException: if an analysis task with the
                same name as this analysis task already exists in the
                data set with different parameters.
        """
        self.dataSet.save_analysis_task(self, overwrite)

    def run(self, overwrite=True) -> None:
        """Run this AnalysisTask.
        
        Upon completion of the analysis, this function informs the DataSet
        that analysis is complete.

        Args:
            overwrite: flag indicating if previous analysis from this
                analysis task should be overwritten.
        Raises:
            AnalysisAlreadyStartedException: if this analysis task is currently
                already running or if overwrite is not True and this analysis
                task has already completed or exited with an error.
        """
        logger = self.dataSet.get_logger(self)
        logger.info('Beginning ' + self.get_analysis_name())

        try:
            if self.is_running():
                raise AnalysisAlreadyStartedException(
                    'Unable to run %s since it is already running'
                    % self.analysisName)

            if overwrite:
                self._reset_analysis()

            if self.is_complete() or self.is_error():
                raise AnalysisAlreadyStartedException(
                    'Unable to run %s since it has already run'
                    % self.analysisName)

            self.dataSet.record_analysis_started(self)
            self._indicate_running()
            self._run_analysis()
            self.dataSet.record_analysis_complete(self)
            logger.info('Completed ' + self.get_analysis_name())
            self.dataSet.close_logger(self)
        except Exception as e:
            logger.exception(e)
            self.dataSet.record_analysis_error(self)
            self.dataSet.close_logger(self)
            raise e

    def _reset_analysis(self) -> None:
        """Remove files created by this analysis task and remove markers
        indicating that this analysis has been started, or has completed.

        This function should be overridden by subclasses so that they
        can delete the analysis files.
        """
        self.dataSet.reset_analysis_status(self)

    def _indicate_running(self) -> None:
        """A loop that regularly signals to the dataset that this analysis
        task is still running successfully. 

        Once this function is called, the dataset will be notified every 
        minute that this analysis is still running until the analysis
        completes.
        """
        if self.is_complete() or self.is_error():
            return

        self.dataSet.record_analysis_running(self)
        self.runTimer = threading.Timer(30, self._indicate_running)
        self.runTimer.daemon = True
        self.runTimer.start()

    @abstractmethod
    def _run_analysis(self) -> None:
        """Perform the analysis for this AnalysisTask.

        This function should be implemented in all subclasses with the
        logic to complete the analysis.
        """
        pass

    @abstractmethod
    def get_estimated_memory(self) -> float:
        """Get an estimate of how much memory is required for this
        AnalysisTask.

        Returns:
            a memory estimate in megabytes.
        """
        pass

    @abstractmethod
    def get_estimated_time(self) -> float:
        """Get an estimate for the amount of time required to complete
        this AnalysisTask.

        Returns:
            a time estimate in minutes.
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get the analysis tasks that must be completed before this
        analysis task can proceed.

        Returns:
            a list containing the names of the analysis tasks that 
                this analysis task depends on. If there are no dependencies,
                an empty list is returned.
        """
        pass

    def get_parameters(self):
        """Get the parameters for this analysis task.

        Returns:
            the parameter dictionary
        """
        return self.parameters

    def is_error(self):
        """Determines if an error has occurred while running this analysis
        
        Returns:
            True if the analysis is complete and otherwise False.
        """
        return self.dataSet.check_analysis_error(self)

    def is_complete(self):
        """Determines if this analysis has completed successfully
        
        Returns:
            True if the analysis is complete and otherwise False.
        """
        return self.dataSet.check_analysis_done(self)

    def is_started(self):
        """Determines if this analysis has started.
        
        Returns:
            True if the analysis has begun and otherwise False.
        """
        return self.dataSet.check_analysis_started(self)

    def is_running(self):
        """Determines if this analysis task is expected to be running,
        but has unexpectedly stopped for more than two minutes.
        """
        if not self.is_started():
            return False
        if self.is_complete():
            return False

        return not self.dataSet.is_analysis_idle(self)

    def get_analysis_name(self):
        """Get the name for this AnalysisTask.

        Returns:
            the name of this AnalysisTask
        """
        return self.analysisName

    def is_parallel(self):
        """Determine if this analysis task uses multiple cores."""
        return False


class InternallyParallelAnalysisTask(AnalysisTask):

    """
    An abstract class for analysis that can only be run in one part,
    but can internally be sped up using multiple processes. Subclasses
    should implement the analysis to perform in the run_analysis() function.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.coreCount = multiprocessing.cpu_count()

    def set_core_count(self, coreCount):
        """Set the number of parallel processes this analysis task is
        allowed to use.
        """
        self.coreCount = coreCount

    def is_parallel(self):
        return True 


class ParallelAnalysisTask(AnalysisTask):

    # TODO - this can be restructured so that AnalysisTask is instead a subclass
    # of ParallelAnalysisTask where fragment count is set to 1. This could
    # help remove some of the redundant code

    """
    An abstract class for analysis that can be run in multiple parts 
    independently. Subclasses should implement the analysis to perform in 
    the run_analysis() function
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def run(self, fragmentIndex: int=None, overwrite=True) -> None:
        """Run the specified index of this analysis task.

        If fragment index is not provided. All fragments for this analysis
        task are run in serial.

        Args:
            fragmentIndex: the index of the analysis fragment to run or None
                if all fragments should be run.
        """
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self.run(i, overwrite)
        else:
            logger = self.dataSet.get_logger(self, fragmentIndex)
            logger.info(
                'Beginning %s %i' % (self.get_analysis_name(), fragmentIndex))
            try:
                if self.is_running(fragmentIndex):
                    raise AnalysisAlreadyStartedException(
                        ('Unable to run %s fragment %i since it is already ' +
                         'running')
                        % (self.analysisName, fragmentIndex))

                if overwrite:
                    self._reset_analysis(fragmentIndex)

                if self.is_complete(fragmentIndex) \
                        or self.is_error(fragmentIndex):
                    raise AnalysisAlreadyStartedException(
                        'Unable to run %s fragment %i since it has already run'
                        % (self.analysisName, fragmentIndex))

                self.dataSet.record_analysis_started(self, fragmentIndex)
                self._indicate_running(fragmentIndex)
                self._run_analysis(fragmentIndex)
                self.dataSet.record_analysis_complete(self, fragmentIndex)
                logger.info('Completed %s %i'
                            % (self.get_analysis_name(), fragmentIndex))
                self.dataSet.close_logger(self, fragmentIndex)
            except Exception as e:
                logger.exception(e)
                self.dataSet.record_analysis_error(self, fragmentIndex)
                self.dataSet.close_logger(self, fragmentIndex)
                raise e

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def _reset_analysis(self, fragmentIndex: int=None) -> None:
        """Remove files created by this analysis task and remove markers
        indicating that this analysis has been started, or has completed.
        """
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self._reset_analysis(i)

        else:
            self.dataSet.reset_analysis_status(self, fragmentIndex)

    def _indicate_running(self, fragmentIndex: int) -> None:
        """A loop that regularly signals to the dataset that this analysis
        task is still running successfully. 

        Once this function is called, the dataset will be notified every 
        minute that this analysis is still running until the analysis
        completes.
        """
        if self.is_complete(fragmentIndex) or self.is_error(fragmentIndex):
            return

        self.dataSet.record_analysis_running(self, fragmentIndex)
        self.runTimer = threading.Timer(
                30, self._indicate_running, [fragmentIndex])
        self.runTimer.daemon = True
        self.runTimer.start()

    @abstractmethod
    def _run_analysis(self, fragmentIndex):
        pass

    def is_error(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if self.is_error(i):
                    return True 

            return False

        else:
            return self.dataSet.check_analysis_error(self, fragmentIndex)

    def is_complete(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if not self.is_complete(i):
                    return False

            return True

        else:
            return self.dataSet.check_analysis_done(self, fragmentIndex)

    def is_started(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if self.is_started(i):
                    return True 

            return False

        else:
            return self.dataSet.check_analysis_started(self, fragmentIndex)

    def is_running(self, fragmentIndex=None):
        if not self.is_started(fragmentIndex):
            return False
        if self.is_complete(fragmentIndex):
            return False

        return not self.dataSet.is_analysis_idle(self, fragmentIndex)

    def is_parallel(self):
        return True
