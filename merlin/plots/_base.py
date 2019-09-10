import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt

from merlin.core import analysistask


class AbstractPlot(ABC):

    """
    A base class for generating a plot of the analysis results. Each plot
    should inherit from this class.
    """

    def __init__(self, analysisTask: analysistask.AnalysisTask):
        """ Create a new AbstractPlot

        Args:
            analysisTask: the analysisTask where the plot should be saved.
        """
        self._analysisTask = analysisTask

    def figure_name(self) -> str:
        """ Get the name for identifying this figure.

        Returns: the name of this figure
        """
        return type(self).__name__

    @abstractmethod
    def get_required_tasks(self) -> Dict[str, Tuple[type]]:
        """ Get the tasks that are required to be complete prior to
        generating this plot.

        Returns: A dictionary of the types of tasks as keys and a tuple
            of the accepted classes as values. The keys can include
            decode_task, filter_task, optimize_task, segment_task,
            sum_task, partition_task, and/or global_align_task. If all classes
            of the specified type are allowed, the value should be 'all'. If
            no tasks are required then an empty dictionary should be returned.
        """
        pass

    @abstractmethod
    def get_required_metadata(self) -> List[object]:
        """ Get the plot metadata that is required to generate this plot.

        Returns: A list of class references for the metadata
            objects that are required for this task.
        """
        pass

    @abstractmethod
    def _generate_plot(self, inputTasks: Dict[str, analysistask.AnalysisTask],
                       inputMetadata: Dict[str, 'PlotMetadata']) -> plt.Figure:
        """ Generate the plot.

        This function should be implemented in all subclasses and the generated
        figure handle should be returned.

        Args:
            inputTasks: A dictionary of the input tasks to use to generate the
                plot. Each analysis task is indexed by a string indicating
                the task type as in get_required_tasks.
            inputMetadata: A dictionary of the input metadata for generating
                this plot. Each metadata object is indexed by the name of the
                metadata.
        Returns: the figure handle to the newly generated figure
        """
        pass

    def is_relevant(self, inputTasks: Dict[str, analysistask.AnalysisTask]
                    ) -> bool:
        """ Determine if this plot is relevant given the analysis tasks
        provided.

        Args:
            inputTasks: A dictionary of the analysis tasks indexed with
                strings indicating the task type as in get_required_tasks
        Returns: True if this plot can be generated using the provided
            analysis tasks and false otherwise.
        """
        for rTask, rTypes in self.get_required_tasks().items():
            if rTask not in inputTasks:
                return False
            if rTypes != 'all' \
                    and not isinstance(inputTasks[rTask], rTypes):
                return False
        return True

    def is_ready(self, completeTasks: List[str],
                 completeMetadata: List[str]) -> bool:
        """ Determine if all requirements for generating this plot are
        satisfied.

        Args:
            completeTasks: A list of the types of tasks that are complete.
                The list can contain the same strings as in get_required_tasks
            completeMetadata: A list of the metadata that has been generated.
        Returns: True if all required tasks and all required metadata
            is complete
        """
        return all([t in completeTasks for t in self.get_required_tasks()])\
            and all([m.metadata_name() in completeMetadata
                     for m in self.get_required_metadata()])

    def is_complete(self) -> bool:
        """ Determine if this plot has been generated.

        Returns: True if this plot has been generated and otherwise false.
        """
        return self._analysisTask.dataSet.figure_exists(
            self._analysisTask, self.figure_name(), type(self).__module__)

    def plot(self, inputTasks: Dict[str, analysistask.AnalysisTask],
             inputMetadata: Dict[str, 'PlotMetadata']) -> None:
        """ Generate this plot and save it within the analysis task.

        If the plot is not relevant for the types of analysis tasks passed,
        then the function will return without generating any plot.

        Args:
            inputTasks: A dictionary of the input tasks to use to generate the
                plot. Each analysis task is indexed by a string indicating
                the task type as in get_required_tasks.
            inputMetadata: A dictionary of the input metadata for generating
                this plot. Each metadata object is indexed by the name of the
                metadata.
        """
        if not self.is_relevant(inputTasks):
            return
        f = self._generate_plot(inputTasks, inputMetadata)
        self._analysisTask.dataSet.save_figure(
                self._analysisTask, f, self.figure_name(), 
                type(self).__module__.split('.')[-1])
        plt.close(f)


class PlotMetadata(ABC):

    def __init__(self, analysisTask: analysistask.AnalysisTask,
                 taskDict: Dict[str, analysistask.AnalysisTask]):
        """ Create a new metadata object.

        Args:
            analysisTask: the analysisTask where the metadata should be saved.
            taskDict: a dictionary containing the analysis tasks to use
                to generate the metadata indexed by the type of task as a
                string as in get_required_tasks
        """
        self._analysisTask = analysisTask
        self._taskDict = taskDict

    @classmethod
    def metadata_name(cls) -> str:
        return cls.__module__.split('.')[-1] + '.' + cls.__name__

    def _load_numpy_metadata(self, resultName: str,
                             defaultValue: np.ndarray=None) -> np.ndarray:
        """ Convenience method for reading a result created by this metadata
        from the dataset.

        Args:
            resultName: the name of the metadata result
            defaultValue: the value to return if the metadata is not found
        Returns: a numpy array with the result or defaultValue if an IOError is
            raised while reading the metadata
        """
        try:
            return self._analysisTask.dataSet.load_numpy_analysis_result(
                resultName, self._analysisTask,
                subdirectory=self.metadata_name())
        except IOError:
            return defaultValue

    def _save_numpy_metadata(self, result: np.ndarray, resultName: str) -> None:
        """ Convenience method for saving a result created by this metadata
        from the dataset.

        Args:
            result: the numpy array to save
            resultName: the name of the metadata result
        """
        self._analysisTask.dataSet.save_numpy_analysis_result(
            result, resultName, self._analysisTask,
            subdirectory=self.metadata_name())

    @abstractmethod
    def update(self) -> None:
        """ Update this metadata with the latest analysis results.

        This method should be implemented in all subclasses and implementations
        should not wait for additional data to become available. They should
        only update the metadata as much as possible with the data that is ready
        when the function is called and should not wait for additional
        analysis to complete.
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """ Determine if this metadata is complete.

        Returns: True if the metadata is complete or False if additional
            computation is necessary
        """
        pass
