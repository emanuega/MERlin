import numpy as np
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot
from merlin.plots._base import PlotMetadata


class TestPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'test_task': 'all'}

    def get_required_metadata(self):
        return [TestPlotMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(inputMetadata['testplots/TestPlotMetadata'].get_mean_values(),
                 'x')
        return fig


class TestPlotMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.testTask = self._taskDict['test_task']
        self.completeFragments = [False]*self.testTask.fragment_count()
        self.meanValues = np.zeros(self.testTask.fragment_count())

    def get_mean_values(self) -> np.ndarray:
        return self.meanValues

    def update(self) -> None:
        testTask = self._taskDict['test_task']

        for i in range(testTask.fragment_count()):
            if not self.completeFragments[i] and testTask.is_complete(i):
                self.meanValues[i] = np.mean(self.testTask.get_random_result(i))
                self.completeFragments[i] = True

    def is_complete(self) -> bool:
        return all(self.completeFragments)
