import numpy as np

from merlin.plots._base import PlotMetadata


class TestPlotMetadata(PlotMetadata):

    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)
        self.testTask = self._taskDict['test_task']
        self.completeFragments = [False]*self.testTask.fragment_count()
        self.meanValues = np.zeros(self.testTask.fragment_count())

    @classmethod
    def metadata_name(cls) -> str:
        return cls.__module__ + '.' + cls.__class__

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
