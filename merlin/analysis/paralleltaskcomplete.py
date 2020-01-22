from merlin.core import analysistask


class ParallelTaskComplete(analysistask.AnalysisTask):
    """
    A task to simplify snakemake construction, there is no need to explicitly
    invoke this task in analysis files, it is inferred by the snakewriter.
    If running outside snakemake this class is not required.
    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 100

    def get_dependencies(self):
        # The dependency is inferred by the snakewriter
        return [self.parameters['dependent_task']]

    def get_estimated_time(self):
        return 1

    def _run_analysis(self):
        dependentTask = self.dataSet.load_analysis_task(
            self.parameters['dependent_task'])
        dependentTask.is_complete()
