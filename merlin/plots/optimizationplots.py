import seaborn
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot


class OptimizationScaleFactorsPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'optimization_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(5, 5))
        seaborn.heatmap(
            inputTasks['optimization_task'].get_scale_factor_history())
        plt.xlabel('Bit index')
        plt.ylabel('Iteration number')
        plt.title('Scale factor optimization history')
        return fig


class OptimizationBarcodeCountsPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'optimization_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(5, 5))
        seaborn.heatmap(
            inputTasks['optimization_task'].get_barcode_count_history())
        plt.xlabel('Barcode index')
        plt.ylabel('Iteration number')
        plt.title('Barcode counts optimization history')
        return fig
