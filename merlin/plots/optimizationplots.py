import seaborn
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot


class OptimizationScaleFactorsPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'optimize_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(5, 5))
        seaborn.heatmap(
            inputTasks['optimize_task'].get_scale_factor_history())
        plt.xlabel('Bit index')
        plt.ylabel('Iteration number')
        plt.title('Scale factor optimization history')
        return fig


class ScaleFactorVsBitNumberPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'optimize_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        optimizeTask = inputTasks['optimize_task']
        codebook = optimizeTask.get_codebook()
        dataOrganization = optimizeTask.dataSet.get_data_organization()
        colors = [dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()]

        scaleFactors = optimizeTask.get_scale_factors()
        scaleFactorsByColor = {c: [] for c in set(colors)}
        for i, s in enumerate(scaleFactors):
            scaleFactorsByColor[colors[i]].append((i, s))

        fig = plt.figure(figsize=(5, 5))
        for c, d in scaleFactorsByColor.items():
            plt.plot([x[0] for x in d], [x[1] for x in d], 'o')

        plt.legend(scaleFactorsByColor.keys())
        plt.ylim(bottom=0)
        plt.xlabel('Bit index')
        plt.ylabel('Scale factor magnitude')
        plt.title('Scale factor magnitude vs bit index')
        return fig


class OptimizationBarcodeCountsPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'optimize_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        fig = plt.figure(figsize=(5, 5))
        seaborn.heatmap(
            inputTasks['optimize_task'].get_barcode_count_history())
        plt.xlabel('Barcode index')
        plt.ylabel('Iteration number')
        plt.title('Barcode counts optimization history')
        return fig
