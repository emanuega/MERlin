from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot


class CellBoundaryPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'segment_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        cellBoundaries = inputTasks['segment_task'].get_cell_boundaries()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        def plot_cell_boundary(boundary):
            ax.plot([x[0] for x in boundary], [x[1] for x in boundary])
        for b in cellBoundaries:
            plot_cell_boundary(b)

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Cell boundaries')
        return fig