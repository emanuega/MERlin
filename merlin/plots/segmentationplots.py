from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot


class SegmentationBoundaryPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'segment_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        featureDB = inputTasks['segment_task'].get_feature_database()
        features = featureDB.read_features()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        if len(features) == 0:
            return fig
        zPosition = 0
        if len(features[0].get_boundaries()) > 1:
            zPosition = int(len(features[0].get_boundaries())/2)

        for f in features:
            featureSingleZ = f.get_boundaries()[zPosition]
            plt.plot(featureSingleZ.exterior.coords.xy[0],
                     featureSingleZ.exterior.coords.xy[1])

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Segmentation boundaries')
        return fig