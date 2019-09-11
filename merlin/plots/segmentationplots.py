from matplotlib import pyplot as plt
import numpy as np

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

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        if len(features) == 0:
            return fig

        zPosition = 0
        if len(features[0].get_boundaries()) > 1:
            zPosition = int(len(features[0].get_boundaries())/2)

        featuresSingleZ = [feature.get_boundaries()[int(zPosition)]
                           for feature in features]
        featuresSingleZ = [x for y in featuresSingleZ for x in y]
        allCoords = [[feature.exterior.coords.xy[0].tolist(),
                      feature.exterior.coords.xy[1].tolist()]
                     for feature in featuresSingleZ]
        allCoords = [x for y in allCoords for x in y]
        plt.plot(*allCoords)

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Segmentation boundaries')
        return fig
