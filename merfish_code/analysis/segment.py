import cv2
import numpy as np
from scipy.ndimage import morphology
from starfish.image._segmentation import watershed
from starfish import stats

from merfish_code.core import analysistask

class SegmentCells(analysistask.ParallelAnalysisTask):

    '''
    An analysis task that determines the boundaries of features in the
    image data.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.nucleusThreshold = parameters.get('nucleus_threshold', 0.41)
        self.cellThreshold = parameters.get('cell_thershold', 0.08)
        self.nucleusIndex = parameters.get('nucleus_index', 17)
        self.cellIndex = parameters.get('cell_index', 16)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        #TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        #TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]


    def _label_to_regions(self, inputImage):
        uniqueLabels = sorted(np.unique(inputImage))[1:]

        def extract_contours(labelImage, label):
            filledImage = morphology.binary_fill_holes(
                    labelImage==label)
            im2, contours, hierarchy = cv2.findContours(
                    filledImage.astype(np.uint8),
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_TC89_KCOS)
            return contours

        return [extract_contours(inputImage, i)[0] for i in uniqueLabels]

    def run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        #TODO - extend to 3D
        nucleusImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.nucleusIndex, 0),
                (int(35), int(35)), 8)
        cellImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.cellIndex, 0),
                (int(35), int(35)), 8)

        w = watershed._WatershedSegmenter(nucleusImage, cellImage)
        labels = w.segment(
                self.nucleusThreshold, self.cellThreshold, [10, 100000])

        return cellImage, self._label_to_regions(labels)
