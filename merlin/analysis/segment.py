import cv2
import numpy as np
from scipy.ndimage import morphology
from starfish.image._segmentation import watershed
from starfish import stats

from merlin.core import analysistask

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
        return [self.parameters['warp_task'], \
                self.parameters['global_align_task']]


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

        return np.array(
                [np.array([x[0] for x in extract_contours(inputImage, i)[0]]) \
                for i in uniqueLabels])

    def _transform_contours(self, contours, transform):
        '''Transforms the coordinates in the contours based on the 
        provided transformation.

        Args:
            contours - a n x 2 numpy array specifying the coordinates of the n
                points in the contour
            transform - a 3 x 3 numpy array specifying the transformation 
                matrix
        '''
        reshapedContours = np.reshape(contours, 
                (1, contours.shape[0], 2)).astype(np.float)
        transformedContours = cv2.transform(
                reshapedContours, transform)[0,:,:2]

        return transformedContours


    def get_cell_boundaries(self):
        boundaryList = []
        for f in self.dataSet.get_fovs():
            currentBoundaries = self.dataSet.load_analysis_result(
                    'cell_boundaries', 'SegmentCells', resultIndex=f)
            boundaryList += [x for x in currentBoundaries]

        return boundaryList



    def run_analysis(self, fragmentIndex):
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        #TODO - extend to 3D
        #TODO - this does not do well with image boundaries. Cell 
        #boundaries are not traced past the edge of the field of 
        #view 
        nucleusImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.nucleusIndex, 0),
                (int(35), int(35)), 8)
        cellImage = cv2.GaussianBlur(warpTask.get_aligned_image(
                fragmentIndex, self.cellIndex, 0),
                (int(35), int(35)), 8)

        w = watershed._WatershedSegmenter(nucleusImage, cellImage)
        labels = w.segment(
                self.nucleusThreshold, self.cellThreshold, [10, 100000])

        cellContours = self._label_to_regions(labels)
        transformation = globalTask.fov_to_global_transform(fragmentIndex)
        transformedContours = np.array(
                [self._transform_contours(x, transformation) \
                for x in cellContours])

        self.dataSet.save_analysis_result(
                transformedContours, 'cell_boundaries',
                self.get_analysis_name(), resultIndex=fragmentIndex)
