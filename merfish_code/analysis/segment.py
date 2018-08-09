from merfish_code.core import analysistask

class Segment(analysistask.ParallelAnalysisTask):

    '''
    An analysis task that determines the boundaries of features in the
    image data.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        #TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        #TODO - refine estimate
        return 5

    def run_analysis(self, fragmentIndex):
        pass
