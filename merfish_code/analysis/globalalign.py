from abc import abstractmethod

from merfish_code.core import analysistask

class GlobalAlignment(analysistask.AnalysisTask):

    '''An abstract analysis task that determines the relative position of 
    different field of views relative to each other in order to construct
    a global alignment.
    '''

    @abstractmethod
    def fov_coordinates_to_global(fov, fovCoordinates): 
        '''Calculates the global coordinates based on the local coordinates
        in the specified field of view.

        Args:
            fov: The fov corresponding where the coordinates are measured
            fovCoordinates: A tuple containing the x and y coordinates
                (in pixels) in the specified fov.
        Returns:
            A tuple contaning the global x and y coordinates (in microns)
        '''
        pass
            

class SimpleGlobalAlignment(GlobalAlignment):

    '''A global alignment that uses the theoretical stage positions in 
    order to determine the relative positions of each field of view.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def run_analysis(self):
        #This analysis task does not need computation
        pass

    def fov_coordinates_to_global(fov, fovCoordinates):
        fovStart = self.dataSet.get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        return (fovStart[0] + fovCoordinates[0]*micronsPerPixel, \
                fovStart[1] + fovCoordinates[1]*micornsPerPixel)
        

class CorrelationGlobalAlignment(GlobalAlignment):

    '''A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    '''

    #TODO - implement.  I expect rotation might be needed for this alignment

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        raise NotImplementedError

