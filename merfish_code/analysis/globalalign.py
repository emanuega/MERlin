from abc import abstractmethod
import numpy as np

from merfish_code.core import analysistask

class GlobalAlignment(analysistask.AnalysisTask):

    '''An abstract analysis task that determines the relative position of 
    different field of views relative to each other in order to construct
    a global alignment.
    '''

    @abstractmethod
    def fov_coordinates_to_global(self, fov, fovCoordinates): 
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

    @abstractmethod
    def get_global_extent(self):
        '''Get the extent of the global coordinate system.

        Returns:
            A tuple where the first two indexes correspond to the minimum
            and maximum x extents and the last two indexes correspond to the
            minimum and maximum y extends. All are is units of microns.
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

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        fovStart = self.dataSet.get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        return (fovStart[0] + fovCoordinates[0]*micronsPerPixel, \
                fovStart[1] + fovCoordinates[1]*micronsPerPixel)
       
    def get_global_extent(self):
        fovSize = self.dataSet.get_image_dimensions()
        fovBounds = [self.fov_coordinates_to_global(x, (0, 0)) \
                for x in self.dataSet.get_fovs()] \
                + [self.fov_coordinates_to_global(x, fovSize) \
                for x in self.dataSet.get_fovs()]

        minX = np.min([x[0] for x in fovBounds])
        maxX = np.max([x[0] for x in fovBounds])
        minY = np.min([x[1] for x in fovBounds])
        maxY = np.max([x[1] for x in fovBounds])

        return (minX, maxX, minY, maxY)

class CorrelationGlobalAlignment(GlobalAlignment):

    '''A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    '''

    #TODO - implement.  I expect rotation might be needed for this alignment

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        raise NotImplementedError

