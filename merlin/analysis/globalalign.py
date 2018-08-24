from abc import abstractmethod
import numpy as np

from merlin.core import analysistask

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
    def fov_to_global_transform(self, fov):
        '''Calculates the transformation matrix for an affine transformation
        that transforms the fov coordinates to global coordinates.
        '''
        pass

    @abstractmethod
    def get_global_extent(self):
        '''Get the extent of the global coordinate system.

        Returns:
            A tuple where the first two indexes correspond to the minimum
            and x and y extents and the last two indexes correspond to the
            maximum x and y extents. All are is units of microns.
        '''
        pass

class SimpleGlobalAlignment(GlobalAlignment):

    '''A global alignment that uses the theoretical stage positions in 
    order to determine the relative positions of each field of view.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 0

    def run_analysis(self):
        #This analysis task does not need computation
        pass

    def get_dependencies(self):
        return []

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        fovStart = self.dataSet.get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        return (fovStart[0] + fovCoordinates[0]*micronsPerPixel, \
                fovStart[1] + fovCoordinates[1]*micronsPerPixel)
      
    def fov_to_global_transform(self, fov):
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        globalStart = self.fov_coordinates_to_global(fov, (0,0))

        return np.float32([[micronsPerPixel, 0, globalStart[0]], \
                           [0, micronsPerPixel, globalStart[1]], \
                           [0, 0, 1]])

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

        return (minX, minY, maxX, maxY)

class CorrelationGlobalAlignment(GlobalAlignment):

    '''A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    '''

    #TODO - implement.  I expect rotation might be needed for this alignment
    #if the x-y orientation of the camera is not perfectly oriented with 
    #the microscope stage

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 60

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        raise NotImplementedError

    def fov_to_global_transform(self, fov):
        raise NotImplementedError

    def get_global_extent(self):
        raise NotImplementedError

    def _calculate_overlap_area(self, x1, y1, x2, y2, width, height):
        '''Calculates the overlapping area between two rectangles with 
        equal dimensions.
        '''

        dx = min(x1+width, x2+width) - max(x1, x2)
        dy = min(y1+height, y2+height) - max(y1,y2)

        if dx>0 and dy>0:
            return dx*dy
        else:
            return 0

    def _get_overlapping_regions(self, fov, minArea=2000):
        '''Get a list of all the fovs that overlap with the specified fov.
        '''
        positions = self.dataSet.get_stage_positions()
        pixelToMicron = self.dataSet.get_microns_per_pixel()
        fovMicrons = [x*pixelToMicron \
                for x in self.dataSet.get_image_dimensions()]
        fovPosition = positions.loc[fov]
        overlapAreas = [i for i,p in positions.iterrows() \
                if self._calculate_overlap_area(
                    p['X'], p['Y'], fovPosition['X'], fovPosition['Y'],
                    fovMicrons[0], fovMicrons[1]) > minArea and i!=fov]

        return overlapAreas

    def run_analysis(self):
        fov1 = self.dataSet.get_fiducial_image(0, 0)
        fov2 = self.dataSet.get_fiducial_image(0, 1)

        return fov1, fov2
