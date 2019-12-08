from abc import abstractmethod
import numpy as np
from typing import Tuple
from typing import List

from merlin.core import analysistask


class GlobalAlignment(analysistask.AnalysisTask):

    """
    An abstract analysis task that determines the relative position of
    different field of views relative to each other in order to construct
    a global alignment.
    """

    @abstractmethod
    def fov_coordinates_to_global(
            self, fov: int, fovCoordinates: Tuple[float, float]) \
            -> Tuple[float, float]:
        """Calculates the global coordinates based on the local coordinates
        in the specified field of view.

        Args:
            fov: the fov where the coordinates are measured
            fovCoordinates: a tuple containing the x and y coordinates
                or z, x, and y coordinates (in pixels) in the specified fov.
        Returns:
            A tuple containing the global x and y coordinates or
            z, x, and y coordinates (in microns)
        """
        pass

    @abstractmethod
    def global_coordinates_to_fov(
            self, fov: int, globalCoordinates: List[Tuple[float, float]]) \
            -> List[Tuple[float, float]]:
        """Calculates the fov pixel coordinates for a list of global coordinates
        in the specified field of view.

        Args:
            fov: the fov where the coordinates are measured
            globalCoordinates: a list of tuples containing the x and
                               y coordinates (in pixels) in the specified fov.
        Returns:
            A list of tuples containing the global x and y coordinates
            (in microns)
        """
        pass
        # TODO this can be updated to take either a list or a single coordinate
        # and to convert z position

    @abstractmethod
    def fov_to_global_transform(self, fov: int) -> np.ndarray:
        """Calculates the transformation matrix for an affine transformation
        that transforms the fov coordinates to global coordinates.

        Args:
            fov: the fov to calculate the transformation
        Returns:
            a numpy array containing the transformation matrix
        """
        pass

    @abstractmethod
    def get_global_extent(self) -> Tuple[float, float, float, float]:
        """Get the extent of the global coordinate system.

        Returns:
            a tuple where the first two indexes correspond to the minimum
            and x and y extents and the last two indexes correspond to the
            maximum x and y extents. All are in units of microns.
        """
        pass

    @abstractmethod
    def get_fov_boxes(self) -> List:
        """
        Creates a list of shapely boxes for each fov containing the global
        coordinates as the box coordinates.

        Returns:
            A list of shapely boxes
        """
        pass



class SimpleGlobalAlignment(GlobalAlignment):

    """A global alignment that uses the theoretical stage positions in
    order to determine the relative positions of each field of view.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 0

    def _run_analysis(self):
        # This analysis task does not need computation
        pass

    def get_dependencies(self):
        return []

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        fovStart = self.dataSet.get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        if len(fovCoordinates) == 2:
            return (fovStart[0] + fovCoordinates[0]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[1]*micronsPerPixel)
        elif len(fovCoordinates) == 3:
            zPositions = self.dataSet.get_z_positions()
            return (np.interp(fovCoordinates[0], np.arange(len(zPositions)),
                              zPositions),
                    fovStart[0] + fovCoordinates[1]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[2]*micronsPerPixel)

    def fov_global_extent(self, fov: int) -> List[float]:

        """
        Returns the global extent of an fov, output interleaved as
        xmin, ymin, xmax, ymax

        Args:
            fov: the fov of interest
        Returns:
            a list of four floats, representing the xmin, xmax, ymin, ymax
        """

        return [x for y in (self.fov_coordinates_to_global(fov, (0, 0)),
                            self.fov_coordinates_to_global(fov, (2048, 2048)))
                for x in y]

    def global_coordinates_to_fov(self, fov, globalCoordinates):
        tform = np.linalg.inv(self.fov_to_global_transform(fov))

        def convert_coordinate(coordinateIn):
            coords = np.array([coordinateIn[0], coordinateIn[1], 1])
            return np.matmul(tform, coords).astype(int)[:2]
        pixels = [convert_coordinate(x) for x in globalCoordinates]
        return pixels

    def fov_to_global_transform(self, fov):
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        globalStart = self.fov_coordinates_to_global(fov, (0, 0))

        return np.float32([[micronsPerPixel, 0, globalStart[0]],
                           [0, micronsPerPixel, globalStart[1]],
                           [0, 0, 1]])

    def get_global_extent(self):
        fovSize = self.dataSet.get_image_dimensions()
        fovBounds = [self.fov_coordinates_to_global(x, (0, 0))
                     for x in self.dataSet.get_fovs()] + \
                    [self.fov_coordinates_to_global(x, fovSize)
                     for x in self.dataSet.get_fovs()]

        minX = np.min([x[0] for x in fovBounds])
        maxX = np.max([x[0] for x in fovBounds])
        minY = np.min([x[1] for x in fovBounds])
        maxY = np.max([x[1] for x in fovBounds])

        return minX, minY, maxX, maxY

    def get_fov_boxes(self) -> List:
        fovs = self.dataSet.get_fovs()
        coords = [self.fov_global_extent(f) for f in fovs]
        coordsDF = pandas.DataFrame(coords,
                                    columns=['minx', 'miny', 'maxx', 'maxy'],
                                    index=fovs)
        boxes = [geometry.box(x[0], x[1], x[2], x[3]) for x in
                 coordsDF.loc[:, ['minx', 'miny', 'maxx', 'maxy']].values]

        return boxes


class CorrelationGlobalAlignment(GlobalAlignment):

    """
    A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    """

    # TODO - implement.  I expect rotation might be needed for this alignment
    # if the x-y orientation of the camera is not perfectly oriented with
    # the microscope stage

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

    @staticmethod
    def _calculate_overlap_area(x1, y1, x2, y2, width, height):
        """Calculates the overlapping area between two rectangles with
        equal dimensions.
        """

        dx = min(x1+width, x2+width) - max(x1, x2)
        dy = min(y1+height, y2+height) - max(y1, y2)

        if dx > 0 and dy > 0:
            return dx*dy
        else:
            return 0

    def _get_overlapping_regions(self, fov: int, minArea: int = 2000):
        """Get a list of all the fovs that overlap with the specified fov.
        """
        positions = self.dataSet.get_stage_positions()
        pixelToMicron = self.dataSet.get_microns_per_pixel()
        fovMicrons = [x*pixelToMicron
                      for x in self.dataSet.get_image_dimensions()]
        fovPosition = positions.loc[fov]
        overlapAreas = [i for i, p in positions.iterrows()
                        if self._calculate_overlap_area(
                p['X'], p['Y'], fovPosition['X'], fovPosition['Y'],
                fovMicrons[0], fovMicrons[1]) > minArea and i != fov]

        return overlapAreas

    def _run_analysis(self):
        fov1 = self.dataSet.get_fiducial_image(0, 0)
        fov2 = self.dataSet.get_fiducial_image(0, 1)

        return fov1, fov2
