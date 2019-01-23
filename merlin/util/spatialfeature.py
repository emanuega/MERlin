import numpy as np
import uuid
import cv2
from typing import List
from typing import Tuple
from shapely import geometry


class SpatialFeature(object):

    """
    A spatial feature is a collection of contiguous voxels.
    """

    def __init__(self, boundaryList: List[List[geometry.Polygon]], fov: int,
                 uniqueID=None) -> None:
        """Create a new feature specified by a list of pixels

        Args:
            boundaryList: a least of boundaries that define this feature.
                The first index of the list corresponds with the z index.
                The second index corresponds with the index of the shape since
                some regions might split in some z indexes.
            fov: the index of the field of view that this feature belongs to.
                The pixel list specifies pixel in the local fov reference
                frame.
        """
        self._boundaryList = boundaryList
        self._fov = fov

        if uniqueID is None:
            self._uniqueID = uuid.uuid4()
        else:
            self._uniqueID = uniqueID

    @staticmethod
    def feature_from_label_matrix(labelMatrix: np.ndarray, fov: int,
                                  transformationMatrix: np.ndarray=None):
        """Generate a new feature from the specified label matrix.

        Args:
            labelMatrix: a 3d matrix indicating the z, x, y position
                of voxels that contain the feature. Voxels corresponding
                to the feature have a value of True while voxels outside of the
                feature should have a value of False.
            fov: the index of the field of view corresponding to the
                label matrix.
            transformationMatrix: a 3x3 numpy array specifying the
                transformation from fov to global coordinates. If None,
                the feature coordinates are not transformed.

        Returns: the new feature
        """

        boundaries = [SpatialFeature._extract_boundaries(x)
                      for x in labelMatrix]

        if transformationMatrix is not None:
            boundaries = [SpatialFeature._transform_boundaries(
                x, transformationMatrix) for x in boundaries]

        return SpatialFeature([[geometry.Polygon(x) for x in b if len(x) > 2]
                               for b in boundaries], fov)

    @staticmethod
    def _extract_boundaries(labelMatrix: np.ndarray) -> List[np.ndarray]:
        """Determine the boundaries of the feature indicated in the
        label matrix.

        Args:
            labelMatrix: a 2 dimensional numpy array indicating the x, y
                position of pixels that contain the feature.
        Returns: a list of n x 2 numpy arrays indicating the x, y coordinates
            of the boundaries where n is the number of boundary coordinates
        """
        _, boundaries, _ = cv2.findContours(
            labelMatrix.copy().astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        return [np.array([[x[0][0], x[0][1]] for x in y]) for y in boundaries]

    @staticmethod
    def _transform_boundaries(
            boundaries: List[np.ndarray],
            transformationMatrix: np.ndarray) -> List[np.ndarray]:

        transformedList = []
        for b in boundaries:
            reshapedBoundaries = np.reshape(
                b, (1, b.shape[0], 2)).astype(np.float)
            transformedBoundaries = cv2.transform(
                reshapedBoundaries, transformationMatrix)[0, :, :2]
            transformedList.append(transformedBoundaries)

        return transformedList

    def get_fov(self) -> int:
        return self._fov

    def get_boundaries(self) -> List[List[geometry.Polygon]]:
        return self._boundaryList

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the 2d box that contains all boundaries in all z plans of this
        feature.

        Returns:
            a tuple containing (x1, y1, x2, y2) coordinates of the bounding box
        """
        boundarySet = []
        for f in self.get_boundaries():
            for b in f:
                boundarySet.append(b)

        multiPolygon = geometry.MultiPolygon(boundarySet)
        return multiPolygon.bounds

    def is_contained_in(self, inFeature) -> bool:
        """Determine if the boundary of this feature is contained in the
        boundary of the specified feature.

        Args:
            inFeature: the feature to check for overlap with
        Returns:
            True if inFeature contains pixels that are within inFeature,
                otherwise False. This returns false if inFeature only shares
                a boundary with this feature.
        """
        raise NotImplementedError


class SpatialFeatureDB(object):

    """A database for storing spatial features"""

    def __init__(self, dataSet, analysisTask):
        self._dataSet = dataSet
        self._analysisTask = analysisTask

    def _get_masterDB(self):
        return self._dataSet.get_database_engine(self._analysisTask)

    def write_features(self, feature: List[SpatialFeature], fov=None) -> None:
        pass


    def get_features(self, fov=None):
        pass
