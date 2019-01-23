import numpy as np


class VolumeFeature(object):

    """
    A feature is a collection of contiguous voxels.
    """

    def __init__(self, pixelList: np.ndarray, fov: int) -> None:
        """Create a new feature specified by a list of pixels

        Args:
            pixelList: a list of pixels that are included in this feature.
                Each pixel is specified by x, y, and z integers.
            fov: the index of the field of view that this feature belongs to.
                The pixel list specifies pixel in the local fov reference
                frame.
        """

        self._pixelList = pixelList.copy().astype(np.uint32)
        self._fov = fov

    def get_fov(self):
        return self._fov

    def get_pixels(self):
        return self._pixelList

    def overlaps_in_fov(self, inFeature) -> bool:
        """Determine if this feature overlaps with the specified feature.

        This function only checks for overlap between features associated
        with the same field of view. It is possible that two features
        from different field of views overlap because of their global
        arrangement, but this function will not detect that overlap.

        Args:
            inFeature: the feature to check for overlap with
        Returns:
            True if this feature and inFeature are in the same field of view
                and contains pixels that are also in inFeature,
                otherwise False.
        """
        if self.get_fov() != inFeature.get_fov():
            return False

        for p1 in self.get_pixels():
            for p2 in inFeature.get_pixels():
                if np.array_equal(p1, p2):
                    return True

        return False


