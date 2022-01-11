class Intersection():
    """
    Find the intersection of two differrent sets of landmarks.
    Different pose datasets and poise estimation systems have different sets of
    landmarks. In order to facilitate comparing between different datasets/
    systems this class finds the intersection of the two and provides methods
    to easily subset and reorder keypoints to obtain two directly comparable
    keypoint arrays. The new array will be ordered based on the order of the
    target set.
    Attributes
    ----------
    landmarks: list of strings, defining the landmarks contained in the
    intersection and their order. Exactly equivalent to the original landmark
    attributes of datasets used to compute the intersection.
    Methods
    -------
    source(keypoints)
        Subset and reorder the original keypoint list.
    destination(keypoints)
        Subset the destination keypoint list.
    """
    def __init__(self, src, dest):
        """
        Parameters
        ----------
        src : list of strings
            list of landmark names defining their order in the source data
        dest : list of strings
            list of landmark names defining their order in the destination data
        """
        self._src_vec = []
        self._dest_vec = []
        self.landmarks = []
        for index, landmark in enumerate(dest):
            if landmark in src:
                self._src_vec.append(src.index(landmark))
                self._dest_vec.append(index)
                self.landmarks.append(landmark)

    def source(self, keypoints):
        """ Subset and reorder the original keypoint list. """
        if len(keypoints.shape) == 2:
            return keypoints[self._src_vec]
        elif len(keypoints.shape) == 3:
            return keypoints[:, self._src_vec]

    def destination(self, keypoints):
        """ Subset destination keypoint list. """
        if len(keypoints.shape) == 2:
            return keypoints[self._dest_vec]
        elif len(keypoints.shape) == 3:
            return keypoints[:, self._dest_vec]
