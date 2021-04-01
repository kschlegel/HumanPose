from .skeleton2d import Skeleton2D
from .skeleton3d import Skeleton3D


class SkeletonND(Skeleton2D, Skeleton3D):
    """
    This class is to combine functionality for drawing in 2D and 3D.

    No work needs to be done here, just inheriting from each of the
    classes that handle drawing in N dimensions.
    """
    pass
