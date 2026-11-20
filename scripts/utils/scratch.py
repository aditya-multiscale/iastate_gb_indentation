import numpy as np
from .rotation_matrix import *
from .fundamental_zone import *

def zone_classifier(euler_angles: Sequence[float]):
    """
    Classifies which texture component in the pole figure a given set of Euler angles is closest to
    Components are defined by specific Euler angles:
    - Cube component: (0, 0, 0)
    - Goss component: (np.pi/2, np.pi/2, np.pi/2)
    - Brass component: (np.pi, np.pi/2, np.pi/4)

    Returns:
        str: Name of the closest texture component.
    """