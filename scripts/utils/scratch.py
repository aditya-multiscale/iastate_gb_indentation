import numpy as np
from .rotation_matrix import *
from .fundamental_zone import *
from typing import Sequence

def zone_classifier(euler_angles: Sequence[float], degrees: bool = False):
    """
    Classifies which texture component in the pole figure a given set of Euler angles is closest to
    Components are defined by specific Euler angles:
    - Cube component: (0, 0, 0)
    - Goss component: (1, 1, 0)
    - Brass component: (1, 1, 1)

    Args:
        euler_angles (Sequence[float]): Euler angles to classify.
        degrees (bool): Whether the input Euler angles are in degrees. Default is False.

    Returns:
        str: Name of the closest texture component.
    """

    # Convert to radians if the input is in degrees
    if degrees:
        euler_angles = [np.deg2rad(angle) for angle in euler_angles]

    # Project the Euler angles into the fundamental zone first
    phi1, Phi, phi2 = project_to_fundamental_zone(euler_angles)

    # Find z-direction in crystalline axes
    rot = rotation_matrix(phi1, Phi, phi2)

    z_crystal = rot.T @ np.array([0, 0, 1])

    # Find cosine of the angle between the z-direction in the crystal and the reference axes
    cos_cube = np.dot(z_crystal, np.array([1, 0, 0]))
    cos_goss = np.dot(z_crystal, np.array([1, 1, 0])) / np.linalg.norm(np.array([1, 1, 0]))
    cos_brass = np.dot(z_crystal, np.array([1, 1, 1])) / np.linalg.norm(np.array([1, 1, 1]))

    # Determine the closest component based on the maximum cosine value
    max_cos = max(cos_cube, cos_goss, cos_brass)
    if max_cos == cos_cube:
        return "Cube"
    elif max_cos == cos_goss:
        return "Goss"
    else:
        return "Brass"