"""
Script to project Euler angles into the fundamental zone of a crystal symmetry.
Methods:
- project_to_fundamental_zone: Projects given Euler angles into the fundamental zone of a crystal symmetry.
"""
import numpy as np
from typing import Sequence

from .rotation_matrix import *

# 24 proper rotations of the cubic point group O(432), as listed in
# Kocks, Tome & Wenk, Ch. 1 Table II (cubic branch), with the errata noted in
# Rollett's 27-750 lecture notes applied: the 33 element of the x-diad
# diag(1, -1, -1) and of the last two z-tetrads is corrected.
# Convention (passive / Bunge): crystal symmetry pre-multiplies the orientation
# matrix, g' = O_c @ g.
_CUBIC_SYMMETRY_OPERATORS = np.array([
    # --- Column 1 ---
    # identity
    [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],
    # 3-fold rotations about <111>
    [[ 0,  0,  1], [ 1,  0,  0], [ 0,  1,  0]],
    [[ 0,  1,  0], [ 0,  0,  1], [ 1,  0,  0]],
    [[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]],
    [[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]],
    [[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]],
    [[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]],
    [[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]],
    [[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]],
    # 2-fold rotations about <100> (y, z, x diads)
    [[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]],
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],
    # --- Column 2 ---
    # 2-fold about <101>, 4-fold about y
    [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],
    [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],
    [[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]],
    [[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]],
    # 2-fold about <011>, 4-fold about x
    [[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]],
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],
    [[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]],
    # 2-fold about <110>, 4-fold about z
    [[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]],
    [[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]],
    [[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]],
    [[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]],
], dtype=float)


def project_to_fundamental_zone(euler_angles: Sequence[float], degrees: bool = False):
    """
    Projects the given Euler angles into the fundamental zone of the cubic crystal symmetry.

    Parameters:
    - euler_angles (Sequence[float]): Euler angles (phi1, Phi, phi2) in radians if degrees is False, otherwise in degrees.
    - degrees (bool): If True, the input and output Euler angles are in degrees. If False, they are in radians.

    Returns:
    Sequence[float]: Euler angles projected into the fundamental zone, following these constraints:
        - phi1 in [0, 2*pi]
        - Phi in [arccos(cos(phi2)/sqrt(1 + cos(phi2)**2)), pi/2]
        - phi2 in [0, pi/4]
    """
    phi1, Phi, phi2 = euler_angles
    if degrees:
        phi1 = np.deg2rad(phi1)
        Phi = np.deg2rad(Phi)
        phi2 = np.deg2rad(phi2)
    # Pathological check: if angles are already within the fundamental zone, return them immediately
    if 0 <= phi1 < 2*np.pi and np.arccos(np.cos(phi2)/np.sqrt(1 + np.cos(phi2)**2)) <= Phi <= np.pi/2 and 0 <= phi2 <= np.pi/4:
        if degrees:
            return np.rad2deg(phi1), np.rad2deg(Phi), np.rad2deg(phi2)
        return phi1, Phi, phi2
    R0 = rotation_matrix(phi1, Phi, phi2)
    for array in _CUBIC_SYMMETRY_OPERATORS:
        R_prime = array @ R0
        phi1_z, Phi_z, phi2_z = rotation_matrix_to_euler(R_prime)
        # If the transformed Euler angles satisfy the fundamental zone constraints, return them
        if 0 <= phi1_z < 2*np.pi and np.arccos(np.cos(phi2_z)/np.sqrt(1 + np.cos(phi2_z)**2)) <= Phi_z <= np.pi/2 and 0 <= phi2_z <= np.pi/4:
            if degrees:
                return np.rad2deg(phi1_z), np.rad2deg(Phi_z), np.rad2deg(phi2_z)
            return phi1_z, Phi_z, phi2_z
    # If no symmetry operator brings the Euler angles into the fundamental zone, return the original angles
    if degrees:
        return np.rad2deg(phi1), np.rad2deg(Phi), np.rad2deg(phi2)
    return phi1, Phi, phi2