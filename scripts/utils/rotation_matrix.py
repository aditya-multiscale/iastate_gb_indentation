import numpy as np


# Rotation matrix using bunge euler angles
def rotation_matrix(phi1, PHI, phi2, degrees=False):
    """
    Compute the rotation matrix from Bunge Euler angles.

    Args:
        phi1 (float): First Euler angle in radians.
        PHI (float): Second Euler angle in radians.
        phi2 (float): Third Euler angle in radians.
        degrees (bool): If True, the input angles are in degrees. Default is False.
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    if degrees:
        phi1 = np.radians(phi1)
        PHI = np.radians(PHI)
        phi2 = np.radians(phi2)

    R = np.array(
        [
            [
                np.cos(phi1) * np.cos(phi2) - np.sin(phi1) * np.sin(phi2) * np.cos(PHI),
                -np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(PHI),
                np.sin(phi1) * np.sin(PHI),
            ],
            [
                np.sin(phi1) * np.cos(phi2) + np.cos(phi1) * np.sin(phi2) * np.cos(PHI),
                -np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(PHI),
                -np.cos(phi1) * np.sin(PHI),
            ],
            [np.sin(phi2) * np.sin(PHI), np.cos(phi2) * np.sin(PHI), np.cos(PHI)],
        ]
    )
    return R

def get_misorientation_angle_axis(R1, R2, degrees=False):
    """
    Compute the misorientation angle between two rotation matrices.

    Args:
        R1 (np.ndarray): First rotation matrix (3x3).
        R2 (np.ndarray): Second rotation matrix (3x3).

    Returns:
        tuple: (misorientation_angle, misorientation_axis)
    """
    R = np.dot(R1.T, R2)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    axis = axis / np.linalg.norm(axis)
    if degrees:
        theta = np.degrees(theta)
    return theta, axis


import numpy as np

def rotation_matrix_to_euler(R: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    r33 = np.clip(R[2, 2], -1.0, 1.0)
    Phi = np.arccos(r33)

    if np.isclose(np.sin(Phi), 0.0, atol=1e-8):
        if Phi < np.pi / 2:
            Phi = 0.0
            phi1 = np.arctan2(-R[1, 0], R[0, 0]) % (2 * np.pi)
        else:
            Phi = np.pi
            phi1 = np.arctan2(R[1, 0], R[0, 0]) % (2 * np.pi)
        phi2 = 0.0
    else:
        phi1 = np.arctan2(R[2, 0], -R[2, 1]) % (2 * np.pi)
        phi2 = np.arctan2(R[0, 2], R[1, 2]) % (2 * np.pi)

    angles = (phi1, Phi, phi2)
    return tuple(np.degrees(a) for a in angles) if degrees else angles

