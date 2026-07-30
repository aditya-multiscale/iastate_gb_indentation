import numpy as np


# Rotation matrix using bunge euler angles
def rotation_matrix(phi1, PHI, phi2):
    """
    Compute the rotation matrix from Bunge Euler angles.

    Args:
        phi1 (float): First Euler angle in radians.
        PHI (float): Second Euler angle in radians.
        phi2 (float): Third Euler angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
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

def get_misorientation_angle(R1, R2):
    """
    Compute the misorientation angle between two rotation matrices.

    Args:
        R1 (np.ndarray): First rotation matrix (3x3).
        R2 (np.ndarray): Second rotation matrix (3x3).

    Returns:
        float: Misorientation angle in radians.
    """
    R = np.dot(R1.T, R2)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)
    return theta