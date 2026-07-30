import numpy as np
from typing import Tuple
from joblib import Parallel, delayed


def voce_to_tensile(
    sigma_0: float,
    Q: float,
    B: float,
) -> Tuple[float, float, float, float, float]:
    """
    Convert Voce parameters to tensile properties: yield strength and tensile strength.

    Parameters
    ----------
    sigma_0 : float
        Initial yield stress from Voce model.
    Q : float
        Saturation stress from Voce model.
    B : float
        Hardening rate from Voce model.

    Returns
    -------
    Tuple[float, float]
        A tuple containing yield strength and tensile strength.
    """
    yield_strength = sigma_0
    uniform_elongation = 1 / B * np.log((Q * (B + 1)) / (sigma_0 + Q))
    ultimate_tensile_strength = (sigma_0 + Q) * B / (B + 1)
    engineering_uts = ultimate_tensile_strength * np.exp(-uniform_elongation)
    engineering_ue = np.exp(uniform_elongation) - 1

    return (
        yield_strength,
        ultimate_tensile_strength,
        uniform_elongation,
        engineering_uts,
        engineering_ue,
    )


def extract_tensile_properties(
    samples: np.ndarray,
) -> np.ndarray:
    """
    Extract tensile properties from Voce parameters for a set of samples.

    Parameters
    ----------
    samples : np.ndarray
        An array of shape (n_samples, 3) containing Voce parameters (sigma_0, Q, B).

    Returns
    -------
    np.ndarray
        An array of shape (n_samples, 5) containing tensile properties (yield strength, ultimate tensile strength, uniform elongation, engineering UTS, engineering UE).
    """
    tensile_properties = Parallel(n_jobs=-1)(
        delayed(voce_to_tensile)(*sample) for sample in samples
    )
    return np.array(tensile_properties)
