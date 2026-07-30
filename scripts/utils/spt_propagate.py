import pandas as pd
import os
import numpy as np
from typing import Any
from joblib import Parallel, delayed


def spt_propagate(
    model: Any,
    samples: np.ndarray,
    nu: float,
    thickness: float,
):
    """Propagate the samples through the model in parallel."""
    predict = lambda sample: model.predict(
        np.array([sample[0], nu, *sample[1:], thickness]).reshape(1, -1)
    )[0].flatten()

    predictions = Parallel(n_jobs=-1)(delayed(predict)(sample) for sample in samples)
    return np.array(predictions)


def spt_propagate_no_modulus(
    model: Any,
    samples: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
):
    """Propagate the samples through the model in parallel, without modulus."""
    predict = lambda sample: model.predict(
        np.array([E, nu, *sample, thickness]).reshape(1, -1)
    )[0].flatten()

    predictions = Parallel(n_jobs=-1)(delayed(predict)(sample) for sample in samples)
    return np.array(predictions)


def spt_propagate_hardening(
    model: Any,
    samples: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    sigma_0: float,
):
    """Propagate the samples through the model in parallel, with hardening."""
    predict = lambda sample: model.predict(
        np.array([E, nu, sigma_0, *sample, thickness]).reshape(1, -1)
    )[0].flatten()

    predictions = Parallel(n_jobs=-1)(delayed(predict)(sample) for sample in samples)
    return np.array(predictions)
