from typing import Any, Sequence, Union
import numpy as np
from scipy.stats import invgamma
from functools import partial


def spt_likelihood(
    x: np.ndarray,
    force: np.ndarray,
    model: Any,
    nu: float,
    thickness: Union[float, np.ndarray],
    n_properties: int,
) -> float:
    """
    Compute the likelihood of the observed forces given the model predictions and uncertainties.
    Can consistently handle experimental replicates by summing the log-likelihoods across all observations.
    Args:
        x (np.ndarray): Input parameters for the model.
                        The first n_properties elements correspond to material properties,
                        and the remaining elements correspond to force variances.
        force (np.ndarray): Observed forces.
        model (Any): A model object that has a predict method.
        nu (float): Poisson's ratio.
        thickness (Union[float, np.ndarray]): Thickness of the material.
        n_properties (int): Number of material properties.
    """

    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            "The provided model does not have a callable 'predict' method."
        )

    force_var = x[n_properties:]
    x_prop = x[:n_properties]
    x_prop = np.array([x_prop[0], nu, *x_prop[1:], thickness])
    try:
        pred_force, pred_std = model.predict(x_prop.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    len_force = min(len(force), len(pred_force.flatten()))
    force_pred_mean = pred_force.flatten()[:len_force]
    force_pred_std = pred_std.flatten()[:len_force]
    force_pred_var = force_pred_std**2  # Convert std to variance
    force_var = force_var[:len_force]  # Ensure force_var matches the length of force
    force = force[:len_force]  # Ensure force matches the length of predictions

    total_var = force_var + force_pred_var
    likelihoods = [
        -0.5 * np.log(2 * np.pi * total_var)
        - 0.5 * ((force[:, k] - force_pred_mean) ** 2 / total_var)
        for k in range(force.shape[1])
    ]
    total_likelihood = np.sum(likelihoods)
    return total_likelihood


def spt_uniform_prior(
    x: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    if np.any(x < lower_bounds) or np.any(x > upper_bounds):
        return -np.inf
    else:
        return 0.0


def spt_invgamma_prior(
    x: np.ndarray,
    experimental_variance: np.ndarray,
    dof: int = 3,
) -> float:
    if np.any(x <= 0):
        return -np.inf
    priors = invgamma.logpdf(x, a=dof / 2, scale=dof * experimental_variance**2 / 2)
    std_prior = np.sum(priors)
    return std_prior


def spt_total_prior(
    x: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    experimental_variance: np.ndarray,
    n_properties: int,
    dof: int = 3,
) -> float:
    x_prop = x[:n_properties]
    x_var = x[n_properties:]
    if len(lower_bounds) != n_properties or len(upper_bounds) != n_properties:
        raise ValueError("Length of bounds must match the number of properties.")
    if len(x_var) != len(experimental_variance):
        raise ValueError(
            "Length of variance parameters must match the length of experimental variance."
        )
    uniform_prior = spt_uniform_prior(x_prop, lower_bounds, upper_bounds)
    invgamma_prior = spt_invgamma_prior(x_var, experimental_variance, dof)
    total_prior = uniform_prior + invgamma_prior
    return total_prior


def spt_likelihood_no_modulus(
    x: np.ndarray,
    force: np.ndarray,
    model: Any,
    E: float,
    nu: float,
    thickness: Union[float, np.ndarray],
    n_properties: int,
) -> float:
    """
    Compute the likelihood of the observed forces given the model predictions and uncertainties.
    This version does not modify the input parameters to include Poisson's ratio and thickness.
    Args:
        x (np.ndarray): Input parameters for the model.
                        The first n_properties elements correspond to material properties,
                        and the remaining elements correspond to force variances.
        force (np.ndarray): Observed forces.
        model (Any): A model object that has a predict method.
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio.
        thickness (Union[float, np.ndarray]): Thickness of the material.
        n_properties (int): Number of material properties.
    """
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            "The provided model does not have a callable 'predict' method."
        )

    force_var = x[n_properties:]
    x_prop = x[:n_properties]
    x_prop = np.array([E, nu, *x_prop, thickness])
    try:
        pred_force, pred_std = model.predict(x_prop.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    len_force = min(len(force), len(pred_force.flatten()))
    force_pred_mean = pred_force.flatten()[:len_force]
    force_pred_std = pred_std.flatten()[:len_force]
    force_pred_var = force_pred_std**2  # Convert std to variance
    force_var = force_var[:len_force]  # Ensure force_var matches the length of force
    force = force[:len_force]  # Ensure force matches the length of predictions
    total_var = force_var + force_pred_var
    likelihoods = [
        -0.5 * np.log(2 * np.pi * total_var)
        - 0.5 * ((force[:, k] - force_pred_mean) ** 2 / total_var)
        for k in range(force.shape[1])
    ]
    total_likelihood = np.sum(likelihoods)
    return total_likelihood


def spt_likelihood_prop_only(
    x: np.ndarray,
    force: np.ndarray,
    force_std: np.ndarray,
    model: Any,
    nu: float,
    thickness: Union[float, np.ndarray],
):
    """
    Compute the likelihood of the observed forces given the model predictions and uncertainties.
    This version only considers the material properties and assumes a fixed variance.
    Args:
        x (np.ndarray): Input parameters for the model, corresponding to material properties.
        force (np.ndarray): Observed forces.
        force_std (np.ndarray): Standard deviation of the observed forces.
        model (Any): A model object that has a predict method.
        nu (float): Poisson's ratio.
        thickness (Union[float, np.ndarray]): Thickness of the material.
    """
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            "The provided model does not have a callable 'predict' method."
        )

    x_prop = x
    x_prop = np.array([x_prop[0], nu, *x_prop[1:], thickness])
    try:
        pred_force, pred_std = model.predict(x_prop.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    len_force = min(len(force), len(pred_force.flatten()))
    force_pred_mean = pred_force.flatten()[:len_force]
    force_pred_std = pred_std.flatten()[:len_force]
    force_pred_var = force_pred_std**2  # Convert std to variance
    force = force[:len_force]  # Ensure force matches the length of predictions
    force_var = (
        force_std[:len_force] ** 2
    )  # Use provided standard deviation for variance
    total_var = force_var + force_pred_var
    likelihoods = [
        -0.5 * np.log(2 * np.pi * total_var)
        - 0.5 * ((force[:, k] - force_pred_mean) ** 2 / total_var)
        for k in range(force.shape[1])
    ]
    total_likelihood = np.sum(likelihoods)
    return total_likelihood


def spt_likelihood_prop_only_no_modulus(
    x: np.ndarray,
    force: np.ndarray,
    force_std: np.ndarray,
    model: Any,
    E: float,
    nu: float,
    thickness: Union[float, np.ndarray],
):
    """
    Compute the likelihood of the observed forces given the model predictions and uncertainties.
    This version only considers the material properties and assumes a fixed variance, without modifying the input parameters to include Poisson's ratio and thickness.
    Args:
        x (np.ndarray): Input parameters for the model, corresponding to material properties.
        force (np.ndarray): Observed forces.
        force_std (np.ndarray): Standard deviation of the observed forces.
        model (Any): A model object that has a predict method.
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio.
        thickness (Union[float, np.ndarray]): Thickness of the material.
    """
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            "The provided model does not have a callable 'predict' method."
        )

    x_prop = x
    x_prop = np.array([E, nu, *x_prop, thickness])
    try:
        pred_force, pred_std = model.predict(x_prop.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    len_force = min(len(force), len(pred_force.flatten()))
    force_pred_mean = pred_force.flatten()[:len_force]
    force_pred_std = pred_std.flatten()[:len_force]
    force_pred_var = force_pred_std**2  # Convert std to variance
    force = force[:len_force]  # Ensure force matches the length of predictions
    force_var = (
        force_std[:len_force] ** 2
    )  # Use provided standard deviation for variance
    total_var = force_var + force_pred_var
    likelihoods = [
        -0.5 * np.log(2 * np.pi * total_var)
        - 0.5 * ((force[:, k] - force_pred_mean) ** 2 / total_var)
        for k in range(force.shape[1])
    ]
    total_likelihood = np.sum(likelihoods)
    return total_likelihood


def spt_likelihood_hardening(
    x: np.ndarray,
    sigma_0: float,
    E: float,
    nu: float,
    thickness: Union[float, np.ndarray],
    force: np.ndarray,
    force_std: np.ndarray,
    model: Any,
) -> float:
    """
    Compute the likelihood of the observed forces given the model predictions and uncertainties, specifically for hardening behavior.
    Args:
        x (np.ndarray): Input parameters for the model, corresponding to hardening parameters.
        sigma_0 (float): Initial yield stress.
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio.
        thickness (Union[float, np.ndarray]): Thickness of the material.
        force (np.ndarray): Observed forces.
        force_std (np.ndarray): Standard deviation of the observed forces.
        model (Any): A model object that has a predict method.
    """
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(
            "The provided model does not have a callable 'predict' method."
        )

    x_prop = x
    x_prop = np.array([E, nu, sigma_0, *x_prop, thickness])
    try:
        pred_force, pred_std = model.predict(x_prop.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    len_force = min(len(force), len(pred_force.flatten()))
    force_pred_mean = pred_force.flatten()[:len_force]
    force_pred_std = pred_std.flatten()[:len_force]
    force_pred_var = force_pred_std**2  # Convert std to variance
    force = force[:len_force]  # Ensure force matches the length of predictions
    force_var = (
        force_std[:len_force] ** 2
    )  # Use provided standard deviation for variance
    total_var = force_var + force_pred_var
    likelihoods = [
        -0.5 * np.log(2 * np.pi * total_var)
        - 0.5 * ((force[:, k] - force_pred_mean) ** 2 / total_var)
        for k in range(force.shape[1])
    ]
    total_likelihood = np.sum(likelihoods)
    return total_likelihood
