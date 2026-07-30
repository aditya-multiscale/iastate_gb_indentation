from GPy.kern import RBF, Matern32, Matern52, RatQuad
from typing import Optional, Union
import GPy.kern as kern
from GPy.mappings import Constant, Linear


def create_kernel(kernel_type: str = "RBF", input_dim: int = 1) -> kern:
    """
    Create a GPy kernel based on the specified type.
    Args:
        kernel_type (str): Type of kernel to create ('RBF', 'Matern32', 'Matern52', 'RatQuad').
        input_dim (int): Dimensionality of the input data.
    Returns:
        kern: An instance of the specified GPy kernel.
    """
    if kernel_type == "RBF":
        base_kernel = RBF(input_dim=input_dim, ARD=True)
    elif kernel_type == "Matern32":
        base_kernel = Matern32(input_dim=input_dim, ARD=True)
    elif kernel_type == "Matern52":
        base_kernel = Matern52(input_dim=input_dim, ARD=True)
    elif kernel_type == "RatQuad":
        base_kernel = RatQuad(input_dim=input_dim, ARD=True)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    return base_kernel


def create_mean_function(
    mean_type: str = "constant", input_dim: int = 1, output_dim: int = 1
) -> Optional[Union[Constant, Linear]]:
    """
    Create a GPy mean function based on the specified type.
    Args:
        mean_type (str): Type of mean function to create ('constant', 'linear', 'none').
        input_dim (int): Dimensionality of the input data.
    Returns:
        Optional[Union[Constant, Linear]]: An instance of the specified GPy mean function or None.
    """
    if mean_type == "constant":
        return Constant(input_dim=input_dim, output_dim=output_dim)
    elif mean_type == "linear":
        return Linear(input_dim=input_dim, output_dim=output_dim)
    elif mean_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported mean type: {mean_type}")
