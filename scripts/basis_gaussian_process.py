import numpy as np
from typing import List, Optional, Any, Tuple, Literal, Union
from joblib import Parallel, delayed
import sys
from .utils.special_interp import *
from .utils.smoothing import smoothed
import optuna
from GPy.kern import RBF, Matern32, Matern52, RatQuad
from GPy.models import GPRegression
from GPy.mappings import Constant, Linear
import GPy.kern as kern
import GPy.mappings as mappings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from .utils import (
    create_kernel,
    create_mean_function,
    optimize_hyperparameters_optuna,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BasisGaussianProcess:
    def __init__(
        self,
        kernel: Literal["RBF", "Matern32", "Matern52", "RatQuad"] = "RBF",
        mean_type: Literal["constant", "linear", "none"] = "constant",
        hp_optimize: bool = False,
        n_restarts: int = 10,
        hp_n_jobs: int = 1,
        random_state: Optional[int] = 999,
        smoothing: bool = False,
        interpolation: Literal["fourier", "legendre", "chebyschev"] = "fourier",
        degree: int = 5,
        hp_test_size: float = 0.2,
        hp_n_trials: int = 20,
        test_size: float = 0.2,
    ):
        """
        Basis Gaussian Process Regression Model

        Args:
            kernel: Kernel type for Gaussian Process. Options are "RBF", "Matern32", "Matern52", "RatQuad"
            mean_type: Type of mean function. Options are "constant", "linear", "none"
            hp_optimize: Whether to optimize hyperparameters using Optuna
            n_restarts: Number of restarts for hyperparameter optimization
            hp_n_jobs: Number of parallel jobs for hyperparameter optimization
            random_state: Random seed for reproducibility
            smoothing: Whether to apply smoothing to predictions
            interpolation: Type of special interpolation to use. Options are "fourier", "legendre", "chebyschev"
            degree: Degree of the special interpolation
            hp_test_size: Proportion of data to use for validation during hyperparameter optimization
            test_size: Proportion of data to use for testing
        """
        self.kernel_type = kernel
        self.mean_type = mean_type
        self.hp_optimize = hp_optimize
        self.n_restarts = n_restarts
        self.hp_n_jobs = hp_n_jobs
        self.random_state = random_state
        self.smoothing = smoothing
        self.interpolation = interpolation
        self.degree = degree
        self.hp_test_size = hp_test_size
        self.test_size = test_size
        self.hp_n_trials = hp_n_trials
        self.model_dict = {}

        self.model_dict["default_kernel"] = self.kernel_type
        self.model_dict["default_mean_type"] = self.mean_type
        self.model_dict["hp_optimize"] = self.hp_optimize
        self.model_dict["n_restarts"] = self.n_restarts
        self.model_dict["hp_n_jobs"] = self.hp_n_jobs
        self.model_dict["random_state"] = self.random_state
        self.model_dict["smoothing"] = self.smoothing
        self.model_dict["interpolation"] = self.interpolation
        self.model_dict["degree"] = self.degree
        self.model_dict["hp_test_size"] = self.hp_test_size
        self.model_dict["test_size"] = self.test_size
        self.model_dict["hp_n_trials"] = self.hp_n_trials

    def fit(self, X: np.ndarray, y: np.ndarray, y_scale: float = 1.0) -> None:
        # Interpolate y using chosen interpolation method

        self.y_index = np.linspace(0, 1, y.shape[1]) * y_scale
        self.X, self.y = X, y
        self.dim, self.dim_y = X.shape[1], y.shape[1]

        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.Xs = self.X_scaler.fit_transform(self.X)
        self.ys = self.y_scaler.fit_transform(self.y)

        self.model_dict["X_scaler"] = self.X_scaler
        self.model_dict["y_scaler"] = self.y_scaler
        self.model_dict["y_index"] = self.y_index
        self.model_dict["X_scaled"] = self.Xs
        self.model_dict["y_scaled"] = self.ys
        self.model_dict["dim"] = self.dim

        if self.interpolation == "fourier":
            fn_interp = trig_interp
        elif self.interpolation == "legendre":
            fn_interp = legendre_interp
        elif self.interpolation == "chebyschev":
            fn_interp = chebyshev_interp
        else:
            raise ValueError(f"Unsupported interpolation type: {self.interpolation}")

        res = Parallel(n_jobs=-2)(
            delayed(fn_interp)(self.y_index, yr, self.degree) for yr in self.ys
        )

        self.y_coeff = np.vstack([res[i][1] for i in range(self.ys.shape[0])])

        self.coeff_scaler = MinMaxScaler()
        self.ys_coeff = self.coeff_scaler.fit_transform(self.y_coeff)
        self.model_dict["coefficient_scaler"] = self.coeff_scaler
        self.model_dict["coefficients_scaled"] = self.ys_coeff
        self.dim_coeff = self.ys_coeff.shape[1]

        self.models = []

        for i in range(self.dim_coeff):
            yi_coeff = self.ys_coeff[:, i : i + 1].reshape(-1, 1)

            if self.hp_optimize:
                best_params = optimize_hyperparameters_optuna(
                    self.Xs,
                    yi_coeff,
                    n_restarts=self.n_restarts,
                    hp_test_size=self.hp_test_size,
                    hp_n_jobs=self.hp_n_jobs,
                    hp_n_trials=self.hp_n_trials,
                    random_state=self.random_state,
                )
                kernel = create_kernel(best_params["kernel"], input_dim=self.dim)
                mean_function = create_mean_function(
                    best_params["mean_type"], input_dim=self.dim
                )
                self.model_dict[f"kernel_{i + 1}"] = best_params["kernel"]
                self.model_dict[f"mean_type_{i + 1}"] = best_params["mean_type"]
            else:
                kernel = create_kernel(self.kernel_type, input_dim=self.dim)
                mean_function = create_mean_function(self.mean_type, input_dim=self.dim)

            model = GPRegression(
                self.Xs, yi_coeff, kernel=kernel, mean_function=mean_function
            )
            model.optimize_restarts(num_restarts=self.n_restarts, verbose=False)
            self.models.append(model)
            self.model_dict[f"params_{i + 1}"] = model.param_array.copy()

    def predict(self, Xtest: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xtest_s = self.X_scaler.transform(Xtest)

        coeffs_pred = []
        coeffs_var = []

        for i in range(self.dim_coeff):
            model = self.models[i]
            coeff_pred, coeff_var = model.predict(Xtest_s)
            coeffs_pred.append(coeff_pred)
            coeffs_var.append(coeff_var)

        coeffs_pred = np.hstack(coeffs_pred)
        coeffs_var = np.hstack(coeffs_var)

        coeffs_pred_rescaled = self.coeff_scaler.inverse_transform(coeffs_pred)
        coeffs_var_rescaled = (
            coeffs_var
            * (self.coeff_scaler.data_max_ - self.coeff_scaler.data_min_) ** 2
        )

        if self.interpolation == "fourier":
            fn_predict = trig_predict
        elif self.interpolation == "legendre":
            fn_predict = legendre_predict
        elif self.interpolation == "chebyschev":
            fn_predict = chebyschev_predict
        else:
            raise ValueError(f"Unsupported interpolation type: {self.interpolation}")

        y_preds = []
        y_vars = []

        y_pred = Parallel(n_jobs=-2)(
            delayed(fn_predict)(
                coeffs_pred_rescaled[i, :], self.y_index, variance=False
            )
            for i in range(Xtest.shape[0])
        )
        y_var = Parallel(n_jobs=-2)(
            delayed(fn_predict)(coeffs_var_rescaled[i, :], self.y_index, variance=True)
            for i in range(Xtest.shape[0])
        )

        y_preds = np.vstack(y_pred)
        y_vars = np.vstack(y_var)

        y_preds_rescaled = self.y_scaler.inverse_transform(y_preds)
        y_vars_rescaled = (
            y_vars * (self.y_scaler.data_max_ - self.y_scaler.data_min_) ** 2
        )

        if self.smoothing:
            y_preds_rescaled, y_vars_rescaled = smoothed(
                y_preds_rescaled, y_vars_rescaled
            )

        return y_preds_rescaled, y_vars_rescaled

    def save_model(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.model_dict, f)

    @classmethod
    def load_model(cls, filepath: str) -> "BasisGaussianProcess":
        with open(filepath, "rb") as f:
            model_dict = pickle.load(f)

        model = cls(
            kernel=model_dict["default_kernel"],
            mean_type=model_dict["default_mean_type"],
            hp_optimize=model_dict["hp_optimize"],
            n_restarts=model_dict["n_restarts"],
            hp_n_jobs=model_dict["hp_n_jobs"],
            random_state=model_dict["random_state"],
            smoothing=model_dict["smoothing"],
            interpolation=model_dict["interpolation"],
            degree=model_dict["degree"],
            hp_test_size=model_dict["hp_test_size"],
            test_size=model_dict["test_size"],
            hp_n_trials=model_dict["hp_n_trials"],
        )

        model.X_scaler = model_dict["X_scaler"]
        model.y_scaler = model_dict["y_scaler"]
        model.y_index = model_dict["y_index"]
        model.Xs = model_dict["X_scaled"]
        model.ys = model_dict["y_scaled"]
        model.coeff_scaler = model_dict["coefficient_scaler"]
        model.ys_coeff = model_dict["coefficients_scaled"]
        model.dim_coeff = model.ys_coeff.shape[1]

        model.models = []

        for i in range(model.degree + 1):
            kernel_type = model_dict.get(
                f"kernel_{i + 1}", model_dict["default_kernel"]
            )
            mean_type = model_dict.get(
                f"mean_type_{i + 1}", model_dict["default_mean_type"]
            )

            kernel = create_kernel(kernel_type, input_dim=model_dict["dim"])
            mean_function = create_mean_function(mean_type, input_dim=model_dict["dim"])

            gp_model = GPRegression(
                model.Xs,
                model.ys_coeff[:, i : i + 1],
                kernel=kernel,
                mean_function=mean_function,
            )
            gp_model[:] = model_dict[f"params_{i + 1}"]
            gp_model.update_model(True)
            model.models.append(gp_model)

        return model
