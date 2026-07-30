import pickle
from typing import Optional, Tuple
import numpy as np
import optuna
from GPy.models import GPRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from GPy.util.multioutput import LCM
from .utils.special_interp import *
from .utils.smoothing import smoothed
from .utils import (
    create_kernel,
    create_mean_function,
    optimize_hyperparameters_optuna,
    model_dict_assign,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CoregGaussianProcess:
    def __init__(
        self,
        kernel: str = "RBF",
        mean_type: str = "constant",
        test_size: float = 0.2,
        hp_test_size: float = 0.2,
        n_restarts: int = 10,
        random_state: Optional[int] = None,
        hp_n_jobs: int = 1,
        hp_n_trials: int = 20,
        hp_optimize: bool = False,
        n_latents: int = 1,
    ):
        """
        Coregionalized Gaussian Process Regression model.

        Args:
            kernel (str): Type of kernel to use ('RBF', 'Matern32', 'Matern52', 'RatQuad').
            mean_type (str): Type of mean function ('constant' or 'linear').
            test_size (float): Proportion of data to use for testing.
            hp_test_size (float): Proportion of data to use for hyperparameter tuning.
            n_restarts (int): Number of restarts for hyperparameter optimization.
            random_state (Optional[int]): Random seed for reproducibility.
            hp_n_jobs (int): Number of parallel jobs for hyperparameter optimization.
            hp_n_trials (int): Number of trials for hyperparameter optimization.
            hp_optimize (bool): Whether to perform hyperparameter optimization.
            n_latents (int): Number of latent processes for the LCM kernel.
        """
        self.kernel = kernel
        self.mean_type = mean_type
        self.test_size = test_size
        self.hp_test_size = hp_test_size
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.hp_n_jobs = hp_n_jobs
        self.hp_n_trials = hp_n_trials
        self.hp_optimize = hp_optimize
        self.n_latents = n_latents
        self.model_dict = model_dict_assign(
            default_kernel=kernel,
            default_mean_type=mean_type,
            n_latents=n_latents,
            random_state=random_state,
            test_size=test_size,
            hp_optimize=hp_optimize,
            hp_n_trials=hp_n_trials,
            hp_n_jobs=hp_n_jobs,
            n_restarts=n_restarts,
        )

    def _optimize_coregionalized_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_latents: int = 1,
        n_restarts: int = 10,
        hp_test_size: float = 0.2,
        hp_n_trials: int = 20,
        hp_n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> dict:
        def func(trial, Xtrain, ytrain, Xval, yval, n_restarts, n_latents):
            kernels = []
            for i in range(n_latents):
                kernel_choice = trial.suggest_categorical(
                    f"kernel_{i + 1}", ["RBF", "Matern32", "Matern52", "RatQuad"]
                )
                base_kernel = create_kernel(kernel_choice, input_dim=Xtrain.shape[1])
                kernels.append(base_kernel)

            mean_choice = trial.suggest_categorical(
                "mean_type", ["constant", "linear", "none"]
            )

            mean_function = create_mean_function(
                mean_choice, input_dim=Xtrain.shape[1] + 1, output_dim=1
            )

            lcm = LCM(
                input_dim=Xtrain.shape[1],
                num_outputs=ytrain.shape[1],
                kernels_list=kernels,
            )

            Xtrain = np.hstack(
                [
                    np.tile(Xtrain, (ytrain.shape[1], 1)),
                    np.repeat(
                        np.arange(ytrain.shape[1]).reshape(-1, 1),
                        Xtrain.shape[0],
                        axis=0,
                    ),
                ]
            )

            Xval = np.hstack(
                [
                    np.tile(Xval, (yval.shape[1], 1)),
                    np.repeat(
                        np.arange(yval.shape[1]).reshape(-1, 1),
                        Xval.shape[0],
                        axis=0,
                    ),
                ]
            )

            ytrain = ytrain.flatten(order="F").reshape(-1, 1)
            yval = yval.flatten(order="F").reshape(-1, 1)

            model = GPRegression(
                Xtrain, ytrain, kernel=lcm, mean_function=mean_function, normalizer=True
            )
            model.optimize_restarts(num_restarts=n_restarts, verbose=False)

            yval_pred, _ = model.predict(Xval)
            mse = np.mean((yval - yval_pred) ** 2)
            return mse

        Xtrain, Xval, ytrain, yval = train_test_split(
            X, y, test_size=hp_test_size, random_state=random_state
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: func(
                trial, Xtrain, ytrain, Xval, yval, n_restarts, n_latents
            ),
            n_trials=hp_n_trials,
            n_jobs=hp_n_jobs,
        )
        return study.best_params

    def fit(self, X: np.ndarray, y: np.ndarray, y_scale: float = 1.0):
        self.y_index = np.linspace(0, 1, y.shape[1]) * y_scale
        self.X, self.y = X, y
        self.dim, self.dim_y = X.shape[1], y.shape[1]

        self.X_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()
        self.Xs = self.X_scaler.fit_transform(self.X)
        self.ys = self.y_scaler.fit_transform(self.y)

        self.model_dict["X_scaler"] = self.X_scaler
        self.model_dict["y_index"] = self.y_index
        self.model_dict["X_scaled"] = self.Xs
        self.model_dict["dim"] = self.dim
        self.model_dict["dim_y"] = self.dim_y
        self.model_dict["y_scaler"] = self.y_scaler
        self.model_dict["y_scaled"] = self.ys

        if self.hp_optimize:
            best_params = self._optimize_coregionalized_hyperparameters(
                self.Xs,
                self.ys,
                n_latents=self.n_latents,
                n_restarts=self.n_restarts,
                hp_test_size=self.hp_test_size,
                hp_n_jobs=self.hp_n_jobs,
                hp_n_trials=self.hp_n_trials,
                random_state=self.random_state,
            )

            kernels = []
            for i in range(self.n_latents):
                kernel = create_kernel(
                    best_params[f"kernel_{i + 1}"], input_dim=self.dim
                )
                kernels.append(kernel)
                self.model_dict[f"kernel_{i + 1}"] = best_params[f"kernel_{i + 1}"]

            mean_function = create_mean_function(
                best_params["mean_type"], input_dim=self.dim + 1, output_dim=1
            )

            self.model_dict["mean_type"] = best_params["mean_type"]

        else:
            kernels = []
            for i in range(self.n_latents):
                kernel = create_kernel(self.kernel, input_dim=self.dim)
                kernels.append(kernel)

            mean_function = create_mean_function(
                self.mean_type, input_dim=self.dim + 1, output_dim=1
            )

        lcm = LCM(
            input_dim=self.dim,
            num_outputs=self.dim_y,
            kernels_list=kernels,
        )
        Xs_coreg = np.hstack(
            [
                np.tile(self.Xs, (self.ys.shape[1], 1)),
                np.repeat(
                    np.arange(self.ys.shape[1]).reshape(-1, 1),
                    self.Xs.shape[0],
                    axis=0,
                ),
            ]
        )
        ys_coreg = self.ys.flatten(order="F").reshape(-1, 1)
        self.model = GPRegression(
            Xs_coreg, ys_coreg, kernel=lcm, mean_function=mean_function, normalizer=True
        )
        self.model.optimize_restarts(
            num_restarts=self.n_restarts, verbose=True, num_processes=self.hp_n_jobs
        )
        self.model_dict["params"] = self.model.param_array.copy()
        self.model_dict["Xs_coreg"] = Xs_coreg
        self.model_dict["ys_coreg"] = ys_coreg

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.X_scaler.transform(X)
        Xs_coreg = np.hstack(
            [
                np.tile(Xs, (self.dim_y, 1)),
                np.repeat(
                    np.arange(self.dim_y).reshape(-1, 1),
                    Xs.shape[0],
                    axis=0,
                ),
            ]
        )
        y_coreg_pred, y_coreg_var = self.model.predict(Xs_coreg)
        y_preds = y_coreg_pred.reshape(X.shape[0], self.dim_y, order="F")
        y_vars = y_coreg_var.reshape(X.shape[0], self.dim_y, order="F")

        y_pred = self.y_scaler.inverse_transform(y_preds)
        y_std = self.y_scaler.scale_ * np.sqrt(y_vars)

        return y_pred, y_std

    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self.model_dict, f)

    @classmethod
    def load_model(cls, filename: str) -> "CoregGaussianProcess":
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)

        obj = cls(
            kernel=model_dict.get("default_kernel", "RBF"),
            mean_type=model_dict.get("default_mean_type", "constant"),
            test_size=model_dict.get("test_size", 0.2),
            hp_optimize=model_dict.get("hp_optimize", False),
            hp_n_trials=model_dict.get("hp_n_trials", 20),
            hp_n_jobs=model_dict.get("hp_n_jobs", 1),
            n_restarts=model_dict.get("n_restarts", 10),
            random_state=model_dict.get("random_state", None),
            n_latents=model_dict.get("n_latents", 1),
        )

        obj.model_dict = model_dict
        obj.X_scaler = model_dict["X_scaler"]
        obj.y_scaler = model_dict["y_scaler"]
        obj.Xs = model_dict["X_scaled"]
        obj.y_index = model_dict["y_index"]
        obj.dim = model_dict["dim"]
        obj.dim_y = model_dict["dim_y"]

        kernels = []
        for i in range(obj.n_latents):
            kernel = create_kernel(
                model_dict.get(f"kernel_{i + 1}", obj.kernel), input_dim=obj.dim
            )
            kernels.append(kernel)

        mean_function = create_mean_function(
            model_dict.get("mean_type", obj.mean_type),
            input_dim=obj.dim + 1,
            output_dim=1,
        )

        lcm = LCM(
            input_dim=obj.dim,
            num_outputs=obj.dim_y,
            kernels_list=kernels,
        )

        obj.model = GPRegression(
            model_dict["Xs_coreg"],
            model_dict["ys_coreg"],
            kernel=lcm,
            mean_function=mean_function,
            normalizer=True,
        )
        obj.model[:] = model_dict["params"]
        obj.model.update_model(True)

        return obj
