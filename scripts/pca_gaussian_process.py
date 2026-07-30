import pickle
from typing import Optional, Tuple
import numpy as np
import optuna
from GPy.models import GPRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from .utils.special_interp import *
from .utils.smoothing import smoothed
from .utils import (
    create_kernel,
    create_mean_function,
    optimize_hyperparameters_optuna,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class PCAGaussianProcess:
    def __init__(
        self,
        n_components: int = 5,
        whiten: bool = True,
        kernel: str = "RBF",
        mean_type: str = "constant",
        test_size: float = 0.2,
        hp_test_size: float = 0.2,
        n_restarts: int = 10,
        random_state: Optional[int] = None,
        hp_n_jobs: int = 1,
        hp_n_trials: int = 20,
        hp_optimize: bool = True,
    ):
        """
        Gaussian Process Regression model on PCA transformed data.

        Args:
            n_components (int): Number of PCA components.
            whiten (bool): Whether to whiten PCA components.
            kernel (str): Type of kernel to use ('RBF', 'Matern32', 'Matern52', 'RatQuad').
            mean_type (str): Type of mean function ('constant' or 'linear').
            test_size (float): Proportion of data to use for testing.
            hp_test_size (float): Proportion of data to use for hyperparameter tuning.
            n_restarts (int): Number of restarts for hyperparameter optimization.
            random_state (Optional[int]): Random seed for reproducibility.
            hp_n_jobs (int): Number of parallel jobs for hyperparameter optimization.
            hp_n_trials (int): Number of trials for hyperparameter optimization.
            hp_optimize (bool): Whether to perform hyperparameter optimization.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.whiten = whiten
        self.mean_type = mean_type
        self.test_size = test_size
        self.hp_test_size = hp_test_size
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.hp_n_jobs = hp_n_jobs
        self.hp_n_trials = hp_n_trials
        self.hp_optimize = hp_optimize

        self.model_dict = {}

        self.model_dict["n_components"] = self.n_components
        self.model_dict["default_kernel"] = self.kernel
        self.model_dict["default_mean_type"] = self.mean_type
        self.model_dict["test_size"] = self.test_size
        self.model_dict["hp_test_size"] = self.hp_test_size
        self.model_dict["n_restarts"] = self.n_restarts
        self.model_dict["random_state"] = self.random_state
        self.model_dict["hp_n_jobs"] = self.hp_n_jobs
        self.model_dict["hp_n_trials"] = self.hp_n_trials
        self.model_dict["hp_optimize"] = self.hp_optimize
        self.model_dict["whiten"] = self.whiten

    def fit(self, X: np.ndarray, y: np.ndarray, y_scale: float = 1.0) -> None:
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

        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self.y_pca = self.pca.fit_transform(self.ys)
        self.model_dict["pca"] = self.pca
        self.model_dict["y_pca"] = self.y_pca
        self.model_dict["explained_variance_ratio_"] = (
            self.pca.explained_variance_ratio_
        )

        self.models = []

        for i in range(self.n_components):
            print(f"Training GP for PCA component {i + 1}/{self.n_components}")
            y_comp = self.y_pca[:, i].reshape(-1, 1)
            if self.hp_optimize:
                best_params = optimize_hyperparameters_optuna(
                    self.Xs,
                    y_comp,
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
                kernel = create_kernel(self.kernel, input_dim=self.dim)
                mean_function = create_mean_function(self.mean_type, input_dim=self.dim)

            model = GPRegression(
                self.Xs,
                y_comp,
                kernel=kernel,
                mean_function=mean_function,
                normalizer=True,
            )

            model.optimize_restarts(
                num_restarts=self.n_restarts,
                verbose=False,
                num_processes=self.hp_n_jobs,
            )

            self.models.append(model)
            self.model_dict[f"params_{i + 1}"] = model.param_array.copy()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.X_scaler.transform(X)
        y_pca_preds = []
        y_pca_vars = []
        for i, model in enumerate(self.models):
            y_pca_pred, y_pca_var = model.predict(Xs)
            y_pca_preds.append(y_pca_pred.reshape(-1))
            y_pca_vars.append(y_pca_var.reshape(-1))
        y_pca_preds = np.array(y_pca_preds).T
        y_pca_vars = np.array(y_pca_vars).T
        y_preds = self.pca.inverse_transform(y_pca_preds)
        y_pred = self.y_scaler.inverse_transform(y_preds)

        # Variance calculation
        # 1. Handle PCA Whitening
        # If whitened, inverse_transform multiplies by sqrt(explained_variance)
        # So variance must be multiplied by explained_variance_
        if self.whiten:
            y_pca_vars = y_pca_vars * self.pca.explained_variance_

        # 2. Project variance to feature space
        # Equivalent to your einsum: np.einsum('ji,kj,ji->ki', self.pca.components_, y_pca_vars, self.pca.components_)
        y_vars_scaled = np.dot(y_pca_vars, self.pca.components_**2)

        # 3. Handle StandardScaler scaling
        # Inverse transform multiplies by std (sqrt(var_)), so variance multiplies by var_
        y_vars = y_vars_scaled * self.y_scaler.var_
        y_std = np.sqrt(y_vars)

        return y_pred, y_std

    def save_model(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.model_dict, f)

    @classmethod
    def load_model(cls, filename: str) -> "PCAGaussianProcess":
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)

        model = cls(
            n_components=model_dict["n_components"],
            kernel=model_dict["default_kernel"],
            mean_type=model_dict["default_mean_type"],
            test_size=model_dict["test_size"],
            hp_test_size=model_dict["hp_test_size"],
            n_restarts=model_dict["n_restarts"],
            random_state=model_dict["random_state"],
            hp_n_jobs=model_dict["hp_n_jobs"],
            hp_n_trials=model_dict["hp_n_trials"],
            hp_optimize=model_dict["hp_optimize"],
            whiten=model_dict["whiten"],
        )

        model.X_scaler = model_dict["X_scaler"]
        model.y_index = model_dict["y_index"]
        model.Xs = model_dict["X_scaled"]
        model.pca = model_dict["pca"]
        model.y_pca = model_dict["y_pca"]
        model.y_scaler = model_dict["y_scaler"]
        model.models = []

        for i in range(model.n_components):
            kernel = model_dict.get(f"kernel_{i + 1}", model_dict["default_kernel"])
            mean_type = model_dict.get(
                f"mean_type_{i + 1}", model_dict["default_mean_type"]
            )

            kernel = create_kernel(kernel, input_dim=model_dict["dim"])
            mean_function = create_mean_function(mean_type, input_dim=model_dict["dim"])

            gp_model = GPRegression(
                model.Xs,
                model.y_pca[:, i].reshape(-1, 1),
                kernel=kernel,
                mean_function=mean_function,
                normalizer=True,
            )
            gp_model[:] = model_dict[f"params_{i + 1}"]
            gp_model.update_model(True)
            model.models.append(gp_model)

        return model
