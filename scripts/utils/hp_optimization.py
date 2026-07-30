from typing import Optional
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from GPy.models import GPRegression
from ms_amp_ai_use_cases.utils import create_kernel, create_mean_function


def optimize_hyperparameters_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 10,
    hp_test_size: float = 0.2,
    hp_n_jobs: int = 1,
    hp_n_trials: int = 20,
    random_state: Optional[int] = None,
) -> dict:
    """
    Optimize hyperparameters for a Gaussian Process model using Optuna.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        n_restarts (int): Number of restarts for hyperparameter optimization.
        hp_test_size (float): Proportion of data to use for hyperparameter tuning.
        hp_n_jobs (int): Number of parallel jobs for hyperparameter optimization.
        hp_n_trials (int): Number of trials for hyperparameter optimization.
        random_state (Optional[int]): Random seed for reproducibility.
    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    def func(trial, Xtrain, ytrain, Xval, yval, n_restarts):
        kernel_choice = trial.suggest_categorical(
            "kernel", ["RBF", "Matern32", "Matern52", "RatQuad"]
        )
        kernel = create_kernel(kernel_choice, input_dim=Xtrain.shape[1])

        mean_choice = trial.suggest_categorical(
            "mean_type", ["constant", "linear", "none"]
        )

        mean_function = create_mean_function(
            mean_choice, input_dim=Xtrain.shape[1], output_dim=ytrain.shape[1]
        )

        model = GPRegression(
            Xtrain, ytrain, kernel=kernel, mean_function=mean_function, normalizer=True
        )
        model.optimize_restarts(num_restarts=n_restarts, verbose=False)

        yval_pred, _ = model.predict(Xval)
        mse = np.mean((yval - yval_pred) ** 2)
        return mse

    Xtrain, Xval, ytrain, yval = train_test_split(
        X, y, test_size=hp_test_size, random_state=random_state
    )

    objective = lambda trial: func(trial, Xtrain, ytrain, Xval, yval, n_restarts)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=hp_n_trials, n_jobs=hp_n_jobs)
    return study.best_params
