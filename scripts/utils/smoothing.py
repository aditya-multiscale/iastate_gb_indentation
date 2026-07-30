import numpy as np
from typing import Tuple


def smoothed(
    mu: np.ndarray, var: np.ndarray, alpha=0.8, beta=0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Private module: Smoothes predictions using holt exponential moving average

    Args:
        mu: Array of mean predictions
        var: Array of variance predictions
    Returns:
        mu: Array of mean predictions (n_samples,n_outputs)
        var: Array of variance predictions (n_samples,n_outputs)

    """

    def holt_ema(z, alpha=alpha, beta=beta):
        s, b = np.zeros_like(z), np.zeros_like(z)
        s[0] = z[0]
        b[0] = z[1] - z[0]

        for t in range(1, len(z)):
            s[t] = alpha * z[t] + (1 - alpha) * (s[t - 1] + b[t - 1])
            b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
        return s, b

    for i in range(mu.shape[0]):
        mu[i, :], var[i, :] = holt_ema(mu[i, :])[0], holt_ema(var[i, :])[0]

    return mu, var
