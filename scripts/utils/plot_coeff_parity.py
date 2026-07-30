import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def plot_sindycoeff_parity(
    coeff: np.ndarray, coeffmean: np.ndarray, coeffvar: np.ndarray
) -> None:
    """
    Plot parity plots for the coefficients.

    Args:
        coeff (np.ndarray): Ground truth coefficients.
        coeffmean (np.ndarray): Predicted coefficient means.
        coeffvar (np.ndarray): Predicted coefficient variances.
    """
    rc("font", size=10)
    rc("font", family="serif")
    rc("axes", labelsize=10)
    rc("text", usetex=True)

    if coeff.shape != coeffmean.shape or coeff.shape != coeffvar.shape:
        raise ValueError("Shapes of all arrays must match")
    if coeff.ndim != 2:
        raise ValueError("2-D arrays needed: (n_samples x n_coefficients)")
    dimcoeff = coeff.shape[1]

    fig_kw = {"figsize": (4, int(dimcoeff * 5)), "dpi": 300}
    fig, ax = plt.subplots(
        nrows=dimcoeff, ncols=1, gridspec_kw={"wspace": 0.4, "hspace": 0.2}, **fig_kw
    )
    for d in range(dimcoeff):
        ax[d].errorbar(
            coeff[:, d],
            coeffmean[:, d],
            yerr=np.abs(coeffvar[:, d]) ** 0.5,
            marker="o",
            ls="",
            ecolor="k",
            mfc="k",
            mec=None,
        )
        concat = np.concatenate([coeff[:, d], coeffmean[:, d]])
        xlim = [min(concat), max(concat)]
        if d == 0:
            xlabel, ylabel = (
                "FE Prediction - Initial Current (mA/m)",
                "ML Prediction - Initial Current (mA/m)",
            )
        else:
            xlabel = "FE Prediction - Coefficient %d" % (d - 1)
            ylabel = "GP Prediction - Coefficient %d" % (d - 1)
        ax[d].set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return


def plot_basiscoeff_parity(
    coeff: np.ndarray, coeffmean: np.ndarray, coeffvar: np.ndarray
) -> None:
    """
    Plot parity plots for the coefficients.

    Args:
        coeff (np.ndarray): Ground truth coefficients.
        coeffmean (np.ndarray): Predicted coefficient means.
        coeffvar (np.ndarray): Predicted coefficient variances.
    """
    rc("font", size=14)
    rc("font", family="serif")
    rc("axes", labelsize=14)
    rc("text", usetex=True)

    if coeff.shape != coeffmean.shape or coeff.shape != coeffvar.shape:
        raise ValueError("Shapes of all arrays must match")
    if coeff.ndim != 2:
        raise ValueError("2-D arrays needed: (n_samples x n_coefficients)")
    dimcoeff = coeff.shape[1]

    fig_kw = {"figsize": (4, int(dimcoeff * 5)), "dpi": 300}
    fig, ax = plt.subplots(
        nrows=dimcoeff, ncols=1, gridspec_kw={"wspace": 0.4, "hspace": 0.2}, **fig_kw
    )
    for d in range(dimcoeff):
        ax[d].errorbar(
            coeff[:, d],
            coeffmean[:, d],
            yerr=np.abs(coeffvar[:, d]) ** 0.5,
            marker="o",
            ls="",
            ecolor="k",
            mfc="k",
            mec=None,
        )
        concat = np.concatenate([coeff[:, d], coeffmean[:, d]])
        xlim = [min(concat), max(concat)]
        xlabel = "FE Prediction - Coefficient %d" % (d)
        ylabel = "GP Prediction - Coefficient %d" % (d)
        ax[d].set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return
