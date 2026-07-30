import abc
from typing import Optional, Tuple, Callable, Any, Literal

import corner
import numpy as np
from scipy import signal
from scipy.stats import multivariate_normal
from scipy.io import savemat
from scipy.optimize import minimize


class BaseMCMC(abc.ABC):
    """
    Abstract base class for MCMC samplers.

    Subclasses must implement:
      - prior(x: np.ndarray) -> float
      - likelihood(x: np.ndarray) -> float
      - sample(n_samples: int, burn_in: int = 0, **kwargs) -> np.ndarray

    Notes:
      - prior() and likelihood() may return probabilities or log-probabilities
        (or +/- np.inf). Use the `log` parameter on posterior() to control how
        they are combined (sum for log-values, multiply for probabilities).
      - Subclasses can accept variant-specific __init__ args; the base __init__
        provides a reproducible RNG via `seed`.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Minimal common initializer. Subclasses may accept other parameters
        but should call super().__init__(seed=...) if they want RNG support.
        """
        self.rng = np.random.default_rng(seed)
        # Storage for samples produced by `infer`.
        # For non-ensemble samplers: expected shape (n_samples, n_dim) or (n_samples,)
        # For ensemble samplers: expected shape (n_chains, n_samples, n_dim) or (n_chains, n_samples)
        self.samples: Optional[np.ndarray] = None
        # Whether this sampler produces parallel/ensemble chains (e.g. emcee)
        self.is_ensemble: bool = False

    @abc.abstractmethod
    def prior(self, x: np.ndarray, log: bool = True) -> float:
        """Return prior probability or log-prior for x."""

    @abc.abstractmethod
    def likelihood(self, x: np.ndarray, log: bool = True) -> float:
        """Return likelihood probability or log-likelihood for x."""

    def posterior(self, x: np.ndarray, *, log: bool = True) -> float:
        """
        Combine prior and likelihood to produce posterior.

        If log is True, assumes prior() and likelihood() return log-values and
        returns their sum. Otherwise multiplies them.

        This method intentionally keeps behavior simple — sampler implementations
        should document whether their prior/likelihood return log-values.
        """
        priorval = self.prior(x, log=log)
        likelihoodval = self.likelihood(x, log=log)

        if log:
            return priorval + likelihoodval
        return priorval * likelihoodval

    @property
    def has_samples(self) -> bool:
        """Return True if samples are present on this object."""
        return self.samples is not None

    def _check_samples(self) -> None:
        if not self.has_samples:
            raise RuntimeError(
                "No samples available. Run `infer()` to generate samples first."
            )

    @staticmethod
    def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation for a 1D array up to max_lag (inclusive).

        Returns array of length (max_lag+1) where index 0 is lag 0 (autocorr=1).
        """
        x = np.asarray(x)
        if x.size <= 1:
            return np.ones(max_lag + 1)

        # Use scipy's correlate for efficiency
        x_centered = x - x.mean()
        autocorr_full = signal.correlate(x_centered, x_centered, mode="full")
        # Take only the positive lags (second half)
        mid = len(autocorr_full) // 2
        autocorr = autocorr_full[mid : mid + max_lag + 1]
        # Normalize by lag-0 value
        return autocorr / autocorr[0] if autocorr[0] != 0 else np.ones(max_lag + 1)

    def compute_autocorrelation(self, max_lag: int = 100) -> np.ndarray:
        """Compute autocorrelation(s) for the stored samples.

        For non-ensemble samplers this returns an array of shape (n_dim, max_lag+1).
        For 1D samples shape will be (max_lag+1,).

        Raises RuntimeError if `infer()` has not been run.
        """
        self._check_samples()
        s = self.samples
        # Normalize shape to (n_samples, n_dim)
        if s.ndim == 1:
            # 1D chain
            return self._autocorr_1d(s, max_lag)
        if self.is_ensemble:
            # ensemble: (n_chains, n_samples, n_dim) or (n_chains, n_samples)
            if s.ndim == 2:
                # (n_chains, n_samples) -> treat each chain separately, then average
                per_chain = np.stack([self._autocorr_1d(chain, max_lag) for chain in s])
                return per_chain.mean(axis=0)
            # (n_chains, n_samples, n_dim)
            n_chains, n_samples, n_dim = s.shape
            # compute mean autocorr across chains for each dim
            result = np.zeros((n_dim, max_lag + 1))
            for d in range(n_dim):
                per_chain = np.stack(
                    [self._autocorr_1d(s[c, :, d], max_lag) for c in range(n_chains)]
                )
                result[d] = per_chain.mean(axis=0)
            return result
        # non-ensemble multi-dim: (n_samples, n_dim)
        if s.ndim == 2:
            n_samples, n_dim = s.shape
            result = np.zeros((n_dim, max_lag + 1))
            for d in range(n_dim):
                result[d] = self._autocorr_1d(s[:, d], max_lag)
            return result
        # fallback
        raise RuntimeError(
            "Unsupported samples array shape for autocorrelation: %s" % (s.shape,)
        )

    def compute_cross_correlation(self, max_lag: int = 100) -> np.ndarray:
        """Compute cross-correlation lags between parallel chains if ensemble.

        If `self.is_ensemble` is False, this falls back to `compute_autocorrelation`.

        For ensemble samplers, returns array of shape (n_dim, max_lag+1) representing
        the mean cross-correlation across chain pairs for each parameter dimension.
        """
        self._check_samples()
        if not self.is_ensemble:
            return self.compute_autocorrelation(max_lag=max_lag)

        s = self.samples
        if s.ndim == 2:
            # (n_chains, n_samples) - 1D parameter
            n_chains = s.shape[0]
            if n_chains < 2:
                return self.compute_autocorrelation(max_lag=max_lag)

            correlations = []
            for i in range(n_chains):
                for j in range(i + 1, n_chains):
                    # Use scipy.signal.correlate for cross-correlation
                    xi = s[i] - s[i].mean()
                    xj = s[j] - s[j].mean()
                    if xi.std() == 0 or xj.std() == 0:
                        correlations.append(np.ones(max_lag + 1))
                        continue

                    xcorr = signal.correlate(xi, xj, mode="full")
                    mid = len(xcorr) // 2
                    xcorr_pos = xcorr[mid : mid + max_lag + 1]
                    # Normalize by standard deviations
                    xcorr_norm = xcorr_pos / (xi.std() * xj.std() * len(xi))
                    correlations.append(xcorr_norm)

            return np.mean(correlations, axis=0)

        # (n_chains, n_samples, n_dim)
        n_chains, n_samples, n_dim = s.shape
        if n_chains < 2:
            return self.compute_autocorrelation(max_lag=max_lag)

        result = np.zeros((n_dim, max_lag + 1))
        for d in range(n_dim):
            correlations = []
            for i in range(n_chains):
                for j in range(i + 1, n_chains):
                    xi = s[i, :, d] - s[i, :, d].mean()
                    xj = s[j, :, d] - s[j, :, d].mean()
                    if xi.std() == 0 or xj.std() == 0:
                        correlations.append(np.ones(max_lag + 1))
                        continue

                    xcorr = signal.correlate(xi, xj, mode="full")
                    mid = len(xcorr) // 2
                    xcorr_pos = xcorr[mid : mid + max_lag + 1]
                    xcorr_norm = xcorr_pos / (xi.std() * xj.std() * len(xi))
                    correlations.append(xcorr_norm)

            result[d] = (
                np.mean(correlations, axis=0) if correlations else np.ones(max_lag + 1)
            )
        return result

    def summary_statistics(self) -> dict:
        """Return summary statistics computed from `self.samples`.

        Returns a dict mapping statistic names to arrays.
        For multi-dim parameters, each statistic is an array of length n_dim.
        Raises RuntimeError if `infer()` has not been run.
        The dict has the following keys:
            - "mean": Mean of samples
            - "std": Standard deviation of samples
            - "median": Median of samples
        """
        self._check_samples()
        s = self.samples

        # Normalize to (n_samples, n_dim) by collapsing ensemble chains if needed
        if self.is_ensemble:
            collapsed = s.mean(axis=0)  # Average across chains
        else:
            collapsed = s

        # Use numpy for all statistics
        if collapsed.ndim == 1:
            # 1D case
            stats = {
                "mean": float(np.mean(collapsed)),
                "std": float(np.std(collapsed, ddof=1)),
                "median": float(np.median(collapsed)),
            }
        else:
            # Multi-dimensional case - use axis=0 for all operations
            stats = {
                "mean": np.mean(collapsed, axis=0),
                "std": np.std(collapsed, axis=0, ddof=1),
                "median": np.median(collapsed, axis=0),
            }

        return stats

    @abc.abstractmethod
    def infer(self, n_samples: int, burn_in: int = 0, **kwargs) -> np.ndarray:
        """
        Draw samples from the posterior. Subclasses must implement sampling logic.

        Returns:
            np.ndarray: Array of samples (shape and dtype up to implementation).
        """
        # At least two samples are necessary
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if "thin" in kwargs and (
            not isinstance(kwargs["thin"], int) or kwargs["thin"] < 1
        ):
            raise ValueError("thin must be a positive integer if provided")

    def evidence(self, N_1=5000, N_2=5000) -> float:
        """
        Estimate model evidence using Bridge sampling
        1. Draw N_1 samples from the posterior: {x_i} ~ p(x)
        2. Draw N_2 samples from a proposal distribution q(x) (e.g. Gaussian approx from posterior)
        3. Compute evidence estimate using the bridge sampling formula
        4. Iterate until convergence

        Inputs:
            N_1 (int): Number of posterior samples to use per iteration
            N_2 (int): Number of proposal samples to use per iteration
        Returns:
            evidence_estimate (float): Estimated model evidence
        Raises:
            RuntimeError: If `infer()` has not been run to generate samples.
        """

        self._check_samples()

        if self.is_ensemble:
            # Collapse ensemble chains by averaging
            samples = self.samples.mean(axis=0)
        else:
            samples = self.samples
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]  # Make it 2D for consistency
        n_samples, n_dim = samples.shape
        if n_samples < 2:
            raise RuntimeError(
                "At least two samples are required to estimate evidence."
            )
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        proposal_dist = multivariate_normal(mean=mean, cov=cov)
        # Initial evidence estimate
        evidence_estimate = 1.0

        # Optimal ratio constant
        optimal_ratio = N_2 / (len(samples) + N_2)

        def bridge_function(x, evidence: float) -> float:
            p = self.posterior(x, log=False)
            q = proposal_dist.pdf(x)
            return (
                1.0 / (optimal_ratio * p + (1 - optimal_ratio) * evidence * q)
                if (p > 0 and q > 0)
                else 0.0
            )

        delta_evidence = np.inf
        iteration = 0
        while (
            delta_evidence > 1e-4 and iteration < 10
        ):  # Limit to 10 iterations for convergence
            iteration += 1
            old_evidence = evidence_estimate
            # Draw samples from proposal
            numerator = (
                1
                / N_2
                * sum(
                    [
                        self.posterior(x, log=False)
                        * bridge_function(x, evidence_estimate)
                        for x in proposal_dist.rvs(size=N_2)
                    ]
                )
            )
            denominator = (
                1
                / len(samples)
                * sum(
                    [
                        bridge_function(x, evidence_estimate) * proposal_dist.pdf(x)
                        for x in samples
                    ]
                )
            )
            evidence_estimate = (
                numerator / denominator if denominator > 0 else evidence_estimate
            )
            delta_evidence = (
                abs(evidence_estimate - old_evidence) / abs(old_evidence)
                if old_evidence != 0
                else np.inf
            )
            # print(f"Iteration {iteration}: Evidence estimate = {evidence_estimate}, Change

        self.converged = delta_evidence <= 1e-4
        self.evidence_estimate = evidence_estimate

        return evidence_estimate

    def max_a_posteriori(
        self,
        n_starts: Optional[int] = None,
        method: Literal["Nelder-Mead", "Powell", "L-BFGS-B"] = "Nelder-Mead",
    ) -> dict:
        """
        Estimate Maximum A Posteriori (MAP) using optimization.

        Args:
            n_starts: Number of optimization starts (multi-start for robustness)
            method: scipy.optimize method ('Nelder-Mead', 'Powell', 'L-BFGS-B')

        Returns:
            dict with keys: 'map_estimate', 'map_log_posterior', 'optimization_success'
        """

        self._check_samples()
        if n_starts is None:
            if self.is_ensemble:
                if self.samples.ndim == 3:
                    n_starts = int(5 * self.samples.shape[-1])  # 5 times the dimension
                elif self.samples.ndim == 2:
                    n_starts = 10
            else:
                if self.samples.ndim == 2:
                    n_starts = int(5 * self.samples.shape[1])  # 5 times the dimension
                elif self.samples.ndim == 1:
                    n_starts = 10

        # Objective: negative log posterior (for minimization)
        def neg_log_posterior(x):
            try:
                return -self.posterior(x, log=True)
            except Exception:
                return np.inf

        # Smart initialization: use best samples as starting points
        samples = self.samples
        if self.is_ensemble:
            samples = samples.reshape(-1, samples.shape[-1])  # Flatten chains

        # Evaluate posterior at all samples to find best ones
        log_posteriors = [self.posterior(x, log=True) for x in samples]
        best_indices = np.argsort(log_posteriors)[-n_starts:]  # Top n_starts samples

        best_result = None
        best_value = np.inf

        for idx in best_indices:
            result = minimize(
                neg_log_posterior,
                x0=samples[idx],
                method=method,
                options={"maxiter": 1000},
            )

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result

        if best_result is None:
            raise RuntimeError("MAP optimization failed for all starting points")

        return best_result.x

    def propagate(
        self, model: Any, x_d: Optional[np.ndarray] = None, full_cov: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through a given model.
        Inputs:
            model (BaseModel): A fitted model with a predict method
            x_d (np.ndarray): Input data of shape (n_samples, n_features);
                                will be appended to each row of self.samples

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and covariance/variance of model predictions

        Note:
            - Requires that `infer()` has been run to generate samples.
            - Assumes model.predict() can handle batch inputs.
            - If model outputs 1D predictions, returns variance; for multi-D, returns covariance matrix.
            - Ignores interpolation uncertainty for simplicity; more complex methods TBD...
        """
        self._check_samples()
        if x_d is not None:
            samples = np.hstack(
                [self.samples, np.tile(x_d, (self.samples.shape[0], 1))]
            )
        else:
            samples = self.samples

        # Check that the model has a predict method
        if not hasattr(model, "predict"):
            raise AttributeError("Provided model does not have a 'predict' method.")

        if not isinstance(model.predict, Callable):
            raise AttributeError("'predict' attribute of the model is not callable.")

        # Get predictions
        preds = model.predict(samples)

        if isinstance(preds, tuple):
            preds = preds[0]  # Ignore uncertainty for now

        if preds.ndim == 1:
            return preds.mean(), preds.var(ddof=1)
        else:
            if full_cov:
                return preds.mean(axis=0), np.cov(preds, rowvar=False)
            else:
                return preds.mean(axis=0), np.var(preds, axis=0, ddof=1)

    def save_statistics(self, file_path: str, full_cov: bool = False) -> None:
        """
        Save summary statistics and evidence estimate to a text file.

        Args:
            file_path: Path to the output text file.
        """
        stats = {}
        self._check_samples()
        # Find ensemble-wide mean if needed
        if self.is_ensemble:
            samples = self.samples.mean(axis=0)
        else:
            samples = self.samples
        stats["samples"] = samples

        if samples.ndim == 1:
            stats["mean"] = float(np.mean(samples))
            stats["variance"] = float(np.std(samples, ddof=1))
        else:
            if full_cov:
                stats["mean"] = np.mean(samples, axis=0)
                stats["variance"] = np.cov(samples, rowvar=False)
            else:
                stats["mean"] = np.mean(samples, axis=0)
                stats["variance"] = np.var(samples, axis=0, ddof=1)

        # Calculate effective sample size
        autocorr = self.compute_autocorrelation(max_lag=100)
        ess = []
        for d in range(autocorr.shape[0] if autocorr.ndim > 1 else 1):
            acf = autocorr[d] if autocorr.ndim > 1 else autocorr
            # Use initial positive sequence
            positive_acf = acf[acf > 0]
            tau = 1 + 2 * np.sum(positive_acf[1:])  # Exclude lag 0
            n = samples.shape[0]
            ess.append(n / tau if tau > 0 else n)
        stats["effective_sample_size"] = ess if len(ess) > 1 else ess[0]
        if not hasattr(self, "evidence_estimate"):
            self.evidence(N_1=5000, N_2=5000)
        stats["evidence_estimate"] = float(self.evidence_estimate)
        stats["evidence_converged"] = bool(self.converged)
        # Save to file as JSON for simplicity
        _, file_ext = file_path.rsplit(".", 1)
        if file_ext != "mat":
            raise ValueError("Only .mat file format is supported currently")
        savemat(file_path, stats)

    @property
    def effective_sample_size(self) -> Optional[np.ndarray]:
        """Return effective sample size(s) computed from autocorrelation.

        For non-ensemble samplers, returns array of length n_dim or a scalar for 1D.
        For ensemble samplers, returns mean effective sample size across chains.

        Returns None if `infer()` has not been run.
        """
        if not self.has_samples:
            return None

        autocorr = self.compute_autocorrelation(max_lag=100)
        ess = []
        for d in range(autocorr.shape[0] if autocorr.ndim > 1 else 1):
            acf = autocorr[d] if autocorr.ndim > 1 else autocorr
            # Use initial positive sequence
            positive_acf = acf[acf > 0]
            tau = 1 + 2 * np.sum(positive_acf[1:])  # Exclude lag 0
            n = self.samples.shape[0] if not self.is_ensemble else self.samples.shape[1]
            ess.append(n / tau if tau > 0 else n)

        return np.array(ess) if len(ess) > 1 else ess[0]

    def create_corner_plots(
        self, dim: Optional[int] = None, labels: Optional[list[str]] = None
    ) -> Any:
        """Create corner plots for the samples.

        Args:
            dim: If specified, only plot the first `dim` dimensions of the samples.
                 If None, plot all dimensions.
            labels: Optional list of labels for the dimensions.
        """

        self._check_samples()
        if self.is_ensemble:
            samples = self.samples.mean(axis=0)
        else:
            samples = self.samples

        if dim is not None and samples.shape[1] > dim:
            samples = samples[:, :dim]

        return corner.corner(samples, labels=labels)
