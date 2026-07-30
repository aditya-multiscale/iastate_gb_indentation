"""
Delayed Rejection Adaptive Metropolis (DRAM) MCMC Sampler.

Implements the DRAM algorithm combining:
- Delayed Rejection (DR): When a proposal is rejected, make a second proposal with smaller step
- Adaptive Metropolis (AM): Adapt proposal covariance based on chain history

References:
    Haario, H., Laine, M., Mira, A., & Saksman, E. (2006).
    "DRAM: Efficient adaptive MCMC"
    Statistics and Computing, 16(4), 339-354.
"""

from typing import Callable, Optional, Sequence, Tuple, Union, Literal

import numpy as np
from scipy.linalg import cholesky, LinAlgError

from .utils import BaseMCMC
from tqdm import tqdm


class DelayedRejectionAdaptiveMetropolis(BaseMCMC):
    """
    Delayed Rejection Adaptive Metropolis (DRAM) sampler.

    Combines delayed rejection with adaptive Metropolis for improved sampling
    efficiency. When initial_position is None, uses MAP estimate from a
    preliminary run as the starting point.

    Args:
        prior: Callable returning log-prior for parameter vector
        likelihood: Callable returning log-likelihood for parameter vector
        n_dim: Number of dimensions (required if initial_position is None)
        initial_position: Starting point for the chain. If None, uses MAP estimate
            from a preliminary sampling run.
        initial_cov: Initial proposal covariance matrix (default: 0.1 * I)
        dr_scale: Scaling factor for second-stage delayed rejection proposal (default: 0.2)
        sd_scale: Scaling factor for adapted covariance, typically 2.4^2/d (default: auto)
        adaptation_start: Start adapting covariance after this many iterations (default: 100)
        adaptation_interval: Update covariance every N iterations (default: 10)
        limits: Optional parameter bounds as (lower, upper) tuple
        log: Whether prior/likelihood return log values (default: True)
        seed: Random seed for reproducibility
        eps: Small value for covariance regularization (default: 1e-6)
    """

    def __init__(
        self,
        prior: Callable[[np.ndarray], float],
        likelihood: Callable[[np.ndarray], float],
        n_dim: Optional[int] = None,
        initial_position: Optional[Union[Sequence[float], np.ndarray]] = None,
        initial_cov: Optional[np.ndarray] = None,
        dr_scale: float = 0.2,
        sd_scale: Optional[float] = None,
        adaptation_start: int = 100,
        adaptation_interval: int = 10,
        limits: Optional[
            Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]
        ] = None,
        log: bool = True,
        seed: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(seed=seed)

        if not callable(prior) or not callable(likelihood):
            raise TypeError("prior and likelihood must be callables")

        self._prior = prior
        self._likelihood = likelihood
        self.log = bool(log)

        # Determine dimensionality
        if initial_position is not None:
            self.current = np.atleast_1d(np.array(initial_position, dtype=float))
            self.dim = self.current.size
            self._use_map_start = False
        elif n_dim is not None:
            self.dim = int(n_dim)
            self.current = None  # Will be set via MAP
            self._use_map_start = True
        else:
            raise ValueError("Either initial_position or n_dim must be provided")

        # Initialize covariance
        if initial_cov is None:
            self.initial_cov = 0.1 * np.eye(self.dim)
        else:
            self.initial_cov = np.asarray(initial_cov, dtype=float)
            if self.initial_cov.shape != (self.dim, self.dim):
                raise ValueError(f"initial_cov must be {self.dim}x{self.dim} matrix")

        self.cov = self.initial_cov.copy()

        # DRAM parameters
        if not 0 < dr_scale < 1:
            raise ValueError("dr_scale must be in (0, 1)")
        self.dr_scale = float(dr_scale)

        # Optimal scaling: 2.4^2 / d for Gaussian targets
        if sd_scale is None:
            self.sd_scale = (2.4**2) / self.dim
        else:
            if sd_scale <= 0:
                raise ValueError("sd_scale must be positive")
            self.sd_scale = float(sd_scale)

        if adaptation_start < 0:
            raise ValueError("adaptation_start must be non-negative")
        self.adaptation_start = int(adaptation_start)

        if adaptation_interval < 1:
            raise ValueError("adaptation_interval must be at least 1")
        self.adaptation_interval = int(adaptation_interval)

        # Regularization
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

        # Bounds
        if limits is not None:
            lower, upper = limits
            lower_arr = None if lower is None else np.asarray(lower, dtype=float)
            upper_arr = None if upper is None else np.asarray(upper, dtype=float)
            if lower_arr is not None and lower_arr.size != self.dim:
                raise ValueError("lower limits must match dimension")
            if upper_arr is not None and upper_arr.size != self.dim:
                raise ValueError("upper limits must match dimension")
            self.limits = (lower_arr, upper_arr)
        else:
            self.limits = (None, None)

        # Validate initial position if provided
        if self.current is not None:
            if not self._within_limits(self.current):
                raise ValueError("initial_position must be within limits")
            try:
                p0 = float(self._prior(self.current))
            except Exception as e:
                raise RuntimeError(f"prior(initial_position) raised: {e}")
            if not np.isfinite(p0):
                raise ValueError("prior(initial_position) must be finite")

        # Initialize tracking
        self.samples = None
        self.is_ensemble = False
        self._accept_count = 0
        self._accept_count_dr = 0  # Second-stage acceptances
        self._proposal_count = 0
        self._dr_proposal_count = 0

        # Running statistics for AM adaptation
        self._chain_mean = np.zeros(self.dim)
        self._chain_cov = np.zeros((self.dim, self.dim))
        self._n_stored = 0

    def prior(self, x: np.ndarray, log: bool = True) -> float:
        """Return prior probability or log-prior for x."""
        val = float(self._prior(np.asarray(x, dtype=float)))
        if not log:
            return np.exp(val) if np.isfinite(val) else 0.0
        return val

    def likelihood(self, x: np.ndarray, log: bool = True) -> float:
        """Return likelihood probability or log-likelihood for x."""
        val = float(self._likelihood(np.asarray(x, dtype=float)))
        if not log:
            return np.exp(val) if np.isfinite(val) else 0.0
        return val

    def _within_limits(self, x: np.ndarray) -> bool:
        """Check if x is within parameter bounds."""
        lower, upper = self.limits
        if lower is not None and np.any(x < lower):
            return False
        if upper is not None and np.any(x > upper):
            return False
        return True

    def _log_posterior(self, x: np.ndarray) -> float:
        """Compute log-posterior at x."""
        if not self._within_limits(x):
            return -np.inf
        try:
            if self.log:
                return self.prior(x, log=True) + self.likelihood(x, log=True)
            else:
                p = self.prior(x, log=False) * self.likelihood(x, log=False)
                return np.log(p) if p > 0 else -np.inf
        except Exception:
            return -np.inf

    def _proposal_logpdf(
        self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> float:
        """Compute log-pdf of multivariate normal proposal."""
        try:
            diff = x - mean
            # Add regularization for numerical stability
            cov_reg = cov + self.eps * np.eye(self.dim)
            L = cholesky(cov_reg, lower=True)
            solve = np.linalg.solve(L, diff)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            return -0.5 * (
                self.dim * np.log(2 * np.pi) + log_det + np.dot(solve, solve)
            )
        except LinAlgError:
            return -np.inf

    def _update_running_statistics(self, x: np.ndarray) -> None:
        """Update running mean and covariance for AM adaptation."""
        self._n_stored += 1
        n = self._n_stored

        # Welford's online algorithm for mean and covariance
        delta = x - self._chain_mean
        self._chain_mean = self._chain_mean + delta / n

        if n > 1:
            delta2 = x - self._chain_mean
            # Update covariance incrementally
            self._chain_cov = (n - 2) / (n - 1) * self._chain_cov + np.outer(
                delta, delta2
            ) / n

    def _get_adapted_covariance(self) -> np.ndarray:
        """Get the adapted proposal covariance with regularization."""
        if self._n_stored < 2:
            return self.initial_cov.copy()

        # AM formula: sd_scale * empirical_cov + eps * I
        adapted = self.sd_scale * self._chain_cov + self.eps * np.eye(self.dim)
        return adapted

    def _propose(self, current: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate proposal from multivariate normal."""
        try:
            return current + self.rng.multivariate_normal(np.zeros(self.dim), cov)
        except (LinAlgError, ValueError):
            # Fallback to diagonal if covariance is singular
            return current + self.rng.normal(0, np.sqrt(np.diag(cov) + self.eps))

    def _delayed_rejection_step(
        self,
        current: np.ndarray,
        current_lp: float,
        prop1: np.ndarray,
        prop1_lp: float,
        cov1: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Perform second-stage delayed rejection.

        Returns:
            Tuple of (new_position, new_log_posterior, accepted)
        """
        self._dr_proposal_count += 1

        # Second-stage covariance is scaled down
        cov2 = self.dr_scale**2 * cov1

        # Propose from scaled covariance centered at current
        prop2 = self._propose(current, cov2)

        if not self._within_limits(prop2):
            return current, current_lp, False

        prop2_lp = self._log_posterior(prop2)

        if not np.isfinite(prop2_lp):
            return current, current_lp, False

        # DR acceptance probability (Mira 2001, Haario et al. 2006)
        # α2 = min(1, π(y2)/π(x) * q1(y1|y2)/q1(y1|x) * (1 - α1(y2,y1))/(1 - α1(x,y1)))

        # Compute auxiliary terms
        # q1(y1|y2): probability of proposing prop1 from prop2 using cov1
        log_q1_y1_given_y2 = self._proposal_logpdf(prop1, prop2, cov1)
        # q1(y1|x): probability of proposing prop1 from current using cov1
        log_q1_y1_given_x = self._proposal_logpdf(prop1, current, cov1)

        # (1 - α1(y2, y1)): rejection prob if we were at prop2 and proposed prop1
        if np.isfinite(prop1_lp) and np.isfinite(prop2_lp):
            log_alpha1_y2_y1 = min(0.0, prop1_lp - prop2_lp)
            one_minus_alpha1_y2_y1 = 1.0 - np.exp(log_alpha1_y2_y1)
        else:
            one_minus_alpha1_y2_y1 = 1.0

        # (1 - α1(x, y1)): rejection prob at current proposing prop1 (already rejected)
        if np.isfinite(prop1_lp) and np.isfinite(current_lp):
            log_alpha1_x_y1 = min(0.0, prop1_lp - current_lp)
            one_minus_alpha1_x_y1 = 1.0 - np.exp(log_alpha1_x_y1)
        else:
            one_minus_alpha1_x_y1 = 1.0

        # Avoid division by zero
        if one_minus_alpha1_x_y1 < 1e-300:
            return current, current_lp, False

        # Log acceptance ratio
        log_ratio = (
            prop2_lp
            - current_lp
            + log_q1_y1_given_y2
            - log_q1_y1_given_x
            + np.log(max(one_minus_alpha1_y2_y1, 1e-300))
            - np.log(one_minus_alpha1_x_y1)
        )

        log_alpha2 = min(0.0, log_ratio)

        if np.log(self.rng.random()) < log_alpha2:
            self._accept_count_dr += 1
            return prop2, prop2_lp, True

        return current, current_lp, False

    def _initialize_from_map(self, n_warmup: int = 500, 
                             map_optimization_method: Literal["Nelder-Mead", "Powell", "L-BFGS-B"] = "Nelder-Mead") -> np.ndarray:
        """
        Initialize starting position using MAP estimate.

        Runs a short preliminary chain to enable MAP estimation.
        """
        # Start from prior sample or center of bounds
        if self.limits[0] is not None and self.limits[1] is not None:
            # Use center of bounds
            start = 0.5 * (self.limits[0] + self.limits[1])
        elif self.limits[0] is not None:
            start = self.limits[0] + 1.0
        elif self.limits[1] is not None:
            start = self.limits[1] - 1.0
        else:
            start = np.zeros(self.dim)

        # Verify start is valid
        if not np.isfinite(self._log_posterior(start)):
            # Try random positions
            for _ in range(100):
                start = self.rng.standard_normal(self.dim)
                if self.limits[0] is not None:
                    start = np.maximum(start, self.limits[0] + 0.1)
                if self.limits[1] is not None:
                    start = np.minimum(start, self.limits[1] - 0.1)
                if np.isfinite(self._log_posterior(start)):
                    break
            else:
                raise RuntimeError(
                    "Could not find valid starting position for MAP initialization"
                )

        # Run preliminary chain
        current = start.copy()
        current_lp = self._log_posterior(current)
        preliminary_samples = [current.copy()]

        cov = self.initial_cov.copy()

        for _ in range(n_warmup):
            prop = self._propose(current, cov)
            prop_lp = self._log_posterior(prop)

            if np.isfinite(prop_lp):
                log_alpha = min(0.0, prop_lp - current_lp)
                if np.log(self.rng.random()) < log_alpha:
                    current = prop
                    current_lp = prop_lp

            preliminary_samples.append(current.copy())

        # Store preliminary samples temporarily
        self.samples = np.array(preliminary_samples)

        # Get MAP estimate
        try:
            map_estimate = self.max_a_posteriori(n_starts=min(5, n_warmup // 10), method=map_optimization_method)
        except RuntimeError:
            # If MAP fails, use best sample
            log_posts = [self._log_posterior(s) for s in preliminary_samples]
            best_idx = np.argmax(log_posts)
            map_estimate = preliminary_samples[best_idx]

        map_estimate = self.max_a_posteriori(n_starts=None, method=map_optimization_method)

        # Clear preliminary samples
        self.samples = None

        return np.atleast_1d(map_estimate)

    def infer(
        self,
        n_samples: int,
        burn_in: int = 0,
        thin: int = 1,
        n_warmup_for_map: int = 500,
        map_optimization_method: Literal["Nelder-Mead", "Powell", "L-BFGS-B"] = "Nelder-Mead",
    ) -> np.ndarray:
        """
        Run DRAM sampling.

        Args:
            n_samples: Number of samples to generate (after burn-in and thinning)
            burn_in: Number of burn-in iterations
            thin: Thinning factor
            n_warmup_for_map: Number of warmup iterations for MAP initialization
                (only used if initial_position was None)

        Returns:
            Array of samples with shape (n_samples, dim) or (n_samples,) if 1D
        """
        super().infer(n_samples=n_samples, burn_in=burn_in, thin=thin)

        # Initialize starting position
        if self._use_map_start or self.current is None:
            self.current = self._initialize_from_map(n_warmup=n_warmup_for_map,
                                                     map_optimization_method=map_optimization_method,
                                                    )

        total_needed = burn_in + n_samples * thin

        current = np.array(self.current, dtype=float)
        current_lp = self._log_posterior(current)

        if not np.isfinite(current_lp):
            raise ValueError("Starting position has non-finite log-posterior")

        samples = []
        kept = 0

        # Reset counters
        self._accept_count = 0
        self._accept_count_dr = 0
        self._proposal_count = 0
        self._dr_proposal_count = 0

        for it in tqdm(range(int(total_needed)), desc="DRAM Sampling", unit="iter"):
            # Get current proposal covariance
            if it >= self.adaptation_start and self._n_stored >= 2:
                cov = self._get_adapted_covariance()
            else:
                cov = self.initial_cov

            # Stage 1: Standard MH proposal
            prop1 = self._propose(current, cov)
            self._proposal_count += 1

            prop1_lp = self._log_posterior(prop1)

            accepted = False

            if np.isfinite(prop1_lp):
                log_alpha1 = min(0.0, prop1_lp - current_lp)
                if np.log(self.rng.random()) < log_alpha1:
                    # Accept first-stage proposal
                    current = prop1
                    current_lp = prop1_lp
                    self._accept_count += 1
                    accepted = True

            # Stage 2: Delayed rejection if first proposal rejected
            if not accepted:
                current, current_lp, accepted = self._delayed_rejection_step(
                    current, current_lp, prop1, prop1_lp, cov
                )

            # Update running statistics for adaptation
            self._update_running_statistics(current)

            # Store sample after burn-in with thinning
            if it >= burn_in and ((it - burn_in) % thin == 0):
                samples.append(current.copy())
                kept += 1
                if kept >= n_samples:
                    break

        samples_arr = np.asarray(samples)

        # Reshape for 1D case
        if samples_arr.ndim == 2 and samples_arr.shape[1] == 1:
            samples_arr = samples_arr.reshape(-1)

        self.samples = samples_arr
        self.current = np.atleast_1d(current)
        self.cov = self._get_adapted_covariance()

        return self.samples

    @property
    def acceptance_rate(self) -> float:
        """Return first-stage acceptance rate."""
        if self._proposal_count <= 0:
            return 0.0
        return float(self._accept_count) / float(self._proposal_count)

    @property
    def dr_acceptance_rate(self) -> float:
        """Return second-stage (delayed rejection) acceptance rate."""
        if self._dr_proposal_count <= 0:
            return 0.0
        return float(self._accept_count_dr) / float(self._dr_proposal_count)

    @property
    def total_acceptance_rate(self) -> float:
        """Return combined acceptance rate (stage 1 + stage 2)."""
        total_proposals = self._proposal_count
        total_accepts = self._accept_count + self._accept_count_dr
        if total_proposals <= 0:
            return 0.0
        return float(total_accepts) / float(total_proposals)

    @property
    def acceptance_count(self) -> int:
        """Return total number of accepted proposals (both stages)."""
        return int(self._accept_count + self._accept_count_dr)

    @property
    def n_proposals(self) -> int:
        """Return total number of first-stage proposals."""
        return int(self._proposal_count)

    @property
    def adapted_covariance(self) -> np.ndarray:
        """Return the current adapted proposal covariance matrix."""
        return self._get_adapted_covariance()
