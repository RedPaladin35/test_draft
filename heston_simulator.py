"""
numerics/heston_simulator.py
-----------------------------
Monte Carlo path simulation under the Heston stochastic volatility model.

The Heston Joint Process (under Q)
------------------------------------
    dS_t = (r - q) S_t dt + √v_t S_t dW_t^S
    dv_t = κ(v̄ - v_t) dt + ξ √v_t dW_t^v
    dW_t^S dW_t^v = ρ dt

Unlike GBM (where σ is constant), the diffusion coefficient √v_t
evolves stochastically. This means:
  - At each time step, the volatility governing S is different
  - The correlation ρ < 0 causes vol to spike when spot falls
  - The forward distribution of S is non-lognormal (skewed, fat-tailed)

This is why Heston paths produce different barrier and Asian prices
than flat BS paths even when both are calibrated to the same vanilla surface.

Discretisation Schemes
-----------------------
Euler-Maruyama (simple, O(√dt) strong error):
    v_{t+dt} = v_t + κ(v̄ - v_t)dt + ξ√v_t √dt Z_v
    log(S_{t+dt}) = log(S_t) + (r-q-v_t/2)dt + √v_t √dt Z_S

Problem: Euler can produce negative variance v_t < 0 when ξ is large
or dt is not small enough. Standard fixes:

  1. Full truncation:   v_{t+dt} = max(v_{t+dt}, 0)     [used here]
  2. Reflection:        v_{t+dt} = |v_{t+dt}|
  3. Partial truncation: use max(v_t, 0) in diffusion only

Milstein Correction (O(dt) strong error):
Adds a correction term to the variance process:
    v_{t+dt} = v_t + κ(v̄ - v_t)dt + ξ√v_t √dt Z_v + (1/4)ξ²(Z_v²-1)dt

This reduces the strong discretisation error from O(√dt) to O(dt),
which means you need ~10x fewer time steps for the same path accuracy.
Important for path-dependent options (barriers, Asians) that are
sensitive to the vol path, not just terminal distribution.

QE Scheme (Andersen 2008) - Best in class, not implemented here:
The Quadratic-Exponential scheme matches the conditional distribution
of v_t analytically at each step, virtually eliminating bias for any dt.
It is the gold standard for production Heston simulation but more complex
to implement. The Milstein scheme is a good practical compromise.

Correlated Brownians
--------------------
W_S and W_v must be correlated with correlation ρ. Generated via
Cholesky decomposition:
    Z_S = ρ Z_v + √(1-ρ²) Z_⊥
where Z_v, Z_⊥ are independent standard normals.

References
----------
Andersen, L. (2008). "Simple and efficient simulation of the Heston
stochastic volatility model." Journal of Computational Finance, 11(3).

Lord, R., Koekkoek, R., Van Dijk, D. (2010). "A comparison of biased
simulation schemes for stochastic volatility models."
Quantitative Finance, 10(2), 177-194.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_lib.models.heston import HestonParams


@dataclass
class HestonSimulator:
    """
    Monte Carlo path simulator for the Heston stochastic volatility model.

    Simulates joint paths (S_t, v_t) under the risk-neutral measure Q.

    Parameters
    ----------
    params : HestonParams
        Heston model parameters (v0, kappa, v_bar, xi, rho).
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps. More steps = better accuracy for path-
        dependent options. 252 (daily) recommended for barriers and Asians.
    antithetic : bool
        Antithetic variates: for each path Z, also simulate -Z.
        Reduces variance by exploiting symmetry of the normal distribution.
        Note: for Heston with ρ ≠ 0, antithetic is applied to BOTH
        Z_v and Z_⊥ simultaneously to preserve the correlation structure.
    milstein : bool
        Use Milstein correction for variance process. Reduces strong
        discretisation error from O(√dt) to O(dt). Recommended for
        barrier options and short-dated Asians.
    truncation : str
        How to handle negative variance:
        'full'     — max(v, 0) at every step (default, most common)
        'reflect'  — |v| at every step
    seed : int, optional
        Random seed for reproducibility.

    Usage
    -----
    >>> params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
    >>> sim = HestonSimulator(params, n_paths=100_000, n_steps=252)
    >>> S_paths, v_paths = sim.simulate(S0=100, T=1.0, r=0.05, q=0.0)
    >>> S_paths.shape   # (100_000, 253)
    """

    params     : HestonParams
    n_paths    : int   = 100_000
    n_steps    : int   = 252
    antithetic : bool  = True
    milstein   : bool  = True
    truncation : str   = 'full'
    seed       : Optional[int] = None

    def __post_init__(self):
        if self.truncation not in ('full', 'reflect'):
            raise ValueError(f"truncation must be 'full' or 'reflect', got {self.truncation}")

    def simulate(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> tuple:
        """
        Simulate Heston paths under Q.

        Parameters
        ----------
        S0 : float   Initial spot price.
        T  : float   Time horizon in years.
        r  : float   Risk-free rate.
        q  : float   Continuous dividend yield.

        Returns
        -------
        S_paths : np.ndarray, shape (n_paths, n_steps + 1)
            Spot price paths. S_paths[:, 0] = S0.
        v_paths : np.ndarray, shape (n_paths, n_steps + 1)
            Variance paths. v_paths[:, 0] = v0.
        """
        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi    = self.params.xi
        rho   = self.params.rho
        v0    = self.params.v0

        dt        = T / self.n_steps
        sqrt_dt   = np.sqrt(dt)
        n_paths   = self.n_paths
        rho_perp  = np.sqrt(max(1 - rho**2, 0.0))  # √(1-ρ²)

        if self.seed is not None:
            np.random.seed(self.seed)

        # ------------------------------------------------------------------
        # Generate correlated random increments
        # Z_v   : drives the variance process
        # Z_perp: drives the orthogonal component of the spot process
        # Z_S   = ρ Z_v + √(1-ρ²) Z_perp  (Cholesky decomposition)
        # ------------------------------------------------------------------
        if self.antithetic:
            half      = n_paths // 2
            Zv_half   = np.random.standard_normal((half, self.n_steps))
            Zp_half   = np.random.standard_normal((half, self.n_steps))
            Z_v       = np.concatenate([Zv_half, -Zv_half], axis=0)
            Z_perp    = np.concatenate([Zp_half, -Zp_half], axis=0)
        else:
            Z_v    = np.random.standard_normal((n_paths, self.n_steps))
            Z_perp = np.random.standard_normal((n_paths, self.n_steps))

        # Correlated spot Brownian: Z_S = ρ Z_v + √(1-ρ²) Z_⊥
        Z_S = rho * Z_v + rho_perp * Z_perp

        # ------------------------------------------------------------------
        # Initialise path arrays
        # ------------------------------------------------------------------
        S_paths        = np.zeros((n_paths, self.n_steps + 1))
        v_paths        = np.zeros((n_paths, self.n_steps + 1))
        S_paths[:, 0]  = S0
        v_paths[:, 0]  = v0

        # ------------------------------------------------------------------
        # Time-stepping loop
        # ------------------------------------------------------------------
        for j in range(self.n_steps):
            v_j = v_paths[:, j]
            S_j = S_paths[:, j]

            # Variance used in diffusion: max(v, 0) for full truncation
            v_pos = np.maximum(v_j, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # ----------------------------------------------------------
            # Variance step: Euler-Maruyama (± Milstein correction)
            # dv = κ(v̄ - v) dt + ξ √v dW_v
            # ----------------------------------------------------------
            v_drift    = kappa * (v_bar - v_pos) * dt
            v_diffusion = xi * sqrt_v * sqrt_dt * Z_v[:, j]

            if self.milstein:
                # Milstein: + (1/4) ξ² (Z_v² - 1) dt
                v_milstein = 0.25 * xi**2 * (Z_v[:, j]**2 - 1) * dt
                v_new = v_j + v_drift + v_diffusion + v_milstein
            else:
                v_new = v_j + v_drift + v_diffusion

            # Apply truncation to keep variance non-negative
            if self.truncation == 'full':
                v_new = np.maximum(v_new, 0.0)
            else:  # reflect
                v_new = np.abs(v_new)

            v_paths[:, j + 1] = v_new

            # ----------------------------------------------------------
            # Spot step (log-Euler for stability)
            # d(log S) = (r - q - v/2) dt + √v dW_S
            # ----------------------------------------------------------
            log_drift    = (r - q - 0.5 * v_pos) * dt
            log_diffusion = sqrt_v * sqrt_dt * Z_S[:, j]

            S_paths[:, j + 1] = S_j * np.exp(log_drift + log_diffusion)

        return S_paths, v_paths

    def terminal_distribution(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> np.ndarray:
        """
        Return only terminal spot prices S_T (more memory efficient
        when path structure is not needed, e.g., European options).

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        S_paths, _ = self.simulate(S0, T, r, q)
        return S_paths[:, -1]

    def realised_variance(self, v_paths: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute realised variance for each path:
            RV = (1/T) ∫_0^T v_t dt ≈ (1/N) Σ v_{t_j}

        This is the payoff of a variance swap (times T).

        Parameters
        ----------
        v_paths : np.ndarray, shape (n_paths, n_steps+1)
        dt : float   Time step size.

        Returns
        -------
        np.ndarray, shape (n_paths,)   Realised variance per path.
        """
        # Trapezoidal integration of v_t along each path
        return np.trapezoid(v_paths, dx=dt, axis=1) / (v_paths.shape[1] * dt)

    def __repr__(self) -> str:
        return (
            f"HestonSimulator(v0={self.params.v0:.4f}, "
            f"κ={self.params.kappa:.2f}, v̄={self.params.v_bar:.4f}, "
            f"ξ={self.params.xi:.3f}, ρ={self.params.rho:.3f}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"milstein={self.milstein})"
        )
