"""Markowitz portfolio weight computation from predictive moments.

Two variants:

Unconstrained — closed-form solution, allows short selling:
  markowitz_step            — single time step, returns weight vector
  compute_weights_array     — full time series, returns (T, n) array
                              (shared by both variants: long_only=True/False)
  markowitz_unconstrained   — entry point: mu_sigma_dict + returns_df → weights DataFrame

Long-only — SLSQP constrained optimisation, no short selling:
  markowitz_step_long_only  — single time step, optional warm start w0
  markowitz_long_only       — entry point: mu_sigma_dict + returns_df → weights DataFrame

Both entry points take:
  mu_sigma_dict : forecast dict (predictive.py): "mu_hat" (T, n), "sigma_hat" (T, n, n),
                  "index" (checked against returns_df.index if present)
  returns_df    : pd.DataFrame (T, n) — used only for index/columns alignment
  start         : first row with weights (the pipeline computes burn_in + periods_until_investment once)
  theta         : risk-aversion scalar (default 1.0)

Unconstrained weights don't sum to 1 by design: the remainder (1 - sum(w))
sits in the risk-free asset. Long-only weights sum to 1 (fully invested).

Weights at row t come from the moments at row t and are held over t+1 (the
shift happens in portfolio_returns; timing convention: predictive.py).
Last row is always NaN — predictions at T-1 have no subsequent return to invest in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from possibilistic_portfolio.config import THETA
from possibilistic_portfolio.predictive import check_aligned, warn_nan_rows


def markowitz_step(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = THETA,
) -> np.ndarray:
    """Unconstrained Markowitz weights at one time step: Sigma^{-1} mu / theta."""
    return np.linalg.solve(sigma_t, mu_t) / theta


def markowitz_step_long_only(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = THETA,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    """Compute long-only Markowitz weights at one time step.

    Solves: max  mu^T w - (theta/2) w^T Sigma w
            s.t. sum(w) = 1,  w_i >= 0
    w0 is the optimiser's starting point (default 1/N).
    Returns NaN vector if optimisation fails.
    """
    n = len(mu_t)
    if w0 is None:
        w0 = np.ones(n) / n

    def neg_utility(w: np.ndarray) -> float:
        return float(-mu_t @ w + (theta / 2.0) * w @ sigma_t @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu_t + theta * sigma_t @ w

    result = minimize(
        neg_utility,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    return result.x if result.success else np.full(n, np.nan)


def compute_weights_array(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    start: int,
    theta: float,
    long_only: bool = False,
) -> np.ndarray:
    """Compute weights array across time.

    Rows from start to T-2 are filled; the last
    row has no following return to invest in. Rows with NaN mu are skipped.
    Long-only warm-starts each step from the previous successful solution.

    Rows that can't be computed (singular Sigma, long-only
    optimiser failure) are left NaN, with one warning giving the counts.
    """
    if theta <= 0:
        raise ValueError(f"theta (risk aversion) must be > 0, got {theta}")

    T, n = mu_hat.shape
    weights = np.full((T, n), np.nan)
    w_prev = np.ones(n) / n  # long-only starting point
    skipped = {"singular Sigma": 0, "optimiser failed": 0}

    for t in range(start, T - 1):
        if np.isnan(mu_hat[t]).any():
            continue
        if long_only:
            w = markowitz_step_long_only(
                mu_hat[t], sigma_hat[t], theta, w0=w_prev
            )
            if np.isnan(w).any():
                skipped["optimiser failed"] += 1
                continue  # keep w_prev so the next start is still sensible
            weights[t] = w_prev = w
        else:
            try:
                weights[t] = markowitz_step(mu_hat[t], sigma_hat[t], theta)
            except np.linalg.LinAlgError:
                skipped["singular Sigma"] += 1

    warn_nan_rows(
        "markowitz", skipped, max(T - 1 - start, 0), "investable rows"
    )
    return weights


def markowitz_long_only(
    mu_sigma_dict: dict[str, np.ndarray],
    returns_df: pd.DataFrame,
    start: int,
    theta: float = THETA,
) -> pd.DataFrame:
    """Full pipeline: extract → long-only optimise → return DataFrame."""
    check_aligned(mu_sigma_dict, returns_df.index)
    mu_hat, sigma_hat = mu_sigma_dict["mu_hat"], mu_sigma_dict["sigma_hat"]

    weights_array = compute_weights_array(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        start=start,
        theta=theta,
        long_only=True,
    )

    return pd.DataFrame(
        weights_array, index=returns_df.index, columns=returns_df.columns
    )


def markowitz_unconstrained(
    mu_sigma_dict: dict[str, np.ndarray],
    returns_df: pd.DataFrame,
    start: int,
    theta: float = THETA,
) -> pd.DataFrame:
    """Full pipeline: extract → compute → return DataFrame."""
    check_aligned(mu_sigma_dict, returns_df.index)
    mu_hat, sigma_hat = mu_sigma_dict["mu_hat"], mu_sigma_dict["sigma_hat"]

    weights_array = compute_weights_array(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        start=start,
        theta=theta,
    )

    return pd.DataFrame(
        weights_array, index=returns_df.index, columns=returns_df.columns
    )
