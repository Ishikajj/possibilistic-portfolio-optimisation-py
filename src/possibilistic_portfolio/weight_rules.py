"""Weight rules: everything that turns data or a forecast into portfolio weights.

Forecasts (predictive moments) come from the forecasters
(alternative_investment_rules, bayesian_averaging, possibilistic_bayesian);
this module only makes weights. The pipeline applies the Markowitz rules to
every forecast and runs the weight-only rules on the returns.

Markowitz, from a forecast (mu_hat, sigma_hat):
  markowitz_unconstrained   — closed form (1/theta) Sigma^{-1} mu, short selling
                              allowed; weights don't sum to 1 (the rest is in
                              the risk-free asset)
  markowitz_long_only       — SLSQP: max mu'w - (theta/2) w' Sigma w,
                              sum(w) = 1, w >= 0
  markowitz_step, markowitz_step_long_only — one time step
  compute_weights_array     — the loop over time shared by both

Weight-only rules, from the returns (no forecast):
  equal_weights             — 1/N
  market_weights            — everything in the market column (Mkt-RF)
  minimum_variance_weights  — Sigma^{-1} 1 / (1' Sigma^{-1} 1), rolling
                              sample covariance

Every rule gives weights indexed like the returns: NaN before `start` and on
the last row. Weights at row t are decided with data up to t and held over
t+1 (the shift happens in portfolio_returns; timing convention: predictive.py).
The Markowitz entry points take the forecast dict (predictive.make_predictive;
"index" is checked against returns_df.index), returns_df (index and columns
only), start and theta (risk aversion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from possibilistic_portfolio.config import ROLLING_WINDOW_LONG, THETA
from possibilistic_portfolio.predictive import check_aligned, warn_nan_rows


def solve_or_pinv(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return x solving A x = b; fall back to pinv(A) b if A is singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _weights_frame(
    weights: np.ndarray, returns_df: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        weights, index=returns_df.index, columns=returns_df.columns
    )


# ── Markowitz (from a forecast) ──────────────────────────────────────────────


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

    return _weights_frame(weights_array, returns_df)


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

    return _weights_frame(weights_array, returns_df)


# ── Weight-only rules (from the returns) ─────────────────────────────────────


def equal_weights(returns_df: pd.DataFrame, *, start: int) -> pd.DataFrame:
    """1/N weights from `start`, last row excluded."""
    T, n = returns_df.shape
    weights = np.full((T, n), np.nan)
    end = T - 1  # last row excluded
    if start < end:
        weights[start:end] = 1.0 / n
    return _weights_frame(weights, returns_df)


def market_weights(
    returns_df: pd.DataFrame,
    *,
    start: int,
    market_col: str = "Mkt-RF",
) -> pd.DataFrame:
    """Everything in the market column from `start`, last row excluded."""
    T, n = returns_df.shape
    if market_col not in returns_df.columns:
        raise KeyError(
            f"Market column '{market_col}' not found in returns DataFrame."
        )
    weights = np.full((T, n), np.nan)
    end = T - 1  # last row excluded
    if start < end:
        weights[start:end, :] = 0.0
        weights[start:end, returns_df.columns.get_loc(market_col)] = 1.0
    return _weights_frame(weights, returns_df)


def minimum_variance_weights(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> pd.DataFrame:
    """Minimum variance weights Sigma^{-1} 1 / (1' Sigma^{-1} 1) from the
    rolling sample covariance (ddof=1) of the `window` returns up to and
    including t, from `start` (or the first full window, if later). A
    singular Sigma falls back to the pseudo-inverse (solve_or_pinv)."""
    T, n = returns_df.shape
    weights = np.full((T, n), np.nan)
    first, end = max(window - 1, start), T - 1  # last row excluded
    ones = np.ones(n)
    skipped = {"1' Sigma^{-1} 1 zero or not finite": 0}

    for t in range(first, end):
        ret_window = returns_df.iloc[t - window + 1 : t + 1]
        w = solve_or_pinv(ret_window.cov(ddof=1).values, ones)
        total = w.sum()
        if total == 0.0 or not np.isfinite(total):
            skipped["1' Sigma^{-1} 1 zero or not finite"] += 1
            continue
        weights[t] = w / total  # normalise to sum to 1

    warn_nan_rows("minimum_variance", skipped, max(end - first, 0))
    return _weights_frame(weights, returns_df)
