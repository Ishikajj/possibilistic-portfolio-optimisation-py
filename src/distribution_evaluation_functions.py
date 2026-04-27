"""Period-by-period predictive distribution scoring under a multivariate Gaussian.

Inputs: returns_df (T, n) and mu_sigma_dict {"mu_hat": (T,n), "sigma_hat": (T,n,n)}.
Moments at t score realised returns at t+1 (lag=1).

mahalanobis_distance_series  — (x-mu)' Sigma^{-1} (x-mu) per period, lower is better.
multivariate_log_likelihood_series — Gaussian log-likelihood per period, higher is better.
average_log_likelihood        — scalar mean of the above, used in evaluation_pipeline.py.
"""

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, slogdet
from typing import Dict


def _prepare_scoring_inputs(
    returns_df: pd.DataFrame,
    mu_sigma_dict: Dict[str, np.ndarray],
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Prepare realized returns and lagged predicted moments for scoring.

    Expects:
    - returns_df: shape (T, n)
    - mu_sigma_dict["mu_hat"]: shape (T, n)
    - mu_sigma_dict["sigma_hat"]: shape (T, n, n)

    Assumes the predicted moments are already indexed in the same time order and
    asset order as `returns_df`.
    """
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(lag, int) and lag >= 0

    mu_hat = mu_sigma_dict["mu_hat"]
    sigma_hat = mu_sigma_dict["sigma_hat"]

    assert isinstance(mu_hat, np.ndarray) and mu_hat.ndim == 2
    assert isinstance(sigma_hat, np.ndarray) and sigma_hat.ndim == 3
    assert mu_hat.shape[0] == len(returns_df.index)
    assert mu_hat.shape[1] == len(returns_df.columns)
    assert sigma_hat.shape[0] == len(returns_df.index)
    assert sigma_hat.shape[1] == sigma_hat.shape[2] == len(returns_df.columns)

    returns_aligned = returns_df.to_numpy(dtype=float, copy=True)
    mu_scoring = np.full_like(mu_hat, np.nan, dtype=float)
    sigma_scoring = np.full_like(sigma_hat, np.nan, dtype=float)
    if lag == 0:
        mu_scoring[:] = mu_hat
        sigma_scoring[:] = sigma_hat
    else:
        mu_scoring[lag:] = mu_hat[:-lag]
        sigma_scoring[lag:] = sigma_hat[:-lag]

    return returns_aligned, mu_scoring, sigma_scoring, returns_df.index


def _gaussian_scoring_terms(
    x_t: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    regularization: float,
) -> tuple[float, float, int] | None:
    """Return quadratic form, log-determinant, and dimension for Gaussian scoring.

    Returns None if inputs contain NaNs or the covariance matrix is numerically
    invalid.
    """
    if np.isnan(x_t).any() or np.isnan(mu_t).any() or np.isnan(Sigma_t).any():
        return None

    n_assets = Sigma_t.shape[0]
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
    if regularization > 0.0:
        Sigma_t += regularization * np.eye(n_assets, dtype=float)

    diff = x_t - mu_t
    try:
        sign, logdet = slogdet(Sigma_t)
        if sign <= 0:
            return None
        quad = float(diff.T @ np.linalg.solve(Sigma_t, diff))
    except LinAlgError:
        return None

    return quad, float(logdet), n_assets


def mahalanobis_distance_series(
    returns_df: pd.DataFrame,
    mu_sigma_dict: Dict[str, np.ndarray],
    *,
    lag: int = 1,
    squared: bool = False,
    regularization: float = 1e-10,
) -> pd.Series:
    """Compute period-by-period Mahalanobis distance of realized returns.

    Convention: moments indexed at t are used to score realized returns at t+lag.
    """
    assert isinstance(regularization, (int, float)) and regularization >= 0

    returns_aligned, mu_scoring, sigma_scoring, scoring_index = (
        _prepare_scoring_inputs(returns_df, mu_sigma_dict, lag)
    )

    out = np.full(len(scoring_index), np.nan, dtype=float)

    for t in range(len(scoring_index)):
        x_t = returns_aligned[t]
        mu_t = mu_scoring[t]
        Sigma_t = np.array(sigma_scoring[t], dtype=float, copy=True)

        terms = _gaussian_scoring_terms(x_t, mu_t, Sigma_t, regularization)
        if terms is None:
            continue

        md2, _, _ = terms
        out[t] = md2 if squared else float(np.sqrt(max(md2, 0.0)))

    return pd.Series(out, index=scoring_index, name="mahalanobis_distance")


def multivariate_log_likelihood_series(
    returns_df: pd.DataFrame,
    mu_sigma_dict: Dict[str, np.ndarray],
    *,
    lag: int = 1,
    regularization: float = 1e-10,
) -> pd.Series:
    """Compute period-by-period multivariate Gaussian log likelihood.

    Convention: moments indexed at t are used to score realized returns at t+lag.
    """
    assert isinstance(regularization, (int, float)) and regularization >= 0

    returns_aligned, mu_scoring, sigma_scoring, scoring_index = (
        _prepare_scoring_inputs(returns_df, mu_sigma_dict, lag)
    )

    out = np.full(len(scoring_index), np.nan, dtype=float)

    for t in range(len(scoring_index)):
        x_t = returns_aligned[t]
        mu_t = mu_scoring[t]
        Sigma_t = np.array(sigma_scoring[t], dtype=float, copy=True)

        terms = _gaussian_scoring_terms(x_t, mu_t, Sigma_t, regularization)
        if terms is None:
            continue

        quad, logdet, n_assets = terms
        out[t] = -0.5 * (n_assets * np.log(2.0 * np.pi) + logdet + quad)

    return pd.Series(out, index=scoring_index, name="log_likelihood")


def average_log_likelihood(
    returns_df: pd.DataFrame,
    mu_sigma_dict: Dict[str, np.ndarray],
    *,
    lag: int = 1,
    regularization: float = 1e-10,
) -> float:
    """Average multivariate Gaussian log likelihood across valid periods."""
    ll = multivariate_log_likelihood_series(
        returns_df,
        mu_sigma_dict,
        lag=lag,
        regularization=regularization,
    )
    ll = ll.dropna()
    if ll.empty:
        return float("nan")
    return float(ll.mean())
