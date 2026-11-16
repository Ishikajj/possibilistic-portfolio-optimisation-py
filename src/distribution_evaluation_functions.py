"""Period-by-period predictive distribution scoring under a multivariate Gaussian.

Inputs: returns_df (T, n) and a forecast dict (predictive.py: "mu_hat" (T,n),
"sigma_hat" (T,n,n), "index" checked against returns_df.index), same asset order.
Moments at t score the realised returns at t+1.

mahalanobis_distance_series        — sqrt((x-mu)' Sigma^{-1} (x-mu)) per period
                                     (squared=True drops the sqrt), lower is better.
multivariate_log_likelihood_series — Gaussian log-likelihood per period, higher is better.
Averages over the valid periods are taken in evaluation_pipeline.py.

Scoring is Gaussian with the forecast's mean and covariance for every strategy.
The Bayesian / possibilistic predictives are really Student-t mixtures (heavier
tails), but the benchmarks have no t predictive, so one Gaussian rule keeps the
scores comparable across strategies.
"""

import warnings

import numpy as np
import pandas as pd

from predictive import check_aligned


def _gaussian_terms(
    returns_df: pd.DataFrame,
    mu_sigma_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-period quadratic form (x-mu)' Sigma^{-1} (x-mu) and log det(Sigma).

    Row t scores the returns at t with the moments from t-1. Rows are NaN
    where the inputs contain NaN (expected before investment, not reported)
    or Sigma can't be used: singular / not positive definite, no
    regularisation (one RuntimeWarning with the count).
    """
    check_aligned(mu_sigma_dict, returns_df.index)
    mu_hat = mu_sigma_dict["mu_hat"]
    sigma_hat = mu_sigma_dict["sigma_hat"]
    T, n = returns_df.shape
    if mu_hat.shape != (T, n) or sigma_hat.shape != (T, n, n):
        raise ValueError(
            f"mu_hat / sigma_hat must have shapes {(T, n)} / {(T, n, n)} to "
            f"match returns_df, got {mu_hat.shape} / {sigma_hat.shape}"
        )

    x = returns_df.to_numpy(dtype=float)[1:]  # returns at t ...
    mu, sigma = mu_hat[:-1], sigma_hat[:-1]  # ... scored with moments at t-1
    quad = np.full(T, np.nan)
    logdet = np.full(T, np.nan)

    # ok[t] is True when the day has no NaN anywhere: not in the returns,
    # the mean forecast or the covariance forecast (axis=(1, 2) = whole matrix)
    ok = ~(
        np.isnan(x).any(axis=1)
        | np.isnan(mu).any(axis=1)
        | np.isnan(sigma).any(axis=(1, 2))
    )
    sigma_ok = 0.5 * (sigma[ok] + sigma[ok].mT)  # .mT: transpose each matrix
    # Two mask levels: ok has one entry per day; sign, ld and good only cover
    # the ok days (they come from sigma_ok). rows maps the good ones back to
    # output rows: positions of ok days -> keep the good ones -> +1 because
    # x, mu, sigma start one day late (x = returns[1:]).
    sign, ld = np.linalg.slogdet(sigma_ok)
    good = sign > 0  # singular or det < 0 -> row invalid
    rows = np.flatnonzero(ok)[good] + 1
    diff = (x[ok] - mu[ok])[good]
    # one solve per row: a single batched solve differs in the last bit
    quad[rows] = [
        d @ np.linalg.solve(s, d) for s, d in zip(sigma_ok[good], diff)
    ]
    logdet[rows] = ld[good]

    n_bad = int((~good).sum())
    if n_bad:
        warnings.warn(
            f"gaussian scoring: {n_bad} of {int(ok.sum())} scorable rows "
            f"left NaN (Sigma singular or det < 0: {n_bad})",
            RuntimeWarning,
            stacklevel=3,  # point at the caller of the public function
        )

    return quad, logdet


def mahalanobis_distance_series(
    returns_df: pd.DataFrame,
    mu_sigma_dict: dict[str, np.ndarray],
    *,
    squared: bool = False,
) -> pd.Series:
    """Period-by-period Mahalanobis distance of the realised returns.

    Moments at t score the returns at t+1. ``squared=True`` returns
    (x-mu)' Sigma^{-1} (x-mu) without the square root.
    """
    quad, _ = _gaussian_terms(returns_df, mu_sigma_dict)
    out = quad if squared else np.sqrt(np.maximum(quad, 0.0))
    return pd.Series(out, index=returns_df.index, name="mahalanobis_distance")


def multivariate_log_likelihood_series(
    returns_df: pd.DataFrame,
    mu_sigma_dict: dict[str, np.ndarray],
) -> pd.Series:
    """Period-by-period multivariate Gaussian log-likelihood.

    Moments at t score the returns at t+1.
    """
    quad, logdet = _gaussian_terms(returns_df, mu_sigma_dict)
    n_assets = returns_df.shape[1]
    out = -0.5 * (n_assets * np.log(2.0 * np.pi) + logdet + quad)
    return pd.Series(out, index=returns_df.index, name="log_likelihood")
