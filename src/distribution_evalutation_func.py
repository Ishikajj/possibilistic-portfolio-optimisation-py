import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, slogdet
from typing import Optional


def _align_predicted_moments(
    returns_df: pd.DataFrame,
    mu_hat_df: pd.DataFrame,
    sigma_hat_arr: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Align realized returns with predicted means/covariances.

    Expects:
    - returns_df: shape (T, n)
    - mu_hat_df: shape (T, n), same asset columns as returns_df
    - sigma_hat_arr: shape (T, n, n)
    """
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(mu_hat_df, pd.DataFrame)
    assert isinstance(sigma_hat_arr, np.ndarray) and sigma_hat_arr.ndim == 3

    common_index = returns_df.index.intersection(mu_hat_df.index)
    returns_aligned = returns_df.loc[common_index].copy()
    mu_aligned = mu_hat_df.loc[common_index, returns_aligned.columns].copy()

    assert sigma_hat_arr.shape[0] == len(mu_hat_df.index)
    assert sigma_hat_arr.shape[1] == sigma_hat_arr.shape[2] == len(mu_hat_df.columns)

    sigma_df_indexer = mu_hat_df.index.get_indexer(common_index)
    sigma_col_indexer = np.array(
        [mu_hat_df.columns.get_loc(col) for col in returns_aligned.columns], dtype=int
    )
    sigma_aligned = sigma_hat_arr[sigma_df_indexer][:, sigma_col_indexer][
        :, :, sigma_col_indexer
    ]

    return returns_aligned, mu_aligned, sigma_aligned


def mahalanobis_distance_series(
    returns_df: pd.DataFrame,
    mu_hat_df: pd.DataFrame,
    sigma_hat_arr: np.ndarray,
    *,
    lag: int = 1,
    squared: bool = False,
    regularization: float = 1e-10,
) -> pd.Series:
    """Compute period-by-period Mahalanobis distance of realized returns.

    Convention: moments indexed at t are used to score realized returns at t+lag.
    """
    assert isinstance(lag, int) and lag >= 0
    assert isinstance(regularization, (int, float)) and regularization >= 0

    returns_aligned, mu_aligned, sigma_aligned = _align_predicted_moments(
        returns_df, mu_hat_df, sigma_hat_arr
    )

    mu_scoring = mu_aligned.shift(lag)
    sigma_scoring = np.full_like(sigma_aligned, np.nan, dtype=float)
    if lag == 0:
        sigma_scoring[:] = sigma_aligned
    else:
        sigma_scoring[lag:] = sigma_aligned[:-lag]

    out = np.full(len(returns_aligned.index), np.nan, dtype=float)

    for t in range(len(returns_aligned.index)):
        x_t = returns_aligned.iloc[t].to_numpy(dtype=float)
        mu_t = mu_scoring.iloc[t].to_numpy(dtype=float)
        Sigma_t = np.array(sigma_scoring[t], dtype=float, copy=True)

        if np.isnan(x_t).any() or np.isnan(mu_t).any() or np.isnan(Sigma_t).any():
            continue

        n_assets = Sigma_t.shape[0]
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
        if regularization > 0.0:
            Sigma_t += regularization * np.eye(n_assets, dtype=float)

        diff = x_t - mu_t
        try:
            md2 = float(diff.T @ np.linalg.solve(Sigma_t, diff))
        except LinAlgError:
            continue

        out[t] = md2 if squared else float(np.sqrt(max(md2, 0.0)))

    return pd.Series(out, index=returns_aligned.index, name="mahalanobis_distance")


def multivariate_log_likelihood_series(
    returns_df: pd.DataFrame,
    mu_hat_df: pd.DataFrame,
    sigma_hat_arr: np.ndarray,
    *,
    lag: int = 1,
    regularization: float = 1e-10,
) -> pd.Series:
    """Compute period-by-period multivariate Gaussian log likelihood.

    Convention: moments indexed at t are used to score realized returns at t+lag.
    """
    assert isinstance(lag, int) and lag >= 0
    assert isinstance(regularization, (int, float)) and regularization >= 0

    returns_aligned, mu_aligned, sigma_aligned = _align_predicted_moments(
        returns_df, mu_hat_df, sigma_hat_arr
    )

    mu_scoring = mu_aligned.shift(lag)
    sigma_scoring = np.full_like(sigma_aligned, np.nan, dtype=float)
    if lag == 0:
        sigma_scoring[:] = sigma_aligned
    else:
        sigma_scoring[lag:] = sigma_aligned[:-lag]

    out = np.full(len(returns_aligned.index), np.nan, dtype=float)

    for t in range(len(returns_aligned.index)):
        x_t = returns_aligned.iloc[t].to_numpy(dtype=float)
        mu_t = mu_scoring.iloc[t].to_numpy(dtype=float)
        Sigma_t = np.array(sigma_scoring[t], dtype=float, copy=True)

        if np.isnan(x_t).any() or np.isnan(mu_t).any() or np.isnan(Sigma_t).any():
            continue

        n_assets = Sigma_t.shape[0]
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
        if regularization > 0.0:
            Sigma_t += regularization * np.eye(n_assets, dtype=float)

        diff = x_t - mu_t
        try:
            sign, logdet = slogdet(Sigma_t)
            if sign <= 0:
                continue
            quad = float(diff.T @ np.linalg.solve(Sigma_t, diff))
        except LinAlgError:
            continue

        out[t] = -0.5 * (n_assets * np.log(2.0 * np.pi) + logdet + quad)

    return pd.Series(out, index=returns_aligned.index, name="log_likelihood")


def average_log_likelihood(
    returns_df: pd.DataFrame,
    mu_hat_df: pd.DataFrame,
    sigma_hat_arr: np.ndarray,
    *,
    lag: int = 1,
    regularization: float = 1e-10,
) -> float:
    """Average multivariate Gaussian log likelihood across valid periods."""
    ll = multivariate_log_likelihood_series(
        returns_df,
        mu_hat_df,
        sigma_hat_arr,
        lag=lag,
        regularization=regularization,
    )
    ll = ll.dropna()
    if ll.empty:
        return float("nan")
    return float(ll.mean())
