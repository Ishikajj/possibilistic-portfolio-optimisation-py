from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# note that the runtime for this is Tn^3, due to the matrix inversion
# but, since n is constant and small, this is negligible.


def markowitz_step(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = 1.0,
) -> np.ndarray:
    """Compute unconstrained Markowitz weights at one time step."""
    inv_sigma = np.linalg.inv(sigma_t)
    w = (1.0 / theta) * inv_sigma @ mu_t
    return w


def compute_weights_array(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    burn_in: int,
    periods_until_investment: int,
    theta: float,
) -> np.ndarray:
    """Compute weights array across time."""
    T, n = mu_hat.shape
    weights = np.full((T, n), np.nan)

    # we compute weights for until T-1 only since the predictions we receive contain info upto time T.
    for t in range(burn_in + periods_until_investment, T - 1):
        if np.isnan(mu_hat[t]).any():
            continue
        weights[t] = markowitz_step(mu_hat[t], sigma_hat[t], theta)

    return weights


def markowitz_step_long_only(
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    theta: float = 1.0,
) -> np.ndarray:
    """Compute long-only Markowitz weights at one time step.

    Solves: max  mu^T w - (theta/2) w^T Sigma w
            s.t. sum(w) = 1,  w_i >= 0
    Returns NaN vector if optimisation fails.
    """
    n = len(mu_t)
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


def compute_weights_array_long_only(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    burn_in: int,
    periods_until_investment: int,
    theta: float,
) -> np.ndarray:
    """Compute long-only weights array across time, warm-starting from previous solution."""
    T, n = mu_hat.shape
    weights = np.full((T, n), np.nan)
    w_prev = np.ones(n) / n  # fallback initial guess

    for t in range(burn_in + periods_until_investment, T - 1):
        if np.isnan(mu_hat[t]).any():
            continue

        result = minimize(
            lambda w: float(-mu_hat[t] @ w + (theta / 2.0) * w @ sigma_hat[t] @ w),
            w_prev,
            jac=lambda w: -mu_hat[t] + theta * sigma_hat[t] @ w,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success:
            weights[t] = result.x
            w_prev = result.x  # warm-start next period
        else:
            weights[t] = np.full(n, np.nan)
            # keep w_prev unchanged so next period still has a sensible start

    return weights


def markowitz_long_only(
    mu_sigma_dict: dict[str, np.ndarray],
    returns_df: pd.DataFrame,
    burn_in: int,
    periods_until_investment: int,
    theta: float = 1.0,
) -> pd.DataFrame:
    """Full pipeline: extract → long-only optimise → return DataFrame."""
    mu_hat, sigma_hat = mu_sigma_dict["mu_hat"], mu_sigma_dict["sigma_hat"]

    weights_array = compute_weights_array_long_only(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )

    return pd.DataFrame(
        weights_array, index=returns_df.index, columns=returns_df.columns
    )


def markowitz_unconstrained(
    mu_sigma_dict: dict[str, np.ndarray],
    returns_df: pd.DataFrame,
    burn_in: int,
    periods_until_investment: int,
    theta: float = 1.0,
) -> pd.DataFrame:
    """Full pipeline: extract → compute → return DataFrame."""
    mu_hat, sigma_hat = mu_sigma_dict["mu_hat"], mu_sigma_dict["sigma_hat"]

    weights_array = compute_weights_array(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )

    return pd.DataFrame(
        weights_array, index=returns_df.index, columns=returns_df.columns
    )
