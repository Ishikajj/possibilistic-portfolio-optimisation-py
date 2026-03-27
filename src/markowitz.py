from __future__ import annotations

import numpy as np
import pandas as pd

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
