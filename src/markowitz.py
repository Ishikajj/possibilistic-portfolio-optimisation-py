import numpy as np
import pandas as pd
from typing import Dict

# note that the runtime for this is Tn^3, due to the matrix inversion
# but, since n is constant and small, this is negligible.
Array3D = (
    np.ndarray
)  # shape (T, n, n) #this is the shape of the covariance array across time
Array2D = (
    np.ndarray
)  # shape (T, n) #this is the shape of the mean returns array across time


def extract_mu_sigma(
    mu_sigma_dict: Dict[str, Array2D | Array3D],
) -> tuple[Array2D, Array3D]:
    """Extract arrays from dictionary."""
    mu_hat = mu_sigma_dict["mu_hat"]
    sigma_hat = mu_sigma_dict["sigma_hat"]

    assert isinstance(mu_hat, np.ndarray)
    assert isinstance(sigma_hat, np.ndarray)

    return mu_hat, sigma_hat


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
    mu_hat: Array2D,
    sigma_hat: Array3D,
    burn_in: int,
    periods_until_investment: int,
    theta: float,
) -> Array2D:
    """Compute weights array across time."""
    T, n = mu_hat.shape
    weights = np.full((T, n), np.nan)

    # we compute weights for until T-1 only since the predictions we receive are for t-1.
    for t in range(burn_in + periods_until_investment, T - 1):
        if np.isnan(mu_hat[t]).any():
            continue
        weights[t] = markowitz_step(mu_hat[t], sigma_hat[t], theta)

    return weights


def weights_to_dataframe(
    weights: Array2D,
    returns_index: pd.Index,
    asset_columns: pd.Index,
) -> pd.DataFrame:
    """Convert weight array to DataFrame."""
    return pd.DataFrame(weights, index=returns_index, columns=asset_columns)


def markowitz_unconstrained(
    mu_sigma_dict: Dict[str, Array2D | Array3D],
    returns_df: pd.DataFrame,
    burn_in: int,
    periods_until_investment: int,
    theta: float = 1.0,
) -> pd.DataFrame:
    """Full pipeline: extract → compute → return DataFrame."""
    mu_hat, sigma_hat = extract_mu_sigma(mu_sigma_dict)

    weights_array = compute_weights_array(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )

    return weights_to_dataframe(
        weights_array,
        returns_index=returns_df.index,
        asset_columns=returns_df.columns,
    )
