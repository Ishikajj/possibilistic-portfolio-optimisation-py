import numpy as np
import pandas as pd
from markowitz import markowitz_unconstrained
from typing import Optional


def historical_expectations_weights(
    returns_df: pd.DataFrame,
    burn_in: int,
    periods_until_investment: int,
) -> pd.DataFrame:
    """Compute historical means and covariances across time."""
    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(burn_in + periods_until_investment, T):
        ret_window = returns_df.iloc[t - burn_in - periods_until_investment : t]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values  # auto set c = w-1

    mu_sigma_dict = {"mu_hat": mu_hat, "sigma_hat": sigma_hat}
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, burn_in, periods_until_investment
    )

    return investment_weights


def rolling_window_weights(
    returns_df: pd.DataFrame, window: int, burn: int, periods_until_investment: int
) -> pd.DataFrame:

    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    start = max(window, burn + periods_until_investment)

    for t in range(start, T):
        ret_window = returns_df.iloc[t - window : t]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values

    mu_sigma_dict = {"mu_hat": mu_hat, "sigma_hat": sigma_hat}
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, burn, periods_until_investment
    )

    return investment_weights


def equal_weight_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
) -> pd.DataFrame:
    """Create 1/N weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)

    start = burn_in + periods_until_investment
    end = T - 1  # last row excluded

    if start < end:
        weights[start:end] = 1.0 / n

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )


def market_weight_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    market_col: str = "Mkt-RF",
) -> pd.DataFrame:
    """Create market weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)
    if market_col not in returns_df.columns:
        raise KeyError(f"Market column '{market_col}' not found in returns DataFrame.")

    start = burn_in + periods_until_investment
    end = T - 1  # last row excluded
    market_idx = returns_df.columns.get_loc(market_col)

    if start < end:
        weights[start:end, :] = 0.0
        weights[start:end, market_idx] = 1.0

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )


def minimum_variance_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    window: int = 250,
) -> pd.DataFrame:
    """Create minimum variance weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)

    start = max(window, burn_in + periods_until_investment)
    end = T - 1  # last row excluded

    for t in range(start, end):
        ret_window = returns_df.iloc[t - window : t]
        sigma_hat_t = ret_window.cov().values
        inv_sigma = np.linalg.inv(sigma_hat_t)
        w = inv_sigma @ np.ones(n)
        w /= w.sum()  # normalize to sum to 1
        weights[t] = w

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )
