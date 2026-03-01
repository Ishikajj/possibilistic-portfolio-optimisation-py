import numpy as np
import pandas as pd
from markowitz import markowitz_unconstrained


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
        sigma_hat[t] = ret_window.cov().values

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
