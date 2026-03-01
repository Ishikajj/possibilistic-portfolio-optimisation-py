# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.
import numpy as np
import pandas as pd
from typing import Optional


def certainty_equivalents(*):
    pass

    
# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.

# basic assumption is that weights at index t are applied to returns at index t+1.


def calculate_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    scaling_factor: Optional[int] = None,
) -> float:
    """Compute daily Sharpe ratio from excess returns and weights."""
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)

    # Align and ensure same columns/order
    returns_aligned = returns_df.loc[weights_df.index, weights_df.columns]

    # Shift returns backward so weights_t multiplies returns_{t+1}
    shifted_returns = returns_aligned.shift(-1)

    # Elementwise multiply and sum across assets
    portfolio_returns = (weights_df * shifted_returns).sum(axis=1)

    # Drop last NaN (from shift) and any burn-in NaNs
    portfolio_returns = portfolio_returns.dropna()

    mean_ret: float = float(portfolio_returns.mean())
    std_ret: float = float(portfolio_returns.std(ddof=1))

    if std_ret == 0.0:
        return 0.0

    sharpe_daily: float = mean_ret / std_ret

    if not scaling_factor:
        return float(sharpe_daily)

    sharpe_annualised: float = sharpe_daily * np.sqrt(scaling_factor)

    return float(sharpe_annualised)


def certainty_equivalents():
    pass


def rolling_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    window: int,
    scaling_factor: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute a rolling Sharpe ratio series from (excess) returns and weights.

    Convention: weights at index t are applied to returns at index t+1.

    Args: Rest of them are same as above.
        window: Rolling window length (in rows / periods) used to estimate mean and std.
        min_periods: Minimum periods required to compute rolling stats. Defaults to `window`.

    Returns:
        pd.Series of rolling Sharpe ratios indexed by time (same index as `weights_df`).
    """
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(window, int) and window > 0

    if min_periods is None:
        min_periods = window
    assert isinstance(min_periods, int) and 1 <= min_periods <= window

    # Align and ensure same columns/order
    returns_aligned = returns_df.loc[weights_df.index, weights_df.columns]

    # Shift returns backward so weights_t multiplies returns_{t+1}
    shifted_returns = returns_aligned.shift(-1)

    # Portfolio return at time t is realised over (t -> t+1]
    portfolio_returns = (weights_df * shifted_returns).sum(axis=1)

    # Rolling mean/std of portfolio returns
    rolling_mean = portfolio_returns.rolling(
        window=window, min_periods=min_periods
    ).mean()
    rolling_std = portfolio_returns.rolling(window=window, min_periods=min_periods).std(
        ddof=1
    )

    sharpe = rolling_mean / rolling_std

    # Avoid inf/-inf when std is zero
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    if scaling_factor:
        sharpe = sharpe * np.sqrt(scaling_factor)

    return sharpe
