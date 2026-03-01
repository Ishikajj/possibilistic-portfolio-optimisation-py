# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.
import numpy as np
import pandas as pd
from typing import Optional

# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.

# basic assumption is that weights at index t are applied to returns at index t+1.


def portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    lag: int = 1,
    dropna: bool = True,
) -> pd.Series:
    """Compute the portfolio return series implied by weights and asset returns.

    Convention: weights at index t are applied to returns at index t+lag.

    Returns:
        pd.Series of portfolio returns inno dexed like `weights_df`.
    """
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(lag, int) and lag >= 0

    # Align to weights index/columns (enforces same asset set and order)
    returns_aligned = returns_df.loc[weights_df.index, weights_df.columns]

    # Shift weights so w_{t-1} multiplies r_t
    shifted_weights = weights_df.shift(lag)

    # Elementwise multiply and sum across assets.
    # Use skipna=False so any NaN propagates.
    port = (shifted_weights * returns_aligned).sum(axis=1, skipna=False)

    if dropna:
        port = port.dropna()

    return port


def calculate_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    scaling_factor: Optional[int] = None,
) -> float:
    """Compute daily Sharpe ratio from excess returns and weights."""
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)

    portfolio = portfolio_returns(returns_df, weights_df, lag=1, dropna=True)

    mean_ret: float = float(portfolio.mean())
    std_ret: float = float(portfolio.std(ddof=1))

    if std_ret == 0.0:
        return 0.0

    sharpe_daily: float = mean_ret / std_ret

    if not scaling_factor:
        return float(sharpe_daily)

    sharpe_annualised: float = sharpe_daily * np.sqrt(scaling_factor)

    return float(sharpe_annualised)


def rolling_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    window: int,
    scaling_factor: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute a rolling Sharpe ratio series from (excess) returns and weights.

    Convention: weights at index t are applied to returns at index t+1.

    """
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(window, int) and window > 0

    if min_periods is None:
        min_periods = window
    assert isinstance(min_periods, int) and 1 <= min_periods <= window

    portfolio = portfolio_returns(returns_df, weights_df, lag=1, dropna=False)

    # Rolling mean/std of portfolio returns
    rolling_mean = portfolio.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = portfolio.rolling(window=window, min_periods=min_periods).std(ddof=1)

    sharpe = rolling_mean / rolling_std

    # Avoid inf/-inf when std is zero
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    if scaling_factor:
        sharpe = sharpe * np.sqrt(scaling_factor)

    return sharpe


def certainty_equivalent(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    risk_free_df: pd.Series | pd.DataFrame,
    theta: float = 1.0,
    *,
    lag: int = 1,
    include_rf_in_mean: bool = True,
) -> float:
    """Certainty equivalent of the (next-period) portfolio return.

    Uses a mean-variance utility approximation:
        CE = E[R_p] + E [R_f]- (theta/2) Var(R_p)

    Returns:
        Scalar certainty equivalent.
    """
    assert isinstance(theta, (int, float)) and theta >= 0

    port = portfolio_returns(returns_df, weights_df, lag=lag, dropna=True)

    # Coerce risk-free to a Series
    if isinstance(risk_free_df, pd.DataFrame):
        assert risk_free_df.shape[1] == 1
        rf = risk_free_df.iloc[:, 0]
    else:
        rf = risk_free_df

    assert isinstance(rf, pd.Series)

    rf_realized = rf.loc[port.index]

    mean_term = float(port.mean())
    if include_rf_in_mean:
        mean_term += float(rf_realized.mean())

    var_term = float(port.var(ddof=1))

    ce = mean_term - (float(theta) / 2.0) * var_term
    return float(ce)
