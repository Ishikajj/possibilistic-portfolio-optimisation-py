# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.
import numpy as np
import pandas as pd
from typing import Optional

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
