# invests in each industry equally.
# assumes the burn in and the periods until investment to be the same as markowtiz
# takes the same t+1 assumption.


import numpy as np
import pandas as pd
from typing import Optional


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
