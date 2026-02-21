"""We use the same data sources as the original robust bayesian portfolio choices paper.

This used 24 datasets FROM CRPS and Kenneth French

find other data sets in ntulibrary. Databases, W, and Wharton Research Data Services (WRDS).
"""

# TODO: incorporate this with other data sets
# add functionality to cycle dynamically through specific column names and do so and so for each file in ../datasets
# each file generates a  new dataset
# would likely require dynamic column determination from downstream applications as well.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Path to datasets directory (relative to this file)
DATA_PATH = (
    Path(__file__).resolve().parent / "../../datasets/12_Industry_Portfolios_Daily.csv"
)


def load_industry_portfolios(
    as_decimal: bool = True, data_path=DATA_PATH
) -> pd.DataFrame:

    # Find the header row that contains the column names (e.g., NoDur, Durbl, ...)
    header_row: Optional[int] = None
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "NoDur" in line and "Durbl" in line and "Manuf" in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError(
            "Could not locate the industry table header row (expected a line containing 'NoDur', 'Durbl', 'Manuf')."
        )

    df = pd.read_csv(
        data_path,
        skiprows=header_row,
        header=0,
        index_col=0,
        na_values=[-99.99, -999, "-99.99", "-999"],
    )

    # Parse and clean the date index
    idx = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df.loc[~idx.isna()].copy()
    df.index = idx[~idx.isna()]
    df = df.sort_index()

    # Convert to numeric (defensive) and drop empty columns
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if as_decimal:
        df = df / 100.0

    return df


def time_index_to_integer(
    df: pd.DataFrame,
    old_index_col: str = "time",
    int_index_name: str = "t",
) -> pd.DataFrame:
    """Convert a time-indexed DataFrame to integer indexing."""
    out = df.copy()
    out[old_index_col] = out.index
    out.index = pd.RangeIndex(start=0, stop=len(out), step=1, name=int_index_name)
    return out


def slice_timeframe(df, start_date="1963-01-01", end_date=None):
    """
    Slice a DataFrame by date index.
    start_date and end_date can be None or strings / datetime-like.
    """
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df.loc[df.index >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df.loc[df.index <= end_date]
    df = df.sort_index()
    df = time_index_to_integer(df)
    return df


if __name__ == "__main__":
    df = load_industry_portfolios()
    df = slice_timeframe(df)
    df = time_index_to_integer(df)
    print(df.head())
    print(df.columns)
    print(df.dtypes)


# output received its a dataframe with yyyy-mm-dd index, 11 industry cols and other col
# each col had value of return for each day, earlier in % base converted to normal number.

# canonical paper used 11 industries: ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems', 'BusEq', 'Telcm', 'Utils','Shops', 'Hlth', 'Money']

# check data input file again it is weird!!


# TODO: fix data input
# create markowitz, sharpe, profitability, market strat, 1/t strat, notebook
