"""Data loading and preparation for Kenneth French portfolio returns.

Two functions form the public API:

load_excess_returns_from_kenneth_french_path(portfolios_path, risk_free_path, ...)
    Loads a Kenneth French CSV, extracts the requested return block (value- or
    equal-weighted), subtracts the risk-free rate, and slices by date. Returns a
    DataFrame that still carries Date, RF, and Mkt-RF columns alongside the asset
    returns — useful for any downstream step that needs date alignment or the RF series.

prepare_returns(df, drop_cols=("Date", "RF"))
    Strips non-numeric metadata columns from the output of the above, leaving a clean
    numeric-only DataFrame ready to pass to the algorithms. Kept separate so callers
    can inspect or slice the dated DataFrame before discarding the metadata.

All returns are stored in decimal units (percent / 100).
"""

from __future__ import annotations

from pathlib import Path
from io import StringIO
import pandas as pd

# Paths (relative to this file)
DATA_PATH = (
    Path(__file__).resolve().parent
    / "../../datasets/10_Industry_Portfolios_Daily.csv"
)

RISK_FREE_RATE_PATH = (
    Path(__file__).resolve().parent
    / "../../datasets/F-F_Research_Data_Factors_daily.csv"
)


def _parse_yyyymmdd_to_datetime(s: pd.Series) -> pd.Series:
    """Parse dates like 19260701 (YYYYMMDD) to datetime."""
    ss = s.astype(str).str.replace(r"\.0$", "", regex=True)
    # If it looks like YYYYMMDD digits, use strict format; otherwise let pandas infer.
    if ss.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(ss, format="%Y%m%d")
    return pd.to_datetime(ss, errors="coerce")


def _read_csv_block_between_markers(
    path: Path,
    start_marker: str,
    end_marker: str,
) -> pd.DataFrame:
    """Read a CSV-like block from a text file between two marker lines (exclusive).

    Many Ken French-style downloads embed multiple tables and notes in one file.
    This helper finds the lines containing `start_marker` and `end_marker`, then
    parses the intervening lines as CSV.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Locate markers
    start_idx = next(
        (i for i, ln in enumerate(lines) if start_marker in ln),
        None,
    )
    end_idx = next(
        (i for i, ln in enumerate(lines) if end_marker in ln),
        None,
    )
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError(
            f"Could not find a valid block between markers in {path}. "
            f"start_marker={start_marker!r}, end_marker={end_marker!r}"
        )

    # The table starts after the marker line and continues until the line before end marker.
    block_lines = [ln for ln in lines[start_idx + 1 : end_idx] if ln.strip()]

    # Defensive: drop obvious non-data lines that sometimes appear inside blocks.
    # Keep lines that start with a digit (date) or a header starting with a letter.
    cleaned: list[str] = []
    for ln in block_lines:
        s = ln.strip()
        if not s:
            continue
        cleaned.append(ln)

    if not cleaned:
        raise ValueError(f"Marker block in {path} was empty after cleaning.")

    csv_text = "\n".join(cleaned)
    return pd.read_csv(StringIO(csv_text))


def load_kenneth_french_portfolios(
    path: Path | None = None,
    start_marker: str = "Average Value Weighted Returns -- Daily",
    end_marker: str | None = "Average Equal Weighted Returns -- Daily",
) -> pd.DataFrame:
    """Load industry portfolios daily returns and return a DataFrame with a datetime Date column.

    Expected: first column is a date-like column (often unnamed) with YYYYMMDD.
    Returns are converted from percent to decimal.

    Parameters
    ----------
    start_marker : str
        Line in the file that precedes the desired data block.
    end_marker : str or None
        Line that terminates the block.  Pass ``None`` to read from
        ``start_marker`` to EOF (used for the equal-weighted block, which
        has no following section header).
    """
    p = path or DATA_PATH

    # extract the requested block if present.
    try:
        if end_marker is not None:
            df = _read_csv_block_between_markers(
                p,
                start_marker=start_marker,
                end_marker=end_marker,
            )
        else:
            # Read from start_marker to EOF, stopping at copyright or next section header.
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            start_idx = next(
                (i for i, ln in enumerate(lines) if start_marker in ln), None
            )
            if start_idx is None:
                raise ValueError(
                    f"start_marker={start_marker!r} not found in {p}"
                )
            cleaned: list[str] = []
            for ln in lines[start_idx + 1 :]:
                s = ln.strip()
                if not s:
                    continue
                if " -- " in s and not s[0].isdigit() and not s.startswith(","):
                    break
                if s.lower().startswith("copyright"):
                    break
                cleaned.append(ln)
            if not cleaned:
                raise ValueError(
                    f"No data found after marker {start_marker!r} in {p}"
                )
            from io import StringIO as _StringIO

            df = pd.read_csv(_StringIO("\n".join(cleaned)))
    except Exception:
        # Fallback: treat as a normal CSV
        df = pd.read_csv(p)

    # Rename first column to Date (often "Unnamed: 0" or similar)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = _parse_yyyymmdd_to_datetime(df["Date"])

    # Convert all non-Date columns to numeric and percent->decimal
    for c in df.columns:
        if c == "Date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100

    # Drop rows with bad dates
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_risk_free_rate(path: Path | None = None) -> pd.DataFrame:
    """Load the risk-free rate (RF) and return a DataFrame with a datetime Date column.

    The Fama-French-style CSV usually has header text; we skip the first 4 rows.
    RF is converted from percent to decimal.
    """
    p = path or RISK_FREE_RATE_PATH
    df = pd.read_csv(p, skiprows=4)

    # Rename unnamed date column to "Date"
    df = df.rename(columns={df.columns[0]: "Date"})

    # Convert YYYYMMDD integer/string to datetime
    df["Date"] = _parse_yyyymmdd_to_datetime(df["Date"])

    # Keep only Date and RF
    if "RF" not in df.columns:
        raise KeyError("RF column not found in risk-free rate file.")
    df = df[["Date", "RF", "Mkt-RF"]]

    # Convert percent columns to numeric (individually)
    df["RF"] = pd.to_numeric(df["RF"], errors="coerce")
    df["Mkt-RF"] = pd.to_numeric(df["Mkt-RF"], errors="coerce")

    df = (
        df.dropna(subset=["Date", "RF", "Mkt-RF"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Percent -> decimal
    df[["RF", "Mkt-RF"]] = df[["RF", "Mkt-RF"]] / 100
    return df


def calculate_excess_returns(
    portfolios_df: pd.DataFrame, risk_free_rate_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute excess returns by subtracting RF from each risky asset return (aligned by Date)."""
    left = portfolios_df.copy()
    right = risk_free_rate_df.copy()

    if "Date" not in left.columns or "Date" not in right.columns:
        raise KeyError("Both DataFrames must have a 'Date' column.")

    out = left.merge(right[["Date", "RF", "Mkt-RF"]], on="Date", how="inner")

    # Subtract RF from each return column (exclude Date, RF, and any metadata columns)
    exclude = {"Date", "RF", "Mkt-RF"}
    ret_cols = [c for c in out.columns if c not in exclude]
    for c in ret_cols:
        out[c] = out[c] - out["RF"]

    return out


def slice_timeframe(
    df: pd.DataFrame,
    start_date: str | None = "1963-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:

    if "Date" not in df.columns:
        raise KeyError("DataFrame must contain a 'Date' column.")

    out = df.copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        out = out.loc[out["Date"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        out = out.loc[out["Date"] <= end_date]

    return out.sort_values("Date").reset_index(drop=True)


def load_excess_returns_from_kenneth_french_path(
    portfolios_path: Path | str = DATA_PATH,
    risk_free_path: Path | str = RISK_FREE_RATE_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    start_marker: str = "Average Value Weighted Returns -- Daily",
    end_marker: str | None = "Average Equal Weighted Returns -- Daily",
) -> pd.DataFrame:
    """
    Minimum callable wrapper to:
    1. Load Kenneth French portfolios
    2. Load risk-free rate
    3. Slice both by Date
    4. Return excess returns DataFrame
    """

    portfolios_path = Path(portfolios_path)
    risk_free_path = Path(risk_free_path)

    portfolios_df = load_kenneth_french_portfolios(
        portfolios_path, start_marker=start_marker, end_marker=end_marker
    )
    rf_df = load_risk_free_rate(risk_free_path)

    portfolios_df = slice_timeframe(portfolios_df, start_date, end_date)
    rf_df = slice_timeframe(rf_df, start_date, end_date)

    excess_df = calculate_excess_returns(portfolios_df, rf_df)

    return excess_df


def prepare_returns(
    df: pd.DataFrame, drop_cols: tuple[str, ...] = ("Date", "RF")
) -> pd.DataFrame:
    """Return numeric returns-only DataFrame."""
    out = df.copy()
    for c in drop_cols:
        if c in out.columns:
            out = out.drop(columns=c)
    out = out.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return out


if __name__ == "__main__":
    df_risk_free = load_risk_free_rate()
    print(df_risk_free.head())
    print(df_risk_free.shape)
    df = load_excess_returns_from_kenneth_french_path(
        DATA_PATH, RISK_FREE_RATE_PATH
    )
    print(df.head())
    print(df.columns)
    print(df.dtypes)
