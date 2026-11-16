"""Load Kenneth French portfolio returns and the Fama-French risk-free rate.

Every table in these CSV downloads has the same shape:

    <title line>                    e.g. "Average Value Weighted Returns -- Daily"
    ,<col 1>,<col 2>,...            header; the empty first field is the date
    YYYYMMDD,<value>,<value>,...    data rows, in percent

The Fama-French factors file has one such table with no title. Tables are
found by this shape, so no start/end markers or skipped-row counts are needed,
and extra tables (e.g. "Number of Firms in Portfolios") are simply not asked for.

Public API:
  read_french_tables(path)            every table in a file, keyed by title (percent)
  load_portfolios(path, weighting)    value- or equal-weighted block (decimals)
  load_risk_free(path)                Date, RF, Mkt-RF (decimals)
  load_excess_returns(ports, rf, ...) portfolios minus RF, plus RF and Mkt-RF
  prepare_returns(df)                 drop Date / RF -> numeric frame for the algorithms

All loaders take an optional inclusive [start_date, end_date] window.
Missing-value codes (-99.99, -999) become NaN. Mkt-RF is already an excess
return, so RF is not subtracted from it.
Portfolio days with no RF are dropped with a warning (error above 1%).
"""

from __future__ import annotations

import re
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_TITLES = {
    "value": "Average Value Weighted Returns -- Daily",
    "equal": "Average Equal Weighted Returns -- Daily",
}
MISSING_CODES = [-99.99, -999.0]  # file headers: "indicated by -99.99 or -999."
MAX_MISSING_RF_SHARE = 0.01  # more portfolio days without RF -> error

_DATA_ROW = re.compile(r"\s*\d{6,8}\s*,")


def _parse_yyyymmdd_to_datetime(s: pd.Series) -> pd.Series:
    """Parse dates like 19260701 (YYYYMMDD) to datetime."""
    ss = s.astype(str).str.replace(r"\.0$", "", regex=True)
    # If it looks like YYYYMMDD digits, use strict format; otherwise let pandas infer.
    if ss.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(ss, format="%Y%m%d")
    return pd.to_datetime(ss, errors="coerce")


def _strip(line: str) -> str:
    """Line without surrounding whitespace and trailing commas (Excel padding)."""
    return line.strip().rstrip(",").strip()


def _slice_dates(
    df: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Rows with start_date <= Date <= end_date (None = open), sorted by Date."""
    out = df.dropna(subset=["Date"])
    if start_date is not None:
        out = out.loc[out["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out.loc[out["Date"] <= pd.to_datetime(end_date)]
    return out.sort_values("Date").reset_index(drop=True)


def _from_percent(df: pd.DataFrame, percent: bool) -> pd.DataFrame:
    """The files store values in percent; return decimals unless percent=True.

    The only place units are converted, so every loader follows one convention.
    """
    if not percent:
        values = df.columns[1:]  # everything except Date
        df[values] = df[values] / 100
    return df


def read_french_tables(path: Path | str) -> dict[str, pd.DataFrame]:
    """Every table in a Kenneth French / Fama-French CSV, keyed by its title.

    A table is a header line starting with "," followed by date rows; its
    title is the line right above the header ("" if that line is empty).
    The first column becomes "Date"; values stay in percent, with the
    missing-value codes replaced by NaN.
    """
    lines: list[str] = (
        Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    )
    tables: dict[str, pd.DataFrame] = {}
    i = 0
    while i < len(lines) - 1:
        is_header = lines[i].lstrip().startswith(",") and _strip(lines[i])
        if not (is_header and _DATA_ROW.match(lines[i + 1])):
            i += 1
            continue
        end = i + 1  # i is at the header, end will be at data row.
        while end < len(lines) and _DATA_ROW.match(lines[end]):
            end += 1
        title = (
            _strip(lines[i - 1]) if i > 0 else ""
        )  # the value weighted or eq weighted"

        df = pd.read_csv(StringIO("\n".join(lines[i:end])))
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = _parse_yyyymmdd_to_datetime(df["Date"])
        values = df.columns[1:]
        df[values] = (
            df[values]
            .apply(pd.to_numeric, errors="coerce")
            .replace(MISSING_CODES, np.nan)
        )
        tables[title] = df
        i = end
    return tables


def load_portfolios(
    path: Path | str,
    weighting: str = "value",
    start_date: str | None = None,
    end_date: str | None = None,
    percent: bool = False,
) -> pd.DataFrame:
    """Value- or equal-weighted daily portfolio returns: Date + one column per portfolio.

    Decimals by default; percent=True keeps the file's percent units.
    Dates: inclusive window [start_date, end_date]; None = no bound.
    """
    if weighting not in TABLE_TITLES:
        raise ValueError(
            f"weighting must be one of {list(TABLE_TITLES)}, got {weighting!r}"
        )
    title = TABLE_TITLES[weighting]
    tables = read_french_tables(path)
    if title not in tables:
        raise ValueError(
            f"No table {title!r} in {path}; tables found: {list(tables)}"
        )
    df = _from_percent(tables[title], percent)
    return _slice_dates(df, start_date, end_date)


def load_risk_free(
    path: Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
    percent: bool = False,
) -> pd.DataFrame:
    """Daily risk-free rate and market excess return: Date, RF, Mkt-RF.

    Decimals by default; percent=True keeps the file's percent units.
    Dates: inclusive window [start_date, end_date]; None = no bound.
    """
    tables = read_french_tables(path)
    df = next((t for t in tables.values() if "RF" in t.columns), None)
    if df is None:
        raise ValueError(f"No table with an RF column in {path}")
    df = _from_percent(df[["Date", "RF", "Mkt-RF"]].dropna(), percent)
    return _slice_dates(df, start_date, end_date)


def load_excess_returns(
    portfolios_path: Path | str,
    risk_free_path: Path | str,
    weighting: str = "value",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Portfolio excess returns on the dates both files share.

    Columns: Date, <portfolios minus RF>, RF, Mkt-RF.
    Dates: inclusive window [start_date, end_date]; None = no bound.
    Portfolio days with no RF value are dropped with a warning; if more than
    MAX_MISSING_RF_SHARE of them are missing, the files probably don't belong
    together (e.g. monthly vs daily) and a ValueError is raised instead.
    """
    ports = load_portfolios(portfolios_path, weighting, start_date, end_date)
    rf = load_risk_free(risk_free_path, start_date, end_date)

    missing = ports.loc[~ports["Date"].isin(rf["Date"]), "Date"]
    if len(missing) > MAX_MISSING_RF_SHARE * len(ports):
        raise ValueError(
            f"{len(missing)} of {len(ports)} portfolio days in {portfolios_path} "
            f"have no RF in {risk_free_path} — do the files belong together?"
        )
    if len(missing) > 0:
        first = ", ".join(d.strftime("%Y-%m-%d") for d in missing.iloc[:5])
        warnings.warn(
            f"{len(missing)} of {len(ports)} portfolio days have no RF and were "
            f"dropped: {first}{', ...' if len(missing) > 5 else ''}",
            RuntimeWarning,
            stacklevel=2,
        )

    out = ports.merge(rf, on="Date", how="inner")
    assets = [c for c in ports.columns if c != "Date"]
    out[assets] = out[assets].sub(out["RF"], axis=0)
    return out


def prepare_returns(
    df: pd.DataFrame, drop_cols: tuple[str, ...] = ("RF",)
) -> pd.DataFrame:
    """Numeric returns only, indexed by Date (if there is a Date column).

    Drops ``drop_cols`` (RF by default) and rows with any missing value.
    Mkt-RF and every portfolio are kept.
    """
    out = df.set_index("Date") if "Date" in df.columns else df.copy()
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out.apply(pd.to_numeric, errors="coerce").dropna(how="any")
