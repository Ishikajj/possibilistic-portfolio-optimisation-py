"""Tests for data_input.py — Kenneth French / Fama-French file parsing and excess returns."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_input as di

VW = "Average Value Weighted Returns -- Daily"
EW = "Average Equal Weighted Returns -- Daily"
DATASETS = Path(__file__).resolve().parents[2] / "datasets"

# Same layout as the real downloads: header text, VW block, EW block, copyright.
FRENCH_FILE = f"""This file was created by CMPT_IND_RETS_DAILY using the 202601 CRSP database.
It contains value- and equal-weighted returns for 2 portfolios.

  {VW}
,Alpha,Beta
19800103,  1.00,  -2.00
19800102,  0.50,   0.25
19800104, -1.00,   3.00

  {EW}
,Alpha,Beta
19800102,  0.10,   0.20
19800103,  0.30,   0.40
19800104,  0.50,   0.60

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

# Same layout as F-F_Research_Data_Factors_daily.csv: 4 lines before the header.
FF_FILE = """This file was created by using the 202601 CRSP database.
The Tbill return is the simple daily rate.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
19800102,    1.00,   -0.25,   -0.27,    0.10
19800103,    2.00,   -0.25,   -0.27,    0.20
19800104,    3.00,   -0.25,   -0.27,    0.30
19800107,    4.00,   -0.25,   -0.27,    0.40

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


@pytest.fixture
def french_path(tmp_path: Path) -> Path:
    p = tmp_path / "portfolios.csv"
    p.write_text(FRENCH_FILE)
    return p


@pytest.fixture
def ff_path(tmp_path: Path) -> Path:
    p = tmp_path / "ff.csv"
    p.write_text(FF_FILE)
    return p


# --- dates ----------------------------------------------------------------


def test_parse_yyyymmdd_from_ints_and_float_strings():
    out = di._parse_yyyymmdd_to_datetime(pd.Series([19800102, "19800103.0"]))
    assert list(out) == [pd.Timestamp("1980-01-02"), pd.Timestamp("1980-01-03")]


# --- Kenneth French portfolio blocks --------------------------------------


def test_value_weighted_block(french_path):
    df = di.load_kenneth_french_portfolios(french_path, VW, EW)
    assert list(df.columns) == ["Date", "Alpha", "Beta"]
    assert list(df["Date"]) == list(
        pd.to_datetime(["1980-01-02", "1980-01-03", "1980-01-04"])
    )
    # sorted by date and converted from percent to decimal
    assert np.allclose(df["Alpha"], [0.005, 0.01, -0.01])
    assert np.allclose(df["Beta"], [0.0025, -0.02, 0.03])


def test_equal_weighted_block_runs_to_copyright(french_path):
    df = di.load_kenneth_french_portfolios(french_path, EW, None)
    assert len(df) == 3
    assert np.allclose(df["Alpha"], [0.001, 0.003, 0.005])
    assert np.allclose(df["Beta"], [0.002, 0.004, 0.006])


def test_equal_weighted_block_stops_at_explicit_end_marker(tmp_path):
    p = tmp_path / "with_firms.csv"
    p.write_text(
        FRENCH_FILE.replace(
            "\nCopyright",
            "\n  Number of Firms in Portfolios\n,Alpha,Beta\n19800102, 500, 600\n\nCopyright",
        )
    )
    df = di.load_kenneth_french_portfolios(
        p, EW, "Number of Firms in Portfolios"
    )
    assert len(df) == 3
    assert df["Alpha"].max() < 0.01  # firm counts (500) not read as returns


# --- risk-free file -------------------------------------------------------


def test_load_risk_free_rate(ff_path):
    rf = di.load_risk_free_rate(ff_path)
    assert list(rf.columns) == ["Date", "RF", "Mkt-RF"]  # SMB / HML dropped
    assert len(rf) == 4
    assert np.allclose(rf["RF"], [0.001, 0.002, 0.003, 0.004])
    assert np.allclose(rf["Mkt-RF"], [0.01, 0.02, 0.03, 0.04])


# --- excess returns, slicing, preparation ---------------------------------


def test_excess_returns_subtract_rf_but_not_from_market(french_path, ff_path):
    ports = di.load_kenneth_french_portfolios(french_path, VW, EW)
    rf = di.load_risk_free_rate(ff_path)
    ex = di.calculate_excess_returns(ports, rf)

    assert list(ex.columns) == ["Date", "Alpha", "Beta", "RF", "Mkt-RF"]
    assert np.allclose(ex["Alpha"], ports["Alpha"] - rf["RF"].iloc[:3])
    # Mkt-RF is already an excess return in the FF file -> unchanged
    assert np.allclose(ex["Mkt-RF"], rf["Mkt-RF"].iloc[:3])


def test_excess_returns_keep_only_common_dates(french_path, ff_path):
    ports = di.load_kenneth_french_portfolios(french_path, VW, EW)
    rf = di.load_risk_free_rate(ff_path)  # has an extra date 1980-01-07
    ex = di.calculate_excess_returns(
        ports, rf.iloc[1:]
    )  # drop 1980-01-02 from RF
    assert list(ex["Date"]) == list(
        pd.to_datetime(["1980-01-03", "1980-01-04"])
    )


def test_slice_timeframe_bounds_are_inclusive():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            "x": [1, 2, 3],
        }
    )
    assert list(di.slice_timeframe(df, "2000-01-02", "2000-01-03")["x"]) == [
        2,
        3,
    ]
    assert list(di.slice_timeframe(df, None, "2000-01-01")["x"]) == [1]
    assert list(di.slice_timeframe(df, None, None)["x"]) == [1, 2, 3]


def test_slice_timeframe_requires_date_column():
    with pytest.raises(KeyError):
        di.slice_timeframe(pd.DataFrame({"x": [1]}))


def test_wrapper_equals_manual_chain(french_path, ff_path):
    ex = di.load_excess_returns_from_kenneth_french_path(
        french_path,
        ff_path,
        start_date="1980-01-03",
        start_marker=VW,
        end_marker=EW,
    )
    manual = di.calculate_excess_returns(
        di.slice_timeframe(
            di.load_kenneth_french_portfolios(french_path, VW, EW), "1980-01-03"
        ),
        di.slice_timeframe(di.load_risk_free_rate(ff_path), "1980-01-03"),
    )
    pd.testing.assert_frame_equal(ex, manual)


def test_prepare_returns_drops_date_and_rf_only(french_path, ff_path):
    ex = di.load_excess_returns_from_kenneth_french_path(
        french_path, ff_path, start_marker=VW, end_marker=EW
    )
    out = di.prepare_returns(ex)
    assert list(out.columns) == [
        "Alpha",
        "Beta",
        "Mkt-RF",
    ]  # market kept as an asset
    assert all(np.issubdtype(t, np.number) for t in out.dtypes)


# --- real data (skipped if ../datasets is missing) ------------------------


@pytest.mark.skipif(not DATASETS.exists(), reason="../datasets not available")
def test_real_industry_file_loads_cleanly():
    ex = di.load_excess_returns_from_kenneth_french_path(
        DATASETS / "10_Industry_Portfolios_Daily.csv",
        DATASETS / "F-F_Research_Data_Factors_daily.csv",
        start_date="1980-01-01",
        start_marker=VW,
        end_marker=EW,
    )
    out = di.prepare_returns(ex)
    assert out.shape[1] == 11  # 10 industries (incl. "Other") + Mkt-RF
    assert "Other" in out.columns and "Mkt-RF" in out.columns
    assert (
        ex["Date"].is_monotonic_increasing and not ex["Date"].duplicated().any()
    )
    assert out.notna().all().all()
    assert (
        (out.abs() < 0.5).all().all()
    )  # decimals, no -99.99 missing-value codes
