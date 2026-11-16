"""Tests for data_input.py — Kenneth French / Fama-French file parsing and excess returns."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio import data_input as di
from possibilistic_portfolio.config import DATASETS_DIR, RF_PATH

VW = di.TABLE_TITLES["value"]
EW = di.TABLE_TITLES["equal"]

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

# Same layout as F-F_Research_Data_Factors_daily.csv: text, blank line, untitled table.
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


# --- finding tables -------------------------------------------------------


def test_read_french_tables_finds_every_table_by_title(french_path, ff_path):
    tables = di.read_french_tables(french_path)
    assert list(tables) == [VW, EW]
    assert list(tables[VW].columns) == ["Date", "Alpha", "Beta"]
    assert tables[VW]["Alpha"].iloc[0] == 1.00  # still percent, file order
    assert list(di.read_french_tables(ff_path)) == [""]  # untitled table


def test_read_french_tables_handles_excel_comma_padding(tmp_path):
    # Some downloads pad every line with commas (e.g. booktomarket).
    p = tmp_path / "padded.csv"
    p.write_text(
        f"Header text,,,\n,,,\n{VW},,,\n,A,B\n19800102,1.0,2.0\n,,,\nCopyright,,,\n"
    )
    tables = di.read_french_tables(p)
    assert list(tables) == [VW]
    assert len(tables[VW]) == 1


@pytest.mark.parametrize("code", ["-99.99", "-999"])
def test_missing_value_codes_become_nan(tmp_path, code):
    p = tmp_path / "missing.csv"
    p.write_text(f"{VW}\n,A,B\n19800102, 1.00, {code}\n19800103, 2.00, 3.00\n")
    df = di.load_portfolios(p, "value")
    assert np.isnan(df["B"].iloc[0])
    assert np.allclose(df["A"], [0.01, 0.02])  # other cells untouched


# --- Kenneth French portfolio blocks --------------------------------------


def test_value_weighted_block(french_path):
    df = di.load_portfolios(french_path, "value")
    assert list(df.columns) == ["Date", "Alpha", "Beta"]
    assert list(df["Date"]) == list(
        pd.to_datetime(["1980-01-02", "1980-01-03", "1980-01-04"])
    )
    # sorted by date and converted from percent to decimal
    assert np.allclose(df["Alpha"], [0.005, 0.01, -0.01])
    assert np.allclose(df["Beta"], [0.0025, -0.02, 0.03])


def test_equal_weighted_block_stops_before_copyright(french_path):
    df = di.load_portfolios(french_path, "equal")
    assert len(df) == 3
    assert np.allclose(df["Alpha"], [0.001, 0.003, 0.005])
    assert np.allclose(df["Beta"], [0.002, 0.004, 0.006])


def test_extra_tables_after_equal_weighted_are_not_read(tmp_path):
    # 6_Portfolios_size_btm has "Number of Firms" (and more) after the EW block.
    p = tmp_path / "with_firms.csv"
    p.write_text(
        FRENCH_FILE.replace(
            "\nCopyright",
            "\n  Number of Firms in Portfolios\n,Alpha,Beta\n19800102, 500, 600\n\nCopyright",
        )
    )
    df = di.load_portfolios(p, "equal")
    assert len(df) == 3
    assert df["Alpha"].max() < 0.01  # firm counts (500) not read as returns


def test_percent_option_keeps_file_units(french_path):
    df = di.load_portfolios(french_path, "value", percent=True)
    assert np.allclose(df["Alpha"], [0.5, 1.0, -1.0])


def test_missing_table_gives_clear_error(tmp_path):
    p = tmp_path / "not_french.csv"
    p.write_text("not, a\nfrench, file\n")
    with pytest.raises(ValueError, match="tables found: \\[\\]"):
        di.load_portfolios(p, "value")


def test_unknown_weighting_rejected(french_path):
    with pytest.raises(ValueError, match="weighting"):
        di.load_portfolios(french_path, "market")


# --- risk-free file -------------------------------------------------------


def test_load_risk_free(ff_path):
    rf = di.load_risk_free(ff_path)
    assert list(rf.columns) == ["Date", "RF", "Mkt-RF"]  # SMB / HML dropped
    assert len(rf) == 4
    assert np.allclose(rf["RF"], [0.001, 0.002, 0.003, 0.004])
    assert np.allclose(rf["Mkt-RF"], [0.01, 0.02, 0.03, 0.04])


def test_percent_option_same_units_for_portfolios_and_rf(french_path, ff_path):
    for percent, scale in [(False, 0.01), (True, 1.0)]:
        ports = di.load_portfolios(french_path, "value", percent=percent)
        rf = di.load_risk_free(ff_path, percent=percent)
        assert np.allclose(ports["Alpha"], np.array([0.5, 1.0, -1.0]) * scale)
        assert np.allclose(rf["RF"], np.array([0.1, 0.2, 0.3, 0.4]) * scale)


def test_load_risk_free_does_not_depend_on_header_length(tmp_path):
    # Old code skipped exactly 4 rows; the table is now found by its shape.
    p = tmp_path / "ff_long_header.csv"
    p.write_text("extra line\nanother one\n" + FF_FILE)
    assert len(di.load_risk_free(p)) == 4


# --- excess returns and date window ---------------------------------------


def test_excess_returns_subtract_rf_but_not_from_market(french_path, ff_path):
    ports = di.load_portfolios(french_path, "value")
    rf = di.load_risk_free(ff_path)
    ex = di.load_excess_returns(french_path, ff_path, "value")

    assert list(ex.columns) == ["Date", "Alpha", "Beta", "RF", "Mkt-RF"]
    assert np.allclose(ex["Alpha"], ports["Alpha"] - rf["RF"].iloc[:3])
    # Mkt-RF is already an excess return in the FF file -> unchanged
    assert np.allclose(ex["Mkt-RF"], rf["Mkt-RF"].iloc[:3])


def _long_files(tmp_path: Path, n_days: int, drop_rf: int) -> tuple[Path, Path]:
    """Portfolio + FF files over n_days business days; the RF file lacks the
    first drop_rf of them."""
    days = pd.bdate_range("1980-01-02", periods=n_days).strftime("%Y%m%d")
    ports = tmp_path / "long_ports.csv"
    ports.write_text(f"{VW}\n,A\n" + "".join(f"{d}, 1.0\n" for d in days))
    ff = tmp_path / "long_ff.csv"
    ff.write_text(
        ",Mkt-RF,SMB,HML,RF\n"
        + "".join(f"{d}, 1.0, 0.0, 0.0, 0.1\n" for d in days[drop_rf:])
    )
    return ports, ff


def test_portfolio_days_without_rf_are_dropped_with_warning(tmp_path):
    ports, ff = _long_files(tmp_path, n_days=200, drop_rf=1)  # 0.5% missing
    with pytest.warns(RuntimeWarning, match="1 of 200 .* 1980-01-02"):
        ex = di.load_excess_returns(ports, ff)
    assert len(ex) == 199
    assert ex["Date"].iloc[0] == pd.Timestamp("1980-01-03")


def test_many_days_without_rf_is_an_error(french_path, tmp_path):
    p = tmp_path / "ff_missing_day.csv"  # 1 of 3 days missing = 33%
    p.write_text(
        FF_FILE.replace("19800102,    1.00,   -0.25,   -0.27,    0.10\n", "")
    )
    with pytest.raises(ValueError, match="1 of 3 portfolio days"):
        di.load_excess_returns(french_path, p, "value")


def test_matching_files_give_no_warning(french_path, ff_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        di.load_excess_returns(french_path, ff_path)


def test_date_window_is_inclusive(french_path, ff_path):
    def dates(start, end):
        ex = di.load_excess_returns(french_path, ff_path, "value", start, end)
        return [d.day for d in ex["Date"]]

    assert dates("1980-01-03", "1980-01-04") == [3, 4]
    assert dates(None, "1980-01-02") == [2]
    assert dates(None, None) == [2, 3, 4]


def test_prepare_returns_indexes_by_date_and_drops_rf(french_path, ff_path):
    excess = di.load_excess_returns(french_path, ff_path)
    out = di.prepare_returns(excess)
    assert list(out.columns) == ["Alpha", "Beta", "Mkt-RF"]  # market kept
    assert all(np.issubdtype(t, np.number) for t in out.dtypes)
    assert out.index.name == "Date"
    assert out.index.equals(pd.DatetimeIndex(excess["Date"], name="Date"))


# --- real data (skipped if ../datasets is missing) ------------------------

REAL_FILES = sorted(
    p for p in DATASETS_DIR.glob("*_*.csv") if not p.name.startswith("F-F")
)


@pytest.mark.skipif(
    not DATASETS_DIR.exists(), reason="../datasets not available"
)
@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.stem)
def test_real_files_load_both_blocks_cleanly(path):
    for weighting in ("value", "equal"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no days without RF
            ex = di.load_excess_returns(path, RF_PATH, weighting, "1980-01-01")
        out = di.prepare_returns(ex)
        assert out.shape[1] in (7, 11)  # 6 or 10 portfolios + Mkt-RF
        assert "Mkt-RF" in out.columns
        assert ex["Date"].is_monotonic_increasing
        assert not ex["Date"].duplicated().any()
        assert out.notna().all().all()
        assert (out.abs() < 0.5).all().all()  # decimals, no missing codes


@pytest.mark.skipif(
    not DATASETS_DIR.exists(), reason="../datasets not available"
)
def test_real_industry_file_keeps_other_column():
    out = di.prepare_returns(
        di.load_excess_returns(
            DATASETS_DIR / "10_Industry_Portfolios_Daily.csv", RF_PATH
        )
    )
    assert "Other" in out.columns  # an industry, not junk
