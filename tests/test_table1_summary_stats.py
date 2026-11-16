"""Tests for table1_summary_stats.py — Table 1 summary statistics (percent units)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio import simulated_datasets as sd
from possibilistic_portfolio import table1_summary_stats as t1
from possibilistic_portfolio.data_input import TABLE_TITLES

VW, EW = TABLE_TITLES["value"], TABLE_TITLES["equal"]


@pytest.fixture
def config(tmp_path: Path) -> t1.PortfolioConfig:
    p = tmp_path / "ports.csv"
    p.write_text(
        f"header\n\n  {VW}\n,A,B\n"
        "19800102,  1.00, -99.99\n19800103, -2.00,  0.50\n19800104,  3.00,  1.50\n"
        f"\n  {EW}\n,A,B\n"
        "19800102,  0.10,  0.20\n19800103,  0.30,  0.40\n"
        "\nCopyright 2026\n"
    )
    return t1.PortfolioConfig(name="Test", label="tst", path=p)


def test_load_raw_pct_keeps_percent_and_masks_missing_code(config):
    df = t1.load_raw_pct(config, "value")
    assert np.allclose(df["A"], [1.0, -2.0, 3.0])  # percent, not decimals
    assert np.isnan(df["B"].iloc[0])  # -99.99 -> NaN
    assert np.allclose(df["B"].iloc[1:], [0.5, 1.5])


def test_load_raw_pct_equal_block_and_dates(config):
    df = t1.load_raw_pct(config, "equal", start_date="1980-01-03")
    assert len(df) == 1 and np.isclose(df["A"].iloc[0], 0.3)


def test_load_raw_pct_rejects_unknown_weighting(config):
    with pytest.raises(ValueError):
        t1.load_raw_pct(config, "market")


def test_compute_stats():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]),
            "A": [1.0, 2.0, 3.0],
            "B": [-1.0, 0.0, 4.0],
        }
    )
    s = t1.compute_stats(df)
    assert s["n_assets"] == 2 and s["sample_size"] == 3
    assert s["start_date"] == "2000-01-03" and s["end_date"] == "2000-01-05"
    assert np.isclose(s["lo_mean"], 1.0) and np.isclose(s["hi_mean"], 2.0)
    assert np.isclose(s["lo_std"], 1.0) and np.isclose(
        s["hi_std"], np.std([-1, 0, 4], ddof=1)
    )
    assert s["lo_return"] == -1.0 and s["hi_return"] == 4.0


def test_build_panel_one_row_per_config(config):
    panel = t1.build_panel([config, config], "value")
    assert len(panel) == 2
    assert {"Name", "Label", "N", "T", "Mean low", "Return high"} <= set(
        panel.columns
    )
    assert panel["T"].iloc[0] == 3


def test_sim_datasets_round_trip(tmp_path):
    sd.generate_and_save_sim_datasets(
        ["iid"], T=50, n_assets=3, seed=1, data_dir=tmp_path
    )
    df = t1.load_sim_pct("iid", T=50, n_assets=3, seed=1, data_dir=tmp_path)
    ports, _ = sd.simulate_iid(T=50, n_assets=3, seed=1)
    assert np.allclose(df.values, ports.values * 100)  # percent units
    assert (tmp_path / "rf_iid_T50_N3_seed1.csv").exists()
    panel = t1.build_sim_panel(
        ["iid"], T=50, n_assets=3, seed=1, data_dir=tmp_path
    )
    assert panel["N"].iloc[0] == 3 and panel["T"].iloc[0] == 50
    assert panel["Name"].iloc[0] == "IID"


def test_sim_names_cover_every_dgp():
    assert list(t1.SIM_NAMES) == list(sd.DGPS)


def test_load_sim_pct_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        t1.load_sim_pct("iid", T=50, n_assets=3, seed=1, data_dir=tmp_path)


def test_portfolio_configs_point_at_existing_files():
    if not t1.DATASETS_DIR.exists():
        pytest.skip("../datasets not available")
    assert len(t1.PORTFOLIO_CONFIGS) == 10
    for cfg in t1.PORTFOLIO_CONFIGS:
        assert cfg.path.exists(), cfg.path
