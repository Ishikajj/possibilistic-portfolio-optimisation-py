"""Tests for pipeline_wrapper.py — dataset loops and per-dataset output files.

The numbers themselves are covered by tests/test_snapshots.py; here we test
the orchestration: which datasets get run, under which names, which files land
where.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio import evaluation_pipeline as ev
from possibilistic_portfolio import pipeline_wrapper as pw
from possibilistic_portfolio import possibilistic_bayesian
from possibilistic_portfolio.predictive import load_predictive, make_predictive

VW = "Average Value Weighted Returns -- Daily"
EW = "Average Equal Weighted Returns -- Daily"


def _french_file(path: Path, extra_table: bool = False) -> Path:
    dates = pd.bdate_range("1980-01-02", periods=6).strftime("%Y%m%d")
    vw = "\n".join(f"{d},  1.0,  2.0" for d in dates)
    ew = "\n".join(f"{d},  3.0,  4.0" for d in dates)
    firms = (
        f"\n  Number of Firms in Portfolios\n,P1,P2\n{dates[0]}, 500, 600\n"
        if extra_table
        else ""
    )
    path.write_text(
        f"header text\n\n  {VW}\n,P1,P2\n{vw}\n\n  {EW}\n,P1,P2\n{ew}\n{firms}\nCopyright 2026\n"
    )
    return path


def _ff_file(path: Path) -> Path:
    dates = pd.bdate_range("1980-01-02", periods=6).strftime("%Y%m%d")
    rows = "\n".join(f"{d},  0.5,  0.0,  0.0,  0.1" for d in dates)
    path.write_text(f"l1\nl2\nl3\n\n,Mkt-RF,SMB,HML,RF\n{rows}\n\nCopyright\n")
    return path


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Replace the per-dataset run with a stub that records its inputs."""
    calls = []

    def fake(dataset, dataset_name, output_dir=None, **params):
        calls.append(
            {"name": dataset_name, "data": dataset.copy(), "params": params}
        )
        folder = Path(output_dir or tmp_path) / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    monkeypatch.setattr(pw, "calculate_all_functions_perdatasets", fake)
    return calls


# --- call_all_datasets ----------------------------------------------------


def test_real_loop_runs_value_and_equal_weighted_per_file(recorder, tmp_path):
    a = _french_file(tmp_path / "10_A.csv")
    b = _french_file(tmp_path / "6_B.csv")
    folders = pw.call_all_datasets(
        [a, b],
        risk_free_path=_ff_file(tmp_path / "ff.csv"),
        start_date=None,
        output_dir=tmp_path / "out",
        burn_in=7,
        theta=2.0,
    )
    names = [c["name"] for c in recorder]
    assert names == [
        "10_A_value_weighted",
        "10_A_equal_weighted",
        "6_B_value_weighted",
        "6_B_equal_weighted",
    ]
    assert [f.name for f in folders] == names
    # VW block 1%/2% minus RF 0.1% ; EW block 3%/4% minus RF ; market kept
    vw, ew = recorder[0]["data"], recorder[1]["data"]
    assert list(vw.columns) == ["P1", "P2", "Mkt-RF"]
    assert vw.index.equals(pd.bdate_range("1980-01-02", periods=6))  # dates
    # RF saved next to the outputs, on the same dates
    rf = ev.load_rf(folders[0])
    assert rf.index.equals(vw.index) and np.allclose(rf, 0.001)
    assert np.allclose(vw["P1"], 0.009) and np.allclose(ew["P1"], 0.029)
    assert (
        recorder[0]["params"]["burn_in"] == 7
        and recorder[0]["params"]["theta"] == 2.0
    )


def test_real_loop_ignores_extra_tables(recorder, tmp_path):
    f = _french_file(tmp_path / "6_btm.csv", extra_table=True)
    pw.call_all_datasets(
        [f],
        risk_free_path=_ff_file(tmp_path / "ff.csv"),
        start_date=None,
        output_dir=tmp_path / "out",
    )
    ew = recorder[1]["data"]
    assert (
        len(ew) == 6 and ew["P1"].max() < 0.1
    )  # firm counts not read as returns


def test_real_loop_applies_date_window(recorder, tmp_path):
    pw.call_all_datasets(
        [_french_file(tmp_path / "10_A.csv")],
        risk_free_path=_ff_file(tmp_path / "ff.csv"),
        start_date="1980-01-04",
        end_date="1980-01-08",
        output_dir=tmp_path / "out",
    )
    assert len(recorder[0]["data"]) == 3


def test_real_loop_skips_unreadable_file(recorder, tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("not, a\nfrench, file\n")
    good = _french_file(tmp_path / "10_A.csv")
    pw.call_all_datasets(
        [bad, good],
        risk_free_path=_ff_file(tmp_path / "ff.csv"),
        start_date=None,
        output_dir=tmp_path / "out",
    )
    assert [c["name"] for c in recorder] == [
        "10_A_value_weighted",
        "10_A_equal_weighted",
    ]
    assert "WARNING" in capsys.readouterr().out


# --- call_all_simulated_datasets -----------------------------------------


def _sim_files(d: Path, name: str, T: int = 5) -> tuple[str, str, str]:
    rng = np.random.default_rng(0)
    pd.DataFrame(rng.normal(size=(T, 2)), columns=["Asset1", "Asset2"]).to_csv(
        d / f"{name}.csv", index=False
    )
    pd.DataFrame({"RF": np.full(T, 1e-4)}).to_csv(
        d / f"rf_{name}.csv", index=False
    )
    return name, f"{name}.csv", f"rf_{name}.csv"


def test_simulated_loop_runs_each_dataset_and_copies_rf(recorder, tmp_path):
    files = [_sim_files(tmp_path, "iid"), _sim_files(tmp_path, "mr")]
    folders = pw.call_all_simulated_datasets(
        tmp_path, files, output_dir=tmp_path / "out"
    )
    assert [c["name"] for c in recorder] == ["iid", "mr"]
    assert list(recorder[0]["data"].columns) == ["Asset1", "Asset2"]
    for f in folders:
        rf = pd.read_csv(f / "rf.csv")
        assert np.allclose(rf["RF"], 1e-4)


def test_simulated_loop_skips_missing_files(recorder, tmp_path, capsys):
    files = [
        ("ghost", "ghost.csv", "rf_ghost.csv"),
        _sim_files(tmp_path, "iid"),
    ]
    pw.call_all_simulated_datasets(tmp_path, files, output_dir=tmp_path / "out")
    assert [c["name"] for c in recorder] == ["iid"]
    assert "WARNING" in capsys.readouterr().out


# --- per-dataset run: which files are written -----------------------------


def test_save_algo_outputs(tmp_path):
    dates = pd.bdate_range("2000-01-03", periods=2, name="Date")
    w = pd.DataFrame({"a": [0.5, np.nan]}, index=dates)
    ms = make_predictive(np.zeros((2, 1)), np.ones((2, 1, 1)), dates)
    pw._save_algo_outputs(tmp_path, "algo", weights=w, mu_sigma_dict=ms)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "algo_predictive.npz",
        "algo_weights.csv",
    ]
    back = load_predictive(tmp_path, "algo")
    assert back is not None
    assert np.array_equal(back["sigma_hat"], ms["sigma_hat"])
    assert back["index"].equals(dates)  # forecast keeps its dates


PREDICTIVE = [
    "historical_expanding",
    "rolling_window_long",
    "rolling_window_short",
    "jorion_bayes_stein",
    "kan_zhou_three_fund",
    "bayesian_averaging",
    "possibilistic_masked",
    "possibilistic_power",
    "possibilistic_exp",
]


@pytest.mark.parametrize("with_market", [True, False])
def test_per_dataset_run_writes_expected_files(
    monkeypatch, tmp_path, with_market
):
    monkeypatch.setattr(
        possibilistic_bayesian,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.ones(len(mus)),
    )
    rng = np.random.default_rng(0)
    cols = ["P1", "P2", "Mkt-RF"] if with_market else ["P1", "P2", "P3"]
    data = pd.DataFrame(rng.normal(0.0005, 0.01, (90, 3)), columns=cols)
    folder = pw.calculate_all_functions_perdatasets(
        data,
        "ds",
        burn_in=30,
        periods_until_investment=10,
        max_models=8,
        rolling_window_long=20,
        rolling_window_short=10,
        output_dir=tmp_path,
    )
    expected = {"returns.csv", "possibilistic_diagnostics.csv"}
    expected |= {
        f"{a}_weights.csv"
        for a in ["equal_weight", "minimum_variance", *PREDICTIVE]
    }
    expected |= {
        f"{a}_{s}"
        for a in PREDICTIVE
        for s in ["predictive.npz", "longonly_weights.csv"]
    }
    if with_market:
        expected.add("market_weight_weights.csv")
    assert {p.name for p in folder.iterdir()} == expected
    pd.testing.assert_frame_equal(
        pd.read_csv(folder / "returns.csv", index_col=0), data
    )


def test_dates_reach_every_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        possibilistic_bayesian,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.ones(len(mus)),
    )
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("1990-01-02", periods=90, name="Date")
    data = pd.DataFrame(
        rng.normal(0.0005, 0.01, (90, 3)),
        columns=["P1", "P2", "Mkt-RF"],
        index=dates,
    )
    folder = pw.calculate_all_functions_perdatasets(
        data,
        "ds",
        burn_in=30,
        periods_until_investment=10,
        max_models=8,
        rolling_window_long=20,
        rolling_window_short=10,
        output_dir=tmp_path,
    )
    assert ev.load_returns(folder).index.equals(dates)
    for algo in ev.discover_strategies(folder):
        w, pred = ev.load_strategy_outputs(folder, algo)
        assert w.index.equals(dates), algo
        if pred is not None:
            assert pred["index"].equals(dates), algo
