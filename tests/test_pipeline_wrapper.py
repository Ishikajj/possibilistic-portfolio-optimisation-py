"""Tests for pipeline_wrapper.py — dataset loops and per-dataset output files.

The numbers themselves are covered by tests/test_snapshots.py; here we test
the orchestration: which datasets get run, under which names, which files land
where.
"""

import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

    def fake(
        dataset,
        dataset_name,
        *,
        settings=None,
        algorithms=None,
        output_dir=None,
        resume_done=None,
        on_done=None,
    ):
        calls.append(
            {
                "name": dataset_name,
                "data": dataset.copy(),
                "settings": settings,
                "algorithms": algorithms,
            }
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
        settings=pw.RunSettings(burn_in=7, theta=2.0),
        algorithms=["bayesian_averaging"],
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
    # settings and algorithm selection reach the per-dataset run
    assert recorder[0]["settings"] == pw.RunSettings(burn_in=7, theta=2.0)
    assert recorder[0]["algorithms"] == ["bayesian_averaging"]


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


def test_real_loop_raises_on_missing_path_before_running(recorder, tmp_path):
    """Was: any loading error skipped the dataset with a printed warning, so
    a typo in a path silently dropped it from the run."""
    good = _french_file(tmp_path / "10_A.csv")
    with pytest.raises(FileNotFoundError, match="10_typo.csv"):
        pw.call_all_datasets(
            [good, tmp_path / "10_typo.csv"],
            risk_free_path=_ff_file(tmp_path / "ff.csv"),
            output_dir=tmp_path / "out",
        )
    assert recorder == [] and not (tmp_path / "out").exists()


def test_real_loop_skips_unreadable_file(recorder, tmp_path):
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
    assert (
        "WARNING Could not load" in (tmp_path / "out" / "run.log").read_text()
    )


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


def test_simulated_loop_skips_missing_files(recorder, tmp_path):
    files = [
        ("ghost", "ghost.csv", "rf_ghost.csv"),
        _sim_files(tmp_path, "iid"),
    ]
    pw.call_all_simulated_datasets(tmp_path, files, output_dir=tmp_path / "out")
    assert [c["name"] for c in recorder] == ["iid"]
    log = (tmp_path / "out" / "run.log").read_text()
    assert "WARNING" in log and "ghost skipped" in log


# --- run folder: config.json, no overwriting ------------------------------


def test_run_writes_its_settings_to_config_json(recorder, tmp_path):
    out = tmp_path / "out"
    settings = pw.RunSettings(burn_in=7, theta=2.0)
    pw.call_all_datasets(
        [_french_file(tmp_path / "10_A.csv")],
        risk_free_path=_ff_file(tmp_path / "ff.csv"),
        start_date="1980-01-03",
        output_dir=out,
        settings=settings,
        algorithms=["bayesian_averaging"],
    )
    config = json.loads((out / "config.json").read_text())
    assert config["SCALING_FACTOR"] == 1
    assert config["settings"]["burn_in"] == 7
    assert config["settings"]["theta"] == 2.0
    assert config["settings"]["necessity"] is False
    assert config["settings"]["start"] == settings.start
    (call,) = config["calls"]
    assert call["data"] == "real" and call["datasets"] == ["10_A"]
    assert call["start_date"] == "1980-01-03" and call["end_date"] is None
    assert call["algorithms"] == ["bayesian_averaging"]
    assert call["finished"] == ["10_A_value_weighted", "10_A_equal_weighted"]
    assert "end_time" in call and call["duration_min"] >= 0

    # simulated data into the same run folder, same settings: appended
    pw.call_all_simulated_datasets(
        tmp_path,
        [_sim_files(tmp_path, "iid")],
        output_dir=out,
        settings=settings,
    )
    config = json.loads((out / "config.json").read_text())
    assert [c["data"] for c in config["calls"]] == ["real", "simulated"]
    assert config["calls"][1]["algorithms"] == list(pw.ALGORITHMS)
    assert config["calls"][1]["finished"] == ["iid"]


def test_interrupted_run_shows_what_finished(monkeypatch, tmp_path):
    """A crash mid-run: config.json lists the datasets that finished, and the
    call has no end_time."""
    real = pw.calculate_all_functions_perdatasets

    def crash_on_equal(dataset, dataset_name, **kw):
        if dataset_name.endswith("equal_weighted"):
            raise RuntimeError("crash")
        return real(dataset, dataset_name, **kw)

    monkeypatch.setattr(
        pw, "calculate_all_functions_perdatasets", crash_on_equal
    )
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        pw.call_all_datasets(
            [_french_file(tmp_path / "10_A.csv")],
            risk_free_path=_ff_file(tmp_path / "ff.csv"),
            output_dir=out,
            algorithms=["equal_weight"],
        )
    (call,) = json.loads((out / "config.json").read_text())["calls"]
    assert call["finished"] == ["10_A_value_weighted"]
    assert "end_time" not in call


def test_settings_reach_the_averaging_methods():
    """keep_newest, the Bayesian prune threshold and n_jobs are run settings
    (recorded in config.json), passed on to the algorithms."""
    s = pw.RunSettings(
        keep_newest=False, bayesian_prune_threshold=0.5, n_jobs=2
    )
    bayes = pw.ALGORITHMS["bayesian_averaging"][1](s)
    poss = pw.ALGORITHMS["possibilistic"][1](s)
    assert bayes["keep_newest"] is False and bayes["prune_threshold"] == 0.5
    assert poss["keep_newest"] is False and poss["n_jobs"] == 2


def test_run_folder_with_other_settings_is_refused(recorder, tmp_path):
    out = tmp_path / "out"
    files = [_sim_files(tmp_path, "iid")]
    pw.call_all_simulated_datasets(tmp_path, files, output_dir=out)
    with pytest.raises(ValueError, match="other settings"):
        pw.call_all_simulated_datasets(
            tmp_path, files, output_dir=out, settings=pw.RunSettings(theta=3.0)
        )
    assert len(recorder) == 1  # the second run never started


def test_existing_dataset_folder_is_never_overwritten(tmp_path):
    data = pd.DataFrame(
        np.random.default_rng(0).normal(0.0005, 0.01, (90, 2)),
        columns=["P1", "P2"],
    )
    run: dict[str, Any] = dict(
        settings=SMALL, algorithms=["equal_weight"], output_dir=tmp_path
    )
    folder = pw.calculate_all_functions_perdatasets(data, "ds", **run)
    before = (folder / "equal_weight_weights.csv").read_text()
    with pytest.raises(FileExistsError, match="never overwritten"):
        pw.calculate_all_functions_perdatasets(data * 2, "ds", **run)
    assert (folder / "equal_weight_weights.csv").read_text() == before
    pd.testing.assert_frame_equal(
        pd.read_csv(folder / "returns.csv", index_col=0), data
    )


# --- per-dataset run: which files are written -----------------------------


def test_save_run_applies_every_portfolio_rule(tmp_path):
    """A forecast is saved with its index, plus the weights of every rule in
    PORTFOLIO_RULES with the run's theta; a weight-only rule's weights are
    saved as they are."""
    dates = pd.bdate_range("2000-01-03", periods=4, name="Date")
    returns = pd.DataFrame(
        {"a": [0.01, 0.02, -0.01, 0.0], "b": [0.0, 0.01, 0.02, -0.01]},
        index=dates,
    )
    forecast = make_predictive(
        np.full((4, 2), 0.01), np.tile(0.01 * np.eye(2), (4, 1, 1)), dates
    )
    run = pw.AlgorithmRun(
        forecasts={"fc": forecast},
        weights={"rule": pd.DataFrame(0.5, index=dates, columns=["a", "b"])},
    )
    settings = pw.RunSettings(burn_in=0, periods_until_investment=1, theta=2.0)
    pw._save_run(tmp_path, run, returns, settings)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "fc_longonly_weights.csv",
        "fc_predictive.npz",
        "fc_weights.csv",
        "rule_weights.csv",
    ]
    back = load_predictive(tmp_path, "fc")
    assert back is not None and back["index"].equals(dates)
    unconstrained = pd.read_csv(tmp_path / "fc_weights.csv", index_col=0)
    # (1/theta) Sigma^{-1} mu = (1/2) * 1 per asset, from start to T-2
    assert np.allclose(unconstrained.iloc[1:3], 0.5)
    assert unconstrained.iloc[[0, 3]].isna().all().all()


# Small settings so the per-dataset run takes a second
SMALL = pw.RunSettings(
    burn_in=30,
    periods_until_investment=10,
    possibilistic_max_models=8,
    rolling_window_long=20,
    rolling_window_short=10,
)

PREDICTIVE = [
    "historical_expanding",
    "rolling_window_long",
    "rolling_window_short",
    "jorion_bayes_stein",
    "kan_zhou_three_fund",
    "bayesian_averaging",
    "possibilistic_plain",
]
NECESSITY = ["possibilistic_masked", "possibilistic_power", "possibilistic_exp"]


def _expected_files(strategies: list[str]) -> set[str]:
    """Weights, forecast and long-only weights of predictive strategies."""
    return {
        f"{a}_{s}"
        for a in strategies
        for s in ["weights.csv", "predictive.npz", "longonly_weights.csv"]
    }


@pytest.mark.parametrize("with_market", [True, False])
def test_per_dataset_run_writes_expected_files(tmp_path, with_market):
    rng = np.random.default_rng(0)
    cols = ["P1", "P2", "Mkt-RF"] if with_market else ["P1", "P2", "P3"]
    data = pd.DataFrame(rng.normal(0.0005, 0.01, (90, 3)), columns=cols)
    folder = pw.calculate_all_functions_perdatasets(
        data,
        "ds",
        settings=SMALL,
        output_dir=tmp_path,
    )
    expected = {"returns.csv", "possibilistic_diagnostics.csv"}
    expected |= {"equal_weight_weights.csv", "minimum_variance_weights.csv"}
    expected |= _expected_files(PREDICTIVE)
    if with_market:
        expected.add("market_weight_weights.csv")
    assert {p.name for p in folder.iterdir()} == expected
    pd.testing.assert_frame_equal(
        pd.read_csv(folder / "returns.csv", index_col=0), data
    )


def test_necessity_adds_the_three_schemes(monkeypatch, tmp_path):
    """Off by default; switched on, the same run also saves masked / power /
    exp next to plain (scores replaced here: the real ones are slow)."""
    monkeypatch.setattr(
        possibilistic_bayesian,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.arange(1, len(mus) + 1) / len(mus),
    )
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(0.0005, 0.01, (90, 3)), columns=["P1", "P2", "P3"]
    )
    run: dict[str, Any] = dict(
        algorithms=["possibilistic"], output_dir=tmp_path
    )
    off = pw.calculate_all_functions_perdatasets(
        data, "off", settings=SMALL, **run
    )
    on = pw.calculate_all_functions_perdatasets(
        data, "on", settings=dataclasses.replace(SMALL, necessity=True), **run
    )
    base = {"returns.csv", "possibilistic_diagnostics.csv"}
    plain = _expected_files(["possibilistic_plain"])
    assert {p.name for p in off.iterdir()} == base | plain
    assert {p.name for p in on.iterdir()} == (
        base | plain | _expected_files(NECESSITY)
    )
    name = "possibilistic_plain_weights.csv"
    assert (off / name).read_text() == (on / name).read_text()


def test_dates_reach_every_output(tmp_path):
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
        settings=SMALL,
        output_dir=tmp_path,
    )
    assert ev.load_returns(folder).index.equals(dates)
    for algo in ev.discover_strategies(folder):
        w, pred = ev.load_strategy_outputs(folder, algo)
        assert w.index.equals(dates), algo
        if pred is not None:
            assert pred["index"].equals(dates), algo
    diag = pd.read_csv(
        folder / "possibilistic_diagnostics.csv", index_col=0, parse_dates=True
    )
    assert diag.index.equals(dates[SMALL.burn_in :])  # from burn_in on


# --- algorithm registry and selection (P1) ---------------------------------


def test_registry_lists_every_algorithm_in_run_order():
    assert list(pw.ALGORITHMS) == [
        "equal_weight",
        "market_weight",
        "minimum_variance",
        "historical_expanding",
        "rolling_window_long",
        "rolling_window_short",
        "jorion_bayes_stein",
        "kan_zhou_three_fund",
        "bayesian_averaging",
        "possibilistic",
    ]


def test_selected_algorithms_write_only_their_files(tmp_path):
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(0.0005, 0.01, (90, 3)), columns=["P1", "P2", "Mkt-RF"]
    )
    # given out of order: still run in registry order
    folder = pw.calculate_all_functions_perdatasets(
        data,
        "ds",
        settings=SMALL,
        algorithms=["bayesian_averaging", "equal_weight"],
        output_dir=tmp_path,
    )
    assert {p.name for p in folder.iterdir()} == {
        "returns.csv",
        "equal_weight_weights.csv",
        "bayesian_averaging_weights.csv",
        "bayesian_averaging_predictive.npz",
        "bayesian_averaging_longonly_weights.csv",
    }


def test_unknown_algorithm_raises_before_writing(tmp_path):
    data = pd.DataFrame({"P1": [0.01, 0.02]})
    with pytest.raises(ValueError, match="available"):
        pw.calculate_all_functions_perdatasets(
            data, "ds", algorithms=["bayesian"], output_dir=tmp_path
        )
    assert not (tmp_path / "ds").exists()


# --- logging and failing algorithms ---------------------------------------


def _broken(*args, **kwargs):
    raise np.linalg.LinAlgError("broken on purpose")


def test_failing_algorithm_does_not_stop_the_others(monkeypatch, tmp_path):
    """The others run and are saved; then AlgorithmsFailed lists the failure."""
    monkeypatch.setitem(
        pw.ALGORITHMS, "minimum_variance", (_broken, lambda s: {})
    )
    data = pd.DataFrame(
        np.random.default_rng(0).normal(0.0005, 0.01, (90, 2)),
        columns=["P1", "P2"],
    )
    with pytest.raises(pw.AlgorithmsFailed) as info:
        pw.calculate_all_functions_perdatasets(
            data,
            "ds",
            settings=SMALL,
            algorithms=[
                "equal_weight",
                "minimum_variance",
                "rolling_window_long",
            ],
            output_dir=tmp_path,
        )
    assert list(info.value.failed) == ["minimum_variance"]
    assert (
        "LinAlgError: broken on purpose"
        in info.value.failed["minimum_variance"]
    )
    saved = {p.name for p in (tmp_path / "ds").iterdir()}
    assert "equal_weight_weights.csv" in saved
    assert "rolling_window_long_weights.csv" in saved  # ran after the failure
    assert not any(name.startswith("minimum_variance") for name in saved)


def test_run_continues_after_a_failure_and_reports_it(monkeypatch, tmp_path):
    monkeypatch.setitem(
        pw.ALGORITHMS, "minimum_variance", (_broken, lambda s: {})
    )
    out = tmp_path / "out"
    files = [
        _sim_files(tmp_path, "iid", T=90),
        _sim_files(tmp_path, "mr", T=90),
    ]
    folders = pw.call_all_simulated_datasets(
        tmp_path,
        files,
        output_dir=out,
        settings=SMALL,
        algorithms=["equal_weight", "minimum_variance"],
    )
    assert [f.name for f in folders] == ["iid", "mr"]  # both datasets ran
    assert all((f / "rf.csv").exists() for f in folders)
    (call,) = json.loads((out / "config.json").read_text())["calls"]
    assert call["finished"] == []  # a dataset with a failure isn't finished
    assert set(call["failed"]) == {"iid", "mr"}
    assert "end_time" in call
    log = (out / "run.log").read_text()
    assert "minimum_variance FAILED" in log and "Traceback" in log
    assert "Run finished with 2 failed algorithm run(s)" in log


def test_run_log_has_timestamps_timings_and_warnings(tmp_path):
    """Timestamped entries, the time per algorithm, and warnings (here
    Markowitz on a singular covariance: a constant column) land in run.log."""
    out = tmp_path / "out"
    _sim_files(tmp_path, "iid", T=90)
    data = pd.read_csv(tmp_path / "iid.csv")
    data["Asset2"] = 0.0  # singular covariance -> NaN rows + a warning
    data.to_csv(tmp_path / "iid.csv", index=False)
    pw.call_all_simulated_datasets(
        tmp_path,
        [("iid", "iid.csv", "rf_iid.csv")],
        output_dir=out,
        settings=SMALL,
        algorithms=["rolling_window_long"],
    )
    lines = (out / "run.log").read_text().splitlines()
    entries = [ln for ln in lines if ln and not ln.startswith(" ")]
    timestamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    assert all(re.match(timestamp, ln) for ln in entries)
    assert any("[iid] rolling_window_long done in" in ln for ln in entries)
    assert any("RuntimeWarning: markowitz" in ln for ln in entries)


# --- resume ----------------------------------------------------------------


class _HardStop(BaseException):
    """Like Ctrl-C / a killed process: not caught as a failing algorithm."""


def _stop_on_second_call(real: Callable[..., Any]) -> Callable[..., Any]:
    calls = {"n": 0}

    def stopping(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _HardStop
        return real(*args, **kwargs)

    return stopping


def _mtimes(folder: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}


def test_resume_continues_a_stopped_run(monkeypatch, tmp_path):
    """Stopped in the 2nd dataset's 2nd algorithm; the resumed run skips the
    1st dataset and the 2nd dataset's 1st algorithm (files untouched) and
    runs the rest."""
    out = tmp_path / "out"
    files = [
        _sim_files(tmp_path, "iid", T=90),
        _sim_files(tmp_path, "mr", T=90),
    ]
    run: dict[str, Any] = dict(
        output_dir=out,
        settings=SMALL,
        algorithms=["equal_weight", "rolling_window_long"],
    )
    fn, takes = pw.ALGORITHMS["rolling_window_long"]
    monkeypatch.setitem(
        pw.ALGORITHMS, "rolling_window_long", (_stop_on_second_call(fn), takes)
    )
    with pytest.raises(_HardStop):
        pw.call_all_simulated_datasets(tmp_path, files, **run)
    (call,) = json.loads((out / "config.json").read_text())["calls"]
    assert call["done"] == {
        "iid": ["equal_weight", "rolling_window_long"],
        "mr": ["equal_weight"],
    }
    assert "end_time" not in call
    before_iid, before_mr = _mtimes(out / "iid"), _mtimes(out / "mr")

    monkeypatch.setitem(pw.ALGORITHMS, "rolling_window_long", (fn, takes))
    with pytest.raises(FileExistsError):  # not without resume
        pw.call_all_simulated_datasets(tmp_path, files, **run)
    pw.call_all_simulated_datasets(tmp_path, files, **run, resume=True)

    assert _mtimes(out / "iid") == before_iid  # skipped entirely
    after_mr = _mtimes(out / "mr")
    for name in ("returns.csv", "equal_weight_weights.csv"):
        assert after_mr[name] == before_mr[name]  # done: not touched
    assert "rolling_window_long_weights.csv" in after_mr
    assert "rf.csv" in after_mr
    last = json.loads((out / "config.json").read_text())["calls"][-1]
    assert last["finished"] == ["mr"] and "end_time" in last
    assert last["done"] == {"mr": ["rolling_window_long"]}
    log = (out / "run.log").read_text()
    assert "[iid] already done, skipped (resume)" in log
    assert "[mr] resuming; already done: equal_weight" in log


def test_resume_refuses_a_folder_with_other_data(tmp_path):
    out = tmp_path / "out"
    data = pd.DataFrame(
        np.random.default_rng(0).normal(0.0005, 0.01, (90, 2)),
        columns=["P1", "P2"],
    )
    run: dict[str, Any] = dict(
        settings=SMALL, algorithms=["equal_weight"], output_dir=out
    )
    pw.calculate_all_functions_perdatasets(data, "ds", **run)
    with pytest.raises(ValueError, match="can't resume"):
        pw.calculate_all_functions_perdatasets(
            data.iloc[:-5], "ds", **run, resume_done=set()
        )


# --- command line ------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "merge_threshold"),
    [([], None), (["--merge-threshold", "0.15"], 0.15)],
)
def test_cli_merge_threshold(monkeypatch, argv, merge_threshold):
    calls = []
    monkeypatch.setattr(pw, "main", lambda *a, **k: calls.append(k))
    monkeypatch.setattr("sys.argv", ["pipeline_wrapper", *argv])
    pw._cli()
    assert calls[0]["settings"].merge_threshold == merge_threshold
