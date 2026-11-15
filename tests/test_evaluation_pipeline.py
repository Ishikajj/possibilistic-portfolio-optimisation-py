"""Tests for evaluation_pipeline.py — evaluating a saved run folder.

Metric maths is tested in test_portfolio_/test_distribution_evaluation_functions;
here we test discovery, loading, wiring and the files written back.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluation_pipeline as ev
from distribution_evaluation_functions import multivariate_log_likelihood_series
from portfolio_evaluation_functions import (
    calculate_sharpe_ratio,
    certainty_equivalent,
)

T, N = 80, 2


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    """A minimal pipeline output folder: one weights-only and one predictive strategy."""
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.001, 0.01, (T, N)), columns=["a", "b"])
    returns.to_csv(tmp_path / "returns.csv")

    w = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    w.iloc[10:-1] = 0.5
    w.to_csv(tmp_path / "equal_weight_weights.csv")

    mu = np.full((T, N), np.nan)
    sigma = np.full((T, N, N), np.nan)
    mu[10:] = 0.001
    sigma[10:] = 1e-4 * np.eye(N)
    w2 = w.copy()
    w2.iloc[10:-1] = [0.8, 0.2]
    w2.to_csv(tmp_path / "model_weights.csv")
    np.save(tmp_path / "model_mus.npy", mu)
    np.save(tmp_path / "model_sigmas.npy", sigma)
    w2.clip(lower=0).to_csv(tmp_path / "model_longonly_weights.csv")
    return tmp_path


@pytest.fixture
def rf() -> pd.Series:
    return pd.Series(1e-4, index=range(T))


def test_discover_strategies_includes_long_only(run_folder):
    assert ev.discover_strategies(run_folder) == [
        "equal_weight",
        "model",
        "model_longonly",
    ]


def test_load_strategy_outputs(run_folder):
    w, pred = ev.load_strategy_outputs(run_folder, "model")
    assert w.shape == (T, N) and pred["sigma_hat"].shape == (T, N, N)
    _, none = ev.load_strategy_outputs(run_folder, "equal_weight")
    assert none is None
    with pytest.raises(FileNotFoundError):
        ev.load_strategy_outputs(run_folder, "ghost")


def test_load_returns_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.load_returns(tmp_path)


def test_scalar_metrics_wire_up_the_metric_functions(run_folder, rf):
    returns = ev.load_returns(run_folder)
    w, pred = ev.load_strategy_outputs(run_folder, "model")
    out = ev.compute_scalar_metrics(returns, w, pred, rf, theta=2.0)
    assert np.isclose(
        out["sharpe"], calculate_sharpe_ratio(returns, w, ev.SCALING_FACTOR)
    )  # daily
    assert np.isclose(
        out["ceq"], certainty_equivalent(returns, w, rf, theta=2.0)
    )
    assert np.isclose(
        out["avg_log_likelihood"],
        multivariate_log_likelihood_series(returns, pred).mean(),
    )
    assert np.isfinite(out["avg_mahalanobis"]) and np.isfinite(
        out["avg_turnover"]
    )


def test_scalar_metrics_without_rf_or_predictives_are_nan(run_folder):
    returns = ev.load_returns(run_folder)
    w, _ = ev.load_strategy_outputs(run_folder, "equal_weight")
    out = ev.compute_scalar_metrics(returns, w)
    assert np.isnan(out["ceq"])
    assert np.isnan(out["avg_log_likelihood"]) and np.isnan(
        out["avg_mahalanobis"]
    )
    assert np.isfinite(out["sharpe"])


def test_run_evaluation_writes_all_files(run_folder, rf):
    results = ev.run_evaluation(run_folder, rf_series=rf, rolling_window=20)
    assert set(results) == {"equal_weight", "model", "model_longonly"}

    scalars = pd.read_csv(run_folder / "scalars.csv", index_col=0)
    assert list(scalars.index) == ["equal_weight", "model", "model_longonly"]
    assert list(scalars.columns) == [
        "sharpe",
        "ceq",
        "avg_log_likelihood",
        "avg_mahalanobis",
        "avg_turnover",
    ]
    rolling = pd.read_csv(run_folder / "rolling_sharpe.csv", index_col=0)
    assert list(rolling.columns) == ["equal_weight", "model", "model_longonly"]
    # predictive series only for strategies with mus / sigmas
    for name in ["log_likelihood.csv", "mahalanobis.csv"]:
        assert list(pd.read_csv(run_folder / name, index_col=0).columns) == [
            "model"
        ]


def test_one_broken_strategy_does_not_stop_the_rest(run_folder, rf, capsys):
    (run_folder / "broken_weights.csv").write_text("not,a\nweights,file\n")
    results = ev.evaluate_all_strategies(run_folder, rf)
    assert results["broken"] == {"scalars": {}, "series": {}}
    assert results["model"]["scalars"]
    assert "WARNING" in capsys.readouterr().out
    scalars = ev.scalars_to_dataframe(results)
    assert "broken" not in scalars.index  # silently dropped from scalars.csv


def test_metric_warnings_are_tagged_with_strategy_name(run_folder):
    w = pd.DataFrame(np.nan, index=range(T), columns=["a", "b"])
    w.iloc[10] = 0.5  # one invested period -> Sharpe undefined
    for name in ["lonely_1", "lonely_2"]:
        w.to_csv(run_folder / f"{name}_weights.csv")
    with pytest.warns(RuntimeWarning) as rec:
        ev.evaluate_all_strategies(run_folder)
    msgs = [str(r.message) for r in rec]
    # same underlying warning, shown once per strategy
    assert any(m.startswith("[lonely_1] sharpe") for m in msgs)
    assert any(m.startswith("[lonely_2] sharpe") for m in msgs)


def test_empty_folder_raises(tmp_path):
    pd.DataFrame({"a": [0.0]}).to_csv(tmp_path / "returns.csv")
    with pytest.raises(ValueError):
        ev.evaluate_all_strategies(tmp_path)
