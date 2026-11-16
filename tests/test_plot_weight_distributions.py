"""Tests for plot_weight_distributions.py — pooled weight boxplots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from possibilistic_portfolio import plot_weight_distributions as pwd  # noqa: E402


def _write_weights(folder: Path, algo: str, values: list[float]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": values}).to_csv(folder / f"{algo}_weights.csv")


def test_collect_weights_pools_across_folders_and_drops_nan(tmp_path):
    _write_weights(tmp_path / "ds1", "bayesian_averaging", [np.nan, 0.5, 1.0])
    _write_weights(tmp_path / "ds2", "bayesian_averaging", [2.0, np.nan])
    _write_weights(
        tmp_path / "ds2", "equal_weight", [0.1]
    )  # not a plotted strategy
    (tmp_path / "notes.txt").write_text("ignored")
    pooled = pwd.collect_weights(tmp_path)
    assert set(pooled) == {s[0] for s in pwd.STRATEGIES}
    assert sorted(pooled["bayesian_averaging"]) == [0.5, 1.0, 2.0]
    assert len(pooled["possibilistic_exp"]) == 0


def test_boxplot_stats_quartiles_and_whiskers():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    s = pwd._boxplot_stats(vals)
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    assert (s["q1"], s["med"], s["q3"]) == (q1, med, q3)
    assert s["whislo"] == 1.0 and s["whishi"] == 4.0  # 100 is beyond 1.5 IQR
    assert list(s["fliers"]) == [100.0]
    assert np.isclose(s["mean"], vals.mean())


def test_plot_writes_png(monkeypatch, tmp_path):
    out = tmp_path / "w.png"
    monkeypatch.setattr(pwd, "OUTPUT_PATH", out)
    rng = np.random.default_rng(0)
    pooled = {name: rng.normal(size=200) for name, _, _ in pwd.STRATEGIES}
    pooled["possibilistic_exp"] = np.array([])  # skipped, not a crash
    pwd.plot(pooled)
    assert out.exists() and out.stat().st_size > 0
