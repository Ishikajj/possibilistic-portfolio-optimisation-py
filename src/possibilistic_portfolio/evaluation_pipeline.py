"""Evaluation pipeline: load saved strategy outputs and compute performance metrics.

Each dataset folder produced by the pipeline is self-contained:
  returns.csv                  ← prepared returns (saved by calculate_all_functions_perdatasets)
  <algo>_weights.csv           ← Markowitz weights, same index as returns.csv
  <algo>_predictive.npz        ← forecast + its index (predictive algos only; predictive.py)
                                 (runs before 2026-11-16: <algo>_mus.npy / _sigmas.npy)
  rf.csv                       ← risk-free rate (real runs since 2026-11-16, simulated runs)

Everything is matched by index label: dates for real data, row numbers for
simulated data. Weights or forecasts whose index differs from returns.csv raise.

After evaluation the same folder also receives:
  scalars.csv                  ← one row per algo, one col per scalar metric
  rolling_sharpe.csv           ← time-indexed, one col per algo
  log_likelihood.csv           ← time-indexed, predictive algos only
  mahalanobis.csv              ← time-indexed, predictive algos only

All strategies are scored over one common window: from the latest first
weight row in the folder (see ``evaluation_start``).

Entry point: ``run_evaluation(folder, rf_series=load_rf(folder))``.

Command line: evaluate every dataset folder of a run (default
``config.RESULTS_DIR``; another run folder as argument, relative to dev/;
``--help`` for the usage)::

    uv run python -m possibilistic_portfolio.evaluation_pipeline [run_folder]
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from possibilistic_portfolio.config import (
    RESULTS_DIR,
    RF_PATH,
    ROOT,
    SCALING_FACTOR,
)
from possibilistic_portfolio.data_input import load_risk_free
from possibilistic_portfolio.distribution_evaluation_functions import (
    mahalanobis_distance_series,
    multivariate_log_likelihood_series,
)
from possibilistic_portfolio.portfolio_evaluation_functions import (
    average_turnover,
    calculate_sharpe_ratio,
    certainty_equivalent,
    rolling_sharpe_ratio,
)
from possibilistic_portfolio.predictive import load_predictive

_ROLLING_WINDOW = 252


# ── 1. I/O helpers ─────────────────


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    """Read a CSV saved with its index; a ``Date`` index is parsed to dates."""
    df = pd.read_csv(path, index_col=0)
    if df.index.name == "Date":
        df.index = pd.to_datetime(df.index)
    return df


def load_returns(folder: Path | str) -> pd.DataFrame:
    """Load ``returns.csv`` from a run folder.

    Indexed by Date for real data, by row number for simulated data (and for
    real runs before 2026-11-16).
    """
    path = Path(folder) / "returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"returns.csv not found in {folder}. "
            "Was calculate_all_functions_perdatasets run on this folder?"
        )
    return _read_indexed_csv(path)


def load_rf(
    folder: Path | str, legacy_start_date: str = "1980-01-01"
) -> pd.Series:
    """Risk-free series for a run folder, matched to returns.csv by index.

    - ``rf.csv`` with a Date column → indexed by Date.
    - ``rf.csv`` without Date (simulated runs since 2026-04-27) → row numbers,
      like their returns.csv.
    - no ``rf.csv`` (real runs before 2026-11-16, row-number index) → the
      global French RF from ``legacy_start_date``, by row number as before.
    """
    rf_csv = Path(folder) / "rf.csv"
    if not rf_csv.exists():
        return load_risk_free(RF_PATH, legacy_start_date)["RF"]
    rf = pd.read_csv(rf_csv)
    if "Date" in rf.columns:
        rf = rf.set_index(pd.to_datetime(rf["Date"]))
    return rf["RF"]


def load_strategy_outputs(
    folder: Path | str,
    algo_name: str,
    index: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """Load saved weights and, if available, the forecast for one algorithm.

    Returns ``(weights_df, forecast)``; ``forecast`` is the dict from
    ``predictive.load_predictive``, or ``None`` for weight-only algorithms.
    With ``index`` given (the returns' index), weights with a different
    index raise.
    """
    return _load_weights(folder, algo_name, index), load_predictive(
        folder, algo_name
    )


def _load_weights(
    folder: Path | str, algo_name: str, index: pd.Index | None = None
) -> pd.DataFrame:
    weights_path = Path(folder) / f"{algo_name}_weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    weights_df = _read_indexed_csv(weights_path)
    if index is not None and not weights_df.index.equals(index):
        raise ValueError(
            f"{algo_name}_weights.csv index does not match returns.csv"
        )
    return weights_df


def discover_strategies(folder: Path | str) -> list[str]:
    """Sorted algorithm names in ``folder`` (``*_weights.csv``, suffix stripped)."""
    return sorted(
        p.name.replace("_weights.csv", "")
        for p in Path(folder).glob("*_weights.csv")
    )


# ── 2. Metrics ────────────────────────────────────────────────────────────────


@dataclass
class StrategyData:
    """Everything a metric needs for one strategy.

    The log-likelihood and Mahalanobis series are computed on first use and
    cached, so the series metrics and their averages share one computation.
    Both are ``None`` without a forecast.
    """

    returns: pd.DataFrame
    weights: pd.DataFrame
    forecast: dict | None = None
    rf: pd.Series | None = None
    theta: float = 1.0
    rolling_window: int = _ROLLING_WINDOW

    @cached_property
    def log_likelihood(self) -> pd.Series | None:
        if self.forecast is None:
            return None
        return multivariate_log_likelihood_series(self.returns, self.forecast)

    @cached_property
    def mahalanobis(self) -> pd.Series | None:
        if self.forecast is None:
            return None
        return mahalanobis_distance_series(self.returns, self.forecast)


def _mean(series: pd.Series | None) -> float:
    return float("nan") if series is None else float(series.dropna().mean())


# Each metric: name -> function of StrategyData. A missing prerequisite gives
# NaN (scalars) or None (series: no column for that strategy). To add a
# metric, add an entry; scalars.csv columns follow this order.
SCALAR_METRICS: dict[str, Callable[[StrategyData], float]] = {
    "sharpe": lambda d: calculate_sharpe_ratio(
        d.returns, d.weights, scaling_factor=SCALING_FACTOR
    ),
    "ceq": lambda d: (
        float("nan")
        if d.rf is None
        else certainty_equivalent(d.returns, d.weights, d.rf, theta=d.theta)
    ),
    "avg_log_likelihood": lambda d: _mean(d.log_likelihood),
    "avg_mahalanobis": lambda d: _mean(d.mahalanobis),  # unsquared
    "avg_turnover": lambda d: average_turnover(
        d.weights, scaling_factor=SCALING_FACTOR
    ),
}

# One <name>.csv each, one column per strategy that has the series.
SERIES_METRICS: dict[str, Callable[[StrategyData], pd.Series | None]] = {
    "rolling_sharpe": lambda d: rolling_sharpe_ratio(
        d.returns,
        d.weights,
        window=d.rolling_window,
        scaling_factor=SCALING_FACTOR,
    ),
    "log_likelihood": lambda d: d.log_likelihood,
    "mahalanobis": lambda d: d.mahalanobis,
}


def _select(
    requested: list[str] | None, available: list[str], what: str
) -> list[str]:
    """``requested`` filtered from ``available`` (keeps ``available`` order);
    ``None`` = all. Unknown names raise, listing what is available."""
    if requested is None:
        return list(available)
    unknown = [r for r in requested if r not in available]
    if unknown:
        raise ValueError(f"Unknown {what} {unknown}; available: {available}")
    return [a for a in available if a in requested]


def compute_metrics(
    data: StrategyData, metrics: list[str] | None = None
) -> dict:
    """``{"scalars": {...}, "series": {...}}`` for one strategy.

    ``metrics``: names from SCALAR_METRICS / SERIES_METRICS (``None`` = all).
    """
    names = _select(metrics, [*SCALAR_METRICS, *SERIES_METRICS], "metric")
    series = {n: SERIES_METRICS[n](data) for n in names if n in SERIES_METRICS}
    return {
        "scalars": {
            n: SCALAR_METRICS[n](data) for n in names if n in SCALAR_METRICS
        },
        "series": {n: s for n, s in series.items() if s is not None},
    }


# ── 3. Common evaluation window ───────────────────────────────────────────────


def evaluation_start(weights: dict[str, pd.DataFrame]) -> int:
    """Row position where the common evaluation window starts.

    Every strategy is scored over the same days: from the latest first
    weight row among ``weights`` (normally ``start`` for all of them).
    Guards against one broken strategy silently cutting the window:
    - a strategy with no weights at all is left out, with a warning;
    - strategies starting later than the earliest are named in a warning.
    Raises if no strategy has weights.
    """
    first: dict[str, int] = {}
    for name, w in weights.items():
        rows = np.flatnonzero(w.notna().any(axis=1).to_numpy())
        if rows.size:
            first[name] = int(rows[0])
        else:
            warnings.warn(
                f"{name} has no weights; left out of the common evaluation "
                "window",
                RuntimeWarning,
                stacklevel=2,
            )
    if not first:
        raise ValueError("No strategy has any weights")
    earliest, start = min(first.values()), max(first.values())
    if start > earliest:
        index = next(iter(weights.values())).index
        late = ", ".join(
            f"{n} (from {index[r]})" for n, r in first.items() if r > earliest
        )
        warnings.warn(
            f"Common evaluation window starts at {index[start]} instead of "
            f"{index[earliest]}, because these strategies start later: {late}",
            RuntimeWarning,
            stacklevel=2,
        )
    return start


def _from_row(
    weights_df: pd.DataFrame, forecast: dict | None, start: int
) -> tuple[pd.DataFrame, dict | None]:
    """Copies with weights and forecast rows before ``start`` set to NaN,
    so no metric uses them (forecast row t scores the return at t+1)."""
    weights_df = weights_df.copy()
    weights_df.iloc[:start] = np.nan
    if forecast is not None:
        forecast = dict(forecast)
        for key in ("mu_hat", "sigma_hat"):
            forecast[key] = forecast[key].copy()
            forecast[key][:start] = np.nan
    return weights_df, forecast


# ── 4. Whole-folder evaluation ────────────────────────────────────────────────


def evaluate_all_strategies(
    folder: Path | str,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
    rolling_window: int = _ROLLING_WINDOW,
    strategies: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate the strategies saved in ``folder`` against its returns.csv.

    ``strategies`` / ``metrics``: names to evaluate (``None`` = all); unknown
    names raise. A strategy that fails is printed as a warning and returns
    empty results; the others carry on.

    All strategies are scored over one common window (``evaluation_start``,
    computed from every strategy in the folder, not only the selected
    ones): weights and forecasts before it are ignored.

    Returns ``{algo_name: {"scalars": {...}, "series": {...}}}``.
    """
    folder = Path(folder)
    returns_df = load_returns(folder)
    available = discover_strategies(folder)
    if not available:
        raise ValueError(f"No *_weights.csv files found in {folder}")
    selected = _select(strategies, available, "strategy")
    _select(metrics, [*SCALAR_METRICS, *SERIES_METRICS], "metric")  # fail early

    weights: dict[str, pd.DataFrame] = {}
    load_errors: dict[str, Exception] = {}
    for algo_name in available:
        try:
            weights[algo_name] = _load_weights(
                folder, algo_name, returns_df.index
            )
        except Exception as exc:
            load_errors[algo_name] = exc
            warnings.warn(
                f"{algo_name} could not be loaded ({exc}); left out of the "
                "common evaluation window",
                RuntimeWarning,
                stacklevel=2,
            )
    start = evaluation_start(weights)
    print(f"  Common evaluation window from {returns_df.index[start]}")

    results: dict[str, dict] = {}
    for algo_name in selected:
        print(f"  Evaluating {algo_name}...")
        # Catch warnings from the metric functions and re-raise them tagged
        # with the strategy name (also stops Python showing identical
        # messages from different strategies only once).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if algo_name in load_errors:
                    raise load_errors[algo_name]
                weights_df, forecast = _from_row(
                    weights[algo_name],
                    load_predictive(folder, algo_name),
                    start,
                )
                data = StrategyData(
                    returns_df,
                    weights_df,
                    forecast,
                    rf_series,
                    theta,
                    rolling_window,
                )
                results[algo_name] = compute_metrics(data, metrics)
            except Exception as exc:
                print(f"  [WARNING] {algo_name} failed: {exc}")
                results[algo_name] = {"scalars": {}, "series": {}}
        for w in caught:
            warnings.warn(
                f"[{algo_name}] {w.message}", w.category, stacklevel=2
            )

    return results


# ── 5. Saving ─────────────────────────────────────────────────────────────────


def save_evaluation_results(
    all_results: dict[str, dict],
    folder: Path | str,
) -> None:
    """Write ``scalars.csv`` (one row per strategy) and one ``<name>.csv``
    per series metric (one column per strategy that has it) into ``folder``.
    A strategy that failed (empty results) gets an all-NaN row in
    ``scalars.csv``, so it shows up as missing, and no series columns.
    """
    folder = Path(folder)
    scalars = {a: r["scalars"] for a, r in all_results.items()}
    pd.DataFrame(scalars).T.reindex(list(scalars)).to_csv(
        folder / "scalars.csv"
    )

    names = dict.fromkeys(n for r in all_results.values() for n in r["series"])
    for name in names:
        cols = {
            a: r["series"][name]
            for a, r in all_results.items()
            if name in r["series"]
        }
        pd.DataFrame(cols).to_csv(folder / f"{name}.csv")


# ── 6. End-to-end entry point ─────────────────────────────────────────────────


def run_evaluation(
    folder: Path | str,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
    rolling_window: int = _ROLLING_WINDOW,
    strategies: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate the strategies in a run folder and save the CSVs back into it.

    Parameters
    ----------
    folder : run folder made by ``calculate_all_functions_perdatasets``.
    rf_series : daily risk-free rate, matched to returns.csv by index
        (``load_rf(folder)``). Needed for CEQ; ``None`` → CEQ is NaN.
    theta : risk aversion for CEQ.
    rolling_window : rolling Sharpe window in days (default 252).
    strategies, metrics : names to evaluate (``None`` = all); see
        ``discover_strategies`` and SCALAR_METRICS / SERIES_METRICS.

    Returns the results dict of :func:`evaluate_all_strategies`.
    """
    folder = Path(folder)
    print(f"\nEvaluating: {folder.name}")
    results = evaluate_all_strategies(
        folder, rf_series, theta, rolling_window, strategies, metrics
    )
    save_evaluation_results(results, folder)
    print(f"  Results saved to: {folder}")
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every dataset folder of a run folder."
    )
    parser.add_argument(
        "run_folder",
        nargs="?",
        type=Path,
        default=RESULTS_DIR,
        help=f"run folder (default {RESULTS_DIR.name}/); relative to dev/",
    )
    run_folder = ROOT / parser.parse_args().run_folder  # absolute stays
    if not run_folder.is_dir():
        parser.error(f"{run_folder} is not a folder")
    for folder in sorted(run_folder.iterdir()):
        if folder.is_dir():
            run_evaluation(folder, rf_series=load_rf(folder))


if __name__ == "__main__":
    _cli()
