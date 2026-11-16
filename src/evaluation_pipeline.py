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

Module layout
-------------
load_returns               -- read returns.csv from a dataset folder
load_rf                    -- read the risk-free series for a dataset folder
load_strategy_outputs      -- read weights + optional forecast for one algo
discover_strategies        -- list every algo saved in a folder
compute_scalar_metrics     -- sharpe, CEQ, avg log-likelihood, avg mahalanobis
compute_series_metrics     -- rolling sharpe, LL series, mahalanobis series
evaluate_strategy          -- load + compute all metrics for one algo
evaluate_all_strategies    -- run evaluate_strategy over every algo in a folder
scalars_to_dataframe       -- tidy scalar results into a DataFrame
save_evaluation_results    -- write all CSVs back into the dataset folder
run_evaluation             -- end-to-end: load → evaluate → save
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RF_PATH, SCALING_FACTOR
from data_input import load_risk_free
from distribution_evaluation_functions import (
    mahalanobis_distance_series,
    multivariate_log_likelihood_series,
)
from portfolio_evaluation_functions import (
    average_turnover,
    calculate_sharpe_ratio,
    certainty_equivalent,
    rolling_sharpe_ratio,
)
from predictive import load_predictive

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
) -> tuple[pd.DataFrame, dict | None]:
    """Load saved weights and, if available, predictive moments for one algorithm.

    Returns
    -------
    weights_df : pd.DataFrame
    mu_sigma_dict : dict or None
        Forecast dict (``predictive.load_predictive``), or ``None`` for
        weight-only algorithms.
    """
    folder = Path(folder)

    weights_path = folder / f"{algo_name}_weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return _read_indexed_csv(weights_path), load_predictive(folder, algo_name)


def discover_strategies(folder: Path | str) -> list[str]:
    """Return a sorted list of algorithm names saved in ``folder``.

    Scans for ``*_weights.csv`` files and strips the suffix.
    """
    return sorted(
        p.name.replace("_weights.csv", "")
        for p in Path(folder).glob("*_weights.csv")
    )


# ── 2. Metric computation ─────────────────────────────────────────────────────


def compute_scalar_metrics(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    mu_sigma_dict: dict | None = None,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
) -> dict:
    """Compute all scalar performance metrics for one strategy.

    Always computed
    ---------------
    sharpe            Sharpe ratio, scaled by config.SCALING_FACTOR (1 = daily)

    Requires ``rf_series``
    ----------------------
    ceq               certainty equivalent (mean-variance utility)

    Requires ``mu_sigma_dict``
    --------------------------
    avg_log_likelihood    mean multivariate Gaussian log-likelihood
    avg_mahalanobis       mean Mahalanobis distance (unsquared)

    Missing prerequisites produce NaN rather than raising.
    """
    results: dict = {}

    results["sharpe"] = calculate_sharpe_ratio(
        returns_df, weights_df, scaling_factor=SCALING_FACTOR
    )

    if rf_series is not None:
        results["ceq"] = certainty_equivalent(
            returns_df, weights_df, rf_series, theta=theta
        )
    else:
        results["ceq"] = float("nan")

    if mu_sigma_dict is not None:
        ll = multivariate_log_likelihood_series(
            returns_df, mu_sigma_dict
        ).dropna()
        results["avg_log_likelihood"] = (
            float(ll.mean()) if not ll.empty else float("nan")
        )
        mah = mahalanobis_distance_series(returns_df, mu_sigma_dict).dropna()
        results["avg_mahalanobis"] = (
            float(mah.mean()) if not mah.empty else float("nan")
        )
    else:
        results["avg_log_likelihood"] = float("nan")
        results["avg_mahalanobis"] = float("nan")

    results["avg_turnover"] = average_turnover(
        weights_df, scaling_factor=SCALING_FACTOR
    )

    return results


def compute_series_metrics(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    mu_sigma_dict: dict | None = None,
    rolling_window: int = _ROLLING_WINDOW,
) -> dict:
    """Compute all time-series performance metrics for one strategy.

    Always computed
    ---------------
    rolling_sharpe            rolling Sharpe over ``rolling_window`` days, scaled by config.SCALING_FACTOR

    Requires ``mu_sigma_dict``
    --------------------------
    log_likelihood_series     period-by-period log-likelihood (or None)
    mahalanobis_series        period-by-period Mahalanobis distance (or None)
    """
    results: dict = {}

    results["rolling_sharpe"] = rolling_sharpe_ratio(
        returns_df,
        weights_df,
        window=rolling_window,
        scaling_factor=SCALING_FACTOR,
    )

    if mu_sigma_dict is not None:
        results["log_likelihood_series"] = multivariate_log_likelihood_series(
            returns_df, mu_sigma_dict
        )
        results["mahalanobis_series"] = mahalanobis_distance_series(
            returns_df, mu_sigma_dict
        )
    else:
        results["log_likelihood_series"] = None
        results["mahalanobis_series"] = None

    return results


# ── 3. Per-strategy and whole-folder evaluation ───────────────────────────────


def evaluate_strategy(
    returns_df: pd.DataFrame,
    folder: Path | str,
    algo_name: str,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
    rolling_window: int = _ROLLING_WINDOW,
) -> dict:
    """Load and fully evaluate one saved strategy.

    Returns
    -------
    dict with keys ``"scalars"`` and ``"series"``.
    """
    weights_df, mu_sigma_dict = load_strategy_outputs(folder, algo_name)
    if not weights_df.index.equals(returns_df.index):
        raise ValueError(
            f"{algo_name}_weights.csv index does not match returns.csv"
        )
    scalars = compute_scalar_metrics(
        returns_df, weights_df, mu_sigma_dict, rf_series, theta
    )
    series = compute_series_metrics(
        returns_df, weights_df, mu_sigma_dict, rolling_window
    )
    return {"scalars": scalars, "series": series}


def evaluate_all_strategies(
    folder: Path | str,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
    rolling_window: int = _ROLLING_WINDOW,
) -> dict[str, dict]:
    """Load returns and evaluate every strategy saved in ``folder``.

    Loads ``returns.csv`` automatically from ``folder``, then calls
    :func:`evaluate_strategy` for every algo discovered there.

    Returns
    -------
    results : dict[str, dict]
        ``{algo_name: {"scalars": {...}, "series": {...}}}``
    """
    folder = Path(folder)
    returns_df = load_returns(folder)
    strategies = discover_strategies(folder)

    if not strategies:
        raise ValueError(f"No *_weights.csv files found in {folder}")

    results: dict[str, dict] = {}
    for algo_name in strategies:
        print(f"  Evaluating {algo_name}...")
        # Catch warnings from the metric functions and re-raise them tagged
        # with the strategy name (also stops Python showing identical
        # messages from different strategies only once).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                results[algo_name] = evaluate_strategy(
                    returns_df,
                    folder,
                    algo_name,
                    rf_series,
                    theta,
                    rolling_window,
                )
            except Exception as exc:
                print(f"  [WARNING] {algo_name} failed: {exc}")
                results[algo_name] = {"scalars": {}, "series": {}}
        for w in caught:
            warnings.warn(
                f"[{algo_name}] {w.message}", w.category, stacklevel=2
            )

    return results


# ── 4. Results formatting and persistence ─────────────────────────────────────


def scalars_to_dataframe(all_results: dict[str, dict]) -> pd.DataFrame:
    """Convert scalar metrics to a tidy DataFrame.

    Rows = strategies, columns = metric names.
    """
    rows = {
        algo: result["scalars"]
        for algo, result in all_results.items()
        if result.get("scalars")
    }
    return pd.DataFrame(rows).T


def save_evaluation_results(
    all_results: dict[str, dict],
    folder: Path | str,
) -> None:  # writes CSVs, no return value
    """Write all evaluation outputs back into the dataset folder.

    Files written
    -------------
    scalars.csv           one row per strategy, one col per scalar metric
    rolling_sharpe.csv    time-indexed, one col per strategy
    log_likelihood.csv    time-indexed, predictive strategies only
    mahalanobis.csv       time-indexed, predictive strategies only
    """
    folder = Path(folder)

    scalars_to_dataframe(all_results).to_csv(folder / "scalars.csv")

    rolling = {
        algo: res["series"]["rolling_sharpe"]
        for algo, res in all_results.items()
        if res.get("series") and res["series"].get("rolling_sharpe") is not None
    }
    if rolling:
        pd.DataFrame(rolling).to_csv(folder / "rolling_sharpe.csv")

    ll = {
        algo: res["series"]["log_likelihood_series"]
        for algo, res in all_results.items()
        if res.get("series")
        and res["series"].get("log_likelihood_series") is not None
    }
    if ll:
        pd.DataFrame(ll).to_csv(folder / "log_likelihood.csv")

    mah = {
        algo: res["series"]["mahalanobis_series"]
        for algo, res in all_results.items()
        if res.get("series")
        and res["series"].get("mahalanobis_series") is not None
    }
    if mah:
        pd.DataFrame(mah).to_csv(folder / "mahalanobis.csv")


# ── 5. End-to-end entry point ─────────────────────────────────────────────────


def run_evaluation(
    folder: Path | str,
    rf_series: pd.Series | None = None,
    theta: float = 1.0,
    rolling_window: int = _ROLLING_WINDOW,
) -> dict[str, dict]:
    """Load, evaluate, and save results for every strategy in ``folder``.

    This is the single call needed after ``calculate_all_functions_perdatasets``
    has been run.  It reads ``returns.csv`` from the folder, evaluates all
    saved strategies, saves the four output CSVs back into the same folder,
    and returns the full results dict.

    Parameters
    ----------
    folder : Path or str
        Self-contained dataset folder produced by the pipeline.
    rf_series : pd.Series, optional
        Daily risk-free rate aligned to the returns index.  Required for CEQ;
        pass ``None`` to skip CEQ (it will appear as NaN in the output).
    theta : float
        Risk-aversion parameter for CEQ.
    rolling_window : int
        Look-back window for rolling Sharpe (default 150 days).

    Returns
    -------
    results : dict[str, dict]
        Full results as returned by :func:`evaluate_all_strategies`.
    """
    folder = Path(folder)
    print(f"\nEvaluating: {folder.name}")

    results = evaluate_all_strategies(folder, rf_series, theta, rolling_window)
    save_evaluation_results(results, folder)

    print(f"  Results saved to: {folder}")
    return results


if __name__ == "__main__":
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
    for folder in sorted(RESULTS_DIR.iterdir()):
        if folder.is_dir():
            run_evaluation(folder, rf_series=load_rf(folder))
