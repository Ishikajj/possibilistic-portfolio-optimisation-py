"""Pipeline wrapper: runs the algorithms on one dataset and loops over
datasets.

Public entry points
-------------------
calculate_all_functions_perdatasets(dataset, dataset_name, *, settings, algorithms)
    Run the algorithms in ``ALGORITHMS`` (all, or the named ones) on one
    prepared returns DataFrame and save their outputs.

call_all_datasets(portfolios_paths, ...)
    Loop over Kenneth French CSV files, running two passes each (value- and
    equal-weighted blocks).

call_all_simulated_datasets(sim_data_dir, sim_files, ...)
    Loop over pre-generated simulated dataset CSVs (produced by
    simulated_datasets.generate_and_save_sim_datasets).

Run settings travel as one ``RunSettings`` object (defaults from config.py).
``ALGORITHMS`` lists each algorithm with the settings it takes; every
algorithm returns ``predictive.StrategyOutput`` (weights + optional forecast),
possibilistic one per weighting scheme. The pipeline saves them all the same
way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from possibilistic_portfolio import bayesian_averaging, possibilistic_bayesian
from possibilistic_portfolio.alternative_investment_rules import (
    equal_weight_strategy,
    historical_expectations_weights,
    jorion_bayes_stein_strategy,
    kan_zhou_three_fund_strategy,
    market_weight_strategy,
    minimum_variance_strategy,
    rolling_window_weights,
)
from possibilistic_portfolio.config import (
    BAYESIAN_MAX_MODELS,
    BURN_IN,
    DATASETS_DIR,
    FRENCH_PORTFOLIO_FILES,
    K_NEIGHBOURS,
    MERGE_THRESHOLD,
    NECESSITY_DEVICE,
    NECESSITY_ETA,
    NECESSITY_GAMMA,
    NU_BANDWIDTH,
    PERIODS_UNTIL_INVESTMENT,
    POSSIBILISTIC_MAX_MODELS,
    RESULTS_DIR,
    RF_PATH,
    ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_SHORT,
    SIM_DATA_DIR,
    THETA,
    USE_NECESSITY,
)
from possibilistic_portfolio.data_input import (
    load_excess_returns,
    prepare_returns,
)
from possibilistic_portfolio.markowitz import markowitz_long_only
from possibilistic_portfolio.predictive import StrategyOutput, save_predictive
from possibilistic_portfolio.simulated_datasets import DGPS, sim_paths

# ── Settings and output types ────────────────────────────────────────────────


@dataclass(frozen=True)
class RunSettings:
    """Settings of one pipeline run. Defaults come from config.py (the
    values of the FYP runs). Change one with e.g. ``RunSettings(theta=2.0)``.
    """

    burn_in: int = BURN_IN
    periods_until_investment: int = PERIODS_UNTIL_INVESTMENT
    theta: float = THETA
    rolling_window_long: int = ROLLING_WINDOW_LONG
    rolling_window_short: int = ROLLING_WINDOW_SHORT
    bayesian_max_models: int = BAYESIAN_MAX_MODELS
    possibilistic_max_models: int = POSSIBILISTIC_MAX_MODELS
    merge_threshold: float = MERGE_THRESHOLD
    nu_bandwidth: int = NU_BANDWIDTH
    k_neighbours: int = K_NEIGHBOURS
    necessity: bool = USE_NECESSITY  # also run masked / power / exp
    gamma: float = NECESSITY_GAMMA
    eta: float = NECESSITY_ETA
    device: str = NECESSITY_DEVICE

    @property
    def start(self) -> int:
        """First row with weights, for every strategy (``burn_in`` is also
        passed to the averaging methods, which use it to set their prior)."""
        return self.burn_in + self.periods_until_investment


@dataclass
class AlgorithmRun:
    """What one algorithm produces: its strategies (file prefix → output,
    saved as ``<prefix>_weights.csv`` / ``_predictive.npz`` /
    ``_longonly_weights.csv``) and extra tables (file stem → table, saved as
    ``<stem>.csv``)."""

    strategies: dict[str, StrategyOutput]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


# ── The algorithms ────────────────────────────────────────────────────────────

# Algorithm name -> (function, the settings it takes besides `start`), in run
# order. Every function is (returns_df, *, start, ...) -> StrategyOutput,
# except possibilistic (see _run_algorithm). Select with ``algorithms=[...]``;
# to add an algorithm, add an entry.
ALGORITHMS: dict[
    str, tuple[Callable[..., Any], Callable[[RunSettings], dict]]
] = {
    "equal_weight": (equal_weight_strategy, lambda s: {}),
    "market_weight": (market_weight_strategy, lambda s: {}),
    "minimum_variance": (
        minimum_variance_strategy,
        lambda s: {"window": s.rolling_window_long},
    ),
    "historical_expanding": (
        historical_expectations_weights,
        lambda s: {"theta": s.theta},
    ),
    "rolling_window_long": (
        rolling_window_weights,
        lambda s: {"window": s.rolling_window_long, "theta": s.theta},
    ),
    "rolling_window_short": (
        rolling_window_weights,
        lambda s: {"window": s.rolling_window_short, "theta": s.theta},
    ),
    "jorion_bayes_stein": (
        jorion_bayes_stein_strategy,
        lambda s: {"window": s.rolling_window_long, "theta": s.theta},
    ),
    "kan_zhou_three_fund": (
        kan_zhou_three_fund_strategy,
        lambda s: {"window": s.rolling_window_long, "theta": s.theta},
    ),
    "bayesian_averaging": (
        bayesian_averaging.run_core,
        lambda s: {
            "burn_in": s.burn_in,
            "theta": s.theta,
            "max_models": s.bayesian_max_models,
        },
    ),
    "possibilistic": (
        possibilistic_bayesian.run_core,
        lambda s: {
            "burn_in": s.burn_in,
            "theta": s.theta,
            "max_models": s.possibilistic_max_models,
            "merge_threshold": s.merge_threshold,
            "nu_bandwidth": s.nu_bandwidth,
            "k_neighbours": s.k_neighbours,
            "necessity": s.necessity,
            "gamma": s.gamma,
            "eta": s.eta,
            "device": s.device,
        },
    ),
}


def _run_algorithm(
    name: str, returns_df: pd.DataFrame, settings: RunSettings
) -> AlgorithmRun:
    """Run one algorithm from ALGORITHMS with its settings."""
    if name == "market_weight" and "Mkt-RF" not in returns_df.columns:
        return AlgorithmRun({})  # e.g. simulated data: no market column
    fn, settings_for = ALGORITHMS[name]
    result = fn(returns_df, start=settings.start, **settings_for(settings))
    if name == "possibilistic":
        # (diagnostics, {scheme: output}): one strategy per weighting scheme
        # ("plain"; with settings.necessity also masked / power / exp)
        diag_df, outputs = result
        return AlgorithmRun(
            {f"possibilistic_{scheme}": o for scheme, o in outputs.items()},
            tables={"possibilistic_diagnostics": diag_df},
        )
    return AlgorithmRun({name: result})


def _select_algorithms(algorithms: Sequence[str] | None) -> list[str]:
    """Names from ALGORITHMS in run order (``None`` = all). Unknown names
    raise, listing what is available."""
    if algorithms is None:
        return list(ALGORITHMS)
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unknown algorithms {unknown}; available: {list(ALGORITHMS)}"
        )
    return [a for a in ALGORITHMS if a in algorithms]


# ── Saving ────────────────────────────────────────────────────────────────────


def _save_algo_outputs(
    folder: Path,
    algo_name: str,
    weights: pd.DataFrame | None = None,
    mu_sigma_dict: dict | None = None,
) -> None:
    """Persist weights (CSV, indexed like the returns) and, where available,
    the forecast with its index (<algo_name>_predictive.npz, see predictive.py).

    File naming convention:
      <algo_name>_weights.csv
      <algo_name>_predictive.npz
    """
    if weights is not None:
        weights.to_csv(folder / f"{algo_name}_weights.csv")
    if mu_sigma_dict is not None:
        save_predictive(folder, algo_name, mu_sigma_dict)


# ── Public API ────────────────────────────────────────────────────────────────


def calculate_all_functions_perdatasets(
    dataset: pd.DataFrame,
    dataset_name: str,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Run the algorithms on one dataset and save every output.

    Writes into ``<output_dir>/<dataset_name>/``: ``returns.csv``, and for
    each strategy ``<strategy>_weights.csv``; predictive strategies also
    ``<strategy>_predictive.npz`` (forecast + index) and
    ``<strategy>_longonly_weights.csv``; the possibilistic run also
    ``possibilistic_diagnostics.csv``.

    Parameters
    ----------
    dataset : numeric returns, no RF column; indexed by Date for real data
        (row numbers for simulated data). The index is carried to every output.
    dataset_name : output sub-folder name.
    settings : run settings (``None`` = ``RunSettings()``, the config.py
        defaults).
    algorithms : names from ``ALGORITHMS`` to run (``None`` = all), always in
        registry order. Unknown names raise.
    output_dir : root folder for the dataset folder (default: current working
        directory).

    Returns the dataset folder.
    """
    settings = RunSettings() if settings is None else settings
    names = _select_algorithms(algorithms)

    base = Path(output_dir) if output_dir is not None else Path.cwd()
    folder = base / dataset_name
    folder.mkdir(parents=True, exist_ok=True)

    returns_df = dataset.copy()
    returns_df.to_csv(folder / "returns.csv")

    for name in names:
        print(f"[{dataset_name}] {name}...")
        run = _run_algorithm(name, returns_df, settings)
        for strategy, out in run.strategies.items():
            _save_algo_outputs(folder, strategy, out.weights, out.forecast)
            if out.forecast is not None:
                long_only = markowitz_long_only(
                    mu_sigma_dict=out.forecast,
                    returns_df=returns_df,
                    start=settings.start,
                    theta=settings.theta,
                )
                long_only.to_csv(folder / f"{strategy}_longonly_weights.csv")
        for stem, table in run.tables.items():
            table.to_csv(folder / f"{stem}.csv")

    print(f"[{dataset_name}] All outputs saved to: {folder}")
    return folder


def call_all_datasets(
    portfolios_paths: Sequence[Path | str],
    risk_free_path: Path | str = RF_PATH,
    start_date: str | None = "1963-01-01",
    end_date: str | None = None,
    output_dir: Path | str | None = None,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
) -> list[Path]:
    """Run the pipeline on every Kenneth French dataset in two passes.

    For each path in ``portfolios_paths`` the file is read twice: the
    value-weighted block (folder ``<stem>_value_weighted``), then the
    equal-weighted block (folder ``<stem>_equal_weighted``). Other tables in
    the file (e.g. "Number of Firms in Portfolios") are ignored.

    In both passes the excess-returns preparation (subtract RF, slice dates,
    drop non-numeric columns) is applied before handing the data to
    ``calculate_all_functions_perdatasets``. A file that can't be loaded is
    skipped with a warning.

    Parameters
    ----------
    portfolios_paths : paths to Kenneth French multi-table CSV files.
    risk_free_path : Fama-French factors CSV with the ``RF`` column.
    start_date, end_date : date window (``None`` = no bound).
    output_dir : root folder for the dataset folders (default: current
        working directory).
    settings, algorithms : passed to ``calculate_all_functions_perdatasets``.

    Returns one folder per (dataset, weighting) that ran.
    """
    output_folders: list[Path] = []

    for raw_path in portfolios_paths:
        p = Path(raw_path)
        stem = p.stem  # e.g. "12_Industry_Portfolios_Daily"

        for weighting in ("value", "equal"):
            dataset_name = f"{stem}_{weighting}_weighted"
            print(f"\n{'=' * 60}")
            print(f"Dataset : {dataset_name}")
            print(f"File    : {p.name}")
            print(f"{'=' * 60}")

            try:
                excess_df = load_excess_returns(
                    portfolios_path=p,
                    risk_free_path=risk_free_path,
                    weighting=weighting,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                print(
                    f"  [WARNING] Could not load {weighting} block "
                    f"from {p.name}: {exc}"
                )
                continue

            returns_df = prepare_returns(excess_df)  # indexed by Date

            folder = calculate_all_functions_perdatasets(
                returns_df,
                dataset_name,
                settings=settings,
                algorithms=algorithms,
                output_dir=output_dir,
            )
            # RF on the same dates, so evaluation matches it by date (CEQ).
            excess_df[["Date", "RF"]].to_csv(folder / "rf.csv", index=False)
            output_folders.append(folder)

    return output_folders


# ── Simulated-data pipeline ───────────────────────────────────────────────────


def call_all_simulated_datasets(
    sim_data_dir: Path | str,
    sim_files: list[tuple[str, str, str]],
    output_dir: Path | str | None = None,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
) -> list[Path]:
    """Run the pipeline on saved simulated datasets.

    Loads portfolio CSVs and their paired RF CSVs from ``sim_data_dir``
    (written by ``simulated_datasets.generate_and_save_sim_datasets``).
    Data is already excess returns in decimals. The RF CSV is copied into
    each output folder so the evaluation pipeline can compute CEQ. Missing
    files are skipped with a warning.

    Parameters
    ----------
    sim_data_dir : folder with the saved CSVs.
    sim_files : one ``(dataset_name, portfolio_filename, rf_filename)`` per DGP.
    output_dir : root folder for the dataset folders.
    settings, algorithms : passed to ``calculate_all_functions_perdatasets``.
    """
    sim_data_dir = Path(sim_data_dir)
    output_folders: list[Path] = []

    for dataset_name, port_file, rf_file in sim_files:
        port_path = sim_data_dir / port_file
        rf_path = sim_data_dir / rf_file

        for p in (port_path, rf_path):
            if not p.exists():
                print(
                    f"  [WARNING] {p} not found — skipping {dataset_name}. "
                    "Run simulated_datasets.generate_and_save_sim_datasets() first."
                )
                break
        else:
            print(f"\n{'=' * 60}")
            print(f"Simulation : {dataset_name}")
            print(f"{'=' * 60}")

            portfolios_df = pd.read_csv(port_path)
            returns_df = portfolios_df[
                [c for c in portfolios_df.columns if c.startswith("Asset")]
            ]

            folder = calculate_all_functions_perdatasets(
                returns_df,
                dataset_name,
                settings=settings,
                algorithms=algorithms,
                output_dir=output_dir,
            )

            # Copy RF into output folder so evaluation pipeline can load it.
            pd.read_csv(rf_path).to_csv(folder / "rf.csv", index=False)

            output_folders.append(folder)

    return output_folders


# ── Entry points (settings = config.py defaults, the FYP values) ─────────────


def main_simulated() -> list[Path]:
    # Saved CSVs of every DGP (see simulated_datasets.sim_paths / DGPS).
    sim_files = []
    for label in DGPS:
        data_csv, rf_csv = sim_paths(label)
        sim_files.append((label, data_csv.name, rf_csv.name))

    return call_all_simulated_datasets(
        SIM_DATA_DIR, sim_files, output_dir=RESULTS_DIR, settings=RunSettings()
    )


def main() -> list[Path]:
    return call_all_datasets(
        [DATASETS_DIR / f for f in FRENCH_PORTFOLIO_FILES],
        risk_free_path=RF_PATH,
        start_date="1980-01-01",
        output_dir=RESULTS_DIR,
        settings=RunSettings(),
    )


if __name__ == "__main__":
    main()
