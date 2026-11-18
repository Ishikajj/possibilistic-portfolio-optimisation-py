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

Run folders: ``output_dir`` is one run. It gets a ``config.json`` with the
settings used, and one sub-folder per dataset. Past results are never
overwritten: a dataset folder that already exists, or a run folder made with
other settings, raises. ``config.json`` also records, per call, which
datasets finished (and which algorithms failed) and the end time and
duration: a call without an end time didn't finish, and under "done" the
algorithms that finished per dataset. ``resume=True`` (``--resume``) continues
a stopped run in the same folder: datasets already done are skipped, and in a
half-finished dataset only the algorithms not done yet run (the files of
finished algorithms are never touched).

Logging: every run logs with timestamps to the terminal and to ``run.log`` in
the run folder (appended), including warnings and the time each algorithm
took. An algorithm that fails is logged with its traceback; the others still
run, and the end of the run lists the failures.

Command line (``--help`` for the options)::

    uv run python -m possibilistic_portfolio.pipeline_wrapper
    uv run python -m possibilistic_portfolio.pipeline_wrapper --simulated \
        --out results_new --algorithms bayesian_averaging possibilistic
    uv run python -m possibilistic_portfolio.pipeline_wrapper --out results_new \
        --resume        # continue after a crash / Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
    BAYESIAN_PRUNE_THRESHOLD,
    BURN_IN,
    DATASETS_DIR,
    END_DATE,
    FRENCH_PORTFOLIO_FILES,
    K_NEIGHBOURS,
    KEEP_NEWEST,
    MERGE_THRESHOLD,
    NECESSITY_DEVICE,
    NECESSITY_ETA,
    NECESSITY_GAMMA,
    NECESSITY_N_JOBS,
    NU_BANDWIDTH,
    PERIODS_UNTIL_INVESTMENT,
    POSSIBILISTIC_MAX_MODELS,
    RESULTS_DIR,
    RF_PATH,
    ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_SHORT,
    ROOT,
    RUN_ALGORITHMS,
    SCALING_FACTOR,
    SIM_DATA_DIR,
    START_DATE,
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

# by name, not __name__: run with `python -m`, __name__ is "__main__"
log = logging.getLogger("possibilistic_portfolio.pipeline_wrapper")


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
    keep_newest: bool = KEEP_NEWEST  # pruning keeps the newborn (both methods)
    bayesian_max_models: int = BAYESIAN_MAX_MODELS
    bayesian_prune_threshold: float = BAYESIAN_PRUNE_THRESHOLD
    possibilistic_max_models: int = POSSIBILISTIC_MAX_MODELS
    merge_threshold: float = MERGE_THRESHOLD
    nu_bandwidth: int = NU_BANDWIDTH
    k_neighbours: int = K_NEIGHBOURS
    necessity: bool = USE_NECESSITY  # also run masked / power / exp
    gamma: float = NECESSITY_GAMMA
    eta: float = NECESSITY_ETA
    device: str = NECESSITY_DEVICE
    n_jobs: int = NECESSITY_N_JOBS  # necessity workers on CPU

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


class AlgorithmsFailed(RuntimeError):
    """Raised by ``calculate_all_functions_perdatasets`` after it ran every
    algorithm, if some failed: ``failed`` = name -> error. The others'
    outputs are saved in ``folder``; the failed ones have no files."""

    def __init__(self, folder: Path, failed: dict[str, str]) -> None:
        self.folder, self.failed = folder, failed
        super().__init__(
            f"{folder.name}: {len(failed)} algorithm(s) failed: "
            + "; ".join(f"{name}: {err}" for name, err in failed.items())
        )


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
            "prune_threshold": s.bayesian_prune_threshold,
            "keep_newest": s.keep_newest,
        },
    ),
    "possibilistic": (
        possibilistic_bayesian.run_core,
        lambda s: {
            "burn_in": s.burn_in,
            "theta": s.theta,
            "max_models": s.possibilistic_max_models,
            "keep_newest": s.keep_newest,
            "merge_threshold": s.merge_threshold,
            "nu_bandwidth": s.nu_bandwidth,
            "k_neighbours": s.k_neighbours,
            "necessity": s.necessity,
            "gamma": s.gamma,
            "eta": s.eta,
            "device": s.device,
            "n_jobs": s.n_jobs,
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


def _code_version() -> str | None:
    """Short git commit of the code, "+ uncommitted changes" if the working
    tree differs from it; None when git isn't available."""
    try:
        commit, status = (
            subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for args in (
                ["rev-parse", "--short", "HEAD"],
                ["status", "--porcelain", "--untracked-files=no"],
            )
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit} + uncommitted changes" if status else commit


def record_run_config(
    output_dir: Path | str, settings: RunSettings, **call: Any
) -> Path:
    """Write the settings of a run to ``<output_dir>/config.json``.

    One run folder = one set of settings: if the folder already has a
    config.json with other settings, this raises instead of mixing results.
    Each call (real data, simulated data) is appended under ``"calls"`` with
    its own details (`call`: datasets, date window, algorithms) and the code
    version. ``SCALING_FACTOR`` is what the evaluation will use (config.py).
    """
    output_dir = Path(output_dir)
    path = output_dir / "config.json"
    current = {**asdict(settings), "start": settings.start}
    config: dict[str, Any] = {
        "SCALING_FACTOR": SCALING_FACTOR,
        "settings": current,
        "calls": [],
    }
    if path.exists():
        config = json.loads(path.read_text())
        if config.get("settings") != current:
            raise ValueError(
                f"{path} was written by a run with other settings; use a new "
                "output folder (one run folder = one set of settings)"
            )
    config["calls"].append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "code": _code_version(),
            **call,
            "finished": [],
            "failed": {},
            "done": {},
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, config)
    return path


def _update_run_call(
    output_dir: Path,
    *,
    finished: str | None = None,
    failed: tuple[str, dict[str, str]] | None = None,
    done: tuple[str, str] | None = None,
    end: bool = False,
) -> None:
    """Update this run's entry (the last call) in config.json: add a dataset
    whose outputs are all saved (`finished`), or one where some algorithms
    failed (`failed` = (dataset, {algorithm: error})), or an algorithm whose
    outputs are all saved (`done` = (dataset, algorithm)), or record the end
    time and the duration (`end`). A call without "end_time" didn't finish."""
    path = output_dir / "config.json"
    config = json.loads(path.read_text())
    call = config["calls"][-1]
    if finished is not None:
        call["finished"].append(finished)
    if failed is not None:
        dataset, errors = failed
        call["failed"][dataset] = errors
    if done is not None:
        dataset, name = done
        call["done"].setdefault(dataset, []).append(name)
    if end:
        now = datetime.now()
        started = datetime.fromisoformat(call["time"])
        call["end_time"] = now.isoformat(timespec="seconds")
        call["duration_min"] = round((now - started).total_seconds() / 60, 1)
    _write_json(path, config)


def _done_algorithms(output_dir: Path) -> dict[str, set[str]]:
    """Dataset -> algorithms whose outputs are saved, over every call of
    the run folder so far (what a resumed run doesn't redo)."""
    path = output_dir / "config.json"
    done: dict[str, set[str]] = {}
    if path.exists():
        for call in json.loads(path.read_text())["calls"]:
            for dataset, names in call.get("done", {}).items():
                done.setdefault(dataset, set()).update(names)
    return done


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, default=str) + "\n")


# ── Logging ───────────────────────────────────────────────────────────────────


@contextmanager
def _run_log(output_dir: Path) -> Iterator[None]:
    """Log the package (every ``possibilistic_portfolio`` module) and all
    warnings, with timestamps, to the terminal and to
    ``<output_dir>/run.log`` (appended) while the block runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log"),
    ]
    loggers = [
        logging.getLogger("possibilistic_portfolio"),
        logging.getLogger("py.warnings"),  # warnings.warn, via captureWarnings
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    for logger in loggers:
        logger.setLevel(logging.INFO)
        for handler in handlers:
            logger.addHandler(handler)
    logging.captureWarnings(True)
    try:
        yield
    finally:
        logging.captureWarnings(False)
        for logger in loggers:
            for handler in handlers:
                logger.removeHandler(handler)
        for handler in handlers:
            handler.close()


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{seconds:.1f} s"


def _log_failures(failures: dict[str, dict[str, str]]) -> None:
    """The summary at the end of a run."""
    if not failures:
        log.info("Run finished: every algorithm ran on every dataset")
        return
    n = sum(len(errors) for errors in failures.values())
    log.error("Run finished with %d failed algorithm run(s):", n)
    for dataset, errors in failures.items():
        for name, err in errors.items():
            log.error("  %s / %s: %s", dataset, name, err)


# ── Public API ────────────────────────────────────────────────────────────────


def calculate_all_functions_perdatasets(
    dataset: pd.DataFrame,
    dataset_name: str,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    resume_done: Collection[str] | None = None,
    on_done: Callable[[str], None] | None = None,
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
    resume_done : continue in an existing dataset folder (resume): these
        algorithms are done and are neither run nor touched. Its
        ``returns.csv`` must equal this dataset. ``None`` = a new folder.
    on_done : called with an algorithm's name once its outputs are saved.

    Returns the dataset folder. Raises FileExistsError if it already exists
    and ``resume_done`` is None (results are never overwritten; choose
    another ``output_dir``, or resume). An
    algorithm that fails is logged and the others still run; at the end
    ``AlgorithmsFailed`` is raised, listing the failed ones.
    """
    settings = RunSettings() if settings is None else settings
    names = _select_algorithms(algorithms)

    base = Path(output_dir) if output_dir is not None else Path.cwd()
    folder = base / dataset_name
    if folder.exists() and resume_done is None:
        raise FileExistsError(
            f"{folder} already exists; results are never overwritten. "
            "Run into a new output folder, or resume."
        )
    folder.mkdir(parents=True, exist_ok=True)

    returns_df = dataset.copy()
    returns_path = folder / "returns.csv"
    if returns_path.exists():  # resuming: must be the same data
        if returns_path.read_text() != returns_df.to_csv():
            raise ValueError(
                f"{returns_path} differs from this dataset (other data or "
                "date window?); can't resume into it"
            )
    else:
        returns_df.to_csv(returns_path)

    done = set(resume_done or ())
    if done & set(names):
        log.info(
            "[%s] resuming; already done: %s",
            dataset_name,
            ", ".join(n for n in names if n in done),
        )
    failed: dict[str, str] = {}
    for name in names:
        if name in done:
            continue
        log.info("[%s] %s ...", dataset_name, name)
        started = time.perf_counter()
        try:
            run = _run_algorithm(name, returns_df, settings)
            _save_run(folder, run, returns_df, settings)
        except Exception as exc:  # logged + raised below; the rest still run
            log.exception(
                "[%s] %s FAILED after %s",
                dataset_name,
                name,
                _duration(time.perf_counter() - started),
            )
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        log.info(
            "[%s] %s done in %s",
            dataset_name,
            name,
            _duration(time.perf_counter() - started),
        )
        if on_done is not None:
            on_done(name)
    if failed:
        raise AlgorithmsFailed(folder, failed)
    log.info("[%s] all outputs saved to %s", dataset_name, folder)
    return folder


def _save_run(
    folder: Path,
    run: AlgorithmRun,
    returns_df: pd.DataFrame,
    settings: RunSettings,
) -> None:
    """Save one algorithm's strategies (weights, forecast, long-only
    weights) and tables into the dataset folder."""
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


def _run_dataset(
    base: Path,
    dataset_name: str,
    returns_df: pd.DataFrame,
    save_rf: Callable[[Path], None],
    *,
    settings: RunSettings,
    algorithms: Sequence[str] | None,
    failures: dict[str, dict[str, str]],
    resume_done: dict[str, set[str]] | None,
) -> Path:
    """One dataset of a run: the algorithms, then ``rf.csv`` (``save_rf``),
    then its entry in config.json — "finished", or "failed" with the
    algorithms that failed (also added to ``failures``). Each algorithm is
    recorded under "done" once saved. ``resume_done`` (resume: dataset ->
    algorithms done in the run folder) skips what is done already."""
    folder = base / dataset_name
    done = None if resume_done is None else resume_done.get(dataset_name, set())
    if done is not None and set(_select_algorithms(algorithms)) <= done:
        if (folder / "rf.csv").exists():
            log.info("[%s] already done, skipped (resume)", dataset_name)
            return folder
    started = time.perf_counter()
    try:
        folder = calculate_all_functions_perdatasets(
            returns_df,
            dataset_name,
            settings=settings,
            algorithms=algorithms,
            output_dir=base,
            resume_done=done,
            on_done=lambda name: _update_run_call(
                base, done=(dataset_name, name)
            ),
        )
        failed: dict[str, str] = {}
    except AlgorithmsFailed as exc:
        folder, failed = exc.folder, exc.failed
    save_rf(folder)
    if failed:
        failures[dataset_name] = failed
        _update_run_call(base, failed=(dataset_name, failed))
    else:
        _update_run_call(base, finished=dataset_name)
    log.info(
        "[%s] dataset done in %s",
        dataset_name,
        _duration(time.perf_counter() - started),
    )
    return folder


def call_all_datasets(
    portfolios_paths: Sequence[Path | str],
    risk_free_path: Path | str = RF_PATH,
    start_date: str | None = START_DATE,
    end_date: str | None = END_DATE,
    output_dir: Path | str | None = None,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
    resume: bool = False,
) -> list[Path]:
    """Run the pipeline on every Kenneth French dataset in two passes.

    For each path in ``portfolios_paths`` the file is read twice: the
    value-weighted block (folder ``<stem>_value_weighted``), then the
    equal-weighted block (folder ``<stem>_equal_weighted``). Other tables in
    the file (e.g. "Number of Firms in Portfolios") are ignored.

    In both passes the excess-returns preparation (subtract RF, slice dates,
    drop non-numeric columns) is applied before handing the data to
    ``calculate_all_functions_perdatasets``. A path that doesn't exist raises
    before anything runs (a typo shouldn't cost a dataset); a file that exists
    but isn't a Kenneth French table file is skipped with a warning.

    Parameters
    ----------
    portfolios_paths : paths to Kenneth French multi-table CSV files.
    risk_free_path : Fama-French factors CSV with the ``RF`` column.
    start_date, end_date : date window (``None`` = no bound).
    output_dir : root folder for the dataset folders (default: current
        working directory).
    settings, algorithms : passed to ``calculate_all_functions_perdatasets``.
    resume : continue a stopped run in ``output_dir`` (same settings): skip
        what config.json lists as done, run the rest.

    Returns one folder per (dataset, weighting) that ran.
    """
    paths = [Path(p) for p in portfolios_paths]
    missing = [str(p) for p in [*paths, Path(risk_free_path)] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {missing}")

    settings = RunSettings() if settings is None else settings
    base = Path(output_dir) if output_dir is not None else Path.cwd()
    resume_done = _done_algorithms(base) if resume else None
    record_run_config(
        base,
        settings,
        data="real",
        datasets=[p.stem for p in paths],
        start_date=start_date,
        end_date=end_date,
        algorithms=_select_algorithms(algorithms),
    )
    output_folders: list[Path] = []
    failures: dict[str, dict[str, str]] = {}

    with _run_log(base):
        log.info("Run folder %s (settings: config.json)", base)
        for p in paths:
            for weighting in ("value", "equal"):
                dataset_name = f"{p.stem}_{weighting}_weighted"
                log.info("Dataset %s (file %s)", dataset_name, p.name)
                try:
                    excess_df = load_excess_returns(
                        portfolios_path=p,
                        risk_free_path=risk_free_path,
                        weighting=weighting,
                        start_date=start_date,
                        end_date=end_date,
                    )
                except ValueError as exc:  # not a French table file
                    log.warning(
                        "Could not load the %s block from %s, skipped: %s",
                        weighting,
                        p.name,
                        exc,
                    )
                    continue

                rf = excess_df[["Date", "RF"]]  # same dates (CEQ by date)
                output_folders.append(
                    _run_dataset(
                        base,
                        dataset_name,
                        prepare_returns(excess_df),  # indexed by Date
                        lambda folder, rf=rf: rf.to_csv(
                            folder / "rf.csv", index=False
                        ),
                        settings=settings,
                        algorithms=algorithms,
                        failures=failures,
                        resume_done=resume_done,
                    )
                )
        _update_run_call(base, end=True)
        _log_failures(failures)
    return output_folders


# ── Simulated-data pipeline ───────────────────────────────────────────────────


def call_all_simulated_datasets(
    sim_data_dir: Path | str,
    sim_files: list[tuple[str, str, str]],
    output_dir: Path | str | None = None,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = None,
    resume: bool = False,
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
    resume : continue a stopped run in ``output_dir`` (same settings): skip
        what config.json lists as done, run the rest.
    """
    sim_data_dir = Path(sim_data_dir)
    settings = RunSettings() if settings is None else settings
    base = Path(output_dir) if output_dir is not None else Path.cwd()
    resume_done = _done_algorithms(base) if resume else None
    record_run_config(
        base,
        settings,
        data="simulated",
        datasets=[name for name, _, _ in sim_files],
        files=[port_file for _, port_file, _ in sim_files],
        algorithms=_select_algorithms(algorithms),
    )
    output_folders: list[Path] = []
    failures: dict[str, dict[str, str]] = {}

    with _run_log(base):
        log.info("Run folder %s (settings: config.json)", base)
        for dataset_name, port_file, rf_file in sim_files:
            port_path = sim_data_dir / port_file
            rf_path = sim_data_dir / rf_file
            missing = [p for p in (port_path, rf_path) if not p.exists()]
            if missing:
                log.warning(
                    "%s not found, %s skipped. Run "
                    "simulated_datasets.generate_and_save_sim_datasets() "
                    "first.",
                    missing[0],
                    dataset_name,
                )
                continue

            log.info("Simulation %s", dataset_name)
            portfolios_df = pd.read_csv(port_path)
            returns_df = portfolios_df[
                [c for c in portfolios_df.columns if c.startswith("Asset")]
            ]
            output_folders.append(
                _run_dataset(
                    base,
                    dataset_name,
                    returns_df,
                    # copy RF so the evaluation pipeline can compute CEQ
                    lambda folder, rf_path=rf_path: pd.read_csv(rf_path).to_csv(
                        folder / "rf.csv", index=False
                    ),
                    settings=settings,
                    algorithms=algorithms,
                    failures=failures,
                    resume_done=resume_done,
                )
            )
        _update_run_call(base, end=True)
        _log_failures(failures)
    return output_folders


# ── Entry points ─────────────────────────────────────────────────────────────
# Settings default to RunSettings() = config.py (the FYP values), the
# algorithms to config.RUN_ALGORITHMS (None = all); output to
# config.RESULTS_DIR unless another run folder is given.


def main_simulated(
    output_dir: Path | str = RESULTS_DIR,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = RUN_ALGORITHMS,
    resume: bool = False,
) -> list[Path]:
    """Run on the saved CSVs of every DGP (simulated_datasets.DGPS)."""
    sim_files = []
    for label in DGPS:
        data_csv, rf_csv = sim_paths(label)
        sim_files.append((label, data_csv.name, rf_csv.name))

    return call_all_simulated_datasets(
        SIM_DATA_DIR,
        sim_files,
        output_dir=output_dir,
        settings=settings,
        algorithms=algorithms,
        resume=resume,
    )


def main(
    output_dir: Path | str = RESULTS_DIR,
    *,
    settings: RunSettings | None = None,
    algorithms: Sequence[str] | None = RUN_ALGORITHMS,
    resume: bool = False,
) -> list[Path]:
    """Run on the Kenneth French files in config.FRENCH_PORTFOLIO_FILES, in
    the date window config.START_DATE … config.END_DATE."""
    return call_all_datasets(
        [DATASETS_DIR / f for f in FRENCH_PORTFOLIO_FILES],
        risk_free_path=RF_PATH,
        output_dir=output_dir,
        settings=settings,
        algorithms=algorithms,
        resume=resume,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the algorithms on every dataset (settings: config.py)."
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="run on the simulated datasets instead of the Kenneth French files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR,
        help=f"run folder (default {RESULTS_DIR.name}/); relative to dev/. "
        "Existing dataset folders are never overwritten",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHMS),
        metavar="NAME",
        help="only these algorithms (default: config.RUN_ALGORITHMS, None = "
        f"all): {', '.join(ALGORITHMS)}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a stopped run in --out (same settings): datasets and "
        "algorithms already done are skipped, the rest runs",
    )
    parser.add_argument(
        "--necessity",
        action="store_true",
        help="possibilistic: also run the necessity schemes (slow)",
    )
    args = parser.parse_args()
    run = main_simulated if args.simulated else main
    run(
        ROOT / args.out,  # an absolute --out stays as it is
        settings=RunSettings(necessity=args.necessity or USE_NECESSITY),
        algorithms=args.algorithms or RUN_ALGORITHMS,
        resume=args.resume,
    )


if __name__ == "__main__":
    _cli()
