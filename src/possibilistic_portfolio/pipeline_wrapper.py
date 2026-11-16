"""Pipeline wrapper: orchestrates all algorithms on a single dataset and loops
over multiple datasets.

Public entry points
-------------------
calculate_all_functions_perdatasets(dataset, dataset_name, ...)
    Run every algorithm on one prepared returns DataFrame and persist outputs.

call_all_datasets(portfolios_paths, ...)
    Loop over Kenneth French CSV files, running two passes each (value- and
    equal-weighted blocks).

call_all_simulated_datasets(sim_data_dir, sim_files, ...)
    Loop over pre-generated simulated dataset CSVs (produced by
    simulated_datasets.generate_and_save_sim_datasets).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
)
from possibilistic_portfolio.data_input import (
    load_excess_returns,
    prepare_returns,
)
from possibilistic_portfolio.markowitz import markowitz_long_only
from possibilistic_portfolio.predictive import save_predictive
from possibilistic_portfolio.simulated_datasets import DGPS, sim_paths

# ── Internal helpers ─────────────────────────────────────────────────────────


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


# ── Public API ─────────────────────────────────


def calculate_all_functions_perdatasets(
    dataset: pd.DataFrame,
    dataset_name: str,
    burn_in: int = BURN_IN,
    periods_until_investment: int = PERIODS_UNTIL_INVESTMENT,
    theta: float = THETA,
    max_models: int = POSSIBILISTIC_MAX_MODELS,
    merge_threshold: float = MERGE_THRESHOLD,
    nu_bandwidth: int = NU_BANDWIDTH,
    k_neighbours: int = K_NEIGHBOURS,
    gamma: float = NECESSITY_GAMMA,
    eta: float = NECESSITY_ETA,
    rolling_window_long: int = ROLLING_WINDOW_LONG,
    rolling_window_short: int = ROLLING_WINDOW_SHORT,
    output_dir: Path | str | None = None,
    device: str = NECESSITY_DEVICE,
) -> Path:
    """Run every portfolio algorithm on one dataset and persist all outputs.

    Algorithms run (in order):
      1.  1/N equal weight
      2.  Market weight           (only when ``Mkt-RF`` column is present)
      3.  Minimum variance
      4.  Historical expanding-window Markowitz
      5.  Rolling-window Markowitz
      6.  Jorion Bayes-Stein
      7.  Kan-Zhou three-fund
      8.  Bayesian model averaging (probabilistic)
      9.  Possibilistic Bayesian averaging — masked weighting
      10. Possibilistic Bayesian averaging — power penalty weighting
      11. Possibilistic Bayesian averaging — exponential penalty weighting

    Outputs are written to ``<output_dir>/<dataset_name>/``.
    Algorithms that produce predictive moments store three files each:
      ``<algo>_weights.csv``, ``<algo>_mus.npy``, ``<algo>_sigmas.npy``.
    Weight-only algorithms store only ``<algo>_weights.csv``.
    The possibilistic run also stores ``possibilistic_diagnostics.csv``.

    Parameters
    ----------
    dataset : pd.DataFrame
        Numeric returns only, no RF column; indexed by Date for real data
        (row numbers for simulated data). The index is carried to every output.
    dataset_name : str
        Used as the output sub-folder name.
    burn_in : int
        Number of initial observations used to seed model priors.
    periods_until_investment : int
        Additional lag before weights are applied (default 1 = next-day).
    theta : float
        Risk-aversion parameter for Markowitz weight computation.
    max_models : int
        Maximum model-pool size for possibilistic and Bayesian averaging.
    merge_threshold : float
        Hellinger distance threshold for merging models (possibilistic only).
    nu_bandwidth : int
        Maximum |nu_i - nu_j| for merge candidacy (possibilistic only).
    k_neighbours : int
        Number of forward neighbours to scan when merging (possibilistic only).
    gamma : float
        Power exponent for power-weighting scheme.
    eta : float
        Penalty scale for exponential-weighting scheme.
    rolling_window_long : int
        Look-back window for long rolling/shrinkage strategies.
    rolling_window_short : int
        Look-back window for short rolling strategies.
    output_dir : Path or str, optional
        Root directory for output folders.  Defaults to current working dir.

    Returns
    -------
    folder : Path
        Path to the created output folder.
    """
    base = Path(output_dir) if output_dir is not None else Path.cwd()
    folder = base / dataset_name
    folder.mkdir(parents=True, exist_ok=True)

    returns_df = dataset.copy()
    returns_df.to_csv(folder / "returns.csv")

    # First row with weights, for every strategy (burn_in is also passed on
    # to the averaging methods, which use it to set their prior).
    start = burn_in + periods_until_investment

    # 1. 1/N ───────────────────────────────────────────────
    print(f"[{dataset_name}] 1/N equal weight...")
    weights_1n = equal_weight_strategy(returns_df, start=start)
    _save_algo_outputs(folder, "equal_weight", weights=weights_1n)

    # 2. Market weight ───────────────────────────────────
    if "Mkt-RF" in returns_df.columns:
        print(f"[{dataset_name}] Market weight...")
        weights_mkt = market_weight_strategy(returns_df, start=start)
        _save_algo_outputs(folder, "market_weight", weights=weights_mkt)

    # 3. Minimum variance ───────────────────────────────────────
    print(f"[{dataset_name}] Minimum variance...")
    weights_mv = minimum_variance_strategy(
        returns_df,
        start=start,
        window=rolling_window_long,
    )
    _save_algo_outputs(folder, "minimum_variance", weights=weights_mv)

    # 4. Historical expanding window ─────────────────────────
    print(f"[{dataset_name}] Historical expanding-window Markowitz...")
    hist_ms, weights_hist = historical_expectations_weights(
        returns_df, start=start, theta=theta
    )
    _save_algo_outputs(
        folder,
        "historical_expanding",
        weights=weights_hist,
        mu_sigma_dict=hist_ms,
    )

    # 5. Rolling window ────────────────────────────────────────
    print(
        f"[{dataset_name}] Rolling-window Markowitz (window={rolling_window_long})..."
    )
    roll_long_ms, weights_roll = rolling_window_weights(
        returns_df, start=start, window=rolling_window_long, theta=theta
    )
    _save_algo_outputs(
        folder,
        "rolling_window_long",
        weights=weights_roll,
        mu_sigma_dict=roll_long_ms,
    )

    # 5b. Rolling window (short) ──────────────────────────────
    print(
        f"[{dataset_name}] Rolling-window Markowitz (window={rolling_window_short})..."
    )
    roll_short_ms, weights_roll = rolling_window_weights(
        returns_df, start=start, window=rolling_window_short, theta=theta
    )
    _save_algo_outputs(
        folder,
        "rolling_window_short",
        weights=weights_roll,
        mu_sigma_dict=roll_short_ms,
    )

    # 6. Jorion Bayes-Stein ──────────────────────────────────
    print(f"[{dataset_name}] Jorion Bayes-Stein...")
    jbs_ms, weights_jbs = jorion_bayes_stein_strategy(
        returns_df,
        window=rolling_window_long,
        start=start,
        theta=theta,
    )
    _save_algo_outputs(
        folder, "jorion_bayes_stein", weights=weights_jbs, mu_sigma_dict=jbs_ms
    )

    # 7. Kan-Zhou three-fund ────────────────────────────────────
    print(f"[{dataset_name}] Kan-Zhou three-fund...")
    kz_ms, weights_kz = kan_zhou_three_fund_strategy(
        returns_df,
        window=rolling_window_long,
        start=start,
        theta=theta,
    )
    _save_algo_outputs(
        folder, "kan_zhou_three_fund", weights=weights_kz, mu_sigma_dict=kz_ms
    )

    # 8. Bayesian model averaging ────────────────────────────
    print(f"[{dataset_name}] Bayesian model averaging...")
    # Bayesian has its own pool cap (config.BAYESIAN_MAX_MODELS);
    # `max_models` is the possibilistic one. Passed through properly in P2.
    bay_ms, weights_bay = bayesian_averaging.run_core(
        returns_df,
        start=start,
        burn_in=burn_in,
        theta=theta,
        max_models=BAYESIAN_MAX_MODELS,
    )
    _save_algo_outputs(
        folder, "bayesian_averaging", weights=weights_bay, mu_sigma_dict=bay_ms
    )

    # 9–11. Possibilistic Bayesian averaging (three weighting schemes) ───
    print(f"[{dataset_name}] Possibilistic Bayesian averaging...")
    (
        diag_df,
        masked_pred,
        weights_masked,
        power_pred,
        weights_power,
        exp_pred,
        weights_exp,
    ) = possibilistic_bayesian.run_core(
        returns_df,
        start=start,
        burn_in=burn_in,
        theta=theta,
        merge_threshold=merge_threshold,
        max_models=max_models,
        nu_bandwidth=nu_bandwidth,
        k_neighbours=k_neighbours,
        gamma=gamma,
        eta=eta,
        device=device,
    )
    _save_algo_outputs(
        folder,
        "possibilistic_masked",
        weights=weights_masked,
        mu_sigma_dict=masked_pred,
    )
    _save_algo_outputs(
        folder,
        "possibilistic_power",
        weights=weights_power,
        mu_sigma_dict=power_pred,
    )
    _save_algo_outputs(
        folder, "possibilistic_exp", weights=weights_exp, mu_sigma_dict=exp_pred
    )
    diag_df.to_csv(folder / "possibilistic_diagnostics.csv")

    for algo_name, mu_sigma in [
        ("historical_expanding", hist_ms),
        ("rolling_window_long", roll_long_ms),
        ("rolling_window_short", roll_short_ms),
        ("jorion_bayes_stein", jbs_ms),
        ("kan_zhou_three_fund", kz_ms),
        ("bayesian_averaging", bay_ms),
        ("possibilistic_masked", masked_pred),
        ("possibilistic_power", power_pred),
        ("possibilistic_exp", exp_pred),
    ]:
        long_only_weights = markowitz_long_only(
            mu_sigma_dict=mu_sigma,
            returns_df=returns_df,
            start=start,
            theta=theta,
        )
        long_only_weights.to_csv(folder / f"{algo_name}_longonly_weights.csv")

    print(f"[{dataset_name}] All outputs saved to: {folder}")
    return folder


def call_all_datasets(
    portfolios_paths: Sequence[Path | str],
    risk_free_path: Path | str = RF_PATH,
    start_date: str | None = "1963-01-01",
    end_date: str | None = None,
    output_dir: Path | str | None = None,
    burn_in: int = BURN_IN,
    periods_until_investment: int = PERIODS_UNTIL_INVESTMENT,
    theta: float = THETA,
    max_models: int = POSSIBILISTIC_MAX_MODELS,
    merge_threshold: float = MERGE_THRESHOLD,
    nu_bandwidth: int = NU_BANDWIDTH,
    k_neighbours: int = K_NEIGHBOURS,
    gamma: float = NECESSITY_GAMMA,
    eta: float = NECESSITY_ETA,
    rolling_window_long: int = ROLLING_WINDOW_LONG,
    rolling_window_short: int = ROLLING_WINDOW_SHORT,
    device: str = NECESSITY_DEVICE,
) -> list[Path]:
    """Run the full pipeline on every Kenneth French dataset in two passes.

    For each path in ``portfolios_paths`` the file is read twice: the
    value-weighted block (folder ``<stem>_value_weighted``), then the
    equal-weighted block (folder ``<stem>_equal_weighted``). Other tables in
    the file (e.g. "Number of Firms in Portfolios") are ignored.

    In both passes the excess-returns preparation (subtract RF, slice dates,
    drop non-numeric columns) is applied before handing the data to
    ``calculate_all_functions_perdatasets``.

    Parameters
    ----------
    portfolios_paths : list of Path or str
        Paths to Kenneth French multi-table CSV files.
    risk_free_path : Path or str
        Path to the Fama-French factors CSV containing the ``RF`` column.
    start_date : str, optional
        Earliest date to include (default ``"1963-01-01"``).
    end_date : str, optional
        Latest date to include (default: no upper bound).
    output_dir : Path or str, optional
        Root directory for output folders.  Defaults to current working dir.
    burn_in, periods_until_investment, theta, max_models, merge_threshold,
    nu_bandwidth, k_neighbours, gamma, eta, rolling_window :
        Forwarded verbatim to ``calculate_all_functions_perdatasets``.

    Returns
    -------
    output_folders : list of Path
        One Path per successful (dataset, weighting) combination.
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
                dataset=returns_df,
                dataset_name=dataset_name,
                burn_in=burn_in,
                periods_until_investment=periods_until_investment,
                theta=theta,
                max_models=max_models,
                merge_threshold=merge_threshold,
                nu_bandwidth=nu_bandwidth,
                k_neighbours=k_neighbours,
                gamma=gamma,
                eta=eta,
                rolling_window_long=rolling_window_long,
                rolling_window_short=rolling_window_short,
                output_dir=output_dir,
                device=device,
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
    burn_in: int = BURN_IN,
    periods_until_investment: int = PERIODS_UNTIL_INVESTMENT,
    theta: float = THETA,
    max_models: int = POSSIBILISTIC_MAX_MODELS,
    merge_threshold: float = MERGE_THRESHOLD,
    nu_bandwidth: int = NU_BANDWIDTH,
    k_neighbours: int = K_NEIGHBOURS,
    gamma: float = NECESSITY_GAMMA,
    eta: float = NECESSITY_ETA,
    rolling_window_long: int = ROLLING_WINDOW_LONG,
    rolling_window_short: int = ROLLING_WINDOW_SHORT,
    device: str = NECESSITY_DEVICE,
) -> list[Path]:
    """Run the full algorithm suite on all saved simulated datasets.

    Loads portfolio CSVs and their paired RF CSVs from ``sim_data_dir``
    (written by ``simulated_datasets.generate_and_save_sim_datasets``).
    Data is already excess returns in decimals — no marker parsing or RF
    subtraction needed.  The RF CSV is copied into each output folder so
    the evaluation pipeline can compute CEQ.

    Parameters
    ----------
    sim_data_dir : Path or str
        Directory containing the saved CSVs.
    sim_files : list of (dataset_name, portfolio_filename, rf_filename)
        One tuple per DGP.
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
                dataset=returns_df,
                dataset_name=dataset_name,
                burn_in=burn_in,
                periods_until_investment=periods_until_investment,
                theta=theta,
                max_models=max_models,
                merge_threshold=merge_threshold,
                nu_bandwidth=nu_bandwidth,
                k_neighbours=k_neighbours,
                gamma=gamma,
                eta=eta,
                rolling_window_long=rolling_window_long,
                rolling_window_short=rolling_window_short,
                output_dir=output_dir,
                device=device,
            )

            # Copy RF into output folder so evaluation pipeline can load it.
            pd.read_csv(rf_path).to_csv(folder / "rf.csv", index=False)

            output_folders.append(folder)

    return output_folders


def main_simulated():
    # Saved CSVs of every DGP (see simulated_datasets.sim_paths / DGPS).
    sim_files = []
    for label in DGPS:
        data_csv, rf_csv = sim_paths(label)
        sim_files.append((label, data_csv.name, rf_csv.name))

    return call_all_simulated_datasets(
        sim_data_dir=SIM_DATA_DIR,
        sim_files=sim_files,
        burn_in=500,
        periods_until_investment=500,
        theta=1.0,
        max_models=100,
        merge_threshold=0.15,
        nu_bandwidth=50,
        k_neighbours=5,
        gamma=1.0,
        eta=1.0,
        rolling_window_long=252,
        rolling_window_short=63,
        device="cuda",
        output_dir=RESULTS_DIR,
    )


def main():
    portfolio_paths = [DATASETS_DIR / f for f in FRENCH_PORTFOLIO_FILES]
    risk_free_path = RF_PATH

    start_date = "1980-01-01"

    burn_in = 500

    periods_until_investment = 500

    theta = 1

    max_models = 100

    merge_threshold = 0.15

    nu_bandwidth = 50

    k_neighbours = 5
    gamma = 1.0

    eta = 1.0

    rolling_window_long = 252

    rolling_window_short = 63

    return call_all_datasets(
        portfolios_paths=portfolio_paths,
        risk_free_path=risk_free_path,
        start_date=start_date,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
        max_models=max_models,
        merge_threshold=merge_threshold,
        nu_bandwidth=nu_bandwidth,
        k_neighbours=k_neighbours,
        gamma=gamma,
        eta=eta,
        rolling_window_long=rolling_window_long,
        rolling_window_short=rolling_window_short,
        device="cuda",
        output_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
