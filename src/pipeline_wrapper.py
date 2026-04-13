"""Pipeline wrapper: orchestrates all algorithms on a single dataset,
and loops over multiple Kenneth French datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_input import (
    load_excess_returns_from_kenneth_french_path,
    load_risk_free_rate,
    prepare_returns,
    DATA_PATH,
    RISK_FREE_RATE_PATH,
)
from typing import Callable
import possibilistic_bayesian
import bayesian_averaging
from alternative_investment_rules import (
    equal_weight_strategy,
    market_weight_strategy,
    minimum_variance_strategy,
    jorion_bayes_stein_strategy,
    kan_zhou_three_fund_strategy,
    historical_expectations_weights,
    rolling_window_weights,
)

# ── Kenneth French marker strings ────────────────────────────────────────────
_VW_START = "Average Value Weighted Returns -- Daily"
_VW_END = "Average Equal Weighted Returns -- Daily"
_EW_START = "Average Equal Weighted Returns -- Daily"
# The EW block runs to EOF (only a copyright line follows), so no end marker.


# ── Internal helpers ─────────────────────────────────────────────────────────


# RETURNS ARE SAVED WITHOUT SHIFTING/ LAG
def _save_algo_outputs(
    folder: Path,
    algo_name: str,
    weights: pd.DataFrame | None = None,
    mu_sigma_dict: dict | None = None,
) -> None:
    """Persist weights (CSV) and, where available, predictive moments (npy).

    File naming convention:
      <algo_name>_weights.csv
      <algo_name>_mus.npy
      <algo_name>_sigmas.npy
    """
    if weights is not None:
        weights.to_csv(folder / f"{algo_name}_weights.csv")
    if mu_sigma_dict is not None:
        mu_hat = mu_sigma_dict.get("mu_hat")
        sigma_hat = mu_sigma_dict.get("sigma_hat")
        if mu_hat is not None:
            np.save(folder / f"{algo_name}_mus.npy", mu_hat)
        if sigma_hat is not None:
            np.save(folder / f"{algo_name}_sigmas.npy", sigma_hat)


# ── Public API ─────────────────────────────────


def calculate_all_functions_perdatasets(
    dataset: pd.DataFrame,
    dataset_name: str,
    burn_in: int = 1000,
    periods_until_investment: int = 1,
    theta: float = 1.0,
    max_models: int = 100,
    merge_threshold: float = 0.01,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
    rolling_window_long: int = 250,
    rolling_window_short: int = 63,
    output_dir: Path | str | None = None,
    device: str = "cpu",
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
      10. Possibilistic Bayesian averaging — power weighting
      11. Possibilistic Bayesian averaging — exponential penalty weighting

    Outputs are written to ``<output_dir>/<dataset_name>/``.
    Algorithms that produce predictive moments store three files each:
      ``<algo>_weights.csv``, ``<algo>_mus.npy``, ``<algo>_sigmas.npy``.
    Weight-only algorithms store only ``<algo>_weights.csv``.
    The possibilistic run also stores ``possibilistic_diagnostics.csv``.

    Parameters
    ----------
    dataset : pd.DataFrame
        Numeric returns-only DataFrame (no Date / RF column).
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
    rolling_window : int
        Look-back window for rolling/shrinkage strategies.
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

    # 1. 1/N ───────────────────────────────────────────────
    print(f"[{dataset_name}] 1/N equal weight...")
    weights_1n = equal_weight_strategy(
        returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )
    _save_algo_outputs(folder, "equal_weight", weights=weights_1n)

    # 2. Market weight ───────────────────────────────────
    if "Mkt-RF" in returns_df.columns:
        print(f"[{dataset_name}] Market weight...")
        weights_mkt = market_weight_strategy(
            returns_df,
            burn_in=burn_in,
            periods_until_investment=periods_until_investment,
        )
        _save_algo_outputs(folder, "market_weight", weights=weights_mkt)

    # 3. Minimum variance ───────────────────────────────────────
    print(f"[{dataset_name}] Minimum variance...")
    weights_mv = minimum_variance_strategy(
        returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        window=rolling_window_long,
    )
    _save_algo_outputs(folder, "minimum_variance", weights=weights_mv)

    # 4. Historical expanding window ─────────────────────────
    print(f"[{dataset_name}] Historical expanding-window Markowitz...")
    hist_ms, weights_hist = historical_expectations_weights(
        returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
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
    roll_ms, weights_roll = rolling_window_weights(
        returns_df,
        window=rolling_window_long,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )
    _save_algo_outputs(
        folder,
        "rolling_window_long",
        weights=weights_roll,
        mu_sigma_dict=roll_ms,
    )

    # 5. Rolling window ────────────────────────────────────────
    print(
        f"[{dataset_name}] Rolling-window Markowitz (window={rolling_window_short})..."
    )
    roll_ms, weights_roll = rolling_window_weights(
        returns_df,
        window=rolling_window_short,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )
    _save_algo_outputs(
        folder,
        "rolling_window_short",
        weights=weights_roll,
        mu_sigma_dict=roll_ms,
    )

    # 6. Jorion Bayes-Stein ──────────────────────────────────
    print(f"[{dataset_name}] Jorion Bayes-Stein...")
    jbs_ms, weights_jbs = jorion_bayes_stein_strategy(
        returns_df,
        window=rolling_window_long,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
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
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )
    _save_algo_outputs(
        folder, "kan_zhou_three_fund", weights=weights_kz, mu_sigma_dict=kz_ms
    )

    # 8. Bayesian model averaging ────────────────────────────
    print(f"[{dataset_name}] Bayesian model averaging...")
    bay_ms, weights_bay = bayesian_averaging.run_core(
        returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
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
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
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

    print(f"[{dataset_name}] All outputs saved to: {folder}")
    return folder


def call_all_datasets(
    portfolios_paths: list[Path | str | tuple[Path | str, str | None]],
    risk_free_path: Path | str = RISK_FREE_RATE_PATH,
    start_date: str | None = "1963-01-01",
    end_date: str | None = None,
    output_dir: Path | str | None = None,
    burn_in: int = 1000,
    periods_until_investment: int = 1,
    theta: float = 1.0,
    max_models: int = 100,
    merge_threshold: float = 0.15,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
    rolling_window_long: int = 250,
    rolling_window_short: int = 63,
) -> list[Path]:
    """Run the full pipeline on every Kenneth French dataset in two passes.

    For each path in ``portfolios_paths`` the file is read twice:

      Pass 1 — Value-weighted block
        start marker : "Average Value Weighted Returns -- Daily"
        end   marker : "Average Equal Weighted Returns -- Daily"
        folder name  : ``<stem>_value_weighted``

      Pass 2 — Equal-weighted block
        start marker : "Average Equal Weighted Returns -- Daily"
        end   marker : (EOF — the EW section runs to the end of the file)
        folder name  : ``<stem>_equal_weighted``

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
    for entry in portfolios_paths:
        if isinstance(entry, tuple):
            raw_path, ew_end_override = entry
        else
            raw_path = entry
            ew_end_override = None
        p = Path(raw_path)
        stem = p.stem  # e.g. "12_Industry_Portfolios_Daily"

        for weighting, start_marker, end_marker in [
            ("value_weighted", _VW_START, _VW_END),
            ("equal_weighted", _EW_START, ew_end_override),
        ]:
            dataset_name = f"{stem}_{weighting}"
            print(f"\n{'=' * 60}")
            print(f"Dataset : {dataset_name}")
            print(f"File    : {p.name}")
            print(f"{'=' * 60}")

            try:
                excess_df = load_excess_returns_from_kenneth_french_path(
                    portfolios_path=p,
                    risk_free_path=risk_free_path,
                    start_date=start_date,
                    end_date=end_date,
                    start_marker=start_marker,
                    end_marker=end_marker,
                )
            except Exception as exc:
                print(
                    f"  [WARNING] Could not load {weighting} block "
                    f"from {p.name}: {exc}"
                )
                continue

            returns_df = prepare_returns(excess_df)

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
            )
            output_folders.append(folder)

    return output_folders


# ── Simulated-data pipeline ───────────────────────────────────────────────────


def call_all_simulated_datasets(
    sim_data_dir: Path | str,
    sim_files: list[tuple[str, str, str]],
    output_dir: Path | str | None = None,
    burn_in: int = 500,
    periods_until_investment: int = 1,
    theta: float = 1.0,
    max_models: int = 100,
    merge_threshold: float = 0.01,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
    rolling_window_long: int = 250,
    rolling_window_short: int = 63,
) -> list[Path]:
    """Run the full algorithm suite on all saved simulated datasets.

    Loads portfolio CSVs and their paired RF CSVs from ``sim_data_dir``
    (written by ``generate_and_save_sim_datasets`` in table1_summary_stats.py).
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
    from simulated_datasets import prepare_sim_returns

    sim_data_dir = Path(sim_data_dir)
    output_folders: list[Path] = []

    for dataset_name, port_file, rf_file in sim_files:
        port_path = sim_data_dir / port_file
        rf_path = sim_data_dir / rf_file

        for p in (port_path, rf_path):
            if not p.exists():
                print(
                    f"  [WARNING] {p} not found — skipping {dataset_name}. "
                    "Run generate_and_save_sim_datasets() in table1_summary_stats.py first."
                )
                break
        else:
            print(f"\n{'=' * 60}")
            print(f"Simulation : {dataset_name}")
            print(f"{'=' * 60}")

            portfolios_df = pd.read_csv(port_path, parse_dates=["Date"])
            n_assets = sum(
                1 for c in portfolios_df.columns if c.startswith("Asset")
            )
            returns_df = prepare_sim_returns(portfolios_df, n_assets=n_assets)

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
            )

            # Copy RF into output folder so evaluation pipeline can load it.
            pd.read_csv(rf_path, parse_dates=["Date"]).to_csv(
                folder / "rf.csv", index=False
            )

            output_folders.append(folder)

    return output_folders


def main2():
    SIM_DATA_DIR = Path("../datasets/simulated")
    SIM_T = 6000
    SIM_N = 5
    SIM_SEED = 42

    sim_files = [
        (
            "iid",
            f"iid_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_iid_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
        (
            "reg",
            f"reg_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_reg_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
        (
            "break",
            f"break_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_break_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
        (
            "mr",
            f"mr_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_mr_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
        (
            "cov_break",
            f"cov_break_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_cov_break_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
        (
            "stoch_vol",
            f"stoch_vol_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
            f"rf_stoch_vol_T{SIM_T}_N{SIM_N}_seed{SIM_SEED}.csv",
        ),
    ]

    return call_all_simulated_datasets(
        sim_data_dir=SIM_DATA_DIR,
        sim_files=sim_files,
        burn_in=500,
        periods_until_investment=500,
        theta=1.0,
        max_models=80,
        merge_threshold=1.0,
        nu_bandwidth=50,
        k_neighbours=5,
        gamma=1.0,
        eta=1.0,
        rolling_window_long=252,
        rolling_window_short=63,
    )


def main():
    portfolio_paths = [
        "../datasets/6_Portfolios_size_momentum.csv",
        "../datasets/10_Portfolios_ltr.csv",
        "../datasets/6_Portfolios_size_str.csv",
        "../datasets/10_Portfolios_Prior_momentum.csv",
        "../datasets/10_Portfolios_Prior_str.csv",
        ("../datasets/6_Portfolios_size_btm.csv", "Number of Firms in Portfolios")
    ]
    risk_free_path = "../datasets/F-F_Research_Data_Factors_daily.csv"

    start_date = "1980-01-01"

    burn_in = 500

    periods_until_investment = 500

    theta = 1

    max_models = 80

    merge_threshold = 1.0

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
    )


if __name__ == "__main__":
    main2()
