"""Table 1 – Daily summary statistics for Kenneth French and simulated datasets.

Produces three panels:
  Panel A: Ken French value-weighted portfolios
  Panel B: Ken French equal-weighted portfolios
  Panel C: Simulated datasets (one row per DGP)

Statistics are in raw percentage returns (RF not subtracted). Ken French missing-value
sentinels (-99.99) are replaced with NaN. Simulated datasets are in decimals internally
and multiplied by 100 for display consistency with Panels A and B.

Entry point:
    python src/table1_summary_stats.py
Writes table1.tex to src/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_input import load_kenneth_french_portfolios
from simulated_datasets import (
    simulate_iid,
    simulate_regime_shifts,
    simulate_sudden_break,
    simulate_mean_reverting_factor,
    simulate_stochastic_volatility,
    simulate_sudden_covariance_break,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

VW_MARKER = "Average Value Weighted Returns -- Daily"
EW_MARKER = "Average Equal Weighted Returns -- Daily"
FIRMS_MARKER = "Number of Firms in Portfolios"

MISSING_SENTINEL = -99.99  # Ken French fill value for missing observations

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConfig:
    name: str  # Long name used in the table
    label: str  # Short label (e.g. "ind")
    path: Path
    # end markers control which block load_kenneth_french_portfolios extracts
    vw_end_marker: str = EW_MARKER  # ends value-weighted block
    ew_end_marker: Optional[str] = None  # None → read to EOF


PORTFOLIO_CONFIGS: list[PortfolioConfig] = [
    PortfolioConfig(
        name="Industry",
        label="ind",
        path=DATASETS_DIR / "10_Industry_Portfolios_Daily.csv",
    ),
    PortfolioConfig(
        name="Size",
        label="size",
        path=DATASETS_DIR / "10_Portfolios_Formed_on_ME_size.csv",
    ),
    PortfolioConfig(
        name="Book-to-market",
        label="beme",
        path=DATASETS_DIR / "10_Portfolios_Formed_on_booktomarket.csv",
    ),
    PortfolioConfig(
        name="Long-term reversal",
        label="ltr",
        path=DATASETS_DIR / "10_Portfolios_ltr.csv",
    ),
    PortfolioConfig(
        name="Momentum",
        label="mom",
        path=DATASETS_DIR / "10_Portfolios_Prior_momentum.csv",
    ),
    PortfolioConfig(
        name="Short-term reversal",
        label="str",
        path=DATASETS_DIR / "10_Portfolios_Prior_str.csv",
    ),
    PortfolioConfig(
        name="Size and book-to-market",
        label="size-beme",
        path=DATASETS_DIR / "6_Portfolios_size_btm.csv",
        # has extra sections after EW block; must stop before them
        ew_end_marker=FIRMS_MARKER,
    ),
    PortfolioConfig(
        name="Size and long-term reversal",
        label="size-ltr",
        path=DATASETS_DIR / "6_Portfolios_size_ltr.csv",
    ),
    PortfolioConfig(
        name="Size and momentum",
        label="size-mom",
        path=DATASETS_DIR / "6_Portfolios_size_momentum.csv",
    ),
    PortfolioConfig(
        name="Size and short-term reversal",
        label="size-str",
        path=DATASETS_DIR / "6_Portfolios_size_str.csv",
    ),
]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_pct(
    config: PortfolioConfig,
    weighting: str,  # "value" or "equal"
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load one block (value- or equal-weighted) in raw percentage terms.

    load_kenneth_french_portfolios divides by 100 internally; we multiply
    back so that statistics are in percent (matching the paper's table).
    Missing-value sentinels (-99.99) are replaced with NaN before any
    computation.
    """
    if weighting == "value":
        start_marker = VW_MARKER
        end_marker = config.vw_end_marker
    elif weighting == "equal":
        start_marker = EW_MARKER
        end_marker = config.ew_end_marker
    else:
        raise ValueError(
            f"weighting must be 'value' or 'equal', got {weighting!r}"
        )

    df = load_kenneth_french_portfolios(
        path=config.path,
        start_marker=start_marker,
        end_marker=end_marker,
    )

    ret_cols = [c for c in df.columns if c != "Date"]

    # Restore to percent (load_kenneth_french_portfolios divides by 100)
    df[ret_cols] = df[ret_cols] * 100

    # Replace Ken French missing-value sentinel
    df[ret_cols] = df[ret_cols].replace(MISSING_SENTINEL, np.nan)

    if start_date is not None:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Compute Table 1 column values from a raw-% DataFrame with a Date column.

    Returned keys:
        n_assets, sample_size, start_date, end_date,
        lo_mean, hi_mean, lo_std, hi_std, lo_return, hi_return
    """
    ret_cols = [c for c in df.columns if c != "Date"]
    returns = df[ret_cols]

    means = returns.mean()
    stds = returns.std()

    return {
        "n_assets": len(ret_cols),
        "sample_size": len(df),
        "start_date": df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
        "end_date": df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
        "lo_mean": means.min(),
        "hi_mean": means.max(),
        "lo_std": stds.min(),
        "hi_std": stds.max(),
        "lo_return": returns.min().min(),
        "hi_return": returns.max().max(),
    }


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def build_panel(
    configs: list[PortfolioConfig],
    weighting: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame of Table 1 rows for one panel (value or equal weighted)."""
    rows = []
    for cfg in configs:
        df = load_raw_pct(cfg, weighting, start_date, end_date)
        stats = compute_stats(df)
        rows.append({"Name": cfg.name, "Label": cfg.label, **stats})

    panel = pd.DataFrame(rows).rename(
        columns={
            "n_assets": "N",
            "sample_size": "T",
            "start_date": "Start",
            "end_date": "End",
            "lo_mean": "Mean low",
            "hi_mean": "Mean high",
            "lo_std": "Std low",
            "hi_std": "Std high",
            "lo_return": "Return low",
            "hi_return": "Return high",
        }
    )
    return panel


# ---------------------------------------------------------------------------
# Simulated datasets
# ---------------------------------------------------------------------------

SIM_DATA_DIR = DATASETS_DIR / "simulated"


@dataclass
class SimConfig:
    name: str
    label: str
    sim_func: object  # callable: (T, n_assets, seed, **kwargs) -> (portfolios_df, rf_df)
    kwargs: dict  # extra keyword arguments forwarded to sim_func


SIM_CONFIGS: list[SimConfig] = [
    SimConfig(
        name="IID",
        label="iid",
        sim_func=simulate_iid,
        kwargs={},
    ),
    SimConfig(
        name="Regime shifts",
        label="reg",
        sim_func=simulate_regime_shifts,
        kwargs={},
    ),
    SimConfig(
        name="Sudden break",
        label="break",
        sim_func=simulate_sudden_break,
        kwargs={},
    ),
    SimConfig(
        name="Mean reverting",
        label="mr",
        sim_func=simulate_mean_reverting_factor,
        kwargs={},
    ),
    SimConfig(
        name="Covariance break",
        label="cov_break",
        sim_func=simulate_sudden_covariance_break,
        kwargs={},
    ),
    SimConfig(
        name="Stochastic volatility",
        label="stoch_vol",
        sim_func=simulate_stochastic_volatility,
        kwargs={},
    ),
]


def _sim_csv_path(config: SimConfig, T: int, n_assets: int, seed: int) -> Path:
    """Canonical path for a saved simulated dataset."""
    return SIM_DATA_DIR / f"{config.label}_T{T}_N{n_assets}_seed{seed}.csv"


def _sim_rf_csv_path(
    config: SimConfig, T: int, n_assets: int, seed: int
) -> Path:
    """Canonical path for the RF series paired with a simulated dataset."""
    return SIM_DATA_DIR / f"rf_{config.label}_T{T}_N{n_assets}_seed{seed}.csv"


def generate_and_save_sim_datasets(
    configs: list[SimConfig] = SIM_CONFIGS,
    T: int = 6000,
    n_assets: int = 5,
    seed: int = 42,
) -> None:
    """
    Generate all simulated datasets and persist them to CSV.

    CSVs are stored in decimal form (exactly as the sim functions return them)
    so they can also be fed directly into the pipeline. Re-run this function
    whenever T / n_assets / seed change.
    """
    SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        out_path = _sim_csv_path(cfg, T, n_assets, seed)
        rf_out_path = _sim_rf_csv_path(cfg, T, n_assets, seed)
        portfolios_df, rf_df = cfg.sim_func(
            T=T, n_assets=n_assets, seed=seed, **cfg.kwargs
        )
        portfolios_df.to_csv(out_path, index=False)
        rf_df.to_csv(rf_out_path, index=False)
        print(f"  saved {out_path.name}")
        print(f"  saved {rf_out_path.name}")


def load_sim_pct(
    config: SimConfig,
    T: int = 6000,
    n_assets: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a saved simulated dataset and return it in raw percentage terms.

    Raises FileNotFoundError if the CSV has not been generated yet —
    call generate_and_save_sim_datasets() first.
    Sim functions save decimals; multiply by 100 so compute_stats produces
    the same units as the Ken French panels.
    """
    csv_path = _sim_csv_path(config, T, n_assets, seed)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Simulated dataset not found: {csv_path}\n"
            "Run generate_and_save_sim_datasets() to create it."
        )
    df = pd.read_csv(csv_path)
    ret_cols = [c for c in df.columns if c != "Date"]
    df[ret_cols] = df[ret_cols] * 100
    return df


def build_sim_panel(
    configs: list[SimConfig] = SIM_CONFIGS,
    T: int = 6000,
    n_assets: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a tidy DataFrame of Table 1 rows for the simulated-data panel."""
    rows = []
    for cfg in configs:
        df = load_sim_pct(cfg, T=T, n_assets=n_assets, seed=seed)
        stats = compute_stats(df)
        rows.append({"Name": cfg.name, "Label": cfg.label, **stats})

    return pd.DataFrame(rows).rename(
        columns={
            "n_assets": "N",
            "sample_size": "T",
            "start_date": "Start",
            "end_date": "End",
            "lo_mean": "Mean low",
            "hi_mean": "Mean high",
            "lo_std": "Std low",
            "hi_std": "Std high",
            "lo_return": "Return low",
            "hi_return": "Return high",
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Ken French date bounds ---
    START = "1980-01-01"
    END = None

    # --- Simulation parameters ---
    SIM_T = 10000
    SIM_N_ASSETS = 10
    SIM_SEED = 42
    # Set REGENERATE = True to overwrite existing simulated CSVs
    REGENERATE = True

    # Generate simulated CSVs if missing (or forced)
    missing = any(
        not _sim_csv_path(c, SIM_T, SIM_N_ASSETS, SIM_SEED).exists()
        for c in SIM_CONFIGS
    )
    if missing or REGENERATE:
        print("Generating simulated datasets...")
        generate_and_save_sim_datasets(
            T=SIM_T, n_assets=SIM_N_ASSETS, seed=SIM_SEED
        )

    print("Building Panel A (value-weighted)...")
    panel_a = build_panel(PORTFOLIO_CONFIGS, "value", START, END)

    print("Building Panel B (equal-weighted)...")
    panel_b = build_panel(PORTFOLIO_CONFIGS, "equal", START, END)

    print("Building Panel C (simulated)...")
    panel_c = build_sim_panel(T=SIM_T, n_assets=SIM_N_ASSETS, seed=SIM_SEED)

    print("\n--- Panel A ---")
    print(panel_a.to_string(index=False))
    print("\n--- Panel B ---")
    print(panel_b.to_string(index=False))
    print("\n--- Panel C ---")
    print(panel_c.to_string(index=False))

    float_fmt = "{:.3f}".format
    col_fmt = "llrrllrrrrrr"

    latex_a = panel_a.to_latex(
        index=False,
        float_format=float_fmt,
        column_format=col_fmt,
        caption="Daily summary statistics --- Panel A: Ken French value-weighted",
        label="tab:summary_a",
    )
    latex_b = panel_b.to_latex(
        index=False,
        float_format=float_fmt,
        column_format=col_fmt,
        caption="Daily summary statistics --- Panel B: Ken French equal-weighted",
        label="tab:summary_b",
    )
    latex_c = panel_c.to_latex(
        index=False,
        float_format=float_fmt,
        column_format=col_fmt,
        caption=f"Daily summary statistics --- Panel C: Simulated data (T={SIM_T}, N={SIM_N_ASSETS}, seed={SIM_SEED})",
        label="tab:summary_c",
    )

    out = Path(__file__).parent / "table1.tex"
    out.write_text(latex_a + "\n\n" + latex_b + "\n\n" + latex_c)
    print(f"\nLaTeX written to {out}")
