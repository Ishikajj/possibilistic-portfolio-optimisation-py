"""Table 1 – Daily summary statistics for Kenneth French and simulated datasets.

Produces three panels:
  Panel A: Ken French value-weighted portfolios
  Panel B: Ken French equal-weighted portfolios
  Panel C: Simulated datasets (one row per DGP)

Statistics are in raw percentage returns (RF not subtracted). Ken French missing-value
codes (-99.99, -999) are replaced with NaN. Simulated datasets are in decimals internally
and multiplied by 100 for display consistency with Panels A and B.

Entry point:
    uv run python -m possibilistic_portfolio.table1_summary_stats
Writes table1.tex to config.FIGURES_DIR (figures/, gitignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from possibilistic_portfolio.config import (
    DATASETS_DIR,
    FIGURES_DIR,
    SIM_N_ASSETS,
    SIM_SEED,
    SIM_T,
)
from possibilistic_portfolio.data_input import load_portfolios
from possibilistic_portfolio.simulated_datasets import (
    generate_and_save_sim_datasets,
    sim_paths,
)

# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConfig:
    name: str  # Long name used in the table
    label: str  # Short label (e.g. "ind")
    path: Path


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
    """Load one block (value- or equal-weighted) in raw percentage terms.

    Missing-value codes (-99.99, -999) are NaN (handled in data_input).
    """
    return load_portfolios(
        config.path, weighting, start_date, end_date, percent=True
    )


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
        "start_date": df["Date"].min().strftime("%Y-%m-%d")
        if "Date" in df.columns
        else "N/A",
        "end_date": df["Date"].max().strftime("%Y-%m-%d")
        if "Date" in df.columns
        else "N/A",
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

# Table name per simulated DGP (keys = simulated_datasets.DGPS labels).
SIM_NAMES: dict[str, str] = {
    "iid": "IID",
    "reg": "Regime shifts",
    "break": "Sudden break",
    "mr": "Mean reverting",
    "cov_break": "Covariance break",
    "stoch_vol": "Stochastic volatility",
}


def load_sim_pct(
    label: str,
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load a saved simulated dataset and return it in raw percentage terms.

    Raises FileNotFoundError if the CSV has not been generated yet —
    call simulated_datasets.generate_and_save_sim_datasets() first.
    Sim functions save decimals; multiply by 100 so compute_stats produces
    the same units as the Ken French panels.
    """
    csv_path, _ = sim_paths(label, T, n_assets, seed, data_dir)
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
    labels: list[str] | None = None,
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Table 1 rows for the simulated-data panel (``None`` = all DGPs)."""
    rows = []
    for label in SIM_NAMES if labels is None else labels:
        df = load_sim_pct(label, T, n_assets, seed, data_dir)
        stats = compute_stats(df)
        rows.append({"Name": SIM_NAMES[label], "Label": label, **stats})

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

    # Simulated data: size and seed from config.SIM_T / SIM_N_ASSETS / SIM_SEED.
    # Set REGENERATE = True to overwrite existing simulated CSVs
    REGENERATE = True

    # Generate simulated CSVs if missing (or forced)
    missing = any(not sim_paths(label)[0].exists() for label in SIM_NAMES)
    if missing or REGENERATE:
        print("Generating simulated datasets...")
        generate_and_save_sim_datasets(list(SIM_NAMES))

    print("Building Panel A (value-weighted)...")
    panel_a = build_panel(PORTFOLIO_CONFIGS, "value", START, END)

    print("Building Panel B (equal-weighted)...")
    panel_b = build_panel(PORTFOLIO_CONFIGS, "equal", START, END)

    print("Building Panel C (simulated)...")
    panel_c = build_sim_panel()

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

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "table1.tex"
    out.write_text(latex_a + "\n\n" + latex_b + "\n\n" + latex_c)
    print(f"\nLaTeX written to {out}")
