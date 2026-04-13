"""Compute long-only Markowitz weights from saved predictives.

For every strategy in every results folder that has *_mus.npy / *_sigmas.npy,
computes long-only weights and saves them as *_longonly_weights.csv.
The evaluation pipeline will pick these up automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from markowitz import markowitz_long_only

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Must match what was used in the original pipeline run
BURN_IN = 500
PERIODS_UNTIL_INVESTMENT = 500
THETA = 1.0


def process_folder(folder: Path) -> None:
    returns_path = folder / "returns.csv"
    if not returns_path.exists():
        print(f"  [SKIP] no returns.csv in {folder.name}")
        return

    returns_df = pd.read_csv(returns_path, index_col=0)

    for mus_path in sorted(folder.glob("*_mus.npy")):
        algo_name = mus_path.name.replace("_mus.npy", "")
        sigmas_path = folder / f"{algo_name}_sigmas.npy"

        if not sigmas_path.exists():
            print(f"  [SKIP] no sigmas for {algo_name} in {folder.name}")
            continue

        out_path = folder / f"{algo_name}_longonly_weights.csv"
        if out_path.exists():
            print(f"  [SKIP] {algo_name}_longonly_weights.csv already exists")
            continue

        print(f"  {algo_name}...", end=" ", flush=True)
        mu_hat = np.load(mus_path)
        sigma_hat = np.load(sigmas_path)

        weights_df = markowitz_long_only(
            mu_sigma_dict={"mu_hat": mu_hat, "sigma_hat": sigma_hat},
            returns_df=returns_df,
            burn_in=BURN_IN,
            periods_until_investment=PERIODS_UNTIL_INVESTMENT,
            theta=THETA,
        )
        weights_df.to_csv(out_path)
        print("done")


if __name__ == "__main__":
    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        print(f"\n{folder.name}")
        process_folder(folder)
