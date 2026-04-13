import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

assert torch.cuda.is_available(), "CUDA not available — check drivers and torch installation"

from data_input import (
    load_excess_returns_from_kenneth_french_path,
    prepare_returns,
)
from pipeline_wrapper import calculate_all_functions_perdatasets

PORTFOLIOS_PATH = Path("../datasets/6_Portfolios_size_btm.csv")
RF_PATH = Path("../datasets/F-F_Research_Data_Factors_daily.csv")

excess_df = load_excess_returns_from_kenneth_french_path(
    portfolios_path=PORTFOLIOS_PATH,
    risk_free_path=RF_PATH,
    start_date="1980-01-01",
    start_marker="Average Value Weighted Returns -- Daily",
    end_marker="Average Equal Weighted Returns -- Daily",
)
returns_df = prepare_returns(excess_df)

t0 = time.perf_counter()
calculate_all_functions_perdatasets(
    dataset=returns_df,
    dataset_name="6_Portfolios_size_btm_equal_weighted_gpu",
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
    device="cuda",
    output_dir=Path("results"),
)
gpu_time = time.perf_counter() - t0
print(f"\nGPU time: {gpu_time:.1f}s ({gpu_time / 60:.1f} min)")


# ── Comparison

CPU_FOLDER = Path("results/6_Portfolios_size_btm_equal_weighted")
GPU_FOLDER = Path("results/6_Portfolios_size_btm_equal_weighted_gpu")

POSSIBILISTIC_ALGOS = [
    "possibilistic_masked",
    "possibilistic_power",
    "possibilistic_exp",
]


def _rel_error(abs_diff: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Element-wise relative error, ignoring near-zero reference values."""
    denom = np.abs(ref)
    mask = denom > 1e-12
    out = np.full_like(abs_diff, np.nan)
    out[mask] = abs_diff[mask] / denom[mask]
    return out


def compare_weights(algo: str) -> None:
    cpu = pd.read_csv(CPU_FOLDER / f"{algo}_weights.csv", index_col=0)
    gpu = pd.read_csv(GPU_FOLDER / f"{algo}_weights.csv", index_col=0)
    abs_diff = (cpu - gpu).abs().values
    rel = _rel_error(abs_diff, cpu.values)
    print(
        f"  {algo}_weights"
        f"  abs max={abs_diff.max():.2e} mean={abs_diff.mean():.2e}"
        f"  rel max={np.nanmax(rel):.2e} mean={np.nanmean(rel):.2e}"
    )


def compare_npy(algo: str, suffix: str) -> None:
    cpu = np.load(CPU_FOLDER / f"{algo}_{suffix}.npy")
    gpu = np.load(GPU_FOLDER / f"{algo}_{suffix}.npy")
    abs_diff = np.abs(cpu - gpu)
    rel = _rel_error(abs_diff, cpu)
    print(
        f"  {algo}_{suffix}"
        f"  abs max={abs_diff.max():.2e} mean={abs_diff.mean():.2e}"
        f"  rel max={np.nanmax(rel):.2e} mean={np.nanmean(rel):.2e}"
    )


print("\n" + "=" * 60)
print("CPU vs GPU comparison (possibilistic algos only)")
print("=" * 60)
for algo in POSSIBILISTIC_ALGOS:
    print(f"\n{algo}:")
    compare_weights(algo)
    compare_npy(algo, "mus")
    compare_npy(algo, "sigmas")
