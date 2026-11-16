"""Project-wide paths and constants. Import from scripts, notebooks and tests.

Dataset paths, simulated-data settings and the evaluation scaling so far; run
parameters (burn-in, theta, results folder, ...) move in later (REFACTOR_PLAN).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # dev/
DATASETS_DIR = ROOT.parent / "datasets"
RF_PATH = DATASETS_DIR / "F-F_Research_Data_Factors_daily.csv"

# Simulated datasets (simulated_datasets, table1_summary_stats, pipeline):
# default size and seed of every DGP and of the saved CSVs.
SIM_DATA_DIR = DATASETS_DIR / "simulated"
SIM_T = 10_000
SIM_N_ASSETS = 10
SIM_SEED = 42

# Scaling of the evaluation metrics (evaluation_pipeline): 1 = daily, as in
# Anderson & Cheng (2016); 252 = annualised (Sharpe × √252, turnover × 252).
SCALING_FACTOR = 1
