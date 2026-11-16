"""Project-wide paths and constants. Import from scripts, notebooks and tests.

Dataset paths, simulated-data settings, algorithm defaults and the evaluation
scaling. The results folder and run lists move in with the pipeline (2.10).
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

# ── Algorithm defaults ─────────────────────────────────────────────────────
# One place for the defaults of every algorithm (alternative_investment_rules,
# markowitz, bayesian_averaging, possibilistic_bayesian) and pipeline_wrapper.
# Values = the FYP runs (pipeline_wrapper.main). Algorithms take
# (returns_df, *, start, ...): everything after returns_df is keyword-only,
# `start` (first row with weights) is always passed explicitly.
BURN_IN = 500  # rows that only set the averaging methods' prior
PERIODS_UNTIL_INVESTMENT = 500  # start = BURN_IN + PERIODS_UNTIL_INVESTMENT
THETA = 1.0  # risk aversion (Markowitz weights of every predictive strategy)
ROLLING_WINDOW_LONG = 252  # rolling (long), min-variance, Jorion, Kan-Zhou
ROLLING_WINDOW_SHORT = 63  # rolling (short)

# Model pools (both averaging methods)
KEEP_NEWEST = True  # pruning always keeps the newborn model

# Bayesian model averaging
BAYESIAN_MAX_MODELS = 2000
BAYESIAN_PRUNE_THRESHOLD = 1e-6

# Possibilistic model averaging
POSSIBILISTIC_MAX_MODELS = 100
MERGE_THRESHOLD = 0.15  # Hellinger distance below which models merge
NU_BANDWIDTH = 50  # max |nu_i - nu_j| for merge candidates
K_NEIGHBOURS = 5  # forward neighbours scanned per model when merging

# Necessity (possibilistic only; kept until the necessity decision)
NECESSITY_GAMMA = 1.0  # power weighting exponent
NECESSITY_ETA = 1.0  # exponential weighting decay
NECESSITY_DEVICE = "cpu"  # "cuda" needs PyTorch
NECESSITY_N_JOBS = 6  # joblib workers on CPU

# Scaling of the evaluation metrics (evaluation_pipeline): 1 = daily, as in
# Anderson & Cheng (2016); 252 = annualised (Sharpe × √252, turnover × 252).
SCALING_FACTOR = 1
