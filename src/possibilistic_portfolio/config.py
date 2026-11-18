"""Project-wide paths and constants. Import from scripts, notebooks and tests:
``from possibilistic_portfolio import config``.

Every path in the code is built from the constants here (no relative paths,
so scripts work from any working directory), plus simulated-data settings,
algorithm defaults and the evaluation scaling.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # dev/ (this file: dev/src/<pkg>/)
DATASETS_DIR = ROOT.parent / "datasets"  # sibling of dev/, not in the repo
RF_PATH = DATASETS_DIR / "F-F_Research_Data_Factors_daily.csv"
RESULTS_DIR = ROOT / "results"  # default pipeline output (one folder/dataset)
FIGURES_DIR = ROOT / "figures"  # regenerable plots / LaTeX tables (gitignored)

# Kenneth French portfolio files run by pipeline_wrapper.main (in DATASETS_DIR)
FRENCH_PORTFOLIO_FILES = [
    "6_Portfolios_size_momentum.csv",
    "10_Portfolios_ltr.csv",
    "6_Portfolios_size_str.csv",
    "10_Portfolios_Prior_momentum.csv",
    "10_Portfolios_Prior_str.csv",
    "10_Industry_Portfolios_Daily.csv",
    "10_Portfolios_Formed_on_booktomarket.csv",
    "10_Portfolios_Formed_on_ME_size.csv",
    "6_Portfolios_size_ltr.csv",
    "6_Portfolios_size_btm.csv",
]

# Algorithms run by pipeline_wrapper.main / main_simulated: None = all, or
# names from pipeline_wrapper.ALGORITHMS (`uv run python -m
# possibilistic_portfolio.pipeline_wrapper --help` lists them), e.g.
# ["bayesian_averaging", "possibilistic"]. --algorithms overrides this.
RUN_ALGORITHMS: list[str] | None = None

# Date window of the Kenneth French runs (YYYY-MM-DD, inclusive); END_DATE
# None = up to the last date in the files.
START_DATE: str | None = "1980-01-01"
END_DATE: str | None = None

# Simulated datasets (simulated_datasets, table1_summary_stats, pipeline):
# default size and seed of every DGP and of the saved CSVs.
SIM_DATA_DIR = DATASETS_DIR / "simulated"
SIM_T = 10_000
SIM_N_ASSETS = 10
SIM_SEED = 42

# ── Algorithm defaults ─────────────────────────────────────────────────────
# One place for the defaults of every algorithm (alternative_investment_rules,
# weight_rules, bayesian_averaging, possibilistic_bayesian) and pipeline_wrapper.
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

# Necessity (possibilistic only). Optional and off by default: plain
# possibility weighting is the default scheme. Switched on, the possibilistic
# run also returns the masked / power / exp schemes (slow: ~90 min per dataset).
USE_NECESSITY = False
NECESSITY_GAMMA = 1.0  # power weighting exponent
NECESSITY_ETA = 1.0  # exponential weighting decay
NECESSITY_DEVICE = "cpu"  # "cuda" needs PyTorch
NECESSITY_N_JOBS = 6  # joblib workers on CPU

# Scaling of the evaluation metrics (evaluation_pipeline): 1 = daily, as in
# Anderson & Cheng (2016); 252 = annualised (Sharpe × √252, turnover × 252).
SCALING_FACTOR = 1
