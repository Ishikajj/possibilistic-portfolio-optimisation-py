"""Project-wide paths and constants. Import from scripts, notebooks and tests.

Only the dataset paths live here so far; run parameters (burn-in, theta,
results folder, ...) move in during the package step (REFACTOR_PLAN Part 1).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # dev/
DATASETS_DIR = ROOT.parent / "datasets"
RF_PATH = DATASETS_DIR / "F-F_Research_Data_Factors_daily.csv"
