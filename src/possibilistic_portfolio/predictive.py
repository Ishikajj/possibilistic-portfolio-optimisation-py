"""Predictive moments ("forecasts"): the dict format and its file format.

A forecast dict has, for every row t of the returns it was built from:
    "mu_hat"    (T, n)     predictive mean for day t+1
    "sigma_hat" (T, n, n)  predictive covariance for day t+1
    "index"     (T,)       the returns' index (dates; row numbers for simulated data)
Rows without a forecast (before it starts) are NaN.

Timing convention (the whole pipeline):
    Everything is stored at row t = the day it is DECIDED, not the day it is for.
    - forecast row t  uses returns up to and including t, predicts t+1
    - weights row t   come from forecast row t (same row), are held over t+1
    - evaluation      pairs row t with the returns at t+1. This +1 shift lives
                      in exactly two places: portfolio_returns
                      (weights.shift(1)) and _gaussian_terms (returns[1:] vs
                      mu_hat[:-1]).
    Never use t+1 when building row t. tests/test_timing.py checks this for
    every algorithm.
check_aligned only checks that forecast and returns are the same rows (same
labels, same order); the timing above is a convention, enforced by those tests.

StrategyOutput   — what every algorithm returns: weights + forecast (None for
                   weight-only strategies).
warn_nan_rows    — the one warning an algorithm gives for rows it had to leave
                   NaN after it started (counts per reason).
make_predictive  — build the dict; every algorithm uses it, so "index" is always set.
check_aligned    — raise unless the forecast's index equals the returns' index.
save_predictive  — write <algo>_predictive.npz (mu_hat, sigma_hat, index) to a run folder.
load_predictive  — read it back; falls back to the old <algo>_mus.npy / _sigmas.npy
                   (runs before 2026-11-16, no index: matched by position only).
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StrategyOutput:
    """One strategy's result: weights indexed like the returns and, for
    predictive strategies, the forecast dict (``make_predictive``)."""

    weights: pd.DataFrame
    forecast: dict | None = None


def warn_nan_rows(
    source: str, skipped: dict[str, int], n_rows: int, what: str = "rows"
) -> None:
    """Warn once that `source` left rows NaN, with the count per reason.

    Rows that are NaN by design (before the start, the last weights row) are
    not the caller's business here; `skipped` counts only rows that should
    have had a value. Nothing happens when all counts are zero.
    """
    total = sum(skipped.values())
    if total == 0:
        return
    reasons = ", ".join(f"{k}: {v}" for k, v in skipped.items() if v)
    warnings.warn(
        f"{source}: {total} of {n_rows} {what} left NaN ({reasons})",
        RuntimeWarning,
        stacklevel=3,  # the caller of the algorithm
    )


def make_predictive(
    mu_hat: np.ndarray, sigma_hat: np.ndarray, index: pd.Index
) -> dict:
    """Bundle predictive moments with the returns index they belong to."""
    if not len(mu_hat) == len(sigma_hat) == len(index):
        raise ValueError(
            f"mu_hat, sigma_hat and index must have the same length, got "
            f"{len(mu_hat)}, {len(sigma_hat)}, {len(index)}"
        )
    return {"mu_hat": mu_hat, "sigma_hat": sigma_hat, "index": index}


def check_aligned(predictive: dict, index: pd.Index) -> None:
    """Raise ValueError unless the forecast belongs to returns with this index.

    Dicts without "index" (built by hand, or loaded from an old run) can't be
    checked and are matched by position, as before.
    """
    pred_index = predictive.get("index")
    if pred_index is None:
        return
    pred_index = pd.Index(pred_index)
    if pred_index.equals(index):
        return
    if len(pred_index) != len(index):
        detail = f"{len(pred_index)} forecast rows vs {len(index)} return rows"
    else:
        i = int(np.flatnonzero(pred_index != index)[0])
        detail = f"row {i}: forecast {pred_index[i]} vs returns {index[i]}"
    raise ValueError(f"forecast index does not match the returns ({detail})")


def save_predictive(
    folder: Path | str, algo_name: str, predictive: dict
) -> None:
    """Write <algo_name>_predictive.npz (mu_hat, sigma_hat, index)."""
    np.savez(
        Path(folder) / f"{algo_name}_predictive.npz",
        mu_hat=predictive["mu_hat"],
        sigma_hat=predictive["sigma_hat"],
        index=np.asarray(predictive["index"]),
    )


def load_predictive(folder: Path | str, algo_name: str) -> dict | None:
    """Read a saved forecast; None if the algorithm saved none (weights only)."""
    folder = Path(folder)
    npz = folder / f"{algo_name}_predictive.npz"
    if npz.exists():
        with np.load(npz) as z:
            return make_predictive(
                z["mu_hat"], z["sigma_hat"], pd.Index(z["index"])
            )
    mus, sigmas = (
        folder / f"{algo_name}_mus.npy",
        folder / f"{algo_name}_sigmas.npy",
    )
    if mus.exists() and sigmas.exists():  # old run: no index saved
        return {"mu_hat": np.load(mus), "sigma_hat": np.load(sigmas)}
    return None
