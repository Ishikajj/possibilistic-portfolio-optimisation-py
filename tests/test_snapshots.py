"""Snapshot test: the whole pipeline + evaluation must reproduce the FYP code's outputs.

Reference: tests/snapshots/reference_no_necessity.npz, made from the `dev` branch
(FYP code) with necessity bypassed (see make_snapshots.py). Runs in ~25 s.

Deliberate differences from the FYP code (reference regenerated from post-fyp):
- 2026-11-16 avg_turnover (scalars, column 4): NaN rows no longer counted as
  zero turnover. Only that column changed (checked with --compare).
  tests/snapshots/reference.npz (real necessity) still has the old values.
- 2026-11-16 log_likelihood, mahalanobis (+ avg_log_likelihood, avg_mahalanobis
  in scalars): no 1e-10 * I added to Sigma before scoring (max rel. diff 6e-4,
  near-singular benchmark covariances); same NaN pattern. reference.npz too.
- 2026-11-16 benchmark timing: minimum variance, historical, rolling (long +
  short), Jorion, Kan-Zhou estimate at t from returns up to and including t
  (were up to t-1). Changed: their forecasts, weights (+ long-only), their
  columns in log_likelihood / mahalanobis / rolling_sharpe and their rows in
  scalars. Bayesian, possibilistic, 1/N, market: bit for bit (checked per
  strategy). reference.npz (real necessity) still has the pre-fix values.
- 2026-11-16 historical: truly expanding (iloc[:t+1]), was a rolling window
  of length start. Only historical_expanding (+ long-only) changed.
- 2026-11-16 possibilistic weights start at `start`, not `burn_in` (forecasts
  unchanged). Only possibilistic unconstrained weights and their Sharpe, CEQ,
  turnover, rolling Sharpe changed. reference.npz still pre-fix.
- 2026-11-16 Bayesian newborn prior divides by t / t-1 (was t-1 / t-2; P4).
  Only bayesian_averaging (+ long-only) changed, slightly.
- 2026-11-16 common evaluation window: every strategy is scored from the
  latest first weight row in the folder (`start`). Bayesian and possibilistic
  forecasts start at `burn_in`, so their warm-up rows are no longer scored:
  only their log_likelihood / mahalanobis columns (100 fewer valid rows, same
  values otherwise) and avg_log_likelihood / avg_mahalanobis changed.

When a result-changing fix lands on purpose, check that only the expected keys
differ (run make_snapshots.py --compare), then regenerate the reference from the
fixed code. The .npz files are gitignored, so note in the fix's commit message
that the reference changed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from possibilistic_portfolio import possibilistic_bayesian

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "snapshots" / "reference_no_necessity.npz"


def _load_make_snapshots():
    spec = importlib.util.spec_from_file_location(
        "make_snapshots", HERE / "snapshots" / "make_snapshots.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = _load_make_snapshots()


@pytest.mark.skipif(
    not ms.DATASETS_DIR.exists(), reason="../datasets not available"
)
@pytest.mark.skipif(
    not REFERENCE.exists(),
    reason=f"{REFERENCE.name} missing (gitignored); regenerate from the "
    "current code (`dev` + the fixes in the module docstring): uv run python "
    "tests/snapshots/make_snapshots.py "
    f"--out {REFERENCE.relative_to(HERE.parent)} --bypass-necessity",
)
def test_pipeline_matches_fyp_reference(monkeypatch, capsys):
    monkeypatch.setattr(
        possibilistic_bayesian,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.ones(len(mus)),
    )
    new = ms.build(bypass_necessity=False)  # bypass already patched above
    capsys.readouterr()  # drop the pipeline's progress prints
    ok = ms.compare(new, REFERENCE)
    report = capsys.readouterr().out
    assert ok, report
