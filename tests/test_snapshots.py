"""Snapshot test: the whole pipeline + evaluation must reproduce the saved reference.

Reference: tests/snapshots/reference_no_necessity.npz = the default pipeline
(possibilistic with plain possibility weighting, no necessity) on two small
slices, see make_snapshots.py. Runs in ~15 s. It started as the output of the
FYP code (`dev` branch) with necessity held at 1, and has been regenerated at
every deliberate change since:

- 2026-11-16 avg_turnover (scalars, column 4): NaN rows no longer counted as
  zero turnover. Only that column changed (checked with --compare).
- 2026-11-16 log_likelihood, mahalanobis (+ avg_log_likelihood, avg_mahalanobis
  in scalars): no 1e-10 * I added to Sigma before scoring (max rel. diff 6e-4,
  near-singular benchmark covariances); same NaN pattern.
- 2026-11-16 benchmark timing: minimum variance, historical, rolling (long +
  short), Jorion, Kan-Zhou estimate at t from returns up to and including t
  (were up to t-1). Changed: their forecasts, weights (+ long-only), their
  columns in log_likelihood / mahalanobis / rolling_sharpe and their rows in
  scalars. Bayesian, possibilistic, 1/N, market: bit for bit (checked per
  strategy).
- 2026-11-16 historical: truly expanding (iloc[:t+1]), was a rolling window
  of length start. Only historical_expanding (+ long-only) changed.
- 2026-11-16 possibilistic weights start at `start`, not `burn_in` (forecasts
  unchanged). Only possibilistic unconstrained weights and their Sharpe, CEQ,
  turnover, rolling Sharpe changed.
- 2026-11-16 Bayesian newborn prior divides by t / t-1 (was t-1 / t-2; P4).
  Only bayesian_averaging (+ long-only) changed, slightly.
- 2026-11-16 common evaluation window: every strategy is scored from the
  latest first weight row in the folder (`start`). Bayesian and possibilistic
  forecasts start at `burn_in`, so their warm-up rows are no longer scored:
  only their log_likelihood / mahalanobis columns (100 fewer valid rows, same
  values otherwise) and avg_log_likelihood / avg_mahalanobis changed.
- 2026-11-17, four steps, each checked against the previous reference:
  1. Structure only (shared niw.py, weighting schemes, one merge function):
     bit for bit. The possibilistic strategy is now `possibilistic_plain`
     (= the old masked / power / exp with necessity 1, identical values);
     the three necessity schemes are no longer in this reference.
  2. keep_newest pruning evicts the lowest-ranked model: nothing changed on
     these slices (the case doesn't occur in them; unit-tested instead).
  3. Kan-Zhou: adjusted psi^2 as in Kan & Zhou (2007) eq. 66 (exponent
     -(w-2)/2, beta parameter (w-n+1)/2) and forecast (mu*, Sigma/h) instead
     of (h mu*, Sigma). Only kan_zhou_three_fund (+ long-only) changed.
     Jorion and Kan-Zhou forecasts now also fill the last row (values
     otherwise identical).
  4. Markowitz and minimum variance use solve(Sigma, b) instead of
     inv(Sigma) @ b: unconstrained weights (and the metrics from them) moved
     by at most 3e-13; forecasts and long-only weights bit for bit.
  `possibilistic_diagnostics` has no necessity columns any more.

tests/snapshots/reference.npz is the same run with real necessity
(`make_snapshots.py --necessity`, ~70 min; regenerated 2026-11-17 from the
final code, the FYP-code one is kept as reference_fyp_code_2026-11-13.npz).
Whatever doesn't depend on necessity is identical in the two references.
tests/snapshots/necessity_baseline.py is the cheap check of the necessity
path (~5 min).

When a result-changing fix lands on purpose, check that only the expected keys
differ (run make_snapshots.py --compare), then regenerate the reference from the
fixed code. The .npz files are gitignored, so note in the fix's commit message
that the reference changed.
"""

import importlib.util
from pathlib import Path

import pytest

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
    "current code: uv run python tests/snapshots/make_snapshots.py "
    f"--out {REFERENCE.relative_to(HERE.parent)}",
)
def test_pipeline_matches_reference(capsys):
    new = ms.build()
    capsys.readouterr()  # drop the pipeline's progress prints
    ok = ms.compare(new, REFERENCE)
    report = capsys.readouterr().out
    assert ok, report
