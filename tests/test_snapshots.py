"""Snapshot test: the whole pipeline + evaluation must reproduce the FYP code's outputs.

Reference: tests/snapshots/reference_no_necessity.npz, made from the `dev` branch
(FYP code) with necessity bypassed (see make_snapshots.py). Runs in ~25 s.

When a result-changing fix lands on purpose, check that only the expected keys
differ (run make_snapshots.py --compare), then regenerate the reference from the
fixed code. The .npz files are gitignored, so note in the fix's commit message
that the reference changed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import possibilistic_bayesian

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
REFERENCE = HERE / "snapshots" / "reference_no_necessity.npz"


def _load_make_snapshots():
    spec = importlib.util.spec_from_file_location(
        "make_snapshots", HERE / "snapshots" / "make_snapshots.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = _load_make_snapshots()


@pytest.mark.skipif(
    not ms.DATASETS_DIR.exists(), reason="../datasets not available"
)
@pytest.mark.skipif(
    not REFERENCE.exists(),
    reason=f"{REFERENCE.name} missing (gitignored); regenerate from `dev`: "
    "git archive dev src | tar -x -C <dir> && uv run python "
    "tests/snapshots/make_snapshots.py --src <dir>/src "
    f"--out {REFERENCE.relative_to(HERE.parent)} --bypass-necessity",
)
def test_pipeline_matches_fyp_reference(monkeypatch, capsys):
    monkeypatch.setattr(
        possibilistic_bayesian,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.ones(len(mus)),
    )
    new = ms.build(SRC, bypass_necessity=False)  # bypass already patched above
    capsys.readouterr()  # drop the pipeline's progress prints
    ok = ms.compare(new, REFERENCE)
    report = capsys.readouterr().out
    assert ok, report
