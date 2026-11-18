"""Build (or check) snapshot outputs of the full pipeline on two small slices.

Covers, end to end:
  - data_input: Kenneth French VW + EW blocks, risk-free rate, excess returns
  - pipeline_wrapper.calculate_all_functions_perdatasets: every algorithm + long-only
  - evaluation_pipeline.run_evaluation: Sharpe, CEQ, log-likelihood, Mahalanobis

By default this is the default pipeline: possibilistic with plain possibility
weighting, no necessity (~15 s). Pass --necessity to also run the necessity
schemes (masked / power / exp) with real scores on CPU (joblib, seeded per
time step; ~80 min): a separate reference, e.g. reference.npz.

Runs the installed package (possibilistic_portfolio, `uv sync`), i.e. the
current code. References from older code have to be made on a checkout of
that code (the FYP code on `dev` has the flat src/ layout, before the
package move; this script then needs its pre-move version).

Usage (from dev/):
  # make the reference pytest uses (tests/test_snapshots.py) from the current code
  uv run python tests/snapshots/make_snapshots.py --out tests/snapshots/reference_no_necessity.npz
  # check the current code against it
  uv run python tests/snapshots/make_snapshots.py --compare tests/snapshots/reference_no_necessity.npz
  # the same with real necessity
  uv run python tests/snapshots/make_snapshots.py --necessity --out tests/snapshots/reference.npz
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from possibilistic_portfolio.config import DATASETS_DIR, RF_PATH, SIM_DATA_DIR

T = 1400  # rows per slice
N_ASSETS = 3  # portfolios / assets kept (+ Mkt-RF for the real slice)
# pipeline_wrapper.RunSettings fields (Bayesian keeps its config pool cap)
PARAMS: dict[str, Any] = dict(
    burn_in=300,
    periods_until_investment=100,  # first weights at row 400 -> ~1000 invested days
    theta=1.0,
    possibilistic_max_models=20,
    merge_threshold=0.15,
    nu_bandwidth=50,
    k_neighbours=5,
    gamma=1.0,
    eta=1.0,
    rolling_window_long=252,
    rolling_window_short=63,
    device="cpu",
)
START_DATE = "1980-01-01"


def _frame_to_arrays(prefix: str, df: pd.DataFrame, out: dict) -> None:
    """Store a DataFrame as two arrays, since .npz only holds plain arrays:
    the values, and the column names (which .to_numpy() drops)."""
    out[prefix] = df.to_numpy(dtype=float)
    out[prefix + "__cols"] = np.array([str(c) for c in df.columns])


def build(necessity: bool = False) -> dict[str, np.ndarray]:
    """Load the data, run the full pipeline + evaluation on the `real` and
    `sim` slices, and collect every output into one dict of arrays.
    `necessity` switches the necessity schemes on (real scores, slow)."""
    from possibilistic_portfolio import (
        data_input,
        evaluation_pipeline,
        pipeline_wrapper,
    )

    settings = pipeline_wrapper.RunSettings(**PARAMS, necessity=necessity)
    out: dict[str, np.ndarray] = {}

    # --- data_input, end to end on one real file (both blocks) -------------
    industry = DATASETS_DIR / "10_Industry_Portfolios_Daily.csv"
    loaded = {}
    for name, weighting in [("vw", "value"), ("ew", "equal")]:
        excess = data_input.load_excess_returns(
            industry, RF_PATH, weighting, start_date=START_DATE
        )
        out[f"data_input__{name}__dates"] = (
            excess["Date"].dt.strftime("%Y%m%d").astype(int).to_numpy()
        )
        prepared = data_input.prepare_returns(excess)
        _frame_to_arrays(f"data_input__{name}", prepared, out)
        loaded[name] = prepared
    # Date-indexed, like the returns (matched by date in CEQ)
    rf_df = data_input.load_risk_free(RF_PATH, START_DATE)
    global_rf = rf_df.set_index("Date")["RF"]
    out["data_input__rf"] = global_rf.to_numpy(dtype=float)

    # --- the two pipeline slices --------------------------------------------
    real = loaded["vw"].iloc[:T]
    real = real[list(real.columns[:N_ASSETS]) + ["Mkt-RF"]]
    real_rf = global_rf.iloc[:T]  # same rows as `real` (both from START_DATE)

    sim_dir = SIM_DATA_DIR
    sim = pd.read_csv(sim_dir / "iid_T10000_N10_seed42.csv").iloc[:T, :N_ASSETS]
    sim_rf = pd.read_csv(sim_dir / "rf_iid_T10000_N10_seed42.csv")["RF"].iloc[
        :T
    ]

    with tempfile.TemporaryDirectory() as tmp:
        for name, returns_df, rf in [
            ("real", real, real_rf),
            ("sim", sim, sim_rf),
        ]:
            t0 = time.time()
            folder = pipeline_wrapper.calculate_all_functions_perdatasets(
                returns_df,
                name,
                settings=settings,
                output_dir=tmp,
            )
            evaluation_pipeline.run_evaluation(folder, rf_series=rf, theta=1.0)
            for f in sorted(folder.iterdir()):
                key = f"{name}__{f.stem}"
                if f.suffix == ".npz":  # <algo>_predictive.npz
                    algo = f.stem.removesuffix("_predictive")
                    with np.load(f) as z:
                        out[f"{name}__{algo}_mus"] = z["mu_hat"]
                        out[f"{name}__{algo}_sigmas"] = z["sigma_hat"]
                elif f.suffix == ".csv":
                    _frame_to_arrays(key, pd.read_csv(f, index_col=0), out)
            print(f"[{name}] done in {time.time() - t0:.1f}s")
    return out


def compare(new: dict, ref_path: Path) -> bool:
    """Check a `build` result against a saved reference, key by key.
    Exact match only (NaN == NaN); prints what differs."""
    ref = dict(np.load(ref_path))
    ok = True
    for k in sorted(set(ref) | set(new)):
        if k not in new or k not in ref:
            print(f"  MISSING in {'new' if k not in new else 'reference'}: {k}")
            ok = False
            continue
        a, b = ref[k], new[k]
        if a.dtype.kind in "US":
            same = a.shape == b.shape and (a == b).all()
            detail = ""
        else:
            same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            detail = (
                f" max|diff|={np.nanmax(np.abs(a - b)):.3g}"
                if a.shape == b.shape
                else f" shape {a.shape} vs {b.shape}"
            )
        if not same:
            print(f"  DIFF {k}{detail}")
            ok = False
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--compare", type=Path)
    p.add_argument("--necessity", action="store_true")
    args = p.parse_args()

    result = build(necessity=args.necessity)
    if args.out:
        np.savez_compressed(args.out, **result)
        print(f"saved {len(result)} arrays to {args.out}")
    else:
        sys.exit(0 if compare(result, args.compare) else 1)
