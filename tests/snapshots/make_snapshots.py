"""Build (or check) snapshot outputs of the full pipeline on two small slices.

Covers, end to end:
  - data_input: Kenneth French VW + EW blocks, risk-free rate, excess returns
  - pipeline_wrapper.calculate_all_functions_perdatasets: all 11 methods + long-only
  - evaluation_pipeline.run_evaluation: Sharpe, CEQ, log-likelihood, Mahalanobis

Necessity runs for real on CPU (joblib, seeded per time step). Pass
--bypass-necessity to replace it by all ones instead: masked / power / exp then
equal plain possibility weighting, and faster_necessity.py is never run.

Usage (from dev/):
  # make reference from another checkout's src (e.g. the FYP code on `dev`)
  uv run python tests/snapshots/make_snapshots.py --src <checkout>/src --out tests/snapshots/reference.npz
  # check the current code against it
  uv run python tests/snapshots/make_snapshots.py --src src --compare tests/snapshots/reference.npz
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"

T = 1400  # rows per slice
N_ASSETS = 3  # portfolios / assets kept (+ Mkt-RF for the real slice)
PARAMS = dict(
    burn_in=300,
    periods_until_investment=100,  # first weights at row 400 -> ~1000 invested days
    theta=1.0,
    max_models=20,
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
VW = "Average Value Weighted Returns -- Daily"
EW = "Average Equal Weighted Returns -- Daily"


def _bypass_necessity(possibilistic_bayesian) -> None:
    possibilistic_bayesian.necessity_scores_fast = lambda y_next, mus, **kw: (
        np.ones(len(mus))
    )


def _frame_to_arrays(prefix: str, df: pd.DataFrame, out: dict) -> None:
    out[prefix] = df.to_numpy(dtype=float)
    out[prefix + "__cols"] = np.array([str(c) for c in df.columns])


def build(src: Path, bypass_necessity: bool = False) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(src.resolve()))
    import data_input
    import evaluation_pipeline
    import pipeline_wrapper
    import possibilistic_bayesian

    if bypass_necessity:
        _bypass_necessity(possibilistic_bayesian)
    out: dict[str, np.ndarray] = {}

    # --- data_input, end to end on one real file (both blocks) -------------
    industry = DATASETS_DIR / "10_Industry_Portfolios_Daily.csv"
    rf_path = DATASETS_DIR / "F-F_Research_Data_Factors_daily.csv"
    loaded = {}
    for name, start, end in [("vw", VW, EW), ("ew", EW, None)]:
        excess = data_input.load_excess_returns_from_kenneth_french_path(
            portfolios_path=industry,
            risk_free_path=rf_path,
            start_date=START_DATE,
            start_marker=start,
            end_marker=end,
        )
        out[f"data_input__{name}__dates"] = (
            excess["Date"].dt.strftime("%Y%m%d").astype(int).to_numpy()
        )
        prepared = data_input.prepare_returns(excess)
        _frame_to_arrays(f"data_input__{name}", prepared, out)
        loaded[name] = prepared
    rf_df = data_input.load_risk_free_rate(rf_path)
    global_rf = data_input.slice_timeframe(rf_df, START_DATE, None)[
        "RF"
    ].reset_index(drop=True)
    out["data_input__rf"] = global_rf.to_numpy(dtype=float)

    # --- the two pipeline slices --------------------------------------------
    real = loaded["vw"].iloc[:T]
    real = real[list(real.columns[:N_ASSETS]) + ["Mkt-RF"]]
    real_rf = global_rf.iloc[:T]

    sim_dir = DATASETS_DIR / "simulated"
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
                dataset=returns_df, dataset_name=name, output_dir=tmp, **PARAMS
            )
            evaluation_pipeline.run_evaluation(folder, rf_series=rf, theta=1.0)
            for f in sorted(folder.iterdir()):
                key = f"{name}__{f.stem}"
                if f.suffix == ".npy":
                    out[key] = np.load(f)
                elif f.suffix == ".csv":
                    _frame_to_arrays(key, pd.read_csv(f, index_col=0), out)
            print(f"[{name}] done in {time.time() - t0:.1f}s")
    return out


def compare(new: dict, ref_path: Path) -> bool:
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
    p.add_argument("--src", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--compare", type=Path)
    p.add_argument("--bypass-necessity", action="store_true")
    args = p.parse_args()

    result = build(args.src, bypass_necessity=args.bypass_necessity)
    if args.out:
        np.savez_compressed(args.out, **result)
        print(f"saved {len(result)} arrays to {args.out}")
    else:
        sys.exit(0 if compare(result, args.compare) else 1)
