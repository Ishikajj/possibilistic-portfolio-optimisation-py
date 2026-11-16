"""Small possibilistic run WITH real necessity, compared bit for bit.

The pytest snapshot (test_snapshots.py) bypasses necessity to stay fast, and
reference.npz (real necessity, ~80 min) is stale. This covers the necessity
path cheaply (~5 min): 160 rows of 10 Industry VW (3 portfolios), 8 models,
CPU, one joblib worker. Deterministic run to run.

Usage (from dev/):
  uv run python tests/snapshots/necessity_baseline.py --out tests/snapshots/necessity_baseline.npz
  uv run python tests/snapshots/necessity_baseline.py --compare tests/snapshots/necessity_baseline.npz

necessity_baseline.npz (gitignored) was made 2026-11-16 just before the
package move (commit 516b013) and matched after it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from possibilistic_portfolio import config, data_input
from possibilistic_portfolio import possibilistic_bayesian as pb

T, N_ASSETS = 160, 3


def build() -> dict[str, np.ndarray]:
    returns = data_input.prepare_returns(
        data_input.load_excess_returns(
            config.DATASETS_DIR / "10_Industry_Portfolios_Daily.csv",
            config.RF_PATH,
            "value",
            start_date="1980-01-01",
        )
    ).iloc[:T, :N_ASSETS]
    t0 = time.time()
    diag, m, w_m, p, w_p, e, w_e = pb.run_core(
        returns, start=60, burn_in=40, max_models=8, n_jobs=1, device="cpu"
    )
    print(f"run_core with real necessity: {time.time() - t0:.1f}s")
    out = {"returns": returns.to_numpy()}
    for name, pred, w in [
        ("masked", m, w_m),
        ("power", p, w_p),
        ("exp", e, w_e),
    ]:
        out[f"{name}_mu"] = pred["mu_hat"]
        out[f"{name}_sigma"] = pred["sigma_hat"]
        out[f"{name}_weights"] = w.to_numpy()
    num = diag.select_dtypes("number")
    out["diag"] = num.to_numpy(dtype=float)
    out["diag_cols"] = np.array(list(num.columns))
    return out


def compare(new: dict, ref_path: str) -> bool:
    """Exact match only (NaN == NaN); prints the keys that differ."""
    ref = dict(np.load(ref_path))
    ok = set(ref) == set(new)
    for k in sorted(set(ref) & set(new)):
        a, b = ref[k], new[k]
        same = a.shape == b.shape and (
            bool((a == b).all())
            if a.dtype.kind in "US"
            else np.array_equal(a, b, equal_nan=True)
        )
        if not same:
            print("  DIFF", k)
            ok = False
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out")
    g.add_argument("--compare")
    args = ap.parse_args()
    result = build()
    if args.out:
        np.savez_compressed(args.out, **result)
        print(f"saved {len(result)} arrays to {args.out}")
    else:
        sys.exit(0 if compare(result, args.compare) else 1)
