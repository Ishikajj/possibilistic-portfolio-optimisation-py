"""Boxplot of portfolio weight distributions pooled across all result folders.

One box per strategy, weights pooled across all datasets (KF + simulated).
Whiskers at 1.5x IQR, fliers subsampled for rendering performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_PATH = Path(__file__).resolve().parent / "weight_distributions.png"

# (algo_name, display_label, group) — group drives colour
STRATEGIES = [
    ("bayesian_averaging", "Bayesian MA", "bayesian"),
    ("possibilistic_masked", "Poss. Masked", "possibilistic"),
    ("possibilistic_power", "Poss. Power", "possibilistic"),
    ("possibilistic_exp", "Poss. Exp", "possibilistic"),
]

GROUP_COLOURS = {
    "benchmark": "#7f7f7f",
    "frequentist": "#4C72B0",
    "shrinkage": "#C44E52",
    "bayesian": "#55A868",
    "possibilistic": "#DD8452",
}

MAX_FLIERS = 20000  # per strategy — subsampled for rendering speed
RNG = np.random.default_rng(42)


def collect_weights(results_dir: Path) -> dict[str, np.ndarray]:
    """Pool all weight values (T x n flattened) per strategy across all folders."""
    pooled: dict[str, list[np.ndarray]] = {s[0]: [] for s in STRATEGIES}
    algo_names = {s[0] for s in STRATEGIES}

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        for algo_name in algo_names:
            path = folder / f"{algo_name}_weights.csv"
            if not path.exists():
                continue
            vals = (
                pd.read_csv(path, index_col=0).to_numpy(dtype=float).flatten()
            )
            vals = vals[~np.isnan(vals)]
            if len(vals):
                pooled[algo_name].append(vals)

    return {
        name: np.concatenate(arrays) if arrays else np.array([])
        for name, arrays in pooled.items()
    }


def _boxplot_stats(vals: np.ndarray) -> dict:
    """Compute boxplot statistics and return a bxp-compatible dict."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr

    non_outliers = vals[(vals >= lo_fence) & (vals <= hi_fence)]
    whislo = float(non_outliers.min()) if len(non_outliers) else float(q1)
    whishi = float(non_outliers.max()) if len(non_outliers) else float(q3)

    fliers = vals[(vals < lo_fence) | (vals > hi_fence)]
    if len(fliers) > MAX_FLIERS:
        fliers = RNG.choice(fliers, size=MAX_FLIERS, replace=False)

    return {
        "med": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "whislo": whislo,
        "whishi": whishi,
        "fliers": fliers,
        "mean": float(vals.mean()),
        "label": "",
    }


def plot(pooled: dict[str, np.ndarray]) -> None:
    stats_list = []
    labels = []
    colours = []

    for algo_name, label, group in STRATEGIES:
        vals = pooled[algo_name]
        if not len(vals):
            print(f"  [SKIP] no data for {algo_name}")
            continue
        stats_list.append(_boxplot_stats(vals))
        labels.append(label)
        colours.append(GROUP_COLOURS[group])

    fig, ax = plt.subplots(figsize=(15, 6))

    bp = ax.bxp(
        stats_list,
        showfliers=True,
        patch_artist=True,
        flierprops=dict(
            marker=".", markersize=1.5, alpha=0.15, linestyle="none"
        ),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    for fliers, colour in zip(bp["fliers"], colours):
        fliers.set_color(colour)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Weight Distributions — All Datasets & Strategies")

    # Legend for groups
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=c, alpha=0.75, label=g.capitalize())
        for g, c in GROUP_COLOURS.items()
        if any(s[2] == g for s in STRATEGIES)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    print("Collecting weights...")
    pooled = collect_weights(RESULTS_DIR)
    for algo_name, _, _ in STRATEGIES:
        n = len(pooled[algo_name])
        print(f"  {algo_name}: {n:,} values")
    print("Plotting...")
    plot(pooled)
