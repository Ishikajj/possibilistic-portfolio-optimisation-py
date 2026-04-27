# Possibilistic Portfolio Optimisation

Repository for the final year project on possibilistic portfolio optimisation in Python.

---

## Overview

Standard Bayesian model averaging (BMA) treats uncertainty over *which model is correct* the same way it treats uncertainty over *returns* — as a probability. This project tries to change that by using possibility theory to quantify knowledge based uncertainty. While portfolio returns are genuinely random, our ignorance of the best model is not a matter of chance: it reflects a state of knowledge, not a frequency. Possibility theory provides a cleaner language for that kind of uncertainty.

The project implements a possibilistic analogue of BMA over Normal-Inverse-Wishart (NIW) model pools. The key change is replacing the probabilistic normalisation constant (a sum / integral) with a supremum, yielding possibility distributions instead of probability distributions. Three necessity-based weighting schemes (masked, power, exponential) then convert possibility scores into portfolio weights via geometric aggregation. The possibilistic approach is evaluated against BMA and a set of classical benchmarks across 20 Kenneth French daily return datasets.

---

## Getting started

### 1. Environment

Dependencies are declared in `pyproject.toml`. Install with `uv`:

```bash
uv sync
```

**PyTorch (strongly recommended):** The necessity computation in `faster_necessity.py` benefits substantially from GPU acceleration. Install PyTorch separately, matching your CUDA version, by following the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Without it the code falls back to CPU via `joblib`.

### 2. Datasets

Download the daily return files you want from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and place them in the `datasets/` directory alongside `F-F_Research_Data_Factors_daily.csv` (also from the same library — needed for the risk-free rate). Note that `datasets/` is not inside this repository — it sits as a sibling folder alongside the cloned repo in your local project directory.

### 3. Configure and run the pipeline

Open `src/pipeline_wrapper.py` and edit the `main()` function at the bottom:

- Point `portfolios_paths` to the CSV files you downloaded.
- Set `output_dir` to wherever you want results saved.

Then run:

```bash
uv run src/pipeline_wrapper.py
```

This produces one output folder per dataset (e.g. `results/10_Industry_Portfolios_Daily_value_weighted/`) containing portfolio weights, predictive moments, and a copy of the returns series.

### 4. Evaluate

```bash
uv run src/evaluation_pipeline.py
```

This reads every output folder, computes Sharpe ratios, certainty equivalents, log-likelihoods, and Mahalanobis distances, and writes summary CSVs back into each folder.

### 5. Visualise

Open `analysis.ipynb` and point `RESULTS_DIR` to your results folder. Run top to bottom. This notebook is also the **best entry point** if you just want to inspect results without re-running the pipeline — it covers a theoretical overview, predictive performance, and portfolio outcomes end to end.

---

## Codebase

### Core algorithms

| File | Role |
|---|---|
| `possibilistic_bayesian.py` | Main possibilistic model-averaging engine — supremum normalisation, NIW updates, geometric aggregation |
| `faster_necessity.py` | Parallelised necessity score computation (joblib + optional GPU) — the computationally heavy step |
| `bayesian_averaging.py` | Probabilistic BMA baseline — same NIW pool structure, sum normalisation |

### Benchmarks and helpers

| File | Role |
|---|---|
| `alternative_investment_rules.py` | 1/N, market, min-variance, historical expanding, rolling window, jorion-bayes stein, kan-zhou 3 fund rule |
| `markowitz.py` | Converts predictive moments into unconstrained or long-only Markowitz weights |
| `portfolio_evaluation_functions.py` | Sharpe, rolling Sharpe, portfolio returns, certainty equivalent |
| `distribution_evaluation_functions.py` | Log-likelihood and Mahalanobis distance under NIW Student-*t* predictive |
| `predictive_diagnostics.ipynb` | Exploratory file for result synthesis |

### Pipeline

| File | Role |
|---|---|
| `pipeline_wrapper.py` | Top-level orchestration — runs all algorithms across all datasets, saves outputs |
| `evaluation_pipeline.py` | Loads saved outputs, computes all performance metrics, writes summary CSVs |
| `data_input.py` | Kenneth French CSV loading, excess-return construction, date slicing |

---

## Notes

- A full possibilistic run over ~12 000 trading days takes roughly 90 minutes per dataset. The main controls are `max_models` (pruning) and `merge_threshold` / `nu_bandwidth` (merging).
- `pipeline_wrapper.py` and `evaluation_pipeline.py` are designed to be run in sequence; the evaluation step expects the folder structure that the pipeline step produces.
-Import paths assume you run scripts from the `dev/` root.
