# Possibilistic Portfolio Optimisation

Repository for the final year project on possibilistic portfolio optimisation in Python.

---

## Overview

Standard Bayesian model averaging (BMA) treats uncertainty over *which model is correct* the same way it treats uncertainty over *returns* — as a probability. This project tries to change that by using possibility theory to quantify knowledge based uncertainty. While portfolio returns are genuinely random, our ignorance of the best model is not a matter of chance: it reflects a state of knowledge, not a frequency. Possibility theory provides a cleaner language for that kind of uncertainty.

The project implements a possibilistic analogue of BMA over Normal-Inverse-Wishart (NIW) model pools. The key change is replacing the probabilistic normalisation constant (a sum / integral) with a supremum, yielding possibility distributions instead of probability distributions. The models are then pooled geometrically, weighted by their possibilities (plain possibility weighting, the default), and the pooled forecast goes to Markowitz. Three necessity-based weighting schemes (masked, power, exponential) are available as an option. The possibilistic approach is evaluated against BMA and a set of classical benchmarks across 20 Kenneth French daily return datasets.

---

## Getting started

### 1. Environment

Dependencies are declared in `pyproject.toml`. Install with `uv`:

```bash
uv sync
```

**PyTorch (optional):** only needed for the necessity schemes on a GPU (`--necessity` with `device="cuda"`). The default run doesn't compute necessity, and necessity on CPU uses `joblib`. To install PyTorch, match your CUDA version following [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 2. Datasets

Download the daily return files you want from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and place them in the `datasets/` directory alongside `F-F_Research_Data_Factors_daily.csv` (also from the same library — needed for the risk-free rate). Note that `datasets/` is not inside this repository — it sits as a sibling folder alongside the cloned repo in your local project directory.

### 3. Configure and run the pipeline

Paths and default settings live in `src/possibilistic_portfolio/config.py`:

- `FRENCH_PORTFOLIO_FILES` lists the Kenneth French CSV files to run (looked up in `DATASETS_DIR`).
- `START_DATE` / `END_DATE` set the date window of the Kenneth French runs (`END_DATE = None`: up to the last date in the files).
- `RUN_ALGORITHMS` selects the algorithms (`None` = all).
- `RESULTS_DIR` is the default run folder (`results/`).
- The algorithm defaults (burn-in, windows, model-pool sizes, risk aversion, `USE_NECESSITY`, ...) are the run settings (`pipeline_wrapper.RunSettings`).

Then run:

```bash
uv run python -m possibilistic_portfolio.pipeline_wrapper                 # Kenneth French datasets
uv run python -m possibilistic_portfolio.pipeline_wrapper --simulated     # simulated datasets
```

Options: `--out <folder>` (run folder, default `results/`), `--algorithms <names>` (only some algorithms, e.g. `--algorithms bayesian_averaging possibilistic`; overrides `RUN_ALGORITHMS`), `--necessity` (also run the three necessity schemes; slow), `--resume` (see below). `--help` lists the algorithm names.

This produces one output folder per dataset (e.g. `results/10_Industry_Portfolios_Daily_value_weighted/`) containing portfolio weights, predictive moments, and a copy of the returns series. Each algorithm's files are saved as soon as it finishes. The run folder also gets:

- `config.json`: the settings used and, per call, the code version (git commit), the datasets and algorithms, which datasets finished or had failed algorithms, and the end time and duration (a call without an end time didn't finish).
- `run.log`: the log with timestamps (also shown in the terminal), including warnings and the time each algorithm took.

An algorithm that fails is logged with its traceback and the run continues with the others; the end of the log lists every failure.

Existing results are never overwritten: run into a new folder (`--out`) or rename the old one. To continue a run that stopped (crash, Ctrl-C), run the same command again with `--resume`: datasets and algorithms already done are skipped, and their files aren't touched.

```bash
uv run python -m possibilistic_portfolio.pipeline_wrapper --out results_new --resume
```

### 4. Evaluate

```bash
uv run python -m possibilistic_portfolio.evaluation_pipeline              # results/
uv run python -m possibilistic_portfolio.evaluation_pipeline results_new  # another run folder
```

This reads every dataset folder of the run, computes Sharpe ratios, certainty equivalents, log-likelihoods, Mahalanobis distances and turnover, and writes them back into each dataset folder: `scalars.csv` (one row per strategy), `rolling_sharpe.csv`, `log_likelihood.csv` and `mahalanobis.csv`. Evaluating again overwrites these four files (they are recomputed from the saved weights). `--help` shows the usage.

### 5. Visualise

Open `analysis.ipynb` and point `RESULTS_DIR` to your results folder. Run top to bottom. This notebook is also the **best entry point** if you just want to inspect results without re-running the pipeline — it covers a theoretical overview, predictive performance, and portfolio outcomes end to end.

---

## Codebase

### Core algorithms

| File | Role |
|---|---|
| `possibilistic_bayesian.py` | Main possibilistic model-averaging engine — supremum normalisation, NIW updates, geometric aggregation, weighting schemes |
| `bayesian_averaging.py` | Probabilistic BMA baseline — same NIW pool structure, sum normalisation |
| `niw.py` | What the two methods share: newborn prior, NIW update, burn-in, top-models selection for pruning |
| `faster_necessity.py` | Optional: parallelised necessity score computation (joblib + optional GPU) for the masked / power / exp schemes — computationally heavy |

### Benchmarks and helpers

| File | Role |
|---|---|
| `alternative_investment_rules.py` | Benchmark forecasts (predictive moments): historical expanding, rolling window, Jorion Bayes-Stein, Kan-Zhou three-fund |
| `weight_rules.py` | Everything that makes weights: Markowitz (unconstrained, long-only) from any forecast, and the weight-only benchmarks 1/N, market, minimum variance |
| `portfolio_evaluation_functions.py` | Sharpe, rolling Sharpe, portfolio returns, certainty equivalent |
| `distribution_evaluation_functions.py` | Log-likelihood and Mahalanobis distance of the realised returns under each forecast (scored as a Gaussian) |
| `predictive.py` | The forecast format every algorithm returns and its file format; timing convention |
| `predictive_diagnostics.ipynb` | Exploratory file for result synthesis |

### Pipeline

| File | Role |
|---|---|
| `pipeline_wrapper.py` | Top-level orchestration — runs all (or selected) algorithms across all datasets, saves outputs and the run's `config.json` |
| `evaluation_pipeline.py` | Loads saved outputs, computes all performance metrics, writes summary CSVs |
| `data_input.py` | Kenneth French CSV loading, excess-return construction, date slicing |

---

## Notes

- Runtime: with the defaults the possibilistic method takes about 20 seconds per dataset (~11 600 trading days, 11 assets, laptop CPU). Merging similar models (off by default; `merge_threshold`) makes it about 3 minutes, and `--necessity` roughly 90 minutes. The main controls are `max_models` (pruning) and, when merging is on, `merge_threshold` / `nu_bandwidth`.
- `pipeline_wrapper.py` and `evaluation_pipeline.py` are designed to be run in sequence; the evaluation step expects the folder structure that the pipeline step produces.
- The code is an installable package (`src/possibilistic_portfolio/`, installed by `uv sync`). Import modules as `from possibilistic_portfolio import possibilistic_bayesian`; all paths come from `config.py`, so scripts work from any directory.
- Tests: `uv run pytest`.
