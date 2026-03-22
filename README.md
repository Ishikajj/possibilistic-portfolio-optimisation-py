# Possibilistic Portfolio Optimisation in Python

## Overview

This project studies portfolio construction under **estimation uncertainty** and implements both **probabilistic (Bayesian)** and **possibilistic model-averaging frameworks**.

### Motivation

Standard Bayesian model averaging approaches can exhibit instability and performance decay under rolling estimation. This repository introduces a **possibilistic analogue using supremum normalization** and compares it empirically against classical and shrinkage-based portfolio strategies.

### Key Contributions

- Dynamic Bayesian model averaging over NIW-updated models with pruning
- Possibilistic analogue replacing probability normalization with supremum-based normalization
- Hybrid necessity estimation (Monte Carlo + deterministic optimization)
- Benchmark comparison against classical (Markowitz, 1/N) and shrinkage-based methods (Jorion, Kan-Zhou)

### Pipeline

1. Load and clean return data  
2. Estimate next-period means and covariances  
3. Convert moments into portfolio weights  
4. Evaluate performance (Sharpe, certainty equivalent, distributional scores)

## What The Project Implements

### Portfolio rules and benchmarks

`alternative_investment_rules.py` implements:

- `equal_weight_strategy`: 1/N allocation
- `market_weight_strategy`: market-only allocation using the `Mkt-RF` column
- `historical_expectations_weights`: expanding-window sample means and covariances followed by Markowitz
- `rolling_window_weights`: rolling-window sample means and covariances followed by Markowitz
- `minimum_variance_strategy`: minimum-variance portfolio
- `jorion_bayes_stein_estimates` and `jorion_bayes_stein_strategy`: Jorion Bayes-Stein shrinkage
- `kan_zhou_three_fund_estimates` and `kan_zhou_three_fund_strategy`: Kan-Zhou three-fund shrinkage

### Bayesian averaging

`bayesian_averaging.py` implements a dynamic Bayesian model averaging procedure over many candidate NIW-updated models. It:

- creates a new "newborn" model each period
- updates model probabilities using a marginal likelihood
- prunes low-probability models to control runtime
- aggregates model-specific moments into a single predictive mean and covariance

This is the probabilistic version of the model-combination engine and is the closest module to the Anderson-Cheng-style setup. The implementation also adds **pruning**, which is important because the raw procedure grows quickly with o(T**2) complexity with time.

### Possibilistic Bayesian averaging

`possibilistic_bayesian.py` implements the **possibility-theoretic analogue** of the Bayesian averaging logic. Relative to the probabilistic version, it:

- replaces posterior probability integral based normalization with **supremum normalization**
- avoids prior probability specification in the same sense as the probabilistic weighting step, keeping the new model more "uninformative"
- keeps NIW-style parameter updates
- aggregates surviving models into a possibilistic predictive mean and covariance through geometric aggregration.

### Necessity calculations

Two files address necessity-style weighting for possibilistic models:

- `necessity_calculation.py`: a **naive Monte Carlo** approximation of model necessity / internal validity
- `necessity_deterministically.py`: a more **deterministic hybrid** approach using analytic starts, optional Monte Carlo seeds, and local optimization

### Evaluation

`sharpe_ratio.py` computes:

- realized portfolio returns
- overall Sharpe ratio
- rolling Sharpe ratio
- certainty equivalent

with formulas expressed below.

### Data and simulation

- `data_input.py` dynamically loads and cleans data from Kenneth French style CSV files
- `simulated_datasets.py` provides two synthetic datasets: an i.i.d. factor DGP and a regime-shifting factor-mean DGP

## Results (Summary)

- Bayesian averaging exhibits rolling Sharpe decay over long horizons
- Possibilistic aggregation retains model influence longer but may introduce persistence effects
- Classical benchmarks (1/N, minimum variance) remain competitive under estimation error
- Model pruning is necessary to control computational growth without degrading performance materially

## Mathematical Conventions

### Markowitz weights

For predicted mean vector `\mu_t` and covariance matrix `\Sigma_t`, the unconstrained Markowitz rule used here is:

w_t = (1 / θ) Σ_t^{-1} μ_t

where `\theta` is the risk-aversion parameter.

### Portfolio return timing

The repository generally assumes:

w_t is applied to r_{t+1}

This lagging convention appears in the weight generation and evaluation code.

### Sharpe ratio

The daily Sharpe ratio is:

Sharpe = E[R_p] / sqrt(Var(R_p))

and if annualized with scaling factor `K`:

Sharpe_annual = Sharpe_daily * sqrt(K)

### Rolling Sharpe ratio

For a rolling window `W`, the rolling Sharpe at time `t` is:

RollingSharpe_t = mean(R_{p,t-W+1:t}) / std(R_{p,t-W+1:t})

with the same optional `\sqrt{K}` annualization.

### Certainty equivalent

The certainty equivalent used in `sharpe_ratio.py` is the mean-variance approximation:

CE = E[R_p] + E[R_f] - (θ/2) Var(R_p)

When `include_rf_in_mean=False`, the `\mathbb{E}[R_f]` term is omitted.

## Getting Started

### Requirements

- Python `3.11+`
- one of:
  - `uv` for environment management
  - standard `venv` plus `pip`

### Install with `uv`

From the repository root:

```bash
cd possibilistic-portfolio-optimisation-py
uv sync
```

This installs the dependencies declared in [`pyproject.toml`](possibilistic-portfolio-optimisation-py/pyproject.toml).

### Install with `venv`

```bash
cd possibilistic-portfolio-optimisation-py
python3.11 -m venv .venv
source .venv/bin/activate
pip install ipykernel matplotlib mpmath mypy numpy pandas pandas-stubs scipy
```

### Important import-path note

The modules in `possibilistic-portfolio-optimisation-py/src` use direct imports such as `from markowitz import ...`, so the easiest ways to run the code are:

1. run notebooks or scripts with the working directory set to `possibilistic-portfolio-optimisation-py/src`, or
2. add `possibilistic-portfolio-optimisation-py/src` to `PYTHONPATH`

### `possibilistic-portfolio-optimisation-py/src/`

- [`alternative_investment_rules.py`](src/alternative_investment_rules.py): benchmark and shrinkage-based portfolio rules
- [`bayesian_averaging.py`](src/bayesian_averaging.py): probabilistic dynamic Bayesian averaging engine
- [`data_input.py`](src/data_input.py): loaders and cleaning utilities for Kenneth French style data
- [`distribution_evaluation_func.py`](src/distribution_evaluation_func.py): predictive-distribution evaluation helpers
- [`markowitz.py`](src/markowitz.py): unconstrained Markowitz weight computation
- [`necessity_calculation.py`](src/necessity_calculation.py): Monte Carlo necessity approximation
- [`necessity_deterministically.py`](src/necessity_deterministically.py): deterministic / hybrid necessity approximation and weighting schemes
- [`possibilistic_bayesian.py`](src/possibilistic_bayesian.py): possibilistic dynamic model averaging engine
- [`prior_selection.py`](src/prior_selection.py): prior-sharing helper for Bayesian averaging
- [`sharpe_ratio.py`](src/sharpe_ratio.py): realized-return and utility metrics
- [`simulated_datasets.py`](src/simulated_datasets.py): simulated datasets and DGP helpers
- [`analysis.ipynb`](src/analysis.ipynb): main exploratory notebook for generating results and comparing methods
- [`testing_data_input.ipynb`](src/testing_data_input.ipynb): notebook used to test and inspect the data loading utilities
- [`file.pkl`](src/file.pkl): serialized experiment artifact; the codebase does not document its schema directly

### `datasets/`

The Datasets have been sourced from Kenneth-French Fama factor models library, as well as the CRSP database.

## Notes And Caveats

- The project is currently organized more like a research codebase than a packaged library.
- The Bayesian and possibilistic model-combination algorithms can be computationally expensive because the active model set grows over time; both files therefore include pruning support.

## Suggested Entry Points

If you are new to the repository, the best starting points are:

1. [`data_input.py`](src/data_input.py) to understand the data schema.
2. [`analysis.ipynb`](src/analysis.ipynb) to see how the modules are composed.
3. [`bayesian_averaging.py`](src/bayesian_averaging.py) for the probabilistic baseline.
4. [`possibilistic_bayesian.py`](src/possibilistic_bayesian.py) for the main novel contribution.
5. [`sharpe_ratio.py`](src/sharpe_ratio.py) and [`distribution_evaluation_func.py`](src/distribution_evaluation_func.py) for evaluation.
