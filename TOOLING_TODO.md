# Dev TODO

The refactor (merged into `post-fyp` on 2026-11-19) is in git history: its
plan and the done items of this file are in commit `599c77a`
(`git show 599c77a:REFACTOR_PLAN.md`, `git show 599c77a:TOOLING_TODO.md`).

## Open

- [ ] **Delete `results_baseline/`** (2026-11-17, `411052a + uncommitted
  changes`): replaced by `results_2026-11-19` (the baseline: defaults, all
  algorithms, real + simulated, `d9d44db`, evaluated). `results_merge0.15`
  (same with merging at 0.15, `599c77a`) is the merging-on run of the ablation.
- [ ] **`predictive_diagnostics.ipynb`**: same fixes as `analysis.ipynb` got
  (2026-11-19): run folder from `config.ROOT`, `possibilistic_plain` instead of
  the necessity schemes, `predictive.load_predictive` instead of the old
  `*_mus.npy` / `*_sigmas.npy`, diagnostics indexed by Date (no `"t"`, no
  `nec_*` without `--necessity`).
- [ ] **`analysis.ipynb` narrative** (code is current; the text is the FYP's):
  update the percentages in cells 18 / 23 (now: turnover −51% / −69%, Sharpe
  cost 7.2% / 2.2%, unconstrained / long-only); cell 5 says BMA merges (it
  doesn't); cell 10 says Student-t predictive (scoring is Gaussian); the
  Mahalanobis heatmap says "lower is better" (calibrated ≈ √n: BMA's lower
  value means a too-wide Σ).
- [ ] **`plot_weight_distributions.py`**: `RESULTS_DIR` points at a results folder
  that may not exist; `STRATEGIES` only covers Bayesian + possibilistic, though
  `GROUP_COLOURS` lists benchmark groups too.
- [~] **Necessity on GPU** (`device="cuda"`): untested (PyTorch not installed);
  GPU scoring is no longer planned.

Minor tidying (no effect on results; fix when next in those files):
- [ ] One list of French files: `config.FRENCH_PORTFOLIO_FILES` and `table1_summary_stats.PORTFOLIO_CONFIGS` (+ its own `START`/`END`, duplicating `config.START_DATE`/`END_DATE`) can drift apart.
- [ ] Demo mains in algorithm modules (`bayesian_averaging.main`, `possibilistic_bayesian.demo_run_core`) are the only reason they import data loading; `--algorithms` does the same.
- [ ] `faster_necessity.py`: GPU path (~400 lines) into its own file; masked / power / exp formulas next to `possibilistic_bayesian._scheme_weights`.
- [ ] `pipeline_wrapper.py` (~920 lines): run record (`config.json`, `run.log`, resume) into its own module; one driver for the real and simulated loops.
- [ ] Rolling-window slice `iloc[t-w+1 : t+1]` written 5 times (historical, rolling, Jorion, Kan-Zhou, min-var): one helper.
- [ ] Simulated CSVs read in two places with different column rules (pipeline: `Asset*`; Table 1: all but `Date`): one `simulated_datasets.load_sim`.

## Research directions (one branch each, merged into `post-fyp` when done)

- [ ] **Possibilistic over-confidence on simulated data** (branch
  `research/sim-overconfidence`). Mean Mahalanobis d²/n = 2.50 on every
  simulated DGP (calibrated = 1; historical / Jorion / Kan-Zhou ≈ 1.0, BMA
  1.40), but ≈ 1.0 on real data; same in the FYP results (2.51 / 2.51 / 2.02
  masked / exp / power), so not from the refactor. The pool is always exactly
  the 100 newest models on simulated data (ν from 2 to 101, `nu_mean` = 51.5
  at every step); on real data it sometimes keeps older models (`nu_mean` up
  to 365) but also falls back to the newest 100. Unverified guess: older
  models' possibilities collapse relative to the newborns (or underflow to
  ties) and pruning keeps the newest; simulated returns are ~3× more volatile
  (daily sd ≈ 0.03 vs 0.01). May also drive possibilistic's lower turnover and
  smoother means in general.
- [ ] **Anderson & Cheng's integral** (branch `research/ac-integral`): add the
  integral as defined in Anderson & Cheng (2016) and see how the results
  change. *Which equation (number) to confirm before starting.*
- [ ] **Why possibilistic allocations are more stable** (branch
  `research/sharing-prior`). Hypothesis: the perfect sharing prior in BMA forces
  the models to redistribute probability mass at every step (sum
  constraint); the uninformed possibilistic prior has no such constraint.
  Experiment: sweep BMA's sharing-prior parameter and plot Sharpe and turnover
  against it, with possibilistic as a reference point; then the other priors
  Anderson & Cheng use.
- [ ] **Robust Bayesian averaging** (branch `research/robust-bayes`): A&C's
  robust BA under probability and under possibility, and whether it fits the
  possibilistic framework.
- [ ] **Model drift after ~12,000 periods** (branch `research/model-drift`):
  appears inherent (not tied to a time period) after about 12,000 periods;
  find out why.

## After the paper — research design

- [~] **Possibilistic merge: ν of the merged model** — `_merge_two_models` uses
  ν′ = max(ν₁ + ν₂ − n, n + 2) (thesis eq. 5.5), borrowed from the probabilistic
  inverse-Wishart. Possibilistic newborns start at ν = 1, so merging two young
  models jumps ν up to n + 2. A modelling choice; the code implements the thesis.
- [~] **Simulated datasets: settings for the extension** (to be regenerated with
  new settings; changing them means changing tests).
  - A&C (2016) Appendix A, DGP 1 (iid) and DGP 2 (regime) are matched exactly by `simulate_iid` / `simulate_regime_shifts` (RF, factor, B, Σ, p_switch = 0.5, ρ = 0.95, σ_μ = 0.001, σ_g). Differences in the experiment: A&C use **41 datasets per DGP** (Monte Carlo) of **12,000 days**; we use 1 dataset (seed 42) of 10,000 days → one sample path mixes real differences with luck; use several seeds for the paper.
  - A&C's "regime" DGP is effectively a persistent AR(1) mean with jumpy innovations (persistence 0.975/day, half-life ≈ 27 days, moving mean ≈ 10% of factor variance), not discrete Markov regimes — same family as our OU `mean_reverting` DGP.
  - Our OU `mean_reverting` (half-life ≈ 13.5 days, moving mean ≈ 40% of factor variance) and `stochastic_volatility` (vol half-life ≈ 14 days) are far more extreme than the literature (predictive R² ~1–2% monthly, half-lives of months–years; daily vol persistence ≈ 0.98–0.99). Either present them as stress tests or recalibrate towards literature values.
  - `cov_break` average vol ≈ 19% and `stochastic_volatility` average vol ≈ 17.7% (σ̄² is the median of v, not the mean) are not matched to the iid 16%, unlike regime / mean-reverting. Decide whether DGPs should share the unconditional variance.
  - The comment above `_simulate_factor_regime_mean` ("this is weird - better use the ou reversion process") was a wish for an OU mean; it is kept as is (the OU version exists as `simulate_mean_reverting_factor`; the regime helper is A&C's DGP 2 as published).

## Notes for the paper

- **Merging ablation (2026-11-19) → merging off by default.** Possibilistic
  with merging at 0.15 (FYP, `results_merge0.15`) vs off (`results_2026-11-19`)
  on all 26 datasets (`merge_ablation.ipynb`; Newey-West, 22 lags, Sharpe:
  `sharpe_difference_test`, daily log-likelihood and turnover:
  `mean_difference_test`, Diebold-Mariano). Real data (20): log-likelihood
  better without merging on 20/20, significant on **19/20**; Sharpe
  significant 6/20 (unconstrained) and 2/20 (long-only), all in favour of off,
  none in favour of merging; turnover slightly lower with merging (significant
  7/20, mean −1.3 on ~80, ~1.5%). Simulated (6): no measurable difference.
  Calibration (mean d²/n) ≈ 1.0 for both on real data. A first look on 16
  datasets also had 0.224 (BC 95%) worse than 0.15 on log-likelihood 16/16.
  Likely cause: a merged model adds both models' κ and ν although merge
  candidates (similar ν) have seen mostly the same days, so it is
  over-confident. Merging was ~90% of possibilistic's run time (3 min → ~20 s
  per dataset). The 0.15 threshold has no recorded source. Kept as an option:
  `--merge-threshold H` / `RunSettings(merge_threshold=H)`. Datasets overlap
  (EW / VW of the same portfolios) and each is one sample path, so read it as
  a consistent pattern, not 20 independent results. Optional extra: 0.10 and
  0.224 runs (`--algorithms possibilistic --merge-threshold H`), picked up by
  the notebook automatically.
- **Headline results hold after the fixes** (`results_2026-11-19`, 20 real
  datasets, possibilistic vs BMA, means): turnover −51% unconstrained / −69%
  long-only, Sharpe −7.2% / −2.2% (FYP: ~−50% / ~−69%, ~6.7% / ~3.4%);
  avg log-likelihood 39.38 vs 37.21. Significance not tested yet.

- **Masked = plain in practice:** on the snapshot slices no model ever got a
  necessity score of exactly 0 (`frac_nec_zero` = 0 at every step; mean score
  0.38 real / 0.30 simulated), so the masked scheme equals plain possibility
  weighting there; only power and exp differ.
- **Kan-Zhou:** the incomplete-beta parameters follow Kan & Zhou's own paper
  (eq. 66): B((N−1)/2, (T−N+1)/2) and exponent −(T−2)/2. Anderson & Cheng print
  the second beta parameter as (w+1)/2, a typo (only (w−n+1)/2 gives the
  exponent a+b−1 = (w−2)/2). The forecast is (μ\*, Σ̂/h) (A&C fn. 17, K&Z eq. 75),
  so its log-likelihood / Mahalanobis scores use Σ\* = Σ̂/h (≈ 10% wider than Σ̂
  for w = 252, n = 11): state this in the paper.
- **Bayesian averaging prunes before aggregating** (the thesis' Algorithm 1
  aggregates first; possibilistic follows its Algorithm 2: aggregate, then
  prune). Measured on the full 10 Industry VW run: daily Sharpe 0.069206 (code)
  vs 0.069207 (thesis order), max weight difference 1.7e-3 on weights of
  typical size 25. The Bayesian pool holds ~20 models on average (max 41), so
  the cap of 2000 never binds; the 1e-6 threshold does all the pruning.
- **`keep_newest` pruning** (fixed 2026-11-17: the newborn replaces the
  lowest-ranked kept model, not the youngest): no effect on real data — on the
  full 10 Industry VW run, with the possibilistic pool at its cap of 100 almost
  every step, old and new rule give identical forecasts.
- **FYP results vs now** (changes that move numbers, all deliberate): turnover
  no longer diluted by NaN rows; benchmarks estimate at t from returns up to
  and including t (were t−1); historical is truly expanding; possibilistic
  weights start at `start` (were `burn_in`); every strategy scored on one
  common window; Bayesian newborn prior divides by t / t−1 (were t−1 / t−2);
  Kan-Zhou formula and forecast as above; default possibilistic output is
  plain possibility weighting; possibilistic merging off (above).
