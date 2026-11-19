# Dev TODO

The refactor (merged into `post-fyp` on 2026-11-19) is in git history: its
plan and the done items of this file are in commit `599c77a`
(`git show 599c77a:REFACTOR_PLAN.md`, `git show 599c77a:TOOLING_TODO.md`).

## Open

- [ ] **Fresh baseline run** (defaults: no merging, no necessity; all
  algorithms, real + simulated) from a clean commit into `results_2026-11-19`
  (~30 min), so its `config.json` names an exact commit; then evaluate it
  (`evaluation_pipeline <folder>`, writes `scalars.csv`) and point the
  notebooks at it (next item). `results_merge0.15` (real + simulated,
  2026-11-19 at `599c77a`, was `results_2026-11-20`) is the same with merging
  at 0.15: keep it as the merging-on run of the ablation. Delete the old
  `results_baseline/` (2026-11-17, `411052a + uncommitted changes`).
- [ ] **Notebooks** (`analysis.ipynb`, `predictive_diagnostics.ipynb`):
  `analysis.ipynb` hard-codes `RESULTS_DIR = Path("results")` (relative, not
  from `config.py`): point it at `results_2026-11-19`. Strategy
  lists lack `possibilistic_plain`; the diagnostics cell does
  `set_index("t")` and reads `nec_mean`, but new runs index the diagnostics by
  Date and only have necessity columns with `--necessity`; some cells load the
  old `*_mus.npy` / `*_sigmas.npy` (new runs: `<algo>_predictive.npz`, read with
  `predictive.load_predictive`).
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
  with merging at 0.15 (FYP) vs off vs 0.224 (Bhattacharyya overlap 95%), on
  the 16 real datasets then finished in `results_merge0.15`; Newey-West
  Sharpe-difference test (`sharpe_difference_test`, 22 lags), all scored from
  row 1000. Average log-likelihood ranks off > 0.15 > 0.224 on **all 16**
  datasets (≈ 0.06 per day between neighbours). Sharpe, 0.15 vs off:
  significant at 5% on 4/16 (unconstrained) and 2/16 (long-only), every one
  in favour of off (largest: 6 size–STR EW t = −5.4, VW t = −4.0). 0.15 vs
  0.224: 3/16 significant (unconstrained), all in favour of 0.15. Likely
  cause: a merged model adds both models' κ and ν although merge candidates
  (similar ν) have seen mostly the same days, so it is over-confident.
  Merging was ~90% of possibilistic's run time (3 min → ~20 s per dataset).
  The 0.15 threshold has no recorded source. Kept as an option:
  `RunSettings(merge_threshold=0.15)`. Datasets overlap (EW / VW of the same
  portfolios) and each is one sample path, so read it as a consistent
  pattern, not 16 independent results.

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
