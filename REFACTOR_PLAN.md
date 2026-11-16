# Refactor plan: src layout + tests

Branch: `post-fyp`. FYP reference state: `dev` @ `e57fd62`.

How we work: one file at a time → outline tests → write + run them → list edge cases → decide changes → commit.
Main goal: remove duplication and confusing structure; edge cases second.
Nothing gets changed without a yes.

`[ ]` open · `[x]` done · `[?]` needs your decision · `[~]` deferred

---

## Part 0 — Open decisions (settle before starting)

- [x] **0.1 Package name**: `possibilistic_portfolio` → `src/possibilistic_portfolio/`, `import possibilistic_portfolio`.
- [x] **0.2 Structure inside the package**: flat (same files, just moved). Regroup into subpackages later if needed, once tests exist.
- [x] **0.3 Order**: tests first, against the current flat `src/` (bare imports like `from markowitz import ...`). Then the package move; the full test suite confirms it broke nothing. Test files' import lines get updated during the move.
- [x] **0.4 Results folders + constants**: no fixed `results/`; each run gets its own named folder.
  - `config.py` holds all constants + run parameters: paths (`DATASETS_DIR`, `SIM_DATA_DIR`, `RF_PATH`), `RESULTS_DIR` (set per run), `START_DATE`, file markers, `BURN_IN`, `PERIODS_UNTIL_INVESTMENT`, `THETA`, `MAX_MODELS`, merge params, rolling windows, `DEVICE`, … Importable from scripts and notebooks.
  - Each run writes the settings it used to `<run folder>/config.json`. Notebooks analysing a past run read that file, not `config.py`.
  - Pipeline refuses to write into a run folder that already exists (protects past runs).
  - [x] `config.json` written by hand (2026-11-13) for both existing runs; each file notes its source commit and which values are confirmed from the outputs vs taken from code only.
    - `results_fyp_submitted/` — read-only. Apr 14, code `4e8cb38`, `DEVICE="cuda"`, sim T=10000 N=10, `MAX_MODELS=100`.
    - `results_cpu/` — superseded. Real data: Mar 28–29, code `5274011`, `MAX_MODELS=80`. Simulated: Apr 2–3, T=6000 N=5, runner uncommitted at the time, settings inferred from `ba2b8e4` (`MERGE_THRESHOLD=1.0`). Odd folder `6_Portfolios_size_btm_rename_weighted`.
  - Overlaps with TODO P2 (params through 3 layers) and "Save `rf.csv` / shared `START_DATE`".
- [x] **0.5 Snapshot (golden) tests**: `tests/snapshots/make_snapshots.py` + `tests/snapshots/reference.npz` (made from the `dev` / FYP code).
  - Two slices, 1400 rows each: real (10 Industry VW, 3 portfolios + `Mkt-RF`, from 1980) and simulated (`iid_T10000_N10_seed42`, 3 assets). `burn_in=300`, `piu=100`, `max_models=20`, CPU, real necessity (`--bypass-necessity` optional).
  - Covers: `data_input` end to end (VW + EW blocks of 10 Industry, RF), the whole `calculate_all_functions_perdatasets` output (all 11 methods + long-only + diagnostics) and `run_evaluation` output (scalars, rolling Sharpe, log-likelihood, Mahalanobis). Compared exactly (bit for bit).
  - Not covered: `call_all_datasets` / `call_all_simulated_datasets` loops, `main()`s, GPU path, other simulated DGPs.
  - [ ] Turn the `--compare` run into a pytest test (`tests/test_snapshots.py`).
  - Finding (2026-11-15): `reference_no_necessity.npz` (made 13 Nov) had long-only weights ~1e-12 off from any run today, so `tests/test_snapshots.py` failed on unchanged code. Same code, NumPy/SciPy versions and inputs; reruns deterministic → one-off quirk of the run that made it (BLAS/threads?). Regenerated from `dev`; current `post-fyp` code matches it bit for bit, test green. `reference.npz` (real necessity, same night) not re-checked (~80 min).
  - Finding: `evaluation_pipeline.py` `__main__` reads a simulated run's `rf.csv` with `index_col=0` → expects a `Date` column. The simulated CSVs were regenerated on 2026-04-27 without `Date`, so the pipeline now writes a Date-less `rf.csv` and evaluating a new simulated run would fail (`KeyError: 'RF'`). Old run folders (with `Date`) still work.
- [x] **0.6 Generated outputs + leftovers** (done 2026-11-13, not committed):
  - `src/weight_distributions*.png` deleted by user (in history).
  - New gitignored `figures/` for anything regenerable. [ ] In Part 1: `config.FIGURES_DIR`; `plot_weight_distributions.py` and `table1_summary_stats.py` write there instead of `src/`.
  - `predictive_diagnostics.ipynb`: removed last cell `Image("src/weight_distributions.png")` (file gone).
  - `src/tests/` 3 diagnostic PNGs (Mar 24–25, code no longer exists) → `archive/diagnostic_figures/` (tracked). Rest of `src/tests/` deleted (in history; local copies only differed by the Nov 6 ruff formatting).
  - `diagnostics_26_march/` deleted (never committed; old run, code since fixed).
  - `.gitignore`: removed `src/tests/`, `diagnostics_26_march`; added `figures/`.
  - Later, optional: `predictive_diagnostics.ipynb` is 34 MB (154 embedded plots); consider stripping outputs before commits (nbstripout).
- [x] **0.7 One-off scripts** (done 2026-11-13): `comparison_cpu_gpu.py` deleted (in history: `git show ba2b8e4:src/comparison_cpu_gpu.py`). `compute_longonly_weights.py` deleted: long-only part redundant (pipeline does it); Sharpe-difference test moved into `portfolio_evaluation_functions.py` → TODO item in `TOOLING_TODO.md`. Both `.gitignore` lines removed.
- [x] **0.8 `necessity_alternative_methods/`**: renamed by user to `src/necessity_archive/` (not committed yet). Necessity will be removed from the project altogether later, so this folder isn't worth much effort.

---

## Part 1 — Package move (mechanical, no behaviour change)

### 1.1 `pyproject.toml`
- [ ] Add `[build-system]` with `uv_build` backend.
- [ ] Add `[tool.uv.build-backend] module-name = "possibilistic_portfolio"` (project name `possibilistic-portfolio-optimisation-py` would otherwise map to a different module name).
- [ ] `[tool.pytest.ini_options]`: remove `pythonpath = ["src"]` (package is installed into the venv instead); fix the odd indentation.
- [ ] `requires-python = ">=3.11,<3.14"` vs `.python-version` 3.13 and ruff `target-version = "py311"` — keep as is?

### 1.2 New files
- [ ] `src/possibilistic_portfolio/__init__.py` (empty).
- [ ] `src/possibilistic_portfolio/config.py` — see 0.4. All paths, `RESULTS_DIR`, `START_DATE`, markers and run parameters.

### 1.3 Moves (`git mv`, keeps history)
All of `src/*.py` + `src/necessity_alternative_methods/` → `src/possibilistic_portfolio/` (subject to 0.7, 0.8).

### 1.4 Per-file touch list (imports + paths)

| File | `sys.path` hack | Bare imports to change | Hard-coded paths to replace |
|---|---|---|---|
| `pipeline_wrapper.py` | L28 | L26 `markowitz`, L31 `bayesian_averaging`, L32 `possibilistic_bayesian`, L33 `alternative_investment_rules`, L42 `data_input` | L548 `../datasets/simulated`; L601, L662 `parent.parent / "results"`; L607–617 ten `../datasets/*.csv`; L621 RF path |
| `evaluation_pipeline.py` | L37 | L39 `distribution_evaluation_functions`, L44 `portfolio_evaluation_functions`, L385 `data_input` (inside `__main__`) | L387 results dir; L389 datasets dir |
| `table1_summary_stats.py` | L26 | L27 `data_input`, L28 `simulated_datasets` | L41 `DATASETS_DIR`; L247 `SIM_DATA_DIR`; L459 `table1.tex` output into `src/` |
| `data_input.py` | — | — | — (paths moved to `config.py` in 2.2) |
| `bayesian_averaging.py` | — | L45 `data_input`, L49 `markowitz` | — |
| `possibilistic_bayesian.py` | — | L30 `data_input`, L34 `faster_necessity`, L40 `markowitz` | — |
| `alternative_investment_rules.py` | — | L28 `markowitz` | — |
| `plot_weight_distributions.py` | — | — | L13 results dir; L14 PNG output into `src/` |
| `markowitz.py`, `portfolio_evaluation_functions.py`, `distribution_evaluation_functions.py`, `simulated_datasets.py`, `faster_necessity.py` | — | — | — |
| `tests/test_markowitz.py` | — | L6 `markowitz` | — |

### 1.5 Other places that mention the old layout
- [ ] `README.md` L33, L41, L49 — `uv run src/...` → `uv run python -m possibilistic_portfolio.<module>`.
- [ ] `CLAUDE.md` — layout tree + commands.
- [ ] `.gitignore` — `src/tests/` path; `comparison_cpu_gpu.py` / `compute_longonly_weights.py` entries (depends on 0.7); add `scratch/` if used.
- [ ] `src/tests/` (gitignored scratch notebooks + `testing_pipeline.py`) → `scratch/`.
- [ ] `possibilistic_bayesian.py` L830 `test_run_core()` lives in the module — rename (e.g. `demo_run_core`) or move into `tests/`, so it isn't mistaken for a test.

### 1.6 Verify
- [ ] `uv sync` installs the package.
- [ ] `uv run pytest` passes.
- [ ] Every module imports: `uv run python -c "import possibilistic_portfolio.<m>"` for each.
- [ ] `uv run ruff check .` no new errors.
- [ ] Snapshot tests (0.5) still match.

---

## Part 2 — Tests, file by file

Order follows dependencies (each file's tests rely only on files already covered).
Tests go in `tests/test_<module>.py`. Shared tiny fixtures in `tests/conftest.py`.

**Rule: unexpected NaNs are reported where they are made (2026-11-16).** Evaluation (2.3, 2.4) treats NaN inputs as "no forecast / not invested" and does not report them: it can't tell a planned NaN (before investment starts, last row) from garbage. So every algorithm that leaves a row NaN **after** investment starts must say so itself: one `RuntimeWarning` per call with counts per reason (pattern: `markowitz.py`, 2.1). No silent `continue` / `except: pass` in the investment window. Silent sites found 2026-11-16:
- `alternative_investment_rules`: `_minvar_mean_target` (denominator 0 / non-finite → NaN), Jorion (2 skip sites), Kan-Zhou (5 skip sites) → 2.5.
- `possibilistic_bayesian.run_core`: `except Exception: pass` around aggregation → 2.9.
- Already reporting: `markowitz` (singular Σ, optimiser failure). Bayesian: no silent NaN path found.

### Date index (cross-cutting) — [x] done 2026-11-16, values unchanged (snapshot bit for bit)
Done as its own step, before 2.11; the timing fix (benchmarks at t-1) stays separate.
- `data_input.prepare_returns` keeps `Date` as the index (simulated data: row numbers).
- Forecasts carry their index: new `src/predictive.py` — `make_predictive` (every algorithm builds its `mu_sigma_dict` with it → keys `mu_hat`, `sigma_hat`, `index`), `check_aligned` (used by both Markowitz entry points and `distribution_evaluation_functions`), `save_predictive` / `load_predictive` (`<algo>_predictive.npz` replaces `<algo>_mus.npy` / `_sigmas.npy`; old `.npy` still read, without index).
- `pipeline_wrapper`: weights / `returns.csv` carry dates; real runs also save `rf.csv` (Date, RF).
- `evaluation_pipeline`: reads dates back; refuses weights whose index differs from `returns.csv`; `load_rf(folder)` (dated `rf.csv` / Date-less simulated `rf.csv` / legacy global RF by row) replaces the `__main__` logic.
- Tests: `tests/test_predictive.py` (6), dates reach every output, misaligned weights / forecasts raise, RF by date / row / legacy, Sharpe test + CE by date. Old FYP folders evaluate identically to the previous commit.
- [~] Notebooks load `*_mus.npy` / `*_sigmas.npy` directly: left as is on purpose (2026-11-16) — they show the FYP results with saved outputs. New notebooks later will use `predictive.load_predictive` / `evaluation_pipeline.load_strategy_outputs`.
- [ ] `possibilistic_diagnostics.csv` is still indexed by step `t`, not date (2.9).

### 2.1 `markowitz.py` — [x] done 2026-11-15 (not committed)
Used by: `bayesian_averaging`, `possibilistic_bayesian`, `alternative_investment_rules` (unconstrained); `pipeline_wrapper` (long-only).

Tests (`tests/test_markowitz.py`, 23 passing): closed form Σ⁻¹μ/θ, 1/θ scaling, long-only valid / interior / corner solutions, warm-started long-only entry point == per-step (θ passed through), NaN structure + index/columns + NaN-μ rows + too-short series for both entry points, unconstrained == per-step formula; singular Σ → NaN row + warning, θ ≤ 0 raises (both entry points), optimiser failure → NaN row + warning + warm start kept, clean run → no warning.

Decisions:
- [x] **Duplication:** one loop `compute_weights_array(..., long_only=False)` for both variants; `markowitz_step_long_only` got a `w0` arg and is called by the loop; `compute_weights_array_long_only` deleted. No extra `markowitz_weights` entry point: `markowitz_unconstrained` / `markowitz_long_only` stay, callers unchanged. Old vs new weights bit-for-bit identical (both variants). B023 lint warning gone.
- [x] **Duplication:** `burn_in + periods_until_investment` was re-added in every function → done 2026-11-16 as its own step (before the timing fix, values unchanged, snapshot bit for bit): `pipeline_wrapper` computes `start` once; Markowitz (`compute_weights_array`, both entry points) and all benchmarks take `start` (+ `window`); Bayesian / possibilistic `run_core(burn_in, start=None)` (`burn_in` = prior period, `start` defaults to it). Possibilistic still starts at `burn_in` (logged bug, explicit `weights_start = burn_in` + comment, fix in 2.9). Historical's window length is `start` until the expanding fix.
- [x] Singular Σ → was `LinAlgError` killing the run. Now NaN row + one `RuntimeWarning` with counts per reason (unconstrained only; SLSQP doesn't invert Σ). `inv` kept.
- [x] NaN in Σ but not μ → **not possible, no handling.** Every producer fills μ and Σ together (possibilistic assigns them in pairs after the `try`); 0 of 2,448,648 valid-μ rows in `results_fyp_submitted` have NaN in Σ.
- [x] θ ≤ 0 → now `ValueError` up front (was `ZeroDivisionError` / silent all-in).
- [x] Near-singular Σ → **leave.** Markowitz is right for the Σ it gets; huge weights come from estimation error in Σ (smallest eigenvalues underestimated, "error maximisation"), which is the estimator's job (shrinkage: Bayesian, Jorion, Kan-Zhou). Capping in Markowitz would hide the differences the thesis measures. FYP condition numbers: Bayesian ~20–100, VW rolling/historical up to ~4·10⁶ (Mkt-RF ≈ combination of VW portfolios); simulated < 60.
- [x] Long-only optimiser failure → NaN row, counted in the warning, warm start kept. 0 real failures in FYP outputs.
- [x] `inv(Σ) @ μ` → `solve(Σ, μ)`: **not changed** (exact snapshot).
- [x] Docstring: unconstrained weights don't sum to 1 (rest in risk-free); long-only does.

### 2.2 `data_input.py` — [x] rewritten 2026-11-15 (not committed), 305 → 175 lines
Rewrite: every table in the French / FF files has the same shape (title line, header starting with `,`, date rows), checked on all 11 files. One reader finds tables by that shape, so markers, the read-to-EOF branch, the `except` fallback and `skiprows=4` are gone.
New API: `read_french_tables(path)` (all tables by title, percent) · `load_portfolios(path, weighting, start, end, percent=False)` · `load_risk_free(path, start, end, percent=False)` (both loaders convert units through one helper `_from_percent`) · `load_excess_returns(ports, rf, weighting, start, end)` · `prepare_returns` (unchanged). Removed: `load_kenneth_french_portfolios`, `load_risk_free_rate`, `calculate_excess_returns`, `slice_timeframe`, `load_excess_returns_from_kenneth_french_path`, `_read_csv_block_between_markers`, the `__main__` demo.
Output unchanged: all 10 real files × VW/EW + RF bit-for-bit identical to the old loader; snapshot test green; `make_snapshots.py` still runs on FYP `dev` (falls back to the old API).
Callers updated: `pipeline_wrapper` (loops `weighting in ("value", "equal")`; `(path, end_marker)` tuples and marker constants gone), `table1_summary_stats.load_raw_pct` (uses `load_portfolios(percent=True)`; its markers, end-marker config fields, ×100 and sentinel code gone), `evaluation_pipeline` `__main__`, `bayesian_averaging.main`, `possibilistic_bayesian.test_run_core`, `make_snapshots.py`.

Tests (`tests/test_data_input.py`, 31 passing): YYYYMMDD parsing, tables found by title (untitled FF table → `""`), Excel comma padding, `-99.99` / `-999` → NaN, VW block (sorted, % → decimal), EW block stops before copyright, extra tables after EW not read, `percent=True` gives the same units for portfolios and RF, missing table → clear error, bad weighting rejected, RF file, RF header length doesn't matter, excess returns (RF not subtracted from Mkt-RF), days without RF dropped with a warning / error above 1% / no warning when files match, inclusive date window, `prepare_returns` drops only Date/RF, **all 10 real files load both blocks cleanly** (and without warnings), `Other` kept.

Edge-case notes:
- [x] `-99.99` (and `-999`, also listed in the file headers) → NaN, in one place (`read_french_tables`), compared in percent before dividing (table1's old ×100 round trip could miss it). None in the current data.
- [x] Extra tables after the EW block ("Number of Firms in Portfolios", "Average Firm Size" in `6_Portfolios_size_btm`) are never read; no special case needed. A new file with extra tables can't break it.
- [x] Wrong / missing table → `ValueError` naming the title and listing the tables found (was: silent empty data or confusing `ParserError`).
- [x] `skiprows=4` gone: the FF header row is found by its shape.
- [x] `DATA_PATH` / `RISK_FREE_RATE_PATH` → new `src/config.py` (`DATASETS_DIR`, `RF_PATH`); other constants follow in Part 1.
- [x] Portfolio days with no RF: still dropped (inner join), now with a warning (count + first dates); more than 1% missing → `ValueError` (files probably don't belong together, e.g. monthly vs daily). 0 missing in the real data.

### 2.3 `portfolio_evaluation_functions.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_portfolio_evaluation_functions.py`, 16 passing + 2 xfail = logged bugs): weights at t × returns at t+1, NaN rows kept/dropped, column alignment by name, Sharpe = mean/std and √scaling, rolling Sharpe last window + warm-up NaNs, CE formula with/without RF and RF as DataFrame, turnover on constant weights; Sharpe-difference test: Newey-West(0 lags) = sample cov, symmetric, point estimates, detects clear outperformance, antisymmetric t, common-index alignment, too-few-obs error, `compare_to_benchmark` one row per strategy.
Bugs (TOOLING_TODO): ~~turnover diluted by NaN rows~~ fixed 2026-11-16 (xfail removed, NaN-row test added, snapshot reference regenerated: only `avg_turnover` changed); ~~`compare_to_benchmark` loses column names~~ fixed 2026-11-15 (xfail removed).

Structure cleanup (2026-11-15, 392 → 317 lines, snapshot bit for bit):
- [x] Unused options removed: `lag` (`portfolio_returns`, `certainty_equivalent`; only 1 was ever used, 0 would be look-ahead), `include_rf_in_mean` and RF-as-DataFrame in `certainty_equivalent` (RF is now `rf_series: pd.Series`).
- [x] Clutter: redundant `float()` / annotated temporaries, one `rolling` object instead of two, unused `k` in `_newey_west_cov`, garbled Jacobian comment → one-line derivation, single `J_d` array, `2·norm.sf` p-value (no underflow to 0), stale module / `compare_to_benchmark` docstrings.
- [x] `analysis.ipynb` cells 20 / 25 titled the `scalars.csv` Sharpe heatmaps "Annualised" but plotted daily values → labels now follow `SCALING_FACTOR` from the run's `config.json` (cell 2; assumes 1 with a printed note if missing) (2026-11-16). Outputs still show the old titles until re-run. `predictive_diagnostics.ipynb` cell 3 annualises its own rolling Sharpe with √window (right only because window = 252).
- [x] `scaling_factor` kept (removed and restored 2026-11-16): Sharpe / rolling Sharpe × √factor, turnover × factor, as originally written. Pipeline passes `config.SCALING_FACTOR = 1` (daily, as Anderson & Cheng) instead of the hard-coded `_SCALING_FACTOR`; `evaluation_pipeline` docstrings say so. `"SCALING_FACTOR": 1` added to both `results_*/config.json` (verified from saved Sharpe). Once the pipeline writes `config.json` (0.4), include it.
- Kept: `SharpeTestResult` NamedTuple (named fields, used by the tests and turned into the `compare_to_benchmark` table); `dropna` argument (rolling Sharpe needs the NaN rows).

Edge-case notes:
- [x] Sharpe with std = 0 returns 0.0 (not NaN); with a single return → NaN. → Both now NaN + `RuntimeWarning` with the reason (2026-11-15). Tests added.
- [x] Rolling Sharpe replaces ±inf by NaN silently → zero-std windows now NaN + one warning with the count; warm-up NaNs (`min_periods` = full window, kept) stay silent (2026-11-15). Test added.
- [x] CE adds mean RF to excess returns (`include_rf_in_mean=True`) — **intended**: matches Anderson & Cheng (2016) eq. (28), `mean(R_p) + mean(R_f) − θ/2·Var(R_p)`, sample variance, R_f on return dates t+1. Docstring now states the formula + reference (2026-11-15). Open part: RF matched by row number, not date → fixed by keeping `Date` as the index (proposed 2026-11-15, with the TODO timing fix; not scheduled yet).
- [x] Sharpe test with identical series → `t = nan` + RuntimeWarning (TODO item). → Now t = 0, p = 1, no warning (2026-11-15). Test added.
- [x] Docstring typo "inno dexed" in `portfolio_returns` — fixed 2026-11-15.
- [x] (done 2026-11-16, see "Date index" under Part 2) **Date index change touches this file** (when `Date` is kept as the index, with the TODO timing fix): `lag` is already gone (fixed 1-day shift); `portfolio_returns` output becomes indexed by return date; `certainty_equivalent` then matches RF by date (`rf_series.loc[port.index]` fails loudly on missing days instead of picking the wrong row); `sharpe_difference_test` aligns on the common index labels (`index.intersection`) → row numbers today, dates then; no code change needed, add a test that it aligns by date. Tests here build integer-indexed frames → update fixtures (date index).
- [x] (2026-11-15) Input checks: `assert`s → `ValueError` with a message for value checks (`lag ≥ 0`, `window > 0`, `1 ≤ min_periods ≤ window`, `θ ≥ 0`, RF one column); `isinstance` asserts dropped (type hints + Pylance). One `assert` kept for type narrowing. Parametrised test added.

### 2.4 `distribution_evaluation_functions.py` — [x] done 2026-11-16 (tests drafted 2026-11-13)
Tests (`tests/test_distribution_evaluation_functions.py`, 8 passing): log-likelihood == `scipy.stats.multivariate_normal.logpdf` with lag 1 and lag 0, Mahalanobis == `scipy.spatial.distance.mahalanobis`, squared option, NaN rows before predictions + index kept, average = mean of valid rows, all-NaN → NaN, shape mismatch rejected.

Structure cleanup (2026-11-16, 174 → 114 lines, snapshot bit for bit):
- [x] **Duplication:** `mahalanobis_distance_series` / `multivariate_log_likelihood_series` were the same loop → one private `_gaussian_terms` returns the `quad` and `logdet` arrays; the public functions are ~3 lines each. `_prepare_scoring_inputs` / `_gaussian_scoring_terms` merged into it (no `n_assets` return, no extra copies of Σ / returns).
- [x] `lag` removed (pipeline always 1; `lag=0` scores a forecast against a return it already used = in-sample). Fixed 1-step shift. `lag=0` test removed.
- [x] `assert`s → `ValueError` with a message (shapes, `regularization ≥ 0`); `isinstance` asserts dropped. Tests: shape mismatch, negative regularization, inputs not modified.
- [x] Module docstring: Mahalanobis is the square root (`squared=True` drops it).
- Kept: `squared` option.
- [x] `average_log_likelihood` removed (2026-11-16): the pipeline averages the log-lik series with `.dropna().mean()`, the same way it already did Mahalanobis.
- [x] `_gaussian_terms` shortened: NaN check, symmetrising and `slogdet` on all rows at once; `solve` stays per row (a batched solve differs in the last bit on 11-asset data). "linear algebra error" reason gone (`slogdet` and `solve` share the LU, so `sign > 0` can't fail in `solve`). Bit for bit.
- [x] (done 2026-11-16, forecasts now carry `index` and are checked) **Date index change touches this file** (with the 2.3 date item): the 1-day shift stays (forecast at t−1 scores the return at t) and so does `rows = flatnonzero(ok)[good] + 1` (positions in the shifted arrays → return rows). Output Series take `returns_df.index`, so they become date-indexed for free. `mu_hat` / `sigma_hat` (`.npy`) have no index and are matched to returns **by position**, guarded only by the shape check → when dates go in, save / check the forecast dates too (test: forecast row t belongs to return date t).
- [~] → 2.11: scoring still runs 4× per strategy (avg log-lik, avg Mahalanobis, then both series again) → compute each series once, average for `scalars.csv`.

Edge-case notes:
- [x] Invalid Σ (non-positive determinant, singular) → row silently skipped, so it's left out of the average with no count. → Now one `RuntimeWarning` with counts per reason (NaN rows before investment not counted) (2026-11-16). FYP results: 0 such rows in all 20 datasets × 9 predictive strategies. (Regularization since removed, see below.)
- [~] Each strategy is averaged over its own valid rows (ties in with the possibilistic row-500 bug). → 2.11 (common evaluation window is an evaluation decision).
- [x] Scoring is Gaussian even though NIW predictives are Student-t — kept (same rule for all strategies; benchmarks have no t predictive); module docstring says so (2026-11-16).
- [x] Default `regularization=1e-10` was added to Σ before scoring → **removed** (2026-11-16): a singular / non-PD Σ makes the row invalid (NaN, counted in the warning) instead of being scored on ε·I (extreme values that can dominate an average). Not negligible: VW benchmark Σ have smallest eigenvalues down to ~1e-9, so FYP avg log-lik moves in the 2nd decimal (e.g. 10-Industry VW rolling-short 42.4805 → 42.4662), no rows lost, no ranking change; Bayesian / possibilistic unchanged (min eigenvalue ≥ 1e-6). Snapshot reference regenerated (log-lik / Mahalanobis keys only).

### 2.5 `alternative_investment_rules.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_alternative_investment_rules.py`, 17 passing): incomplete beta vs numerical integral, `_solve_or_pinv` singular fallback, min-variance mean target = GMV portfolio mean, 1/N, market (and KeyError without `Mkt-RF`), min-variance vs Σ⁻¹1 normalised + delayed start when window > burn-in, historical and rolling moments + weights, **Jorion vs an independent implementation of the 1986 formulas**, Jorion shrinkage in (0,1) + strategy uses μ*/Σ*, Kan-Zhou Σ (MLE), μ* convex combination, h formula, strategy weights = (h/θ)Σ⁻¹μ*, window-too-short errors.
Window timing tests pin the CURRENT t-1 windows via one helper `_window_rows` → flips with the TODO fix.
Bugs / checks (TOOLING_TODO): Kan-Zhou saved mean scaled by h; verify Kan-Zhou beta parameters vs paper. Existing: benchmarks one day behind; "historical expanding" is rolling.

Edge-case notes:
- [?] `minimum_variance_strategy` uses `np.linalg.inv` → singular window crashes (same as Markowitz).
- [ ] Jorion / Kan-Zhou skip a row silently (NaN weights) when q ≤ 0, ψ < 0, or the beta is non-finite — no count. Also `_minvar_mean_target` → NaN on a zero / non-finite denominator. → Report with counts per reason (Part 2 NaN rule).
- [?] Kan-Zhou ε can be negative → v outside (0,1) (no check).
- [?] Jorion / Kan-Zhou compute moments up to T-2; historical / rolling up to T-1 (last row has no use anyway).
- [?] `_incomplete_beta_unregularized`: dead `if _betainc is not None` branch (TODO item), docstring typo "incomp2lete".

### 2.6 `simulated_datasets.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_simulated_datasets.py`, 24 passing): every DGP — shapes, column names, integer index, finite values, seed reproducibility; iid means ≈ B·μ̄ and RF moments; loadings 0.5…1.5; sudden break flips mean sign; covariance break raises vol; regime parameters that make Var(g) negative are rejected; stochastic vol varies; **snapshot: all 6 DGPs regenerate the on-disk `*_T10000_N10_seed42.csv` + RF files bit for bit**.

Edge-case notes:
- [?] `generate_and_save_sim_datasets` defaults `T=6000, n_assets=5` — stale (data in use is 10000 × 10). → `config.py`.
- [?] Defaults differ between DGPs: `simulate_regime_shifts` seed=0 / n_assets=10, others seed=42 / n_assets=5.
- [?] `simulate_regime_shifts` default `p_switch=0.5` → mean can switch every other day; check that's intended.
- [?] `simulate_mean_reverting_factor` floors σ_g² at 1e-8 silently if the mean's variance exceeds the target.
- [?] Leftover comment above `_simulate_factor_regime_mean`: "this is weird - better use the ou reversion process."

### 2.7 `bayesian_averaging.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_bayesian_averaging.py`, 15 passing): **marginal likelihood == NIW predictive Student-t (`scipy.stats.multivariate_t`)**, invalid scale → −inf, NIW conjugate update, in-place update of all models, newborn model uses the common prior, sharing prior (2-model case, sums to 1, α=0 means no sharing, first model), probability update == Bayes' rule and stable with extreme likelihoods, BMA moments == mixture mean/covariance, pruning (threshold, renormalise, cap keeps the most probable + newest), `run_core` shapes / NaN layout / weights == Markowitz, Σ̂ symmetric PD, **no look-ahead** (changing returns after t leaves predictions up to t unchanged).
Correction: `sharing_prior_update` IS used (called every step in `run_core`); earlier "unused helper" remark was wrong.
Existing TODO: P3 (θ, `max_models` not passed), P4 (`t-1`/`t-2` denominators in `_new_model_prior`, on hold).

Edge-case notes:
- [?] `burn_obs = min(burn_in, T // 2)` silently shortens the burn-in on short data (same as possibilistic).
- [?] `_prune_models` default `max_models=500`, `run_core` default 2000, pipeline passes nothing → 2000 (vs 100 for possibilistic). → `config.py`.
- [?] Progress `print` every 100 steps (noisy in tests/pipelines; logging?).
- [?] `main()` loads data from the default `DATA_PATH` (dev helper, not used by the pipeline).

### 2.8 `faster_necessity.py` — [~] no tests
Necessity is being removed from the project. No unit tests for this file.

### 2.9 `possibilistic_bayesian.py` — [x] tests drafted 2026-11-13 (necessity bypassed)
Tests (`tests/test_possibilistic_bayesian.py`, 21 passing + 3 xfail = logged bugs): newborn prior = pooled sample mean / variance, newborn possibility 1 with κ=ν=1, log score == docstring formula, possibility update == sup-normalised Bayes (max 1), all-invalid → ones, NIW update identical to Bayesian, pruning (under cap, top-k by possibility, length check), merge of two models (statistics, ν floor), Hellinger (0 for identical, symmetric, 1-D closed form), merge core (close pair merged, far kept, ν bandwidth), aggregation (single model = its mode, scale-invariant weights, κ-weighted mean), `run_core` with necessity = 1 (all three outputs identical, NaN layout, diagnostics, pool ≤ max_models, weights == Markowitz, Σ̂ SPD, **no look-ahead**).
`tests/test_snapshots.py` also covers `run_core` end to end against the FYP code.
Bugs (TOOLING_TODO, xfail tests): weights start at row 500 not 1000; `keep_newest` pruning evicts the wrong model (also Bayesian); Mahalanobis merge distance √2 off (only `results_cpu` used it).

Edge-case notes:
- [?] Merging floors ν at n+2, but newborn possibilistic models start at ν=1 → merging two young models jumps ν up.
- [?] Merge diagnostic `hellinger_avg` is the average from the *last* merge pass only (reset each pass).
- [?] Hellinger returns 1.0 (max distance) on numerical failure, silently.
- [?] `_prune_models` docstring mentions a threshold the function no longer has.
- [ ] `except Exception: pass` around aggregation → report with counts (Part 2 NaN rule); `burn_obs = min(burn_in, T // 2)` (TODO item).
- [ ] L830 `test_run_core` (needs real data, not a pytest test) → rename or delete in Part 1.

### 2.10 `pipeline_wrapper.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_pipeline_wrapper.py`, 9 passing; the per-dataset run is stubbed for loop tests): real-data loop runs VW then EW per file with names `<stem>_value_weighted` / `_equal_weighted`, excess returns + `Mkt-RF` reach the run, parameters are passed through, extra tables ignored (`6_Portfolios_size_btm` case), unreadable file skipped with a warning, start/end date window; simulated loop runs each DGP on `Asset*` columns and copies `rf.csv`, skips missing files with a warning; `_save_algo_outputs` file names; **per-dataset run writes exactly the expected file set** (with and without `Mkt-RF`; necessity bypassed) and `returns.csv` round-trips.
Numbers are covered by `tests/test_snapshots.py`.
~~Bug: file without markers silently loads as empty data and gets run~~ — fixed in 2.2 (2026-11-15), xfail removed. Existing: `device="cuda"` in both `main()`s; real-data runs don't save `rf.csv`.

Edge-case notes:
- [?] `sys.path.insert` at import time (goes away with the package move).
- [?] Default parameters differ between the three functions (e.g. `merge_threshold` 0.01 vs 0.15, `periods_until_investment` 1, `rolling_window_long` 250 vs 252 in `main`) → `config.py`.
- [?] Real-data loop catches every loading exception and prints a warning → a typo in a path just skips that dataset.

### 2.11 `evaluation_pipeline.py` — [x] tests drafted 2026-11-13
Tests (`tests/test_evaluation_pipeline.py`, 8 passing; fake run folder): strategy discovery (incl. `*_longonly`), loading weights ± predictives, missing files raise, scalar metrics == the metric functions (Sharpe daily, CE, log-lik), NaN when RF / predictives missing, `run_evaluation` writes `scalars.csv` (5 columns), `rolling_sharpe.csv`, and `log_likelihood.csv` / `mahalanobis.csv` for predictive strategies only, one broken strategy doesn't stop the rest, empty folder raises.
~~Existing TODO: `__main__` RF loading (`rf.csv` without Date → KeyError; real data re-derives RF from its own `START_DATE`).~~ Fixed 2026-11-16 with the date index: `load_rf(folder)`, tested.

- [x] (2026-11-15) `evaluate_all_strategies` catches each strategy's warnings and shows them again tagged `[algo_name]` (so identical messages from different strategies aren't collapsed into one by Python). Test added.
- [ ] From 2.4: compute the log-likelihood / Mahalanobis series once per strategy (now 4 passes) and average them. Add the Sharpe-difference test (`compare_to_benchmark`) to the outputs (TOOLING_TODO).
- [ ] From 2.4: each strategy is averaged over its own valid rows → decide on a common evaluation window across strategies (different start rows, e.g. possibilistic row-500 bug).
- [ ] **Duplication / flexibility** (~1–1.5 h, agreed 2026-11-15):
  - `strategies: list[str] | None = None` on `evaluate_all_strategies` / `run_evaluation` (`None` = all in folder; unknown name → error listing what's there).
  - Metrics as dicts `SCALAR_METRICS` / `SERIES_METRICS` (name → function of one bundle: returns, weights, predictives, rf, theta, window; returns NaN / `None` when a prerequisite is missing). Overridable via `scalar_metrics=` / `series_metrics=`.
  - Replaces `compute_scalar_metrics` / `compute_series_metrics` with one loop, and the three copy-pasted blocks in `save_evaluation_results` with one loop writing `<name>.csv`.
  - Keep file names, `scalars.csv` column order and the `run_evaluation(folder, rf_series=, theta=)` signature → snapshot stays bit for bit, `make_snapshots.py` unaffected. Update the two tests that call `compute_scalar_metrics` directly.

Edge-case notes:
- [?] **Sharpe in `scalars.csv` is daily** (`_SCALING_FACTOR = 1`), but `compute_scalar_metrics`' docstring says "annualised (scaling = 252)". FYP values are daily (market ≈ 0.033 ≈ 0.52 annualised) — check the thesis tables are labelled daily. Fix the docstring.
- [?] `run_evaluation` docstring says rolling window default 150 days; it's 252 (CLAUDE.md corrected).
- [?] A strategy that fails to evaluate is printed as a warning and silently dropped from `scalars.csv`.
- [?] Long-only strategies get NaN log-lik / Mahalanobis (no `*_longonly_mus.npy`) — fine, but they share predictives with their unconstrained twin.

### 2.12 Scripts — [x] tests drafted 2026-11-13
`tests/test_table1_summary_stats.py` (8 passing): raw-% loading with `-99.99` → NaN, EW block + date filter, bad weighting rejected, `compute_stats` on a known frame, `build_panel` one row per config, simulated save → load (percent units) → panel round trip in a temp dir, missing simulated file raises, all 10 `PORTFOLIO_CONFIGS` paths exist.
`tests/test_plot_weight_distributions.py` (3 passing): pooling across folders drops NaN and ignores other strategies/files, boxplot quartiles / whiskers / fliers, plot writes a PNG (output path redirected to a temp dir).

Edge-case notes:
- [x] French file headers say missing data is `-99.99` **or `-999.`** → both handled in `data_input` since 2.2 (2026-11-15); `table1` uses it.
- [?] `load_sim_pct` / `build_sim_panel` / `generate_and_save_sim_datasets` default `T=6000` (stale; data is T=10000, N=10) → `config.py`.
- [?] `plot_weight_distributions.RESULTS_DIR` points at the non-existent `results/`; `OUTPUT_PATH` writes into `src/` → `config.py` / `figures/` (0.6).
- [?] Plot `STRATEGIES` only covers Bayesian + 3 possibilistic, though `GROUP_COLOURS` lists benchmark / frequentist / shrinkage groups too.

## Part 3 — After tests + move
Back to `TOOLING_TODO.md` Primary list (device fix, P3, P2/P1, rf.csv, window fixes).
