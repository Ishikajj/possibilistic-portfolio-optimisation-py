# Dev TODO

`[x]` = resolved, `[~]` = deferred, `[ ]` = open.

## Primary — things that will bottleneck development

- [ ] **P1. Pipeline is all-or-nothing** — `calculate_all_functions_perdatasets` (`src/pipeline_wrapper.py`, 256 lines) hard-codes all 11 algorithms in sequence, including the ~90-min possibilistic run. Can't run only Bayesian averaging, or add/swap an algorithm, without editing this function. Idea: a registry of `{name: callable}` and an `algorithms=[...]` argument.
- [ ] **P2. Parameters threaded through 3 layers** — `main()` → `call_all_datasets` (17 params) → `calculate_all_functions_perdatasets` (15 params) → each algorithm. Adding one parameter means editing every layer. Idea: one config dataclass passed down.
- [ ] **P3. Bayesian averaging silently ignores pipeline settings** — `bayesian_averaging.run_core` takes no `theta` (always uses Markowitz default `theta=1`) and isn't passed `max_models` (uses its own default 2000; possibilistic gets 100). Fine today because `theta=1`, but changing `theta` would not affect Bayesian. Matters now that Bayesian averaging is the main method.
- [~] **P4. Check `_new_model_prior` denominators** — Bayesian uses `t-1`/`t-2`, possibilistic `t`/`t-1` for the prior mean/variance. Audit: `sum_R` holds `t` obs at step t, so possibilistic is correct and Bayesian is off by one. **On hold** (user's call).
- [ ] **Benchmarks use one day less info than Bayesian** — decided: all methods use returns up to and including `t` (like Bayesian). Windows `iloc[t-w:t]` → `iloc[t-w+1:t+1]` in `rolling_window_weights`, `minimum_variance_strategy`, `jorion_bayes_stein_estimates`, `kan_zhou_three_fund_estimates` (`alternative_investment_rules.py`). Changes results → fix after tests, own commit.
- [ ] **"Historical expanding" is a rolling window of `burn_in + piu` days** (`alternative_investment_rules.py:83`) — decided: make it truly expanding, `iloc[:t+1]`. Changes results → fix after tests, own commit.
- [ ] **Save `rf.csv` for real datasets too** — only simulated folders get one (`pipeline_wrapper.py:540`); real-data evaluation re-derives RF from its own hard-coded `START_DATE` (`evaluation_pipeline.py:392`), so changing the start date in only one place silently misaligns CEQ. Currently aligned (checked all 20 folders).
- [ ] Hard-coded `device="cuda"` in both `main()`s of `pipeline_wrapper.py` (see 2.6 notes) — breaks the pipeline on the Mac.
- [ ] **Sharpe-difference test** — `sharpe_difference_test` / `compare_to_benchmark` (GMM + Newey-West, 22 lags, Anderson & Cheng 2016 App. B.1) moved into `portfolio_evaluation_functions.py` (2026-11-13), imports fixed, otherwise unchanged and untested. To do: check the Jacobian / formulas against the paper, write tests, handle identical series (currently `t = nan` + warning instead of 0), then add to `evaluation_pipeline` output.
- [ ] **Bug: possibilistic starts investing at row 500, not 1000** — `possibilistic_bayesian.run_core` accepts `periods_until_investment` but passes `periods_until_investment=0` to `markowitz_unconstrained`, so weights start at `burn_in` instead of `burn_in + periods_until_investment` like every other strategy. Confirmed in `results_fyp_submitted/iid` (possibilistic first weight row 500, Bayesian 1000). Evaluation drops each strategy's own NaN rows, so FYP possibilistic Sharpe / CEQ / log-likelihood cover 500 extra days vs the others. Fix in refactor step 2.9; snapshot diff expected (possibilistic weights rows 500–999 + their eval only).
- [ ] **Bug: evaluating a new simulated run fails** — simulated CSVs were regenerated 2026-04-27 without a `Date` column, so the pipeline now writes a Date-less `rf.csv`; `evaluation_pipeline.py` `__main__` reads it with `index_col=0` → `KeyError: 'RF'` (verified). Old run folders still have `Date` and work. Fix in step 2.11.
- [ ] **Bug: singular covariance crashes the run** — `markowitz.markowitz_step` uses `np.linalg.inv`, so a singular Σ raises `LinAlgError` and stops the whole pipeline; near-singular Σ gives weights ~±1e13 with no warning; `theta=0` → `ZeroDivisionError` (long-only silently goes all-in on max μ); NaN in Σ but not μ → silent NaN weights. Decide handling in step 2.1.
- [ ] **Bug (silent failures) in `possibilistic_bayesian.run_core`** — `except Exception: pass` around aggregation leaves NaN rows with no count or warning; `burn_obs = min(burn_in, T // 2)` silently shortens the burn-in on short data. Step 2.9.

## Secondary — tooling & housekeeping

### Done
- [x] 2.1 uv — upgraded 0.7.13 → 0.13.7.
- [x] 2.2 Python pinned to 3.13 (`.python-version`); independent of CUDA build.
- [x] 2.3 Runtime deps — all used; `ipykernel` kept in runtime deps.
- [x] 2.4 Dev deps — ruff, pandas-stubs, pytest. No CLI pyright; Pylance `diagnosticMode: workspace`.
- [x] 2.5 Lockfile in sync.
- [x] 2.6 PyTorch — not adding; GPU necessity no longer planned.
  - History: Apr 2026 (`c035489`) torch 2.11.0 + CUDA 13 from PyPI, reverted (`ba2b8e4`). If GPU needed again: PyTorch CUDA index via `[tool.uv.sources]` (https://docs.astral.sh/uv/guides/integration/pytorch/).
  - Breaks without torch: `device="cuda"` runs (`src/faster_necessity.py:603`); both `main()`s in `pipeline_wrapper.py`; `comparison_cpu_gpu.py`.
  - Works without torch: Bayesian averaging, benchmarks, evaluation, notebooks, possibilistic on `device="cpu"`.
- [x] 3 Formatting — 6 files formatted; all pass `ruff format --check`.
- [x] 4 Linting — 22 safe fixes; B905 disabled. 11 left on purpose (B023 false positive in `markowitz.py`, F821 in `compute_longonly_weights.py`, scratch-notebook noise).
- [x] 5 Type checking — Pylance in VS Code, workspace mode.
- [x] 6 pytest — 3 illustrative tests in `tests/test_markowitz.py`; `.gitignore` narrowed to `src/tests/`.
- [x] 8 Pre-commit / CI — skipped for now.

### Open (low priority)
- [~] 2.7 / 7 Package layout (`src/possibilistic_portfolio/` + `[build-system]`) — removes `sys.path` hacks; nice-to-have.
- [ ] Write proper tests (loader, Bayesian averaging on tiny simulated data, …).
- [x] `src/compute_longonly_weights.py` — deleted (long-only now computed in the pipeline); Sharpe test moved to `portfolio_evaluation_functions.py`.
- [ ] `src/alternative_investment_rules.py:31` — `-> float` function can fall through without returning.
- [ ] `src/necessity_alternative_methods/annealing.py:326` — invalid `seed` argument.
- [ ] `.gitignore` hides `.vscode/`. (`comparison_cpu_gpu.py`, `compute_longonly_weights.py` deleted, lines removed.)
- [ ] `CLAUDE.md` out of date (lists removed files).
