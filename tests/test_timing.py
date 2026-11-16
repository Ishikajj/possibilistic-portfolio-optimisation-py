"""Timing tests for every algorithm: what row t of a forecast / weights may use.

Convention (predictive.py): row t is decided at the end of day t, using returns
up to AND including t, and is for day t+1. Two checks per algorithm:

- no look-ahead: changing returns after t leaves rows <= t unchanged.
- uses day t:    changing only the return at t leaves rows < t unchanged and
                 changes row t.

1/N and market weights don't use returns, so they only get the look-ahead test.
(The benchmarks stopped at t-1 until 2026-11-16; the "uses day t" test caught it.)

End to end: the whole pipeline + evaluation (pipeline_wrapper, Markowitz incl.
long-only, saved files, evaluation_pipeline metrics) gets the same look-ahead
check on every row-level output.

The exact pairings are pinned in the module tests: weights row t = Markowitz
on forecast row t (test_markowitz.py), weights at t earn the returns at t+1
(test_portfolio_evaluation_functions.py), forecast at t scores the returns at
t+1 (test_distribution_evaluation_functions.py).
"""

import numpy as np
import pandas as pd
import pytest

import alternative_investment_rules as air
import bayesian_averaging as ba
import evaluation_pipeline as ev
import pipeline_wrapper as pw
import possibilistic_bayesian as pb

T, N = 150, 3
BURN, START, WIN = 40, 60, 30
T_CHECK = 100  # the day t whose rows are checked


@pytest.fixture(autouse=True)
def no_necessity(monkeypatch):
    """Necessity is being removed; all ones keeps possibilistic fast."""
    monkeypatch.setattr(
        pb, "necessity_scores_fast", lambda y_next, mus, **kw: np.ones(len(mus))
    )


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (T, N)),
        columns=["a", "b", "Mkt-RF"],
        index=pd.bdate_range("2000-01-03", periods=T, name="Date"),
    )


def _possibilistic(r):
    _, m, w_m, p, w_p, e, w_e = pb.run_core(
        r, burn_in=BURN, start=START, max_models=15, n_jobs=1
    )
    return [(m, w_m), (p, w_p), (e, w_e)]


# name -> returns_df -> list of (forecast dict or None, weights DataFrame)
ALGOS = {
    "equal_weight": lambda r: [
        (None, air.equal_weight_strategy(r, start=START))
    ],
    "market_weight": lambda r: [
        (None, air.market_weight_strategy(r, start=START))
    ],
    "minimum_variance": lambda r: [
        (None, air.minimum_variance_strategy(r, start=START, window=WIN))
    ],
    "historical_expanding": lambda r: [
        air.historical_expectations_weights(r, start=START)
    ],
    "rolling_window": lambda r: [
        air.rolling_window_weights(r, start=START, window=WIN)
    ],
    "jorion_bayes_stein": lambda r: [
        air.jorion_bayes_stein_strategy(r, start=START, window=WIN)
    ],
    "kan_zhou_three_fund": lambda r: [
        air.kan_zhou_three_fund_strategy(r, start=START, window=WIN)
    ],
    "bayesian_averaging": lambda r: [ba.run_core(r, burn_in=BURN, start=START)],
    "possibilistic": _possibilistic,
}
IGNORE_RETURNS = {"equal_weight", "market_weight"}


def _assert_same_rows(a, b, rows: slice) -> None:
    """Forecast and weights of two runs agree on `rows` (NaN == NaN)."""
    (fa, wa), (fb, wb) = a, b
    if fa is not None:
        for key in ("mu_hat", "sigma_hat"):
            assert np.allclose(fa[key][rows], fb[key][rows], equal_nan=True), (
                key
            )
    assert np.allclose(wa.iloc[rows], wb.iloc[rows], equal_nan=True), "weights"


@pytest.mark.parametrize("name", ALGOS)
def test_no_look_ahead(name, returns):
    changed = returns.copy()
    noise = np.random.default_rng(1).normal(0, 0.05, (T - T_CHECK - 1, N))
    changed.iloc[T_CHECK + 1 :] += noise

    for a, b in zip(ALGOS[name](returns), ALGOS[name](changed), strict=True):
        _assert_same_rows(a, b, slice(None, T_CHECK + 1))
        if name not in IGNORE_RETURNS:  # the change did reach later rows
            assert not np.allclose(a[1].iloc[-2], b[1].iloc[-2])


@pytest.mark.parametrize(
    "name", [name for name in ALGOS if name not in IGNORE_RETURNS]
)
def test_uses_return_at_t(name, returns):
    changed = returns.copy()
    changed.iloc[T_CHECK] += 0.05

    for a, b in zip(ALGOS[name](returns), ALGOS[name](changed), strict=True):
        _assert_same_rows(a, b, slice(None, T_CHECK))
        (fa, wa), (fb, wb) = a, b
        if fa is not None:
            assert not np.allclose(fa["mu_hat"][T_CHECK], fb["mu_hat"][T_CHECK])
        assert not np.allclose(wa.iloc[T_CHECK], wb.iloc[T_CHECK])


# --- end to end: pipeline + evaluation ------------------------------------


def _run_and_collect(data: pd.DataFrame, output_dir) -> dict[str, pd.DataFrame]:
    """Run pipeline + evaluation; every row-level output, indexed by date."""
    folder = pw.calculate_all_functions_perdatasets(
        data,
        "ds",
        burn_in=30,
        periods_until_investment=10,
        max_models=8,
        rolling_window_long=20,
        rolling_window_short=10,
        output_dir=output_dir,
    )
    ev.run_evaluation(folder, rf_series=None, rolling_window=10)

    out = {}
    for algo in ev.discover_strategies(folder):
        weights, pred = ev.load_strategy_outputs(folder, algo)
        out[f"{algo} weights"] = weights
        if pred is not None:
            T_ = len(pred["index"])
            out[f"{algo} mu_hat"] = pd.DataFrame(pred["mu_hat"], pred["index"])
            out[f"{algo} sigma_hat"] = pd.DataFrame(
                pred["sigma_hat"].reshape(T_, -1), pred["index"]
            )
    for metric in ("rolling_sharpe", "log_likelihood", "mahalanobis"):
        out[metric] = pd.read_csv(
            folder / f"{metric}.csv", index_col=0, parse_dates=True
        )
    return out


def test_whole_pipeline_has_no_look_ahead(tmp_path):
    """Changing returns after t leaves every output row labelled <= t unchanged.

    Rows are labelled by decision day (weights, forecasts) or return day
    (rolling Sharpe, log-likelihood, Mahalanobis); either way nothing at or
    before t may depend on returns after t.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("1990-01-02", periods=90, name="Date")
    data = pd.DataFrame(
        rng.normal(0.0005, 0.01, (90, 3)),
        columns=["P1", "P2", "Mkt-RF"],
        index=dates,
    )
    t = 60
    changed = data.copy()
    changed.iloc[t + 1 :] += rng.normal(0, 0.05, (90 - t - 1, 3))

    a = _run_and_collect(data, tmp_path / "a")
    b = _run_and_collect(changed, tmp_path / "b")

    assert a.keys() == b.keys()
    for name in a:
        assert np.allclose(
            a[name].loc[: dates[t]], b[name].loc[: dates[t]], equal_nan=True
        ), name
    # the change did reach the evaluation, so the check above isn't vacuous
    for metric in ("rolling_sharpe", "log_likelihood", "mahalanobis"):
        assert not np.allclose(a[metric], b[metric], equal_nan=True), metric
