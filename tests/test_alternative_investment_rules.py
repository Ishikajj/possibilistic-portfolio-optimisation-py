"""Tests for alternative_investment_rules.py — benchmark strategies.

Window timing: estimates at t use the w returns up to and including t, i.e.
iloc[t-w+1 : t+1] (`_window_rows`). Until 2026-11-16 they stopped at t-1.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from possibilistic_portfolio import alternative_investment_rules as air
from possibilistic_portfolio.markowitz import markowitz_step
from possibilistic_portfolio.predictive import StrategyOutput

T, N = 120, 3
BURN, PIU = 40, 10
FIRST = BURN + PIU  # = start passed to every strategy


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cov = 1e-4 * (0.3 * np.ones((N, N)) + 0.7 * np.eye(N))
    data = rng.multivariate_normal([0.001, 0.0005, 0.0002], cov, size=T)
    return pd.DataFrame(data, columns=["a", "b", "Mkt-RF"])


def _fw(out: StrategyOutput) -> tuple[dict, pd.DataFrame]:
    """(forecast, weights) of a predictive strategy."""
    assert out.forecast is not None
    return out.forecast, out.weights


def _window_rows(returns: pd.DataFrame, t: int, w: int) -> pd.DataFrame:
    """Rows used for the estimate at t: the w returns up to and including t."""
    return returns.iloc[t - w + 1 : t + 1]


def _assert_nan_layout(weights: pd.DataFrame, first: int) -> None:
    assert weights.iloc[:first].isna().all().all()
    assert weights.iloc[first:-1].notna().all().all()
    assert weights.iloc[-1].isna().all()


# --- helpers --------------------------------------------------------------


def test_incomplete_beta_matches_integral():
    a, b, x = 1.5, 20.0, 0.3
    expected, _ = quad(lambda s: s ** (a - 1) * (1 - s) ** (b - 1), 0, x)
    assert np.isclose(air._incomplete_beta_unregularized(a, b, x), expected)


def test_incomplete_beta_outside_unit_interval_is_nan():
    assert np.isnan(air._incomplete_beta_unregularized(1.0, 2.0, 1.5))


def test_solve_or_pinv_falls_back_on_singular():
    A = np.ones((2, 2))
    b = np.array([2.0, 2.0])
    assert np.allclose(air._solve_or_pinv(A, b), np.linalg.pinv(A) @ b)


def test_minvar_mean_target_is_gmv_portfolio_mean():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(N, N))
    S, mu, ones = A @ A.T + np.eye(N), rng.normal(size=N), np.ones(N)
    w_gmv = np.linalg.solve(S, ones) / (ones @ np.linalg.solve(S, ones))
    mu_g, mu_g_vec, denom = air._minvar_mean_target(mu, S, ones)
    assert np.isclose(mu_g, w_gmv @ mu)
    assert np.allclose(mu_g_vec, mu_g)
    assert np.isclose(denom, ones @ np.linalg.solve(S, ones))


def test_shrinkage_target_psi2_and_undefined_target():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(N, N))
    S, mu, ones = A @ A.T + np.eye(N), rng.normal(size=N), np.ones(N)
    mu_g, mu_g_vec, denom, psi2 = air._shrinkage_target(mu, S, ones)
    d = mu - mu_g_vec
    assert np.isclose(psi2, d @ np.linalg.solve(S, d))
    assert (mu_g, denom) == air._minvar_mean_target(mu, S, ones)[::2]
    # 1' Sigma^{-1} 1 = 0 → no target
    assert air._shrinkage_target(mu, np.zeros((N, N)), ones) is None


def test_kan_zhou_adjusted_psi2_edge_cases(monkeypatch):
    assert air._kan_zhou_adjusted_psi2(0.0, n=N, w=30) == 0.0
    assert air._kan_zhou_adjusted_psi2(1e-13, n=N, w=30) == 0.0
    assert air._kan_zhou_adjusted_psi2(0.5, n=N, w=30) > 0.0
    # incomplete beta of 0 used to divide by zero → NaN (row skipped)
    monkeypatch.setattr(air, "_incomplete_beta_unregularized", lambda **kw: 0.0)
    assert np.isnan(air._kan_zhou_adjusted_psi2(0.5, n=N, w=30))


# --- weights-only strategies ---------------------------------------------


def test_equal_weight(returns):
    w = air.equal_weight_strategy(returns, start=FIRST).weights
    _assert_nan_layout(w, FIRST)
    assert np.allclose(w.iloc[FIRST:-1], 1.0 / N)


def test_market_weight(returns):
    w = air.market_weight_strategy(returns, start=FIRST).weights
    _assert_nan_layout(w, FIRST)
    assert (w["Mkt-RF"].iloc[FIRST:-1] == 1.0).all()
    assert (w[["a", "b"]].iloc[FIRST:-1] == 0.0).all().all()


def test_market_weight_needs_market_column(returns):
    with pytest.raises(KeyError):
        air.market_weight_strategy(returns.drop(columns="Mkt-RF"), start=FIRST)


def test_minimum_variance(returns):
    win = 30
    w = air.minimum_variance_strategy(returns, start=FIRST, window=win).weights
    _assert_nan_layout(w, FIRST)
    assert np.allclose(w.iloc[FIRST:-1].sum(axis=1), 1.0)
    t = 70
    S = _window_rows(returns, t, win).cov().to_numpy()
    expected = np.linalg.solve(S, np.ones(N))
    assert np.allclose(w.iloc[t], expected / expected.sum())


def test_minimum_variance_window_longer_than_burn_in_delays_start(returns):
    w = air.minimum_variance_strategy(returns, start=FIRST, window=80).weights
    _assert_nan_layout(w, 80)


def _with_dead_window(returns: pd.DataFrame, win: int) -> pd.DataFrame:
    """Rows 60 … 99 set to zero: the windows ending at 60+win-1 … 99 have a
    zero covariance, which no benchmark can invert."""
    dead = returns.copy()
    dead.iloc[60:100] = 0.0
    return dead


def test_minimum_variance_reports_rows_it_cannot_compute(returns):
    """Was np.linalg.inv: a singular window stopped the whole run."""
    win = 30
    with pytest.warns(RuntimeWarning, match="minimum_variance: 11 of 69 rows"):
        w = air.minimum_variance_strategy(
            _with_dead_window(returns, win), start=FIRST, window=win
        ).weights
    assert w.iloc[89:100].isna().all().all()
    assert w.iloc[FIRST:89].notna().all().all()


# --- moment-based strategies ----------------------------------------------


def test_historical_moments_and_weights(returns):
    pred, w = _fw(air.historical_expectations_weights(returns, start=FIRST))
    t = 70
    rows = returns.iloc[: t + 1]  # expanding: every return up to and incl. t
    assert np.allclose(pred["mu_hat"][t], rows.mean())
    assert np.allclose(pred["sigma_hat"][t], rows.cov())
    _assert_nan_layout(w, FIRST)
    assert np.allclose(
        w.iloc[t], markowitz_step(pred["mu_hat"][t], pred["sigma_hat"][t])
    )


def test_rolling_window_moments(returns):
    win = 25
    pred, w = _fw(air.rolling_window_weights(returns, start=FIRST, window=win))
    t = 90
    rows = _window_rows(returns, t, win)
    assert np.allclose(pred["mu_hat"][t], rows.mean())
    assert np.allclose(pred["sigma_hat"][t], rows.cov())
    assert np.isnan(pred["mu_hat"][FIRST - 1]).all()
    _assert_nan_layout(w, FIRST)


def _jorion_by_hand(rows: pd.DataFrame):
    """Jorion (1986) written out independently from the paper."""
    w, n = rows.shape
    mu = rows.mean().to_numpy()
    S = (
        rows.cov(ddof=0).to_numpy() * w / (w - n - 2)
    )  # sum of squares / (w-n-2)
    ones = np.ones(n)
    Si = np.linalg.inv(S)
    mu_g = (ones @ Si @ mu) / (ones @ Si @ ones)
    d = mu - mu_g
    lam = (n + 2) / (d @ Si @ d)
    v = lam / (w + lam)
    mu_star = (1 - v) * mu + v * mu_g
    S_star = S * (1 + 1 / (w + lam)) + lam / (w * (w + 1 + lam)) * np.outer(
        ones, ones
    ) / (ones @ Si @ ones)
    return mu_star, S_star, v


def test_jorion_matches_paper_formulas(returns):
    win = 30
    est = air.jorion_bayes_stein_estimates(returns, start=FIRST, window=win)
    t = 70
    mu_star, S_star, v = _jorion_by_hand(_window_rows(returns, t, win))
    assert np.allclose(est["mu_star"][t], mu_star)
    assert np.allclose(est["sigma_star"][t], S_star)
    assert np.isclose(est["v"][t], v)


def test_jorion_shrinkage_is_convex_and_strategy_uses_it(returns):
    win = 30
    est = air.jorion_bayes_stein_estimates(returns, start=FIRST, window=win)
    v = est["v"][FIRST:-1]
    assert ((v > 0) & (v < 1)).all()
    pred, w = _fw(
        air.jorion_bayes_stein_strategy(
            returns, start=FIRST, window=win, theta=2.0
        )
    )
    assert np.allclose(pred["mu_hat"], est["mu_star"], equal_nan=True)
    _assert_nan_layout(w, FIRST)
    t = 80
    assert np.allclose(
        w.iloc[t], markowitz_step(est["mu_star"][t], est["sigma_star"][t], 2.0)
    )


def test_jorion_rejects_too_short_window(returns):
    with pytest.raises(ValueError):
        air.jorion_bayes_stein_estimates(returns, start=FIRST, window=N + 2)


def test_jorion_and_kan_zhou_forecast_every_row_from_start(returns):
    """The last row has a forecast (for the day after the data) like every
    other predictive strategy; only its weights are NaN."""
    for strategy in (
        air.jorion_bayes_stein_strategy,
        air.kan_zhou_three_fund_strategy,
    ):
        pred, w = _fw(strategy(returns, start=FIRST, window=30))
        assert np.isfinite(pred["mu_hat"][FIRST:]).all()
        assert np.isfinite(pred["sigma_hat"][FIRST:]).all()
        _assert_nan_layout(w, FIRST)


def test_jorion_reports_rows_it_cannot_estimate(returns):
    win = 30
    with pytest.warns(RuntimeWarning) as caught:
        est = air.jorion_bayes_stein_estimates(
            _with_dead_window(returns, win), start=FIRST, window=win
        )
    message = str(caught[0].message)
    assert "jorion_bayes_stein: 11 of 70 rows left NaN" in message
    assert "minimum-variance mean undefined: 11" in message
    assert np.isnan(est["mu_star"][89:100]).all()
    assert np.isfinite(est["mu_star"][FIRST:89]).all()


def _kan_zhou_by_hand(rows: pd.DataFrame):
    """Kan & Zhou (2007) eq. 66 / 75 written out independently (T = w, N = n);
    the incomplete beta function by numerical integration."""
    w, n = rows.shape
    mu = rows.mean().to_numpy()
    S = rows.cov(ddof=0).to_numpy()
    ones = np.ones(n)
    Si = np.linalg.inv(S)
    mu_g = (ones @ Si @ mu) / (ones @ Si @ ones)
    psi2 = (mu - mu_g) @ Si @ (mu - mu_g)
    a, b = (n - 1) / 2, (w - n + 1) / 2
    B, _ = quad(
        lambda y: y ** (a - 1) * (1 - y) ** (b - 1), 0, psi2 / (1 + psi2)
    )
    psi2_a = ((w - n - 1) * psi2 - (n - 1)) / w + 2 * psi2 ** ((n - 1) / 2) * (
        1 + psi2
    ) ** (-(w - 2) / 2) / (w * B)
    v = n / (n + w * psi2_a)
    return (1 - v) * mu + v * mu_g, S, v


def test_kan_zhou_matches_paper_formulas(returns):
    win = 30
    est = air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=win)
    for t in (70, 95):
        mu_star, S, v = _kan_zhou_by_hand(_window_rows(returns, t, win))
        assert np.isclose(est["v"][t], v)
        assert np.allclose(est["mu_star"][t], mu_star)
        assert np.allclose(est["Sigma"][t], S)  # MLE covariance (divide by w)
    h = ((win - N - 1) * (win - N - 4)) / (win * (win - 2))
    assert np.isclose(est["h"][0], h)
    assert np.allclose(est["sigma_star"], est["Sigma"] / h, equal_nan=True)


def test_kan_zhou_shrinkage_weight_is_between_0_and_1(returns):
    """The adjusted estimator of psi^2 is positive, so v can't leave (0, 1)."""
    est = air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=30)
    v = est["v"][FIRST:]
    assert ((v > 0) & (v < 1)).all()


def test_kan_zhou_forecast_is_mu_star_and_sigma_over_h(returns):
    """The saved mean is mu* itself (was h * mu*, which the scoring and the
    long-only weights then used); h sits in the covariance: Sigma / h."""
    win = 30
    est = air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=win)
    pred, w = _fw(
        air.kan_zhou_three_fund_strategy(
            returns, start=FIRST, window=win, theta=2.0
        )
    )
    h = est["h"][0]
    assert np.allclose(pred["mu_hat"], est["mu_star"], equal_nan=True)
    assert np.allclose(pred["sigma_hat"], est["Sigma"] / h, equal_nan=True)
    _assert_nan_layout(w, FIRST)
    t = 80
    # the three-fund rule: (h / theta) Sigma^{-1} mu*
    expected = h / 2.0 * np.linalg.solve(est["Sigma"][t], est["mu_star"][t])
    assert np.allclose(w.iloc[t], expected)


def test_kan_zhou_reports_rows_it_cannot_estimate(returns):
    win = 30
    with pytest.warns(RuntimeWarning) as caught:
        est = air.kan_zhou_three_fund_estimates(
            _with_dead_window(returns, win), start=FIRST, window=win
        )
    message = str(caught[0].message)
    assert "kan_zhou_three_fund: 11 of 70 rows left NaN" in message
    assert "minimum-variance mean undefined: 11" in message
    assert np.isnan(est["mu_star"][89:100]).all()


def test_clean_benchmark_runs_give_no_warning(returns):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        air.minimum_variance_strategy(returns, start=FIRST, window=30)
        air.jorion_bayes_stein_strategy(returns, start=FIRST, window=30)
        air.kan_zhou_three_fund_strategy(returns, start=FIRST, window=30)


def test_kan_zhou_rejects_too_short_window(returns):
    with pytest.raises(ValueError):
        air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=N + 4)
