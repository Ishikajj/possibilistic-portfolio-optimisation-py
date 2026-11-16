"""Tests for alternative_investment_rules.py — benchmark strategies.

Window timing: estimates at t use the w returns up to and including t, i.e.
iloc[t-w+1 : t+1] (`_window_rows`). Until 2026-11-16 they stopped at t-1.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from possibilistic_portfolio import alternative_investment_rules as air
from possibilistic_portfolio.markowitz import markowitz_step

T, N = 120, 3
BURN, PIU = 40, 10
FIRST = BURN + PIU  # = start passed to every strategy


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cov = 1e-4 * (0.3 * np.ones((N, N)) + 0.7 * np.eye(N))
    data = rng.multivariate_normal([0.001, 0.0005, 0.0002], cov, size=T)
    return pd.DataFrame(data, columns=["a", "b", "Mkt-RF"])


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


# --- weights-only strategies ---------------------------------------------


def test_equal_weight(returns):
    w = air.equal_weight_strategy(returns, start=FIRST)
    _assert_nan_layout(w, FIRST)
    assert np.allclose(w.iloc[FIRST:-1], 1.0 / N)


def test_market_weight(returns):
    w = air.market_weight_strategy(returns, start=FIRST)
    _assert_nan_layout(w, FIRST)
    assert (w["Mkt-RF"].iloc[FIRST:-1] == 1.0).all()
    assert (w[["a", "b"]].iloc[FIRST:-1] == 0.0).all().all()


def test_market_weight_needs_market_column(returns):
    with pytest.raises(KeyError):
        air.market_weight_strategy(returns.drop(columns="Mkt-RF"), start=FIRST)


def test_minimum_variance(returns):
    win = 30
    w = air.minimum_variance_strategy(returns, start=FIRST, window=win)
    _assert_nan_layout(w, FIRST)
    assert np.allclose(w.iloc[FIRST:-1].sum(axis=1), 1.0)
    t = 70
    S = _window_rows(returns, t, win).cov().to_numpy()
    expected = np.linalg.solve(S, np.ones(N))
    assert np.allclose(w.iloc[t], expected / expected.sum())


def test_minimum_variance_window_longer_than_burn_in_delays_start(returns):
    w = air.minimum_variance_strategy(returns, start=FIRST, window=80)
    _assert_nan_layout(w, 80)


# --- moment-based strategies ----------------------------------------------


def test_historical_moments_and_weights(returns):
    pred, w = air.historical_expectations_weights(returns, start=FIRST)
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
    pred, w = air.rolling_window_weights(returns, start=FIRST, window=win)
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
    pred, w = air.jorion_bayes_stein_strategy(
        returns, start=FIRST, window=win, theta=2.0
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


def test_kan_zhou_estimates(returns):
    win = 30
    est = air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=win)
    t = 70
    rows = _window_rows(returns, t, win)
    S = rows.cov(ddof=0).to_numpy()
    assert np.allclose(est["Sigma"][t], S)  # MLE covariance (divide by w)
    _, mu_g_vec, _ = air._minvar_mean_target(
        rows.mean().to_numpy(), S, np.ones(N)
    )
    v = est["v"][t]
    assert np.allclose(
        est["mu_star"][t], (1 - v) * rows.mean().to_numpy() + v * mu_g_vec
    )
    h = ((win - N - 1) * (win - N - 4)) / (win * (win - 2))
    assert np.isclose(est["h"][0], h)


def test_kan_zhou_strategy_scales_mean_by_h(returns):
    win = 30
    est = air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=win)
    pred, w = air.kan_zhou_three_fund_strategy(
        returns, start=FIRST, window=win, theta=2.0
    )
    assert np.allclose(
        pred["mu_hat"], est["h"][0] * est["mu_star"], equal_nan=True
    )
    _assert_nan_layout(w, FIRST)
    t = 80
    expected = (
        est["h"][0] / 2.0 * np.linalg.solve(est["Sigma"][t], est["mu_star"][t])
    )
    assert np.allclose(w.iloc[t], expected)


def test_kan_zhou_rejects_too_short_window(returns):
    with pytest.raises(ValueError):
        air.kan_zhou_three_fund_estimates(returns, start=FIRST, window=N + 4)
