"""Tests for portfolio_evaluation_functions.py — portfolio returns, Sharpe, CE, turnover, Sharpe test."""

import warnings

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio.portfolio_evaluation_functions import (
    SharpeTestResult,
    _newey_west_cov,
    average_turnover,
    calculate_sharpe_ratio,
    certainty_equivalent,
    compare_to_benchmark,
    portfolio_returns,
    rolling_sharpe_ratio,
    sharpe_difference_test,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.001, 0.01, size=(300, 2)), columns=["a", "b"]
    )


@pytest.fixture
def weights(returns) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    w.iloc[50:-1] = [0.6, 0.4]  # NaN before investment and on the last row
    return w


# --- portfolio_returns ----------------------------------------------------


def test_weights_at_t_apply_to_returns_at_t_plus_1():
    r = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.0, 0.0, 0.0]})
    w = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.0, 0.0, np.nan]})
    port = portfolio_returns(r, w)
    # r_1 * w_0 = 0.2 * 1, r_2 * w_1 = 0.3 * 2
    assert list(port.index) == [1, 2]
    assert np.allclose(port, [0.2, 0.6])


def test_portfolio_returns_keeps_nan_rows_when_asked(returns, weights):
    port = portfolio_returns(returns, weights, dropna=False)
    assert len(port) == len(returns)
    assert port.iloc[:51].isna().all()  # first return earned is at row 51
    assert port.iloc[51:].notna().all()


def test_portfolio_returns_uses_weight_columns_order(returns, weights):
    swapped = returns[["b", "a"]]
    pd.testing.assert_series_equal(
        portfolio_returns(swapped, weights), portfolio_returns(returns, weights)
    )


# --- Sharpe ---------------------------------------------------------------


def test_sharpe_matches_mean_over_std(returns, weights):
    port = portfolio_returns(returns, weights)
    expected = port.mean() / port.std(ddof=1)
    assert np.isclose(calculate_sharpe_ratio(returns, weights), expected)


def test_sharpe_annualised_by_sqrt_scaling(returns, weights):
    daily = calculate_sharpe_ratio(returns, weights)
    assert np.isclose(
        calculate_sharpe_ratio(returns, weights, 252), daily * np.sqrt(252)
    )
    roll = rolling_sharpe_ratio(returns, weights, window=40)
    assert np.allclose(
        rolling_sharpe_ratio(returns, weights, window=40, scaling_factor=252),
        roll * np.sqrt(252),
        equal_nan=True,
    )


def test_rolling_sharpe_last_value_matches_window(returns, weights):
    roll = rolling_sharpe_ratio(returns, weights, window=40)
    port = portfolio_returns(returns, weights, dropna=False)
    last = port.iloc[-40:]
    assert len(roll) == len(returns)
    assert np.isclose(roll.iloc[-1], last.mean() / last.std(ddof=1))
    assert roll.iloc[: 51 + 39].isna().all()  # needs a full window of returns


def test_sharpe_single_return_is_nan_with_warning(returns):
    w = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    w.iloc[10] = [1.0, 0.0]  # one invested period -> one return
    with pytest.warns(RuntimeWarning, match="1 valid return"):
        assert np.isnan(calculate_sharpe_ratio(returns, w))


def test_sharpe_constant_returns_is_nan_with_warning(weights):
    const = pd.DataFrame(0.25, index=weights.index, columns=weights.columns)
    with pytest.warns(RuntimeWarning, match="std = 0"):
        assert np.isnan(calculate_sharpe_ratio(const, weights))


def test_rolling_sharpe_counts_zero_std_windows(returns, weights):
    r = returns.copy()
    r.iloc[200:] = 0.25  # last windows see constant portfolio returns
    with pytest.warns(RuntimeWarning, match="zero std") as rec:
        roll = rolling_sharpe_ratio(r, weights, window=40)
    n_zero = int(str(rec[0].message).split()[1])
    assert n_zero == len(r) - (200 + 39)  # windows fully inside 200..T-1
    assert roll.iloc[200 + 39 :].isna().all()
    assert not np.isinf(roll).any()


@pytest.mark.parametrize(
    "call",
    [
        lambda r, w: rolling_sharpe_ratio(r, w, window=0),
        lambda r, w: rolling_sharpe_ratio(r, w, window=10, min_periods=11),
        lambda r, w: certainty_equivalent(
            r, w, pd.Series(0.0, index=r.index), theta=-1.0
        ),
    ],
    ids=["window=0", "min_periods>window", "theta<0"],
)
def test_bad_inputs_raise_value_error_with_message(returns, weights, call):
    with pytest.raises(ValueError, match="must"):
        call(returns, weights)


# --- certainty equivalent -------------------------------------------------


def test_certainty_equivalent_formula(returns, weights):
    rf = pd.Series(0.0001, index=returns.index)
    port = portfolio_returns(returns, weights)
    expected = port.mean() + 0.0001 - (3.0 / 2.0) * port.var(ddof=1)
    assert np.isclose(
        certainty_equivalent(returns, weights, rf, theta=3.0), expected
    )


def test_certainty_equivalent_uses_rf_on_return_dates(returns, weights):
    # RF differs before / after the first return date (row 51)
    rf = pd.Series(
        np.where(returns.index < 51, 1.0, 0.0001), index=returns.index
    )
    port = portfolio_returns(returns, weights)
    expected = port.mean() + 0.0001 - 0.5 * port.var(ddof=1)
    assert np.isclose(certainty_equivalent(returns, weights, rf), expected)


# --- turnover -------------------------------------------------------------


def test_turnover_on_fully_invested_weights():
    w = pd.DataFrame({"a": [1.0, 0.0, 1.0, 1.0], "b": [0.0, 1.0, 0.0, 0.0]})
    # changes: 2, 2, 0 -> mean 4/3
    assert np.isclose(average_turnover(w), 4.0 / 3.0)
    assert np.isclose(average_turnover(w, scaling_factor=252), 252 * 4.0 / 3.0)


def test_turnover_ignores_rows_without_weights():
    nan = np.nan
    w = pd.DataFrame(
        {
            "a": [nan, nan, 1.0, 0.7, 1.0, nan],
            "b": [nan, nan, 0.0, 0.3, 0.0, nan],
        }
    )
    # only rows 3 and 4 have a previous weight: 0.6, 0.6
    assert np.isclose(average_turnover(w), 0.6)


def test_turnover_constant_weights_is_zero():
    w = pd.DataFrame({"a": [0.5] * 5, "b": [0.5] * 5})
    assert average_turnover(w) == 0.0


# --- Sharpe-difference test (moved from compute_longonly_weights.py) ------


def test_newey_west_with_zero_lags_is_sample_covariance():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(500, 3))
    assert np.allclose(_newey_west_cov(m, 0), np.cov(m, rowvar=False, ddof=0))


def test_newey_west_is_symmetric():
    rng = np.random.default_rng(2)
    S = _newey_west_cov(rng.normal(size=(500, 4)), n_lags=5)
    assert np.allclose(S, S.T)


def test_sharpe_difference_point_estimates():
    rng = np.random.default_rng(3)
    a, b = (
        pd.Series(rng.normal(0.002, 0.01, 2000)),
        pd.Series(rng.normal(0.0, 0.01, 2000)),
    )
    res = sharpe_difference_test(a, b)
    assert isinstance(res, SharpeTestResult)
    assert np.isclose(res.sharpe_strategy, a.mean() / a.std(ddof=1))
    assert np.isclose(res.sharpe_benchmark, b.mean() / b.std(ddof=1))
    assert np.isclose(
        res.difference, res.sharpe_strategy - res.sharpe_benchmark
    )


def test_sharpe_difference_detects_clear_outperformance_and_is_antisymmetric():
    rng = np.random.default_rng(4)
    a, b = (
        pd.Series(rng.normal(0.003, 0.01, 3000)),
        pd.Series(rng.normal(0.0, 0.01, 3000)),
    )
    ab, ba = sharpe_difference_test(a, b), sharpe_difference_test(b, a)
    assert ab.outperforms and not ab.underperforms
    assert ba.underperforms and not ba.outperforms
    assert np.isclose(ab.t_statistic, -ba.t_statistic)
    assert 0.0 <= ab.p_approx < 0.05


def test_sharpe_difference_identical_series_is_no_difference():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning
        res = sharpe_difference_test(r, r.copy())
    assert res.difference == 0.0
    assert res.t_statistic == 0.0 and res.p_approx == 1.0
    assert not res.outperforms and not res.underperforms


def test_sharpe_difference_aligns_on_common_index():
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.001, 0.01, 500))
    b = pd.Series(rng.normal(0.0, 0.01, 500))
    res_full = sharpe_difference_test(a.iloc[100:], b)
    res_cut = sharpe_difference_test(a.iloc[100:], b.iloc[100:])
    assert res_full == res_cut


def test_sharpe_difference_rejects_too_few_observations():
    with pytest.raises(ValueError):
        sharpe_difference_test(
            pd.Series(np.ones(10) * 0.01), pd.Series(np.ones(10) * 0.02)
        )


def _other_weights(weights):
    w = weights.copy()
    w.iloc[50:-1] = [0.1, 0.9]
    return w


def test_compare_to_benchmark_one_row_per_strategy(returns, weights):
    out = compare_to_benchmark(
        returns, weights, {"x": _other_weights(weights), "y": weights * -1}
    )
    assert list(out.index) == ["x", "y"]
    assert out.shape[1] == len(SharpeTestResult._fields)


def test_compare_to_benchmark_has_named_columns(returns, weights):
    out = compare_to_benchmark(returns, weights, {"x": _other_weights(weights)})
    assert list(out.columns) == list(SharpeTestResult._fields)


# --- date index -------------------------------------------------------------


def test_sharpe_difference_and_ce_match_by_date():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2000-01-03", periods=300, name="Date")
    a = pd.Series(rng.normal(0.001, 0.01, 300), index=dates)
    b = pd.Series(rng.normal(0.0, 0.01, 300), index=dates)
    # benchmark starts 50 days later: only the shared dates are compared
    res = sharpe_difference_test(a, b.iloc[50:])
    assert np.isclose(
        res.sharpe_strategy, a.iloc[50:].mean() / a.iloc[50:].std()
    )

    returns = pd.DataFrame({"x": a, "y": b})
    w = pd.DataFrame(0.5, index=dates, columns=["x", "y"])
    rf = pd.Series(1e-4, index=dates)
    certainty_equivalent(returns, w, rf)  # RF found for every return date
    with pytest.raises(KeyError):  # a return date without RF fails loudly
        certainty_equivalent(returns, w, rf.drop(dates[100]))
