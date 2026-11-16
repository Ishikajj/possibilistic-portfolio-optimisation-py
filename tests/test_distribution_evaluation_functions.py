"""Tests for distribution_evaluation_functions.py — Gaussian predictive scoring."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

from possibilistic_portfolio.distribution_evaluation_functions import (
    mahalanobis_distance_series,
    multivariate_log_likelihood_series,
)
from possibilistic_portfolio.predictive import make_predictive


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    T, n = 30, 3
    returns = pd.DataFrame(rng.normal(0, 0.01, (T, n)), columns=list("abc"))
    mu = rng.normal(0, 0.001, (T, n))
    A = rng.normal(0, 0.01, (T, n, n))
    sigma = A @ A.transpose(0, 2, 1) + 1e-4 * np.eye(n)
    mu[:5] = np.nan  # no prediction before "investment"
    sigma[:5] = np.nan
    return returns, {"mu_hat": mu, "sigma_hat": sigma}


def test_log_likelihood_matches_scipy(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(returns, pred)
    for t in range(6, len(returns)):  # moments at t-1 score returns at t
        expected = multivariate_normal.logpdf(
            returns.iloc[t], pred["mu_hat"][t - 1], pred["sigma_hat"][t - 1]
        )
        assert np.isclose(ll.iloc[t], expected)


def test_mahalanobis_matches_scipy(setup):
    returns, pred = setup
    md = mahalanobis_distance_series(returns, pred)
    t = 10
    VI = np.linalg.inv(pred["sigma_hat"][t - 1])
    assert np.isclose(
        md.iloc[t], mahalanobis(returns.iloc[t], pred["mu_hat"][t - 1], VI)
    )


def test_squared_mahalanobis_is_square(setup):
    returns, pred = setup
    md = mahalanobis_distance_series(returns, pred)
    md2 = mahalanobis_distance_series(returns, pred, squared=True)
    assert np.allclose(md.dropna() ** 2, md2.dropna())


def test_rows_without_prediction_are_nan_and_index_kept(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(returns, pred)
    assert ll.index.equals(returns.index)
    assert (
        ll.iloc[:6].isna().all()
    )  # predictions start at 5 -> first score at 6
    assert ll.iloc[6:].notna().all()


def test_all_nan_predictions_give_all_nan_scores(setup):
    returns, pred = setup
    pred = {k: np.full_like(v, np.nan) for k, v in pred.items()}
    assert multivariate_log_likelihood_series(returns, pred).isna().all()
    assert mahalanobis_distance_series(returns, pred).isna().all()


def test_shape_mismatch_is_rejected(setup):
    returns, pred = setup
    with pytest.raises(ValueError, match="must have shapes"):
        multivariate_log_likelihood_series(returns.iloc[:, :2], pred)
    with pytest.raises(ValueError, match="must have shapes"):
        mahalanobis_distance_series(returns.iloc[:-1], pred)


def test_inputs_are_not_modified(setup):
    returns, pred = setup
    before = {k: v.copy() for k, v in pred.items()}
    multivariate_log_likelihood_series(returns, pred)
    for k in pred:
        np.testing.assert_array_equal(pred[k], before[k])


def test_unusable_sigma_rows_are_counted_in_one_warning(setup):
    returns, pred = setup
    sigma = pred["sigma_hat"].copy()
    sigma[10] = np.diag([1.0, -1.0, 1.0])  # det < 0
    sigma[11] = 0.0  # singular -> det = 0
    bad = {"mu_hat": pred["mu_hat"], "sigma_hat": sigma}
    with pytest.warns(RuntimeWarning, match=r"2 of 24 scorable rows") as rec:
        ll = multivariate_log_likelihood_series(returns, bad)
    assert len(rec) == 1
    assert "Sigma singular or det < 0: 2" in str(rec[0].message)
    assert ll.iloc[11:13].isna().all()  # moments at t-1 score row t


def test_clean_scoring_does_not_warn(setup):
    returns, pred = setup
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # rows before investment are expected
        multivariate_log_likelihood_series(returns, pred)
        mahalanobis_distance_series(returns, pred)


def test_forecast_from_other_dates_is_refused(setup):
    returns, pred = setup
    dates = pd.bdate_range("2000-01-03", periods=len(returns), name="Date")
    returns = returns.set_axis(dates)
    shifted = make_predictive(
        pred["mu_hat"], pred["sigma_hat"], dates + pd.offsets.BDay(1)
    )
    with pytest.raises(ValueError, match="forecast index does not match"):
        multivariate_log_likelihood_series(returns, shifted)
    with pytest.raises(ValueError, match="forecast index does not match"):
        mahalanobis_distance_series(returns, shifted)
