"""Tests for distribution_evaluation_functions.py — Gaussian predictive scoring."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

from distribution_evaluation_functions import (
    average_log_likelihood,
    mahalanobis_distance_series,
    multivariate_log_likelihood_series,
)


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


def test_log_likelihood_matches_scipy_with_lag_1(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(returns, pred, regularization=0.0)
    for t in range(6, len(returns)):  # moments at t-1 score returns at t
        expected = multivariate_normal.logpdf(
            returns.iloc[t], pred["mu_hat"][t - 1], pred["sigma_hat"][t - 1]
        )
        assert np.isclose(ll.iloc[t], expected)


def test_mahalanobis_matches_scipy(setup):
    returns, pred = setup
    md = mahalanobis_distance_series(returns, pred, regularization=0.0)
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


def test_lag_0_scores_same_row(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(
        returns, pred, lag=0, regularization=0.0
    )
    t = 10
    expected = multivariate_normal.logpdf(
        returns.iloc[t], pred["mu_hat"][t], pred["sigma_hat"][t]
    )
    assert np.isclose(ll.iloc[t], expected)


def test_rows_without_prediction_are_nan_and_index_kept(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(returns, pred)
    assert ll.index.equals(returns.index)
    assert (
        ll.iloc[:6].isna().all()
    )  # predictions start at 5 -> first score at 6
    assert ll.iloc[6:].notna().all()


def test_average_log_likelihood_is_mean_of_valid_rows(setup):
    returns, pred = setup
    ll = multivariate_log_likelihood_series(returns, pred)
    assert np.isclose(average_log_likelihood(returns, pred), ll.dropna().mean())


def test_average_log_likelihood_all_nan_is_nan(setup):
    returns, pred = setup
    pred = {k: np.full_like(v, np.nan) for k, v in pred.items()}
    assert np.isnan(average_log_likelihood(returns, pred))


def test_shape_mismatch_is_rejected(setup):
    returns, pred = setup
    with pytest.raises(AssertionError):
        multivariate_log_likelihood_series(returns.iloc[:, :2], pred)
