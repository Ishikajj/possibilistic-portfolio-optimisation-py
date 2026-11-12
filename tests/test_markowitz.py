"""Illustrative tests for markowitz.py — placeholders until proper tests are written."""

import numpy as np
import pandas as pd

from markowitz import (
    markowitz_step,
    markowitz_step_long_only,
    markowitz_unconstrained,
)


def test_identity_covariance_gives_mu():
    # w = (1/theta) * inv(Sigma) @ mu, so Sigma = I and theta = 1 give w == mu
    mu = np.array([0.1, 0.2, 0.3])
    w = markowitz_step(mu, np.eye(3), theta=1.0)
    assert np.allclose(w, mu)


def test_long_only_weights_are_valid():
    mu = np.array([0.1, -0.2, 0.3])
    w = markowitz_step_long_only(mu, np.eye(3))
    assert np.all(w >= -1e-9)
    assert np.isclose(w.sum(), 1.0)


def test_unconstrained_nan_before_investment_and_last_row():
    T, n = 10, 3
    burn_in, periods_until_investment = 2, 3
    mu_sigma_dict = {
        "mu_hat": np.full((T, n), 0.1),
        "sigma_hat": np.tile(np.eye(n), (T, 1, 1)),
    }
    returns_df = pd.DataFrame(np.zeros((T, n)), columns=["a", "b", "c"])

    weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, burn_in, periods_until_investment
    )

    first = burn_in + periods_until_investment
    assert weights.shape == (T, n)
    assert weights.iloc[:first].isna().all().all()
    assert weights.iloc[first:-1].notna().all().all()
    assert weights.iloc[-1].isna().all()  # no return after T-1 to invest in
