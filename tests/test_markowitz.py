"""Tests for markowitz.py — unconstrained (closed form) and long-only (SLSQP) weights."""

import numpy as np
import pandas as pd
import pytest

from markowitz import (
    compute_weights_array_long_only,
    markowitz_long_only,
    markowitz_step,
    markowitz_step_long_only,
    markowitz_unconstrained,
)


def _random_spd(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def _predictives(T: int, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "mu_hat": rng.normal(0.0, 0.1, size=(T, n)),
        "sigma_hat": np.stack([_random_spd(rng, n) for _ in range(T)]),
    }


# --- single step, unconstrained -------------------------------------------


def test_step_identity_covariance_gives_mu():
    mu = np.array([0.1, 0.2, 0.3])
    assert np.allclose(markowitz_step(mu, np.eye(3), theta=1.0), mu)


def test_step_matches_closed_form():
    rng = np.random.default_rng(1)
    mu, sigma = rng.normal(size=4), _random_spd(rng, 4)
    expected = np.linalg.solve(sigma, mu) / 2.5
    assert np.allclose(markowitz_step(mu, sigma, theta=2.5), expected)


def test_step_scales_with_one_over_theta():
    rng = np.random.default_rng(2)
    mu, sigma = rng.normal(size=3), _random_spd(rng, 3)
    w1 = markowitz_step(mu, sigma, theta=1.0)
    assert np.allclose(markowitz_step(mu, sigma, theta=4.0), w1 / 4.0)


# --- single step, long-only -----------------------------------------------


def test_long_only_weights_are_valid():
    w = markowitz_step_long_only(np.array([0.1, -0.2, 0.3]), np.eye(3))
    assert np.all(w >= -1e-9)
    assert np.isclose(w.sum(), 1.0)


def test_long_only_interior_solution():
    # Sigma = I, theta = 1: optimum has w1 - w2 = mu1 - mu2 and w1 + w2 = 1
    w = markowitz_step_long_only(np.array([0.2, 0.1]), np.eye(2), theta=1.0)
    assert np.allclose(w, [0.55, 0.45], atol=1e-6)


def test_long_only_corner_solution():
    # Unconstrained budget solution would be (1.5, -0.5) -> clipped to (1, 0)
    w = markowitz_step_long_only(np.array([2.0, 0.0]), np.eye(2), theta=1.0)
    assert np.allclose(w, [1.0, 0.0], atol=1e-6)


def test_long_only_array_matches_single_steps():
    # The array version warm-starts; it must still find the same optimum.
    T, n, first = 8, 3, 2
    pred = _predictives(T, n)
    arr = compute_weights_array_long_only(
        pred["mu_hat"], pred["sigma_hat"], 1, 1, theta=2.0
    )
    steps = [
        markowitz_step_long_only(pred["mu_hat"][t], pred["sigma_hat"][t], 2.0)
        for t in range(first, T - 1)
    ]
    assert np.allclose(arr[first : T - 1], steps, atol=1e-6)


# --- full series (both entry points) --------------------------------------

ENTRY_POINTS = [markowitz_unconstrained, markowitz_long_only]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_nan_before_investment_and_on_last_row(entry):
    T, n, burn_in, piu = 10, 3, 2, 3
    returns_df = pd.DataFrame(np.zeros((T, n)), columns=["a", "b", "c"])
    w = entry(_predictives(T, n), returns_df, burn_in, piu)

    first = burn_in + piu
    assert w.shape == (T, n)
    assert w.iloc[:first].isna().all().all()
    assert w.iloc[first:-1].notna().all().all()
    assert w.iloc[-1].isna().all()  # no return after T-1 to invest in


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_keeps_index_and_columns(entry):
    T, n = 6, 2
    idx = pd.date_range("2020-01-01", periods=T)
    returns_df = pd.DataFrame(np.zeros((T, n)), index=idx, columns=["x", "y"])
    w = entry(_predictives(T, n), returns_df, 1, 1)
    assert w.index.equals(idx)
    assert list(w.columns) == ["x", "y"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_nan_mu_row_stays_nan(entry):
    T, n = 8, 2
    pred = _predictives(T, n)
    pred["mu_hat"][4] = np.nan
    w = entry(pred, pd.DataFrame(np.zeros((T, n))), 1, 1)
    assert w.iloc[4].isna().all()
    assert w.iloc[[3, 5]].notna().all().all()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_too_short_series_is_all_nan(entry):
    T, n = 3, 2  # first investable row (1 + 1 = 2) is the last row -> nothing
    w = entry(_predictives(T, n), pd.DataFrame(np.zeros((T, n))), 1, 1)
    assert w.isna().all().all()


def test_unconstrained_matches_step_on_every_row():
    T, n = 7, 3
    pred = _predictives(T, n)
    w = markowitz_unconstrained(pred, pd.DataFrame(np.zeros((T, n))), 1, 1, 3.0)
    for t in range(2, T - 1):
        expected = markowitz_step(pred["mu_hat"][t], pred["sigma_hat"][t], 3.0)
        assert np.allclose(w.iloc[t], expected)
