"""Tests for niw.py — the NIW pieces shared by both averaging methods."""

import numpy as np
import pytest

from possibilistic_portfolio import niw

N = 3


def _niw(seed: int = 0) -> tuple[np.ndarray, float, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    return (
        rng.normal(size=N) * 0.01,
        2.0,
        A @ A.T * 1e-3 + 1e-3 * np.eye(N),
        N + 4.0,
    )


# --- burn-in ---------------------------------------------------------------


def test_burn_in_sums_cover_the_rows_before_burn_in():
    R = np.random.default_rng(0).normal(size=(20, N))
    sum_R, sum_R2 = niw.burn_in_sums(R, burn_in=5, start=8)
    assert np.allclose(sum_R, R[:5].sum(0))
    assert np.allclose(sum_R2, (R[:5] ** 2).sum(0))
    zero, zero2 = niw.burn_in_sums(R, burn_in=0, start=0)
    assert not zero.any() and not zero2.any()


@pytest.mark.parametrize(
    "burn_in, start",
    [(-1, 5), (20, 25), (30, 40), (10, 5)],
    ids=["negative", "all rows", "more than rows", "after start"],
)
def test_burn_in_that_does_not_fit_raises(burn_in, start):
    """Was silently shortened to half the data."""
    with pytest.raises(ValueError, match="burn_in"):
        niw.burn_in_sums(np.zeros((20, N)), burn_in=burn_in, start=start)


# --- newborn prior ---------------------------------------------------------


def test_new_model_prior_is_pooled_sample_mean_and_variance():
    rng = np.random.default_rng(1)
    R = rng.normal(0.001, 0.01, (50, N))
    mu_bar, lam_bar = niw.new_model_prior(R.sum(0), (R**2).sum(0), t=50)
    assert np.isclose(mu_bar, R.mean(0).mean())
    assert np.isclose(lam_bar, R.var(0, ddof=1).mean())


def test_new_model_prior_with_fewer_than_two_returns():
    R = np.array([[0.01, 0.02, 0.03]])
    assert niw.new_model_prior(np.zeros(N), np.zeros(N), t=0) == (0.0, 1e-4)
    mu_bar, lam_bar = niw.new_model_prior(R.sum(0), (R**2).sum(0), t=1)
    assert np.isclose(mu_bar, 0.02) and lam_bar == 1e-4


# --- conjugate update ------------------------------------------------------


def test_update_niw_is_conjugate_posterior():
    mu, kappa, Lam, nu = _niw()
    x = np.array([0.01, -0.02, 0.005])
    mu1, k1, L1, nu1 = niw.update_niw(x, mu, kappa, Lam, nu)
    assert np.allclose(mu1, (kappa * mu + x) / (kappa + 1))
    assert k1 == kappa + 1 and nu1 == nu + 1
    assert np.allclose(L1, Lam + kappa / (kappa + 1) * np.outer(x - mu, x - mu))


def test_update_all_models_updates_in_place():
    models = [_niw(1), _niw(2)]
    mus, kappas, Lams, nus = (list(c) for c in zip(*models, strict=True))
    x = np.full(N, 0.01)
    expected = [niw.update_niw(x, *m) for m in models]
    niw.update_all_models(x, mus, kappas, Lams, nus)
    for m in range(2):
        assert np.allclose(mus[m], expected[m][0])
        assert kappas[m] == expected[m][1] and nus[m] == expected[m][3]
        assert np.allclose(Lams[m], expected[m][2])


# --- top models (pruning) --------------------------------------------------


def test_top_models_under_the_cap_keeps_everything():
    assert list(niw.top_models(np.array([0.2, 0.9]), 5, True)) == [0, 1]


def test_top_models_keeps_the_highest_scores_in_pool_order():
    scores = np.array([1.0, 0.1, 0.9, 0.2, 0.8])
    assert list(niw.top_models(scores, 3, keep_newest=True)) == [0, 2, 4]


def test_top_models_newest_replaces_the_lowest_ranked_keeper():
    """The old rule dropped the keeper with the highest index (here model 3,
    the best one) instead of the lowest-ranked (model 0)."""
    scores = np.array([0.8, 0.1, 0.9, 1.0, 0.05])
    assert list(niw.top_models(scores, 3, keep_newest=True)) == [2, 3, 4]
    assert list(niw.top_models(scores, 3, keep_newest=False)) == [0, 2, 3]


def test_select_models_restricts_every_list():
    pool = ([np.zeros(N), np.ones(N)], [1.0, 2.0], [np.eye(N)] * 2, [5.0, 6.0])
    mus, kappas, Lams, nus = niw.select_models(*pool, np.array([1]))
    assert np.allclose(mus[0], 1.0) and kappas == [2.0] and nus == [6.0]
    assert len(Lams) == 1
