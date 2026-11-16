"""Tests for possibilistic_bayesian.py — everything except necessity.

Necessity is being removed from the project, so `necessity_scores_fast` is
replaced by all ones in every run_core test: masked / power / exp then all
reduce to plain possibility weighting.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio import bayesian_averaging as ba
from possibilistic_portfolio import possibilistic_bayesian as pb
from possibilistic_portfolio.markowitz import markowitz_step

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


def _niw_pool(
    *seeds: int,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray], list[float]]:
    """Parallel lists (mus, kappas, Lambdas, nus) of the _niw(seed) models."""
    models = [_niw(s) for s in seeds]
    return (
        [m[0] for m in models],
        [m[1] for m in models],
        [m[2] for m in models],
        [m[3] for m in models],
    )


@pytest.fixture
def no_necessity(monkeypatch):
    monkeypatch.setattr(
        pb, "necessity_scores_fast", lambda y_next, mus, **kw: np.ones(len(mus))
    )


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (150, N)), columns=list("abc"))


RUN: dict[str, Any] = dict(burn_in=40, start=60, max_models=15, n_jobs=1)


# --- newborn model ---------------------------------------------------------


def test_new_model_prior_is_pooled_sample_mean_and_variance():
    rng = np.random.default_rng(1)
    R = rng.normal(0.001, 0.01, (50, N))
    mu_bar, lam_bar = pb._new_model_prior(R.sum(0), (R**2).sum(0), t=50)
    assert np.isclose(mu_bar, R.mean(0).mean())
    assert np.isclose(lam_bar, R.var(0, ddof=1).mean())


def test_new_model_prior_same_as_bayesian():
    rng = np.random.default_rng(2)
    R = rng.normal(0.001, 0.01, (30, N))
    for t in (0, 1, 2, 30):
        args = (R[:t].sum(0), (R[:t] ** 2).sum(0), t)
        assert pb._new_model_prior(*args) == ba._new_model_prior(*args)


def test_run_core_prior_at_t_sees_the_t_returns_before_t(
    no_necessity, monkeypatch, returns
):
    """At step t, sum_R / sum_R2 hold rows 0 … t-1 (t returns, not R[t])."""
    calls = []
    real = pb._new_model_prior

    def spy(sum_R, sum_R2, t):
        calls.append((sum_R.copy(), sum_R2.copy(), t))
        return real(sum_R, sum_R2, t)

    monkeypatch.setattr(pb, "_new_model_prior", spy)
    pb.run_core(returns, **RUN)
    R = returns.to_numpy()
    assert [t for *_, t in calls] == list(range(RUN["burn_in"], len(R)))
    for sum_R, sum_R2, t in calls:
        assert np.allclose(sum_R, R[:t].sum(0))
        assert np.allclose(sum_R2, (R[:t] ** 2).sum(0))


def test_append_new_model_has_full_possibility():
    mus, kappas, Lams, nus = [], [], [], []
    poss = pb._append_new_model(
        mus, kappas, Lams, nus, np.array([0.4]), 0.001, 1e-4, N
    )
    assert np.allclose(poss, [0.4, 1.0])
    assert kappas == [1.0] and nus == [1.0]
    assert np.allclose(mus[0], 0.001) and np.allclose(Lams[0], 1e-4 * np.eye(N))


# --- possibility update ----------------------------------------------------


def test_log_score_matches_docstring_formula():
    mu, kappa, Lam, nu = _niw()
    x = np.array([0.01, -0.02, 0.005])
    L1 = Lam + kappa / (kappa + 1) * np.outer(x - mu, x - mu)
    expected = (
        -N / 2 * np.log(2 * np.pi * np.e)
        + nu / 2 * np.linalg.slogdet(Lam)[1]
        - N * nu / 2 * np.log(nu)
        + N * (nu + 1) / 2 * np.log(nu + 1)
        - (nu + 1) / 2 * np.linalg.slogdet(L1)[1]
    )
    assert np.isclose(
        pb._log_marginal_likelihood(x, mu, kappa, Lam, nu), expected
    )


def test_update_possibilities_is_sup_normalised_bayes():
    models = [_niw(s) for s in range(3)]
    mus, kappas, Lams, nus = _niw_pool(*range(3))
    prior = np.array([0.2, 1.0, 0.5])
    x = np.array([0.01, -0.02, 0.005])
    lik = np.exp([pb._log_marginal_likelihood(x, *m) for m in models])
    expected = prior * lik / (prior * lik).max()
    out = pb._update_possibilities(x, mus, kappas, Lams, nus, prior)
    assert np.allclose(out, expected)
    assert np.isclose(out.max(), 1.0)


def test_update_possibilities_all_invalid_gives_ones():
    mus, kappas, _, nus = _niw_pool(0, 1)
    bad = [-np.eye(N), -np.eye(N)]
    out = pb._update_possibilities(
        np.zeros(N), mus, kappas, bad, nus, np.ones(2)
    )
    assert np.allclose(out, 1.0)


def test_niw_update_identical_to_bayesian():
    mu, kappa, Lam, nu = _niw()
    x = np.array([0.01, -0.02, 0.005])
    for a, b in zip(
        pb._update_niw(x, mu, kappa, Lam, nu),
        ba._update_niw(x, mu, kappa, Lam, nu),
        strict=True,
    ):
        assert np.allclose(a, b)


# --- pruning ---------------------------------------------------------------


def _pool(poss):
    k = len(poss)
    mus = [np.full(N, i, dtype=float) for i in range(k)]
    return (
        mus,
        [1.0] * k,
        [np.eye(N)] * k,
        [N + 2.0] * k,
        np.asarray(poss, float),
    )


def test_prune_under_cap_only_renormalises():
    out = pb._prune_models(*_pool([0.5, 0.25]), max_models=5)
    assert len(out[0]) == 2 and np.allclose(out[4], [1.0, 0.5])


def test_prune_keeps_top_by_possibility():
    out = pb._prune_models(*_pool([1.0, 0.1, 0.9, 0.2, 0.8]), max_models=3)
    assert [int(m[0]) for m in out[0]] == [0, 2, 4]
    assert np.isclose(out[4].max(), 1.0)


def test_prune_rejects_mismatched_lengths():
    mus, kappas, Lams, nus, poss = _pool([1.0, 0.5])
    with pytest.raises(ValueError):
        pb._prune_models(mus, kappas, Lams, nus[:1], poss)


@pytest.mark.xfail(
    strict=True,
    reason="TOOLING_TODO bug: keep_newest evicts the highest-index model, not the least possible",
)
def test_prune_keep_newest_evicts_least_possible():
    out = pb._prune_models(*_pool([0.8, 0.1, 0.9, 1.0, 0.05]), max_models=3)
    assert [int(m[0]) for m in out[0]] == [2, 3, 4]


# --- merging ---------------------------------------------------------------


def test_merge_two_models_combines_statistics():
    mu1, mu2 = np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, -0.3])
    L = np.eye(N)
    mu, kappa, Lam, nu, poss = pb._merge_two_models(
        mu1, 1.0, L, 20.0, 0.4, mu2, 3.0, L, 30.0, 0.9, N
    )
    d = mu1 - mu2
    assert np.allclose(mu, (1 * mu1 + 3 * mu2) / 4) and kappa == 4.0
    assert np.allclose(Lam, 2 * L + (3 / 4) * np.outer(d, d))
    assert nu == 20 + 30 - N and poss == 0.9


def test_merge_two_young_models_floors_nu():
    m = _niw()
    _, _, _, nu, _ = pb._merge_two_models(*m[:3], 1.0, 1.0, *m[:3], 1.0, 1.0, N)
    assert nu == N + 2


def test_hellinger_properties():
    mu, _, Lam, _ = _niw()
    S = Lam / 10
    assert pb._gaussian_hellinger_distance(mu, S, mu, S) == pytest.approx(
        0, abs=1e-6
    )
    other = (mu + 0.05, 2 * S)
    h12 = pb._gaussian_hellinger_distance(mu, S, *other)
    h21 = pb._gaussian_hellinger_distance(*other, mu, S)
    assert np.isclose(h12, h21) and 0 < h12 < 1


def test_hellinger_one_dimensional_closed_form():
    m1, m2, s1, s2 = 0.0, 1.0, 1.0, 2.0
    h2 = 1 - np.sqrt(2 * s1 * s2 / (s1**2 + s2**2)) * np.exp(
        -((m1 - m2) ** 2) / (4 * (s1**2 + s2**2))
    )
    got = pb._gaussian_hellinger_distance(
        np.array([m1]), np.array([[s1**2]]), np.array([m2]), np.array([[s2**2]])
    )
    assert np.isclose(got, np.sqrt(h2))


@pytest.mark.xfail(
    strict=True,
    reason="TOOLING_TODO bug: pooled covariance is S1 + S2, not (S1 + S2) / 2",
)
def test_mahalanobis_between_means():
    mu1, mu2, S1, S2 = (
        np.zeros(2),
        np.array([1.0, 2.0]),
        np.eye(2),
        3 * np.eye(2),
    )
    expected = np.sqrt((mu2 - mu1) @ np.linalg.inv((S1 + S2) / 2) @ (mu2 - mu1))
    assert np.isclose(
        pb._gaussian_mahalanobis_distance(mu1, S1, mu2, S2), expected
    )


def test_merge_core_merges_close_models_and_keeps_far_ones():
    mu, _, Lam, _ = _niw()
    mus = [mu, mu + 1e-6, mu + 1.0]
    out = pb._merge_models_hellinger(
        mus,
        [1.0, 1.0, 1.0],
        [Lam] * 3,
        [20.0, 21.0, 22.0],
        np.array([1.0, 0.5, 0.2]),
        merge_threshold=0.1,
    )
    assert len(out[0]) == 2  # the two near-identical models merged
    assert np.isclose(out[4].max(), 1.0)


def test_merge_core_respects_nu_bandwidth():
    mu, _, Lam, _ = _niw()
    out = pb._merge_models_hellinger(
        [mu, mu],
        [1.0, 1.0],
        [Lam] * 2,
        [10.0, 200.0],
        np.ones(2),
        merge_threshold=0.5,
        nu_bandwidth=50,
    )
    assert len(out[0]) == 2  # identical but too far apart in nu


# --- aggregation -----------------------------------------------------------


def test_aggregate_single_model_is_its_mode():
    mu, kappa, Lam, nu = _niw()
    mu_hat, S_hat = pb._aggregate_possibilistic_niw(
        [mu], [kappa], [Lam], [nu], np.ones(1), N
    )
    assert np.allclose(mu_hat, mu)
    assert np.allclose(S_hat, (Lam + 1e-10 * np.eye(N)) / nu)


def test_aggregate_is_invariant_to_scaling_weights():
    mus, kappas, Lams, nus = _niw_pool(*range(3))
    a = pb._aggregate_possibilistic_niw(
        mus, kappas, Lams, nus, np.array([1, 0.5, 0.2]), N
    )
    b = pb._aggregate_possibilistic_niw(
        mus, kappas, Lams, nus, np.array([2, 1.0, 0.4]), N
    )
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_aggregate_mean_is_kappa_weighted():
    mus, kappas, Lams, nus = _niw_pool(*range(2))
    kappas = [1.0, 3.0]
    w = np.array([0.5, 0.5])
    mu_hat, _ = pb._aggregate_possibilistic_niw(mus, kappas, Lams, nus, w, N)
    assert np.allclose(mu_hat, (1 * mus[0] + 3 * mus[1]) / 4)


# --- run_core (necessity bypassed) ----------------------------------------


def test_run_core_outputs(no_necessity, returns):
    diag, masked, w_m, power, w_p, exp, w_e = pb.run_core(returns, **RUN)
    T = len(returns)
    assert masked["mu_hat"].shape == (T, N) and masked["sigma_hat"].shape == (
        T,
        N,
        N,
    )
    # with necessity = 1, all three weightings are plain possibility weighting
    for other, w_o in [(power, w_p), (exp, w_e)]:
        assert np.array_equal(other["mu_hat"], masked["mu_hat"], equal_nan=True)
        pd.testing.assert_frame_equal(w_o, w_m)
    assert np.isnan(masked["mu_hat"][:40]).all()
    assert np.isfinite(masked["mu_hat"][40:]).all()
    assert list(diag.index) == list(range(40, T))
    assert (diag["n_models_post_merge"] <= diag["n_models_post_prune"]).all()
    assert (diag["n_models_post_prune"] <= RUN["max_models"]).all()
    t = 100
    assert np.allclose(
        w_m.iloc[t], markowitz_step(masked["mu_hat"][t], masked["sigma_hat"][t])
    )


def test_run_core_covariances_symmetric_positive_definite(
    no_necessity, returns
):
    _, masked, *_ = pb.run_core(returns, **RUN)
    for S in masked["sigma_hat"][40:]:
        assert np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0


def test_run_core_uses_returns_up_to_t_only(no_necessity, returns):
    t = 100
    _, a, *_ = pb.run_core(returns, **RUN)
    changed = returns.copy()
    changed.iloc[t + 1 :] = 0.5
    _, b, *_ = pb.run_core(changed, **RUN)
    assert np.allclose(
        a["mu_hat"][: t + 1], b["mu_hat"][: t + 1], equal_nan=True
    )
    assert np.allclose(
        a["sigma_hat"][: t + 1], b["sigma_hat"][: t + 1], equal_nan=True
    )


def test_run_core_weights_start_at_start(no_necessity, returns):
    """Forecasts from burn_in (warm-up), weights only from start."""
    _, masked, w_m, _, w_p, _, w_e = pb.run_core(returns, **RUN)
    burn, start = RUN["burn_in"], RUN["start"]
    assert np.isfinite(masked["mu_hat"][burn:start]).all()
    for w in (w_m, w_p, w_e):
        assert w.iloc[:start].isna().all().all()
        assert w.iloc[start:-1].notna().all().all()


def test_theta_reaches_the_weights(returns, no_necessity):
    """theta was ignored before (always 1): unconstrained weights scale 1/theta."""
    _, _, w1, *_ = pb.run_core(returns, **RUN, theta=1.0)
    _, _, w2, *_ = pb.run_core(returns, **RUN, theta=2.0)
    np.testing.assert_allclose(w2.to_numpy(), w1.to_numpy() / 2.0)
