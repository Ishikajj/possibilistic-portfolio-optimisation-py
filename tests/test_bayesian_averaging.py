"""Tests for bayesian_averaging.py — probabilistic model averaging over NIW models."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_t

import bayesian_averaging as ba
from markowitz import markowitz_step

N = 3


def _niw(seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    return (
        rng.normal(size=N) * 0.01,
        2.0,
        A @ A.T * 1e-3 + 1e-3 * np.eye(N),
        N + 4.0,
    )


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (160, N)), columns=list("abc"))


# --- newborn model ---------------------------------------------------------


def test_new_model_prior_is_pooled_sample_mean_and_variance():
    rng = np.random.default_rng(1)
    R = rng.normal(0.001, 0.01, (50, N))
    mu_bar, lam_bar = ba._new_model_prior(R.sum(0), (R**2).sum(0), t=50)
    assert np.isclose(mu_bar, R.mean(0).mean())
    assert np.isclose(lam_bar, R.var(0, ddof=1).mean())


def test_run_core_prior_at_t_sees_the_t_returns_before_t(monkeypatch, returns):
    """At step t, sum_R / sum_R2 hold rows 0 … t-1 (t returns, not R[t])."""
    calls = []
    real = ba._new_model_prior

    def spy(sum_R, sum_R2, t):
        calls.append((sum_R.copy(), sum_R2.copy(), t))
        return real(sum_R, sum_R2, t)

    monkeypatch.setattr(ba, "_new_model_prior", spy)
    ba.run_core(returns, start=40, burn_in=40)
    R = returns.to_numpy()
    assert [t for *_, t in calls] == list(range(40, len(R)))
    for sum_R, sum_R2, t in calls:
        assert np.allclose(sum_R, R[:t].sum(0))
        assert np.allclose(sum_R2, (R[:t] ** 2).sum(0))


# --- NIW building blocks --------------------------------------------------


def test_marginal_likelihood_is_niw_predictive_student_t():
    mu, kappa, Lam, nu = _niw()
    x = np.array([0.01, -0.02, 0.005])
    df = nu - N + 1
    shape = Lam * (kappa + 1) / (kappa * df)
    expected = multivariate_t(loc=mu, shape=shape, df=df).logpdf(x)
    assert np.isclose(
        ba._log_marginal_likelihood(x, mu, kappa, Lam, nu), expected
    )


def test_marginal_likelihood_of_invalid_scale_is_minus_inf():
    mu, kappa, _, nu = _niw()
    assert (
        ba._log_marginal_likelihood(np.zeros(N), mu, kappa, -np.eye(N), nu)
        == -np.inf
    )


def test_niw_update_is_conjugate_posterior():
    mu, kappa, Lam, nu = _niw()
    x = np.array([0.01, -0.02, 0.005])
    mu1, k1, L1, nu1 = ba._update_niw(x, mu, kappa, Lam, nu)
    assert np.allclose(mu1, (kappa * mu + x) / (kappa + 1))
    assert k1 == kappa + 1 and nu1 == nu + 1
    assert np.allclose(L1, Lam + kappa / (kappa + 1) * np.outer(x - mu, x - mu))


def test_update_all_models_updates_in_place():
    mus, kappas, Lams, nus = map(list, zip(_niw(1), _niw(2), strict=True))
    x = np.full(N, 0.01)
    expected = [
        ba._update_niw(x, mus[m], kappas[m], Lams[m], nus[m]) for m in range(2)
    ]
    ba._update_all_models(x, mus, kappas, Lams, nus)
    for m in range(2):
        assert np.allclose(mus[m], expected[m][0])
        assert np.allclose(Lams[m], expected[m][2])


def test_append_new_model_uses_common_prior():
    mus, kappas, Lams, nus = [], [], [], []
    ba._append_new_model(
        mus, kappas, Lams, nus, mu_bar=0.002, lam_bar=1e-4, n=N
    )
    assert np.allclose(mus[0], 0.002) and kappas[0] == 1.0 and nus[0] == N + 2
    assert np.allclose(Lams[0], 1e-4 * np.eye(N))


# --- probabilities --------------------------------------------------------


def test_sharing_prior_two_models():
    # P(old) = 1 -> half of it is shared with the newborn
    assert np.allclose(ba.sharing_prior_update(np.array([1.0]), 2), [0.5, 0.5])


def test_sharing_prior_sums_to_one_and_respects_alpha():
    prev = np.array([0.5, 0.3, 0.2])
    out = ba.sharing_prior_update(prev, 4)
    assert np.isclose(out.sum(), 1.0) and (out > 0).all()
    no_share = ba.sharing_prior_update(prev, 4, alpha=0.0)
    assert np.allclose(no_share, [0.5, 0.3, 0.2, 0.0])
    assert np.allclose(ba.sharing_prior_update(np.array([]), 1), [1.0])


def test_update_probs_is_bayes_rule():
    models = [_niw(s) for s in range(3)]
    mus, kappas, Lams, nus = map(list, zip(*models, strict=True))
    prior = np.array([0.2, 0.3, 0.5])
    x = np.array([0.01, -0.02, 0.005])
    lik = np.exp([ba._log_marginal_likelihood(x, *m) for m in models])
    expected = prior * lik / (prior * lik).sum()
    assert np.allclose(
        ba._update_probs(x, mus, kappas, Lams, nus, prior), expected
    )


def test_update_probs_survives_extreme_likelihoods():
    mus, kappas, Lams, nus = map(list, zip(_niw(0), _niw(1), strict=True))
    out = ba._update_probs(
        np.full(N, 50.0), mus, kappas, Lams, nus, np.array([0.5, 0.5])
    )
    assert np.isfinite(out).all() and np.isclose(out.sum(), 1.0)


# --- moments and pruning --------------------------------------------------


def test_ba_moments_are_mixture_moments():
    models = [_niw(s) for s in range(2)]
    mus, kappas, Lams, nus = map(list, zip(*models, strict=True))
    p = np.array([0.3, 0.7])
    mu_hat, S_hat = ba._ba_moments(mus, kappas, Lams, nus, p, N)
    covs = [
        (1 + 1 / k) * L / (nu - N - 1)
        for k, L, nu in zip(kappas, Lams, nus, strict=True)
    ]
    exp_mu = p[0] * mus[0] + p[1] * mus[1]
    exp_S = sum(
        p[m] * (covs[m] + np.outer(mus[m] - exp_mu, mus[m] - exp_mu))
        for m in range(2)
    )
    assert np.allclose(mu_hat, exp_mu)
    assert np.allclose(S_hat, exp_S)


def _pool(probs):
    k = len(probs)
    mus = [np.full(N, i, dtype=float) for i in range(k)]
    return (
        mus,
        [1.0] * k,
        [np.eye(N)] * k,
        [N + 2.0] * k,
        np.asarray(probs, float),
    )


def test_prune_drops_below_threshold_and_renormalises():
    out = ba._prune_models(*_pool([0.5, 1e-9, 0.3, 0.2]), prune_threshold=1e-6)
    mus, probs = out[0], out[4]
    assert [m[0] for m in mus] == [0, 2, 3]
    assert np.isclose(probs.sum(), 1.0)


def test_prune_caps_models_but_keeps_newest():
    out = ba._prune_models(*_pool([0.4, 0.3, 0.2, 0.09, 0.01]), max_models=3)
    assert [m[0] for m in out[0]] == [0, 1, 4]  # two most probable + newest


# --- run_core -------------------------------------------------------------


def test_run_core_shapes_and_nan_layout(returns):
    pred, w = ba.run_core(returns, burn_in=40, start=60)
    T = len(returns)
    assert pred["mu_hat"].shape == (T, N) and pred["sigma_hat"].shape == (
        T,
        N,
        N,
    )
    assert (
        np.isnan(pred["mu_hat"][:40]).all()
        and np.isfinite(pred["mu_hat"][40:]).all()
    )
    assert w.iloc[:60].isna().all().all() and w.iloc[60:-1].notna().all().all()
    assert w.iloc[-1].isna().all()
    t = 100
    assert np.allclose(
        w.iloc[t], markowitz_step(pred["mu_hat"][t], pred["sigma_hat"][t])
    )


def test_run_core_covariances_are_symmetric_positive_definite(returns):
    pred, _ = ba.run_core(returns, start=40, burn_in=40)
    for S in pred["sigma_hat"][40:]:
        assert np.allclose(S, S.T)
        assert np.linalg.eigvalsh(S).min() > 0


def test_run_core_uses_returns_up_to_t_only(returns):
    """Changing returns after t must not change the prediction at t."""
    t = 100
    pred, _ = ba.run_core(returns, start=40, burn_in=40)
    changed = returns.copy()
    changed.iloc[t + 1 :] = 0.5
    pred2, _ = ba.run_core(changed, start=40, burn_in=40)
    assert np.allclose(
        pred["mu_hat"][: t + 1], pred2["mu_hat"][: t + 1], equal_nan=True
    )
    assert np.allclose(
        pred["sigma_hat"][: t + 1], pred2["sigma_hat"][: t + 1], equal_nan=True
    )
    assert not np.allclose(pred["mu_hat"][t + 1], pred2["mu_hat"][t + 1])


@pytest.mark.xfail(
    strict=True,
    reason="TOOLING_TODO bug: keep_newest evicts the highest-index model, not the least probable",
)
def test_prune_keep_newest_evicts_least_probable():
    probs = np.array([0.8, 0.1, 0.9, 1.0, 0.05])
    out = ba._prune_models(
        *_pool(probs / probs.sum()), max_models=3, prune_threshold=0
    )
    assert [m[0] for m in out[0]] == [2, 3, 4]


def test_theta_reaches_the_weights(returns):
    """theta was ignored before (always 1): unconstrained weights scale 1/theta."""
    _, w1 = ba.run_core(returns, start=60, burn_in=40, theta=1.0)
    _, w2 = ba.run_core(returns, start=60, burn_in=40, theta=2.0)
    np.testing.assert_allclose(w2.to_numpy(), w1.to_numpy() / 2.0)
