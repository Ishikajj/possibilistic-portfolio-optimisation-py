"""Tests for possibilistic_bayesian.py — everything except the necessity scores.

Plain possibility weighting is the default. Necessity is optional
(`necessity=True`); its scores are slow, so the tests that switch it on replace
`necessity_scores_fast` (fixture `fake_necessity`). The pieces shared with the
Bayesian method (newborn prior, NIW update, burn-in sums, top-models
selection) are tested in test_niw.py.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio import niw
from possibilistic_portfolio import possibilistic_bayesian as pb

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
def fake_necessity(monkeypatch):
    """Necessity scores in (0, 1] that differ between models (model m of M
    gets (m + 1) / M), so the three necessity schemes differ from plain."""
    monkeypatch.setattr(
        pb,
        "necessity_scores_fast",
        lambda y_next, mus, **kw: np.arange(1, len(mus) + 1) / len(mus),
    )


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (150, N)), columns=list("abc"))


RUN: dict[str, Any] = dict(burn_in=40, start=60, max_models=15)


# --- newborn model ---------------------------------------------------------


def test_run_core_prior_at_t_sees_the_t_returns_before_t(monkeypatch, returns):
    """At step t, sum_R / sum_R2 hold rows 0 … t-1 (t returns, not R[t])."""
    calls = []
    real = niw.new_model_prior

    def spy(sum_R, sum_R2, t):
        calls.append((sum_R.copy(), sum_R2.copy(), t))
        return real(sum_R, sum_R2, t)

    monkeypatch.setattr(niw, "new_model_prior", spy)
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


def test_prune_keep_newest_evicts_least_possible():
    """Was: the kept model with the highest index (here the most possible)."""
    out = pb._prune_models(*_pool([0.8, 0.1, 0.9, 1.0, 0.05]), max_models=3)
    assert [int(m[0]) for m in out[0]] == [2, 3, 4]
    assert np.allclose(out[4], [0.9, 1.0, 0.05])


def test_prune_without_keep_newest_is_plain_top_k():
    out = pb._prune_models(
        *_pool([0.8, 0.1, 0.9, 1.0, 0.05]), max_models=3, keep_newest=False
    )
    assert [int(m[0]) for m in out[0]] == [0, 2, 3]


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


def test_mahalanobis_between_means():
    """Pooled covariance is (S1 + S2) / 2 (was S1 + S2: distances √2 short)."""
    mu1, mu2, S1, S2 = (
        np.zeros(2),
        np.array([1.0, 2.0]),
        np.eye(2),
        3 * np.eye(2),
    )
    d = mu2 - mu1
    expected = np.sqrt(d @ np.linalg.solve((S1 + S2) / 2, d))
    assert np.isclose(
        pb._gaussian_mahalanobis_distance(mu1, S1, mu2, S2), expected
    )


def test_merge_core_merges_close_models_and_keeps_far_ones():
    mu, _, Lam, _ = _niw()
    mus = [mu, mu + 1e-6, mu + 1.0]
    out = pb._merge_models(
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
    out = pb._merge_models(
        [mu, mu],
        [1.0, 1.0],
        [Lam] * 2,
        [10.0, 200.0],
        np.ones(2),
        merge_threshold=0.5,
        nu_bandwidth=50,
    )
    assert len(out[0]) == 2  # identical but too far apart in nu


def test_merge_distance_is_pluggable():
    """Mahalanobis only looks at the means: same mean, very different
    covariance merges under it but not under Hellinger."""
    mu, _, Lam, _ = _niw()
    pool = ([mu, mu], [1.0, 1.0], [Lam, 50 * Lam], [20.0, 21.0], np.ones(2))
    assert len(pb._merge_models(*pool, merge_threshold=0.5)[0]) == 2
    merged = pb._merge_models(
        *pool,
        merge_threshold=0.5,
        distance_fn=pb._gaussian_mahalanobis_distance,
    )
    assert len(merged[0]) == 1


def test_merge_reports_average_distance_of_the_final_pool():
    mu, _, Lam, _ = _niw()
    S = Lam / 20.0
    out = pb._merge_models(
        [mu, mu + 1.0],
        [1.0, 1.0],
        [Lam, Lam],
        [20.0, 20.0],
        np.ones(2),
        merge_threshold=0.1,
    )
    assert len(out[0]) == 2  # nothing merged: one pair, its distance reported
    assert np.isclose(
        out[5], pb._gaussian_hellinger_distance(mu, S, mu + 1.0, S)
    )
    single = pb._merge_models([mu], [1.0], [Lam], [20.0], np.ones(1))
    assert np.isnan(single[5])


def test_closest_pair_picks_the_closest_candidates():
    mu, _, Lam, _ = _niw()
    pool = ([mu, mu + 1.0, mu + 1e-3], [Lam] * 3, [20.0, 21.0, 22.0])
    i, j, dist, avg = pb._closest_pair(
        *pool, np.ones(3), 50, 5, pb._gaussian_hellinger_distance
    )
    assert {i, j} == {0, 2} and dist < avg


def test_closest_pair_without_candidates():
    """nu too far apart: no pair is compared at all."""
    mu, _, Lam, _ = _niw()
    i, j, dist, avg = pb._closest_pair(
        [mu, mu], [Lam] * 2, [10.0, 200.0], np.ones(2), 50, 5,
        pb._gaussian_hellinger_distance,
    )  # fmt: skip
    assert (i, j, dist) == (-1, -1, np.inf) and np.isnan(avg)


def test_hellinger_non_finite_is_maximal_distance():
    """Was 0 for a NaN result (the pair would have merged)."""
    _, _, Lam, _ = _niw()
    nan_mean = np.full(N, np.nan)
    assert pb._gaussian_hellinger_distance(nan_mean, Lam, np.zeros(N), Lam) == 1


def test_sup_normalise():
    assert np.allclose(pb._sup_normalise(np.array([0.5, 2.0])), [0.25, 1.0])
    assert np.allclose(pb._sup_normalise(np.zeros(3)), 1.0)


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


# --- weighting schemes -----------------------------------------------------


def test_scheme_weights_plain_only_without_necessity():
    poss = np.array([1.0, 0.5, 0.2])
    weights = pb._scheme_weights(poss, None, gamma=1.0, eta=1.0)
    assert list(weights) == ["plain"]
    assert np.array_equal(weights["plain"], poss)


def test_scheme_weights_with_necessity_match_the_thesis_formulas():
    """Thesis eq. 5.16-5.18, before normalisation."""
    poss, nec = np.array([1.0, 0.5, 0.2]), np.array([0.0, 0.4, 1.0])
    weights = pb._scheme_weights(poss, nec, gamma=2.0, eta=3.0)
    assert list(weights) == ["plain", *pb.NECESSITY_SCHEMES]
    assert np.allclose(weights["masked"], poss * (nec > 0))
    assert np.allclose(weights["power"], poss * nec**2.0)
    assert np.allclose(weights["exp"], poss * np.exp(-3.0 * (1 - nec)))


# --- run_core ---------------------------------------------------------------


def test_run_core_outputs(returns):
    """One forecast per weighting scheme, from burn_in on (rows burn_in …
    start-1 are warm-up; weights come from the pipeline, weight_rules)."""
    diag, forecasts = pb.run_core(returns, **RUN)
    assert list(forecasts) == ["plain"]  # necessity is off by default
    plain = forecasts["plain"]
    T = len(returns)
    assert plain["mu_hat"].shape == (T, N)
    assert plain["sigma_hat"].shape == (T, N, N)
    assert np.isnan(plain["mu_hat"][:40]).all()
    assert np.isfinite(plain["mu_hat"][40:]).all()
    assert list(diag.index) == list(range(40, T))
    assert not [c for c in diag.columns if "nec" in c]
    assert (diag["n_models_post_merge"] <= diag["n_models_post_prune"]).all()
    assert (diag["n_models_post_prune"] <= RUN["max_models"]).all()


def test_run_core_does_not_score_necessity_by_default(monkeypatch, returns):
    def boom(*args, **kwargs):
        raise AssertionError("necessity scores computed with necessity off")

    monkeypatch.setattr(pb, "necessity_scores_fast", boom)
    pb.run_core(returns, **RUN)


def test_necessity_adds_schemes_and_leaves_plain_unchanged(
    fake_necessity, returns
):
    """Switching necessity on adds forecasts; it doesn't change the plain
    scheme (necessity never feeds back into the model pool)."""
    _, off = pb.run_core(returns, **RUN)
    diag, on = pb.run_core(returns, **RUN, necessity=True, n_jobs=1)
    assert list(on) == ["plain", "masked", "power", "exp"]
    plain_off, plain_on = off["plain"], on["plain"]
    for key in ("mu_hat", "sigma_hat"):
        assert np.array_equal(plain_off[key], plain_on[key], equal_nan=True)
    # scores differ between models here, so power / exp differ from plain
    for scheme in ("power", "exp"):
        other = on[scheme]
        assert np.isfinite(other["mu_hat"][40:]).all()
        assert not np.allclose(other["mu_hat"][40:], plain_on["mu_hat"][40:])
    assert {"nec_mean", "nec_max", "frac_nec_zero"} <= set(diag.columns)


def test_necessity_of_one_reduces_every_scheme_to_plain(monkeypatch, returns):
    monkeypatch.setattr(
        pb, "necessity_scores_fast", lambda y_next, mus, **kw: np.ones(len(mus))
    )
    _, forecasts = pb.run_core(returns, **RUN, necessity=True)
    plain = forecasts["plain"]
    for scheme in pb.NECESSITY_SCHEMES:
        for key in ("mu_hat", "sigma_hat"):
            assert np.array_equal(
                forecasts[scheme][key], plain[key], equal_nan=True
            )


def test_diagnostics_are_indexed_like_the_returns(returns):
    dated = returns.set_axis(
        pd.bdate_range("2000-01-03", periods=len(returns), name="Date")
    )
    diag, _ = pb.run_core(dated, **RUN)
    assert diag.index.equals(dated.index[40:]) and diag.index.name == "Date"
    assert pb.run_core(returns, **RUN)[0].index.name == "t"  # row numbers


def test_run_core_covariances_symmetric_positive_definite(returns):
    plain = pb.run_core(returns, **RUN)[1]["plain"]
    for S in plain["sigma_hat"][40:]:
        assert np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0


def test_run_core_uses_returns_up_to_t_only(returns):
    t = 100
    a = pb.run_core(returns, **RUN)[1]["plain"]
    changed = returns.copy()
    changed.iloc[t + 1 :] = 0.5
    b = pb.run_core(changed, **RUN)[1]["plain"]
    assert np.allclose(
        a["mu_hat"][: t + 1], b["mu_hat"][: t + 1], equal_nan=True
    )
    assert np.allclose(
        a["sigma_hat"][: t + 1], b["sigma_hat"][: t + 1], equal_nan=True
    )


def test_failed_aggregation_is_reported_not_swallowed(monkeypatch, returns):
    """Was `except Exception: pass`: NaN rows with no trace."""
    real = pb._aggregate_possibilistic_niw
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("Aggregated kappa must be strictly positive.")
        mu, Sigma = real(*args)
        if calls["n"] == 5:
            mu = np.full_like(mu, np.inf)
        return mu, Sigma

    monkeypatch.setattr(pb, "_aggregate_possibilistic_niw", flaky)
    with pytest.warns(RuntimeWarning) as caught:
        _, forecasts = pb.run_core(returns, **RUN)
    message = str(caught[0].message)
    assert "possibilistic: 2 of 110 forecast rows left NaN" in message
    assert "aggregation failed: 1" in message
    assert "non-finite forecast: 1" in message
    plain = forecasts["plain"]
    assert np.isnan(plain["mu_hat"][[42, 44]]).all()
    assert np.isfinite(plain["mu_hat"][[41, 43, 45]]).all()


def test_clean_run_gives_no_warning(returns):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pb.run_core(returns, **RUN)


def test_burn_in_longer_than_the_data_raises(returns):
    """Was silently shortened to half the data."""
    with pytest.raises(ValueError, match="burn_in"):
        pb.run_core(returns, start=500, burn_in=500)
