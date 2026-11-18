"""Tests for faster_necessity.py — necessity scores (CPU path) and the three
weighting schemes built on them.

Necessity is optional (off by default). The score of one model is
    V = 1 - sup_{mu, Sigma} (1 - h_bar(mu, Sigma | y)) * f_bar(mu, Sigma)
(thesis eq. 4.38). For a fixed Sigma the supremum over mu is reduced to one
dimension (eq. 5.6-5.15); the supremum over Sigma is a numerical search. The
reduction is checked here against a brute-force search over mu; the search
over Sigma only for its properties (range, reproducibility, direction).
The GPU path needs PyTorch and is not tested. The pipeline-level check with
real scores is tests/snapshots/necessity_baseline.py.
"""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize

from possibilistic_portfolio import faster_necessity as fn

N = 2
# small search settings: the tests check behaviour, not the search quality
SEARCH: dict[str, Any] = dict(
    initial_iw_samples=16,
    adaptive_rounds=2,
    elite_count=4,
    per_round_samples=16,
    n_jobs=1,
)


def _model(seed: int = 0) -> tuple[np.ndarray, float, np.ndarray, float]:
    """(mu0, kappa, Lambda, nu) of one NIW model."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    return (
        rng.normal(size=N) * 0.01,
        3.0,
        A @ A.T * 1e-3 + 1e-3 * np.eye(N),
        8.0,
    )


# Not positive definite. (The code checks the sign of the determinant, so
# the test matrix needs a negative one: -I has determinant +1 for even n.)
BAD_SIGMA = np.diag([1.0, -1.0])


def _pool(*seeds: int):
    models = [_model(s) for s in seeds]
    return tuple(list(column) for column in zip(*models, strict=True))


# --- NIW kernel and its mode ------------------------------------------------


def test_log_kernel_matches_formula():
    mu0, kappa, Lam, nu = _model()
    mu, Sigma = mu0 + 0.01, 2 * Lam / nu
    Si = np.linalg.inv(Sigma)
    expected = (
        -0.5 * nu * np.linalg.slogdet(Sigma)[1]
        - 0.5 * np.trace(Lam @ Si)
        - 0.5 * kappa * (mu - mu0) @ Si @ (mu - mu0)
    )
    got = fn._log_possibilistic_niw_kernel(mu, Sigma, mu0, kappa, Lam, nu)
    assert np.isclose(got, expected)


def test_log_kernel_of_invalid_sigma_is_minus_inf():
    mu0, kappa, Lam, nu = _model()
    got = fn._log_possibilistic_niw_kernel(mu0, BAD_SIGMA, mu0, kappa, Lam, nu)
    assert got == -np.inf


def test_mode_is_mu0_and_lambda_over_nu_and_maximises_the_kernel():
    mu0, kappa, Lam, nu = _model()
    mu_mode, Sigma_mode = fn._niw_mode(mu0, Lam, nu)
    assert np.allclose(mu_mode, mu0) and np.allclose(Sigma_mode, Lam / nu)
    at_mode = fn._log_fbar_mode(mu0, kappa, Lam, nu)
    rng = np.random.default_rng(1)
    for _ in range(50):
        B = rng.normal(size=(N, N)) * 0.02
        Sigma = Sigma_mode + B @ B.T * rng.uniform(0, 1)
        mu = mu0 + rng.normal(size=N) * 0.01
        other = fn._log_possibilistic_niw_kernel(mu, Sigma, mu0, kappa, Lam, nu)
        assert other <= at_mode + 1e-12


# --- the 1-D reduction for a fixed Sigma -----------------------------------


@pytest.mark.parametrize("t, d2, kappa", [(0.3, 2.0, 1.0), (1.0, 0.5, 4.0)])
def test_optimal_z_matches_closed_form_and_maximises(t, d2, kappa):
    """Thesis eq. 5.13."""
    A = np.exp(-0.5 * (1 - t) ** 2 * d2)
    expected = (
        0.0 if A <= kappa / (kappa + 1) else 2 * np.log(A * (kappa + 1) / kappa)
    )
    z_star = fn._optimal_z_for_t(t, d2, kappa)
    assert np.isclose(z_star, expected)

    def objective(z):
        return (1 - np.exp(-0.5 * ((1 - t) ** 2 * d2 + z))) * np.exp(
            -0.5 * kappa * (t**2 * d2 + z)
        )

    zs = np.linspace(0, 10, 20001)
    assert objective(z_star) >= objective(zs).max() - 1e-9


def test_optimal_z_when_the_return_equals_the_model_mean():
    """d2 = 0: A_t = 1 for every t, so z* = 2 log((kappa+1)/kappa) > 0.
    (The CPU code returned 0 here until 2026-11-17, which gave necessity 1
    for a return exactly at the mean; the GPU code never had that branch.)"""
    kappa = 3.0
    assert np.isclose(
        fn._optimal_z_for_t(0.5, 0.0, kappa), 2 * np.log((kappa + 1) / kappa)
    )


def test_bracket_is_one_minus_h_times_f_at_the_optimal_z():
    t, d2, C, kappa = 0.4, 1.5, -0.2, 2.0
    z = fn._optimal_z_for_t(t, d2, kappa)
    h_bar = np.exp(-0.5 * ((1 - t) ** 2 * d2 + z))
    f_bar = np.exp(C - 0.5 * kappa * (t**2 * d2 + z))
    assert np.isclose(
        fn._bracket_1d_cached(t, d2, C, kappa), (1 - h_bar) * f_bar
    )


@pytest.mark.parametrize("shift", [0.0, 0.02, 0.2])
def test_reduction_equals_brute_force_supremum_over_mu(shift):
    """For a fixed Sigma, sup over mu in R^n of (1 - h_bar) * f_bar equals the
    1-D search over t (thesis eq. 5.15). Brute force: optimiser over mu from
    many starting points."""
    mu0, kappa, Lam, nu = _model()
    Sigma = Lam / nu
    Si = np.linalg.inv(Sigma)
    y = mu0 + shift * np.array([1.0, -0.5])

    def neg_objective(mu):
        h_bar = np.exp(-0.5 * (y - mu) @ Si @ (y - mu))
        f_bar = np.exp(-0.5 * kappa * (mu - mu0) @ Si @ (mu - mu0))
        return -(1 - h_bar) * f_bar

    rng = np.random.default_rng(0)
    scale = np.sqrt(np.diag(Sigma))
    starts = [mu0 + rng.normal(size=N) * scale * 2 for _ in range(40)]
    brute = max(
        -minimize(
            neg_objective, x0, method="Nelder-Mead", options={"xatol": 1e-10}
        ).fun
        for x0 in starts
    )
    cache = fn._build_sigma_cache(
        Sigma, y, mu0, Lam, kappa, nu, fn._log_fbar_mode(mu0, kappa, Lam, nu)
    )
    assert cache is not None and np.isclose(cache["C"], 0.0)  # Sigma = mode
    assert np.isclose(cache["d2"], (y - mu0) @ Si @ (y - mu0))
    reduced = fn._best_bracket_for_sigma_cache(cache, kappa)
    assert reduced == pytest.approx(brute, abs=1e-5)


def test_sigma_cache_of_invalid_sigma_is_none():
    mu0, kappa, Lam, nu = _model()
    cache = fn._build_sigma_cache(BAD_SIGMA, mu0, mu0, Lam, kappa, nu, 0.0)
    assert cache is None


# --- candidates for Sigma ---------------------------------------------------


def test_cholesky_packing_round_trip():
    _, _, Lam, nu = _model()
    Sigma = Lam / nu
    theta = fn._pack_sigma_cholesky(Sigma)
    assert theta.shape == (N * (N + 1) // 2,)
    back = fn._unpack_sigma_cholesky(theta, N, sigma_floor=1e-8)
    assert np.allclose(back, Sigma + 1e-8 * np.eye(N))


def test_deterministic_seeds_are_symmetric_positive_definite():
    mu0, kappa, Lam, nu = _model()
    seeds = fn._deterministic_sigma_seeds(mu0 + 0.05, mu0, kappa, Lam, nu, 1e-8)
    assert len(seeds) == 5
    assert np.allclose(seeds[0], Lam / nu + 1e-8 * np.eye(N))  # the mode
    for Sigma in seeds:
        assert np.allclose(Sigma, Sigma.T)
        assert np.linalg.eigvalsh(Sigma).min() > 0


def test_inverse_wishart_samples():
    _, _, Lam, nu = _model()
    rng = np.random.default_rng(0)
    samples = fn._sample_iw_sigmas(Lam, nu, 7, 1e-8, rng)
    assert len(samples) == 7
    for Sigma in samples:
        assert Sigma.shape == (N, N) and np.linalg.eigvalsh(Sigma).min() > 0
    assert fn._sample_iw_sigmas(Lam, nu, 0, 1e-8, rng) == []
    # inverse-Wishart needs nu > n - 1 (newborn possibilistic models: nu = 1)
    assert fn._sample_iw_sigmas(Lam, N - 1.0, 5, 1e-8, rng) == []
    one = fn._sample_iw_sigmas(Lam, nu, 1, 1e-8, rng)  # shape quirk of rvs
    assert len(one) == 1 and one[0].shape == (N, N)


# --- necessity scores -------------------------------------------------------


def test_scores_are_in_unit_interval_and_reproducible():
    mus, kappas, Lams, nus = _pool(0, 1, 2)
    y = np.array([0.01, -0.02])
    a = fn.necessity_scores_fast(
        y, mus, kappas, Lams, nus, random_state=5, **SEARCH
    )
    b = fn.necessity_scores_fast(
        y, mus, kappas, Lams, nus, random_state=5, **SEARCH
    )
    assert a.shape == (3,) and ((a >= 0) & (a <= 1)).all()
    assert np.array_equal(a, b)


def test_return_far_from_the_model_has_no_necessity():
    """Far away, (1 - h_bar) is ~1 at the mode, where f_bar = 1: V = 0."""
    mu0, kappa, Lam, nu = _model()
    far = mu0 + 100 * np.sqrt(np.diag(Lam / nu))
    score = fn.necessity_scores_fast(
        far, [mu0], [kappa], [Lam], [nu], random_state=0, **SEARCH
    )
    assert score[0] == pytest.approx(0.0, abs=1e-6)


def test_return_at_the_model_mean_scores_highest_and_respects_its_bound():
    """At y = mu0 the candidate Sigma = mode alone gives
    sup >= (1 / (kappa + 1)) * (kappa / (kappa + 1)) ** kappa, an upper bound
    on V (up to the 1e-8 floor added to every candidate); the score falls as
    the return moves away."""
    mu0, kappa, Lam, nu = _model()
    step = np.sqrt(np.diag(Lam / nu))
    scores = [
        fn.necessity_scores_fast(
            mu0 + k * step,
            [mu0],
            [kappa],
            [Lam],
            [nu],
            random_state=0,
            **SEARCH,
        )[0]
        for k in (0.0, 1.0, 3.0)
    ]
    bound = 1 - (1 / (kappa + 1)) * (kappa / (kappa + 1)) ** kappa
    assert 0 < scores[0] <= bound + 1e-6
    assert scores[0] > scores[1] > scores[2]


def test_more_search_never_raises_the_score():
    """The score is 1 - (best value found), and the small search's candidates
    for the first round are a subset of the larger one's seeds, so a longer
    search can only find a larger supremum."""
    mu0, kappa, Lam, nu = _model()
    y = mu0 + 0.5 * np.sqrt(np.diag(Lam / nu))
    small = fn.necessity_scores_fast(
        y,
        [mu0],
        [kappa],
        [Lam],
        [nu],
        random_state=0,
        initial_iw_samples=0,
        adaptive_rounds=1,
        elite_count=1,
        per_round_samples=1,
        n_jobs=1,
    )
    large = fn.necessity_scores_fast(
        y, [mu0], [kappa], [Lam], [nu], random_state=0, **SEARCH
    )
    assert large[0] <= small[0] + 1e-12


def test_joblib_workers_give_the_same_scores():
    mus, kappas, Lams, nus = _pool(0, 1, 2)
    y = np.array([0.01, -0.02])
    serial = fn.necessity_scores_fast(
        y, mus, kappas, Lams, nus, random_state=3, **SEARCH
    )
    parallel = fn.necessity_scores_fast(
        y, mus, kappas, Lams, nus, random_state=3, **{**SEARCH, "n_jobs": 2}
    )
    assert np.array_equal(serial, parallel)


def test_empty_pool_and_bad_arguments():
    y = np.zeros(N)
    assert fn.necessity_scores_fast(y, [], [], [], []).size == 0
    mus, kappas, Lams, nus = _pool(0, 1)
    with pytest.raises(ValueError, match="same length"):
        fn.necessity_scores_fast(y, mus, kappas[:1], Lams, nus)
    bad_settings: list[dict[str, Any]] = [
        {"sigma_floor": 0.0},
        {"initial_iw_samples": -1},
        {"adaptive_rounds": 0},
        {"elite_count": 0},
        {"per_round_samples": 0},
        {"perturb_scale": 0.0},
    ]
    for bad in bad_settings:
        with pytest.raises(ValueError):
            fn.necessity_scores_fast(y, mus, kappas, Lams, nus, **bad)


# --- weighting schemes (thesis eq. 5.16-5.18, before normalisation) --------

POSS = np.array([1.0, 0.5, 0.2])
NEC = np.array([0.0, 0.4, 1.0])


def test_mask_keeps_models_with_positive_necessity():
    out = fn.deterministic_mask_weighting(NEC, POSS)
    assert np.allclose(out, [0.0, 0.5, 0.2])


def test_power_weighting():
    out = fn.power_weighting(NEC, POSS, gamma=2.0, epsilon=0.1)
    assert np.allclose(out, POSS * (NEC + 0.1) ** 2.0)


def test_exponential_penalty_weighting():
    out = fn.exponential_penalty_weighting(NEC, POSS, eta=3.0)
    assert np.allclose(out, POSS * np.exp(-3.0 * (1 - NEC)))
    # eta = 0: no penalty
    assert np.allclose(fn.exponential_penalty_weighting(NEC, POSS, eta=0), POSS)


def test_necessity_of_one_leaves_possibilities_unchanged():
    ones = np.ones(3)
    for out in (
        fn.deterministic_mask_weighting(ones, POSS),
        fn.power_weighting(ones, POSS),
        fn.exponential_penalty_weighting(ones, POSS),
    ):
        assert np.array_equal(out, POSS)


def test_schemes_fall_back_to_possibilities_when_everything_is_masked():
    zeros = np.zeros(3)
    assert np.array_equal(fn.deterministic_mask_weighting(zeros, POSS), POSS)
    assert np.array_equal(fn.power_weighting(zeros, POSS), POSS)


def test_schemes_reject_bad_arguments():
    with pytest.raises(ValueError, match="same length"):
        fn.deterministic_mask_weighting(NEC[:2], POSS)
    with pytest.raises(ValueError, match="gamma"):
        fn.power_weighting(NEC, POSS, gamma=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        fn.power_weighting(NEC, POSS, epsilon=-1.0)
    with pytest.raises(ValueError, match="eta"):
        fn.exponential_penalty_weighting(NEC, POSS, eta=-1.0)
