"""Naive Monte Carlo sampling approach to necessity score computation. Not used in the main pipeline.

Abandoned because MC sampling cannot be made dense enough in high dimensions —
required sample size grows exponentially with n, causing necessity scores to be
systematically underestimated.

This is especially true because we are not calculating an expectation, but an extremum.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError, slogdet
from scipy.stats import invwishart


def _log_normal_likelihood(
    y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray
) -> float:
    """Log multivariate normal likelihood of y given (mu, Sigma)."""
    n = len(y)
    Sigma = 0.5 * (Sigma + Sigma.T)
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        return -np.inf

    diff = y - mu
    try:
        quad = float(diff.T @ np.linalg.solve(Sigma, diff))
    except LinAlgError:
        return -np.inf

    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)


def _log_possibilistic_niw_kernel(
    mu: np.ndarray,
    Sigma: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log possibilistic NIW kernel used in possibilistic_bayesian.py.

    Kernel:
        |Sigma|^{-nu/2}
        exp(-0.5 tr(Lambda Sigma^{-1}))
        exp(-0.5 (mu-mu0)' kappa Sigma^{-1} (mu-mu0))
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        return -np.inf

    try:
        Sigma_inv_Lambda = np.linalg.solve(Sigma, Lambda)
        trace_term = float(np.trace(Sigma_inv_Lambda))

        diff = mu - mu0
        quad_term = float(diff.T @ np.linalg.solve(Sigma, diff))
    except LinAlgError:
        return -np.inf

    return -0.5 * nu * logdet - 0.5 * trace_term - 0.5 * kappa * quad_term


def _sample_probabilistic_niw(
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (mu, Sigma) from the probabilistic NIW analogue.

    We use the standard probabilistic NIW sampling law:
        Sigma ~ IW(Lambda, nu)
        mu | Sigma ~ N(mu0, Sigma / kappa)

    This is used as a Monte Carlo device for the possibilistic kernel because the
    probability and possibility forms share the same exponential-quadratic
    structure in (mu, Sigma).
    """
    n = len(mu0)
    if n_samples <= 0:
        raise ValueError("n_samples must be strictly positive.")
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive.")
    if nu <= n - 1:
        raise ValueError(
            f"nu must satisfy nu > n - 1 for inverse-Wishart sampling; got nu={nu}, n={n}."
        )

    Sigmas = invwishart.rvs(
        df=nu, scale=Lambda, size=n_samples, random_state=rng
    )
    if n_samples == 1:
        Sigmas = Sigmas[np.newaxis, :, :]

    mus = np.empty((n_samples, n), dtype=float)
    for s in range(n_samples):
        Sigma_s = 0.5 * (Sigmas[s] + Sigmas[s].T)
        mus[s] = rng.multivariate_normal(mean=mu0, cov=Sigma_s / kappa)

    return mus, Sigmas


def _raw_internal_validity_score(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo approximation of the unnormalized necessity-style score.

    Approximates
        1 - sup_{mu,Sigma} (1 - h_bar(y_next | mu, Sigma)) f_bar(mu, Sigma | F_t)

    by replacing the supremum over the continuous parameter space with the maximum
    over NIW Monte Carlo draws. Both h_bar and f_bar are normalized to have sample
    supremum one.
    """
    mus, Sigmas = _sample_probabilistic_niw(
        mu0, kappa, Lambda, nu, n_samples, rng
    )

    log_h = np.array(
        [
            _log_normal_likelihood(y_next, mus[s], Sigmas[s])
            for s in range(n_samples)
        ],
        dtype=float,
    )
    log_f = np.array(
        [
            _log_possibilistic_niw_kernel(
                mus[s], Sigmas[s], mu0, kappa, Lambda, nu
            )
            for s in range(n_samples)
        ],
        dtype=float,
    )

    finite_h = np.isfinite(log_h)
    finite_f = np.isfinite(log_f)
    finite = finite_h & finite_f
    if not finite.any():
        return 0.0

    h_bar = np.zeros(n_samples, dtype=float)
    f_bar = np.zeros(n_samples, dtype=float)
    h_bar[finite] = np.exp(log_h[finite] - np.max(log_h[finite]))
    f_bar[finite] = np.exp(log_f[finite] - np.max(log_f[finite]))

    bracket = (1.0 - h_bar) * f_bar
    raw_score = 1.0 - float(np.max(bracket))
    return max(raw_score, 0.0)


def necessity_scores(
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    *,
    n_samples: int = 2000,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute normalized necessity-style scores for all active models.

    For each model m, we approximate the score
        V_m^raw = 1 - sup (1 - h_bar_m) f_bar_m

    and then normalize by the maximal raw score across models:
        V_m = V_m^raw / max_q V_q^raw.

    This matches the denominator structure in the thesis expression by choosing q
    as the model with the largest raw internal-validity score.
    """
    n_models = len(mus)
    if not (len(kappas) == len(Lambdas) == len(nus) == n_models):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return np.array([], dtype=float)

    y_next = np.asarray(y_next, dtype=float)
    rng = np.random.default_rng(random_state)

    raw_scores = np.empty(n_models, dtype=float)
    for m in range(n_models):
        raw_scores[m] = _raw_internal_validity_score(
            y_next=y_next,
            mu0=np.asarray(mus[m], dtype=float),
            kappa=float(kappas[m]),
            Lambda=np.asarray(Lambdas[m], dtype=float),
            nu=float(nus[m]),
            n_samples=n_samples,
            rng=rng,
        )

    max_raw = raw_scores.max(initial=0.0)
    if max_raw <= 0.0:
        return np.ones(n_models, dtype=float)
    return raw_scores / max_raw


def necessity_weighted_possibilities(
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    *,
    n_samples: int = 2000,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine necessity scores with current possibilities.

    Returns
    -------
    necessities : np.ndarray
        Model-wise normalized necessity-style scores.
    adjusted_possibilities : np.ndarray
        Elementwise product of necessities and current possibilities, normalized
        to have supremum one.
    """
    possibilities = np.asarray(possibilities, dtype=float)
    necessities = necessity_scores(
        y_next=y_next,
        mus=mus,
        kappas=kappas,
        Lambdas=Lambdas,
        nus=nus,
        n_samples=n_samples,
        random_state=random_state,
    )

    if len(necessities) != len(possibilities):
        raise ValueError(
            "necessities and possibilities must have the same length."
        )

    adjusted = necessities * possibilities
    max_adjusted = adjusted.max(initial=0.0)
    if max_adjusted <= 0.0:
        adjusted = np.ones_like(adjusted)
    else:
        adjusted = adjusted / max_adjusted

    return necessities, adjusted
