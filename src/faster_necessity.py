from __future__ import annotations

"""Adaptive fast necessity scores with fixed-Sigma reduction and cached evaluation.

Mathematical procedure
======================
For a single model m and observed return y_{t+1}, define

    V_m = 1 - sup_{(mu, Sigma) in R^n x S_{++}^n}
              [(1 - h_bar(mu, Sigma | y_{t+1})) f_bar(mu, Sigma)].

Here

    h_bar(mu, Sigma | y)
      = exp(-0.5 (y - mu)' Sigma^{-1} (y - mu)),

and f_bar is the normalized possibilistic NIW kernel,

    f_bar(mu, Sigma)
      propto |Sigma|^{-nu/2}
              exp(-0.5 tr(Lambda Sigma^{-1}))
              exp(-0.5 kappa (mu - mu0)' Sigma^{-1} (mu - mu0)),

normalized by its analytic mode.

For fixed Sigma, write

    r = y_{t+1} - mu0,
    mu = mu0 + t r + w,
    r' Sigma^{-1} w = 0,
    z = w' Sigma^{-1} w >= 0.

Then

    (y - mu)' Sigma^{-1} (y - mu) = (1 - t)^2 d^2 + z,
    (mu - mu0)' Sigma^{-1} (mu - mu0) = t^2 d^2 + z,
    d^2 = r' Sigma^{-1} r.

So for fixed Sigma the inner optimisation reduces to

    sup_mu [(1 - h_bar) f_bar]
      = max_t max_{z >= 0}
          (1 - exp(-0.5 ((1 - t)^2 d^2 + z)))
          exp(C(Sigma) - 0.5 kappa (t^2 d^2 + z)),

where

    C(Sigma) = log f_bar(mu0, Sigma)
             = -0.5 nu log |Sigma|
               -0.5 tr(Lambda Sigma^{-1})
               - log f_mode.

For fixed t, the z-maximizer is available in closed form. Let

    A_t = exp(-0.5 (1 - t)^2 d^2).

Then

    z*(t) = 0,                                  if A_t <= kappa / (kappa + 1),
          = 2 log(A_t (kappa + 1) / kappa),    otherwise.

Hence for each fixed Sigma we only solve a 1D optimisation in t.

Adaptive covariance search
==========================
The outer supremum over Sigma is approximated adaptively.

1. Start from deterministic NIW-style seeds and inverse-Wishart samples.
2. For every Sigma candidate, compute once and cache:
       - log |Sigma|,
       - Sigma^{-1} r,
       - d^2 = r' Sigma^{-1} r,
       - tr(Lambda Sigma^{-1}),
       - C(Sigma).
   These cached quantities are reused across all t-evaluations, so the
   determinant and matrix solves are not recomputed inside the 1D search.
3. Evaluate the exact fixed-Sigma supremum over mu by coarse-grid search in t,
   followed by bounded scalar polishing.
4. Keep the elite Sigma candidates.
5. Resample new Sigma candidates by perturbing elites in packed Cholesky space,
   with perturbation scale shrinking over rounds.
6. Return
       V_m ≈ 1 - max_{screened Sigma} sup_mu [(1 - h_bar) f_bar].

Pseudocode
==========
for each model m:
    compute log_f_mode once
    seeds <- deterministic NIW covariance seeds
    candidates <- seeds + IW draws
    best_value <- 0

    for round in 0, ..., R-1:
        values <- []
        cache_list <- []
        for Sigma in candidates:
            cache <- build_cached_sigma_terms(Sigma)
            value <- max_t bracket_1d_cached(t; cache)
            store (Sigma, cache, value)
            best_value <- max(best_value, value)

        elites <- top K candidates by value
        if round < R-1:
            candidates <- elites
            while len(candidates) < target_count:
                choose elite
                perturb its Cholesky coordinates
                form new Sigma
                append candidate

    score_m <- 1 - best_value

This is still an approximation in Sigma-space, but for each fixed Sigma it uses
an exact reduction of the mu-problem and cached linear-algebra terms.
"""

import numpy as np
from numpy.linalg import LinAlgError, slogdet
from scipy.optimize import minimize_scalar
from scipy.stats import invwishart
from joblib import Parallel, delayed


def _log_possibilistic_niw_kernel(
    mu: np.ndarray,
    Sigma: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log possibilistic NIW kernel (unnormalized)."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        return -np.inf

    try:
        trace_term = float(np.trace(np.linalg.solve(Sigma, Lambda)))
        diff = mu - mu0
        quad_term = float(diff.T @ np.linalg.solve(Sigma, diff))
    except LinAlgError:
        return -np.inf

    return -0.5 * nu * logdet - 0.5 * trace_term - 0.5 * kappa * quad_term


def _niw_mode(
    mu0: np.ndarray,
    Lambda: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode of the possibilistic NIW kernel under the thesis parameterization."""
    mu_mode = np.array(mu0, dtype=float, copy=True)
    Sigma_mode = np.array(Lambda, dtype=float, copy=True) / float(nu)
    Sigma_mode = 0.5 * (Sigma_mode + Sigma_mode.T)
    return mu_mode, Sigma_mode


def _log_fbar_mode(
    mu0: np.ndarray,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log NIW kernel at its analytic mode, used for normalization."""
    mu_mode, Sigma_mode = _niw_mode(mu0, Lambda, nu)
    return _log_possibilistic_niw_kernel(
        mu_mode,
        Sigma_mode,
        mu0,
        kappa=1.0,
        Lambda=Lambda,
        nu=nu,
    )


def _deterministic_sigma_seeds(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
) -> list[np.ndarray]:
    """Construct deterministic covariance seeds."""
    n = len(mu0)
    _, Sigma_mode = _niw_mode(mu0, Lambda, nu)
    diff = y_next - mu0
    posterior_Lambda = Lambda + (kappa / (kappa + 1.0)) * np.outer(diff, diff)
    posterior_Sigma = posterior_Lambda / (nu + 1.0)
    midpoint = 0.5 * (Sigma_mode + posterior_Sigma)
    tight = 0.5 * midpoint

    seeds = [
        Sigma_mode,
        posterior_Sigma,
        midpoint,
        tight,
        sigma_floor * np.eye(n, dtype=float),
    ]

    out: list[np.ndarray] = []
    for Sigma in seeds:
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + sigma_floor * np.eye(n, dtype=float)
        out.append(Sigma)
    return out


def _sample_iw_sigmas(
    Lambda: np.ndarray,
    nu: float,
    n_samples: int,
    sigma_floor: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample covariance candidates from inverse-Wishart."""
    if n_samples <= 0:
        return []
    n = Lambda.shape[0]
    if nu <= n - 1:
        raise ValueError(
            f"nu must satisfy nu > n - 1 for inverse-Wishart sampling; got nu={nu}, n={n}."
        )

    Sigmas = invwishart.rvs(df=nu, scale=Lambda, size=n_samples, random_state=rng)
    if n_samples == 1:
        Sigmas = Sigmas[np.newaxis, :, :]

    out: list[np.ndarray] = []
    for s in range(n_samples):
        Sigma = 0.5 * (Sigmas[s] + Sigmas[s].T)
        Sigma = Sigma + sigma_floor * np.eye(n, dtype=float)
        out.append(Sigma)
    return out


def _pack_sigma_cholesky(Sigma: np.ndarray) -> np.ndarray:
    """Pack Cholesky coordinates of Sigma for adaptive perturbations."""
    chol = np.linalg.cholesky(0.5 * (Sigma + Sigma.T))
    n = Sigma.shape[0]
    pieces: list[np.ndarray] = []
    for i in range(n):
        pieces.append(np.array([np.log(max(chol[i, i], 1e-12))], dtype=float))
        if i > 0:
            pieces.append(np.asarray(chol[i, :i], dtype=float))
    return np.concatenate(pieces)


def _unpack_sigma_cholesky(theta: np.ndarray, n: int, sigma_floor: float) -> np.ndarray:
    """Unpack Cholesky coordinates back into an SPD covariance matrix."""
    theta = np.asarray(theta, dtype=float)
    L = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        L[i, i] = np.exp(theta[idx])
        idx += 1
        if i > 0:
            L[i, :i] = theta[idx : idx + i]
            idx += i
    Sigma = L @ L.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += sigma_floor * np.eye(n, dtype=float)
    return Sigma


def _build_sigma_cache(
    Sigma: np.ndarray,
    y_next: np.ndarray,
    mu0: np.ndarray,
    Lambda: np.ndarray,
    kappa: float,
    nu: float,
    log_f_mode: float,
) -> dict[str, float | np.ndarray] | None:
    """Precompute all fixed-Sigma quantities needed for repeated t evaluations.

    This is the caching step: log-determinant and linear solves are computed once
    per Sigma candidate and then reused throughout the 1D inner optimization.
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        return None

    r = y_next - mu0
    try:
        Qr = np.linalg.solve(Sigma, r)
        QLambda = np.linalg.solve(Sigma, Lambda)
    except LinAlgError:
        return None

    d2 = float(r.T @ Qr)
    trace_term = float(np.trace(QLambda))
    C = -0.5 * nu * logdet - 0.5 * trace_term - log_f_mode
    C = min(C, 0.0)

    return {
        "d2": max(d2, 0.0),
        "C": C,
    }


def _optimal_z_for_t(t: float, d2: float, kappa: float) -> float:
    """Closed-form optimizer over the Q-orthogonal radius z for fixed t."""
    if d2 <= 0.0 or kappa <= 0.0:
        return 0.0

    log_A = -0.5 * (1.0 - t) ** 2 * d2
    A_t = float(np.exp(min(log_A, 0.0)))
    threshold = kappa / (kappa + 1.0)
    if A_t <= threshold:
        return 0.0

    z_star = 2.0 * np.log(A_t * (kappa + 1.0) / kappa)
    return max(0.0, float(z_star))


def _bracket_1d_cached(t: float, d2: float, C: float, kappa: float) -> float:
    """Evaluate the exact fixed-Sigma reduced objective using cached terms."""
    z_star = _optimal_z_for_t(t, d2, kappa)
    log_h = -0.5 * ((1.0 - t) ** 2 * d2 + z_star)
    log_f = C - 0.5 * kappa * (t**2 * d2 + z_star)

    h_bar = float(np.exp(min(log_h, 0.0)))
    f_bar = float(np.exp(min(log_f, 0.0)))
    value = (1.0 - h_bar) * f_bar
    return max(0.0, min(1.0, value))


def _best_bracket_for_sigma_cache(
    cache: dict[str, float | np.ndarray],
    kappa: float,
    t_min: float = -2.5,
    t_max: float = 3.5,
    grid_size: int = 81,
) -> float:
    """Compute sup_mu[(1-h_bar)f_bar] for one fixed Sigma using cached terms."""
    d2 = float(cache["d2"])
    C = float(cache["C"])

    ts = np.linspace(t_min, t_max, grid_size)
    vals = np.array([_bracket_1d_cached(t, d2, C, kappa) for t in ts], dtype=float)
    idx = int(np.argmax(vals))
    best_val = float(vals[idx])

    left = ts[max(idx - 1, 0)]
    right = ts[min(idx + 1, grid_size - 1)]
    if right > left:
        res = minimize_scalar(
            lambda t: -_bracket_1d_cached(t, d2, C, kappa),
            bounds=(left, right),
            method="bounded",
        )
        if res.success and np.isfinite(res.fun):
            best_val = max(best_val, -float(res.fun))

    return max(0.0, min(1.0, best_val))


def _adaptive_sigma_rounds(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
    log_f_mode: float,
    *,
    initial_iw_samples: int,
    adaptive_rounds: int,
    elite_count: int,
    per_round_samples: int,
    perturb_scale: float,
    random_state: int | None,
) -> float:
    """Adaptive screening over Sigma using elite resampling in Cholesky space."""
    rng = np.random.default_rng(random_state)
    n = len(mu0)

    candidates = _deterministic_sigma_seeds(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
    )
    candidates.extend(
        _sample_iw_sigmas(
            Lambda=Lambda,
            nu=nu,
            n_samples=initial_iw_samples,
            sigma_floor=sigma_floor,
            rng=rng,
        )
    )

    best_value = 0.0

    for round_idx in range(adaptive_rounds):
        scored: list[tuple[float, np.ndarray]] = []
        for Sigma in candidates:
            cache = _build_sigma_cache(
                Sigma=Sigma,
                y_next=y_next,
                mu0=mu0,
                Lambda=Lambda,
                kappa=kappa,
                nu=nu,
                log_f_mode=log_f_mode,
            )
            if cache is None:
                continue
            value = _best_bracket_for_sigma_cache(cache, kappa=kappa)
            scored.append((value, Sigma))
            best_value = max(best_value, value)

        if not scored:
            break

        scored.sort(key=lambda pair: pair[0], reverse=True)
        elites = scored[: max(1, min(elite_count, len(scored)))]

        if round_idx == adaptive_rounds - 1:
            break

        next_candidates: list[np.ndarray] = [Sigma for _, Sigma in elites]
        theta_elites = [_pack_sigma_cholesky(Sigma) for _, Sigma in elites]
        current_scale = perturb_scale / (1.0 + round_idx)

        while len(next_candidates) < per_round_samples:
            base = theta_elites[int(rng.integers(0, len(theta_elites)))]
            noise = rng.normal(loc=0.0, scale=current_scale, size=base.shape)
            try:
                Sigma_new = _unpack_sigma_cholesky(
                    base + noise, n=n, sigma_floor=sigma_floor
                )
            except (FloatingPointError, LinAlgError, ValueError):
                continue
            next_candidates.append(Sigma_new)

        candidates = next_candidates

    return best_value


def _raw_score_fast(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
    *,
    initial_iw_samples: int,
    adaptive_rounds: int,
    elite_count: int,
    per_round_samples: int,
    perturb_scale: float,
    random_state: int | None,
) -> float:
    """Fast approximate necessity score for one NIW model."""
    y_next = np.asarray(y_next, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    mu_mode, Sigma_mode = _niw_mode(mu0, Lambda, nu)
    log_f_mode = _log_possibilistic_niw_kernel(
        mu_mode, Sigma_mode, mu0, kappa, Lambda, nu
    )
    if not np.isfinite(log_f_mode):
        return 1.0

    best_value = _adaptive_sigma_rounds(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
        log_f_mode=log_f_mode,
        initial_iw_samples=initial_iw_samples,
        adaptive_rounds=adaptive_rounds,
        elite_count=elite_count,
        per_round_samples=per_round_samples,
        perturb_scale=perturb_scale,
        random_state=random_state,
    )

    raw_score = 1.0 - best_value
    return max(0.0, float(raw_score))


def _score_single_model(
    m: int,
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    sigma_floor: float,
    initial_iw_samples: int,
    adaptive_rounds: int,
    elite_count: int,
    per_round_samples: int,
    perturb_scale: float,
    random_state: int | None,
) -> float:
    seed_m = None if random_state is None else int(random_state + m)
    return _raw_score_fast(
        y_next=y_next,
        mu0=np.asarray(mus[m], dtype=float),
        kappa=float(kappas[m]),
        Lambda=np.asarray(Lambdas[m], dtype=float),
        nu=float(nus[m]),
        sigma_floor=float(sigma_floor),
        initial_iw_samples=int(initial_iw_samples),
        adaptive_rounds=int(adaptive_rounds),
        elite_count=int(elite_count),
        per_round_samples=int(per_round_samples),
        perturb_scale=float(perturb_scale),
        random_state=seed_m,
    )


def necessity_scores_fast(
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    *,
    sigma_floor: float = 1e-8,
    initial_iw_samples: int = 128,
    adaptive_rounds: int = 3,
    elite_count: int = 12,
    per_round_samples: int = 96,
    perturb_scale: float = 0.20,
    random_state: int | None = None,
    n_jobs: int = 8,
) -> np.ndarray:
    """Compute fast approximate necessity scores for all models.

    The inner mu-optimization is exact conditional on Sigma after the reduction
    above. The outer Sigma supremum is approximated by adaptive elite-resampling
    over covariance candidates, with cached linear-algebra terms for each Sigma.
    """
    n_models = len(mus)
    if not (len(kappas) == len(Lambdas) == len(nus) == n_models):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return np.array([], dtype=float)
    if sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be strictly positive.")
    if initial_iw_samples < 0:
        raise ValueError("initial_iw_samples must be nonnegative.")
    if adaptive_rounds <= 0:
        raise ValueError("adaptive_rounds must be strictly positive.")
    if elite_count <= 0:
        raise ValueError("elite_count must be strictly positive.")
    if per_round_samples <= 0:
        raise ValueError("per_round_samples must be strictly positive.")
    if perturb_scale <= 0.0:
        raise ValueError("perturb_scale must be strictly positive.")

    y_next = np.asarray(y_next, dtype=float)
    raw_scores = np.empty(n_models, dtype=float)

    if n_jobs == 1:
        raw_scores = np.empty(n_models, dtype=float)
        for m in range(n_models):
            raw_scores[m] = _score_single_model(
                m=m,
                y_next=y_next,
                mus=mus,
                kappas=kappas,
                Lambdas=Lambdas,
                nus=nus,
                sigma_floor=float(sigma_floor),
                initial_iw_samples=int(initial_iw_samples),
                adaptive_rounds=int(adaptive_rounds),
                elite_count=int(elite_count),
                per_round_samples=int(per_round_samples),
                perturb_scale=float(perturb_scale),
                random_state=random_state,
            )
        return raw_scores

    raw_scores_list = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_score_single_model)(
            m=m,
            y_next=y_next,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            sigma_floor=float(sigma_floor),
            initial_iw_samples=int(initial_iw_samples),
            adaptive_rounds=int(adaptive_rounds),
            elite_count=int(elite_count),
            per_round_samples=int(per_round_samples),
            perturb_scale=float(perturb_scale),
            random_state=random_state,
        )
        for m in range(n_models)
    )
    return np.asarray(raw_scores_list, dtype=float)
