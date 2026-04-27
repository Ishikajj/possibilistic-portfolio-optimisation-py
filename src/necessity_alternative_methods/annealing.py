"""Simulated annealing approach to necessity score computation. Not used in the main pipeline.

Abandoned because  — the annealing schedule is inherently sequential,
each candidate depending on the current temperature and previously accepted state.
While independent chains could be run in parallel across models,
each chain remains sequential and the per-model compute cost
is too high
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError, slogdet
from scipy.optimize import dual_annealing, minimize
from scipy.stats import invwishart

"""too slow."""


# correct
def _log_possibilistic_niw_kernel(
    mu: np.ndarray,
    Sigma: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log possibilistic NIW kernel.

    Kernel:
        |Sigma|^{-nu/2}
        exp(-0.5 tr(Lambda Sigma^{-1}))
        exp(-0.5 (mu-mu0)' kappa Sigma^{-1} (mu-mu0))

        note that the -nv is constant, so is the v in the denominator
    """
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


# makes sense
def _niw_mode(
    mu0: np.ndarray, Lambda: np.ndarray, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mode of the possibilistic NIW kernel under the thesis parameterization."""
    mu_mode = np.array(mu0, dtype=float, copy=True)
    Sigma_mode = np.array(Lambda, dtype=float, copy=True) / float(nu)
    Sigma_mode = 0.5 * (Sigma_mode + Sigma_mode.T)
    return mu_mode, Sigma_mode


# makes sense
def _pack_theta(mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Pack mean vector and Cholesky-factor parameters into one vector."""
    n = len(mu)
    out = [np.asarray(mu, dtype=float)]
    for i in range(n):
        out.append(np.array([np.log(max(chol[i, i], 1e-12))], dtype=float))
        if i > 0:
            out.append(np.asarray(chol[i, :i], dtype=float))
    return np.concatenate(out)


# makes sense
def _unpack_theta(
    theta: np.ndarray, n: int, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack parameter vector into (mu, Sigma).

    Sigma is parameterized by a lower-triangular Cholesky factor with log-diagonal
    entries to guarantee positive definiteness.
    """
    theta = np.asarray(theta, dtype=float)
    mu = theta[:n]
    L = np.zeros((n, n), dtype=float)
    idx = n
    for i in range(n):
        L[i, i] = np.exp(theta[idx])
        idx += 1
        if i > 0:
            L[i, :i] = theta[idx : idx + i]
            idx += i

    Sigma = L @ L.T
    if ridge > 0.0:
        Sigma += ridge * np.eye(n, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma


def _theta_bounds(
    y_next: np.ndarray,
    mu0: np.ndarray,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
    mu_radius: float = 8.0,
    chol_offdiag_scale: float = 6.0,
    logdiag_pad: float = 4.0,
) -> list[tuple[float, float]]:
    """Construct box bounds for dual annealing in packed-theta coordinates.

    The search variable is theta = (mu, packed Cholesky parameters). Because
    scipy's dual_annealing requires finite box constraints, we build pragmatic
    bounds centered around the NIW structure.
    """
    y_next = np.asarray(y_next, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = len(mu0)

    Sigma_mode = 0.5 * (Lambda + Lambda.T) / float(nu)
    Sigma_mode += sigma_floor * np.eye(n, dtype=float)
    chol_mode = np.linalg.cholesky(Sigma_mode)

    center = 0.5 * (mu0 + y_next)
    spread = np.sqrt(np.maximum(np.diag(Sigma_mode), sigma_floor))
    mu_lower = center - mu_radius * spread
    mu_upper = center + mu_radius * spread

    bounds: list[tuple[float, float]] = []
    for i in range(n):
        bounds.append((float(mu_lower[i]), float(mu_upper[i])))

    for i in range(n):
        logdiag_mode = float(np.log(max(chol_mode[i, i], 1e-12)))
        bounds.append((logdiag_mode - logdiag_pad, logdiag_mode + logdiag_pad))
        if i > 0:
            row_scale = chol_offdiag_scale * float(
                max(chol_mode[i, i], np.sqrt(sigma_floor))
            )
            for _ in range(i):
                bounds.append((-row_scale, row_scale))

    return bounds


# correct but this is not "normal"
def _log_hbar(
    y: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Log normalized Gaussian likelihood."""

    diff = y - mu
    try:
        quad = float(diff.T @ np.linalg.solve(Sigma, diff))
    except LinAlgError:
        return -np.inf
    log_h = -0.5 * quad
    if not np.isfinite(log_h):
        return -np.inf

    return min(log_h, 0.0)


# correct
def _log_fbar(
    mu: np.ndarray,
    Sigma: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log normalized possibilistic NIW kernel using its analytic mode."""
    log_f = _log_possibilistic_niw_kernel(mu, Sigma, mu0, kappa, Lambda, nu)
    if not np.isfinite(log_f):
        return -np.inf

    mu_mode, Sigma_mode = _niw_mode(mu0, Lambda, nu)
    log_f_mode = _log_possibilistic_niw_kernel(
        mu_mode, Sigma_mode, mu0, kappa, Lambda, nu
    )
    if not np.isfinite(log_f_mode):
        return -np.inf

    # since the kernel isnt the exact formula, you need to then divide by the kernel at the mode to normalise.
    return min(log_f - log_f_mode, 0.0)


# makes sense
def _bracket_value(
    mu: np.ndarray,
    Sigma: np.ndarray,
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Evaluate (1 - h_bar) f_bar at a deterministic candidate point."""
    log_hbar = _log_hbar(y_next, mu, Sigma)
    log_fbar = _log_fbar(mu, Sigma, mu0, kappa, Lambda, nu)
    if not (np.isfinite(log_hbar) and np.isfinite(log_fbar)):
        return 0.0

    h_bar = float(np.exp(log_hbar))
    f_bar = float(np.exp(log_fbar))
    value = (1.0 - h_bar) * f_bar
    return min(
        max(value, 0.0), 1.0
    )  # make sure the value in the bracket is between 1 and 0.


# makes sense
def _deterministic_initial_thetas(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
) -> list[np.ndarray]:
    """Construct analytic deterministic starting points for local optimization. These are generally prior and posterior moments, midpoints and a couple other."""
    n = len(mu0)
    mu_mode, Sigma_mode = _niw_mode(mu0, Lambda, nu)
    posterior_mu = (kappa * mu0 + y_next) / (kappa + 1.0)
    diff = y_next - mu0
    posterior_Lambda = Lambda + (kappa / (kappa + 1.0)) * np.outer(diff, diff)
    posterior_Sigma = posterior_Lambda / (nu + 1.0)

    starts: list[tuple[np.ndarray, np.ndarray]] = []
    starts.append((mu_mode, Sigma_mode))
    starts.append((posterior_mu, posterior_Sigma))
    starts.append((y_next.copy(), sigma_floor * np.eye(n, dtype=float)))
    starts.append((0.5 * (mu_mode + y_next), Sigma_mode))
    starts.append((posterior_mu, 0.5 * (Sigma_mode + posterior_Sigma)))

    thetas: list[np.ndarray] = []
    for mu_start, Sigma_start in starts:
        Sigma_start = 0.5 * (Sigma_start + Sigma_start.T)
        Sigma_start += sigma_floor * np.eye(n, dtype=float)
        chol = np.linalg.cholesky(Sigma_start)
        thetas.append(_pack_theta(mu_start, chol))
    return thetas


def _annealing_start_theta(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
) -> np.ndarray:
    """Use deterministic NIW structure to provide a reasonable annealing start."""
    starts = _deterministic_initial_thetas(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
    )
    return starts[0]


# makes sense
def _raw_internal_validity_score_deterministic(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
    *,
    mc_samples: int = 0,
    top_k: int = 0,
    rng: np.random.Generator | None = None,
) -> float:
    """Approximate the corrected necessity score with scipy dual annealing.

    The target is
        V_m = 1 - sup_{mu, Sigma} (1 - h_bar_m) f_bar_m.

    We optimize directly over packed-theta coordinates using scipy's
    dual_annealing under pragmatic finite bounds, then optionally polish with
    L-BFGS-B starting from the annealing solution.
    """
    y_next = np.asarray(y_next, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = len(mu0)
    if rng is None:
        rng = np.random.default_rng()

    def objective(theta: np.ndarray) -> float:
        mu, Sigma = _unpack_theta(theta, n=n, ridge=sigma_floor)
        value = _bracket_value(mu, Sigma, y_next, mu0, kappa, Lambda, nu)
        return -value

    bounds = _theta_bounds(
        y_next=y_next,
        mu0=mu0,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
    )
    x0 = _annealing_start_theta(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
    )

    anneal_res = dual_annealing(
        objective,
        bounds=bounds,
        x0=x0,
        seed=rng,
        no_local_search=True,
    )

    candidate_thetas: list[np.ndarray] = []
    if anneal_res.x is not None and np.all(np.isfinite(anneal_res.x)):
        candidate_thetas.append(np.asarray(anneal_res.x, dtype=float))
    candidate_thetas.append(x0)

    best_theta = min(candidate_thetas, key=objective)
    local_res = minimize(
        objective, best_theta, method="L-BFGS-B", bounds=bounds
    )
    if local_res.success and np.all(np.isfinite(local_res.x)):
        candidate_thetas.append(np.asarray(local_res.x, dtype=float))

    best_value = 0.0
    for theta in candidate_thetas:
        mu, Sigma = _unpack_theta(theta, n=n, ridge=sigma_floor)
        value = _bracket_value(
            mu,
            Sigma,
            y_next,
            mu0,
            kappa,
            Lambda,
            nu,
        )
        best_value = max(best_value, value)

    raw_score = 1.0 - best_value
    return max(raw_score, 0.0)


def necessity_scores_deterministic(
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    *,
    sigma_floor: float = 1e-8,
    mc_samples: int = 0,
    top_k: int = 0,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute corrected model-wise necessity scores with a hybrid pipeline.

    Each score is the corrected quantity
        V_m = 1 - sup (1 - h_bar_m) f_bar_m.

    The optimization is deterministic once the starting points are fixed. The
    optional Monte Carlo part is used only to supply stronger initial points for
    deterministic local optimization.
    """
    n_models = len(mus)
    if not (len(kappas) == len(Lambdas) == len(nus) == n_models):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return np.array([], dtype=float)
    if sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be strictly positive.")
    if mc_samples < 0:
        raise ValueError("mc_samples must be nonnegative.")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative.")

    y_next = np.asarray(y_next, dtype=float)
    rng = np.random.default_rng(random_state)

    raw_scores = np.empty(n_models, dtype=float)
    for m in range(n_models):
        raw_scores[m] = _raw_internal_validity_score_deterministic(
            y_next=y_next,
            mu0=np.asarray(mus[m], dtype=float),
            kappa=float(kappas[m]),
            Lambda=np.asarray(Lambdas[m], dtype=float),
            nu=float(nus[m]),
            sigma_floor=float(sigma_floor),
            mc_samples=mc_samples,
            top_k=top_k,
            rng=rng,
        )

    return raw_scores


# ------------------- New weighting functions -------------------


def deterministic_mask_weighting(
    necessities: np.ndarray,
    possibilities: np.ndarray,
) -> np.ndarray:
    """Apply the deterministic 1{V_m > 0} mask to existing possibilities.

    This function does not compute necessity scores itself. It expects model-wise
    necessity scores that have already been computed, then masks the supplied
    possibilities and normalizes the result to have supremum one.
    """
    necessities = np.asarray(necessities, dtype=float)
    possibilities = np.asarray(possibilities, dtype=float)

    if len(necessities) != len(possibilities):
        raise ValueError(
            "necessities and possibilities must have the same length."
        )

    validity_mask = (necessities > 0.0).astype(float)
    adjusted = validity_mask * possibilities

    if sum(adjusted) <= 0:
        return possibilities

    return adjusted


def power_weighting(
    necessities: np.ndarray,
    possibilities: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Apply soft power weighting using precomputed necessity scores.

    The adjusted possibilities are proportional to
        possibilities_m * (necessities_m + epsilon) ** gamma
    and are then normalized to have supremum one.
    """
    necessities = np.asarray(necessities, dtype=float)
    possibilities = np.asarray(possibilities, dtype=float)

    if len(necessities) != len(possibilities):
        raise ValueError(
            "necessities and possibilities must have the same length."
        )
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")

    adjusted = possibilities * np.power(
        np.maximum(necessities, 0.0) + epsilon, gamma
    )
    if sum(adjusted) <= 0.0:
        return possibilities

    return adjusted


def exponential_penalty_weighting(
    necessities: np.ndarray,
    possibilities: np.ndarray,
    eta: float = 1.0,
) -> np.ndarray:
    """Apply exponential penalization using precomputed necessity scores.

    The adjusted possibilities are proportional to
        possibilities_m * exp(-eta * (1 - necessities_m))
    and are then normalized to have supremum one.
    """
    necessities = np.asarray(necessities, dtype=float)
    possibilities = np.asarray(possibilities, dtype=float)

    if len(necessities) != len(possibilities):
        raise ValueError(
            "necessities and possibilities must have the same length."
        )
    if eta < 0.0:
        raise ValueError("eta must be nonnegative.")

    adjusted = possibilities * np.exp(
        -eta * (1.0 - np.maximum(necessities, 0.0))
    )

    if sum(adjusted) <= 0:
        return possibilities

    return adjusted
