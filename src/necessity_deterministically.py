from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError, slogdet
from scipy.optimize import minimize
from scipy.stats import invwishart


# makes sense
def _log_normal_likelihood(y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
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
def _sample_probabilistic_niw(
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (mu, Sigma) from the probabilistic NIW analogue.

    These samples are used only to generate strong initial points for local
    optimization. The final necessity score is still computed from deterministic
    local optimization of the corrected supremum objective.
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

    # invwishart has the same mode in probability and possibility. when we draw from sample, things roughly match. its just that one has a normalisation issue.
    Sigmas = invwishart.rvs(df=nu, scale=Lambda, size=n_samples, random_state=rng)
    if n_samples == 1:
        Sigmas = Sigmas[np.newaxis, :, :]

    mus = np.empty((n_samples, n), dtype=float)
    for s in range(n_samples):
        Sigma_s = 0.5 * (Sigmas[s] + Sigmas[s].T)
        mus[s] = rng.multivariate_normal(mean=mu0, cov=Sigma_s / kappa)

    return mus, Sigmas


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


# correct but this is not "normal"
def _log_hbar(
    y: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    sigma_floor: float,
) -> float:
    """Log normalized Gaussian likelihood.

    The unconstrained supremum over (mu, Sigma) is not finite because Sigma can
    collapse to zero at mu = y. We therefore normalize relative to the constrained
    supremum achieved at mu = y and Sigma = sigma_floor * I.
    """
    n = len(y)
    log_h = _log_normal_likelihood(y, mu, Sigma)
    if not np.isfinite(log_h):
        return -np.inf

    log_sup = -0.5 * (n * np.log(2.0 * np.pi) + n * np.log(sigma_floor))
    log_hbar = log_h - log_sup
    return min(log_hbar, 0.0)


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
    sigma_floor: float,
) -> float:
    """Evaluate (1 - h_bar) f_bar at a deterministic candidate point."""
    log_hbar = _log_hbar(y_next, mu, Sigma, sigma_floor)
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


def _hybrid_initial_thetas(
    y_next: np.ndarray,
    mu0: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
    sigma_floor: float,
    mc_samples: int,
    top_k: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Construct hybrid starting points: analytic seeds plus top Monte Carlo seeds.

    We first generate deterministic analytic starts from the NIW structure. We
    then sample from the probabilistic NIW analogue, score those samples with the
    corrected deterministic bracket objective, and keep the top-k samples as
    additional starts for local optimization.
    """
    starts = _deterministic_initial_thetas(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
    )  # we get some fixed initial samples depending on the parameters of the model and the observation we will notice.

    if mc_samples <= 0 or top_k <= 0:
        return starts

    sample_mus, sample_sigmas = (
        _sample_probabilistic_niw(  # some random samples from the NIW
            mu0=mu0,
            kappa=kappa,
            Lambda=Lambda,
            nu=nu,
            n_samples=mc_samples,
            rng=rng,
        )
    )

    values = np.empty(mc_samples, dtype=float)
    for s in range(mc_samples):
        values[s] = _bracket_value(
            mu=sample_mus[s],
            Sigma=sample_sigmas[s],
            y_next=y_next,
            mu0=mu0,
            kappa=kappa,
            Lambda=Lambda,
            nu=nu,
            sigma_floor=sigma_floor,
        )

    order = np.argsort(values)[::-1]
    n_keep = min(top_k, mc_samples)
    for s in order[:n_keep]:
        Sigma_start = 0.5 * (sample_sigmas[s] + sample_sigmas[s].T)
        Sigma_start += sigma_floor * np.eye(len(mu0), dtype=float)
        chol = np.linalg.cholesky(Sigma_start)
        starts.append(_pack_theta(sample_mus[s], chol))

    return starts


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
    """
    finds the necessity of a given prior model, and the observation
    Hybrid local-optimization approximation of the corrected necessity score.

    The target remains the corrected deterministic quantity
        V_m = 1 - sup (1 - h_bar_m) f_bar_m.

    The hybrid part is only in the choice of optimizer starting points:
    analytic NIW-based starts plus top Monte Carlo NIW samples.
    """
    y_next = np.asarray(y_next, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = len(mu0)
    if rng is None:
        rng = np.random.default_rng()

    def objective(theta: np.ndarray) -> float:
        mu, Sigma = _unpack_theta(theta, n=n, ridge=sigma_floor)
        value = _bracket_value(mu, Sigma, y_next, mu0, kappa, Lambda, nu, sigma_floor)
        return -value

    # gets a few start values (of mu and sigma, given our lambda, mu0 , kappa) depending on the number of samples
    best_value = 0.0
    starts = _hybrid_initial_thetas(
        y_next=y_next,
        mu0=mu0,
        kappa=kappa,
        Lambda=Lambda,
        nu=nu,
        sigma_floor=sigma_floor,
        mc_samples=mc_samples,
        top_k=top_k,
        rng=rng,
    )

    for theta0 in starts:
        res = minimize(objective, theta0, method="L-BFGS-B")

        candidate_thetas = [theta0]

        # res function finds some local optimizers of (1 - h) f
        if res.success and np.all(np.isfinite(res.x)):
            candidate_thetas.append(res.x)

        # for each of these candidates, calculate the bracket value, and update it. to note that the maximiser is not used, only the maximised value
        for theta in candidate_thetas:
            mu, Sigma = _unpack_theta(theta, n=n, ridge=sigma_floor)
            value = _bracket_value(
                mu, Sigma, y_next, mu0, kappa, Lambda, nu, sigma_floor
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
        raise ValueError("necessities and possibilities must have the same length.")

    validity_mask = (necessities > 0.0).astype(float)
    adjusted = validity_mask * possibilities
    max_adjusted = adjusted.max(initial=0.0)
    if max_adjusted <= 0.0:
        return np.ones_like(adjusted)
    return adjusted / max_adjusted


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
        raise ValueError("necessities and possibilities must have the same length.")
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")

    adjusted = possibilities * np.power(np.maximum(necessities, 0.0) + epsilon, gamma)
    max_adjusted = adjusted.max(initial=0.0)
    if max_adjusted <= 0.0:
        return np.ones_like(adjusted)
    return adjusted / max_adjusted


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
        raise ValueError("necessities and possibilities must have the same length.")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative.")

    adjusted = possibilities * np.exp(-eta * (1.0 - np.maximum(necessities, 0.0)))
    max_adjusted = adjusted.max(initial=0.0)
    if max_adjusted <= 0.0:
        return np.ones_like(adjusted)
    return adjusted / max_adjusted


def necessity_weighted_possibilities_deterministic(
    y_next: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    *,
    sigma_floor: float = 1e-6,
    mc_samples: int = 0,
    top_k: int = 0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute necessity scores and apply deterministic mask weighting.

    This wrapper first computes model-wise deterministic necessity scores and
    then applies the 1{V_m > 0} weighting rule through
    `deterministic_mask_weighting(...)`.
    """
    possibilities = np.asarray(possibilities, dtype=float)
    necessities = necessity_scores_deterministic(
        y_next=y_next,
        mus=mus,
        kappas=kappas,
        Lambdas=Lambdas,
        nus=nus,
        sigma_floor=sigma_floor,
        mc_samples=mc_samples,
        top_k=top_k,
        random_state=random_state,
    )

    adjusted = deterministic_mask_weighting(necessities, possibilities)
    return necessities, adjusted
