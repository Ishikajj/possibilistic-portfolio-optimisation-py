"""Possibilistic Bayesian model averaging over Normal-Inverse-Wishart (NIW) models.

Replaces the probabilistic normalisation constant (sum) with a supremum, producing
possibility distributions instead of probability distributions. At each time step a
new NIW model is born, all models are updated via the conjugate NIW update, and
possibilities are reweighted using the possibilistic Bayes rule. Three necessity-based
weighting schemes (masked, power, exponential) are applied on top of the raw possibilities
to produce three sets of predictive moments, each passed to Markowitz for portfolio weights.

Algorithm (per time step t):
1. Birth     — spawn a new NIW model with prior set to cross-asset historical moments
2. Observe   — receive realised return vector R_t
3. Score     — compute necessity scores for all models given R_t
4. Reweight  — update possibilities via possibilistic Bayes rule (supremum normalisation)
5. Update    — apply conjugate NIW update to every model
6. Aggregate — form predictive (mu, Sigma) under each of three necessity weighting schemes
7. Prune     — keep top max_models by possibility
8. Merge     — merge similar models by Hellinger distance to reduce redundancy

Entry point: run_core(returns_df, ...) — returns (diag_df, masked_pred, masked_weights,
power_pred, power_weights, exp_pred, exp_weights).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.linalg import slogdet

from data_input import (
    load_excess_returns_from_kenneth_french_path,
    prepare_returns,
)
from faster_necessity import (
    deterministic_mask_weighting,
    exponential_penalty_weighting,
    necessity_scores_fast,
    power_weighting,
)
from markowitz import markowitz_unconstrained


def _new_model_prior(
    sum_R: np.ndarray, sum_R2: np.ndarray, t: int
) -> tuple[float, float]:
    """Compute the scalar prior hyperparameters for the newborn model at time t.

    The newborn model uses a common prior across assets:
    - mean = average across assets of historical sample means
    - covariance scale = average across assets of historical sample variances

    Returns
    -------
    (mu_bar, lambda_bar)
        Scalar common mean and scalar common variance proxy.
    """
    if t <= 0:
        return 0.0, 1e-4

    mu_vec = sum_R / float(t)
    mu_bar = float(mu_vec.mean())

    if t <= 1:
        return mu_bar, 1e-4

    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 1)
    var_vec = np.maximum(var_vec, 1e-8)
    return mu_bar, float(var_vec.mean())


def _append_new_model(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    mu_bar: float,
    lam_bar: float,
    n: int,
) -> np.ndarray:
    """Append a newborn possibilistic model.

    Before observing R_t, the project specification sets the newborn model's
    possibility to 1, while older models retain their previous possibilities.
    """
    mus.append(np.full(n, mu_bar, dtype=float))
    kappas.append(1.0)
    Lambdas.append(lam_bar * np.eye(n, dtype=float))
    nus.append(1.0)

    if possibilities.size == 0:
        return np.array([1.0], dtype=float)
    return np.concatenate([possibilities, np.array([1.0], dtype=float)])


def _log_marginal_likelihood(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Closed-form possibilistic score log L(r_t | m, F_{t-1}).

    L = (2*pi*e)^{-n/2} * |Lambda|^{nu/2} * nu^{-n*nu/2}
        * nu1^{n*nu1/2} * |Lambda1|^{-nu1/2}

    where Lambda1 = Lambda + (kappa/(kappa+1)) * outer(d, d), nu1 = nu + 1.
    """
    n = len(R_t)
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    d = R_t - mu
    Lambda1 = Lambda + (kappa / k1) * np.outer(d, d)

    s0, ld0 = slogdet(Lambda)
    s1, ld1 = slogdet(Lambda1)
    if s0 <= 0 or s1 <= 0:
        return -np.inf

    return (
        -(n / 2.0) * np.log(2.0 * np.pi * np.e)
        + (nu / 2.0) * ld0
        - (n * nu / 2.0) * np.log(nu)
        + (n * nu1 / 2.0) * np.log(nu1)
        - (nu1 / 2.0) * ld1
    )


def _update_possibilities(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
) -> np.ndarray:
    """Update model credibilities using possibilistic Bayes rule.

    P_t(m | F_t) = L(R_t | m, F_{t-1}) P_{t-1}(m | F_{t-1})
                   / sup_q L(R_t | q, F_{t-1}) P_{t-1}(q | F_{t-1}).

    Since the model space is finite, the supremum is the maximum.

    returns an numpy array with the updated model possibilities given the new returns.
    """
    log_joint = np.array(
        [
            _log_marginal_likelihood(
                R_t,
                mus[m],
                kappas[m],
                Lambdas[m],
                nus[m],
            )
            + np.log(max(possibilities[m], 1e-300))
            for m in range(len(mus))
        ],
        dtype=float,
    )  # creates an array of the numerator of the likelihood function for each model

    finite = np.isfinite(log_joint)
    if not finite.any():
        return np.ones(len(mus), dtype=float)

    log_sup = np.max(log_joint[finite])
    updated = np.exp(log_joint - log_sup)
    updated = np.where(
        np.isfinite(updated), updated, 0.0
    )  # replace all non finite updated valeus with 0.

    # the following is just defensive coding to make sure the supremum condition we already ensured is met again.
    max_updated = updated.max(initial=0.0)
    if max_updated <= 0.0:
        return np.ones(len(mus), dtype=float)
    return updated / max_updated


# correct
def _update_niw(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """One-step NIW posterior update after observing R_t."""
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    mu1 = (kappa * mu + R_t) / k1
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(d, d)
    return mu1, k1, L1, nu1


# correct
def _update_all_models(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
) -> None:
    """Update every model's NIW parameters in place."""
    for m in range(len(mus)):
        mus[m], kappas[m], Lambdas[m], nus[m] = _update_niw(
            R_t, mus[m], kappas[m], Lambdas[m], nus[m]
        )


# correct
def _sigma_mode(Lambda: np.ndarray, nu: float, n: int) -> np.ndarray:
    """Mode of the possibilistic inverse-Wishart kernel used in the thesis.

    Appendix A uses E^*(Sigma) = Delta / nu, i.e. the mode under the possibilistic
    inverse-Wishart convention adopted in the project.
    """
    return Lambda / nu


###################
####MODEL MANAGEMENT FUNCTIONS
###################
def _prune_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    max_models: int = 100,
    keep_newest: bool = True,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Prune models to control pool size.

    Strategy: always keep exactly the top max_models by possibility.
    One model is added each step, so at most one is dropped each step —
    no cliff-edge mass die-offs.  The threshold is kept as a secondary
    floor but only acts when max_models is None.

    After pruning, possibilities are renormalized to have supremum one.
    """
    n_models = len(mus)
    if not (
        len(kappas)
        == len(Lambdas)
        == len(nus)
        == len(possibilities)
        == n_models
    ):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return mus, kappas, Lambdas, nus, possibilities

    # Always keep exactly top max_models — smooth one-per-step pruning
    if n_models <= max_models:
        keep_idx = np.arange(n_models, dtype=int)
    else:
        order = np.argsort(possibilities)[::-1]
        keep_idx = np.sort(order[:max_models])
        if keep_newest and (n_models - 1) not in keep_idx:
            # Replace the lowest-possibility keeper with the newborn
            keep_idx[-1] = n_models - 1
            keep_idx = np.sort(keep_idx)

    mus = [mus[i] for i in keep_idx]
    kappas = [kappas[i] for i in keep_idx]
    Lambdas = [Lambdas[i] for i in keep_idx]
    nus = [nus[i] for i in keep_idx]
    possibilities = possibilities[keep_idx].astype(float, copy=False)

    max_possibility = possibilities.max(initial=0.0)
    if max_possibility <= 0.0:
        possibilities = np.ones(len(possibilities), dtype=float)
    else:
        possibilities = possibilities / max_possibility

    return mus, kappas, Lambdas, nus, possibilities


def _merge_two_models(
    mu1: np.ndarray,
    kappa1: float,
    Lambda1: np.ndarray,
    nu1: float,
    poss1: float,
    mu2: np.ndarray,
    kappa2: float,
    Lambda2: np.ndarray,
    nu2: float,
    poss2: float,
    n: int,
) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    """Merge two NIW models by combining their sufficient statistics.

    Equivalent to treating both models as observations from the same process:
        kappa_new  = kappa1 + kappa2
        mu_new     = precision-weighted mean
        Lambda_new = combined scatter + between-mean correction
        nu_new     = nu1 + nu2 - n  (floored at n+2 to keep IW valid)
        poss_new   = max(poss1, poss2)
    """
    kappa_new = kappa1 + kappa2
    mu_new = (kappa1 * mu1 + kappa2 * mu2) / kappa_new
    d = mu1 - mu2
    Lambda_new = (
        Lambda1 + Lambda2 + (kappa1 * kappa2 / kappa_new) * np.outer(d, d)
    )
    nu_new = float(max(nu1 + nu2 - n, float(n + 2)))
    poss_new = float(max(poss1, poss2))
    return mu_new, kappa_new, Lambda_new, nu_new, poss_new


# --- Hellinger distance between Gaussians ---
def _gaussian_hellinger_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """Hellinger distance between two multivariate Gaussian distributions.

    For N(mu1, Sigma1) and N(mu2, Sigma2), the squared Hellinger distance is

        H^2 = 1 - \
            |Sigma1|^{1/4} |Sigma2|^{1/4} / |(Sigma1 + Sigma2)/2|^{1/2}
            * exp(-1/8 * (mu1-mu2)^T ((Sigma1+Sigma2)/2)^{-1} (mu1-mu2)).

    Returns the Hellinger distance H in [0, 1].
    """
    Sigma1 = 0.5 * (Sigma1 + Sigma1.T)
    Sigma2 = 0.5 * (Sigma2 + Sigma2.T)

    eps = 1e-10
    n = Sigma1.shape[0]
    Sigma1 = Sigma1 + eps * np.eye(n, dtype=float)
    Sigma2 = Sigma2 + eps * np.eye(n, dtype=float)

    Sigma_avg = 0.5 * (Sigma1 + Sigma2)
    d = mu1 - mu2

    s1, ld1 = np.linalg.slogdet(Sigma1)
    s2, ld2 = np.linalg.slogdet(Sigma2)
    savg, ldavg = np.linalg.slogdet(Sigma_avg)
    if s1 <= 0 or s2 <= 0 or savg <= 0:
        return 1.0

    try:
        quad = float(d @ np.linalg.solve(Sigma_avg, d))
    except np.linalg.LinAlgError:
        return 1.0

    log_coeff = 0.25 * ld1 + 0.25 * ld2 - 0.5 * ldavg
    log_rho = log_coeff - 0.125 * quad
    rho = float(np.exp(min(log_rho, 0.0)))
    rho = min(max(rho, 0.0), 1.0)

    hellinger_sq = max(0.0, 1.0 - rho)
    return float(np.sqrt(hellinger_sq))


def _gaussian_mahalanobis_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """Mahalanobis distance between two Gaussian means using pooled covariance.

    d = sqrt((mu1 - mu2)^T * ((Sigma1 + Sigma2) / 2)^{-1} * (mu1 - mu2))

    Merging of 2 models was earlier based on mahalnobis distance -
    but this was switched in favour of hellinger which compares distributions directly
    """
    eps = 1e-10
    n = Sigma1.shape[0]
    Sigma_pool = 0.5 * (Sigma1 + Sigma1.T + Sigma2 + Sigma2.T) + eps * np.eye(
        n, dtype=float
    )
    d = mu1 - mu2
    try:
        quad = float(d @ np.linalg.solve(Sigma_pool, d))
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(max(quad, 0.0)))


def _merge_models_core(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    distance_fn,
    merge_threshold: float,
    nu_bandwidth: int,
    k_neighbours: int,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
    float,
]:
    """Merge redundant model pairs using a pluggable distance function.

    Two models are candidates for merging only if:
      1. |nu_i - nu_j| <= nu_bandwidth  — similar window length
      2. distance_fn(mu_i, Sigma_i, mu_j, Sigma_j) < merge_threshold

    Each model is summarised by N(mu_i, Sigma_i) where Sigma_i = Lambda_i / nu_i.
    Search: sort by nu, scan k_neighbours forward neighbours — O(N * k) per pass.
    Merges are applied greedily until no qualifying pairs remain.
    Returns the updated pool and the possibility-weighted average distance.
    """
    if len(mus) < 2:
        return mus, kappas, Lambdas, nus, possibilities, np.nan

    n = len(mus[0])
    poss_arr = np.array(possibilities, dtype=float, copy=True)
    dist_sum = 0.0
    w_sum = 0.0

    while True:
        n_models = len(mus)
        if n_models < 2:
            break

        nu_arr = np.array(nus, dtype=float)
        order = np.argsort(nu_arr)
        poss_norm = poss_arr / (poss_arr.sum() + 1e-300)

        best_dist = np.inf
        best_i = best_j = -1
        dist_sum = 0.0
        w_sum = 0.0

        for rank in range(n_models):
            i = int(order[rank])
            Sigma_i = Lambdas[i] / nu_arr[i]

            for fwd in range(1, k_neighbours + 1):
                if rank + fwd >= n_models:
                    break
                j = int(order[rank + fwd])

                if abs(nu_arr[i] - nu_arr[j]) > nu_bandwidth:
                    break

                Sigma_j = Lambdas[j] / nu_arr[j]
                dist = distance_fn(mus[i], Sigma_i, mus[j], Sigma_j)

                w = poss_norm[i] * poss_norm[j]
                dist_sum += w * dist
                w_sum += w

                if dist < merge_threshold and dist < best_dist:
                    best_dist = dist
                    best_i, best_j = i, j

        if best_i < 0:
            break

        mu_m, kappa_m, Lambda_m, nu_m, poss_m = _merge_two_models(
            mus[best_i],
            kappas[best_i],
            Lambdas[best_i],
            nus[best_i],
            float(poss_arr[best_i]),
            mus[best_j],
            kappas[best_j],
            Lambdas[best_j],
            nus[best_j],
            float(poss_arr[best_j]),
            n,
        )

        keep = [k for k in range(n_models) if k != best_i and k != best_j]
        mus = [mus[k] for k in keep] + [mu_m]
        kappas = [kappas[k] for k in keep] + [kappa_m]
        Lambdas = [Lambdas[k] for k in keep] + [Lambda_m]
        nus = [nus[k] for k in keep] + [nu_m]
        poss_arr = np.concatenate([poss_arr[keep], [poss_m]])

    max_poss = float(poss_arr.max(initial=0.0))
    poss_arr = (
        poss_arr / max_poss
        if max_poss > 0.0
        else np.ones(len(poss_arr), dtype=float)
    )

    dist_avg = dist_sum / w_sum if w_sum > 0 else np.nan
    return mus, kappas, Lambdas, nus, poss_arr, dist_avg


def _merge_models_hellinger(
    mus,
    kappas,
    Lambdas,
    nus,
    possibilities,
    merge_threshold: float = 0.1,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
):
    """Merge redundant model pairs using Hellinger distance. See _merge_models_core."""
    return _merge_models_core(
        mus,
        kappas,
        Lambdas,
        nus,
        possibilities,
        distance_fn=_gaussian_hellinger_distance,
        merge_threshold=merge_threshold,
        nu_bandwidth=nu_bandwidth,
        k_neighbours=k_neighbours,
    )


def _merge_models_mahalanobis(
    mus,
    kappas,
    Lambdas,
    nus,
    possibilities,
    merge_threshold: float = 1.0,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
):
    """Merge redundant model pairs using Mahalanobis distance. See _merge_models_core."""
    return _merge_models_core(
        mus,
        kappas,
        Lambdas,
        nus,
        possibilities,
        distance_fn=_gaussian_mahalanobis_distance,
        merge_threshold=merge_threshold,
        nu_bandwidth=nu_bandwidth,
        k_neighbours=k_neighbours,
    )


#########FINAL PREDICTIVE FORMATION


def _aggregate_possibilistic_niw(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate model-wise NIW possibility functions into a single predictive.

    The possibilistic combination of M NIW kernels raised to weights w_m is itself
    NIW-shaped (up to a proportionality constant). Combining natural parameters gives:

        kappa* = sum_m w_m * kappa_m
        mu*    = (sum_m w_m * kappa_m * mu_m) / kappa*
        Lambda* = sum_m w_m * (Lambda_m + kappa_m * mu_m mu_m') - kappa* * mu* mu*'
        nu*    = sum_m w_m * nu_m

    Predictive covariance is the mode of the aggregated possibilistic IW kernel:
        Sigma* = Lambda* / nu*

    Weights are derived from possibilities and/or necessity scores by the caller.
    """
    n_models = len(mus)
    if n_models == 0:
        raise ValueError("At least one model is required for aggregation.")

    weights = np.array(possibilities, dtype=float)
    weights_sum = weights.sum()

    if weights_sum <= 0.0:
        weights = np.ones(n_models, dtype=float) / n_models
    else:
        weights = weights / weights_sum

    weighted_mean_sum = np.zeros(n, dtype=float)
    weighted_kappa_sum = float(np.sum(weights * np.array(kappas, dtype=float)))
    weighted_nu_sum = float(np.sum(weights * np.array(nus, dtype=float)))

    if weighted_kappa_sum <= 0.0:
        raise ValueError("Aggregated kappa must be strictly positive.")

    for m in range(n_models):
        weighted_mean_sum += weights[m] * kappas[m] * mus[m]
    mu_hat = weighted_mean_sum / weighted_kappa_sum

    Lambda_hat = np.zeros((n, n), dtype=float)
    for m in range(n_models):
        centered_outer = np.outer(mus[m], mus[m])
        Lambda_hat += weights[m] * (Lambdas[m] + kappas[m] * centered_outer)
    Lambda_hat -= weighted_kappa_sum * np.outer(mu_hat, mu_hat)

    Lambda_hat = 0.5 * (
        Lambda_hat + Lambda_hat.T
    )  # makes sure the matrix is symmetric.
    Lambda_hat += 1e-10 * np.eye(n, dtype=float)

    Sigma_hat = _sigma_mode(Lambda_hat, weighted_nu_sum, n)
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)
    return mu_hat, Sigma_hat


def run_core(
    returns_df: pd.DataFrame,
    burn_in: int = 1000,
    periods_until_investment: int = 0,
    merge_threshold: float = 0.15,
    max_models: int = 100,
    keep_newest: bool = True,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
    device: str = "cpu",
    n_jobs: int = 6,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """Run the possibilistic NIW model averaging algorithm.

    Args:
        returns_df: (T, n) DataFrame of excess returns in decimals, numeric columns only.
        burn_in: periods used to initialise sum_R / sum_R2 before model birth begins.
        periods_until_investment: additional lag before Markowitz weights are non-NaN.
        merge_threshold: Hellinger distance below which two models are merged.
        max_models: maximum pool size after pruning each step.
        keep_newest: if True, always retain the newborn model after pruning.
        nu_bandwidth: max |nu_i - nu_j| for two models to be merge candidates.
        k_neighbours: number of forward neighbours scanned per model during merging.
        gamma: power exponent for power weighting scheme.
        eta: decay rate for exponential penalty weighting scheme.
        device: "cpu" or "cuda" for necessity score computation.
        n_jobs: parallel workers for necessity score computation.

    Returns 7-tuple:
        diag_df            — per-step diagnostics (model counts, necessity stats, Hellinger avg)
        masked_predictive  — {"mu_hat": (T,n), "sigma_hat": (T,n,n)} under masked weighting
        weights_masked     — (T, n) Markowitz weights from masked predictives
        power_predictive   — {"mu_hat": (T,n), "sigma_hat": (T,n,n)} under power weighting
        weights_power      — (T, n) Markowitz weights from power predictives
        exp_predictive     — {"mu_hat": (T,n), "sigma_hat": (T,n,n)} under exponential weighting
        weights_exp        — (T, n) Markowitz weights from exponential predictives
    """
    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mu_hat_arr_masked = np.full((T, n), np.nan)
    sigma_hat_arr_masked = np.full((T, n, n), np.nan)
    mu_hat_arr_power = np.full((T, n), np.nan)
    sigma_hat_arr_power = np.full((T, n, n), np.nan)
    mu_hat_arr_exp = np.full((T, n), np.nan)
    sigma_hat_arr_exp = np.full((T, n, n), np.nan)

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)
    records = []

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    for t in range(burn_obs, T):
        if t % 100 == 0:
            print(f"Processing time step {t} / {T}...")

        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )

        n_models_raw = len(mus)

        R_t = R[t]
        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            n_jobs=n_jobs,
            random_state=t,  # vary per step for diversity
            device=device,
        )

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )

        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        masked_weights = deterministic_mask_weighting(
            necessities, possibilities
        )
        power_weights = power_weighting(necessities, possibilities, gamma=gamma)
        exp_weights = exponential_penalty_weighting(
            necessities, possibilities, eta=eta
        )

        try:
            mu_masked, Sigma_masked = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, masked_weights, n
            )
            mu_power, Sigma_power = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, power_weights, n
            )
            mu_exp, Sigma_exp = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, exp_weights, n
            )

            mu_hat_arr_masked[t] = mu_masked
            sigma_hat_arr_masked[t] = Sigma_masked
            mu_hat_arr_power[t] = mu_power
            sigma_hat_arr_power[t] = Sigma_power
            mu_hat_arr_exp[t] = mu_exp
            sigma_hat_arr_exp[t] = Sigma_exp
        except Exception:
            pass

        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            max_models=max_models,
            keep_newest=keep_newest,
        )

        n_models_post_prune = len(mus)
        hellinger_avg = np.nan

        mus, kappas, Lambdas, nus, possibilities, hellinger_avg = (
            _merge_models_hellinger(
                mus,
                kappas,
                Lambdas,
                nus,
                possibilities,
                merge_threshold=merge_threshold,
                nu_bandwidth=nu_bandwidth,
                k_neighbours=k_neighbours,
            )
        )

        n_models_post_merge = len(mus)

        records.append(
            {
                "t": t,
                "n_models_raw": n_models_raw,
                "n_models_post_prune": n_models_post_prune,
                "n_models_post_merge": n_models_post_merge,
                "nec_mean": float(necessities.mean()),
                "nec_max": float(necessities.max()),
                "frac_nec_zero": float(np.mean(necessities == 0.0)),
                "nu_mean": float(np.mean(nus)),
                "nu_min": float(np.min(nus)),
                "hellinger_avg": hellinger_avg,
            }
        )
        sum_R += R_t
        sum_R2 += R_t**2

    diag_df = pd.DataFrame(records).set_index("t")

    masked_predictive = {
        "mu_hat": mu_hat_arr_masked,
        "sigma_hat": sigma_hat_arr_masked,
    }
    power_predictive = {
        "mu_hat": mu_hat_arr_power,
        "sigma_hat": sigma_hat_arr_power,
    }
    exp_predictive = {"mu_hat": mu_hat_arr_exp, "sigma_hat": sigma_hat_arr_exp}

    weights_masked = markowitz_unconstrained(
        masked_predictive,
        returns_df,
        burn_in=burn_in,
        periods_until_investment=0,
    )
    weights_power = markowitz_unconstrained(
        power_predictive,
        returns_df,
        burn_in=burn_in,
        periods_until_investment=0,
    )
    weights_exp = markowitz_unconstrained(
        exp_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
    )

    return (
        diag_df,
        masked_predictive,
        weights_masked,
        power_predictive,
        weights_power,
        exp_predictive,
        weights_exp,
    )


def test_run_core() -> tuple[pd.DataFrame, tuple]:
    """Run a minimal smoke test of run_core on a short return sample."""
    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(start_date="2000-01-01")
    )
    df_small = df.iloc[:150].copy()

    results = run_core(
        df_small,
        burn_in=20,
        merge_threshold=0.15,
        max_models=10,
        keep_newest=True,
        nu_bandwidth=20,
        k_neighbours=3,
        gamma=1.0,
        eta=1.0,
    )

    diag_df = results[0]
    if diag_df.empty:
        raise RuntimeError(
            "smoke_test_run_core produced an empty diagnostic DataFrame."
        )

    print("test_run_core passed")
    print(diag_df.tail())
    return diag_df, results


if __name__ == "__main__":
    test_run_core()
