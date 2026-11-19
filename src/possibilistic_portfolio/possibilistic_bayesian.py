"""Possibilistic Bayesian model averaging over Normal-Inverse-Wishart (NIW) models.

Replaces the probabilistic normalisation constant (sum) with a supremum, producing
possibility distributions instead of probability distributions. At each time step a
new NIW model is born, all models are updated via the conjugate NIW update, and
possibilities are reweighted using the possibilistic Bayes rule. The models are
then pooled geometrically into one predictive (mu, Sigma), the forecast; the
pipeline turns it into Markowitz weights (weight_rules).

Weighting schemes (the pooling weights of the models):
  "plain"                   — the possibilities themselves (default, always there)
  "masked", "power", "exp"  — possibilities adjusted by necessity scores
                              (faster_necessity.py). Optional: only with
                              ``necessity=True``, which is slow (the scores are
                              a numerical supremum per model and step).

Algorithm (per time step t):
1. Birth     — spawn a new NIW model with prior set to cross-asset historical moments
2. Observe   — receive realised return vector R_t
3. Score     — (necessity only) necessity scores for all models given R_t
4. Reweight  — update possibilities via possibilistic Bayes rule (supremum normalisation)
5. Update    — apply conjugate NIW update to every model
6. Aggregate — form predictive (mu, Sigma) under each weighting scheme
7. Prune     — keep top max_models by possibility
8. Merge     — (optional, off by default) merge similar models by Hellinger
               distance to reduce redundancy

Entry point: run_core(returns_df, ...) — returns (diag_df, forecasts),
forecasts = {scheme: forecast dict}, one per weighting scheme.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.linalg import slogdet

from possibilistic_portfolio import niw
from possibilistic_portfolio.config import (
    BURN_IN,
    DATASETS_DIR,
    K_NEIGHBOURS,
    KEEP_NEWEST,
    MERGE_THRESHOLD,
    NECESSITY_DEVICE,
    NECESSITY_ETA,
    NECESSITY_GAMMA,
    NECESSITY_N_JOBS,
    NU_BANDWIDTH,
    POSSIBILISTIC_MAX_MODELS,
    RF_PATH,
    USE_NECESSITY,
)
from possibilistic_portfolio.data_input import (
    load_excess_returns,
    prepare_returns,
)
from possibilistic_portfolio.faster_necessity import (
    deterministic_mask_weighting,
    exponential_penalty_weighting,
    necessity_scores_fast,
    power_weighting,
)
from possibilistic_portfolio.predictive import make_predictive, warn_nan_rows

# Weighting schemes that need necessity scores (run_core(necessity=True)).
NECESSITY_SCHEMES = ("masked", "power", "exp")


log = logging.getLogger(__name__)


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


def _sup_normalise(possibilities: np.ndarray) -> np.ndarray:
    """Rescale so the largest possibility is 1 (all ones if none is
    positive)."""
    max_possibility = possibilities.max(initial=0.0)
    if max_possibility <= 0.0:
        return np.ones(len(possibilities), dtype=float)
    return possibilities / max_possibility


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

    Returns the updated model possibilities (maximum 1). If no model has a
    finite score, all possibilities are reset to 1.
    """
    # log numerator of the update, one entry per model
    log_joint = np.array(
        [
            _log_marginal_likelihood(R_t, mus[m], kappas[m], Lambdas[m], nus[m])
            + np.log(max(possibilities[m], 1e-300))
            for m in range(len(mus))
        ],
        dtype=float,
    )
    finite = np.isfinite(log_joint)
    if not finite.any():
        return np.ones(len(mus), dtype=float)

    # subtract the largest log first so exp can't overflow
    updated = np.exp(log_joint - np.max(log_joint[finite]))
    updated = np.where(np.isfinite(updated), updated, 0.0)
    return _sup_normalise(updated)  # max is 1 already; defensive


def _sigma_mode(Lambda: np.ndarray, nu: float) -> np.ndarray:
    """Mode of the possibilistic inverse-Wishart kernel used in the thesis.

    Appendix A uses E^*(Sigma) = Delta / nu, i.e. the mode under the possibilistic
    inverse-Wishart convention adopted in the project.
    """
    return Lambda / nu


# ── Model management: pruning and merging ────────────────────────────────────


def _prune_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    max_models: int = POSSIBILISTIC_MAX_MODELS,
    keep_newest: bool = KEEP_NEWEST,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Keep the top `max_models` models by possibility.

    No threshold (unlike the Bayesian pool): with supremum normalisation one
    model can dominate while all others are tiny, so a threshold would wipe
    the pool out. One model is added each step, so at most one is dropped.
    With `keep_newest` the newborn always survives, taking the place of the
    least possible kept model if it isn't among the top itself.

    After pruning, possibilities are renormalised to have supremum one.
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

    keep_idx = niw.top_models(possibilities, max_models, keep_newest)
    mus, kappas, Lambdas, nus = niw.select_models(
        mus, kappas, Lambdas, nus, keep_idx
    )
    possibilities = possibilities[keep_idx].astype(float, copy=False)
    return mus, kappas, Lambdas, nus, _sup_normalise(possibilities)


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
        nu_new     = nu1 + nu2 - n  (floored at n+2; thesis eq. 5.5)
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

    Returns the Hellinger distance H in [0, 1]. Both covariances get a
    1e-10 ridge first. If a covariance is not positive definite, or the
    result isn't finite, it is 1 (maximal distance), so such a pair is never
    merged.
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

    # log of the Bhattacharyya coefficient rho = 1 - H^2
    log_rho = 0.25 * ld1 + 0.25 * ld2 - 0.5 * ldavg - 0.125 * quad
    if not np.isfinite(log_rho):
        return 1.0
    rho = float(np.exp(min(log_rho, 0.0)))  # in (0, 1]
    return float(np.sqrt(1.0 - rho))


def _gaussian_mahalanobis_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """Mahalanobis distance between two Gaussian means using pooled covariance.

    d = sqrt((mu1 - mu2)^T * ((Sigma1 + Sigma2) / 2)^{-1} * (mu1 - mu2))

    Not used by run_core: merging was first based on this distance, then
    switched to Hellinger, which compares the distributions directly. Kept
    as an alternative ``distance_fn`` for ``_merge_models``. Returns inf (never
    merge) if the pooled covariance is singular.
    """
    eps = 1e-10
    n = Sigma1.shape[0]
    # (Sigma1 + Sigma2) / 2, symmetrised, plus a small ridge
    Sigma_pool = 0.25 * (Sigma1 + Sigma1.T + Sigma2 + Sigma2.T) + eps * np.eye(
        n, dtype=float
    )
    d = mu1 - mu2
    try:
        quad = float(d @ np.linalg.solve(Sigma_pool, d))
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(max(quad, 0.0)))


def _closest_pair(
    mus: list[np.ndarray],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    nu_bandwidth: int,
    k_neighbours: int,
    distance_fn: Callable[..., float],
) -> tuple[int, int, float, float]:
    """The closest pair of merge candidates in the pool.

    Candidates: sort the models by nu; each model is compared with its next
    `k_neighbours` models in that order, as long as |nu_i - nu_j| <=
    nu_bandwidth (similar window length). O(N * k). Each model is summarised
    by N(mu_i, Lambda_i / nu_i) for the distance.

    Returns (i, j, distance, average distance); i = j = -1 and distance inf
    if there is no candidate pair. The average is over all candidate pairs,
    each weighted by the product of the two models' possibility shares (a
    diagnostic; NaN without candidates).
    """
    nu_arr = np.array(nus, dtype=float)
    order = np.argsort(nu_arr)
    share = possibilities / (possibilities.sum() + 1e-300)

    best_i = best_j = -1
    best_dist = np.inf
    dist_sum = weight_sum = 0.0
    for rank, i in enumerate(order):
        Sigma_i = Lambdas[i] / nu_arr[i]
        for j in order[rank + 1 : rank + 1 + k_neighbours]:
            if abs(nu_arr[i] - nu_arr[j]) > nu_bandwidth:
                break
            dist = distance_fn(mus[i], Sigma_i, mus[j], Lambdas[j] / nu_arr[j])
            pair_weight = share[i] * share[j]
            dist_sum += pair_weight * dist
            weight_sum += pair_weight
            if dist < best_dist:
                best_i, best_j, best_dist = int(i), int(j), dist

    avg_dist = dist_sum / weight_sum if weight_sum > 0 else np.nan
    return best_i, best_j, best_dist, avg_dist


def _merge_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    merge_threshold: float | None = MERGE_THRESHOLD,
    nu_bandwidth: int = NU_BANDWIDTH,
    k_neighbours: int = K_NEIGHBOURS,
    distance_fn: Callable[..., float] = _gaussian_hellinger_distance,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
    float,
]:
    """Merge redundant model pairs (Hellinger distance by default).

    Repeatedly merges the closest candidate pair (see _closest_pair) while
    its distance is below `merge_threshold`. `distance_fn` can be swapped
    (e.g. `_gaussian_mahalanobis_distance`, with a threshold on that scale).

    Returns the updated pool and, as a diagnostic, the average distance of
    the last search (the final pool, in which nothing merged); NaN with
    fewer than two models.

    `merge_threshold=None` turns merging off (the default): the pool comes
    back unchanged, the distance search doesn't run, and the distance is NaN.
    """
    if merge_threshold is None or len(mus) < 2:
        return mus, kappas, Lambdas, nus, possibilities, np.nan

    n = len(mus[0])
    possibilities = np.array(possibilities, dtype=float, copy=True)
    avg_dist = np.nan
    while len(mus) >= 2:
        i, j, dist, avg_dist = _closest_pair(
            mus,
            Lambdas,
            nus,
            possibilities,
            nu_bandwidth,
            k_neighbours,
            distance_fn,
        )
        if not dist < merge_threshold:
            break

        merged = _merge_two_models(
            mus[i], kappas[i], Lambdas[i], nus[i], float(possibilities[i]),
            mus[j], kappas[j], Lambdas[j], nus[j], float(possibilities[j]),
            n,
        )  # fmt: skip
        keep = [k for k in range(len(mus)) if k not in (i, j)]
        mus = [mus[k] for k in keep] + [merged[0]]
        kappas = [kappas[k] for k in keep] + [merged[1]]
        Lambdas = [Lambdas[k] for k in keep] + [merged[2]]
        nus = [nus[k] for k in keep] + [merged[3]]
        possibilities = np.concatenate([possibilities[keep], [merged[4]]])

    return mus, kappas, Lambdas, nus, _sup_normalise(possibilities), avg_dist


# ── Predictive moments ───────────────────────────────────────────────────────


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

    `possibilities` are the (unnormalised) pooling weights of one weighting
    scheme (see _scheme_weights); they are normalised to sum to one here.
    Lambda* gets a 1e-10 ridge.
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
        Lambda_hat += weights[m] * (
            Lambdas[m] + kappas[m] * np.outer(mus[m], mus[m])
        )
    Lambda_hat -= weighted_kappa_sum * np.outer(mu_hat, mu_hat)

    # symmetric exactly (A + A' is), so Sigma below is symmetric too
    Lambda_hat = 0.5 * (Lambda_hat + Lambda_hat.T)
    Lambda_hat += 1e-10 * np.eye(n, dtype=float)
    return mu_hat, _sigma_mode(Lambda_hat, weighted_nu_sum)


def _scheme_weights(
    possibilities: np.ndarray,
    necessities: np.ndarray | None,
    gamma: float,
    eta: float,
) -> dict[str, np.ndarray]:
    """Pooling weights of the models under each weighting scheme.

    "plain" (the possibilities) is always there; the necessity schemes are
    added when necessity scores are given. A new scheme is one more entry.
    """
    weights = {"plain": possibilities}
    if necessities is not None:
        weights["masked"] = deterministic_mask_weighting(
            necessities, possibilities
        )
        weights["power"] = power_weighting(
            necessities, possibilities, gamma=gamma
        )
        weights["exp"] = exponential_penalty_weighting(
            necessities, possibilities, eta=eta
        )
    return weights


def _forecast_schemes(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    scheme_weights: dict[str, np.ndarray],
    skipped: dict[str, int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One predictive (mu, Sigma) per weighting scheme.

    A scheme whose aggregation fails, or gives a non-finite forecast, is left
    out and counted in `skipped` (its row stays NaN).
    """
    n = len(mus[0])
    forecasts = {}
    for scheme, weights in scheme_weights.items():
        try:
            mu, Sigma = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, weights, n
            )
        except ValueError:
            skipped["aggregation failed"] += 1
            continue
        if not (np.isfinite(mu).all() and np.isfinite(Sigma).all()):
            skipped["non-finite forecast"] += 1
            continue
        forecasts[scheme] = (mu, Sigma)
    return forecasts


def _step_diagnostics(
    n_raw: int,
    n_pruned: int,
    nus: list[float],
    necessities: np.ndarray | None,
    hellinger_avg: float,
) -> dict[str, float]:
    """One row of diag_df: pool sizes (after birth, pruning, merging),
    necessity statistics (with necessity only), nu and merge distance."""
    record: dict[str, float] = {
        "n_models_raw": n_raw,
        "n_models_post_prune": n_pruned,
        "n_models_post_merge": len(nus),
    }
    if necessities is not None:
        record["nec_mean"] = float(necessities.mean())
        record["nec_max"] = float(necessities.max())
        record["frac_nec_zero"] = float(np.mean(necessities == 0.0))
    record["nu_mean"] = float(np.mean(nus))
    record["nu_min"] = float(np.min(nus))
    record["hellinger_avg"] = hellinger_avg
    return record


def run_core(
    returns_df: pd.DataFrame,
    *,
    start: int,
    burn_in: int = BURN_IN,
    max_models: int = POSSIBILISTIC_MAX_MODELS,
    keep_newest: bool = KEEP_NEWEST,
    merge_threshold: float | None = MERGE_THRESHOLD,
    nu_bandwidth: int = NU_BANDWIDTH,
    k_neighbours: int = K_NEIGHBOURS,
    necessity: bool = USE_NECESSITY,
    gamma: float = NECESSITY_GAMMA,
    eta: float = NECESSITY_ETA,
    device: str = NECESSITY_DEVICE,
    n_jobs: int = NECESSITY_N_JOBS,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run the possibilistic NIW model averaging algorithm.

    Args:
        returns_df: (T, n) DataFrame of excess returns in decimals, numeric columns only.
        Everything after returns_df is keyword-only; defaults come from config.
        start: first row with weights (always passed). Forecasts start at
            burn_in: rows burn_in … start-1 are warm-up (learning, no weights).
        burn_in: periods used to initialise sum_R / sum_R2 before model birth
            begins; must be <= start and < T (raises otherwise).
        merge_threshold: Hellinger distance below which two models are
            merged; None (the default) = no merging.
        max_models: maximum pool size after pruning each step.
        keep_newest: if True, always retain the newborn model after pruning.
        nu_bandwidth: max |nu_i - nu_j| for two models to be merge candidates
            (merging only).
        k_neighbours: number of forward neighbours scanned per model during
            merging (merging only).
        necessity: also compute necessity scores and the three schemes built
            on them ("masked", "power", "exp"). Off by default (slow).
        gamma: power exponent for power weighting scheme (necessity only).
        eta: decay rate for exponential penalty weighting scheme (necessity only).
        device: "cpu" or "cuda" for necessity score computation.
        n_jobs: parallel workers for necessity score computation.

    Returns (diag_df, forecasts):
        diag_df    — per-step diagnostics from burn_in on, indexed like the
                     returns (model counts, nu, Hellinger average (NaN
                     without merging); with necessity also its statistics)
        forecasts  — weighting scheme -> forecast {"mu_hat": (T,n),
                     "sigma_hat": (T,n,n), "index"}: "plain", plus
                     NECESSITY_SCHEMES with necessity.

    A step whose aggregation fails leaves that forecast row NaN; one
    RuntimeWarning at the end gives the counts.
    """
    R = returns_df.to_numpy(dtype=float)
    T, n = R.shape
    sum_R, sum_R2 = niw.burn_in_sums(R, burn_in, start)

    schemes = ("plain", *NECESSITY_SCHEMES) if necessity else ("plain",)
    mu_hat = {s: np.full((T, n), np.nan) for s in schemes}
    sigma_hat = {s: np.full((T, n, n), np.nan) for s in schemes}
    skipped = {"aggregation failed": 0, "non-finite forecast": 0}

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)
    records = []

    for t in range(burn_in, T):
        if t % 1000 == 0:
            log.info("  possibilistic: step %d / %d", t, T)

        R_t = R[t]

        # 1. birth: newborn prior from the returns before t (not R[t] yet)
        mu_bar, lam_bar = niw.new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )
        n_raw = len(mus)

        # 2. necessity scores (optional), on the models before they see R_t
        necessities = None
        if necessity:
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

        # 3. reweight (possibilistic Bayes rule), then update every model
        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        niw.update_all_models(R_t, mus, kappas, Lambdas, nus)

        # 4. aggregate: one forecast for day t+1 per weighting scheme
        scheme_weights = _scheme_weights(possibilities, necessities, gamma, eta)
        forecasts = _forecast_schemes(
            mus, kappas, Lambdas, nus, scheme_weights, skipped
        )
        for scheme, (mu, Sigma) in forecasts.items():
            mu_hat[scheme][t], sigma_hat[scheme][t] = mu, Sigma

        # 5. prune, then merge similar models (no-op when merging is off)
        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            max_models=max_models,
            keep_newest=keep_newest,
        )
        n_pruned = len(mus)
        mus, kappas, Lambdas, nus, possibilities, hellinger_avg = _merge_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            merge_threshold=merge_threshold,
            nu_bandwidth=nu_bandwidth,
            k_neighbours=k_neighbours,
        )

        records.append(
            _step_diagnostics(n_raw, n_pruned, nus, necessities, hellinger_avg)
        )
        sum_R += R_t
        sum_R2 += R_t**2

    warn_nan_rows(
        "possibilistic",
        skipped,
        (T - burn_in) * len(schemes),
        "forecast rows",
    )

    # diagnostics on the returns' index (dates; "t" = row number otherwise)
    index = returns_df.index
    diag_df = pd.DataFrame(
        records, index=index[burn_in:].rename(index.name or "t")
    )

    forecasts = {
        scheme: make_predictive(mu_hat[scheme], sigma_hat[scheme], index)
        for scheme in schemes
    }
    return diag_df, forecasts


def demo_run_core() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run a minimal smoke test of run_core on a short return sample."""
    df = prepare_returns(
        load_excess_returns(
            DATASETS_DIR / "10_Industry_Portfolios_Daily.csv",
            RF_PATH,
            start_date="2000-01-01",
        )
    )
    df_small = df.iloc[:150].copy()

    diag_df, forecasts = run_core(
        df_small,
        start=20,
        burn_in=20,
        merge_threshold=0.15,
        max_models=10,
        keep_newest=True,
        nu_bandwidth=20,
        k_neighbours=3,
        gamma=1.0,
        eta=1.0,
    )

    if diag_df.empty:
        raise RuntimeError(
            "demo_run_core produced an empty diagnostic DataFrame."
        )

    print("demo_run_core passed")
    print(diag_df.tail())
    return diag_df, forecasts


if __name__ == "__main__":
    demo_run_core()
