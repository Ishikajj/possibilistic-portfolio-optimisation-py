"""
this file is doing something unconventional known as the possibilistic analog
takes the data frame from data input


for each asset, creates it into a separate dataframe with the following properties:
mean (from t = 0 to ti), variance(from t = 0 to ti), and delta, k, scale, degrees of freedom,
inferred optimal time window using a normal inverse wishart distribution of the possibilistic form

we have a burn in period of 100 days where we form a very weak (the average of averages priors with possibility one)

now, for each point of time, we determine a series of "models": each defined by the mean, variance, delta
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from prior_selection import sharing_prior_update
from scipy.special import multigammaln
from data_input import load_excess_returns_from_kenneth_french_path, prepare_returns
from necessity_montecarlo_deterministic_hybrid import (
    deterministic_mask_weighting,
    power_weighting,
    exponential_penalty_weighting,
)
from faster_necessity import necessity_scores_fast
from annealing import necessity_scores_deterministic
from markowitz import markowitz_unconstrained

# TODO: main problem is the T^2n^3 complexity of the algorithm, where t is the time steps and n is assets
# the n^3 remains fixed as the number of assets = 11
# but T grows monsterly: we have about 70 years of data of 250 trading days each
# this becomes bad quick
# Update: 1963-2011, T = 12000 took about 40 minutes.

"""step 1
-> create a new model with mean  = common mean across all previous days and assets,
 similarly create new delta and k using the +1 update rule.
 the possibility of this new model is 1.

step 2
-> observe returns for the new day

step 3
-> using the returns, calculate the updated means and covariances using the formula given.

step 4
-> using the likelihood function defined over the supremum, calculate the updated possibilities of each model. this will require the previous convariance, new covariance, previous delta, new delta, previous k, new k, previous v, new v

step 5
-> discard the old (sigma, delta ( v - n - 1), k, mean)

step 6
-> use these updated model possibilities and the updated model variances and means to calculate intersection possibility, as written in the paper. then normalise it using supremum. this would likely have some proportionality to the niw.

step 7
-> get the mean of this model as the intersection mean and the intersection covariance, as the mode.

step 7
-> do steps 1-7 until you reach the end of time

short note:

Weights calculation:
the integer indexed series of bayesian averaged mean and covariances is used to calculate the markowitz weights using a separate function.

this function will return a integer indexed series of weights of each assets.

Portfolio return determination:
using the weights and the returns series that we get, we can run these to calculate the sharpe and certainty equivalents each day.

for each model, we have its mean, covariance, degrees of freedom k and v, and its probability. as time goes on, the number of models increases. step 1.1 indicates the models of t according to the information available at time t-1

okay so each model uses the probability as a scalar, meaning we can not have different windows across different assets even though that might be a better predictor of mean.
this is due to having a common covariance matrix across all assets, even if means differ."""


"""data input is
NoDur           float64
Durbl           float64
Manuf           float64
Enrgy           float64
Chems           float64
BusEq           float64
Telcm           float64
Utils           float64
Shops           float64
Hlth            float64
Money           float64
Other           float64
time           datetime64[ns]
"""


# corrrect
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
    mu_bar = float(mu_vec.mean())  # common initial mean across all assets.

    if t <= 1:
        return mu_bar, 1e-4

    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 1)
    var_vec = np.maximum(var_vec, 1e-8)
    return mu_bar, float(var_vec.mean())


# correct
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
    nus.append(float(n + 2))  # from our initial proof

    if possibilities.size == 0:
        return np.array([1.0], dtype=float)
    return np.concatenate([possibilities, np.array([1.0], dtype=float)])


# to check
def _log_marginal_likelihood(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log NIW marginal likelihood log L(R_t | m, F_{t-1}). This remains the same from the probabilistic case."""
    n = len(R_t)
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(d, d)

    s0, ld0 = np.linalg.slogdet(Lambda)
    s1, ld1 = slogdet(L1)
    if s0 <= 0 or s1 <= 0:
        return -np.inf

    return (
        multigammaln(nu1 / 2.0, n)
        - multigammaln(nu / 2.0, n)
        + (n / 2.0) * np.log(kappa / k1)
        + (nu / 2.0) * ld0
        - (nu1 / 2.0) * ld1
        - (n / 2.0) * np.log(np.pi)
    )


# correct
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


def _prune_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    prune_threshold: float = 1e-6,
    max_models: int | None = 100,
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
    if not (len(kappas) == len(Lambdas) == len(nus) == len(possibilities) == n_models):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return mus, kappas, Lambdas, nus, possibilities

    if max_models is not None:
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
    else:
        # No cap: fall back to threshold-based pruning
        keep_mask = possibilities >= prune_threshold
        if keep_newest:
            keep_mask[-1] = True
        keep_idx = np.flatnonzero(keep_mask)
        if keep_idx.size == 0:
            keep_idx = np.array([n_models - 1], dtype=int)

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
    Lambda_new = Lambda1 + Lambda2 + (kappa1 * kappa2 / kappa_new) * np.outer(d, d)
    nu_new = float(max(nu1 + nu2 - n, float(n + 2)))
    poss_new = float(max(poss1, poss2))
    return mu_new, kappa_new, Lambda_new, nu_new, poss_new


def _merge_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    merge_threshold: float = 1.0,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray], list[float], np.ndarray]:
    """Merge genuinely redundant model pairs based on similarity, not a target count.

    Two models are candidates for merging only if:
      1. |nu_i - nu_j| <= nu_bandwidth  — similar window length
      2. Mahalanobis distance between means < merge_threshold

    The Mahalanobis distance uses the average of the two models' covariance modes:
        Sigma_avg = 0.5 * (Lambda_i/nu_i + Lambda_j/nu_j)
        d_mah^2   = (mu_i - mu_j)^T Sigma_avg^{-1} (mu_i - mu_j)

    Search strategy: sort models by nu, then for each model scan only its
    k_neighbours forward neighbours — each pair is checked exactly once,
    no double-counting.  Complexity: O(N * k) per pass.

    Merges are applied greedily: find the most similar qualifying pair, merge
    it, repeat until no qualifying pairs remain.
    """
    if len(mus) < 2:
        return mus, kappas, Lambdas, nus, possibilities

    n = len(mus[0])
    poss_arr = np.array(possibilities, dtype=float, copy=True)

    while True:
        n_models = len(mus)
        if n_models < 2:
            break

        nu_arr = np.array(nus, dtype=float)
        order = np.argsort(nu_arr)  # sorted indices by nu

        best_dist = np.inf
        best_i = best_j = -1

        for rank in range(n_models):
            i = int(order[rank])
            for fwd in range(1, k_neighbours + 1):
                if rank + fwd >= n_models:
                    break
                j = int(order[rank + fwd])

                # Gate 1: nu proximity
                if abs(nu_arr[i] - nu_arr[j]) > nu_bandwidth:
                    break  # sorted by nu so further neighbours are even farther

                # Gate 2: Mahalanobis distance between means
                Sigma_avg = 0.5 * (Lambdas[i] / nu_arr[i] + Lambdas[j] / nu_arr[j])
                d = mus[i] - mus[j]
                try:
                    mah2 = float(d @ np.linalg.solve(Sigma_avg, d))
                except np.linalg.LinAlgError:
                    continue

                if mah2 < merge_threshold**2 and mah2 < best_dist:
                    best_dist = mah2
                    best_i, best_j = i, j

        if best_i < 0:
            break  # no qualifying pairs left

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
    if max_poss > 0.0:
        poss_arr = poss_arr / max_poss
    else:
        poss_arr = np.ones(len(poss_arr), dtype=float)

    return mus, kappas, Lambdas, nus, poss_arr


def _aggregate_possibilistic_niw(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometrically aggregate model-wise NIW possibility functions.

    Uses the passed weights (derived from possibilities and/or necessity scores)
    to compute a weighted NIW aggregate. Models with higher weight contribute more
    to the aggregate mean and scatter matrix.

    The product of NIW-shaped kernels remains NIW-shaped up to a proportionality
    constant, so we combine natural parameters and return:
    - aggregated mean location
    - aggregated covariance mode Lambda / nu
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
    periods_until_investment=0,
    prune_threshold: float = 1e-6,
    max_models: int | None = 100,
    keep_newest: bool = True,
    gamma=1.0,
    eta=1.0,
):
    """Run the possibilistic model averaging algorithm.

    Implementation choices in this version:
    - newborn model possibility is 1 before observing the new return
    - model possibility update uses supremum normalization
    - NIW parameters are updated exactly as in the probabilistic conjugate case
    - geometric aggregation uses equal weights across models
    - necessity-based reweighting is intentionally omitted for now

    returns 3 dictionaries of the predcitives.
    """
    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mu_hat_arr_masked = np.full((T, n), np.nan)
    sigma_hat_arr_masked = np.full((T, n, n), np.nan)
    mu_hat_arr_power = np.full((T, n), np.nan)
    sigma_hat_arr_power = np.full((T, n, n), np.nan)
    mu_hat_arr_exponential = np.full((T, n), np.nan)
    sigma_hat_arr_exponential = np.full((T, n, n), np.nan)

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    print("enterred loop")

    for t in range(burn_obs, T):
        if t % 100 == 0:
            print(f"Processing time step {t} / {T}...")

        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )

        n_models = len(mus)
        assert len(mus) == len(kappas) == len(Lambdas) == len(nus) == n_models
        assert len(possibilities) == n_models

        R_t = R[t]

        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            random_state=t,  # vary per step for diversity
        )

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        masked_weights = deterministic_mask_weighting(
            necessities=necessities,
            possibilities=possibilities,
        )
        power_weights = power_weighting(
            necessities=necessities,
            possibilities=possibilities,
            gamma=gamma,
        )

        exponential_weights = exponential_penalty_weighting(
            necessities=necessities,
            possibilities=possibilities,
            eta=eta,
        )

        mu_hat_masked, Sigma_hat_masked = _aggregate_possibilistic_niw(
            mus, kappas, Lambdas, nus, masked_weights, n
        )

        mu_hat_power, sigma_hat_power = _aggregate_possibilistic_niw(
            mus, kappas, Lambdas, nus, power_weights, n
        )

        mu_hat_exponential, sigma_hat_exponential = _aggregate_possibilistic_niw(
            mus, kappas, Lambdas, nus, exponential_weights, n
        )

        mu_hat_arr_masked[t] = mu_hat_masked
        sigma_hat_arr_masked[t] = Sigma_hat_masked

        mu_hat_arr_power[t] = mu_hat_power
        sigma_hat_arr_power[t] = sigma_hat_power

        mu_hat_arr_exponential[t] = mu_hat_exponential
        sigma_hat_arr_exponential[t] = sigma_hat_exponential

        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            prune_threshold=prune_threshold,
            max_models=max_models,
            keep_newest=keep_newest,
        )

        sum_R += R_t
        sum_R2 += R_t**2

    masked_predictive = {"mu_hat": mu_hat_arr_masked, "sigma_hat": sigma_hat_arr_masked}
    power_predictive = {"mu_hat": mu_hat_arr_power, "sigma_hat": sigma_hat_arr_power}
    exponential_predictive = {
        "mu_hat": mu_hat_arr_exponential,
        "sigma_hat": sigma_hat_arr_exponential,
    }

    # --- minimal persistence: save full predictive arrays to disk ---
    np.save("mu_hat_masked.npy", mu_hat_arr_masked)
    np.save("sigma_hat_masked.npy", sigma_hat_arr_masked)

    np.save("mu_hat_power.npy", mu_hat_arr_power)
    np.save("sigma_hat_power.npy", sigma_hat_arr_power)

    np.save("mu_hat_exponential.npy", mu_hat_arr_exponential)
    np.save("sigma_hat_exponential.npy", sigma_hat_arr_exponential)

    weights_masked = markowitz_unconstrained(
        mu_sigma_dict=masked_predictive,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    weights_power = markowitz_unconstrained(
        mu_sigma_dict=power_predictive,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    weights_exponential = markowitz_unconstrained(
        mu_sigma_dict=exponential_predictive,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    return (
        masked_predictive,
        weights_masked,
        power_predictive,
        weights_power,
        exponential_predictive,
        weights_exponential,
    )


def run_core_diagnostic(
    returns_df: pd.DataFrame,
    burn_in: int = 100,
    n_steps: int = 300,
    prune_threshold: float = 1e-6,
    max_models: int | None = 200,
    keep_newest: bool = True,
    gamma: float = 1.0,
    eta: float = 1.0,
) -> pd.DataFrame:
    """Run the algorithm for n_steps and record per-step diagnostics.

    Tracks the quantities most likely to reveal why the algorithm is misbehaving:
      - necessity distribution (mean, max, fraction == 0, newborn model score)
      - possibility distribution (max, entropy, effective number of models)
      - whether the masked weighting fell back to raw possibilities
      - aggregate Sigma condition number (predicts Markowitz blow-up)
      - model pool composition (n_models, mean nu, mean kappa)
    """
    import warnings

    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    t_end = min(burn_obs + n_steps, T)
    records = []

    for t in range(burn_obs, t_end):
        print(f"step {t}")
        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )
        n_models = len(mus)
        R_t = R[t]

        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            random_state=t,
        )

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        masked_weights = deterministic_mask_weighting(necessities, possibilities)
        power_weights = power_weighting(necessities, possibilities, gamma=gamma)

        # did masked weighting fall back? (all necessities == 0 → mask all → fallback)
        frac_nec_zero = float(np.mean(necessities == 0.0))
        masked_fallback = bool(np.all(necessities == 0.0))

        # possibility distribution
        poss_max = float(possibilities.max())
        poss_norm = possibilities / (possibilities.sum() + 1e-300)
        poss_entropy = float(-np.sum(poss_norm * np.log(poss_norm + 1e-300)))
        eff_n_models = float(np.exp(poss_entropy))  # effective number of models

        # aggregate Sigma condition number under masked weights
        try:
            mu_agg, Sigma_agg = _aggregate_possibilistic_niw(
                mus,
                kappas,
                Lambdas,
                nus,
                masked_weights if not masked_fallback else possibilities,
                n,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sigma_cond = float(np.linalg.cond(Sigma_agg))
            mu_norm = float(np.linalg.norm(mu_agg))
        except Exception:
            sigma_cond = np.nan
            mu_norm = np.nan

        records.append(
            {
                "t": t,
                "n_models": n_models,
                # necessity
                "nec_mean": float(necessities.mean()),
                "nec_max": float(necessities.max()),
                "nec_min": float(necessities.min()),
                "frac_nec_zero": frac_nec_zero,
                "newborn_necessity": float(necessities[-1]),  # newest model is last
                # possibility
                "poss_max": poss_max,
                "poss_entropy": poss_entropy,
                "eff_n_models": eff_n_models,
                # weighting
                "masked_fallback": masked_fallback,
                # model pool
                "nu_mean": float(np.mean(nus)),
                "nu_min": float(np.min(nus)),
                "kappa_mean": float(np.mean(kappas)),
                # aggregate quality
                "sigma_cond": sigma_cond,
                "mu_norm": mu_norm,
            }
        )

        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            prune_threshold=prune_threshold,
            max_models=max_models,
            keep_newest=keep_newest,
        )
        sum_R += R_t
        sum_R2 += R_t**2

    return pd.DataFrame(records).set_index("t")


def run_core_diagnostic_merging(
    returns_df: pd.DataFrame,
    burn_in: int = 100,
    n_steps: int = 300,
    max_models: int | None = 200,
    prune_threshold: float = 1e-6,
    keep_newest: bool = True,
    merge_threshold: float = 1.0,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
):
    """Run the algorithm with model merging and record per-step diagnostics.

    After each step: prune low-possibility models (top-k cap), then merge
    genuinely redundant pairs whose means are within merge_threshold
    Mahalanobis distance and nu within nu_bandwidth.  Aggregation uses all
    three necessity-weighted schemes (masked, power, exponential) exactly as
    in run_core.  Markowitz weights are computed via markowitz_unconstrained.

    Returns
    -------
    diag_df : pd.DataFrame
        Per-step diagnostics indexed by t.
    masked_predictive, power_predictive, exp_predictive : dict
        Each has keys "mu_hat" (T, n) and "sigma_hat" (T, n, n).
    weights_masked, weights_power, weights_exp : pd.DataFrame
        Markowitz weights for each weighting scheme, aligned to returns_df.
    """
    import warnings

    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    t_end = min(burn_obs + n_steps, T)
    records = []
    mu_hat_arr_masked = np.full((T, n), np.nan)
    sigma_hat_arr_masked = np.full((T, n, n), np.nan)
    mu_hat_arr_power = np.full((T, n), np.nan)
    sigma_hat_arr_power = np.full((T, n, n), np.nan)
    mu_hat_arr_exp = np.full((T, n), np.nan)
    sigma_hat_arr_exp = np.full((T, n, n), np.nan)

    for t in range(burn_obs, t_end):
        if t % 100 == 0:
            print(f"Processing time step {t} / {t_end}...")
        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )
        n_models_raw = len(mus)
        R_t = R[t]

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        # Step 1: prune to max_models
        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            prune_threshold=prune_threshold,
            max_models=max_models,
            keep_newest=keep_newest,
        )
        n_models_post_prune = len(mus)

        # Step 2: merge genuinely redundant pairs
        mus, kappas, Lambdas, nus, possibilities = _merge_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            merge_threshold=merge_threshold,
            nu_bandwidth=nu_bandwidth,
            k_neighbours=k_neighbours,
        )
        n_models_post_merge = len(mus)

        # Step 3: necessity on the final merged pool so lengths match possibilities
        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            random_state=t,
        )

        masked_weights = deterministic_mask_weighting(necessities, possibilities)
        power_weights = power_weighting(necessities, possibilities, gamma=gamma)
        exp_weights = exponential_penalty_weighting(necessities, possibilities, eta=eta)

        sigma_cond = np.nan
        mu_norm = np.nan
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sigma_cond = float(np.linalg.cond(Sigma_masked))
            mu_norm = float(np.linalg.norm(mu_masked))
            mu_hat_arr_masked[t] = mu_masked
            sigma_hat_arr_masked[t] = Sigma_masked
            mu_hat_arr_power[t] = mu_power
            sigma_hat_arr_power[t] = Sigma_power
            mu_hat_arr_exp[t] = mu_exp
            sigma_hat_arr_exp[t] = Sigma_exp
        except Exception:
            pass

        poss_norm = possibilities / (possibilities.sum() + 1e-300)
        poss_entropy = float(-np.sum(poss_norm * np.log(poss_norm + 1e-300)))
        eff_n_models = float(np.exp(poss_entropy))

        records.append(
            {
                "t": t,
                "n_models_raw": n_models_raw,
                "n_models_post_prune": n_models_post_prune,
                "n_models_post_merge": n_models_post_merge,
                "nec_mean": float(necessities.mean()),
                "nec_max": float(necessities.max()),
                "frac_nec_zero": float(np.mean(necessities == 0.0)),
                "masked_fallback": bool(np.all(necessities == 0.0)),
                "poss_entropy": poss_entropy,
                "eff_n_models": eff_n_models,
                "nu_mean": float(np.mean(nus)),
                "nu_min": float(np.min(nus)),
                "sigma_cond": sigma_cond,
                "mu_norm": mu_norm,
            }
        )

        sum_R += R_t
        sum_R2 += R_t**2

    diag_df = pd.DataFrame(records).set_index("t")

    masked_predictive = {"mu_hat": mu_hat_arr_masked, "sigma_hat": sigma_hat_arr_masked}
    power_predictive = {"mu_hat": mu_hat_arr_power, "sigma_hat": sigma_hat_arr_power}
    exp_predictive = {"mu_hat": mu_hat_arr_exp, "sigma_hat": sigma_hat_arr_exp}

    # --- minimal persistence: save diagnostic predictive arrays ---
    np.save("diag_mu_hat_masked.npy", mu_hat_arr_masked)
    np.save("diag_sigma_hat_masked.npy", sigma_hat_arr_masked)

    np.save("diag_mu_hat_power.npy", mu_hat_arr_power)
    np.save("diag_sigma_hat_power.npy", sigma_hat_arr_power)

    np.save("diag_mu_hat_exp.npy", mu_hat_arr_exp)
    np.save("diag_sigma_hat_exp.npy", sigma_hat_arr_exp)

    weights_masked = markowitz_unconstrained(
        masked_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
    )
    weights_power = markowitz_unconstrained(
        power_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
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


def main_diagnostic():
    import matplotlib.pyplot as plt
    import bayesian_averaging
    from portfolio_evaluation_functions import (
        calculate_sharpe_ratio,
        rolling_sharpe_ratio,
        portfolio_returns as pf_returns,
    )
    from distribution_evalutation_func import average_log_likelihood

    BURN_IN = 100
    N_STEPS = 5000
    MAX_MODELS = 100
    MERGE_THRESHOLD = 0.5
    NU_BANDWIDTH = 50
    K_NEIGHBOURS = 3
    ROLL_WINDOW = 60

    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(
            start_date="2000-01-01",
        )
    )
    burn_obs = min(BURN_IN, len(df) // 2)
    t_end = min(burn_obs + N_STEPS, len(df))
    returns_window = df.iloc[burn_obs + 1 : t_end]

    eval_slice = slice(burn_obs + 1, t_end)

    print(len(returns_window.columns))

    print("Running merged possibilistic diagnostic...")
    (
        diag,
        masked_pred,
        weights_masked_df,
        power_pred,
        weights_power_df,
        exp_pred,
        weights_exp_df,
    ) = run_core_diagnostic_merging(
        df,
        burn_in=BURN_IN,
        n_steps=N_STEPS,
        max_models=MAX_MODELS,
        merge_threshold=MERGE_THRESHOLD,
        nu_bandwidth=NU_BANDWIDTH,
        k_neighbours=K_NEIGHBOURS,
    )

    print("Running Bayesian averaging...")
    bay_ms, weights_bay_df = bayesian_averaging.run_core(
        df, burn_in=BURN_IN, periods_until_investment=0
    )
    mu_bay_df = pd.DataFrame(bay_ms["mu_hat"], index=df.index, columns=df.columns)

    # ── Slice weights to evaluation window ───────────────────────────────────
    w_masked = weights_masked_df.iloc[burn_obs + 1 : t_end]
    w_power = weights_power_df.iloc[burn_obs + 1 : t_end]
    w_exp = weights_exp_df.iloc[burn_obs + 1 : t_end]
    w_bay = weights_bay_df.iloc[burn_obs + 1 : t_end]

    strategies = {
        "masked": w_masked,
        "power": w_power,
        "exp": w_exp,
        "bayesian": w_bay,
    }

    # ── Sharpes ───────────────────────────────────────────────
    sharpes = {
        name: calculate_sharpe_ratio(returns_window, w, scaling_factor=252)
        for name, w in strategies.items()
    }
    rolls = {
        name: rolling_sharpe_ratio(
            returns_window, w, window=ROLL_WINDOW, scaling_factor=252
        )
        for name, w in strategies.items()
    }
    pf_rets = {
        name: pf_returns(returns_window, w, lag=1) for name, w in strategies.items()
    }

    # ── Average log-likelihoods ───────────────────────────────
    masked_pred_window = {
        "mu_hat": masked_pred["mu_hat"][eval_slice],
        "sigma_hat": masked_pred["sigma_hat"][eval_slice],
    }
    power_pred_window = {
        "mu_hat": power_pred["mu_hat"][eval_slice],
        "sigma_hat": power_pred["sigma_hat"][eval_slice],
    }
    exp_pred_window = {
        "mu_hat": exp_pred["mu_hat"][eval_slice],
        "sigma_hat": exp_pred["sigma_hat"][eval_slice],
    }
    bay_ms_window = {
        "mu_hat": bay_ms["mu_hat"][eval_slice],
        "sigma_hat": bay_ms["sigma_hat"][eval_slice],
    }

    avg_lls = {
        "masked": average_log_likelihood(returns_window, masked_pred_window),
        "power": average_log_likelihood(returns_window, power_pred_window),
        "exp": average_log_likelihood(returns_window, exp_pred_window),
        "bayesian": average_log_likelihood(returns_window, bay_ms_window),
    }

    # ── Save ────────────────────────────────────────────────────
    pd.DataFrame(pf_rets).to_csv("diagnostic_portfolio_returns.csv")
    pd.DataFrame(rolls).to_csv("diagnostic_rolling_sharpe.csv")
    pd.DataFrame(
        {
            "strategy": list(sharpes.keys()),
            "annualised_sharpe": list(sharpes.values()),
            "avg_log_likelihood": [avg_lls[k] for k in sharpes],
            "burn_in": BURN_IN,
            "n_steps": N_STEPS,
            "max_models": MAX_MODELS,
            "merge_threshold": [
                MERGE_THRESHOLD,
                MERGE_THRESHOLD,
                MERGE_THRESHOLD,
                None,
            ],
            "nu_bandwidth": [NU_BANDWIDTH, NU_BANDWIDTH, NU_BANDWIDTH, None],
            "roll_window": ROLL_WINDOW,
        }
    ).to_csv("diagnostic_sharpe_summary.csv", index=False)
    for name, w in strategies.items():
        w.to_csv(f"diagnostic_weights_{name}.csv")
    mu_bay_df.iloc[eval_slice].to_csv("diagnostic_mu_bayesian.csv")

    print(f"\n── Sharpe ───────────────────────────────────────────")
    for name, s in sharpes.items():
        print(f"  {name:12s}: {s:.4f}  (avg LL: {avg_lls[name]:.4f})")
    print(f"\n── Merge diagnostic ──────────────────────────────────")
    print(
        diag[
            [
                "n_models_raw",
                "n_models_post_prune",
                "n_models_post_merge",
                "nec_max",
                "sigma_cond",
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )

    # ── Figure 1: 6-panel merging diagnostic ────────────────────────────────
    fig1, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig1.suptitle(
        "Possibilistic Bayesian + Merging — Per-Step Diagnostics", fontsize=13
    )

    ax = axes[0, 0]
    ax.plot(diag.index, diag["nec_mean"], label="mean necessity")
    ax.plot(diag.index, diag["nec_max"], label="max necessity", linestyle="--")
    ax.set_title("Necessity scores")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(diag.index, diag["frac_nec_zero"] * 100)
    ax.set_title("% Models with necessity = 0")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(
        diag.index, diag["n_models_raw"], label="post-birth", alpha=0.4, linestyle=":"
    )
    ax.plot(diag.index, diag["n_models_post_prune"], label="post-prune", linestyle="--")
    ax.plot(diag.index, diag["n_models_post_merge"], label="post-merge", linewidth=2)
    ax.axhline(
        diag["n_models_post_merge"].median(),
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"median={diag['n_models_post_merge'].median():.0f}",
    )
    ax.set_title("Model pool size across pipeline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(diag.index, diag["poss_entropy"], label="entropy")
    ax.plot(diag.index, diag["eff_n_models"], label="eff. models", linestyle="--")
    ax.set_title("Possibility entropy + effective models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.semilogy(diag.index, diag["sigma_cond"].clip(lower=1))
    ax.set_title("Aggregate Σ condition number (log scale)")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(diag.index, diag["nu_mean"], label="mean ν")
    ax.plot(diag.index, diag["nu_min"], label="min ν", linestyle="--")
    ax.set_title("NIW degrees of freedom ν across models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig1.tight_layout()
    plt.savefig("diagnostic_merging.png", dpi=150)

    # ── Figure 2: Sharpe comparison ──────────────────────────────────────────
    colors_map = {
        "masked": "steelblue",
        "power": "seagreen",
        "exp": "mediumpurple",
        "bayesian": "darkorange",
    }
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        f"Weighting schemes vs Bayesian  (burn={BURN_IN}, steps={N_STEPS})", fontsize=12
    )

    ax = axes2[0]
    labels = list(sharpes.keys())
    values = list(sharpes.values())
    clrs = [colors_map[k] for k in labels]
    bars = ax.bar(labels, values, color=clrs, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Annualised Sharpe Ratio")
    ax.grid(axis="y", alpha=0.3)

    ax = axes2[1]
    for name, roll in rolls.items():
        ax.plot(
            roll.index, roll.values, label=name, color=colors_map[name], linewidth=1.1
        )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"Rolling Sharpe (window={ROLL_WINDOW}d)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig2.tight_layout()
    plt.savefig("sharpe_comparison_merging.png", dpi=150)
    plt.show()

    return diag, pf_rets


if __name__ == "__main__":
    main_diagnostic()
