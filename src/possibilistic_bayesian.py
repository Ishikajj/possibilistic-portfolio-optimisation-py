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

    mu_vec = sum_R / float(t - 1)
    mu_bar = float(mu_vec.mean())  # common initial mean across all assets.

    if t <= 1:
        return mu_bar, 1e-4

    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 2)
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
    nus.append(float(0.0))  # from our initial proof

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
    max_models: int | None = 200,
    keep_newest: bool = True,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Prune low-possibility models to control runtime growth.

    Strategy:
    - drop models with possibility below `prune_threshold`
    - optionally cap the total number of retained models at `max_models`
    - optionally force retention of the newest model (last index)

    After pruning, possibilities are renormalized to have supremum one.
    """
    n_models = len(mus)
    if not (len(kappas) == len(Lambdas) == len(nus) == len(possibilities) == n_models):
        raise ValueError("All model containers must have the same length.")
    if n_models == 0:
        return mus, kappas, Lambdas, nus, possibilities

    keep_mask = possibilities >= prune_threshold
    if keep_newest:
        keep_mask[-1] = True

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        fallback_count = n_models if max_models is None else min(n_models, max_models)
        keep_idx = np.arange(n_models - fallback_count, n_models, dtype=int)

    if max_models is not None and keep_idx.size > max_models:
        order = np.argsort(possibilities[keep_idx])[::-1]
        keep_idx = keep_idx[order[:max_models]]
        keep_idx = np.sort(keep_idx)
        if keep_newest and keep_idx[-1] != n_models - 1:
            keep_idx[-1] = n_models - 1
            keep_idx = np.unique(np.sort(keep_idx))
            if keep_idx.size > max_models:
                drop_candidates = keep_idx[keep_idx != n_models - 1]
                smallest = drop_candidates[np.argmin(possibilities[drop_candidates])]
                keep_idx = keep_idx[keep_idx != smallest]

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


def _aggregate_possibilistic_niw(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    possibilities: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometrically aggregate model-wise NIW possibility functions.

    For now, use equal geometric weights across all active models.
    Model possibilities still matter because they determine which models survive as
    highly plausible over time, but they are not yet used as aggregation weights.

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
    prune_threshold: float = 1e-6,
    max_models: int | None = 200,
    keep_newest: bool = True,
) -> dict[str, np.ndarray]:
    """Run the possibilistic model averaging algorithm.

    Implementation choices in this version:
    - newborn model possibility is 1 before observing the new return
    - model possibility update uses supremum normalization
    - NIW parameters are updated exactly as in the probabilistic conjugate case
    - geometric aggregation uses equal weights across models
    - necessity-based reweighting is intentionally omitted for now
    """
    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mu_hat_arr = np.full((T, n), np.nan)
    sigma_hat_arr = np.full((T, n, n), np.nan)

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
        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

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

        mu_hat, Sigma_hat = _aggregate_possibilistic_niw(
            mus, kappas, Lambdas, nus, possibilities, n
        )
        mu_hat_arr[t] = mu_hat
        sigma_hat_arr[t] = Sigma_hat

        sum_R += R_t
        sum_R2 += R_t**2

    return {
        "mu_hat": mu_hat_arr,
        "sigma_hat": sigma_hat_arr,
    }


def main():
    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(start_date="2020-01-01")
    )
    results = run_core(df, burn_in=100)
    print(results["mu_hat"])
    print(results["sigma_hat"])


if __name__ == "__main__":
    main()
