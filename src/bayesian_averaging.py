"""Probabilistic Bayesian model averaging over Normal-Inverse-Wishart (NIW) models.

At each time step t the algorithm maintains a pool of NIW models, each representing
a different window for determining the mean and the covariance. The pool grows by one model per period
and is pruned to prevent unbounded growth.

Algorithm (per time step)
-------------------------
1. Birth   — spawn a new NIW model whose prior mean and scale are set to the
             cross-asset average of historical sample moments.
2. Observe — receive the realised return vector R_t.
3. Update  — apply the NIW conjugate update (equations 7a/7b) to every model.
4. Reweight — compute each model's marginal likelihood and update probabilities
              via Bayes' rule; probabilities are normalised to sum to one.
5. Prune   — discard low-probability models to keep the pool tractable.
6. Predict — form the BMA predictive mean and covariance as the
             probability-weighted average across models.

Steps 1–6 repeat for every observation, producing a time-indexed series of
predictive moments (mu_hat, sigma_hat) which are passed to the Markowitz
solver to generate portfolio weights.

Design constraint
-----------------
Each model shares a single scalar probability, so the same candidate window
applies to all assets jointly. Per-asset windows are not supported because the
covariance matrix is common across assets.

Entry point
-----------
run_core(returns_df, burn_in, periods_until_investment, ...)
    Input : returns_df — pd.DataFrame of shape (T, n), numeric excess returns.
    Output: (mu_sigma_dict, weights_df)
        mu_sigma_dict — {"mu_hat": ndarray (T, n), "sigma_hat": ndarray (T, n, n)}
        weights_df    — pd.DataFrame (T, n), Markowitz weights aligned to returns_df.index
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from scipy.special import multigammaln
from data_input import (
    load_excess_returns_from_kenneth_french_path,
    prepare_returns,
)
from markowitz import markowitz_unconstrained


def sharing_prior_update(
    probs_prev: np.ndarray,
    t_models: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Birth-step prior update with sharing parameter alpha in [0,1].

    Data structures:
      - probs_prev: shape (t_models-1,) representing P_{t-1}(m | F_{t-1}) for m=1..t-1
      - returns: shape (t_models,) representing P_t(m | F_{t-1}) for m=1..t

    alpha=1.0 -> perfect sharing (paper's alpha=1 case)
    alpha=0.0 -> no sharing (old models keep their prob; newborn gets 0)
    """
    if t_models < 1:
        raise ValueError("t_models must be >= 1")
    if t_models == 1:
        return np.array([1.0])

    if probs_prev.shape[0] != t_models - 1:
        raise ValueError("probs_prev must have length t_models-1")

    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    # Retained mass for existing models
    out = np.zeros(t_models, dtype=float)
    out[: t_models - 1] = (1.0 - a) * probs_prev

    # Shared mass: each old model q shares alpha*P_{t-1}(q) equally with itself and newer models
    # Contribution from q (1-based) to any m>=q is alpha*P_{t-1}(q)/(t - q + 1)
    q = np.arange(1, t_models)  # 1..t-1
    share_each = a * probs_prev / (t_models - q + 1.0)  # length t-1

    # For model m (1-based), add sum_{q<=m} share_each[q]
    # (cumsum handles this for m=1..t-1)
    out[: t_models - 1] += np.cumsum(share_each)

    # Newborn model m=t gets all shared parts from all previous models
    out[t_models - 1] = (
        share_each.sum()
    )  # cumsum gives the series of pasts (s1, s2, s3) and sum just gives the final element of the cumsum series.

    # Numerical safety
    out = np.maximum(out, 0.0)
    s = out.sum()
    if s <= 0:
        # fallback: uniform
        return np.full(t_models, 1.0 / t_models)
    return out / s  # makes sure that we normalise the vector to sum to 1.


def _new_model_prior(
    sum_R: np.ndarray, sum_R2: np.ndarray, t: int
) -> tuple[float, float]:
    """Compute the scalar prior hyperparameters for the newborn model at time t.
    This implements the paper's *common* (across assets) prior for the new model:
    """
    if t <= 0:
        return 0.0, 1e-4

    mu_vec = sum_R / (
        float(t) - 1
    )  # shape (n,), only information until t-1 is used to form the new models priors.
    mu_bar = float(mu_vec.mean())

    if t <= 1:
        # Cannot compute sample variance with < 2 observations.
        return mu_bar, 1e-4

    # Per-asset sample variance: Var(R_i) = (sum R_i^2 - t * mean_i^2) / (t-1)
    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 2)
    var_vec = np.maximum(var_vec, 1e-8)
    return mu_bar, float(var_vec.mean())


def _append_new_model(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    mu_bar: float,
    lam_bar: float,
    n: int,
) -> None:
    """takes the previously existing models, and the newly caclulated variance and mean, and appends those to the previously existing models."""
    mus.append(np.full(n, mu_bar, dtype=float))
    kappas.append(1.0)
    Lambdas.append(lam_bar * np.eye(n, dtype=float))
    nus.append(
        float(n + 2.0)
    )  # paper used delta = 1 at initialisation, and delta = nu - n - 1


def _update_probs(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
) -> np.ndarray:

    # ll is an array of the log of the likelihood functions for each model given our observed returns.
    ll = np.array(
        [
            _log_marginal_likelihood(
                R_t,
                mus[m],
                kappas[m],
                Lambdas[m],
                nus[m],
            )
            for m in range(len(mus))
        ]
    )

    # logaddexp basically does log(sum(exp(i))).
    # since i is our log of probabilities, this is log (sum of probabilities)
    # so we get the denominator of bayes rule in log space.
    lognum = ll + np.log(np.maximum(probs, 1e-300))
    logZ = np.logaddexp.reduce(
        lognum
    )  # logaddexp only takes 2 inputs, reduce applies repeatedly.

    p = np.exp(lognum - logZ)
    p = np.maximum(
        p, 0.0
    )  # effectively handing cases where the likelihood is -inf
    return p / p.sum()  # normalise to sum 1.


def _log_marginal_likelihood(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> float:
    """Log NIW marginal likelihood log L(R_t | m, F_{t-1}).

    Uses `scipy.special.multigammaln(a, d)`, which returns `log Γ_d(a)` (the natural
    log of the multivariate gamma function). We work in logs because Γ_d(a) and the
    NIW normalizing constants can be extremely large/small; log-space avoids overflow
    and turns products/ratios into sums/differences.

    the formulae used here are from anderson cheng section 3.2.
    """
    n = len(R_t)
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(
        d, d
    )  # Paper uses sigma for updates, which has been simplified here to Lambda. this is also why the delta denominator vanishes.

    s0, ld0 = np.linalg.slogdet(
        Lambda
    )  # computes the sign, log of determinant of Lambda. Due to PD, this must be +.
    s1, ld1 = slogdet(L1)
    if s0 <= 0 or s1 <= 0:
        return (
            -np.inf
        )  # indicates cases of mathematically imposible covariance matrices.

    log_likelihood = (
        multigammaln(nu1 / 2.0, n)
        - multigammaln(nu / 2.0, n)
        + (n / 2.0) * np.log(kappa / k1)
        + (nu / 2.0) * ld0  # determinant was already logged.
        - (nu1 / 2.0) * ld1
        - (n / 2.0) * np.log(np.pi)
    )
    return log_likelihood


def _update_all_models(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
) -> None:
    for m in range(len(mus)):
        mus[m], kappas[m], Lambdas[m], nus[m] = _update_niw(
            R_t, mus[m], kappas[m], Lambdas[m], nus[m]
        )


def _update_niw(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    mu1 = (kappa * mu + R_t) / k1
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(d, d)
    return mu1, k1, L1, nu1


def _ba_moments(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """formulae from section 3.1"""

    mu_hat = sum(probs[m] * mus[m] for m in range(len(mus)))
    Sigma_hat = np.zeros((n, n))
    for m in range(len(mus)):
        Sigma_m = _sigma_m(Lambdas[m], nus[m], n)
        Sigma_bar = (1.0 + 1.0 / kappas[m]) * Sigma_m
        Sigma_hat += probs[m] * (Sigma_bar + np.outer(mus[m], mus[m]))
    Sigma_hat -= np.outer(mu_hat, mu_hat)
    return mu_hat, Sigma_hat


def _sigma_m(Lambda: np.ndarray, nu: float, n: int) -> np.ndarray:
    return Lambda / (nu - n - 1.0)


def _prune_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
    prune_threshold: float = 1e-6,
    max_models: int | None = 500,
    keep_newest: bool = True,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Prune low-probability models to keep runtime sub-quadratic in T.

    Strategy:
    - drop models with posterior probability below `prune_threshold`
    - optionally cap the total number of retained models at `max_models`
    - optionally force retention of the newest model (last index)

    After pruning, probabilities are renormalized to sum to 1.

    Generally the threshhold is kept to 1e-6 = 0.0001% to prevent numerical issues.
    """
    n_models = len(mus)

    keep_mask = probs >= prune_threshold
    if keep_newest:
        keep_mask[-1] = True

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        fallback_count = (
            n_models if max_models is None else min(n_models, max_models)
        )
        keep_idx = np.arange(n_models - fallback_count, n_models, dtype=int)

    if max_models is not None and keep_idx.size > max_models:
        order = np.argsort(probs[keep_idx])[
            ::-1
        ]  # sorts by descending the probabilities.
        keep_idx = keep_idx[order[:max_models]]
        keep_idx = np.sort(keep_idx)
        if keep_newest and keep_idx[-1] != n_models - 1:
            keep_idx[-1] = n_models - 1
            keep_idx = np.unique(np.sort(keep_idx))
            if keep_idx.size > max_models:
                drop_candidates = keep_idx[keep_idx != n_models - 1]
                smallest = drop_candidates[np.argmin(probs[drop_candidates])]
                keep_idx = keep_idx[keep_idx != smallest]

    mus = [mus[i] for i in keep_idx]
    kappas = [kappas[i] for i in keep_idx]
    Lambdas = [Lambdas[i] for i in keep_idx]
    nus = [nus[i] for i in keep_idx]
    probs = probs[keep_idx].astype(float, copy=False)

    prob_sum = probs.sum()
    if prob_sum <= 0.0:
        probs = np.full(len(probs), 1.0 / len(probs))
    else:
        probs = probs / prob_sum

    return mus, kappas, Lambdas, nus, probs


def run_core(
    returns_df: pd.DataFrame,
    burn_in: int = 1000,
    periods_until_investment: int = 0,
    prune_threshold: float = 1e-6,
    max_models: int | None = 2000,
    keep_newest: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Run probabilistic Bayesian model averaging over NIW models.

    Input : returns_df — (T, n) numeric excess returns, no Date/RF column.
    Output: (mu_sigma_dict, weights_df)
        mu_sigma_dict — {"mu_hat": (T, n), "sigma_hat": (T, n, n)}, NaN before burn-in.
        weights_df    — Markowitz weights (T, n), aligned to returns_df.index.
    """
    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mu_hat_arr = np.full((T, n), np.nan)
    sigma_hat_arr = np.full((T, n, n), np.nan)

    # mu is the mean for each model, kappa is the precision about the mean
    # Lambda is the scale parameter for the covariance matrix, and nu is the degrees of freedom.
    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    probs = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    # burn_obs must be an integer for array slicing
    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    for t in range(burn_obs, T):
        if t % 100 == 0:
            print(f"Processing time step {t} / {T}...")
        mu_bar, lam_bar = _new_model_prior(
            sum_R, sum_R2, t
        )  # Note, this is not using the R[t] yet
        _append_new_model(mus, kappas, Lambdas, nus, mu_bar, lam_bar, n)

        n_models = len(mus)

        probs = sharing_prior_update(
            probs_prev=probs, t_models=n_models, alpha=1.0
        )

        # Now, after a new model has been added, it has been given a weak prior based on previous information flow
        # Only after that we observe the returns for the day.
        R_t = R[t]

        probs = _update_probs(R_t, mus, kappas, Lambdas, nus, probs)

        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        mus, kappas, Lambdas, nus, probs = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            probs,
            prune_threshold=prune_threshold,
            max_models=max_models,
            keep_newest=keep_newest,
        )

        mu_hat, Sigma_hat = _ba_moments(mus, kappas, Lambdas, nus, probs, n)
        mu_hat_arr[t] = mu_hat
        sigma_hat_arr[t] = Sigma_hat

        sum_R += R_t
        sum_R2 += R_t**2
    mu_sigma_dict = {"mu_hat": mu_hat_arr, "sigma_hat": sigma_hat_arr}

    weights = markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    # VERY IMPORTANT: the returned mu_hat_arr and sigma_hat_arr have info upto time t, meaning they are predicting t+1.
    return mu_sigma_dict, weights


def main() -> None:
    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(start_date="2020-01-01")
    )
    predictives, weights = run_core(df, burn_in=100)
    print(predictives["mu_hat"])
    print(predictives["sigma_hat"])


if __name__ == "__main__":
    main()
