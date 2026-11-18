"""Probabilistic Bayesian model averaging over Normal-Inverse-Wishart (NIW) models.

At each time step t the algorithm maintains a pool of NIW models, each representing
a different window for determining the mean and the covariance. The pool grows by one model per period
and is pruned to prevent unbounded growth.

Algorithm (per time step)
-------------------------
1. Birth   — spawn a new NIW model whose prior mean and scale are set to the
             cross-asset average of historical sample moments.
2. Observe — receive the realised return vector R_t.
3. Reweight — compute each model's marginal likelihood and update probabilities
              via Bayes' rule; probabilities are normalised to sum to one.
4. Update  — apply the NIW conjugate update (equations 7a/7b) to every model.
5. Prune   — discard low-probability models to keep the pool tractable.
6. Predict — form the BMA predictive mean and covariance as the
             probability-weighted average across models.

Steps 1–6 repeat for every observation, producing a time-indexed series of
predictive moments (mu_hat, sigma_hat); the pipeline turns them into
Markowitz weights (weight_rules).

Design constraint
-----------------
Each model shares a single scalar probability, so the same candidate window
applies to all assets jointly. Per-asset windows are not supported because the
covariance matrix is common across assets.

Entry point
-----------
run_core(returns_df, *, start, burn_in, ...)
    Input : returns_df — pd.DataFrame of shape (T, n), numeric excess returns.
    Output: the forecast dict (predictive.make_predictive):
        {"mu_hat": ndarray (T, n), "sigma_hat": ndarray (T, n, n), "index"}
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from scipy.special import multigammaln

from possibilistic_portfolio import niw
from possibilistic_portfolio.config import (
    BAYESIAN_MAX_MODELS,
    BAYESIAN_PRUNE_THRESHOLD,
    BURN_IN,
    DATASETS_DIR,
    KEEP_NEWEST,
    RF_PATH,
)
from possibilistic_portfolio.data_input import (
    load_excess_returns,
    prepare_returns,
)
from possibilistic_portfolio.predictive import make_predictive

log = logging.getLogger(__name__)


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


def _append_new_model(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    mu_bar: float,
    lam_bar: float,
    n: int,
) -> None:
    """Append the newborn model: the common prior (mu_bar, lam_bar) for every
    asset, kappa = 1 and nu = n + 2 (the smallest nu whose inverse-Wishart
    mean exists; Anderson & Cheng start at delta = nu - n - 1 = 1)."""
    mus.append(np.full(n, mu_bar, dtype=float))
    kappas.append(1.0)
    Lambdas.append(lam_bar * np.eye(n, dtype=float))
    nus.append(float(n + 2.0))


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
    L1 = (
        Lambda + (kappa / k1) * np.outer(d, d)
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


def _ba_moments(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """formulae from section 3.1"""

    mu_hat = sum((probs[m] * mus[m] for m in range(len(mus))), np.zeros(n))
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
    prune_threshold: float = BAYESIAN_PRUNE_THRESHOLD,
    max_models: int | None = BAYESIAN_MAX_MODELS,
    keep_newest: bool = KEEP_NEWEST,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Prune low-probability models to keep runtime sub-quadratic in T.

    - drop models with posterior probability below `prune_threshold`
    - of the rest, keep at most the `max_models` most probable (None = no cap)
    - with `keep_newest` the newest model (last index) survives both steps;
      under the cap it takes the place of the least probable kept model

    After pruning, probabilities are renormalised to sum to 1.
    """
    n_models = len(mus)

    keep_mask = probs >= prune_threshold
    if keep_newest:
        keep_mask[-1] = True

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:  # nothing above the threshold: keep the newest
        fallback_count = (
            n_models if max_models is None else min(n_models, max_models)
        )
        keep_idx = np.arange(n_models - fallback_count, n_models, dtype=int)

    if max_models is not None:
        keep_idx = keep_idx[
            niw.top_models(probs[keep_idx], max_models, keep_newest)
        ]

    mus, kappas, Lambdas, nus = niw.select_models(
        mus, kappas, Lambdas, nus, keep_idx
    )
    probs = probs[keep_idx].astype(float, copy=False)

    prob_sum = probs.sum()
    if prob_sum <= 0.0:
        probs = np.full(len(probs), 1.0 / len(probs))
    else:
        probs = probs / prob_sum

    return mus, kappas, Lambdas, nus, probs


def run_core(
    returns_df: pd.DataFrame,
    *,
    start: int,
    burn_in: int = BURN_IN,
    max_models: int | None = BAYESIAN_MAX_MODELS,
    prune_threshold: float = BAYESIAN_PRUNE_THRESHOLD,
    keep_newest: bool = KEEP_NEWEST,
) -> dict:
    """Run probabilistic Bayesian model averaging over NIW models.

    Input : returns_df — (T, n) numeric excess returns, no RF column.
            start — first row with weights (always passed; only checked
                against burn_in here).
            burn_in — first rows used only to set the prior; must be
                <= start and < T (raises otherwise).
            Everything after returns_df is keyword-only; defaults: config.
    Output: the forecast {"mu_hat": (T, n), "sigma_hat": (T, n, n), "index"},
        NaN before burn_in (rows burn_in … start-1 are warm-up). Row t uses
        returns up to and including t: it predicts t+1.
    """
    R = returns_df.to_numpy(dtype=float)
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

    sum_R, sum_R2 = niw.burn_in_sums(R, burn_in, start)

    for t in range(burn_in, T):
        if t % 1000 == 0:
            log.info("  bayesian averaging: step %d / %d", t, T)
        # prior of the newborn model: returns before t only (not R[t] yet)
        mu_bar, lam_bar = niw.new_model_prior(sum_R, sum_R2, t)
        _append_new_model(mus, kappas, Lambdas, nus, mu_bar, lam_bar, n)

        n_models = len(mus)

        probs = sharing_prior_update(
            probs_prev=probs, t_models=n_models, alpha=1.0
        )

        # Now, after a new model has been added, it has been given a weak prior based on previous information flow
        # Only after that we observe the returns for the day.
        R_t = R[t]

        probs = _update_probs(R_t, mus, kappas, Lambdas, nus, probs)

        niw.update_all_models(R_t, mus, kappas, Lambdas, nus)

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
    return make_predictive(mu_hat_arr, sigma_hat_arr, returns_df.index)


def main() -> None:
    df = prepare_returns(
        load_excess_returns(
            DATASETS_DIR / "10_Industry_Portfolios_Daily.csv",
            RF_PATH,
            start_date="2020-01-01",
        )
    )
    forecast = run_core(df, start=100, burn_in=100)
    print(forecast["mu_hat"])
    print(forecast["sigma_hat"])


if __name__ == "__main__":
    main()
