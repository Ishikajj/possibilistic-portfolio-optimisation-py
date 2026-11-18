"""Normal-Inverse-Wishart (NIW) building blocks shared by both averaging
methods (bayesian_averaging, possibilistic_bayesian).

A model pool is four parallel lists, one entry per model:
    mus      (n,) mean vectors
    kappas   precision of the mean
    Lambdas  (n, n) scale matrices
    nus      degrees of freedom
plus one array of model weights kept by each method (probabilities, summing
to one; possibilities, with maximum one).

Only what is identical in the two methods lives here: the burn-in sums, the
newborn prior, the conjugate update and the "top models" selection used by
pruning. Scores, weight updates, pruning rules and aggregation differ on
purpose and stay in each method's module.
"""

from __future__ import annotations

import numpy as np

Pool = tuple[list[np.ndarray], list[float], list[np.ndarray], list[float]]


def burn_in_sums(
    R: np.ndarray, burn_in: int, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sums of the returns and squared returns over the burn-in rows
    (rows 0 … burn_in-1); they set the prior of the first models.

    Raises unless ``0 <= burn_in <= start`` and ``burn_in < len(R)``: models
    are born from row burn_in on, so weights can't start before it.
    """
    if not (0 <= burn_in <= start and burn_in < len(R)):
        raise ValueError(
            f"need 0 <= burn_in <= start and burn_in < number of rows; got "
            f"burn_in={burn_in}, start={start}, rows={len(R)}"
        )
    R_burn = R[:burn_in]
    return R_burn.sum(axis=0), (R_burn * R_burn).sum(axis=0)


def new_model_prior(
    sum_R: np.ndarray, sum_R2: np.ndarray, t: int
) -> tuple[float, float]:
    """Scalar prior hyperparameters for the model born at step t.

    Anderson & Cheng's common prior across assets, from the t returns seen so
    far (rows 0 … t-1; sum_R / sum_R2 don't include R[t] yet):
    - mean = average across assets of the per-asset sample means
    - covariance scale = average across assets of the per-asset sample
      variances (ddof=1), each floored at 1e-8

    Returns (mu_bar, lambda_bar); lambda_bar is 1e-4 while there are fewer
    than two returns.
    """
    if t <= 0:
        return 0.0, 1e-4

    mu_vec = sum_R / float(t)  # (n,) per-asset sample means
    mu_bar = float(mu_vec.mean())
    if t <= 1:  # no sample variance from one observation
        return mu_bar, 1e-4

    # Per-asset sample variance: (sum R_i^2 - t * mean_i^2) / (t - 1)
    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 1)
    var_vec = np.maximum(var_vec, 1e-8)
    return mu_bar, float(var_vec.mean())


def update_niw(
    R_t: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    Lambda: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """One-step NIW conjugate update of one model after observing R_t."""
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    mu1 = (kappa * mu + R_t) / k1
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(d, d)
    return mu1, k1, L1, nu1


def update_all_models(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
) -> None:
    """Update every model's NIW parameters in place."""
    for m in range(len(mus)):
        mus[m], kappas[m], Lambdas[m], nus[m] = update_niw(
            R_t, mus[m], kappas[m], Lambdas[m], nus[m]
        )


def top_models(
    scores: np.ndarray, max_models: int, keep_newest: bool
) -> np.ndarray:
    """Positions (ascending) of the ``max_models`` highest-scoring models.

    With ``keep_newest`` the newest model (the last one) always stays: if it
    isn't among the top, it takes the place of the lowest-ranked of them.
    """
    n_models = len(scores)
    if n_models <= max_models:
        return np.arange(n_models, dtype=int)
    keep = np.argsort(scores)[::-1][:max_models]  # best first
    if keep_newest and (n_models - 1) not in keep:
        keep[-1] = n_models - 1
    return np.sort(keep)


def select_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    keep_idx: np.ndarray,
) -> Pool:
    """The pool restricted to the models at ``keep_idx``."""
    return (
        [mus[i] for i in keep_idx],
        [kappas[i] for i in keep_idx],
        [Lambdas[i] for i in keep_idx],
        [nus[i] for i in keep_idx],
    )
