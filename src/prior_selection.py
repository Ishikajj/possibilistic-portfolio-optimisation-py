"""Different kinds of prior functions are available for import here."""

import pandas as pd
import numpy as np


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
