"""Benchmark forecasts: predictive moments of the classical strategies.

Each forecaster returns a forecast dict (predictive.make_predictive:
"mu_hat" (T, n), "sigma_hat" (T, n, n), "index"); the pipeline turns every
forecast into Markowitz weights (weight_rules). The weight-only benchmarks
(1/N, market, minimum variance) are in weight_rules.

  historical_expanding_forecast — sample mean / covariance of all returns so far
  rolling_window_forecast       — the same over the last `window` returns
  jorion_bayes_stein_forecast   — Jorion (1986): mean shrunk towards the
                                  minimum-variance mean, inflated covariance
  kan_zhou_three_fund_forecast  — Kan & Zhou (2007): their shrinkage, forecast
                                  (mu*, Sigma / h)

Formulas for Jorion and Kan-Zhou: Anderson & Cheng (2016), sections 6.6 / 6.7.
A row they can't estimate (after their start) is left NaN and counted in one
RuntimeWarning per call.

Signature of every forecaster: (returns_df, *, start, window) — keyword-only
after returns_df; `start`, the first row with weights, is always passed (the
pipeline computes burn_in + periods_until_investment once); window defaults
to config.ROLLING_WINDOW_LONG. Forecasts start at `start` (or at the first
full window, if later). Estimates at t use the w returns up to and including
t: iloc[t-w+1 : t+1] (timing convention: predictive.py).
"""

import numpy as np
import pandas as pd
from scipy.special import beta as _beta  # Beta(a,b)
from scipy.special import (
    betainc as _betainc,
)  # regularized incomplete beta I_x(a,b)

from possibilistic_portfolio.config import ROLLING_WINDOW_LONG
from possibilistic_portfolio.predictive import make_predictive, warn_nan_rows
from possibilistic_portfolio.weight_rules import solve_or_pinv


def _incomplete_beta_unregularized(a: float, b: float, x: float) -> float:
    r"""Return unregularized incomplete beta B_x(a,b) = \int_0^x t^{a-1} (1-t)^{b-1} dt."""
    if not (0.0 <= x <= 1.0) or not np.isfinite(x):
        return float("nan")

    # B_x(a,b) = I_x(a,b) * B(a,b)
    return float(_betainc(a, b, x) * _beta(a, b))


def _minvar_mean_target(
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    ones: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Compute Jorion/Kan-Zhou minimum-variance mean target.

    Returns:
      mu_g_scalar: (mu' Sigma^{-1} 1)/(1' Sigma^{-1} 1)
      mu_g_vec: mu_g_scalar * 1
      denom_1: 1' Sigma^{-1} 1
    """
    invSig_1 = solve_or_pinv(Sigma_t, ones)
    denom_1 = float(ones @ invSig_1)
    if denom_1 == 0.0 or not np.isfinite(denom_1):
        return (
            float("nan"),
            np.full_like(ones, np.nan, dtype=float),
            float("nan"),
        )
    mu_g_scalar = float(mu_t @ invSig_1) / denom_1
    mu_g_vec = mu_g_scalar * ones
    return mu_g_scalar, mu_g_vec, denom_1


def _shrinkage_target(
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    ones: np.ndarray,
) -> tuple[float, np.ndarray, float, float] | None:
    """What Jorion and Kan-Zhou both shrink with, for one window.

    Returns (mu_g_scalar, mu_g_vec, denom_1, psi2) — the minimum-variance
    target (see _minvar_mean_target) and psi2 = d' Sigma^{-1} d, d = mu_t -
    mu_g (Jorion calls it q) — or None if the target is undefined.
    """
    mu_g_scalar, mu_g_vec, denom_1 = _minvar_mean_target(mu_t, Sigma_t, ones)
    if not np.isfinite(mu_g_scalar):
        return None
    d = mu_t - mu_g_vec
    psi2 = float(d @ solve_or_pinv(Sigma_t, d))
    return mu_g_scalar, mu_g_vec, denom_1, psi2


def _kan_zhou_adjusted_psi2(psi2: float, n: int, w: int) -> float:
    """Kan & Zhou's adjusted estimator of psi^2 (their eq. 66; T = w, N = n):

        psi2_a = [(w-n-1) psi2 - (n-1)] / w
                 + 2 psi2^((n-1)/2) (1 + psi2)^(-(w-2)/2)
                   / (w * B_x((n-1)/2, (w-n+1)/2)),      x = psi2 / (1 + psi2)

    Anderson & Cheng (2016, section 6.7) print the second beta parameter as
    (w+1)/2; the original paper has (w-n+1)/2, used here.

    psi2_a is positive by construction and -> 0 as psi2 -> 0: returns 0 for
    psi2 <= 1e-12 (the mean already equals the target), NaN if the result
    isn't positive and finite (e.g. the incomplete beta underflows to 0).
    """
    if psi2 <= 1e-12:
        return 0.0
    a = (n - 1.0) / 2.0
    B = _incomplete_beta_unregularized(
        a=a, b=(w - n + 1.0) / 2.0, x=psi2 / (1.0 + psi2)
    )
    if not B > 0.0:
        return float("nan")
    unbiased = ((w - n - 1.0) * psi2 - (n - 1.0)) / w
    adjustment = 2.0 * psi2**a * (1.0 + psi2) ** (-(w - 2.0) / 2.0) / (w * B)
    psi2_a = unbiased + adjustment
    return psi2_a if np.isfinite(psi2_a) and psi2_a > 0.0 else float("nan")


def historical_expanding_forecast(
    returns_df: pd.DataFrame, *, start: int
) -> dict:
    """Sample mean and covariance (ddof=1) of all returns up to and including
    t (expanding window from row 0), from `start` on."""
    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(start, T):
        ret_window = returns_df.iloc[: t + 1]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov(ddof=1).values

    return make_predictive(mu_hat, sigma_hat, returns_df.index)


def rolling_window_forecast(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict:
    """Sample mean and covariance (ddof=1) of the last `window` returns up to
    and including t, from `start` (or the first full window, if later)."""
    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(max(window - 1, start), T):
        ret_window = returns_df.iloc[t - window + 1 : t + 1]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov(ddof=1).values

    return make_predictive(mu_hat, sigma_hat, returns_df.index)


def jorion_bayes_stein_estimates(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict[str, np.ndarray]:
    """Compute Jorion (1986) Bayes–Stein shrunk mean/covariance.

    Returns arrays aligned to `returns_df.index` (NaN before the start):
      - mu_star: (T,n)
      - sigma_star: (T,n,n)
      - v: (T,) shrinkage intensity
      - J: (T,) auxiliary scalar
      - mu_g_scalar: (T,) minimum-variance mean target (scalar)
    """
    T, n = returns_df.shape
    w = int(window)
    if w <= n + 2:
        raise ValueError(
            f"window must be > n+2 for c=w-n-2 to be positive; got window={w}, n={n}"
        )

    start = max(w - 1, start)  # first full window

    mu_star = np.full((T, n), np.nan)
    sigma_star = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)
    J = np.full(T, np.nan)
    mu_g_scalar_arr = np.full(T, np.nan)

    ones = np.ones(n)
    c = w - n - 2
    skipped = {"minimum-variance mean undefined": 0, "q <= 0 or not finite": 0}

    for t in range(start, T):
        ret_window = returns_df.iloc[t - w + 1 : t + 1]
        mu_t = ret_window.mean().to_numpy()

        # pandas cov(ddof=1) gives (1/(w-1)) S; paper uses (1/c) S
        cov_ddof1 = ret_window.cov(ddof=1).values
        Sigma_t = cov_ddof1 * (w - 1) / c

        target = _shrinkage_target(mu_t, Sigma_t, ones)
        if target is None:
            skipped["minimum-variance mean undefined"] += 1
            continue
        mu_g_scalar, mu_g_vec, denom_1, q = target
        if q <= 0.0 or not np.isfinite(q):
            skipped["q <= 0 or not finite"] += 1
            continue

        v_t = (n + 2.0) / ((n + 2.0) + w * q)
        mu_t_star = (1.0 - v_t) * mu_t + v_t * mu_g_vec

        J_t = (n + 2.0) / q

        Sigma_t_star = (1.0 + 1.0 / (w + J_t)) * Sigma_t + (
            (J_t / (w * (w + 1.0 + J_t))) * (np.outer(ones, ones) / denom_1)
        )

        mu_star[t] = mu_t_star
        sigma_star[t] = Sigma_t_star
        v[t] = v_t
        J[t] = J_t
        mu_g_scalar_arr[t] = mu_g_scalar

    warn_nan_rows("jorion_bayes_stein", skipped, max(T - start, 0))
    return {
        "mu_star": mu_star,
        "sigma_star": sigma_star,
        "v": v,
        "J": J,
        "mu_g_scalar": mu_g_scalar_arr,
    }


def jorion_bayes_stein_forecast(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict:
    """Jorion's Bayes-Stein forecast (mu*, Sigma*), from `start` (or the
    first full window, if later); Markowitz on it is Jorion's rule
    (1/theta) Sigma*^{-1} mu*."""
    est = jorion_bayes_stein_estimates(returns_df, start=start, window=window)
    return make_predictive(est["mu_star"], est["sigma_star"], returns_df.index)


def kan_zhou_three_fund_estimates(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict[str, np.ndarray]:
    """Compute Kan and Zhou (2007) three-fund shrinkage estimates.

    Returns arrays aligned to `returns_df.index` (NaN before the start):
      - mu_star: (T,n) shrunk mean
      - Sigma: (T,n,n) rolling MLE covariance (c=w)
      - sigma_star: (T,n,n) Sigma / h, the covariance the rule works with
      - v: (T,) shrinkage weight, in (0, 1]
      - h: scalar constant of the portfolio rule (one-element array)

    The shrinkage weight is v = n / (n + w * psi2_a), with psi2_a Kan & Zhou's
    adjusted estimator of psi^2 = (mu - mu_g)' Sigma^{-1} (mu - mu_g), see
    _kan_zhou_adjusted_psi2.
    """
    T, n = returns_df.shape
    w = int(window)
    if w <= n + 4:
        raise ValueError(
            f"window must be > n+4 for h to be well-defined; got window={w}, n={n}"
        )

    start = max(w - 1, start)  # first full window

    h = ((w - n - 1.0) * (w - n - 4.0)) / (w * (w - 2.0))

    mu_star = np.full((T, n), np.nan)
    Sigma_arr = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)

    ones = np.ones(n)
    skipped = {
        "minimum-variance mean undefined": 0,
        "psi^2 negative or not finite": 0,
        "adjusted psi^2 not positive": 0,
    }

    for t in range(start, T):
        ret_window = returns_df.iloc[t - w + 1 : t + 1]
        mu_t = ret_window.mean().to_numpy()

        # MLE covariance (c=w)
        Sigma_t = ret_window.cov(ddof=0).values

        target = _shrinkage_target(mu_t, Sigma_t, ones)
        if target is None:
            skipped["minimum-variance mean undefined"] += 1
            continue
        _, mu_g_vec, _, psi2 = target
        if not np.isfinite(psi2) or psi2 < 0.0:
            skipped["psi^2 negative or not finite"] += 1
            continue

        psi2_a = _kan_zhou_adjusted_psi2(psi2, n, w)
        if np.isnan(psi2_a):
            skipped["adjusted psi^2 not positive"] += 1
            continue

        v_t = n / (w * psi2_a + n)  # 1 when psi2_a = 0: mean = target
        mu_star[t] = (1.0 - v_t) * mu_t + v_t * mu_g_vec
        Sigma_arr[t] = Sigma_t
        v[t] = v_t

    warn_nan_rows("kan_zhou_three_fund", skipped, max(T - start, 0))
    return {
        "mu_star": mu_star,
        "Sigma": Sigma_arr,
        "sigma_star": Sigma_arr / h,
        "v": v,
        "h": np.array([h]),
    }


def kan_zhou_three_fund_forecast(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict:
    """Kan and Zhou (2007) three-fund forecast (Anderson & Cheng 2016, section 6.7).

      - mu_t is rolling sample mean over a window of size w
      - Sigma_t uses the MLE covariance with c=w (i.e. divide by w)
      - mu_g is the constant-vector target from the minimum-variance portfolio mean
      - v_t is Kan–Zhou's shrinkage weight (different from Jorion)
      - portfolio rule: (h/theta) * Sigma_t^{-1} * mu_t^*

    The forecast is (mu*, Sigma / h): Anderson & Cheng's footnote 17 and Kan &
    Zhou's eq. 75 read the rule as plain Markowitz on these moments,
    (1/theta) (Sigma/h)^{-1} mu* = (h/theta) Sigma^{-1} mu* (Sigma*/h, A&C
    fn. 17). So the forecast's mean is mu* itself, and the unconstrained and
    long-only weights and the scoring all use the same forecast.
    """
    est = kan_zhou_three_fund_estimates(returns_df, start=start, window=window)
    return make_predictive(est["mu_star"], est["sigma_star"], returns_df.index)
