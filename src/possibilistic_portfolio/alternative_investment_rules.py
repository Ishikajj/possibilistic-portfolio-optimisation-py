"""Benchmark and classical portfolio strategies for comparison against Bayesian/possibilistic methods.

Seven strategies split into two groups:

Weights only (pd.DataFrame):
  equal_weight_strategy          — 1/N uniform allocation
  market_weight_strategy         — full weight on the market column, zero elsewhere
  minimum_variance_strategy      — rolling sample covariance inversion, no mean input

Predictive moments + weights (tuple[mu_sigma_dict, pd.DataFrame]):
  historical_expectations_weights — expanding-window sample mean/covariance → Markowitz
  rolling_window_weights          — fixed rolling window → Markowitz
  jorion_bayes_stein_strategy     — Jorion (1986) shrinkage toward min-variance mean
  kan_zhou_three_fund_strategy    — Kan & Zhou (2007) three-fund rule with incomplete-beta shrinkage

Signature of every strategy: (returns_df, *, start, window, theta) — all
keyword-only after returns_df (only the ones a strategy uses). `start`, the
first row with weights, is always passed (the pipeline computes
burn_in + periods_until_investment once); window / theta default to
config.ROLLING_WINDOW_LONG / config.THETA. Weights are NaN before start and on
the last row (weights at t apply to returns at t+1; timing convention: predictive.py).
Estimates at t use the w returns up to and including t: iloc[t-w+1 : t+1].
mu_sigma_dict: predictive.make_predictive dict ("mu_hat", "sigma_hat", "index" = returns_df.index).
"""

import numpy as np
import pandas as pd
from scipy.special import beta as _beta  # Beta(a,b)
from scipy.special import (
    betainc as _betainc,
)  # regularized incomplete beta I_x(a,b)

from possibilistic_portfolio.config import ROLLING_WINDOW_LONG, THETA
from possibilistic_portfolio.markowitz import markowitz_unconstrained
from possibilistic_portfolio.predictive import StrategyOutput, make_predictive


def _incomplete_beta_unregularized(a: float, b: float, x: float) -> float:
    r"""Return unregularized incomplete beta B_x(a,b) = \int_0^x t^{a-1} (1-t)^{b-1} dt."""
    if not (0.0 <= x <= 1.0) or not np.isfinite(x):
        return float("nan")

    # B_x(a,b) = I_x(a,b) * B(a,b)
    return float(_betainc(a, b, x) * _beta(a, b))


def _solve_or_pinv(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return x solving A x = b; fall back to pinv(A) b if singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


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
    invSig_1 = _solve_or_pinv(Sigma_t, ones)
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


def historical_expectations_weights(
    returns_df: pd.DataFrame,
    *,
    start: int,
    theta: float = THETA,
) -> StrategyOutput:
    """Sample mean and covariance of all returns up to and including t
    (expanding window from row 0), from `start` on."""
    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(start, T):
        ret_window = returns_df.iloc[: t + 1]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values  # auto set c = w-1

    mu_sigma_dict = make_predictive(mu_hat, sigma_hat, returns_df.index)
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, start, theta=theta
    )

    return StrategyOutput(investment_weights, mu_sigma_dict)


def rolling_window_weights(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
    theta: float = THETA,
) -> StrategyOutput:
    """Sample mean and covariance of the last `window` returns up to and
    including t, from `start` (or `window`, if later)."""

    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(max(window, start), T):
        ret_window = returns_df.iloc[t - window + 1 : t + 1]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values

    mu_sigma_dict = make_predictive(mu_hat, sigma_hat, returns_df.index)
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, start, theta=theta
    )

    return StrategyOutput(investment_weights, mu_sigma_dict)


def equal_weight_strategy(
    returns_df: pd.DataFrame,
    *,
    start: int,
) -> StrategyOutput:
    """Create 1/N weights from `start`, last row excluded."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)
    end = T - 1  # last row excluded

    if start < end:
        weights[start:end] = 1.0 / n

    return StrategyOutput(
        pd.DataFrame(
            weights, index=returns_df.index, columns=returns_df.columns
        )
    )


def market_weight_strategy(
    returns_df: pd.DataFrame,
    *,
    start: int,
    market_col: str = "Mkt-RF",
) -> StrategyOutput:
    """Create market weights from `start`, last row excluded."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)
    if market_col not in returns_df.columns:
        raise KeyError(
            f"Market column '{market_col}' not found in returns DataFrame."
        )

    end = T - 1  # last row excluded
    market_idx = returns_df.columns.get_loc(market_col)

    if start < end:
        weights[start:end, :] = 0.0
        weights[start:end, market_idx] = 1.0

    return StrategyOutput(
        pd.DataFrame(
            weights, index=returns_df.index, columns=returns_df.columns
        )
    )


def minimum_variance_strategy(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> StrategyOutput:
    """Create minimum variance weights from `start` (or `window`, if later)."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)
    end = T - 1  # last row excluded

    for t in range(max(window, start), end):
        ret_window = returns_df.iloc[t - window + 1 : t + 1]
        sigma_hat_t = ret_window.cov().values
        inv_sigma = np.linalg.inv(sigma_hat_t)
        w = inv_sigma @ np.ones(n)
        w /= w.sum()  # normalize to sum to 1
        weights[t] = w

    return StrategyOutput(
        pd.DataFrame(
            weights, index=returns_df.index, columns=returns_df.columns
        )
    )


def jorion_bayes_stein_estimates(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict[str, np.ndarray]:
    """Compute Jorion (1986) Bayes–Stein shrunk mean/covariance.

    Returns arrays aligned to `returns_df.index`:
      - mu_star: (T,n)
      - sigma_star: (T,n,n)
      - v: (T,) shrinkage intensity
      - J: (T,) auxiliary scalar
      - mu_g_scalar: (T,) minimum-variance mean target (scalar)
    """
    assert isinstance(returns_df, pd.DataFrame)

    T, n = returns_df.shape
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    if w <= n + 2:
        raise ValueError(
            f"window must be > n+2 for c=w-n-2 to be positive; got window={w}, n={n}"
        )

    start = max(w, start)
    end = T - 1  # last row excluded (since returns_{t+1} is needed downstream)

    mu_star = np.full((T, n), np.nan)
    sigma_star = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)
    J = np.full(T, np.nan)
    mu_g_scalar_arr = np.full(T, np.nan)

    ones = np.ones(n)
    c = w - n - 2

    for t in range(start, end):
        ret_window = returns_df.iloc[t - w + 1 : t + 1]
        mu_t = ret_window.mean().to_numpy()

        # pandas cov(ddof=1) gives (1/(w-1)) S; paper uses (1/c) S
        cov_ddof1 = ret_window.cov(ddof=1).values
        Sigma_t = cov_ddof1 * (w - 1) / c

        mu_g_scalar, mu_g_vec, denom_1 = _minvar_mean_target(
            mu_t, Sigma_t, ones
        )
        if not np.isfinite(mu_g_scalar) or not np.isfinite(denom_1):
            continue
        mu_g_scalar_arr[t] = mu_g_scalar

        d = mu_t - mu_g_vec
        invSig_d = _solve_or_pinv(Sigma_t, d)
        q = float(d @ invSig_d)
        if q <= 0.0 or not np.isfinite(q):
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

    return {
        "mu_star": mu_star,
        "sigma_star": sigma_star,
        "v": v,
        "J": J,
        "mu_g_scalar": mu_g_scalar_arr,
    }


def jorion_bayes_stein_strategy(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
    theta: float = THETA,
) -> StrategyOutput:
    """Create Jorion's Bayes-Stein weights with burn-in and last-row exclusion."""
    est = jorion_bayes_stein_estimates(
        returns_df=returns_df,
        window=window,
        start=start,
    )

    mu_sigma_dict = make_predictive(
        est["mu_star"], est["sigma_star"], returns_df.index
    )

    # Use existing pipeline to turn (mu*, Sigma*) into weights
    weights = markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        start=start,
        theta=theta,
    )
    return StrategyOutput(weights, mu_sigma_dict)


def kan_zhou_three_fund_estimates(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
) -> dict[str, np.ndarray]:
    """Compute Kan and Zhou (2007) three-fund shrinkage estimates.

    Returns arrays aligned to `returns_df.index`:
      - mu_star: (T,n) shrunk mean
      - Sigma: (T,n,n) rolling MLE covariance (c=w)
      - v: (T,) shrinkage weight
      - h: scalar constant used in the portfolio rule
    """
    assert isinstance(returns_df, pd.DataFrame)

    T, n = returns_df.shape
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    if w <= 2:
        raise ValueError("window must be > 2")
    if w <= n + 4:
        raise ValueError(
            f"window must be > n+4 for h to be well-defined; got window={w}, n={n}"
        )

    start = max(w, start)
    end = T - 1

    h = ((w - n - 1.0) * (w - n - 4.0)) / (w * (w - 2.0))

    mu_star = np.full((T, n), np.nan)
    Sigma_arr = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)

    ones = np.ones(n)
    a = (n - 1.0) / 2.0
    b = (w + 1.0) / 2.0

    for t in range(start, end):
        ret_window = returns_df.iloc[t - w + 1 : t + 1]
        mu_t = ret_window.mean().to_numpy()

        # MLE covariance (c=w)
        Sigma_t = ret_window.cov(ddof=0).values

        mu_g_scalar, mu_g_vec, _denom_1 = _minvar_mean_target(
            mu_t, Sigma_t, ones
        )
        if not np.isfinite(mu_g_scalar):
            continue

        d = mu_t - mu_g_vec
        invSig_d = _solve_or_pinv(Sigma_t, d)
        psi_t = float(d @ invSig_d)
        if not np.isfinite(psi_t) or psi_t < 0.0:
            continue

        # If psi is ~0, mu_t already equals the target.
        if psi_t <= 1e-12:
            mu_star[t] = mu_g_vec
            Sigma_arr[t] = Sigma_t
            v[t] = 1.0
            continue

        x_t = float(psi_t / (1.0 + psi_t))
        B_t = _incomplete_beta_unregularized(a=a, b=b, x=x_t)
        if not np.isfinite(B_t) or B_t <= 0.0:
            continue

        term1 = ((w - n - 1.0) * psi_t - (n - 1.0)) / w
        term2_num = 2.0 * (psi_t**a) * ((1.0 + psi_t) ** (-b))
        term2 = (1.0 / (w * B_t)) * term2_num
        epsilon_t = term1 + term2
        if not np.isfinite(epsilon_t):
            continue

        v_t = n / (w * epsilon_t + n)
        mu_t_star = (1.0 - v_t) * mu_t + v_t * mu_g_vec

        mu_star[t] = mu_t_star
        Sigma_arr[t] = Sigma_t
        v[t] = v_t

    return {"mu_star": mu_star, "Sigma": Sigma_arr, "v": v, "h": np.array([h])}


def kan_zhou_three_fund_strategy(
    returns_df: pd.DataFrame,
    *,
    start: int,
    window: int = ROLLING_WINDOW_LONG,
    theta: float = THETA,
) -> StrategyOutput:
    """Kan and Zhou (2007) three-fund rule.

    Implements Section 6.7:
      - mu_t is rolling sample mean over a window of size w
      - Sigma_t uses the MLE covariance with c=w (i.e. divide by w)
      - mu_g is the constant-vector target from the minimum-variance portfolio mean
      - v_t is Kan–Zhou's shrinkage weight (different from Jorion)
      - portfolio rule: (h/theta) * Sigma_t^{-1} * mu_t^*

    Convention: weights at index t are applied to returns at index t+1 (handled elsewhere via shift(-1)).

    Returns:
      StrategyOutput: weights (same index/columns as `returns_df`) and the
      forecast (mu = h * mu_star, see below).
    """
    est = kan_zhou_three_fund_estimates(
        returns_df=returns_df,
        window=window,
        start=start,
    )

    h = float(est["h"][0])

    # Encode (h/theta) factor by scaling the mean: (h/theta) Sigma^{-1} mu_star
    # = (1/theta) Sigma^{-1} (h * mu_star).
    mu_tilde = h * est["mu_star"]
    sigma_hat = est["Sigma"]

    mu_sigma_dict = make_predictive(mu_tilde, sigma_hat, returns_df.index)

    weights = markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        start=start,
        theta=theta,
    )
    return StrategyOutput(weights, mu_sigma_dict)
