import numpy as np
import pandas as pd
from markowitz import markowitz_unconstrained
from typing import Optional

"""only 3 functions here: 1/n, market weights and minimum variance strategy here returns only the weights invested. rest returns the predictives as well as the weights invested."""

# all functions below already return weight dataframes.
# this means we can directly implement portfolio returns > all dem thangs.
try:
    # Preferred: fast special functions
    from scipy.special import (
        betainc as _betainc,
    )  # regularized incomplete beta I_x(a,b)
    from scipy.special import beta as _beta  # Beta(a,b)
except Exception:  # pragma: no cover
    _betainc = None
    _beta = None
    try:
        import mpmath as _mp  # fallback (slower)
    except Exception as e:  # pragma: no cover
        _mp = None
        _mp_import_error = e


def _incomplete_beta_unregularized(a: float, b: float, x: float) -> float:
    """Return unregularized incomplete beta B_x(a,b) = \int_0^x t^{a-1} (1-t)^{b-1} dt."""
    if not (0.0 <= x <= 1.0) or not np.isfinite(x):
        return float("nan")

    if _betainc is not None and _beta is not None:
        # B_x(a,b) = I_x(a,b) * B(a,b)
        return float(_betainc(a, b, x) * _beta(a, b))

    # mpmath fallback (unregularized directly)
    if _mp is None:  # pragma: no cover
        raise ImportError("Need scipy or mpmath installed to compute incomplete beta.")
    return float(_mp.betainc(a, b, 0.0, x, regularized=False))


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
        return float("nan"), np.full_like(ones, np.nan, dtype=float), float("nan")
    mu_g_scalar = float(mu_t @ invSig_1) / denom_1
    mu_g_vec = mu_g_scalar * ones
    return mu_g_scalar, mu_g_vec, denom_1


def historical_expectations_weights(
    returns_df: pd.DataFrame,
    burn_in: int,
    periods_until_investment: int,
) -> pd.DataFrame:
    """Compute historical means and covariances across time."""
    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    for t in range(burn_in + periods_until_investment, T):
        ret_window = returns_df.iloc[t - burn_in - periods_until_investment : t]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values  # auto set c = w-1

    mu_sigma_dict = {"mu_hat": mu_hat, "sigma_hat": sigma_hat}
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, burn_in, periods_until_investment
    )

    return mu_sigma_dict, investment_weights


# returns the mu hat, sigma hat as well as the investment weights
def rolling_window_weights(
    returns_df: pd.DataFrame, window: int, burn_in: int, periods_until_investment: int
):

    T, n = returns_df.shape
    mu_hat = np.full((T, n), np.nan)
    sigma_hat = np.full((T, n, n), np.nan)

    start = max(window, burn_in + periods_until_investment)

    for t in range(start, T):
        ret_window = returns_df.iloc[t - window : t]
        mu_hat[t] = ret_window.mean().values
        sigma_hat[t] = ret_window.cov().values

    mu_sigma_dict = {"mu_hat": mu_hat, "sigma_hat": sigma_hat}
    investment_weights = markowitz_unconstrained(
        mu_sigma_dict, returns_df, burn_in, periods_until_investment
    )

    return mu_sigma_dict, investment_weights


def equal_weight_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
) -> pd.DataFrame:
    """Create 1/N weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)

    start = burn_in + periods_until_investment
    end = T - 1  # last row excluded

    if start < end:
        weights[start:end] = 1.0 / n

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )


def market_weight_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    market_col: str = "Mkt-RF",
) -> pd.DataFrame:
    """Create market weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)
    if market_col not in returns_df.columns:
        raise KeyError(f"Market column '{market_col}' not found in returns DataFrame.")

    start = burn_in + periods_until_investment
    end = T - 1  # last row excluded
    market_idx = returns_df.columns.get_loc(market_col)

    if start < end:
        weights[start:end, :] = 0.0
        weights[start:end, market_idx] = 1.0

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )


def minimum_variance_strategy(
    returns_df: pd.DataFrame,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    window: int = 250,
) -> pd.DataFrame:
    """Create minimum variance weights with burn-in and last-row exclusion."""
    T, n = returns_df.shape

    weights = np.full((T, n), np.nan)

    start = max(window, burn_in + periods_until_investment)
    end = T - 1  # last row excluded

    for t in range(start, end):
        ret_window = returns_df.iloc[t - window : t]
        sigma_hat_t = ret_window.cov().values
        inv_sigma = np.linalg.inv(sigma_hat_t)
        w = inv_sigma @ np.ones(n)
        w /= w.sum()  # normalize to sum to 1
        weights[t] = w

    return pd.DataFrame(
        weights,
        index=returns_df.index,
        columns=returns_df.columns,
    )


def jorion_bayes_stein_estimates(
    returns_df: pd.DataFrame,
    window: int = 250,
    burn_in: int = 0,
    periods_until_investment: int = 0,
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

    start = max(w, burn_in + periods_until_investment)
    end = T - 1  # last row excluded (since returns_{t+1} is needed downstream)

    mu_star = np.full((T, n), np.nan)
    sigma_star = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)
    J = np.full(T, np.nan)
    mu_g_scalar_arr = np.full(T, np.nan)

    ones = np.ones(n)
    c = w - n - 2

    for t in range(start, end):
        ret_window = returns_df.iloc[t - w : t]
        mu_t = ret_window.mean().values

        # pandas cov(ddof=1) gives (1/(w-1)) S; paper uses (1/c) S
        cov_ddof1 = ret_window.cov(ddof=1).values
        Sigma_t = cov_ddof1 * (w - 1) / c

        mu_g_scalar, mu_g_vec, denom_1 = _minvar_mean_target(mu_t, Sigma_t, ones)
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
    window: int = 250,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    theta: float = 1.0,
) -> pd.DataFrame:
    """Create Jorion's Bayes-Stein weights with burn-in and last-row exclusion."""
    est = jorion_bayes_stein_estimates(
        returns_df=returns_df,
        window=window,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    mu_sigma_dict = {"mu_hat": est["mu_star"], "sigma_hat": est["sigma_star"]}

    # Use existing pipeline to turn (mu*, Sigma*) into weights
    return mu_sigma_dict, markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )


def kan_zhou_three_fund_estimates(
    returns_df: pd.DataFrame,
    window: int = 250,
    burn_in: int = 0,
    periods_until_investment: int = 0,
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

    start = max(w, burn_in + periods_until_investment)
    end = T - 1

    h = ((w - n - 1.0) * (w - n - 4.0)) / (w * (w - 2.0))

    mu_star = np.full((T, n), np.nan)
    Sigma_arr = np.full((T, n, n), np.nan)
    v = np.full(T, np.nan)

    ones = np.ones(n)
    a = (n - 1.0) / 2.0
    b = (w + 1.0) / 2.0

    for t in range(start, end):
        ret_window = returns_df.iloc[t - w : t]
        mu_t = ret_window.mean().values

        # MLE covariance (c=w)
        Sigma_t = ret_window.cov(ddof=0).values

        mu_g_scalar, mu_g_vec, _denom_1 = _minvar_mean_target(mu_t, Sigma_t, ones)
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
    window: int = 250,
    burn_in: int = 0,
    periods_until_investment: int = 0,
    theta: float = 1.0,
):
    """Kan and Zhou (2007) three-fund rule.

    Implements Section 6.7 (as shown in your screenshot):
      - mu_t is rolling sample mean over a window of size w
      - Sigma_t uses the MLE covariance with c=w (i.e. divide by w)
      - mu_g is the constant-vector target from the minimum-variance portfolio mean
      - v_t is Kan–Zhou's shrinkage weight (different from Jorion)
      - portfolio rule: (h/theta) * Sigma_t^{-1} * mu_t^*

    Convention: weights at index t are applied to returns at index t+1 (handled elsewhere via shift(-1)).

    Returns:
      DataFrame of weights with same index/columns as `returns_df`.
    """
    est = kan_zhou_three_fund_estimates(
        returns_df=returns_df,
        window=window,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    h = float(est["h"][0])

    # Encode (h/theta) factor by scaling the mean: (h/theta) Sigma^{-1} mu_star
    # = (1/theta) Sigma^{-1} (h * mu_star).
    mu_tilde = h * est["mu_star"]
    sigma_hat = est["Sigma"]

    mu_sigma_dict = {"mu_hat": mu_tilde, "sigma_hat": sigma_hat}

    return mu_sigma_dict, markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
        theta=theta,
    )
