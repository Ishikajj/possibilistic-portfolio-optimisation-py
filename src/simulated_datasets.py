"""
Data input utilities for the Robust Bayesian Portfolio Choices replication.

Conventions:
- Source CSVs often report returns in PERCENT units (e.g., 1.0 means 1%).
  We convert returns to DECIMALS (divide by 100).
- We keep a datetime `Date` column for slicing.
- We also provide `time_index_to_integer()` to convert a Date-indexed frame to
  integer time indexing while preserving the Date in a column.

  Note that before 1953 there were about 300 trading days per year. we choose to focus our performance strictly post 1963 as per the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from io import StringIO
import numpy as np
import pandas as pd

# -----------------
# Simulation helpers
# -----------------


def prepare_sim_returns(portfolios_df: pd.DataFrame, n_assets: int = 5) -> pd.DataFrame:
    """Prepare a simulated portfolios DataFrame for strategy runners.

    Sets the Date column as the index and keeps the first n_assets asset
    columns, returning a date-indexed DataFrame of excess returns in decimals.
    """
    df = portfolios_df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    asset_cols = [c for c in df.columns if c.startswith("Asset")][:n_assets]
    return df[asset_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")


def simulate_sudden_break(
    T: int = 252 * 20,
    start_date: str = "2000-01-03",
    n_assets: int = 5,
    seed: int = 42,
    break_fraction: float = 0.5,
    mu_pre: float = 0.08 / 252.0,
    mu_post: float = -0.04 / 252.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single permanent structural break in factor mean at break_fraction * T.

    Pre-break: positive drift.  Post-break: negative drift.
    The break is abrupt and permanent — models that span the break are actively
    harmful, so strategies that quickly down-weight stale models should win.
    """
    rng = np.random.default_rng(seed)
    dates = _make_trading_dates(start_date, T)
    RF = _simulate_rf(rng, T)

    break_t = int(T * break_fraction)
    mu_t = np.where(np.arange(T) < break_t, mu_pre, mu_post)
    sigma_g = 0.16 / np.sqrt(252.0)
    g = rng.normal(loc=mu_t, scale=sigma_g, size=T)

    excess, _ = _simulate_factor_model_excess_returns(rng, g, n_assets)
    return _pack_simulation_frames(dates, excess, RF, g)


def simulate_mean_reverting_factor(
    T: int = 252 * 20,
    start_date: str = "2000-01-03",
    n_assets: int = 5,
    seed: int = 42,
    theta: float = 0.05,
    mu_bar: float = 0.08 / 252.0,
    sigma_mu: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor mean follows a discrete-time OU process (fast mean reversion).

    mu_{t+1} = mu_t + theta * (mu_bar - mu_t) + sigma_mu * eps_t

    With theta=0.05, the half-life is ~14 trading days.  Short-window models
    consistently out-predict long-window models because recent data is far more
    informative about the current factor mean.  Tests whether model averaging
    concentrates weight on fresh models.
    """
    rng = np.random.default_rng(seed)
    dates = _make_trading_dates(start_date, T)
    RF = _simulate_rf(rng, T)

    mu_t = np.empty(T, dtype=float)
    mu_t[0] = mu_bar
    noise = rng.normal(0.0, sigma_mu, size=T - 1)
    for t in range(T - 1):
        mu_t[t + 1] = mu_t[t] + theta * (mu_bar - mu_t[t]) + noise[t]

    # Keep total Var(g_t) ≈ sigma_bar^2 = (0.16/sqrt(252))^2
    sigma_bar = 0.16 / np.sqrt(252.0)
    steady_state_var_mu = sigma_mu**2 / (2.0 * theta - theta**2)
    sigma_g = float(np.sqrt(max(sigma_bar**2 - steady_state_var_mu, 1e-8)))
    g = rng.normal(loc=mu_t, scale=sigma_g, size=T)

    excess, _ = _simulate_factor_model_excess_returns(rng, g, n_assets)
    return _pack_simulation_frames(dates, excess, RF, g)


def _make_trading_dates(start_date: str, T: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start_date, periods=T)


def _simulate_rf(rng: np.random.Generator, T: int) -> np.ndarray:
    """Daily RF with yearly mean 2% and yearly sd 2%. Returns decimals."""
    mu_rf = 0.02 / 252.0
    sd_rf = 0.02 / np.sqrt(252.0)
    return rng.normal(loc=mu_rf, scale=sd_rf, size=T)


def _simulate_factor_iid(
    rng: np.random.Generator,
    T: int,
    mu_bar: float = 0.08 / 252.0,
    sigma_bar: float = 0.16 / np.sqrt(252.0),
) -> np.ndarray:
    """i.i.d. single factor g_t. Returns decimals."""
    return rng.normal(loc=mu_bar, scale=sigma_bar, size=T)


def _simulate_factor_regime_mean(
    rng: np.random.Generator,
    T: int,
    p_switch: float,
    rho: float,
    sigma_mu: float,
    mu_bar: float = 0.08 / 252.0,
    sigma_bar: float = 0.16 / np.sqrt(252.0),
) -> np.ndarray:
    """Single factor with regime changes in its mean (Appendix A, DGP 2).

    mu_{t+1} equals mu_t with prob 1-p_switch. If switch, mu_{t+1} ~ N(mu_bar + rho(mu_t-mu_bar), sigma_mu^2).
    g_t ~ N(mu_t, sigma_g^2) with sigma_g chosen so Var(g_t) matches the i.i.d. case.
    """
    # sigma_g = sqrt(sigma_bar^2 - sigma_mu^2/(1-rho^2))
    adj = (sigma_mu**2) / (1.0 - rho**2)
    var_g = sigma_bar**2 - adj
    if var_g <= 0.0:
        raise ValueError(
            "Invalid parameters: sigma_bar^2 - sigma_mu^2/(1-rho^2) must be > 0. "
            f"Got sigma_bar^2={sigma_bar**2:.6g}, adj={adj:.6g}."
        )
    sigma_g = float(np.sqrt(var_g))

    mu_t = np.empty(T, dtype=float)
    mu_t[0] = mu_bar

    switches = rng.random(T - 1) < p_switch
    for t in range(T - 1):
        if switches[t]:
            mu_t[t + 1] = rng.normal(
                loc=mu_bar + rho * (mu_t[t] - mu_bar),
                scale=sigma_mu,
            )
        else:
            mu_t[t + 1] = mu_t[t]

    return rng.normal(loc=mu_t, scale=sigma_g, size=T)


def _simulate_factor_model_excess_returns(
    rng: np.random.Generator,
    g: np.ndarray,
    n_assets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate excess returns via r^ex_t = B g_t + eps_t, eps diagonal.

    - B evenly spaced in [0.5, 1.5]
    - yearly idiosyncratic variances ~ Unif(0.1, 0.3), converted to daily by /252

    Returns:
      excess: (T, n_assets)
      B: (n_assets,)
    """
    T = int(g.shape[0])
    B = np.linspace(0.5, 1.5, n_assets)

    yearly_var = rng.uniform(0.1, 0.3, size=n_assets)
    sigma2_daily = yearly_var / 252.0
    eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2_daily), size=(T, n_assets))

    excess = g[:, None] * B[None, :] + eps
    return excess, B


def _pack_simulation_frames(
    dates: pd.DatetimeIndex,
    excess: np.ndarray,
    RF: np.ndarray,
    g: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (portfolios_df, rf_df) in the same schema as the data loaders."""
    n_assets = excess.shape[1]
    asset_cols = [f"Asset{i+1}" for i in range(n_assets)]

    portfolios_df = pd.DataFrame(excess, columns=asset_cols)
    portfolios_df.insert(0, "Date", dates)

    rf_df = pd.DataFrame({"Date": dates, "RF": RF})
    return portfolios_df, rf_df


def simulate_rbpc_data(
    T: int = 252 * 20,  # ~20 years of daily data
    start_date: str = "2000-01-03",
    n_assets: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate RBPC-style i.i.d. data (Appendix A, DGP 1).

    Returns:
      portfolios_df: Date + Asset1..AssetN (EXCESS returns, decimals)
      rf_df: Date + RF + Mkt-RF (decimals), where Mkt-RF is set to the factor g_t

    Note: `portfolios_df` contains excess returns already. Do NOT call
    `calculate_excess_returns` on it unless you first convert it to total returns.
    """
    rng = np.random.default_rng(seed)

    dates = _make_trading_dates(start_date, T)
    RF = _simulate_rf(rng, T)
    g = _simulate_factor_iid(rng, T)

    excess, _B = _simulate_factor_model_excess_returns(rng, g, n_assets)

    return _pack_simulation_frames(dates, excess, RF, g)


def simulate_rbpc_data_regime_shifts(
    T: int = 252 * 20,  # ~20 years of daily data
    start_date: str = "2000-01-03",
    n_assets: int = 10,
    seed: int = 0,
    p_switch: float = 0.5,
    rho: float = 0.95,
    sigma_mu: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate RBPC-style data with regime changes in the factor mean.

    Matches the paper's second DGP (Appendix A):
    """
    rng = np.random.default_rng(seed)

    dates = _make_trading_dates(start_date, T)
    RF = _simulate_rf(rng, T)

    g = _simulate_factor_regime_mean(
        rng=rng,
        T=T,
        p_switch=p_switch,
        rho=rho,
        sigma_mu=sigma_mu,
    )

    excess, _B = _simulate_factor_model_excess_returns(rng, g, n_assets)

    return _pack_simulation_frames(dates, excess, RF, g)
