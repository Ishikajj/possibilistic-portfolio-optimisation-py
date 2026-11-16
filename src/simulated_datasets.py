"""Synthetic return data generators for controlled experiments.

All DGPs share a single-factor structure: r_it = B_i * g_t + eps_it, where g_t is a
common factor, B_i in [0.5, 1.5] are fixed loadings, and eps_it is i.i.d. idiosyncratic
noise. Asset excess returns have positive expected return proportional to B_i * E[g_t].
RF is simulated independently and never subtracted — outputs are already excess returns.

DGPs differ only in how g_t is generated:
  simulate_iid                     — g_t i.i.d. N(mu_bar, sigma_bar^2)
  simulate_regime_shifts           — g_t mean switches discretely with prob p_switch
  simulate_sudden_break            — g_t mean jumps permanently at break_fraction * T
  simulate_mean_reverting_factor   — g_t mean follows a discrete OU process
  simulate_sudden_covariance_break — g_t vol jumps permanently at break_fraction * T
  simulate_stochastic_volatility   — g_t vol follows a log-OU process

Every DGP has the signature ``(T, n_assets, seed, *, <its own settings>)``;
T / n_assets / seed default to config.SIM_T / SIM_N_ASSETS / SIM_SEED, and
the settings are keyword-only. All return (portfolios_df, rf_df),
integer-indexed, decimal units.

``DGPS`` maps each file label to its generator; ``sim_paths`` gives the CSV
paths and ``generate_and_save_sim_datasets`` writes them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from config import SIM_DATA_DIR, SIM_N_ASSETS, SIM_SEED, SIM_T

# -----------------
# Shared constants (annualised figures scaled to daily)
# -----------------

_MU_BAR = 0.08 / 252.0  # annualised factor mean 8%
_SIGMA_BAR = 0.16 / np.sqrt(252.0)  # annualised factor vol 16%
_MU_RF = 0.02 / 252.0  # annualised RF mean 2%
_SIGMA_RF = 0.02 / np.sqrt(252.0)  # annualised RF vol 2%

# -----------------
# Simulation helpers (no defaults: the public DGPs own them)
# -----------------


def _simulate_rf(rng: np.random.Generator, T: int) -> np.ndarray:
    """Daily RF with yearly mean 2% and yearly sd 2%. Returns decimals."""
    return rng.normal(loc=_MU_RF, scale=_SIGMA_RF, size=T)


def _simulate_factor_iid(
    rng: np.random.Generator, T: int, mu_bar: float, sigma_bar: float
) -> np.ndarray:
    """i.i.d. single factor g_t. Returns decimals."""
    return rng.normal(loc=mu_bar, scale=sigma_bar, size=T)


# this is weird - better use the ou reversion process.
# next draws are centered on previous day BUT with some pull back towards mean for containment.
# replicated from anderson cheng.
def _simulate_factor_regime_mean(
    rng: np.random.Generator,
    T: int,
    mu_bar: float,
    sigma_bar: float,
    p_switch: float,
    rho: float,
    sigma_mu: float,
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


def _simulate_factor_mean_break(
    rng: np.random.Generator,
    T: int,
    sigma_bar: float,
    break_fraction: float,
    mu_pre: float,
    mu_post: float,
) -> np.ndarray:
    """Factor mean jumps from mu_pre to mu_post at break_fraction * T."""
    break_t = int(T * break_fraction)
    mu_t = np.where(np.arange(T) < break_t, mu_pre, mu_post)
    return rng.normal(loc=mu_t, scale=sigma_bar, size=T)


def _simulate_factor_mean_reverting(
    rng: np.random.Generator,
    T: int,
    mu_bar: float,
    sigma_bar: float,
    theta_mu: float,
    sigma_mu: float,
) -> np.ndarray:
    """Factor mean follows mu_{t+1} = mu_t + theta_mu (mu_bar - mu_t) + sigma_mu eps_t;
    g_t ~ N(mu_t, sigma_g^2) with total Var(g_t) ≈ sigma_bar^2.

    Raises if the mean's steady-state variance alone reaches sigma_bar^2
    (theta_mu / sigma_mu incompatible with sigma_bar), like the regime DGP.
    """
    steady_state_var_mu = sigma_mu**2 / (2.0 * theta_mu - theta_mu**2)
    var_g = sigma_bar**2 - steady_state_var_mu
    if var_g <= 0.0:
        raise ValueError(
            "Invalid parameters: sigma_bar^2 - sigma_mu^2/(2 theta_mu - "
            f"theta_mu^2) must be > 0. Got sigma_bar^2={sigma_bar**2:.6g}, "
            f"steady-state Var(mu)={steady_state_var_mu:.6g}."
        )
    sigma_g = float(np.sqrt(var_g))

    mu_t = np.empty(T, dtype=float)
    mu_t[0] = mu_bar
    noise = rng.normal(0.0, sigma_mu, size=T - 1)
    for t in range(T - 1):
        mu_t[t + 1] = mu_t[t] + theta_mu * (mu_bar - mu_t[t]) + noise[t]
    return rng.normal(loc=mu_t, scale=sigma_g, size=T)


def _simulate_factor_vol_break(
    rng: np.random.Generator,
    T: int,
    mu_bar: float,
    break_fraction: float,
    sigma_pre: float,
    sigma_post: float,
) -> np.ndarray:
    """Factor with constant mean but a permanent jump in volatility at break_fraction * T."""
    break_t = int(T * break_fraction)
    sigma_t = np.where(np.arange(T) < break_t, sigma_pre, sigma_post)
    return rng.normal(loc=mu_bar, scale=sigma_t, size=T)


def _simulate_factor_stochastic_vol(
    rng: np.random.Generator,
    T: int,
    mu_bar: float,
    sigma_bar: float,
    theta_v: float,
    xi: float,
) -> np.ndarray:
    """Factor with constant mean and log-OU stochastic variance around sigma_bar^2.

    log(v_{t+1}) = log(v_t) + theta_v * (log(v_bar) - log(v_t)) + xi * eps_t
    g_t ~ N(mu_bar, v_t),  v_bar = sigma_bar^2
    """
    v_bar = sigma_bar**2
    log_v = np.empty(T, dtype=float)
    log_v[0] = np.log(v_bar)
    log_v_bar = np.log(v_bar)
    noise = rng.normal(0.0, xi, size=T - 1)
    for t in range(T - 1):
        log_v[t + 1] = log_v[t] + theta_v * (log_v_bar - log_v[t]) + noise[t]
    sigma_t = np.sqrt(np.exp(log_v))
    return rng.normal(loc=mu_bar, scale=sigma_t, size=T)


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


def _simulate(
    T: int,
    n_assets: int,
    seed: int,
    factor: Callable[..., np.ndarray],
    **factor_params: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared body of every DGP. Draw order (fixed, keeps saved data
    reproducible): RF, then the factor, then the idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    rf = _simulate_rf(rng, T)
    g = factor(rng, T, **factor_params)
    excess, _ = _simulate_factor_model_excess_returns(rng, g, n_assets)
    asset_cols = [f"Asset{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(excess, columns=asset_cols), pd.DataFrame({"RF": rf})


# -----------------
# Public DGPs
# -----------------


def simulate_iid(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    mu_bar: float = _MU_BAR,
    sigma_bar: float = _SIGMA_BAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """i.i.d. single-factor model (Appendix A, DGP 1)."""
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_iid,
        mu_bar=mu_bar,
        sigma_bar=sigma_bar,
    )


def simulate_regime_shifts(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    mu_bar: float = _MU_BAR,
    sigma_bar: float = _SIGMA_BAR,
    p_switch: float = 0.5,
    rho: float = 0.95,
    sigma_mu: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor mean undergoes discrete regime changes (Appendix A, DGP 2).

    With probability p_switch the mean jumps to N(mu_bar + rho*(mu_t - mu_bar), sigma_mu^2);
    otherwise it stays constant.
    """
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_regime_mean,
        mu_bar=mu_bar,
        sigma_bar=sigma_bar,
        p_switch=p_switch,
        rho=rho,
        sigma_mu=sigma_mu,
    )


def simulate_sudden_break(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    sigma_bar: float = _SIGMA_BAR,
    break_fraction: float = 0.5,
    mu_pre: float = _MU_BAR,
    mu_post: float = -0.04 / 252.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single permanent structural break in factor mean at break_fraction * T.

    Pre-break: positive drift.  Post-break: negative drift.
    The break is abrupt and permanent — models that span the break are actively
    harmful, so strategies that quickly down-weight stale models should win.
    """
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_mean_break,
        sigma_bar=sigma_bar,
        break_fraction=break_fraction,
        mu_pre=mu_pre,
        mu_post=mu_post,
    )


def simulate_mean_reverting_factor(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    mu_bar: float = _MU_BAR,
    sigma_bar: float = _SIGMA_BAR,
    theta_mu: float = 0.05,
    sigma_mu: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor mean follows a discrete-time OU process (fast mean reversion).

    mu_{t+1} = mu_t + theta_mu * (mu_bar - mu_t) + sigma_mu * eps_t

    With theta_mu=0.05, the half-life is ~14 trading days.  Short-window models
    consistently out-predict long-window models because recent data is far more
    informative about the current factor mean.  Tests whether model averaging
    concentrates weight on fresh models.
    """
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_mean_reverting,
        mu_bar=mu_bar,
        sigma_bar=sigma_bar,
        theta_mu=theta_mu,
        sigma_mu=sigma_mu,
    )


def simulate_sudden_covariance_break(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    mu_bar: float = _MU_BAR,
    break_fraction: float = 0.5,
    sigma_pre: float = 0.10 / np.sqrt(252.0),
    sigma_post: float = 0.25 / np.sqrt(252.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permanent jump in factor volatility at break_fraction * T; mean is constant.

    Pre-break annualised factor vol: ~10%.  Post-break: ~25%.
    Sigma_t = B B' sigma_g_t^2 + Diag(sigma_idio^2) jumps at the break.
    Mirrors DGP 3 (sudden_break) but in covariance rather than mean.
    """
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_vol_break,
        mu_bar=mu_bar,
        break_fraction=break_fraction,
        sigma_pre=sigma_pre,
        sigma_post=sigma_post,
    )


def simulate_stochastic_volatility(
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    *,
    mu_bar: float = _MU_BAR,
    sigma_bar: float = _SIGMA_BAR,
    theta_v: float = 0.05,
    xi: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constant factor mean with log-OU stochastic volatility around sigma_bar.

    log(v_{t+1}) = log(v_t) + theta_v*(log(sigma_bar^2) - log(v_t)) + xi*eps_t

    With theta_v=0.05, half-life ~14 trading days; with xi=0.2, annualised
    vol ranges roughly [8%, 30%].  Covariance varies continuously; tests
    whether model averaging tracks volatility clustering.
    Mirrors DGP 4 (mean_reverting) but in covariance rather than mean.
    """
    return _simulate(
        T,
        n_assets,
        seed,
        _simulate_factor_stochastic_vol,
        mu_bar=mu_bar,
        sigma_bar=sigma_bar,
        theta_v=theta_v,
        xi=xi,
    )


# -----------------
# Registry and saved datasets
# -----------------

# File label -> generator. The labels name the saved CSVs and the pipeline's
# result folders.
DGPS: dict[str, Callable[..., tuple[pd.DataFrame, pd.DataFrame]]] = {
    "iid": simulate_iid,
    "reg": simulate_regime_shifts,
    "break": simulate_sudden_break,
    "mr": simulate_mean_reverting_factor,
    "cov_break": simulate_sudden_covariance_break,
    "stoch_vol": simulate_stochastic_volatility,
}


def sim_paths(
    label: str,
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    data_dir: Path | None = None,
) -> tuple[Path, Path]:
    """(returns CSV, RF CSV) for a saved dataset:
    ``<label>_T<T>_N<n>_seed<seed>.csv`` and ``rf_<same>``."""
    tag = f"{label}_T{T}_N{n_assets}_seed{seed}.csv"
    data_dir = SIM_DATA_DIR if data_dir is None else Path(data_dir)
    return data_dir / tag, data_dir / f"rf_{tag}"


def generate_and_save_sim_datasets(
    labels: list[str] | None = None,
    T: int = SIM_T,
    n_assets: int = SIM_N_ASSETS,
    seed: int = SIM_SEED,
    data_dir: Path | None = None,
) -> None:
    """Generate the DGPs in ``labels`` (``None`` = all of DGPS) and save
    returns and RF as CSVs (decimals, no index) at ``sim_paths``."""
    for label in DGPS if labels is None else labels:
        ports_path, rf_path = sim_paths(label, T, n_assets, seed, data_dir)
        ports_path.parent.mkdir(parents=True, exist_ok=True)
        portfolios_df, rf_df = DGPS[label](T=T, n_assets=n_assets, seed=seed)
        portfolios_df.to_csv(ports_path, index=False)
        rf_df.to_csv(rf_path, index=False)
        print(f"  saved {ports_path.name}, {rf_path.name}")
