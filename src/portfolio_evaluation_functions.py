"""Portfolio performance metrics computed from weights and excess returns.

Inputs: weights_df (T, n) and returns_df (T, n) of excess returns. Weights at
t are decided with data up to t and earn the returns at t+1. Metrics are
daily unless ``scaling_factor`` is given (252 = annualised: Sharpe × √252,
turnover × 252). The pipeline passes ``config.SCALING_FACTOR``.

portfolio_returns       — return series implied by weights and returns.
calculate_sharpe_ratio  — scalar Sharpe, optionally annualised.
rolling_sharpe_ratio    — rolling Sharpe time series, optionally annualised.
certainty_equivalent    — mean-variance certainty equivalent (needs RF).
average_turnover        — mean daily sum of absolute weight changes, optionally annualised.
sharpe_difference_test  — GMM / Newey-West test of a Sharpe-ratio difference.
compare_to_benchmark    — sharpe_difference_test for several strategies vs one benchmark.
"""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    dropna: bool = True,
) -> pd.Series:
    """Portfolio return series: weights decided at t earn the returns at t+1.

    Returns are matched to the weights by index label and column name. The
    output is indexed by the return period (t+1); with ``dropna=True``
    periods without a weight are dropped.
    """
    returns_aligned = returns_df.loc[weights_df.index, weights_df.columns]
    # w_t moves to row t+1 so it multiplies r_{t+1}; any NaN propagates
    port = (weights_df.shift(1) * returns_aligned).sum(axis=1, skipna=False)
    return port.dropna() if dropna else port


def calculate_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    scaling_factor: int | None = None,
) -> float:
    """Sharpe ratio: mean / sample std of the portfolio excess returns.

    Daily; with ``scaling_factor`` (e.g. 252) multiplied by √scaling_factor.

    Undefined cases (fewer than 2 valid returns, or constant returns with
    std = 0) return NaN with a RuntimeWarning giving the reason.
    """
    port = portfolio_returns(returns_df, weights_df)

    if len(port) < 2:
        warnings.warn(
            f"sharpe: undefined with {len(port)} valid return(s); "
            "returning NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    std = float(port.std(ddof=1))
    if std == 0.0:
        warnings.warn(
            "sharpe: portfolio returns are constant (std = 0); returning NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    sharpe_daily = float(port.mean()) / std
    if not scaling_factor:
        return sharpe_daily
    return float(sharpe_daily * np.sqrt(scaling_factor))


def rolling_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    window: int,
    scaling_factor: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling Sharpe ratio, indexed like ``portfolio_returns``.

    Daily; with ``scaling_factor`` (e.g. 252) multiplied by √scaling_factor.

    NaN where the window has fewer than ``min_periods`` returns (warm-up,
    NaN weights; expected, no warning) or zero std (one RuntimeWarning
    with the count).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if min_periods is None:
        min_periods = window
    if not 1 <= min_periods <= window:
        raise ValueError(
            f"min_periods must be in [1, window={window}], got {min_periods}"
        )

    port = portfolio_returns(returns_df, weights_df, dropna=False)
    rolling = port.rolling(window=window, min_periods=min_periods)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=1)

    zero_std = rolling_std == 0
    if zero_std.any():
        warnings.warn(
            f"rolling_sharpe: {int(zero_std.sum())} of "
            f"{int(rolling_std.notna().sum())} windows have zero std; "
            "set to NaN",
            RuntimeWarning,
            stacklevel=2,
        )
    sharpe = rolling_mean / rolling_std.mask(zero_std)
    if scaling_factor:
        sharpe = sharpe * np.sqrt(scaling_factor)
    return sharpe


def certainty_equivalent(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    rf_series: pd.Series,
    theta: float = 1.0,
) -> float:
    """Certainty equivalent, Anderson & Cheng (2016) eq. (28):

        CE = mean(R_p,t+1) + mean(R_f,t+1) - (theta/2) * Var(R_p,t+1)

    with R_p the excess portfolio return, Var the sample variance (ddof=1)
    and R_f taken on the same return periods t+1 (matched by index label).
    """
    if theta < 0:
        raise ValueError(f"theta must be >= 0, got {theta}")

    port = portfolio_returns(returns_df, weights_df)
    mean_total = float(port.mean()) + float(rf_series.loc[port.index].mean())
    return mean_total - theta / 2.0 * float(port.var(ddof=1))


def average_turnover(
    weights_df: pd.DataFrame,
    scaling_factor: int | None = None,
) -> float:
    """Mean daily portfolio turnover.

    Turnover at t = sum_i |w_{i,t} - w_{i,t-1}|.
    Averaged over the periods where both w_t and w_{t-1} exist (rows before
    investment, the first invested row and the last NaN row are left out).
    Multiply by scaling_factor (e.g. 252) to annualise.
    """
    # skipna=False: an all-NaN row must stay NaN (not 0) so dropna removes it
    daily = weights_df.diff().abs().sum(axis=1, skipna=False).dropna()
    if daily.empty:
        return float("nan")
    to = float(daily.mean())
    if scaling_factor:
        to *= scaling_factor
    return to


class SharpeTestResult(NamedTuple):
    sharpe_strategy: float  # Sharpe of the strategy being tested
    sharpe_benchmark: float  # Sharpe of the benchmark
    difference: float  # strategy - benchmark
    t_statistic: float  # GMM t-stat on the difference
    outperforms: bool  # strategy significantly better than benchmark
    underperforms: bool  # strategy significantly worse than benchmark
    p_approx: float  # approximate two-sided p-value (normal)


def _newey_west_cov(moments: np.ndarray, n_lags: int) -> np.ndarray:
    """Newey-West HAC covariance estimator for a (T, k) matrix of moment conditions.

    moments : (T, k) array — each row is the vector of moment conditions at time t
    n_lags  : number of lags to include (paper uses 22)

    Returns the (k, k) long-run covariance matrix S such that
    Avar(sqrt(T) * mean(moments)) = S.
    """
    T = moments.shape[0]
    m = moments - moments.mean(axis=0)
    S = (m.T @ m) / T
    for lag in range(1, n_lags + 1):
        gamma = (m[lag:].T @ m[:-lag]) / T
        weight = 1.0 - lag / (n_lags + 1.0)  # Bartlett kernel
        S += weight * (gamma + gamma.T)
    return S


def sharpe_difference_test(
    returns_strategy: pd.Series,
    returns_benchmark: pd.Series,
    n_lags: int = 22,
    confidence: float = 0.95,
) -> SharpeTestResult:
    """Test whether a strategy's Sharpe ratio significantly differs from a benchmark.

    Implements the GMM approach from Anderson & Cheng (2016) Appendix B.1.
    The four moment conditions are:

        g1 = R_p   - s * sigma
        g2 = g1^2  - sigma^2
        g3 = R*_p  - (s - d) * sigma*
        g4 = g3^2  - sigma*^2

    where s  = Sharpe of strategy, d = difference (strategy - benchmark),
    sigma / sigma* are the respective standard deviations.

    Standard errors use a Newey-West HAC covariance with `n_lags` lags.
    The two series are matched on their common index labels.

    Parameters
    ----------
    returns_strategy  : excess portfolio return series for the strategy under test
    returns_benchmark : excess portfolio return series for the benchmark (e.g. robust BA)
    n_lags            : lags for Newey-West (paper uses 22)
    confidence        : confidence level for the significance flag (default 0.95)

    Returns
    -------
    SharpeTestResult namedtuple
    """
    idx = returns_strategy.index.intersection(returns_benchmark.index)
    r_s = returns_strategy.loc[idx].to_numpy(dtype=float)
    r_b = returns_benchmark.loc[idx].to_numpy(dtype=float)

    T = len(r_s)
    if T < max(n_lags + 2, 10):
        raise ValueError(
            f"Too few observations ({T}) for Newey-West with {n_lags} lags."
        )

    # GMM point estimates (exactly identified, closed form)
    sigma_s = float(np.std(r_s, ddof=1))
    sigma_b = float(np.std(r_b, ddof=1))
    s_hat = float(np.mean(r_s)) / sigma_s  # Sharpe of strategy
    sb_hat = float(np.mean(r_b)) / sigma_b  # Sharpe of benchmark (= s - d)
    d_hat = s_hat - sb_hat

    # (T, 4) moment conditions at the estimates
    g1 = r_s - s_hat * sigma_s
    g2 = g1**2 - sigma_s**2
    g3 = r_b - sb_hat * sigma_b
    g4 = g3**2 - sigma_b**2
    S = _newey_west_cov(np.column_stack([g1, g2, g3, g4]), n_lags)

    # Delta method. Linearising the four conditions and solving for d gives
    #   d_hat - d ≈ ḡ1/σ - s·ḡ2/(2σ²) - ḡ3/σ* + s*·ḡ4/(2σ*²),  s* = s - d
    # so Var(√T·d_hat) = J_d S J_d' with J_d the coefficients below.
    J_d = np.array(
        [
            1.0 / sigma_s,
            -s_hat / (2.0 * sigma_s**2),
            -1.0 / sigma_b,
            sb_hat / (2.0 * sigma_b**2),
        ]
    )
    avar_d = float(J_d @ S @ J_d)
    se_d = np.sqrt(max(avar_d, 0.0) / T)

    if se_d > 0:
        t_stat = d_hat / se_d
    elif d_hat == 0:
        t_stat = 0.0  # identical series: no difference (p = 1)
    else:
        t_stat = np.sign(d_hat) * np.inf

    critical = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    return SharpeTestResult(
        sharpe_strategy=s_hat,
        sharpe_benchmark=sb_hat,
        difference=d_hat,
        t_statistic=float(t_stat),
        outperforms=bool(t_stat > critical),
        underperforms=bool(t_stat < -critical),
        p_approx=float(2.0 * norm.sf(abs(t_stat))),
    )


def compare_to_benchmark(
    returns_df: pd.DataFrame,
    weights_benchmark: pd.DataFrame,
    weights_strategies: dict[str, pd.DataFrame],
    n_lags: int = 22,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare multiple strategies against a single benchmark using the Sharpe GMM test.

    Parameters
    ----------
    returns_df          : asset excess return DataFrame
    weights_benchmark   : weights of the benchmark strategy (e.g. robust BA)
    weights_strategies  : dict of {name: weights_df} for each strategy to test
    n_lags              : Newey-West lags (paper uses 22)
    confidence          : significance threshold

    Returns
    -------
    DataFrame indexed by strategy name, one column per ``SharpeTestResult``
    field (sharpe_strategy, sharpe_benchmark, difference, t_statistic,
    outperforms, underperforms, p_approx).
    """
    r_bench = portfolio_returns(returns_df, weights_benchmark)
    rows = {
        name: sharpe_difference_test(
            portfolio_returns(returns_df, w),
            r_bench,
            n_lags=n_lags,
            confidence=confidence,
        )._asdict()
        for name, w in weights_strategies.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")
