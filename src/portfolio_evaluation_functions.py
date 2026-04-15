# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.
import numpy as np
import pandas as pd
from typing import Optional, NamedTuple

# given a set of weights, excess returns over the market for each said asset, we calculate the sharpe
# the sharpe ratios are calculated daily.
# use non annualised and then annualise it.

# basic assumption is that weights at index t are applied to returns at index t+1.


def portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    lag: int = 1,
    dropna: bool = True,
) -> pd.Series:
    """Compute the portfolio return series implied by weights and asset returns.

    Convention: weights at index t are applied to returns at index t+lag.

    Returns:
        pd.Series of portfolio returns inno dexed like `weights_df`.
    """
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(lag, int) and lag >= 0

    # Align to weights index/columns (enforces same asset set and order)
    returns_aligned = returns_df.loc[weights_df.index, weights_df.columns]

    # Shift weights so w_{t-1} multiplies r_t
    shifted_weights = weights_df.shift(lag)

    # Elementwise multiply and sum across assets.
    # Use skipna=False so any NaN propagates.
    port = (shifted_weights * returns_aligned).sum(axis=1, skipna=False)

    if dropna:
        port = port.dropna()

    return port


def calculate_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    scaling_factor: Optional[int] = None,
) -> float:
    """Compute daily Sharpe ratio from excess returns and weights."""
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)

    portfolio = portfolio_returns(returns_df, weights_df, lag=1, dropna=True)

    mean_ret: float = float(portfolio.mean())
    std_ret: float = float(portfolio.std(ddof=1))

    if std_ret == 0.0:
        return 0.0

    sharpe_daily: float = mean_ret / std_ret

    if not scaling_factor:
        return float(sharpe_daily)

    sharpe_annualised: float = sharpe_daily * np.sqrt(scaling_factor)

    return float(sharpe_annualised)


def rolling_sharpe_ratio(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    window: int,
    scaling_factor: Optional[int] = None,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute a rolling Sharpe ratio series from (excess) returns and weights.

    Convention: weights at index t are applied to returns at index t+1.

    """
    assert isinstance(weights_df, pd.DataFrame)
    assert isinstance(returns_df, pd.DataFrame)
    assert isinstance(window, int) and window > 0

    if min_periods is None:
        min_periods = window
    assert isinstance(min_periods, int) and 1 <= min_periods <= window

    portfolio = portfolio_returns(returns_df, weights_df, lag=1, dropna=False)

    # Rolling mean/std of portfolio returns
    rolling_mean = portfolio.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = portfolio.rolling(window=window, min_periods=min_periods).std(ddof=1)

    sharpe = rolling_mean / rolling_std

    # Avoid inf/-inf when std is zero
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    if scaling_factor:
        sharpe = sharpe * np.sqrt(scaling_factor)

    return sharpe


def certainty_equivalent(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    risk_free_df: pd.Series | pd.DataFrame,
    theta: float = 1.0,
    *,
    lag: int = 1,
    include_rf_in_mean: bool = True,
) -> float:
    """Certainty equivalent of the (next-period) portfolio return.

    Uses a mean-variance utility approximation:
        CE = E[R_p] + E [R_f]- (theta/2) Var(R_p)

    Returns:
        Scalar certainty equivalent.
    """
    assert isinstance(theta, (int, float)) and theta >= 0

    port = portfolio_returns(returns_df, weights_df, lag=lag, dropna=True)

    # Coerce risk-free to a Series
    if isinstance(risk_free_df, pd.DataFrame):
        assert risk_free_df.shape[1] == 1
        rf = risk_free_df.iloc[:, 0]
    else:
        rf = risk_free_df

    assert isinstance(rf, pd.Series)

    rf_realized = rf.loc[port.index]

    mean_term = float(port.mean())
    if include_rf_in_mean:
        mean_term += float(rf_realized.mean())

    var_term = float(port.var(ddof=1))

    ce = mean_term - (float(theta) / 2.0) * var_term
    return float(ce)


class SharpeTestResult(NamedTuple):
    sharpe_strategy: float       # Sharpe of the strategy being tested
    sharpe_benchmark: float      # Sharpe of the benchmark
    difference: float            # strategy - benchmark
    t_statistic: float           # GMM t-stat on the difference
    outperforms: bool            # strategy significantly better than benchmark
    underperforms: bool          # strategy significantly worse than benchmark
    p_approx: float              # approximate two-sided p-value (normal)


def _newey_west_cov(moments: np.ndarray, n_lags: int) -> np.ndarray:
    """Newey-West HAC covariance estimator for a (T, k) matrix of moment conditions.

    moments : (T, k) array — each row is the vector of moment conditions at time t
    n_lags  : number of lags to include (paper uses 22)

    Returns the (k, k) long-run covariance matrix S such that
    Avar(sqrt(T) * mean(moments)) = S.
    """
    T, k = moments.shape
    # demean
    m = moments - moments.mean(axis=0)
    S = (m.T @ m) / T
    for lag in range(1, n_lags + 1):
        gamma = (m[lag:].T @ m[:-lag]) / T
        weight = 1.0 - lag / (n_lags + 1.0)   # Bartlett kernel
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
    # Align on common index
    idx = returns_strategy.index.intersection(returns_benchmark.index)
    r_s = returns_strategy.loc[idx].to_numpy(dtype=float)
    r_b = returns_benchmark.loc[idx].to_numpy(dtype=float)

    T = len(r_s)
    if T < max(n_lags + 2, 10):
        raise ValueError(f"Too few observations ({T}) for Newey-West with {n_lags} lags.")

    # --- GMM point estimates (exactly identified, closed form) ---
    sigma_s  = float(np.std(r_s, ddof=1))
    sigma_b  = float(np.std(r_b, ddof=1))
    s_hat    = float(np.mean(r_s)) / sigma_s       # Sharpe of strategy
    sb_hat   = float(np.mean(r_b)) / sigma_b       # Sharpe of benchmark
    d_hat    = s_hat - sb_hat                       # difference

    # --- Build (T, 4) matrix of demeaned moment conditions ---
    # g1_t = r_s_t  - s_hat * sigma_s
    # g2_t = g1_t^2 - sigma_s^2
    # g3_t = r_b_t  - (s_hat - d_hat) * sigma_b   [= r_b_t - sb_hat * sigma_b]
    # g4_t = g3_t^2 - sigma_b^2
    g1 = r_s - s_hat * sigma_s
    g2 = g1 ** 2 - sigma_s ** 2
    g3 = r_b - sb_hat * sigma_b
    g4 = g3 ** 2 - sigma_b ** 2
    moments = np.column_stack([g1, g2, g3, g4])   # (T, 4)

    # --- Newey-West long-run covariance S ---
    S = _newey_west_cov(moments, n_lags)           # (4, 4)

    # --- Delta method: Jacobian of (s, d) w.r.t. the four moment conditions ---
    # The GMM estimator maps sample moments to (sigma_s, s, sigma_b, d).
    # We want Var(d_hat).  By the delta method:
    #   Var(sqrt(T) * d_hat) = J_d @ S @ J_d'
    # where J_d is the 1x4 row of the full Jacobian corresponding to d.
    #
    # Differentiating the moment conditions:
    #   d(s_hat)/d(mean_g1)  = 1/sigma_s
    #   d(s_hat)/d(sigma_s)  = -s_hat/sigma_s   (via d(sigma_s^2)/d(mean_g2) = 1 => d(sigma_s)/d(mean_g2) = 1/(2*sigma_s))
    #   Similarly for benchmark.
    #   d = s_hat - sb_hat, so J_d = J_s - J_sb.

    # Jacobian rows for s_hat (w.r.t. [mean_g1, mean_g2, mean_g3, mean_g4])
    J_s  = np.array([1.0 / sigma_s,
                     -s_hat / (2.0 * sigma_s ** 2),
                     0.0,
                     0.0])

    # Jacobian rows for sb_hat
    J_sb = np.array([0.0,
                     0.0,
                     1.0 / sigma_b,
                     -sb_hat / (2.0 * sigma_b ** 2)])

    J_d = J_s - J_sb   # 1x4

    # Asymptotic variance of sqrt(T) * d_hat
    avar_d = float(J_d @ S @ J_d)
    se_d   = np.sqrt(max(avar_d, 0.0) / T)

    t_stat = d_hat / se_d if se_d > 0 else np.sign(d_hat) * np.inf

    from scipy.stats import norm
    critical = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_approx = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))

    return SharpeTestResult(
        sharpe_strategy=s_hat,
        sharpe_benchmark=sb_hat,
        difference=d_hat,
        t_statistic=float(t_stat),
        outperforms=bool(t_stat >  critical),   # significantly better
        underperforms=bool(t_stat < -critical), # significantly worse
        p_approx=p_approx,
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
    DataFrame with columns:
        sharpe_strategy, sharpe_benchmark, difference, t_statistic, significant, p_approx
    indexed by strategy name.
    """
    r_bench = portfolio_returns(returns_df, weights_benchmark, lag=1, dropna=True)

    rows = {}
    for name, w in weights_strategies.items():
        r_strat = portfolio_returns(returns_df, w, lag=1, dropna=True)
        rows[name] = sharpe_difference_test(r_strat, r_bench, n_lags=n_lags, confidence=confidence)

    return pd.DataFrame(rows).T


def average_turnover(
    weights_df: pd.DataFrame,
    scaling_factor: int | None = None,
) -> float:
    """Mean daily portfolio turnover.

    Turnover at t = sum_i |w_{i,t} - w_{i,t-1}|.
    Averaged over all periods with a valid previous weight.
    Multiply by scaling_factor (e.g. 252) to annualise.
    """
    assert isinstance(weights_df, pd.DataFrame)

    daily = weights_df.diff().abs().sum(axis=1).dropna()

    if daily.empty:
        return float("nan")

    to = float(daily.mean())
    if scaling_factor:
        to *= scaling_factor
    return to
