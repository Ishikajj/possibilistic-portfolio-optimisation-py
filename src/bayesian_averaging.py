"""check data input, fixed windows"""

import numpy as np
from numpy.linalg import slogdet, inv
from math import lgamma, log, pi


def multivariate_lgamma(a: float, p: int) -> float:
    # log multivariate gamma Γ_p(a)
    return (p * (p - 1) / 4) * log(pi) + sum(
        lgamma(a + (1 - j) / 2) for j in range(1, p + 1)
    )


def niw_posterior_params(
    X: np.ndarray, mu0: np.ndarray, kappa0: float, nu0: float, Lambda0: np.ndarray
):
    """
    NIW prior:
      Σ ~ Inv-Wishart(ν0, Λ0)
      μ | Σ ~ N(μ0, Σ / κ0)

    Posterior after observing X (n x p):
      κn = κ0 + n
      νn = ν0 + n
      μn = (κ0 μ0 + n xbar) / κn
      Λn = Λ0 + S + (κ0 n / κn) (xbar - μ0)(xbar - μ0)'
    where S = sum (xi - xbar)(xi - xbar)'.
    """
    X = np.asarray(X)
    n, p = X.shape
    xbar = X.mean(axis=0)
    Xm = X - xbar
    S = Xm.T @ Xm  # scatter

    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n

    d = (xbar - mu0).reshape(-1, 1)
    Lambda_n = Lambda0 + S + (kappa0 * n / kappa_n) * (d @ d.T)

    return mu_n, kappa_n, nu_n, Lambda_n


def student_t_logpdf(
    x: np.ndarray, m: np.ndarray, Sigma: np.ndarray, df: float
) -> float:
    """
    Multivariate Student-t logpdf with location m, scale Sigma, df.
    """
    x = np.asarray(x).reshape(-1)
    m = np.asarray(m).reshape(-1)
    p = x.size

    delta = (x - m).reshape(-1, 1)
    Sinv = inv(Sigma)
    quad = float(delta.T @ Sinv @ delta)

    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Scale matrix not PD.")

    return (
        multivariate_lgamma((df + p) / 2, p)
        - multivariate_lgamma(df / 2, p)
        - (p / 2) * log(df * pi)
        - 0.5 * logdet
        - ((df + p) / 2) * log(1 + quad / df)
    )


def niw_posterior_predictive_logpdf(
    x: np.ndarray,
    X: np.ndarray,
    mu0: np.ndarray,
    kappa0: float,
    nu0: float,
    Lambda0: np.ndarray,
) -> float:
    """
    Predictive is multivariate Student-t:
      x | X ~ t_{νn - p + 1}(μn, ((κn + 1)/(κn * (νn - p + 1))) Λn )
    """
    X = np.asarray(X)
    n, p = X.shape

    mu_n, kappa_n, nu_n, Lambda_n = niw_posterior_params(X, mu0, kappa0, nu0, Lambda0)

    df = nu_n - p + 1
    if df <= 2:  # practical sanity
        return -np.inf

    scale = (kappa_n + 1) / (kappa_n * df) * Lambda_n
    return student_t_logpdf(x, mu_n, scale, df)


def choose_window_by_prequential_score(
    returns: np.ndarray,
    windows=(20, 40, 60, 120, 252),
    mu0=None,
    kappa0=1.0,
    nu0=None,
    Lambda0=None,
    burn_in=None,
):
    """
    returns: (T,) for univariate OR (T,p) for multivariate.
    windows: candidate rolling window lengths W.
    Scores each W by sum_{t=burn_in..T-1} log p(r_t | r_{t-W:t-1}) using NIW.
    """
    R = np.asarray(returns)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    T, p = R.shape

    if mu0 is None:
        mu0 = np.zeros(p)
    else:
        mu0 = np.asarray(mu0).reshape(-1)

    # weakly-informative defaults
    if nu0 is None:
        nu0 = p + 2.0  # must be > p-1; small-ish df
    if Lambda0 is None:
        Lambda0 = np.eye(p) * 1e-4  # small scale prior
    if burn_in is None:
        burn_in = max(windows)

    scores = {}
    for W in windows:
        if W < 2 or W >= T:
            scores[W] = -np.inf
            continue
        s = 0.0
        # start at max(burn_in, W) so every predictive uses full window
        start = max(burn_in, W)
        for t in range(start, T):
            X = R[t - W : t, :]  # window history
            x = R[t, :]
            lp = niw_posterior_predictive_logpdf(x, X, mu0, kappa0, nu0, Lambda0)
            s += lp
        scores[W] = s

    best_W = max(scores, key=scores.get)
    return best_W, scores


# ---- Example usage ----
if __name__ == "__main__":
    # toy: univariate returns
    rng = np.random.default_rng(0)
    T = 2000
    # regime shift to make window selection nontrivial
    r1 = rng.normal(0.0005, 0.01, size=1200)
    r2 = rng.normal(0.0002, 0.02, size=800)
    r = np.concatenate([r1, r2])

    best_W, scores = choose_window_by_prequential_score(
        r,
        windows=(20, 40, 60, 120, 252),
        # prior can be tuned; these are defaultish
        mu0=np.array([0.0]),
        kappa0=1.0,
        nu0=1 + 2.0,  # p=1 => nu0 > 0; keep small
        Lambda0=np.array([[1e-4]]),
    )

    print("best W:", best_W)
    for W in sorted(scores):
        print(W, scores[W])
