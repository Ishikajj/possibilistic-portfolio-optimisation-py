from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from prior_selection import sharing_prior_update
from scipy.special import multigammaln
from data_input import load_excess_returns_from_kenneth_french_path, prepare_returns
from markowitz import markowitz_unconstrained

# TODO: main problem is the T^2n^3 complexity of the algorithm, where t is the time steps and n is assets
# the n^3 remains fixed as the number of assets = 11
# but T grows monsterly: we have about 70 years of data of 250 trading days each
# this becomes bad quick
# Update: 1963-2011, T = 12000 took about 40 minutes.

"""step 1
-> create a new model with mean  = common mean across all previous days and assets,
 similarly create new delta and k using the +1 update rule.
 the probability of this model is according to the past models probabilities and our prior used.

step 2
-> observe returns for the new day

step 3
-> using the returns, calculate the updated means and covariances using formula 7a and 7b

step 4
-> using the likelihood function, calculate the updated probabilities of each model. this will require the previous convariance, new covariance, previous delta, new delta, previous k, new k, previous v, new v

step 5
-> discard the old (sigma, delta ( v - n - 1), k, mean)

step 6
-> use these updated model probabilities and the updated model variances and means to calculate the expected returns and covariances. store said time indexed series.

step 7
-> do steps 1-6 until you reach the end of time

short note:

Weights calculation:
the integer indexed series of bayesian averaged mean and covariances is used to calculate the markowitz weights using a separate function.

this function will return a integer indexed series of weights of each assets.

Portfolio return determination:
using the weights and the returns series that we get, we can run these to calculate the sharpe and certainty equivalents each day.

for each model, we have its mean, covariance, degrees of freedom k and v, and its probability. as time goes on, the number of models increases. step 1.1 indicates the models of t according to the information available at time t-1

okay so each model uses the probability as a scalar, meaning we can not have different windows across different assets even though that might be a better predictor of mean.
this is due to having a common covariance matrix across all assets, even if means differ."""


"""data input is
NoDur           float64
Durbl           float64
Manuf           float64
Enrgy           float64
Chems           float64
BusEq           float64
Telcm           float64
Utils           float64
Shops           float64
Hlth            float64
Money           float64
Other           float64
time           datetime64[ns]
"""


def _new_model_prior(
    sum_R: np.ndarray, sum_R2: np.ndarray, t: int
) -> tuple[float, float]:
    """Compute the scalar prior hyperparameters for the newborn model at time t.
    Note that at birth, delta = 1, and since Sigma = delta * Lambda, at birth, both these are identical.

    This implements the paper's *common* (across assets) prior for the new model:
    Parameters
    ----------
    sum_R:
        Running column-wise sum of past returns, shape (n,).

    sum_R2:
        Running column-wise sum of past squared returns, shape (n,).

    t:
        Number of past observations included in the running sums.
        Must satisfy t >= 0.

    Returns
    -------
    (mu_bar, lambda_bar):
        mu_bar is the average across assets of the historical sample means.
        lambda_bar is the average across assets of the historical sample variances.

    Note:
    The paper uses t-1 for the mean and t-2 for the variance, this is due to differently defining what the current time period is.
    mathematically this is equivalent.
    """
    if t <= 0:
        return 0.0, 1e-4

    mu_vec = sum_R / (
        float(t) - 1
    )  # shape (n,), only information until t-1 is used to form the new models priors.
    mu_bar = float(mu_vec.mean())

    if t <= 1:
        # Cannot compute sample variance with < 2 observations.
        return mu_bar, 1e-4

    # Per-asset sample variance: Var(R_i) = (sum R_i^2 - t * mean_i^2) / (t-1)
    var_vec = (sum_R2 - t * mu_vec**2) / float(t - 2)
    var_vec = np.maximum(var_vec, 1e-8)
    return mu_bar, float(var_vec.mean())


def _append_new_model(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    mu_bar: float,
    lam_bar: float,
    n: int,
) -> None:
    """takes the previously existing models, and the newly caclulated variance and mean, and appends those to the previously existing models."""
    mus.append(np.full(n, mu_bar, dtype=float))
    kappas.append(1.0)
    Lambdas.append(lam_bar * np.eye(n, dtype=float))
    nus.append(
        float(n + 2.0)
    )  # paper used delta = 1 at initialisation, and delta = nu - n - 1


def _update_probs(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
) -> np.ndarray:

    # ll is an array of the log of the likelihood functions for each model given our observed returns.
    ll = np.array(
        [
            _log_marginal_likelihood(
                R_t,
                mus[m],
                kappas[m],
                Lambdas[m],
                nus[m],
            )
            for m in range(len(mus))
        ]
    )

    # logaddexp basically does log(sum(exp(i))).
    # since i is our log of probabilities, this is log (sum of probabilities)
    # so we get the denominator of bayes rule in log space.
    lognum = ll + np.log(np.maximum(probs, 1e-300))
    logZ = np.logaddexp.reduce(
        lognum
    )  # logaddexp only takes 2 inputs, reduce applies repeatedly.

    p = np.exp(lognum - logZ)
    p = np.maximum(p, 0.0)  # effectively handing cases where the likelihood is -inf
    return p / p.sum()  # normalise to sum 1.


def _log_marginal_likelihood(R_t, mu, kappa, Lambda, nu):
    """Log NIW marginal likelihood log L(R_t | m, F_{t-1}).

    Uses `scipy.special.multigammaln(a, d)`, which returns `log Γ_d(a)` (the natural
    log of the multivariate gamma function). We work in logs because Γ_d(a) and the
    NIW normalizing constants can be extremely large/small; log-space avoids overflow
    and turns products/ratios into sums/differences.

    the formulae used here are from anderson cheng section 3.2.
    """
    n = len(R_t)
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(
        d, d
    )  # Paper uses sigma for updates, which has been simplified here to Lambda. this is also why the delta denominator vanishes.

    s0, ld0 = np.linalg.slogdet(
        Lambda
    )  # computes the sign, log of determinant of Lambda. Due to PD, this must be +.
    s1, ld1 = slogdet(L1)
    if s0 <= 0 or s1 <= 0:
        return (
            -np.inf
        )  # indicates cases of mathematically imposible covariance matrices.

    log_likelihood = (
        multigammaln(nu1 / 2.0, n)
        - multigammaln(nu / 2.0, n)
        + (n / 2.0) * np.log(kappa / k1)
        + (nu / 2.0) * ld0  # determinant was already logged.
        - (nu1 / 2.0) * ld1
        - (n / 2.0) * np.log(np.pi)
    )
    return log_likelihood


def _update_all_models(
    R_t: np.ndarray,
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
) -> None:
    for m in range(len(mus)):
        mus[m], kappas[m], Lambdas[m], nus[m] = _update_niw(
            R_t, mus[m], kappas[m], Lambdas[m], nus[m]
        )


def _update_niw(R_t, mu, kappa, Lambda, nu):
    k1 = kappa + 1.0
    nu1 = nu + 1.0
    mu1 = (kappa * mu + R_t) / k1
    d = R_t - mu
    L1 = Lambda + (kappa / k1) * np.outer(d, d)
    return mu1, k1, L1, nu1


def _ba_moments(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """formulae from section 3.1"""

    mu_hat = sum(probs[m] * mus[m] for m in range(len(mus)))
    Sigma_hat = np.zeros((n, n))
    for m in range(len(mus)):
        Sigma_m = _sigma_m(Lambdas[m], nus[m], n)
        Sigma_bar = (1.0 + 1.0 / kappas[m]) * Sigma_m
        Sigma_hat += probs[m] * (Sigma_bar + np.outer(mus[m], mus[m]))
    Sigma_hat -= np.outer(mu_hat, mu_hat)
    return mu_hat, Sigma_hat


def _sigma_m(Lambda, nu, n):
    return Lambda / (nu - n - 1.0)


def _prune_models(
    mus: list[np.ndarray],
    kappas: list[float],
    Lambdas: list[np.ndarray],
    nus: list[float],
    probs: np.ndarray,
    prune_threshold: float = 1e-6,
    max_models: int | None = 500,
    keep_newest: bool = True,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[float],
    np.ndarray,
]:
    """Prune low-probability models to keep runtime sub-quadratic in T.

    Strategy:
    - drop models with posterior probability below `prune_threshold`
    - optionally cap the total number of retained models at `max_models`
    - optionally force retention of the newest model (last index)

    After pruning, probabilities are renormalized to sum to 1.
    """
    n_models = len(mus)

    keep_mask = probs >= prune_threshold
    if keep_newest:
        keep_mask[-1] = True

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        fallback_count = n_models if max_models is None else min(n_models, max_models)
        keep_idx = np.arange(n_models - fallback_count, n_models, dtype=int)

    if max_models is not None and keep_idx.size > max_models:
        order = np.argsort(probs[keep_idx])[
            ::-1
        ]  # sorts by descending the probabilities.
        keep_idx = keep_idx[order[:max_models]]
        keep_idx = np.sort(keep_idx)
        if keep_newest and keep_idx[-1] != n_models - 1:
            keep_idx[-1] = n_models - 1
            keep_idx = np.unique(np.sort(keep_idx))
            if keep_idx.size > max_models:
                drop_candidates = keep_idx[keep_idx != n_models - 1]
                smallest = drop_candidates[np.argmin(probs[drop_candidates])]
                keep_idx = keep_idx[keep_idx != smallest]

    mus = [mus[i] for i in keep_idx]
    kappas = [kappas[i] for i in keep_idx]
    Lambdas = [Lambdas[i] for i in keep_idx]
    nus = [nus[i] for i in keep_idx]
    probs = probs[keep_idx].astype(float, copy=False)

    prob_sum = probs.sum()
    if prob_sum <= 0.0:
        probs = np.full(len(probs), 1.0 / len(probs))
    else:
        probs = probs / prob_sum

    return mus, kappas, Lambdas, nus, probs


# return core gets the input from data_input file, it receives a file with integer indexing, a column for time also.
# this helps you specify the date you want to slice, the source of the data.
def run_core(
    returns_df: pd.DataFrame,
    burn_in: int = 1000,
    periods_until_investment=0,
    prune_threshold: float = 1e-6,
    max_models: int | None = 2000,
    keep_newest: bool = True,
) -> dict[str, np.ndarray]:

    # important to note that a dataframe with the time index is still retained, and can be appended to the end of our produced weight series if needed.
    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mu_hat_arr = np.full((T, n), np.nan)
    sigma_hat_arr = np.full((T, n, n), np.nan)

    # mu is the mean for each model, kappa is the precision about the mean
    # Lambda is the scale parameter for the covariance matrix, and nu is the degrees of freedom.
    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    probs = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    # burn_obs must be an integer for array slicing
    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    for t in range(burn_obs, T):
        if t % 100 == 0:
            print(f"Processing time step {t} / {T}...")
        mu_bar, lam_bar = _new_model_prior(
            sum_R, sum_R2, t
        )  # Note, this is not using the R[t] yet
        _append_new_model(mus, kappas, Lambdas, nus, mu_bar, lam_bar, n)

        n_models = len(mus)

        probs = sharing_prior_update(probs_prev=probs, t_models=n_models, alpha=1.0)

        # Now, after a new model has been added, it has been given a weak prior based on previous information flow
        # Only after that we observe the returns for the day.
        R_t = R[t]

        probs = _update_probs(R_t, mus, kappas, Lambdas, nus, probs)

        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        mus, kappas, Lambdas, nus, probs = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            probs,
            prune_threshold=prune_threshold,
            max_models=max_models,
            keep_newest=keep_newest,
        )

        mu_hat, Sigma_hat = _ba_moments(mus, kappas, Lambdas, nus, probs, n)
        mu_hat_arr[t] = mu_hat
        sigma_hat_arr[t] = Sigma_hat

        sum_R += R_t
        sum_R2 += R_t**2
    mu_sigma_dict = {"mu_hat": mu_hat_arr, "sigma_hat": sigma_hat_arr}

    weights = markowitz_unconstrained(
        mu_sigma_dict=mu_sigma_dict,
        returns_df=returns_df,
        burn_in=burn_in,
        periods_until_investment=periods_until_investment,
    )

    # VERY IMPORTANT: the returned mu_hat_arr and sigma_hat_arr have info upto time t, meaning they are predicting t+1.
    return mu_sigma_dict, weights


def main():
    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(start_date="2020-01-01")
    )
    results = run_core(df, burn_in=100)
    print(results["mu_hat"])
    print(results["sigma_hat"])


if __name__ == "__main__":
    main()
