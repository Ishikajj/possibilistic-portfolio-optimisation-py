def run_core_diagnostic(
    returns_df: pd.DataFrame,
    burn_in: int = 100,
    n_steps: int = 300,
    max_models: int | None = 200,
    keep_newest: bool = True,
    gamma: float = 1.0,
    eta: float = 1.0,
) -> pd.DataFrame:
    """Run the algorithm for n_steps and record per-step diagnostics.

    Tracks the quantities most likely to reveal why the algorithm is misbehaving:
      - necessity distribution (mean, max, fraction == 0, newborn model score)
      - possibility distribution (max, entropy, effective number of models)
      - whether the masked weighting fell back to raw possibilities
      - aggregate Sigma condition number (predicts Markowitz blow-up)
      - model pool composition (n_models, mean nu, mean kappa)
    """
    import warnings

    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    t_end = min(burn_obs + n_steps, T)
    records = []

    for t in range(burn_obs, t_end):
        print(f"step {t}")
        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )
        n_models = len(mus)
        R_t = R[t]

        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            random_state=t,
        )

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        masked_weights = deterministic_mask_weighting(necessities, possibilities)
        power_weights = power_weighting(necessities, possibilities, gamma=gamma)

        # did masked weighting fall back? (all necessities == 0 → mask all → fallback)
        frac_nec_zero = float(np.mean(necessities == 0.0))
        masked_fallback = bool(np.all(necessities == 0.0))

        # possibility distribution
        poss_max = float(possibilities.max())
        poss_norm = possibilities / (possibilities.sum() + 1e-300)
        poss_entropy = float(-np.sum(poss_norm * np.log(poss_norm + 1e-300)))
        eff_n_models = float(np.exp(poss_entropy))  # effective number of models

        # aggregate Sigma condition number under masked weights
        try:
            mu_agg, Sigma_agg = _aggregate_possibilistic_niw(
                mus,
                kappas,
                Lambdas,
                nus,
                masked_weights if not masked_fallback else possibilities,
                n,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sigma_cond = float(np.linalg.cond(Sigma_agg))
            mu_norm = float(np.linalg.norm(mu_agg))
        except Exception:
            sigma_cond = np.nan
            mu_norm = np.nan

        records.append(
            {
                "t": t,
                "n_models": n_models,
                # necessity
                "nec_mean": float(necessities.mean()),
                "nec_max": float(necessities.max()),
                "nec_min": float(necessities.min()),
                "frac_nec_zero": frac_nec_zero,
                "newborn_necessity": float(necessities[-1]),  # newest model is last
                # possibility
                "poss_max": poss_max,
                "poss_entropy": poss_entropy,
                "eff_n_models": eff_n_models,
                # weighting
                "masked_fallback": masked_fallback,
                # model pool
                "nu_mean": float(np.mean(nus)),
                "nu_min": float(np.min(nus)),
                "kappa_mean": float(np.mean(kappas)),
                # aggregate quality
                "sigma_cond": sigma_cond,
                "mu_norm": mu_norm,
            }
        )

        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            max_models=max_models,
            keep_newest=keep_newest,
        )
        sum_R += R_t
        sum_R2 += R_t**2

    return pd.DataFrame(records).set_index("t")


def run_core_diagnostic_merging(
    returns_df: pd.DataFrame,
    burn_in: int = 100,
    n_steps: int = 300,
    max_models: int | None = 200,
    prune_threshold: float = 1e-6,
    keep_newest: bool = True,
    merge_threshold: float = 1.0,
    nu_bandwidth: int = 50,
    k_neighbours: int = 5,
    gamma: float = 1.0,
    eta: float = 1.0,
):
    """Run the algorithm with model merging and record per-step diagnostics.

    After each step: prune low-possibility models (top-k cap), then merge
    genuinely redundant pairs whose Gaussian summaries are within
    merge_threshold Hellinger distance and nu within nu_bandwidth. Aggregation uses all
    three necessity-weighted schemes (masked, power, exponential) exactly as
    in run_core.  Markowitz weights are computed via markowitz_unconstrained.

    Returns
    -------
    diag_df : pd.DataFrame
        Per-step diagnostics indexed by t.
    masked_predictive, power_predictive, exp_predictive : dict
        Each has keys "mu_hat" (T, n) and "sigma_hat" (T, n, n).
    weights_masked, weights_power, weights_exp : pd.DataFrame
        Markowitz weights for each weighting scheme, aligned to returns_df.
    """
    import warnings

    R_df = returns_df.copy()
    R = R_df.values.astype(float)
    T, n = R.shape

    mus: list[np.ndarray] = []
    kappas: list[float] = []
    Lambdas: list[np.ndarray] = []
    nus: list[float] = []
    possibilities = np.array([], dtype=float)

    sum_R = np.zeros(n)
    sum_R2 = np.zeros(n)

    burn_obs = min(int(burn_in), T // 2)
    if burn_obs > 0:
        R_burn = R[:burn_obs]
        sum_R = R_burn.sum(axis=0)
        sum_R2 = (R_burn * R_burn).sum(axis=0)

    t_end = min(burn_obs + n_steps, T)
    records = []
    mu_hat_arr_masked = np.full((T, n), np.nan)
    sigma_hat_arr_masked = np.full((T, n, n), np.nan)
    mu_hat_arr_power = np.full((T, n), np.nan)
    sigma_hat_arr_power = np.full((T, n, n), np.nan)
    mu_hat_arr_exp = np.full((T, n), np.nan)
    sigma_hat_arr_exp = np.full((T, n, n), np.nan)

    for t in range(burn_obs, t_end):
        if t % 100 == 0:
            print(f"Processing time step {t} / {t_end}...")
        mu_bar, lam_bar = _new_model_prior(sum_R, sum_R2, t)
        possibilities = _append_new_model(
            mus, kappas, Lambdas, nus, possibilities, mu_bar, lam_bar, n
        )
        n_models_raw = len(mus)
        R_t = R[t]

        possibilities = _update_possibilities(
            R_t, mus, kappas, Lambdas, nus, possibilities
        )
        _update_all_models(R_t, mus, kappas, Lambdas, nus)

        # Step 1: prune to max_models
        mus, kappas, Lambdas, nus, possibilities = _prune_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            max_models=max_models,
            keep_newest=keep_newest,
        )
        n_models_post_prune = len(mus)

        # Step 2: merge genuinely redundant pairs
        mus, kappas, Lambdas, nus, possibilities = _merge_models(
            mus,
            kappas,
            Lambdas,
            nus,
            possibilities,
            merge_threshold=merge_threshold,
            nu_bandwidth=nu_bandwidth,
            k_neighbours=k_neighbours,
        )
        n_models_post_merge = len(mus)

        # Step 3: necessity on the final merged pool so lengths match possibilities
        necessities = necessity_scores_fast(
            y_next=R_t,
            mus=mus,
            kappas=kappas,
            Lambdas=Lambdas,
            nus=nus,
            random_state=t,
        )

        masked_weights = deterministic_mask_weighting(necessities, possibilities)
        power_weights = power_weighting(necessities, possibilities, gamma=gamma)
        exp_weights = exponential_penalty_weighting(necessities, possibilities, eta=eta)

        sigma_cond = np.nan
        mu_norm = np.nan
        try:
            mu_masked, Sigma_masked = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, masked_weights, n
            )
            mu_power, Sigma_power = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, power_weights, n
            )
            mu_exp, Sigma_exp = _aggregate_possibilistic_niw(
                mus, kappas, Lambdas, nus, exp_weights, n
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sigma_cond = float(np.linalg.cond(Sigma_masked))
            mu_norm = float(np.linalg.norm(mu_masked))
            mu_hat_arr_masked[t] = mu_masked
            sigma_hat_arr_masked[t] = Sigma_masked
            mu_hat_arr_power[t] = mu_power
            sigma_hat_arr_power[t] = Sigma_power
            mu_hat_arr_exp[t] = mu_exp
            sigma_hat_arr_exp[t] = Sigma_exp
        except Exception:
            pass

        poss_norm = possibilities / (possibilities.sum() + 1e-300)
        poss_entropy = float(-np.sum(poss_norm * np.log(poss_norm + 1e-300)))
        eff_n_models = float(np.exp(poss_entropy))

        records.append(
            {
                "t": t,
                "n_models_raw": n_models_raw,
                "n_models_post_prune": n_models_post_prune,
                "n_models_post_merge": n_models_post_merge,
                "nec_mean": float(necessities.mean()),
                "nec_max": float(necessities.max()),
                "frac_nec_zero": float(np.mean(necessities == 0.0)),
                "masked_fallback": bool(np.all(necessities == 0.0)),
                "poss_entropy": poss_entropy,
                "eff_n_models": eff_n_models,
                "nu_mean": float(np.mean(nus)),
                "nu_min": float(np.min(nus)),
                "sigma_cond": sigma_cond,
                "mu_norm": mu_norm,
            }
        )

        sum_R += R_t
        sum_R2 += R_t**2

    diag_df = pd.DataFrame(records).set_index("t")

    masked_predictive = {"mu_hat": mu_hat_arr_masked, "sigma_hat": sigma_hat_arr_masked}
    power_predictive = {"mu_hat": mu_hat_arr_power, "sigma_hat": sigma_hat_arr_power}
    exp_predictive = {"mu_hat": mu_hat_arr_exp, "sigma_hat": sigma_hat_arr_exp}

    # --- minimal persistence: save diagnostic predictive arrays ---
    np.save("diag_mu_hat_masked.npy", mu_hat_arr_masked)
    np.save("diag_sigma_hat_masked.npy", sigma_hat_arr_masked)

    np.save("diag_mu_hat_power.npy", mu_hat_arr_power)
    np.save("diag_sigma_hat_power.npy", sigma_hat_arr_power)

    np.save("diag_mu_hat_exp.npy", mu_hat_arr_exp)
    np.save("diag_sigma_hat_exp.npy", sigma_hat_arr_exp)

    weights_masked = markowitz_unconstrained(
        masked_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
    )
    weights_power = markowitz_unconstrained(
        power_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
    )
    weights_exp = markowitz_unconstrained(
        exp_predictive, returns_df, burn_in=burn_in, periods_until_investment=0
    )

    return (
        diag_df,
        masked_predictive,
        weights_masked,
        power_predictive,
        weights_power,
        exp_predictive,
        weights_exp,
    )


def main_diagnostic():
    import matplotlib.pyplot as plt
    import bayesian_averaging
    from portfolio_evaluation_functions import (
        calculate_sharpe_ratio,
        rolling_sharpe_ratio,
        portfolio_returns as pf_returns,
    )
    from distribution_evalutation_func import average_log_likelihood

    BURN_IN = 100
    N_STEPS = 5000
    MAX_MODELS = 100

    NU_BANDWIDTH = 50
    K_NEIGHBOURS = 3
    ROLL_WINDOW = 60

    df = prepare_returns(
        load_excess_returns_from_kenneth_french_path(
            start_date="2000-01-01",
        )
    )
    burn_obs = min(BURN_IN, len(df) // 2)
    t_end = min(burn_obs + N_STEPS, len(df))
    returns_window = df.iloc[burn_obs + 1 : t_end]

    eval_slice = slice(burn_obs + 1, t_end)

    print(len(returns_window.columns))

    print("Running merged possibilistic diagnostic...")
    (
        diag,
        masked_pred,
        weights_masked_df,
        power_pred,
        weights_power_df,
        exp_pred,
        weights_exp_df,
    ) = run_core_diagnostic_merging(
        df,
        burn_in=BURN_IN,
        n_steps=N_STEPS,
        max_models=MAX_MODELS,
        merge_threshold=HELLINGER_THRESHOLD,
        nu_bandwidth=NU_BANDWIDTH,
        k_neighbours=K_NEIGHBOURS,
    )

    print("Running Bayesian averaging...")
    bay_ms, weights_bay_df = bayesian_averaging.run_core(
        df, burn_in=BURN_IN, periods_until_investment=0
    )
    mu_bay_df = pd.DataFrame(bay_ms["mu_hat"], index=df.index, columns=df.columns)

    # ── Slice weights to evaluation window ───────────────────────────────────
    w_masked = weights_masked_df.iloc[burn_obs + 1 : t_end]
    w_power = weights_power_df.iloc[burn_obs + 1 : t_end]
    w_exp = weights_exp_df.iloc[burn_obs + 1 : t_end]
    w_bay = weights_bay_df.iloc[burn_obs + 1 : t_end]

    strategies = {
        "masked": w_masked,
        "power": w_power,
        "exp": w_exp,
        "bayesian": w_bay,
    }

    # ── Sharpes ───────────────────────────────────────────────
    sharpes = {
        name: calculate_sharpe_ratio(returns_window, w, scaling_factor=252)
        for name, w in strategies.items()
    }
    rolls = {
        name: rolling_sharpe_ratio(
            returns_window, w, window=ROLL_WINDOW, scaling_factor=252
        )
        for name, w in strategies.items()
    }
    pf_rets = {
        name: pf_returns(returns_window, w, lag=1) for name, w in strategies.items()
    }

    # ── Average log-likelihoods ───────────────────────────────
    masked_pred_window = {
        "mu_hat": masked_pred["mu_hat"][eval_slice],
        "sigma_hat": masked_pred["sigma_hat"][eval_slice],
    }
    power_pred_window = {
        "mu_hat": power_pred["mu_hat"][eval_slice],
        "sigma_hat": power_pred["sigma_hat"][eval_slice],
    }
    exp_pred_window = {
        "mu_hat": exp_pred["mu_hat"][eval_slice],
        "sigma_hat": exp_pred["sigma_hat"][eval_slice],
    }
    bay_ms_window = {
        "mu_hat": bay_ms["mu_hat"][eval_slice],
        "sigma_hat": bay_ms["sigma_hat"][eval_slice],
    }

    avg_lls = {
        "masked": average_log_likelihood(returns_window, masked_pred_window),
        "power": average_log_likelihood(returns_window, power_pred_window),
        "exp": average_log_likelihood(returns_window, exp_pred_window),
        "bayesian": average_log_likelihood(returns_window, bay_ms_window),
    }

    # ── Save ────────────────────────────────────────────────────
    pd.DataFrame(pf_rets).to_csv("diagnostic_portfolio_returns.csv")
    pd.DataFrame(rolls).to_csv("diagnostic_rolling_sharpe.csv")
    pd.DataFrame(
        {
            "strategy": list(sharpes.keys()),
            "annualised_sharpe": list(sharpes.values()),
            "avg_log_likelihood": [avg_lls[k] for k in sharpes],
            "burn_in": BURN_IN,
            "n_steps": N_STEPS,
            "max_models": MAX_MODELS,
            "merge_threshold": [
                HELLINGER_THRESHOLD,
                HELLINGER_THRESHOLD,
                HELLINGER_THRESHOLD,
                None,
            ],
            "nu_bandwidth": [NU_BANDWIDTH, NU_BANDWIDTH, NU_BANDWIDTH, None],
            "roll_window": ROLL_WINDOW,
        }
    ).to_csv("diagnostic_sharpe_summary.csv", index=False)
    for name, w in strategies.items():
        w.to_csv(f"diagnostic_weights_{name}.csv")
    mu_bay_df.iloc[eval_slice].to_csv("diagnostic_mu_bayesian.csv")

    print(f"\n── Sharpe ───────────────────────────────────────────")
    for name, s in sharpes.items():
        print(f"  {name:12s}: {s:.4f}  (avg LL: {avg_lls[name]:.4f})")
    print(f"\n── Merge diagnostic ──────────────────────────────────")
    print(
        diag[
            [
                "n_models_raw",
                "n_models_post_prune",
                "n_models_post_merge",
                "nec_max",
                "sigma_cond",
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )

    # ── Figure 1: 6-panel merging diagnostic ────────────────────────────────
    fig1, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig1.suptitle(
        "Possibilistic Bayesian + Hellinger Merging — Per-Step Diagnostics",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(diag.index, diag["nec_mean"], label="mean necessity")
    ax.plot(diag.index, diag["nec_max"], label="max necessity", linestyle="--")
    ax.set_title("Necessity scores")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(diag.index, diag["frac_nec_zero"] * 100)
    ax.set_title("% Models with necessity = 0")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(
        diag.index, diag["n_models_raw"], label="post-birth", alpha=0.4, linestyle=":"
    )
    ax.plot(diag.index, diag["n_models_post_prune"], label="post-prune", linestyle="--")
    ax.plot(diag.index, diag["n_models_post_merge"], label="post-merge", linewidth=2)
    ax.axhline(
        diag["n_models_post_merge"].median(),
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"median={diag['n_models_post_merge'].median():.0f}",
    )
    ax.set_title("Model pool size across pipeline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(diag.index, diag["poss_entropy"], label="entropy")
    ax.plot(diag.index, diag["eff_n_models"], label="eff. models", linestyle="--")
    ax.set_title("Possibility entropy + effective models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.semilogy(diag.index, diag["sigma_cond"].clip(lower=1))
    ax.set_title("Aggregate Σ condition number (log scale)")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(diag.index, diag["nu_mean"], label="mean ν")
    ax.plot(diag.index, diag["nu_min"], label="min ν", linestyle="--")
    ax.set_title("NIW degrees of freedom ν across models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig1.tight_layout()
    plt.savefig("diagnostic_merging.png", dpi=150)

    # ── Figure 2: Sharpe comparison ──────────────────────────────────────────
    colors_map = {
        "masked": "steelblue",
        "power": "seagreen",
        "exp": "mediumpurple",
        "bayesian": "darkorange",
    }
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        f"Weighting schemes vs Bayesian  (burn={BURN_IN}, steps={N_STEPS})", fontsize=12
    )

    ax = axes2[0]
    labels = list(sharpes.keys())
    values = list(sharpes.values())
    clrs = [colors_map[k] for k in labels]
    bars = ax.bar(labels, values, color=clrs, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Annualised Sharpe Ratio")
    ax.grid(axis="y", alpha=0.3)

    ax = axes2[1]
    for name, roll in rolls.items():
        ax.plot(
            roll.index, roll.values, label=name, color=colors_map[name], linewidth=1.1
        )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"Rolling Sharpe (window={ROLL_WINDOW}d)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig2.tight_layout()
    plt.savefig("sharpe_comparison_merging.png", dpi=150)
    plt.show()

    return diag, pf_rets
