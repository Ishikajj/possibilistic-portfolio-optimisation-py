"""Tests for config.py — every algorithm takes its defaults from config and
the same argument contract: (returns_df, *, start, ...), start required."""

import inspect

import pytest

from possibilistic_portfolio import alternative_investment_rules as air
from possibilistic_portfolio import bayesian_averaging as ba
from possibilistic_portfolio import config
from possibilistic_portfolio import possibilistic_bayesian as pb
from possibilistic_portfolio import weight_rules as wr

WEIGHT_RULES = [
    wr.equal_weights,
    wr.market_weights,
    wr.minimum_variance_weights,
]
FORECASTERS = [
    air.historical_expanding_forecast,
    air.rolling_window_forecast,
    air.jorion_bayes_stein_estimates,
    air.jorion_bayes_stein_forecast,
    air.kan_zhou_three_fund_estimates,
    air.kan_zhou_three_fund_forecast,
    ba.run_core,
    pb.run_core,
]
ALGORITHMS = WEIGHT_RULES + FORECASTERS

# argument name -> config default, wherever an algorithm has that argument
SHARED_DEFAULTS = {
    "window": config.ROLLING_WINDOW_LONG,
    "burn_in": config.BURN_IN,
    "keep_newest": config.KEEP_NEWEST,
}


@pytest.mark.parametrize("fn", ALGORITHMS, ids=lambda f: f.__qualname__)
def test_keyword_only_with_required_start(fn):
    params = list(inspect.signature(fn).parameters.values())
    assert params[0].name == "returns_df"
    assert all(p.kind is p.KEYWORD_ONLY for p in params[1:])
    start = inspect.signature(fn).parameters["start"]
    assert start.default is inspect.Parameter.empty  # must be passed


@pytest.mark.parametrize("fn", ALGORITHMS, ids=lambda f: f.__qualname__)
def test_shared_defaults_come_from_config(fn):
    params = inspect.signature(fn).parameters
    for name, default in SHARED_DEFAULTS.items():
        if name in params:
            assert params[name].default == default, name


@pytest.mark.parametrize("fn", FORECASTERS, ids=lambda f: f.__qualname__)
def test_forecasters_do_not_take_theta(fn):
    """Risk aversion only matters for the weights: theta goes to Markowitz
    (weight_rules) and nowhere else."""
    assert "theta" not in inspect.signature(fn).parameters


@pytest.mark.parametrize(
    "fn", [wr.markowitz_unconstrained, wr.markowitz_long_only]
)
def test_markowitz_theta_default_comes_from_config(fn):
    assert inspect.signature(fn).parameters["theta"].default == config.THETA


def test_method_specific_defaults_come_from_config():
    b = inspect.signature(ba.run_core).parameters
    assert b["max_models"].default == config.BAYESIAN_MAX_MODELS
    assert b["prune_threshold"].default == config.BAYESIAN_PRUNE_THRESHOLD
    p = inspect.signature(pb.run_core).parameters
    assert p["max_models"].default == config.POSSIBILISTIC_MAX_MODELS
    assert p["merge_threshold"].default == config.MERGE_THRESHOLD
    assert p["nu_bandwidth"].default == config.NU_BANDWIDTH
    assert p["k_neighbours"].default == config.K_NEIGHBOURS
    assert p["necessity"].default == config.USE_NECESSITY
    assert p["gamma"].default == config.NECESSITY_GAMMA
    assert p["eta"].default == config.NECESSITY_ETA
    assert p["device"].default == config.NECESSITY_DEVICE
    assert p["n_jobs"].default == config.NECESSITY_N_JOBS


def test_necessity_is_off_by_default():
    """Plain possibility weighting is the default; necessity is an option."""
    assert config.USE_NECESSITY is False


def test_pipeline_settings_default_to_config():
    from possibilistic_portfolio.pipeline_wrapper import RunSettings

    s = RunSettings()
    assert (s.burn_in, s.periods_until_investment, s.theta) == (
        config.BURN_IN,
        config.PERIODS_UNTIL_INVESTMENT,
        config.THETA,
    )
    assert (s.rolling_window_long, s.rolling_window_short) == (
        config.ROLLING_WINDOW_LONG,
        config.ROLLING_WINDOW_SHORT,
    )
    assert s.bayesian_max_models == config.BAYESIAN_MAX_MODELS
    assert s.possibilistic_max_models == config.POSSIBILISTIC_MAX_MODELS
    assert (s.merge_threshold, s.nu_bandwidth, s.k_neighbours) == (
        config.MERGE_THRESHOLD,
        config.NU_BANDWIDTH,
        config.K_NEIGHBOURS,
    )
    assert (s.necessity, s.gamma, s.eta, s.device, s.n_jobs) == (
        config.USE_NECESSITY,
        config.NECESSITY_GAMMA,
        config.NECESSITY_ETA,
        config.NECESSITY_DEVICE,
        config.NECESSITY_N_JOBS,
    )
    assert s.keep_newest == config.KEEP_NEWEST
    assert s.bayesian_prune_threshold == config.BAYESIAN_PRUNE_THRESHOLD
    # start = burn_in + periods_until_investment, for every strategy
    assert RunSettings(burn_in=7, periods_until_investment=3).start == 10


def test_run_algorithms_come_from_config():
    """main / main_simulated run config.RUN_ALGORITHMS (None = all), and
    every name listed there exists."""
    from possibilistic_portfolio import pipeline_wrapper as pw

    for entry in (pw.main, pw.main_simulated):
        default = inspect.signature(entry).parameters["algorithms"].default
        assert default == config.RUN_ALGORITHMS
    if config.RUN_ALGORITHMS is not None:
        assert set(config.RUN_ALGORITHMS) <= set(pw.ALGORITHMS)


def test_date_window_comes_from_config():
    from possibilistic_portfolio import pipeline_wrapper as pw

    p = inspect.signature(pw.call_all_datasets).parameters
    assert p["start_date"].default == config.START_DATE
    assert p["end_date"].default == config.END_DATE
