"""Tests for config.py — every algorithm takes its defaults from config and
the same argument contract: (returns_df, *, start, ...), start required."""

import inspect

import pytest

from possibilistic_portfolio import alternative_investment_rules as air
from possibilistic_portfolio import bayesian_averaging as ba
from possibilistic_portfolio import config
from possibilistic_portfolio import possibilistic_bayesian as pb

ALGORITHMS = [
    air.equal_weight_strategy,
    air.market_weight_strategy,
    air.minimum_variance_strategy,
    air.historical_expectations_weights,
    air.rolling_window_weights,
    air.jorion_bayes_stein_estimates,
    air.jorion_bayes_stein_strategy,
    air.kan_zhou_three_fund_estimates,
    air.kan_zhou_three_fund_strategy,
    ba.run_core,
    pb.run_core,
]

# argument name -> config default, wherever an algorithm has that argument
SHARED_DEFAULTS = {
    "window": config.ROLLING_WINDOW_LONG,
    "theta": config.THETA,
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


def test_method_specific_defaults_come_from_config():
    b = inspect.signature(ba.run_core).parameters
    assert b["max_models"].default == config.BAYESIAN_MAX_MODELS
    assert b["prune_threshold"].default == config.BAYESIAN_PRUNE_THRESHOLD
    p = inspect.signature(pb.run_core).parameters
    assert p["max_models"].default == config.POSSIBILISTIC_MAX_MODELS
    assert p["merge_threshold"].default == config.MERGE_THRESHOLD
    assert p["nu_bandwidth"].default == config.NU_BANDWIDTH
    assert p["k_neighbours"].default == config.K_NEIGHBOURS
    assert p["gamma"].default == config.NECESSITY_GAMMA
    assert p["eta"].default == config.NECESSITY_ETA
    assert p["device"].default == config.NECESSITY_DEVICE
    assert p["n_jobs"].default == config.NECESSITY_N_JOBS


def test_start_is_burn_in_plus_periods_until_investment_in_the_pipeline():
    from possibilistic_portfolio import pipeline_wrapper as pw

    for fn in (
        pw.calculate_all_functions_perdatasets,
        pw.call_all_datasets,
        pw.call_all_simulated_datasets,
    ):
        params = inspect.signature(fn).parameters
        assert params["burn_in"].default == config.BURN_IN
        assert (
            params["periods_until_investment"].default
            == config.PERIODS_UNTIL_INVESTMENT
        )
        assert params["merge_threshold"].default == config.MERGE_THRESHOLD
        assert (
            params["rolling_window_long"].default == config.ROLLING_WINDOW_LONG
        )
