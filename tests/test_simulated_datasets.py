"""Tests for simulated_datasets.py — synthetic single-factor DGPs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import simulated_datasets as sd

SIM_DIR = Path(__file__).resolve().parents[2] / "datasets" / "simulated"

DGPS = {
    "iid": sd.simulate_iid,
    "reg": sd.simulate_regime_shifts,
    "break": sd.simulate_sudden_break,
    "mr": sd.simulate_mean_reverting_factor,
    "cov_break": sd.simulate_sudden_covariance_break,
    "stoch_vol": sd.simulate_stochastic_volatility,
}


@pytest.mark.parametrize("name", DGPS)
def test_shapes_columns_and_index(name):
    ports, rf = DGPS[name](T=500, n_assets=4, seed=1)
    assert ports.shape == (500, 4)
    assert list(ports.columns) == ["Asset1", "Asset2", "Asset3", "Asset4"]
    assert list(rf.columns) == ["RF"] and len(rf) == 500
    assert ports.index.equals(pd.RangeIndex(500))
    assert np.isfinite(ports.values).all() and np.isfinite(rf.values).all()


@pytest.mark.parametrize("name", DGPS)
def test_same_seed_same_data_different_seed_different(name):
    a, _ = DGPS[name](T=300, n_assets=3, seed=7)
    b, _ = DGPS[name](T=300, n_assets=3, seed=7)
    c, _ = DGPS[name](T=300, n_assets=3, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.values, c.values)


def test_iid_moments_match_targets():
    ports, rf = sd.simulate_iid(T=2_000_000, n_assets=3, seed=0)
    B = np.linspace(0.5, 1.5, 3)
    assert np.allclose(ports.mean(), B * sd._MU_BAR, atol=1e-4)  # ~4 std errors
    assert np.isclose(rf["RF"].mean(), sd._MU_RF, atol=1e-5)
    assert np.isclose(rf["RF"].std(), sd._SIGMA_RF, rtol=0.02)


def test_factor_loadings_spread_from_half_to_one_and_a_half():
    rng = np.random.default_rng(0)
    g = np.ones(10)
    _, B = sd._simulate_factor_model_excess_returns(rng, g, 5)
    assert np.allclose(B, [0.5, 0.75, 1.0, 1.25, 1.5])


def test_sudden_break_switches_mean_at_break_point():
    T = 100_000
    ports, _ = sd.simulate_sudden_break(T=T, n_assets=1, seed=0)
    pre, post = ports.iloc[: T // 2, 0], ports.iloc[T // 2 :, 0]
    # asset 1 has loading 0.5 when n_assets = 1 -> linspace(0.5, 1.5, 1) = [0.5]
    assert pre.mean() > 0 > post.mean()


def test_covariance_break_raises_vol_after_break():
    g = sd._simulate_factor_vol_break(np.random.default_rng(0), T=20_000)
    assert g[10_000:].std() > 2 * g[:10_000].std()  # 25% vs 10% annual vol


def test_regime_mean_rejects_impossible_variance():
    with pytest.raises(ValueError):
        sd._simulate_factor_regime_mean(
            np.random.default_rng(0), T=10, p_switch=0.5, rho=0.0, sigma_mu=1.0
        )


def test_stochastic_vol_variance_is_positive_and_varies():
    g = sd._simulate_factor_stochastic_vol(np.random.default_rng(0), T=5000)
    rolling = pd.Series(g).rolling(100).std().dropna()
    assert rolling.max() > 1.5 * rolling.min()


@pytest.mark.skipif(
    not SIM_DIR.exists(), reason="../datasets/simulated not available"
)
@pytest.mark.parametrize("name", DGPS)
def test_generators_reproduce_saved_datasets(name):
    """Snapshot: today's code regenerates the CSVs used for the FYP exactly."""
    ports, rf = DGPS[name](T=10_000, n_assets=10, seed=42)
    tag = f"{name}_T10000_N10_seed42.csv"
    disk = pd.read_csv(SIM_DIR / tag, float_precision="round_trip")
    disk_rf = pd.read_csv(SIM_DIR / f"rf_{tag}", float_precision="round_trip")
    np.testing.assert_array_equal(ports.values, disk.values)
    np.testing.assert_array_equal(rf.values, disk_rf.values)
