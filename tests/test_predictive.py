"""Tests for predictive.py — forecast dict format, index check, file format."""

import numpy as np
import pandas as pd
import pytest

from possibilistic_portfolio.markowitz import markowitz_unconstrained
from possibilistic_portfolio.predictive import (
    check_aligned,
    load_predictive,
    make_predictive,
    save_predictive,
)

DATES = pd.bdate_range("2000-01-03", periods=4, name="Date")


def _pred(index=DATES):
    T, n = len(index), 2
    return make_predictive(
        np.full((T, n), 0.001), np.tile(np.eye(n) * 1e-4, (T, 1, 1)), index
    )


def test_make_predictive_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        make_predictive(np.zeros((4, 2)), np.zeros((3, 2, 2)), DATES)


def test_check_aligned_passes_on_same_index_and_without_index():
    check_aligned(_pred(), DATES)
    check_aligned({"mu_hat": None, "sigma_hat": None}, DATES)  # old format


def test_check_aligned_names_the_first_mismatch():
    shifted = DATES + pd.offsets.BDay(1)
    with pytest.raises(ValueError, match="row 0: forecast 2000-01-04"):
        check_aligned(_pred(shifted), DATES)
    with pytest.raises(ValueError, match="4 forecast rows vs 3 return rows"):
        check_aligned(_pred(), DATES[:3])


@pytest.mark.parametrize("index", [DATES, pd.RangeIndex(4)])
def test_save_load_round_trip_keeps_index(tmp_path, index):
    pred = _pred(index)
    save_predictive(tmp_path, "algo", pred)
    back = load_predictive(tmp_path, "algo")
    assert back is not None
    assert back["index"].equals(index)
    np.testing.assert_array_equal(back["mu_hat"], pred["mu_hat"])
    np.testing.assert_array_equal(back["sigma_hat"], pred["sigma_hat"])


def test_load_falls_back_to_old_npy_files_and_none(tmp_path):
    np.save(tmp_path / "old_mus.npy", np.zeros((4, 2)))
    np.save(tmp_path / "old_sigmas.npy", np.zeros((4, 2, 2)))
    old = load_predictive(tmp_path, "old")
    assert old is not None
    assert set(old) == {"mu_hat", "sigma_hat"}  # no index saved back then
    assert load_predictive(tmp_path, "weights_only") is None


def test_markowitz_refuses_a_forecast_from_other_dates():
    returns = pd.DataFrame(0.0, index=DATES, columns=["a", "b"])
    shifted = _pred(DATES + pd.offsets.BDay(1))
    with pytest.raises(ValueError, match="forecast index does not match"):
        markowitz_unconstrained(shifted, returns, start=0)
