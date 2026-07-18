from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.portfolio.hrp import compute_hrp_weights, compute_returns_matrix


def test_hrp_single_asset_gets_full_weight():
    returns = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 1, 50)})
    w = compute_hrp_weights(returns)
    assert w["A"] == pytest.approx(1.0)


def test_hrp_two_uncorrelated_assets_matches_inverse_variance():
    """For N=2, HRP has a known closed form: it reduces exactly to
    inverse-variance weighting (a single split, each side a single-asset
    cluster). This is the standard sanity check for any HRP implementation."""
    rng = np.random.default_rng(1)
    n = 2000
    a = rng.normal(0, 2.0, n)  # var ~= 4
    b = rng.normal(0, 1.0, n)  # var ~= 1
    returns = pd.DataFrame({"A": a, "B": b})

    w = compute_hrp_weights(returns)
    var_a, var_b = returns["A"].var(), returns["B"].var()
    expected_a = (1 / var_a) / (1 / var_a + 1 / var_b)

    assert w["A"] == pytest.approx(expected_a, abs=0.03)
    assert w.sum() == pytest.approx(1.0)


def test_hrp_weights_always_sum_to_one():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0, 1, (200, 6)), columns=list("ABCDEF"))
    w = compute_hrp_weights(returns)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()  # HRP never shorts or zeroes out an asset


def test_hrp_diversifies_away_from_a_correlated_cluster():
    """Two highly correlated assets should not simply get 2x the weight of
    an independent third asset -- HRP should recognize they're largely
    redundant and treat their *combined* weight comparably to the
    independent asset's individual weight. This is the whole point of HRP
    over naive equal-weighting."""
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(0, 1.0, n)
    y = 0.9 * x + rng.normal(0, 0.3, n)  # highly correlated with x, similar variance
    z = rng.normal(0, 1.0, n)  # independent, same variance as x
    returns = pd.DataFrame({"X": x, "Y": y, "Z": z})

    w = compute_hrp_weights(returns)
    combined_xy = w["X"] + w["Y"]
    assert combined_xy == pytest.approx(w["Z"], rel=0.3)  # roughly comparable, not 2x


def test_hrp_lower_variance_asset_gets_more_weight():
    rng = np.random.default_rng(4)
    n = 1000
    low_vol = rng.normal(0, 0.5, n)
    high_vol = rng.normal(0, 3.0, n)
    returns = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol})

    w = compute_hrp_weights(returns)
    assert w["LOW"] > w["HIGH"]


def test_compute_returns_matrix_forward_fills_isolated_single_day_gaps():
    """Observed live: several otherwise 5-year-complete NSE symbols were
    missing exactly their single most-recent day (a transient
    free-data-source lag), which used to disqualify an otherwise perfectly
    liquid stock from the whole portfolio. An isolated 1-day gap should now
    be forward-filled and the symbol kept."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    complete = pd.DataFrame({"symbol": "AAA", "date": dates, "close_adj": 100 + np.arange(100)})
    # BBB has three separate, non-consecutive single-day gaps.
    gappy_dates = dates.delete([10, 20, 30])
    gappy = pd.DataFrame({"symbol": "BBB", "date": gappy_dates, "close_adj": 50 + np.arange(len(gappy_dates))})
    prices = pd.concat([complete, gappy], ignore_index=True)

    returns = compute_returns_matrix(prices, ["AAA", "BBB"], lookback_days=90)
    assert "AAA" in returns.columns
    assert "BBB" in returns.columns
    assert not returns["BBB"].isna().any()


def test_compute_returns_matrix_still_drops_symbols_with_gaps_larger_than_the_fill_limit():
    dates = pd.bdate_range("2024-01-01", periods=100)
    complete = pd.DataFrame({"symbol": "AAA", "date": dates, "close_adj": 100 + np.arange(100)})
    # CCC is missing 5 consecutive days -- larger than the default max_gap_fill=2.
    gappy_dates = dates.delete([10, 11, 12, 13, 14])
    gappy = pd.DataFrame({"symbol": "CCC", "date": gappy_dates, "close_adj": 50 + np.arange(len(gappy_dates))})
    prices = pd.concat([complete, gappy], ignore_index=True)

    returns = compute_returns_matrix(prices, ["AAA", "CCC"], lookback_days=90)
    assert "AAA" in returns.columns
    assert "CCC" not in returns.columns  # gap too large to fill -- dropped, not fabricated


def test_compute_returns_matrix_respects_lookback_window():
    dates = pd.bdate_range("2024-01-01", periods=200)
    prices = pd.DataFrame({"symbol": "AAA", "date": dates, "close_adj": 100 + np.arange(200)})
    returns = compute_returns_matrix(prices, ["AAA"], lookback_days=30)
    assert len(returns) == 30
