from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.backtest.calibration_curve import compute_return_calibration, lookup_expected_return


def test_compute_return_calibration_is_monotonic_for_informative_scores():
    scores = pd.Series(range(100)).astype(float) / 100
    returns = scores * 0.5  # perfectly monotonic relationship
    calib = compute_return_calibration(scores, returns)

    assert calib["mean_return"].is_monotonic_increasing


def test_compute_return_calibration_empty_input():
    out = compute_return_calibration(pd.Series(dtype="float64"), pd.Series(dtype="float64"))
    assert out.empty


def test_compute_return_calibration_drops_nan_pairs():
    scores = pd.Series([0.1, 0.2, None, 0.4])
    returns = pd.Series([0.01, None, 0.03, 0.04])
    calib = compute_return_calibration(scores, returns)
    # Only rows (0.1, 0.01) and (0.4, 0.04) have both values present.
    assert calib["n_obs"].sum() == 2


def test_compute_return_calibration_enforces_monotonicity_despite_noisy_raw_data():
    """Regression test for a real bug: the previous fixed-decile
    implementation reported each decile's raw (unconstrained) mean
    return independently, which could be non-monotonic from pure sampling
    noise -- observed live, a 90d backtest's lowest-scored decile showed a
    HIGHER historical mean return than its highest-scored decile. This
    fixture reproduces that shape (a low-score block with an outlier-driven
    high return, high-score blocks with lower, tightly-clustered returns)
    and asserts the calibration table's mean_return is non-decreasing
    regardless -- the isotonic fit must never expose that inversion to
    lookup_expected_return."""
    rng = np.random.default_rng(0)
    n = 300
    scores = np.sort(rng.uniform(0, 1, n))
    # True (noisy) relationship: mostly flat/weak, but the lowest-score
    # region gets a few large-return outliers -- exactly the kind of
    # small-sample noise that broke the old fixed-decile approach.
    returns = rng.normal(0.05, 0.02, n)
    returns[:20] += rng.uniform(0.5, 1.0, 20)  # outlier bump concentrated at the LOW-score end

    calib = compute_return_calibration(pd.Series(scores), pd.Series(returns))
    assert calib["mean_return"].is_monotonic_increasing, (
        "isotonic fit must guarantee non-decreasing mean_return even when the raw, "
        "unconstrained decile-level averages would not be monotonic"
    )


def test_lookup_expected_return_interpolates_between_block_centers():
    calib = pd.DataFrame(
        {
            "block": [0, 1],
            "score_min": [0.0, 0.5],
            "score_max": [0.49, 1.0],
            "mean_return": [0.01, 0.05],
            "median_return": [0.01, 0.05],
            "n_obs": [10, 10],
        }
    )
    # Centers: (0.0+0.49)/2=0.245, (0.5+1.0)/2=0.75.
    assert lookup_expected_return(0.245, calib) == pytest.approx(0.01)
    assert lookup_expected_return(0.75, calib) == pytest.approx(0.05)
    midpoint = lookup_expected_return(0.4975, calib)  # exactly between the two centers
    assert 0.01 < midpoint < 0.05


def test_lookup_expected_return_clamps_to_nearest_block_outside_range():
    calib = pd.DataFrame(
        {
            "block": [0, 1],
            "score_min": [0.2, 0.5],
            "score_max": [0.49, 0.8],
            "mean_return": [0.01, 0.05],
            "median_return": [0.01, 0.05],
            "n_obs": [10, 10],
        }
    )
    assert lookup_expected_return(0.95, calib) == pytest.approx(0.05)  # beyond the last center
    assert lookup_expected_return(0.05, calib) == pytest.approx(0.01)  # before the first center


def test_lookup_expected_return_never_inverts_monotonic_blocks():
    """A higher score must never map to a lower expected_return than a
    lower score when looking up against an (isotonic, non-decreasing)
    calibration table -- the whole point of the fix."""
    calib = pd.DataFrame(
        {
            "block": [0, 1, 2],
            "score_min": [0.0, 0.4, 0.7],
            "score_max": [0.39, 0.69, 1.0],
            "mean_return": [0.02, 0.05, 0.09],
            "median_return": [0.02, 0.05, 0.09],
            "n_obs": [50, 50, 50],
        }
    )
    grid = np.linspace(0.0, 1.0, 50)
    values = [lookup_expected_return(s, calib) for s in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_lookup_expected_return_none_when_no_calibration_data():
    assert lookup_expected_return(0.5, pd.DataFrame(columns=["score_min", "score_max", "mean_return"])) is None
