from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.backtest.calibration_curve import compute_decile_return_calibration, lookup_expected_return


def test_compute_decile_calibration_is_monotonic_for_informative_scores():
    scores = pd.Series(range(100)).astype(float) / 100
    returns = scores * 0.5  # perfectly monotonic relationship
    calib = compute_decile_return_calibration(scores, returns, n_deciles=10)

    assert len(calib) == 10
    assert calib["mean_return"].is_monotonic_increasing


def test_compute_decile_calibration_empty_input():
    out = compute_decile_return_calibration(pd.Series(dtype="float64"), pd.Series(dtype="float64"))
    assert out.empty


def test_compute_decile_calibration_drops_nan_pairs():
    scores = pd.Series([0.1, 0.2, None, 0.4])
    returns = pd.Series([0.01, None, 0.03, 0.04])
    calib = compute_decile_return_calibration(scores, returns, n_deciles=2)
    # Only rows (0.1, 0.01) and (0.4, 0.04) have both values present.
    assert calib["n_obs"].sum() == 2


def test_lookup_expected_return_within_range():
    calib = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.05], "median_return": [0.01, 0.05], "n_obs": [10, 10]}
    )
    assert lookup_expected_return(0.2, calib) == pytest.approx(0.01)
    assert lookup_expected_return(0.7, calib) == pytest.approx(0.05)


def test_lookup_expected_return_extrapolates_to_nearest_when_out_of_range():
    calib = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.2, 0.5], "score_max": [0.49, 0.8], "mean_return": [0.01, 0.05], "median_return": [0.01, 0.05], "n_obs": [10, 10]}
    )
    assert lookup_expected_return(0.95, calib) == pytest.approx(0.05)  # nearest to decile 1
    assert lookup_expected_return(0.05, calib) == pytest.approx(0.01)  # nearest to decile 0


def test_lookup_expected_return_none_when_no_calibration_data():
    assert lookup_expected_return(0.5, pd.DataFrame(columns=["score_min", "score_max", "mean_return"])) is None
