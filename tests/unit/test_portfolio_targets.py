from __future__ import annotations

import math

import pandas as pd
import pytest

from stockpredictor.portfolio.targets import (
    MAX_REASONABLE_EXTRAPOLATION_MULTIPLE,
    compute_stock_targets,
    compute_stop_loss_target,
    estimate_return_for_days,
    extrapolation_warning,
)


def test_compute_stop_loss_target_basic_bracket():
    stop, target = compute_stop_loss_target(entry_price=100.0, atr=2.0, stop_multiplier=2.0, reward_risk_ratio=2.0)
    assert stop == pytest.approx(96.0)  # 100 - 2*2
    assert target == pytest.approx(108.0)  # 100 + 2*(2*2)


def test_compute_stop_loss_target_scales_with_reward_risk_ratio():
    _, target_1x = compute_stop_loss_target(100.0, atr=2.0, stop_multiplier=2.0, reward_risk_ratio=1.0)
    _, target_3x = compute_stop_loss_target(100.0, atr=2.0, stop_multiplier=2.0, reward_risk_ratio=3.0)
    assert target_3x - 100.0 == pytest.approx(3 * (target_1x - 100.0))


def test_compute_stop_loss_target_nan_atr_propagates_honest_nan():
    stop, target = compute_stop_loss_target(100.0, atr=float("nan"), stop_multiplier=2.0, reward_risk_ratio=2.0)
    assert math.isnan(stop)
    assert math.isnan(target)


def test_compute_stock_targets_combines_bracket_and_calibration():
    calibration = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.06], "median_return": [0.01, 0.06], "n_obs": [10, 10]}
    )
    # score=0.75 is exactly the second block's center ((0.5+1.0)/2) --
    # lookup_expected_return interpolates between block centers (see
    # test_calibration_curve.py for that math itself), so an exact score
    # here isolates "does calibration flow through correctly" (this test's
    # actual concern) from the interpolation arithmetic.
    result = compute_stock_targets(
        entry_price=100.0, atr=2.0, score=0.75, stop_multiplier=2.0, reward_risk_ratio=2.0, return_calibration=calibration
    )
    assert result.entry_price == 100.0
    assert result.stop_loss == pytest.approx(96.0)
    assert result.target_price == pytest.approx(108.0)
    assert result.expected_return == pytest.approx(0.06)


def test_compute_stock_targets_expected_return_none_without_calibration():
    result = compute_stock_targets(
        entry_price=100.0, atr=2.0, score=0.7, stop_multiplier=2.0, reward_risk_ratio=2.0,
        return_calibration=pd.DataFrame(),
    )
    assert result.expected_return is None


def test_estimate_return_for_days_is_a_no_op_when_n_days_equals_reference_horizon():
    calibration = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.06], "median_return": [0.01, 0.06], "n_obs": [10, 10]}
    )
    # score=0.75 is exactly the second block's center -- see the sibling
    # test above for why this test uses an exact-center score.
    result = estimate_return_for_days(score=0.75, return_calibration=calibration, n_days=5, reference_horizon_days=5)
    assert result == pytest.approx(0.06)


def test_estimate_return_for_days_scales_linearly_with_time():
    # Expected RETURN (the mean) scales linearly with time under the
    # random-walk assumption -- unlike volatility/Sharpe, which scale by
    # sqrt(time) (see constructor.py's expected_sharpe). Using sqrt-of-time
    # on the raw return itself would understate long-horizon extrapolations
    # (sqrt(4)=2 vs. the correct linear factor of 4).
    calibration = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.06], "median_return": [0.01, 0.06], "n_obs": [10, 10]}
    )
    result = estimate_return_for_days(score=0.75, return_calibration=calibration, n_days=20, reference_horizon_days=5)
    assert result == pytest.approx(0.06 * (20 / 5))


def test_estimate_return_for_days_none_without_calibration_data():
    result = estimate_return_for_days(score=0.7, return_calibration=pd.DataFrame(), n_days=10, reference_horizon_days=5)
    assert result is None


def test_estimate_return_for_days_rejects_non_positive_n_days():
    calibration = pd.DataFrame(
        {"decile": [0], "score_min": [0.0], "score_max": [1.0], "mean_return": [0.02], "median_return": [0.02], "n_obs": [10]}
    )
    with pytest.raises(ValueError):
        estimate_return_for_days(score=0.5, return_calibration=calibration, n_days=0, reference_horizon_days=5)
    with pytest.raises(ValueError):
        estimate_return_for_days(score=0.5, return_calibration=calibration, n_days=-3, reference_horizon_days=5)


def test_extrapolation_warning_none_within_reasonable_multiple():
    # Exactly at the threshold multiple (10x a 5d horizon = 50 days) should
    # still be considered reasonable -- the boundary is inclusive.
    assert extrapolation_warning(n_days=50, reference_horizon_days=5) is None
    assert extrapolation_warning(n_days=5, reference_horizon_days=5) is None
    assert extrapolation_warning(n_days=1, reference_horizon_days=5) is None  # 5x, within bounds


def test_extrapolation_warning_fires_far_above_reference_horizon():
    # 5000 days on a 5d curve is 1000x -- the exact scenario that motivated this guard.
    warning = extrapolation_warning(n_days=5000, reference_horizon_days=5)
    assert warning is not None
    assert "1000" in warning
    assert "5d" in warning


def test_extrapolation_warning_fires_far_below_reference_horizon_too():
    # Symmetric: n_days far *below* the reference horizon is just as much
    # an unevidenced extrapolation as one far above it.
    warning = extrapolation_warning(n_days=1, reference_horizon_days=90)
    assert warning is not None


def test_extrapolation_warning_boundary_just_over_threshold():
    just_over = MAX_REASONABLE_EXTRAPOLATION_MULTIPLE * 5 + 1  # 51 days on a 5d horizon
    assert extrapolation_warning(n_days=just_over, reference_horizon_days=5) is not None
