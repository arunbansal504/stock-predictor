from __future__ import annotations

import math

import pandas as pd
import pytest

from stockpredictor.portfolio.targets import compute_stock_targets, compute_stop_loss_target


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
    result = compute_stock_targets(
        entry_price=100.0, atr=2.0, score=0.7, stop_multiplier=2.0, reward_risk_ratio=2.0, return_calibration=calibration
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
