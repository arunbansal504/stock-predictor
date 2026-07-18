from __future__ import annotations

import numpy as np
import pytest

from stockpredictor.models.calibration import IsotonicCalibrator


def test_transform_before_fit_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.transform(np.array([0.5]))


def test_fit_transform_output_is_bounded_and_monotonic():
    rng = np.random.default_rng(0)
    raw_scores = rng.uniform(0, 1, 500)
    # y_true probability increases with raw_scores -- a genuinely calibratable signal.
    y_true = rng.uniform(0, 1, 500) < raw_scores

    cal = IsotonicCalibrator()
    calibrated = cal.fit_transform(raw_scores, y_true)

    assert (calibrated >= 0).all() and (calibrated <= 1).all()

    order = np.argsort(raw_scores)
    sorted_calibrated = calibrated[order]
    # Isotonic regression is non-decreasing by construction.
    assert (np.diff(sorted_calibrated) >= -1e-9).all()


def test_calibrated_high_scores_correspond_to_higher_realized_rate():
    rng = np.random.default_rng(1)
    raw_scores = rng.uniform(0, 1, 1000)
    y_true = rng.uniform(0, 1, 1000) < raw_scores

    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    low = cal.transform(np.array([0.1]))[0]
    high = cal.transform(np.array([0.9]))[0]
    assert high > low
