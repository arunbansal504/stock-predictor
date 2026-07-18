from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.portfolio.sizing import apply_confidence_tilt, apply_position_cap, apply_sector_caps


def test_confidence_tilt_zero_equals_pure_hrp():
    hrp = pd.Series({"A": 0.7, "B": 0.3})
    scores = pd.Series({"A": 0.1, "B": 0.9})
    out = apply_confidence_tilt(hrp, scores, tilt_strength=0.0)
    assert out["A"] == pytest.approx(0.7)
    assert out["B"] == pytest.approx(0.3)


def test_confidence_tilt_one_equals_pure_score_weighting():
    hrp = pd.Series({"A": 0.5, "B": 0.5})
    scores = pd.Series({"A": 0.9, "B": 0.1})
    out = apply_confidence_tilt(hrp, scores, tilt_strength=1.0)
    assert out["A"] == pytest.approx(0.9)
    assert out["B"] == pytest.approx(0.1)


def test_confidence_tilt_always_sums_to_one():
    hrp = pd.Series({"A": 0.6, "B": 0.4})
    scores = pd.Series({"A": 0.3, "B": 0.7})
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = apply_confidence_tilt(hrp, scores, tilt_strength=strength)
        assert out.sum() == pytest.approx(1.0)


def test_position_cap_leaves_uncapped_weights_alone():
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    capped = apply_position_cap(w, max_weight=0.5)
    pd.testing.assert_series_equal(w, capped)


def test_position_cap_redistributes_excess_and_sums_to_one():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    capped = apply_position_cap(w, max_weight=0.35)
    assert (capped <= 0.35 + 1e-9).all()
    assert capped.sum() == pytest.approx(1.0)


def test_position_cap_handles_cascading_violations():
    """Redistributing one position's excess can push another position over
    the cap -- the function must iterate until stable, not just do one pass.
    Uses 4 assets at a 0.3 cap (max achievable sum 1.2 > 1.0) so full
    redistribution to 1.0 is actually mathematically possible -- unlike 3
    assets at a 0.3 cap, which has a hard ceiling of 0.9 no matter what."""
    w = pd.Series({"A": 0.6, "B": 0.25, "C": 0.1, "D": 0.05})
    capped = apply_position_cap(w, max_weight=0.3)
    assert (capped <= 0.3 + 1e-9).all()
    assert capped.sum() == pytest.approx(1.0)


def test_position_cap_all_at_cap_sums_to_less_than_one():
    """If every position is already at the cap, there's nowhere to
    redistribute excess to -- weights should honestly sum below 1 rather
    than silently violate the cap."""
    w = pd.Series({"A": 0.5, "B": 0.5})
    capped = apply_position_cap(w, max_weight=0.4)
    assert (capped <= 0.4 + 1e-9).all()
    assert capped.sum() < 1.0


def test_sector_caps_redistributes_excess_to_other_sectors():
    w = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    sectors = pd.Series({"A": "IT", "B": "IT", "C": "Financials"})
    capped = apply_sector_caps(w, sectors, max_sector_weight=0.5)

    it_total = capped[["A", "B"]].sum()
    assert it_total == pytest.approx(0.5)
    assert capped["C"] == pytest.approx(0.5)
    assert capped.sum() == pytest.approx(1.0)


def test_sector_caps_leaves_compliant_allocation_alone():
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    sectors = pd.Series({"A": "IT", "B": "Financials", "C": "Energy"})
    capped = apply_sector_caps(w, sectors, max_sector_weight=0.5)
    pd.testing.assert_series_equal(w, capped)


def test_sector_caps_within_sector_proportions_preserved():
    """Capping a sector should scale its members down proportionally, not
    change their relative weighting within the sector."""
    w = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})  # A:B ratio 3:1 within IT
    sectors = pd.Series({"A": "IT", "B": "IT", "C": "Financials"})
    capped = apply_sector_caps(w, sectors, max_sector_weight=0.4)
    assert capped["A"] / capped["B"] == pytest.approx(3.0)
