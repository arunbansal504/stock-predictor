from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.portfolio.sizing import (
    apply_confidence_tilt,
    apply_position_and_sector_caps,
    apply_position_cap,
    apply_sector_caps,
)


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


def _both_caps_hold(weights: pd.Series, sectors: pd.Series, max_position_weight: float, max_sector_weight: float) -> bool:
    position_ok = (weights <= max_position_weight + 1e-9).all()
    sector_ok = (weights.groupby(sectors).sum() <= max_sector_weight + 1e-9).all()
    return position_ok and sector_ok


def test_position_and_sector_caps_regression_sequential_application_would_violate_sector_cap():
    """Regression test for a real bug: applying apply_sector_caps then
    apply_position_cap in a fixed sequence (the previous implementation)
    left the sector cap silently violated -- observed live, three
    Financial Services positions each independently satisfying a 15%
    position cap but summing to 45% against a 35% sector cap, because
    position-cap redistribution ran last and wasn't sector-aware. This
    reproduces that exact shape (3 same-sector + 1 other-sector, both caps
    tight enough to interact) and asserts BOTH caps hold in the result."""
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.1})
    sectors = pd.Series({"A": "Financials", "B": "Financials", "C": "Financials", "D": "Energy"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.15, max_sector_weight=0.35)
    assert _both_caps_hold(capped, sectors, 0.15, 0.35)
    # The mathematically correct joint ceiling here: Financials capped at
    # 0.35 total (proportionally split, all equal so 0.35/3 each), Energy's
    # D at its own 0.15 position cap -- total 0.50, not silently higher.
    assert capped["A"] == pytest.approx(0.35 / 3)
    assert capped["D"] == pytest.approx(0.15)
    assert capped.sum() == pytest.approx(0.50)


def test_position_and_sector_caps_does_not_falsely_converge_to_a_two_cycle():
    """A naive 'alternate position-cap and sector-cap until the composed
    pass is a no-op' loop can settle into a 2-cycle where the *composition*
    looks stable even though neither individual constraint holds on that
    state (verified by hand-tracing the case above: alternating would
    oscillate between [0.15]*4 and [0.1167,0.1167,0.1167,0.25], and a
    before/after-of-the-full-pass equality check can't tell those two
    states apart from a genuine fixed point). Directly asserts the
    constraints, not just single-pass stability, to guard against that."""
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.1})
    sectors = pd.Series({"A": "Financials", "B": "Financials", "C": "Financials", "D": "Energy"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.15, max_sector_weight=0.35)
    assert capped["D"] == pytest.approx(0.15), "D must land at its own position cap, not the mis-redistributed 0.25"


def test_position_and_sector_caps_leaves_compliant_allocation_alone():
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    sectors = pd.Series({"A": "IT", "B": "Financials", "C": "Energy"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.5, max_sector_weight=0.5)
    pd.testing.assert_series_equal(w, capped)


def test_position_and_sector_caps_only_position_cap_binds():
    """When sectors are diverse enough that the sector cap never engages,
    the joint function should behave like apply_position_cap alone."""
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    sectors = pd.Series({"A": "S1", "B": "S2", "C": "S3"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.35, max_sector_weight=0.9)
    assert _both_caps_hold(capped, sectors, 0.35, 0.9)
    assert capped.sum() == pytest.approx(1.0)


def test_position_and_sector_caps_only_sector_cap_binds():
    """When no single position is anywhere near the position cap, the
    joint function should behave like apply_sector_caps alone."""
    w = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    sectors = pd.Series({"A": "IT", "B": "IT", "C": "Financials"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.9, max_sector_weight=0.5)
    assert _both_caps_hold(capped, sectors, 0.9, 0.5)
    it_total = capped[["A", "B"]].sum()
    assert it_total == pytest.approx(0.5)


def test_position_and_sector_caps_honest_shortfall_when_infeasible():
    """Every position pinned at its own cap, with no room left anywhere --
    weights must honestly sum below 1 rather than silently violate either
    cap (same convention as the single-constraint functions)."""
    w = pd.Series({"A": 0.5, "B": 0.5})
    sectors = pd.Series({"A": "S1", "B": "S1"})
    capped = apply_position_and_sector_caps(w, sectors, max_position_weight=0.4, max_sector_weight=0.4)
    assert _both_caps_hold(capped, sectors, 0.4, 0.4)
    assert capped.sum() < 1.0
