from __future__ import annotations

import numpy as np
import pytest

from stockpredictor.models.calibration import (
    SEPARATION_ALPHA,
    SEPARATION_NONE,
    SEPARATION_OUTPERFORM,
    SEPARATION_UNDERPERFORM,
    IsotonicCalibrator,
)


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


def test_separation_info_before_fit_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.separation_info(np.array([0.5]))


def test_fit_computes_base_rate_as_calibration_set_mean():
    rng = np.random.default_rng(3)
    raw_scores = rng.uniform(0, 1, 2000)
    y_true = (rng.uniform(0, 1, 2000) < 0.37).astype(int)  # deliberately off-0.5

    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    assert cal.base_rate == pytest.approx(y_true.mean())
    assert cal.base_rate != pytest.approx(0.5, abs=0.05), "fixture is deliberately off-0.5"


def test_base_rate_is_recomputed_on_refit_not_frozen():
    rng = np.random.default_rng(4)
    raw_scores = rng.uniform(0, 1, 2000)

    cal = IsotonicCalibrator()
    y_low = (rng.uniform(0, 1, 2000) < 0.30).astype(int)
    cal.fit(raw_scores, y_low)
    first_base_rate = cal.base_rate

    y_high = (rng.uniform(0, 1, 2000) < 0.65).astype(int)
    cal.fit(raw_scores, y_high)
    second_base_rate = cal.base_rate

    assert first_base_rate == pytest.approx(0.30, abs=0.05)
    assert second_base_rate == pytest.approx(0.65, abs=0.05)
    assert second_base_rate != first_base_rate


def _three_regime_off_center_fixture(seed: int = 13):
    """Synthetic calibration set (NOT pulled from the live lake -- 137K raw
    calibration rows is too large to check into a unit-test fixture) but
    calibrated to reproduce the real 5d model's own global base rate
    (~0.4774, per the base-rate investigation) rather than a fixed 0.5, so
    it actually exercises the corrected null. Three regions in increasing
    raw-score order (required so PAVA doesn't merge them to enforce
    monotonicity):

    - low raw scores, true rate 0.20 -- well below the base rate: confirmed
      UNDERPERFORMANCE.
    - mid raw scores, true rate 0.4774 (matches the real base rate exactly,
      large n) -- dominates the population and sets the overall base rate,
      genuinely indistinguishable from it: NONE.
    - high raw scores, true rate 0.49 -- note this is still BELOW 0.5, but
      above the ~0.477 base rate: this is the exact regression case the
      corrected null exists for. Under the old fixed-H0=0.5 test this block
      would have been (wrongly) flagged CONFIRMED UNDERPERFORMANCE just for
      sitting under 50%; under the corrected test it must flag OUTPERFORMANCE.
    """
    rng = np.random.default_rng(seed)

    n_low, p_low = 2000, 0.20
    raw_low = rng.uniform(0.1, 0.2, n_low)
    y_low = (rng.uniform(0, 1, n_low) < p_low).astype(int)

    n_base, p_base = 150_000, 0.4774
    raw_base = rng.uniform(0.3, 0.7, n_base)
    y_base = (rng.uniform(0, 1, n_base) < p_base).astype(int)

    n_target, p_target = 20_000, 0.49
    raw_target = rng.uniform(0.8, 0.95, n_target)
    y_target = (rng.uniform(0, 1, n_target) < p_target).astype(int)

    raw_scores = np.concatenate([raw_low, raw_base, raw_target])
    y_true = np.concatenate([y_low, y_base, y_target])
    return raw_scores, y_true


def test_separation_direction_uses_horizon_base_rate_not_fixed_half():
    raw_scores, y_true = _three_regime_off_center_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    assert cal.base_rate == pytest.approx(0.4774, abs=0.01)

    info = cal.separation_info(np.array([0.15, 0.5, 0.87]))
    under_row, base_row, target_row = info.iloc[0], info.iloc[1], info.iloc[2]

    assert under_row["separation_direction"] == SEPARATION_UNDERPERFORM
    assert 0.15 < under_row["empirical_rate"] < 0.25

    assert base_row["separation_direction"] == SEPARATION_NONE, (
        "a block whose rate matches this horizon's own base rate must NOT be "
        "flagged as having real separation -- that would fabricate confidence "
        "the data doesn't support"
    )
    assert base_row["p_value"] >= SEPARATION_ALPHA

    # The regression case: rate is BELOW 0.5, but ABOVE the horizon base
    # rate -- must be OUTPERFORM, not underperform, under the corrected null.
    assert target_row["empirical_rate"] < 0.5, "fixture sanity check: this rate is still below 0.5"
    assert target_row["empirical_rate"] > target_row["base_rate"], "fixture sanity check: but above base_rate"
    assert target_row["separation_direction"] == SEPARATION_OUTPERFORM, (
        f"rate={target_row['empirical_rate']:.4f} is below 0.5 but above this horizon's "
        f"base_rate={target_row['base_rate']:.4f} -- a fixed-H0=0.5 test would wrongly call "
        "this confirmed underperformance; the corrected test must call it outperformance"
    )


def test_separation_badge_never_gives_positive_styling_to_a_below_base_rate_block():
    """Regression test for the direction-blindness bug: a statistically
    significant BELOW-base_rate block (confirmed underperformance) must
    never be styled/labeled the same as an above-base_rate block. Sweeps a
    range of base rates, empirical rates below them, and n's -- not just
    one hand-picked example, and explicitly includes base rates far from
    0.5 (this module no longer assumes a fixed 0.5 null anywhere)."""
    for base_rate in (0.35, 0.4774, 0.5, 0.6):
        for rate in (base_rate - 0.30, base_rate - 0.10, base_rate - 0.01):
            for n in (10, 100, 10_000):
                badge = IsotonicCalibrator.separation_badge(SEPARATION_UNDERPERFORM, rate, n, base_rate)
                assert badge["style"] != "positive", (
                    f"rate={rate}, base_rate={base_rate}, n={n}: a confirmed-underperform "
                    "block must never render as positive/green"
                )
                assert "under" in badge["label"].lower()


def test_separation_badge_states_rate_relative_to_base_rate_not_alone():
    """The label must show the comparison (rate vs base_rate), not just the
    raw rate -- a rate like 49% is only self-explanatory once the reader
    knows whether this horizon's base rate is 50% or 40%."""
    badge = IsotonicCalibrator.separation_badge(SEPARATION_OUTPERFORM, 0.492, 9321, 0.47675)
    assert "49.2%" in badge["label"]
    assert "47.7%" in badge["label"]


def test_separation_badge_directions_map_to_distinct_styles():
    outperform = IsotonicCalibrator.separation_badge(SEPARATION_OUTPERFORM, 0.7, 1000, 0.4774)
    underperform = IsotonicCalibrator.separation_badge(SEPARATION_UNDERPERFORM, 0.3, 1000, 0.4774)
    none_ = IsotonicCalibrator.separation_badge(SEPARATION_NONE, 0.48, 1000, 0.4774)

    assert outperform["style"] == "positive"
    assert underperform["style"] == "negative"
    assert none_["style"] == "neutral"
    assert len({outperform["style"], underperform["style"], none_["style"]}) == 3


def _weak_signal_fixture(seed: int = 7, n: int = 2000):
    """A weak-but-real signal that PAVA pools into only a handful of
    blocks -- reproducing the live symptom (many raw scores collapsing
    onto few distinct block values) so CIR interpolation has something
    real to smooth between."""
    rng = np.random.default_rng(seed)
    raw_scores = rng.uniform(0.45, 0.56, n)
    y_true = (rng.uniform(0, 1, n) < (0.4 + 0.1 * (raw_scores - 0.45) / 0.11)).astype(int)
    return raw_scores, y_true


def test_transform_is_strictly_monotonic_within_block_center_range():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    centers = cal.block_stats["raw_center"].to_numpy()
    grid = np.linspace(centers.min(), centers.max(), 200)
    calibrated = cal.transform(grid)
    assert (np.diff(calibrated) > 0).all() or len(centers) == 1


def test_transform_gives_distinct_outputs_for_distinct_inputs_in_range():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    centers = cal.block_stats["raw_center"].to_numpy()
    if len(centers) < 2:
        pytest.skip("fixture collapsed to a single block")
    grid = np.linspace(centers.min(), centers.max(), 50)
    calibrated = cal.transform(grid)
    assert len(np.unique(calibrated)) == len(np.unique(grid)), (
        "distinct raw scores within the fitted center range must map to "
        "distinct calibrated scores under CIR interpolation -- this is the "
        "behavior that fixes the 'many stocks share one identical score' symptom"
    )


def test_transform_at_block_centers_equals_block_empirical_rate():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    blocks = cal.block_stats
    at_centers = cal.transform(blocks["raw_center"].to_numpy())
    np.testing.assert_allclose(at_centers, blocks["calibrated_score"].to_numpy())

    if len(blocks) >= 2:
        midpoint = (blocks["raw_center"].iloc[0] + blocks["raw_center"].iloc[1]) / 2
        mid_value = cal.transform(np.array([midpoint]))[0]
        lo, hi = sorted(blocks["calibrated_score"].iloc[:2])
        assert lo <= mid_value <= hi, "interpolated points must stay within their bracketing block rates"


def test_transform_degenerates_gracefully_for_a_single_pooled_block():
    rng = np.random.default_rng(11)
    raw_scores = np.sort(rng.uniform(0, 1, 500))
    # y strictly anti-monotonic in raw-score order (all 1s then all 0s) --
    # the one pattern PAVA is guaranteed to pool into a single flat block,
    # since any non-decreasing fit to a decreasing target must average out.
    half = len(raw_scores) // 2
    y_true = np.concatenate([np.ones(half, dtype=int), np.zeros(len(raw_scores) - half, dtype=int)])

    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)
    assert len(cal.block_stats) == 1, "fixture sanity check: must fully pool into one block"

    out = cal.transform(np.array([0.0, 0.3, 0.5, 0.7, 1.0]))
    assert np.allclose(out, out[0]), "a single pooled block must produce a constant (not crash)"
    assert out[0] == pytest.approx(cal.base_rate, abs=0.05)


def test_transform_clamps_outside_fitted_range_and_stays_bounded():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    blocks = cal.block_stats
    below = cal.transform(np.array([blocks["raw_center"].min() - 5.0]))[0]
    above = cal.transform(np.array([blocks["raw_center"].max() + 5.0]))[0]
    assert below == pytest.approx(blocks["calibrated_score"].iloc[0])
    assert above == pytest.approx(blocks["calibrated_score"].iloc[-1])

    extreme = cal.transform(np.array([-100.0, 100.0]))
    assert (extreme >= 0).all() and (extreme <= 1).all()


def test_transform_gives_equal_outputs_for_equal_raw_scores():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    out = cal.transform(np.array([0.5, 0.5, 0.5]))
    assert out[0] == out[1] == out[2]


def test_transform_no_longer_identically_equals_separation_empirical_rate():
    """The whole point of CIR: `transform` interpolates between block
    centers, so it diverges from `separation_info`'s block-anchored
    `empirical_rate` for query points that aren't exactly at a block
    center -- this is intentional, not a regression."""
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    blocks = cal.block_stats
    if len(blocks) < 2:
        pytest.skip("fixture collapsed to a single block")
    midpoint = (blocks["raw_center"].iloc[0] + blocks["raw_center"].iloc[1]) / 2
    score = cal.transform(np.array([midpoint]))[0]
    empirical_rate = cal.separation_info(np.array([midpoint]))["empirical_rate"].iloc[0]
    assert score != pytest.approx(empirical_rate)


def test_separation_info_unaffected_by_cir_interpolation():
    """`separation_info`/`block_stats` describe pure PAVA blocks and must
    not change shape or values because of the CIR transform change."""
    raw_scores, y_true = _three_regime_off_center_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    info = cal.separation_info(np.array([0.15, 0.5, 0.87]))
    assert list(info.columns) == ["n", "empirical_rate", "p_value", "separation_direction", "base_rate"]
    assert info.iloc[0]["separation_direction"] == SEPARATION_UNDERPERFORM
    assert info.iloc[2]["separation_direction"] == SEPARATION_OUTPERFORM


def test_block_stats_before_fit_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.block_stats


def test_block_stats_includes_raw_center_between_lo_and_hi():
    raw_scores, y_true = _weak_signal_fixture()
    cal = IsotonicCalibrator()
    cal.fit(raw_scores, y_true)

    blocks = cal.block_stats
    assert "raw_center" in blocks.columns
    assert (blocks["raw_center"] >= blocks["raw_lo"]).all()
    assert (blocks["raw_center"] <= blocks["raw_hi"]).all()
