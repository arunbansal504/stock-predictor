from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.backtest.significance import (
    ic_autocorrelation,
    ic_bootstrap_ci,
    ic_subperiod_stability,
    ic_ttest,
    run_significance_report,
)


def _dated_series(values: np.ndarray) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=dates)


def test_ic_ttest_detects_a_real_signal():
    """A small, consistent positive IC with low noise across many periods
    should be clearly statistically significant -- the textbook case a
    t-test exists to catch."""
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.05, scale=0.02, size=300)
    result = ic_ttest(_dated_series(values))
    assert result["significant_at_5pct"] is True
    assert result["p_value"] < 0.01
    assert result["ci_low"] > 0  # confidence interval excludes zero


def test_ic_ttest_rejects_pure_noise():
    """Mean-zero noise should NOT come back significant -- this is the
    guard against reading a lucky positive mean IC as a real edge."""
    rng = np.random.default_rng(2)
    values = rng.normal(loc=0.0, scale=0.05, size=300)
    result = ic_ttest(_dated_series(values))
    assert result["significant_at_5pct"] is False
    assert result["ci_low"] < 0 < result["ci_high"]  # CI straddles zero


def test_ic_ttest_handles_too_few_periods_without_raising():
    result = ic_ttest(_dated_series(np.array([0.05])))
    assert result["n_periods"] == 1
    assert np.isnan(result["p_value"])
    assert result["significant_at_5pct"] is False

    empty_result = ic_ttest(pd.Series(dtype="float64"))
    assert empty_result["n_periods"] == 0
    assert np.isnan(empty_result["mean_ic"])


def test_ic_bootstrap_ci_matches_ttest_conclusion_for_a_clear_signal():
    rng = np.random.default_rng(3)
    values = rng.normal(loc=0.05, scale=0.02, size=300)
    result = ic_bootstrap_ci(_dated_series(values))
    assert result["ci_low"] > 0
    assert result["fraction_non_positive"] < 0.01


def test_ic_bootstrap_ci_matches_ttest_conclusion_for_noise():
    rng = np.random.default_rng(4)
    values = rng.normal(loc=0.0, scale=0.05, size=300)
    result = ic_bootstrap_ci(_dated_series(values))
    assert result["ci_low"] < 0 < result["ci_high"]
    # Not "roughly 0.5" -- a finite noise sample's own mean can land away
    # from the true zero by chance (a real property of bootstrapping, not
    # a bug), so this only asserts the bootstrap isn't confidently backing
    # either direction, consistent with the straddling-zero CI above.
    assert 0.01 < result["fraction_non_positive"] < 0.99


def test_ic_bootstrap_ci_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(5)
    values = rng.normal(loc=0.03, scale=0.02, size=100)
    series = _dated_series(values)
    result_a = ic_bootstrap_ci(series, seed=42)
    result_b = ic_bootstrap_ci(series, seed=42)
    assert result_a == result_b


def test_ic_autocorrelation_near_zero_for_iid_noise():
    rng = np.random.default_rng(6)
    values = rng.normal(0, 1, 500)
    autocorr = ic_autocorrelation(_dated_series(values))
    assert abs(autocorr) < 0.15  # i.i.d. noise should show little lag-1 structure


def test_ic_autocorrelation_high_for_a_slowly_drifting_series():
    # A smooth sine wave is maximally autocorrelated at lag 1 by construction.
    values = np.sin(np.linspace(0, 20 * np.pi, 500))
    autocorr = ic_autocorrelation(_dated_series(values))
    assert autocorr > 0.9


def test_ic_autocorrelation_nan_when_insufficient_history():
    assert np.isnan(ic_autocorrelation(_dated_series(np.array([0.1, 0.2]))))


def test_ic_subperiod_stability_flags_inconsistent_sign():
    values = np.concatenate([np.full(50, 0.08), np.full(50, -0.04)])
    subperiods = ic_subperiod_stability(_dated_series(values), n_splits=2)
    assert len(subperiods) == 2
    assert subperiods.iloc[0]["mean_ic"] > 0
    assert subperiods.iloc[1]["mean_ic"] < 0


def test_ic_subperiod_stability_consistent_sign_case():
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.03, scale=0.01, size=200)
    subperiods = ic_subperiod_stability(_dated_series(values), n_splits=4)
    assert len(subperiods) == 4
    assert (subperiods["mean_ic"] > 0).all()


def test_ic_subperiod_stability_empty_when_too_little_history():
    result = ic_subperiod_stability(_dated_series(np.array([0.1, 0.2])), n_splits=2)
    assert result.empty


def test_run_significance_report_bundles_everything_consistently():
    rng = np.random.default_rng(8)
    values = rng.normal(loc=0.02, scale=0.03, size=195)  # mirrors the observed live scale
    report = run_significance_report(_dated_series(values))

    assert set(report.keys()) == {
        "ttest", "bootstrap", "subperiods", "lag1_autocorrelation", "consistent_sign_across_subperiods",
    }
    assert report["ttest"]["n_periods"] == 195
    assert isinstance(report["consistent_sign_across_subperiods"], bool)


def test_run_significance_report_consistent_sign_none_when_not_enough_data():
    report = run_significance_report(_dated_series(np.array([0.1])))
    assert report["consistent_sign_across_subperiods"] is None
