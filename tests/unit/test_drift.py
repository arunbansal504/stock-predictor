from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.monitoring.drift import (
    check_and_update_baseline,
    check_drift,
    compute_feature_stats,
    load_baseline,
)


def test_compute_feature_stats_matches_manual_mean_std():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    out = compute_feature_stats(df, ["f1", "f2"])
    out = out.set_index("feature")
    assert out.loc["f1", "mean"] == pytest.approx(2.0)
    assert out.loc["f1", "std"] == pytest.approx(df["f1"].std())


def test_check_drift_flags_large_mean_shift():
    baseline = pd.DataFrame({"feature": ["f1"], "mean": [0.0], "std": [1.0]})
    current = pd.DataFrame({"feature": ["f1"], "mean": [10.0], "std": [1.0]})  # 10 std shift
    out = check_drift(current, baseline, z_threshold=3.0)
    assert out.loc[0, "drifted"] == True  # noqa: E712


def test_check_drift_does_not_flag_small_shift():
    baseline = pd.DataFrame({"feature": ["f1"], "mean": [0.0], "std": [1.0]})
    current = pd.DataFrame({"feature": ["f1"], "mean": [0.5], "std": [1.0]})  # 0.5 std shift
    out = check_drift(current, baseline, z_threshold=3.0)
    assert out.loc[0, "drifted"] == False  # noqa: E712


def test_check_drift_handles_feature_missing_from_baseline():
    baseline = pd.DataFrame({"feature": ["f1"], "mean": [0.0], "std": [1.0]})
    current = pd.DataFrame({"feature": ["f1", "f2_new"], "mean": [0.0, 5.0], "std": [1.0, 2.0]})
    out = check_drift(current, baseline, z_threshold=3.0).set_index("feature")
    assert out.loc["f2_new", "drifted"] == False  # noqa: E712 -- nothing to compare against


def test_check_drift_handles_zero_baseline_std():
    baseline = pd.DataFrame({"feature": ["f1"], "mean": [0.0], "std": [0.0]})
    current = pd.DataFrame({"feature": ["f1"], "mean": [1.0], "std": [1.0]})
    out = check_drift(current, baseline)  # must not raise ZeroDivisionError
    assert out.loc[0, "drifted"] == False  # noqa: E712 -- undefined shift treated as not-drifted


def test_check_and_update_baseline_first_run_establishes_baseline_returns_none(tmp_lake):
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    result = check_and_update_baseline(tmp_lake, df, ["f1"])
    assert result is None
    assert not load_baseline(tmp_lake).empty


def test_check_and_update_baseline_second_run_compares_against_first(tmp_lake):
    df1 = pd.DataFrame({"f1": [0.0, 0.0, 0.0]})
    check_and_update_baseline(tmp_lake, df1, ["f1"])

    df2 = pd.DataFrame({"f1": [100.0, 100.0, 100.0]})  # huge shift, baseline std was 0 -> nan-safe
    result = check_and_update_baseline(tmp_lake, df2, ["f1"])
    assert result is not None
    assert "drifted" in result.columns
