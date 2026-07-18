from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.features.cross_sectional import add_cross_sectional_rank, add_cross_sectional_zscore


def _panel() -> pd.DataFrame:
    # Two dates, three symbols each -- a small but real cross-section.
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "rsi_14": [30.0, 60.0, 90.0, 10.0, 50.0, 95.0],
        }
    )


def test_cross_sectional_rank_is_percentile_within_date_group():
    out = add_cross_sectional_rank(_panel(), ["rsi_14"])
    day1 = out[out["date"] == "2024-01-01"].set_index("symbol")
    # 3 values ranked -> percentiles 1/3, 2/3, 3/3
    assert day1.loc["A", "rsi_14_xrank"] == pytest.approx(1 / 3)
    assert day1.loc["B", "rsi_14_xrank"] == pytest.approx(2 / 3)
    assert day1.loc["C", "rsi_14_xrank"] == pytest.approx(3 / 3)


def test_cross_sectional_rank_is_independent_per_date():
    out = add_cross_sectional_rank(_panel(), ["rsi_14"])
    day2 = out[out["date"] == "2024-01-02"].set_index("symbol")
    # Day 2 has different relative ordering (A now lowest, C highest) --
    # ranks must be computed fresh per date, not globally.
    assert day2.loc["A", "rsi_14_xrank"] == pytest.approx(1 / 3)
    assert day2.loc["C", "rsi_14_xrank"] == pytest.approx(3 / 3)


def test_cross_sectional_zscore_has_zero_mean_within_date_group():
    out = add_cross_sectional_zscore(_panel(), ["rsi_14"])
    day1 = out[out["date"] == "2024-01-01"]
    assert day1["rsi_14_xz"].mean() == pytest.approx(0.0, abs=1e-9)


def test_cross_sectional_rank_handles_nan_gracefully():
    df = _panel()
    df.loc[0, "rsi_14"] = float("nan")
    out = add_cross_sectional_rank(df, ["rsi_14"])
    assert pd.isna(out.loc[0, "rsi_14_xrank"])
    # Remaining non-NaN values in that date group still get ranked.
    day1 = out[out["date"] == "2024-01-01"]
    assert day1["rsi_14_xrank"].notna().sum() == 2
