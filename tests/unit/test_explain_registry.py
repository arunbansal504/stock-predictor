from __future__ import annotations

import pandas as pd

from stockpredictor.explain.registry import persist_explanations, read_explanations


def _explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "factor_blocks": {"Momentum/Trend": 0.1, "Oscillators": -0.02},
                "top_positive_signals": [{"feature": "return_5d", "block": "Momentum/Trend", "contribution": 0.1}],
                "top_negative_signals": [{"feature": "rsi_14", "block": "Oscillators", "contribution": -0.02}],
            }
        ]
    )


def test_persist_and_read_explanations_roundtrip_nested_objects(tmp_lake):
    rows = persist_explanations(tmp_lake, _explanations(), date=pd.Timestamp("2024-01-01"), horizon="5d")
    assert rows == 1

    out = read_explanations(tmp_lake, "5d")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["factor_blocks"] == {"Momentum/Trend": 0.1, "Oscillators": -0.02}
    assert row["top_positive_signals"][0]["feature"] == "return_5d"
    assert row["date"] == pd.Timestamp("2024-01-01")
    assert row["horizon"] == "5d"


def test_persist_explanations_empty_returns_zero(tmp_lake):
    assert persist_explanations(tmp_lake, pd.DataFrame(), date=pd.Timestamp("2024-01-01"), horizon="5d") == 0


def test_read_explanations_empty_when_no_data(tmp_lake):
    assert read_explanations(tmp_lake, "5d").empty
