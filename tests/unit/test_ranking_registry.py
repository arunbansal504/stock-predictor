from __future__ import annotations

import pandas as pd

from stockpredictor.ranking.registry import persist_rankings, read_latest_rankings, read_rankings


def _ranked(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": pd.to_datetime([date, date]),
            "horizon": ["5d", "5d"],
            "score": [0.9, 0.7],
            "rank": [1, 2],
        }
    )


def test_persist_and_read_rankings_roundtrip(tmp_lake):
    rows = persist_rankings(tmp_lake, _ranked("2024-01-01"), horizon="5d")
    assert rows == 2

    out = read_rankings(tmp_lake, "5d")
    assert len(out) == 2


def test_persist_rankings_empty_returns_zero(tmp_lake):
    assert persist_rankings(tmp_lake, pd.DataFrame(), horizon="5d") == 0


def test_read_latest_rankings_returns_only_most_recent_date(tmp_lake):
    persist_rankings(tmp_lake, _ranked("2024-01-01"), horizon="5d")
    persist_rankings(tmp_lake, _ranked("2024-02-01"), horizon="5d")

    latest = read_latest_rankings(tmp_lake, "5d")
    assert (latest["date"] == pd.Timestamp("2024-02-01")).all()
    assert list(latest["rank"]) == [1, 2]


def test_read_latest_rankings_empty_when_no_data(tmp_lake):
    assert read_latest_rankings(tmp_lake, "5d").empty
