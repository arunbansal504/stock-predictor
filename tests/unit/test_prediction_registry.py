from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.prediction.registry import GOLD_DOMAIN, persist_predictions


def test_persist_predictions_writes_and_is_readable_back(tmp_lake):
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "horizon": ["5d", "5d"],
            "score": [0.6, 0.4],
            "disagreement": [0.1, 0.2],
        }
    )
    rows = persist_predictions(tmp_lake, df)
    assert rows == 2

    out = tmp_lake.read_all(DataLayer.GOLD, GOLD_DOMAIN)
    assert len(out) == 2
    assert set(out["symbol"]) == {"AAA", "BBB"}


def test_persist_predictions_empty_returns_zero(tmp_lake):
    assert persist_predictions(tmp_lake, pd.DataFrame()) == 0
