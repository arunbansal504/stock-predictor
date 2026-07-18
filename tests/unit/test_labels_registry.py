from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.common.types import DataLayer
from stockpredictor.labels import registry


def _seed_prices(tmp_lake, symbol: str, closes: list[float]) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"symbol": [symbol] * len(closes), "date": dates, "close_adj": closes})
    tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])


def _seed_benchmark(tmp_lake, closes: list[float]) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"series": ["NIFTY500"] * len(closes), "date": dates, "close": closes})
    tmp_lake.write(df, DataLayer.SILVER, "macro", "NIFTY500", key_cols=["series", "date"])


def test_build_labels_for_universe_empty_when_no_prices(tmp_lake):
    out = registry.build_labels_for_universe(tmp_lake)
    assert out.empty


def test_build_labels_for_universe_raises_without_benchmark(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105])
    with pytest.raises(ValueError, match="Benchmark series"):
        registry.build_labels_for_universe(tmp_lake)


def test_build_labels_for_universe_combines_all_symbols(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105, 106, 107])
    _seed_prices(tmp_lake, "BBB", [50, 51, 49, 52, 53, 54, 55, 56])
    _seed_benchmark(tmp_lake, [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007])

    out = registry.build_labels_for_universe(tmp_lake, horizons={"2d": 2})
    assert set(out["symbol"].unique()) == {"AAA", "BBB"}
    assert set(out["horizon"]) == {"2d"}


def test_persist_labels_writes_to_gold_and_reads_back(tmp_lake):
    _seed_prices(tmp_lake, "AAA", [100, 101, 102, 103, 104, 105])
    _seed_benchmark(tmp_lake, [1000, 1001, 1002, 1003, 1004, 1005])

    labels = registry.build_labels_for_universe(tmp_lake, horizons={"2d": 2})
    rows = registry.persist_labels(tmp_lake, labels)
    assert rows == len(labels)

    gold = tmp_lake.read_all(DataLayer.GOLD, registry.GOLD_DOMAIN)
    assert len(gold) == len(labels)


def test_persist_labels_empty_returns_zero(tmp_lake):
    assert registry.persist_labels(tmp_lake, pd.DataFrame()) == 0
