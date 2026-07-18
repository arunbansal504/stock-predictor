from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.features import registry


def _synthetic_silver_prices(symbol: str, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "close_adj": closes,
            "volume": np.full(n, 1000, dtype="int64"),
            "knowable_date": dates,
        }
    )


def test_build_technical_features_for_universe_empty_when_no_prices(tmp_lake):
    out = registry.build_technical_features_for_universe(tmp_lake)
    assert out.empty


def test_build_technical_features_for_universe_produces_raw_and_ranked_columns(tmp_lake):
    for symbol, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        df = _synthetic_silver_prices(symbol, seed=seed)
        tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])

    matrix = registry.build_technical_features_for_universe(tmp_lake)
    assert set(matrix["symbol"].unique()) == {"AAA", "BBB", "CCC"}

    for col in registry.TECHNICAL_FEATURE_COLUMNS:
        assert col in matrix.columns
        assert f"{col}_xrank" in matrix.columns

    assert (matrix["feature_set_version"] == registry.FEATURE_SET_VERSION).all()


def test_persist_features_writes_to_gold_and_is_readable_back(tmp_lake):
    for symbol, seed in [("AAA", 1), ("BBB", 2)]:
        df = _synthetic_silver_prices(symbol, seed=seed)
        tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])

    matrix = registry.build_technical_features_for_universe(tmp_lake)
    rows = registry.persist_features(tmp_lake, matrix)
    assert rows == len(matrix)

    gold = tmp_lake.read_all(DataLayer.GOLD, registry.GOLD_DOMAIN)
    assert len(gold) == len(matrix)
    assert set(gold["symbol"].unique()) == {"AAA", "BBB"}


def test_persist_features_empty_matrix_writes_nothing(tmp_lake):
    assert registry.persist_features(tmp_lake, pd.DataFrame()) == 0
