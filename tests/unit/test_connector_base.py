from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.connectors.base import PRICE_BRONZE_COLUMNS, validate_schema


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": pd.to_datetime(["2024-01-01"]),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": pd.array([100], dtype="int64"),
            "source": ["yfinance"],
        }
    )


def test_validate_schema_passes_for_conforming_frame():
    validate_schema(_valid_frame(), PRICE_BRONZE_COLUMNS, context="test")  # no raise


def test_validate_schema_rejects_missing_column():
    df = _valid_frame().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(df, PRICE_BRONZE_COLUMNS, context="test")


def test_validate_schema_rejects_wrong_dtype():
    df = _valid_frame()
    df["close"] = df["close"].astype(str)  # should be float64
    with pytest.raises(ValueError, match="dtype"):
        validate_schema(df, PRICE_BRONZE_COLUMNS, context="test")


def test_validate_schema_accepts_float32_as_float64_compatible():
    df = _valid_frame()
    df["close"] = df["close"].astype("float32")
    validate_schema(df, PRICE_BRONZE_COLUMNS, context="test")  # no raise
