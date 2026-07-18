"""Connector interface + a lightweight contract-test harness.

Architecture doc §5: every external source sits behind a common adapter
interface so swapping/adding sources doesn't touch downstream code, and each
one gets a contract test that catches silent schema/endpoint drift (free
feeds break without notice).
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import pandas as pd


class PriceConnector(Protocol):
    """A connector that can fetch daily OHLCV bars for a set of symbols."""

    def fetch_prices(self, symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame: ...


# Schema every price connector's bronze output must satisfy. Downstream
# ingestion (ingestion/prices.py) and tests/contract/ both check against this
# single source of truth, so a provider-side change fails loudly here first.
PRICE_BRONZE_COLUMNS: dict[str, str] = {
    "symbol": "object",
    "date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "int64",
    "source": "object",
}


def validate_schema(df: pd.DataFrame, expected: dict[str, str], context: str) -> None:
    """Raise a clear error the moment a source's output stops matching what
    downstream code expects, instead of letting a malformed frame silently
    propagate into features/predictions."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"[{context}] missing expected columns: {sorted(missing)}")

    for col, dtype in expected.items():
        actual = str(df[col].dtype)
        base_expected = dtype.split("[")[0]
        if base_expected == "int64" and actual in ("int32", "int64"):
            continue
        if base_expected == "float64" and actual in ("float32", "float64"):
            continue
        # pandas 3.0 defaults plain string columns to dtype "str" instead of
        # "object" (PDEP-14) -- both mean "string-like" for our purposes.
        if base_expected == "object" and actual in ("object", "str"):
            continue
        if not actual.startswith(base_expected):
            raise ValueError(f"[{context}] column '{col}' has dtype {actual}, expected {dtype}")
