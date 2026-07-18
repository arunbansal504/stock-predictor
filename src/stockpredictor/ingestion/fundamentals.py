"""Ingestion for annual fundamentals (§5, §7). Written straight to Silver,
same rationale as ingestion/macro.py: no split-adjustment or similar
transform applies to fundamental line items, so there's no bronze->silver
transform step to separate out -- the connector's PIT-stamped output *is*
the silver representation.
"""

from __future__ import annotations

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.connectors.fundamentals_yfinance import fetch_fundamentals
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

DOMAIN = "fundamentals"
KEY_COLS = ["symbol", "period_end"]


def ingest_symbol_fundamentals(lake: Lake, symbol: str, exchange: str = "NSE") -> int:
    """Fetch and store annual fundamentals for one symbol. Returns the
    resulting row count (0 if the fetch failed/was empty) -- callers (the
    orchestration DAG) use this to detect per-symbol gaps without the whole
    run failing, same convention as ingestion/prices.py."""
    df = fetch_fundamentals(symbol, exchange)
    if df.empty:
        return 0
    rows = lake.write(df, DataLayer.SILVER, DOMAIN, symbol, key_cols=KEY_COLS)
    logger.info("Ingested %d silver fundamentals rows for %s", rows, symbol)
    return rows
