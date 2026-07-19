"""Bronze -> Silver ingestion for daily prices (§5, §27 Phase 1 step 4).

Bronze = raw connector output, written as-is (the audit trail; never
mutated). Silver = PIT-stamped, deduplicated, and carries a single
`close_adj` column (split/dividend adjusted, from the provider's Adj Close)
that all downstream feature code must use for return calculations — using
unadjusted `close` for returns across a split/bonus is a classic, silent
correctness bug that would poison every momentum/return feature.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.connectors import prices_yfinance
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

BRONZE_KEY_COLS = ["symbol", "date", "source"]
SILVER_KEY_COLS = ["symbol", "date"]
SILVER_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "close_adj",
    "volume",
    "knowable_date",
]


def bronze_to_silver(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform, unit-testable without hitting the network or disk."""
    if bronze_df.empty:
        return pd.DataFrame(columns=SILVER_COLUMNS)
    df = bronze_df.copy()
    df["close_adj"] = df["adj_close"]
    # Same-day knowability: a daily close is knowable from that trading day
    # onward — event date == knowable date for prices (see common/pit.py).
    df["knowable_date"] = df["date"]
    return df[SILVER_COLUMNS]


def ingest_symbol_prices(
    lake: Lake,
    symbol: str,
    start: dt.date,
    end: dt.date,
    exchange: str = "NSE",
) -> int:
    """Fetch, bronze-write, and silver-transform prices for one symbol.
    Returns the resulting silver row count (0 if the fetch failed/was empty)
    — callers (the orchestration DAG) use this to detect per-symbol gaps
    without the whole run failing.

    If silver data already covers through `end`, skips the network call
    entirely and returns the existing row count. This is what makes a
    same-day manual rerun (after an earlier run already ingested through
    the pinned `end` -- see common/trading_calendar.py) produce byte-
    identical inputs: no new fetch means no chance of a different price
    revision or a different set of per-symbol failures reshuffling the
    universe between runs."""
    existing = lake.read(DataLayer.SILVER, "prices", symbol)
    if not existing.empty:
        existing_max = pd.to_datetime(existing["date"]).max().date()
        if existing_max >= end:
            logger.info(
                "Silver prices for %s already cover through %s (>= end=%s) — skipping fetch",
                symbol, existing_max, end,
            )
            return len(existing)

    bronze_df = prices_yfinance.fetch_prices([symbol], start, end, exchange)
    if bronze_df.empty:
        return 0

    # Defense in depth against a partial/live bar: yfinance's `end` bound is
    # a request hint, not a hard guarantee (a provider-side revision or an
    # in-progress session can still return a row dated after `end`). Every
    # caller here treats `end` as "the last completed session", so nothing
    # later may enter the lake regardless of what the connector returned —
    # see common/trading_calendar.py.
    bronze_df = bronze_df[pd.to_datetime(bronze_df["date"]) <= pd.Timestamp(end)]
    if bronze_df.empty:
        return 0

    lake.write(bronze_df, DataLayer.BRONZE, "prices", symbol, key_cols=BRONZE_KEY_COLS)

    silver_df = bronze_to_silver(bronze_df)
    rows = lake.write(silver_df, DataLayer.SILVER, "prices", symbol, key_cols=SILVER_KEY_COLS)
    logger.info("Ingested %d silver price rows for %s", rows, symbol)
    return rows
