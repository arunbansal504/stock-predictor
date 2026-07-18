"""Ingestion for macro/benchmark series (§5, §7 macro factor block).

Written straight to Silver (no bronze/silver split needed — a single daily
close has no adjustment step, unlike equity prices) using the lake's
per-symbol file convention with the series name standing in for `symbol`.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.connectors import macro_yfinance
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

KEY_COLS = ["series", "date"]
DOMAIN = "macro"


def ingest_macro_series(
    lake: Lake,
    series_names: list[str],
    start: dt.date,
    end: dt.date,
) -> int:
    """Fetch and store each requested macro/benchmark series. Returns total
    rows written across all series."""
    df = macro_yfinance.fetch_macro_series(series_names, start, end)
    if df.empty:
        return 0

    total = 0
    for name, group in df.groupby("series"):
        total += lake.write(group, DataLayer.SILVER, DOMAIN, name, key_cols=KEY_COLS)
    logger.info("Ingested macro series rows for %s", sorted(df["series"].unique()))
    return total


def read_macro_series(lake: Lake, series_name: str) -> pd.DataFrame:
    return lake.read(DataLayer.SILVER, DOMAIN, series_name)
