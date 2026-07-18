"""Persistence for ranked tables (§13 `rankings` table).

Partitioned by horizon (one file per horizon, like macro series), since the
natural read pattern is "give me the whole ranked table for horizon=5d as of
the latest run" -- not one symbol's ranking history.
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

GOLD_DOMAIN = "rankings"
GOLD_KEY_COLS = ["symbol", "date", "horizon"]


def persist_rankings(lake: Lake, ranked: pd.DataFrame, horizon: str) -> int:
    if ranked.empty:
        return 0
    rows = lake.write(ranked, DataLayer.GOLD, GOLD_DOMAIN, horizon, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted rankings for horizon=%s (%d total rows on file)", horizon, rows)
    return rows


def read_rankings(lake: Lake, horizon: str) -> pd.DataFrame:
    return lake.read(DataLayer.GOLD, GOLD_DOMAIN, horizon)


def read_latest_rankings(lake: Lake, horizon: str) -> pd.DataFrame:
    """The most recent date's full ranked table for a horizon -- what the
    UI/API's Top-N screen actually queries."""
    df = read_rankings(lake, horizon)
    if df.empty:
        return df
    latest_date = df["date"].max()
    return df[df["date"] == latest_date].sort_values("rank").reset_index(drop=True)
