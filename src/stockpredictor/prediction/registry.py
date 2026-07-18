"""Persistence for scored predictions (§13 `predictions` table)."""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

GOLD_DOMAIN = "predictions"
GOLD_KEY_COLS = ["symbol", "date", "horizon"]


def persist_predictions(lake: Lake, scored: pd.DataFrame) -> int:
    if scored.empty:
        return 0
    total = 0
    for symbol, group in scored.groupby("symbol"):
        total += lake.write(group, DataLayer.GOLD, GOLD_DOMAIN, symbol, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted %d prediction rows", total)
    return total
