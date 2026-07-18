"""Persistence for SHAP-based explanations (§11, §13).

Explanation records carry nested objects (factor-block dict, positive/
negative signal lists) that are JSON-serialized to plain string columns
before writing to Parquet -- storing genuinely nested/variable-shape data in
Parquet across many small per-write files risks schema-inference mismatches
on concatenation (see storage/lake.py Lake.read_all), whereas a JSON string
column is universally safe and trivially round-tripped.

Computed once during the nightly batch for the Top-N ranked symbols (not
on-demand per API request) -- consistent with the "serving plane reads
pre-computed artifacts" principle in §4: explaining a symbol requires the
just-trained in-memory model, which the API/UI layer doesn't have.
"""

from __future__ import annotations

import json

import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

GOLD_DOMAIN = "explanations"
GOLD_KEY_COLS = ["symbol", "date", "horizon"]
_JSON_COLUMNS = ("factor_blocks", "top_positive_signals", "top_negative_signals")


def persist_explanations(
    lake: Lake, explanations: pd.DataFrame, date: pd.Timestamp, horizon: str
) -> int:
    """`explanations` must have the shape produced by
    explain.signals.explain_predictions (symbol, factor_blocks,
    top_positive_signals, top_negative_signals); `date` and `horizon` are
    attached here since explain_predictions is date/horizon-agnostic."""
    if explanations.empty:
        return 0
    df = explanations.copy()
    df["date"] = date
    df["horizon"] = horizon
    for col in _JSON_COLUMNS:
        df[col] = df[col].apply(json.dumps)

    rows = lake.write(df, DataLayer.GOLD, GOLD_DOMAIN, horizon, key_cols=GOLD_KEY_COLS)
    logger.info("Persisted %d explanation rows for horizon=%s", len(df), horizon)
    return rows


def read_explanations(lake: Lake, horizon: str) -> pd.DataFrame:
    """Read back explanations for a horizon, deserializing the JSON columns."""
    df = lake.read(DataLayer.GOLD, GOLD_DOMAIN, horizon)
    if df.empty:
        return df
    df = df.copy()
    for col in _JSON_COLUMNS:
        df[col] = df[col].apply(json.loads)
    return df
