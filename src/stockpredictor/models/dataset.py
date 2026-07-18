"""Assembles a model-ready dataset by joining Gold-layer features and labels
for one horizon (§27 step 8).
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.features.registry import GOLD_DOMAIN as FEATURES_DOMAIN
from stockpredictor.features.registry import TECHNICAL_FEATURE_COLUMNS
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.storage.lake import Lake

# Cross-sectional rank features are the model's default input (§7: "this is
# what makes it a *relative* ranking model and removes market-wide drift").
FEATURE_RANK_COLUMNS: list[str] = [f"{c}_xrank" for c in TECHNICAL_FEATURE_COLUMNS]


def build_training_frame(lake: Lake, horizon: str) -> pd.DataFrame:
    """Join Gold features and Gold labels on (symbol, date) for one horizon.

    Returns a frame with metadata columns (symbol, date, label_valid_date)
    plus feature columns (raw + cross-sectional rank) plus target columns
    (excess_return, outperform). Rows with an unresolved label are dropped --
    training data must have a real, resolved outcome by definition (an
    unresolved label isn't wrong to keep out; it just isn't training data
    yet)."""
    features = lake.read_all(DataLayer.GOLD, FEATURES_DOMAIN)
    labels = lake.read_all(DataLayer.GOLD, LABELS_DOMAIN)
    if features.empty or labels.empty:
        return pd.DataFrame()

    labels = labels[labels["horizon"] == horizon]
    if labels.empty:
        return pd.DataFrame()

    merged = features.merge(labels, on=["symbol", "date"], how="inner")
    merged = merged.dropna(subset=["outperform"])
    return merged.reset_index(drop=True)


def get_feature_columns(use_cross_sectional: bool = True) -> list[str]:
    """Which columns from the joined frame are model inputs."""
    return FEATURE_RANK_COLUMNS if use_cross_sectional else list(TECHNICAL_FEATURE_COLUMNS)
