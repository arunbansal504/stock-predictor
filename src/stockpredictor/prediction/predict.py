"""Batch inference (§27 step 10): train on all currently-resolved history
and score the latest feature snapshot for the whole universe.

This is the "production" counterpart to models/walk_forward.py's
walk-forward CV: CV proves the ranking approach has out-of-sample skill
across history; this module trains once on everything resolved so far and
scores the decision an investor would actually face today. It reuses
StackedRanker as-is -- the base/meta split inside `fit` (see
models/ensemble.py) already keeps this leak-safe.
"""

from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.features.registry import GOLD_DOMAIN as FEATURES_DOMAIN
from stockpredictor.models.dataset import build_training_frame, get_feature_columns
from stockpredictor.models.ensemble import StackedRanker
from stockpredictor.storage.lake import Lake


def get_latest_feature_snapshot(lake: Lake) -> pd.DataFrame:
    """The most recent feature row per symbol -- what a live run would
    actually score. No label is needed here (that's the point: it's the
    future we're trying to predict)."""
    features = lake.read_all(DataLayer.GOLD, FEATURES_DOMAIN)
    if features.empty:
        return features
    features = features.sort_values("date")
    return features.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


def train_production_model(
    lake: Lake, horizon: str, random_state: int = 42
) -> tuple[StackedRanker, list[str]]:
    """Train on every currently-resolved row for `horizon` -- no
    walk-forward split, this is the single model that scores the latest
    snapshot, not a CV fold."""
    training_frame = build_training_frame(lake, horizon)
    if training_frame.empty:
        raise ValueError(f"No training data available for horizon '{horizon}'")

    feature_cols = get_feature_columns(use_cross_sectional=True)
    model = StackedRanker(random_state=random_state)
    model.fit(training_frame[feature_cols], training_frame["outperform"], training_frame["date"])
    return model, feature_cols


def score_universe(lake: Lake, horizon: str, random_state: int = 42) -> pd.DataFrame:
    """Train a production model for `horizon` and score the latest feature
    snapshot for every symbol with feature history. Returns symbol, date,
    horizon, score (calibrated probability of outperformance), disagreement
    (§6: ensemble-disagreement component of confidence)."""
    model, feature_cols = train_production_model(lake, horizon, random_state)
    snapshot = get_latest_feature_snapshot(lake)
    if snapshot.empty:
        return pd.DataFrame()

    X = snapshot[feature_cols]
    scores = model.predict_proba(X)
    disagreement = model.disagreement(X)

    return pd.DataFrame(
        {
            "symbol": snapshot["symbol"].values,
            "date": snapshot["date"].values,
            "horizon": horizon,
            "score": scores,
            "disagreement": disagreement,
        }
    )
