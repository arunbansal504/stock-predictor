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

import datetime as dt

import pandas as pd

from stockpredictor.common.pit import filter_as_of
from stockpredictor.common.types import DataLayer
from stockpredictor.features.registry import GOLD_DOMAIN as FEATURES_DOMAIN
from stockpredictor.models.dataset import build_training_frame, get_feature_columns
from stockpredictor.models.ensemble import StackedRanker
from stockpredictor.storage.lake import Lake


def get_latest_feature_snapshot(lake: Lake, as_of: dt.date | None = None) -> pd.DataFrame:
    """Every symbol's feature row for the single most recent date -- one
    coherent cross-section, which is what a live run would actually score.
    No label is needed here (that's the point: it's the future we're trying
    to predict).

    Deliberately NOT "the most recent row per symbol": `_xrank` columns
    (features/cross_sectional.py) are percentiles computed within each
    date's cross-section, so a symbol whose latest bar is a stale date
    would carry an `_xrank` computed against a *different* day's universe
    -- mixing cross-sections that don't mean the same thing. A symbol with
    no bar on the latest date is honestly excluded from that day's ranking
    rather than scored on stale data.

    `as_of`, when given, drops feature rows after that date first -- same
    rationale as features/registry.py's `as_of` (common/trading_calendar.py):
    a stale partial bar already on disk can't be picked up as "latest"."""
    features = lake.read_all(DataLayer.GOLD, FEATURES_DOMAIN)
    if features.empty:
        return features
    if as_of is not None:
        features = filter_as_of(features, pd.Timestamp(as_of))
        if features.empty:
            return features
    latest_date = features["date"].max()
    return (
        features[features["date"] == latest_date]
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )


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


def score_universe(
    lake: Lake, horizon: str, random_state: int = 42, as_of: dt.date | None = None
) -> pd.DataFrame:
    """Train a production model for `horizon` and score the latest feature
    snapshot for every symbol with feature history. Returns symbol, date,
    horizon, score (calibrated probability of outperformance), disagreement
    (§6: ensemble-disagreement component of confidence)."""
    model, feature_cols = train_production_model(lake, horizon, random_state)
    snapshot = get_latest_feature_snapshot(lake, as_of=as_of)
    if snapshot.empty:
        return pd.DataFrame()

    X = snapshot[feature_cols]
    # meta_score computed once and reused for the calibrated score, rather
    # than also calling model.predict_proba(X) and redoing the same
    # base-learner inference -- see meta_score's docstring.
    meta_score = model.meta_score(X)
    disagreement = model.disagreement(X)
    separation = model.calibrator.separation_info(meta_score)

    return pd.DataFrame(
        {
            "symbol": snapshot["symbol"].values,
            "date": snapshot["date"].values,
            "horizon": horizon,
            "score": model.calibrator.transform(meta_score),
            "disagreement": disagreement,
            # Pre-calibration ranking tie-break -- see ranking/engine.py's
            # rank_universe docstring for why this is needed.
            "meta_score": meta_score,
            # Calibration-evidence columns -- see IsotonicCalibrator.separation_info.
            "separation_direction": separation["separation_direction"].to_numpy(),
            "separation_n": separation["n"].to_numpy(),
            "separation_empirical_rate": separation["empirical_rate"].to_numpy(),
            "separation_base_rate": separation["base_rate"].to_numpy(),
        }
    )
