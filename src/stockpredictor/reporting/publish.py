"""Weekly publish: freezes this week's official Top-N recommendation set
into `published_predictions` plus `predictions/YYYY-MM-DD.{csv,json}` (ML
Review Board spec Part 1).

Designed to run as its own GH Actions job (weekly_prediction.yml) on a fresh
checkout with no access to nightly's in-memory state: it rebuilds the
(cheap, technical-only, no FinBERT) feature layer and retrains from
committed silver/gold data rather than assuming `data/gold/features`
exists, since that domain is deliberately git-ignored (see .gitignore) and
only ever exists on the runner that just computed it. Because the
determinism fix (common/trading_calendar.py + deterministic LightGBM) makes
retraining on the same `last_completed_nse_session()` date byte-identical
to nightly's own run for that date, this recompute reproduces nightly's
ranking rather than risking disagreement with it -- see
orchestration/nightly_flow.py's task_predict_rank_explain, which this
mirrors.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.common.config import REPO_ROOT
from stockpredictor.common.logging import get_logger
from stockpredictor.common.trading_calendar import last_completed_nse_session
from stockpredictor.common.types import DataLayer
from stockpredictor.features.registry import (
    TECHNICAL_FEATURE_COLUMNS,
    build_technical_features_for_universe,
    persist_features,
)
from stockpredictor.features.sentiment import SENTIMENT_FEATURE_COLUMNS, build_sentiment_features_for_symbol
from stockpredictor.prediction.predict import get_latest_feature_snapshot, train_production_model
from stockpredictor.ranking.engine import (
    apply_ranking_filters,
    compute_liquidity_and_anomaly_flags,
    rank_universe,
)
from stockpredictor.ranking.engine import top_n as top_n_filter
from stockpredictor.reporting.versioning import git_commit_hash, model_version
from stockpredictor.storage.db import session_scope
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import PublishedPrediction

logger = get_logger(__name__)

PREDICTIONS_DIR = REPO_ROOT / "predictions"


def _safe_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _sentiment_snapshot(lake: Lake, symbol: str, as_of: dt.date) -> dict:
    """Best-effort sentiment feature values for one symbol as of `as_of`.
    Not a model input yet (see features/sentiment.py's docstring) -- stored
    here purely as an explainability/audit snapshot, honestly NaN-heavy for
    symbols with thin news coverage rather than fabricated."""
    prices = lake.read(DataLayer.SILVER, "prices", symbol)
    if prices.empty:
        return {c: None for c in SENTIMENT_FEATURE_COLUMNS}
    news = lake.read(DataLayer.SILVER, "news", symbol)
    sentiment = build_sentiment_features_for_symbol(prices, news)
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.normalize()
    row = sentiment[sentiment["date"] == pd.Timestamp(as_of)]
    if row.empty:
        return {c: None for c in SENTIMENT_FEATURE_COLUMNS}
    row = row.iloc[0]
    return {c: _safe_float(row.get(c)) for c in SENTIMENT_FEATURE_COLUMNS}


def _insert_new(session_factory: sessionmaker[Session], rows: list[PublishedPrediction]) -> pd.DataFrame:
    """Insert only the rows whose `prediction_id` isn't already published --
    the "never overwrite previous predictions" discipline (spec Part 1).
    Returns the newly-inserted rows as plain records (for file export),
    empty if every id already existed (a same-day rerun is then a safe
    no-op, not an error)."""
    if not rows:
        return pd.DataFrame()
    ids = [r.prediction_id for r in rows]
    with session_scope(session_factory) as session:
        existing = set(
            session.execute(
                select(PublishedPrediction.prediction_id).where(PublishedPrediction.prediction_id.in_(ids))
            ).scalars()
        )
        new_rows = [r for r in rows if r.prediction_id not in existing]
        session.add_all(new_rows)
        session.flush()
        records = [
            {
                "prediction_id": r.prediction_id,
                "prediction_date": r.prediction_date,
                "prediction_horizon": r.prediction_horizon,
                "stock_symbol": r.stock_symbol,
                "buy_price": r.buy_price,
                "prediction_probability": r.prediction_probability,
                "confidence": r.confidence,
                "rank": r.rank,
                "relative_strength": r.relative_strength,
                "disagreement": r.disagreement,
                "technical_features": r.technical_features,
                "sentiment_features": r.sentiment_features,
                "feature_vector": r.feature_vector,
                "model_version": r.model_version,
                "git_commit_hash": r.git_commit_hash,
            }
            for r in new_rows
        ]
    return pd.DataFrame(records)


def _export_files(published: pd.DataFrame, end: dt.date) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PREDICTIONS_DIR / f"{end:%Y-%m-%d}.csv"
    json_path = PREDICTIONS_DIR / f"{end:%Y-%m-%d}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(
            f"predictions/{end:%Y-%m-%d}.csv/.json already exist -- refusing to overwrite "
            "a previously published snapshot."
        )

    published.to_csv(csv_path, index=False)

    records = published.to_dict("records")
    for rec in records:
        for col in ("technical_features", "sentiment_features", "feature_vector"):
            rec[col] = json.loads(rec[col])
        rec["prediction_date"] = str(rec["prediction_date"])
    json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", csv_path, json_path)


def publish_weekly_predictions(
    lake: Lake,
    session_factory: sessionmaker[Session],
    horizon: str = "90d",
    top_k: int = 10,
) -> pd.DataFrame:
    """Rebuild features from committed silver data, train + score, and
    freeze the Top-`top_k` for `horizon` into `published_predictions` plus
    `predictions/{date}.csv/.json`. Returns the newly-published rows (empty
    if this date/horizon was already published -- reruns are safe no-ops)."""
    end = last_completed_nse_session()

    features = build_technical_features_for_universe(lake, as_of=end)
    if features.empty:
        raise ValueError("No feature data available to publish -- has silver/prices been ingested?")
    persist_features(lake, features)

    model, feature_cols = train_production_model(lake, horizon)
    snapshot = get_latest_feature_snapshot(lake, as_of=end)
    if snapshot.empty:
        raise ValueError(f"No feature snapshot available as of {end} to score")

    X = snapshot[feature_cols]
    meta_score = model.meta_score(X)
    disagreement = model.disagreement(X)
    scored = pd.DataFrame(
        {
            "symbol": snapshot["symbol"].values,
            "date": snapshot["date"].values,
            "horizon": horizon,
            "score": model.calibrator.transform(meta_score),
            "disagreement": disagreement,
            "meta_score": meta_score,
        }
    )

    flags = compute_liquidity_and_anomaly_flags(lake, as_of=end)
    filtered = apply_ranking_filters(scored, flags)
    if filtered.empty:
        raise ValueError("No symbols passed ranking filters -- nothing to publish")
    ranked = rank_universe(filtered)
    top = top_n_filter(ranked, top_k)

    top_symbols = set(top["symbol"])
    mask = snapshot["symbol"].isin(top_symbols)
    top_snapshot = snapshot.loc[mask].reset_index(drop=True)
    # SHAP explanations are deliberately NOT recomputed here: nightly already
    # persists them (gold/explanations, top_n_explain=20 per horizon -- a
    # superset of this top_k=10) and that domain, unlike gold/features, IS
    # git-committed (see .gitignore). explain.registry.read_explanations,
    # joined on (symbol, date, horizon), is the source of truth for Part 8 --
    # see reporting/review.py.

    disagreement_min, disagreement_max = float(disagreement.min()), float(disagreement.max())
    spread = disagreement_max - disagreement_min
    version, commit = model_version(), git_commit_hash()

    rows = []
    for _, rec in top.iterrows():
        symbol = rec["symbol"]
        snap_row = top_snapshot.loc[top_snapshot["symbol"] == symbol].iloc[0]

        technical = {c: _safe_float(snap_row.get(c)) for c in TECHNICAL_FEATURE_COLUMNS}
        sentiment = _sentiment_snapshot(lake, symbol, end)
        vector = {c: _safe_float(snap_row.get(c)) for c in feature_cols}

        norm_disagreement = (float(rec["disagreement"]) - disagreement_min) / spread if spread > 0 else 0.0
        confidence = 1.0 - norm_disagreement

        rows.append(
            PublishedPrediction(
                prediction_id=f"{end:%Y%m%d}-{horizon}-{symbol}",
                prediction_date=end,
                prediction_horizon=horizon,
                stock_symbol=symbol,
                buy_price=float(rec["close_price"]),
                prediction_probability=float(rec["score"]),
                confidence=confidence,
                rank=int(rec["rank"]),
                # `relative_strength` = meta_score, the model's raw pre-
                # calibration signal -- matching the SAME meaning this term
                # already has in the Streamlit UI/USER_GUIDE.md glossary
                # (apps/streamlit_app/app.py renames meta_score to
                # "relative_strength" for display). Deliberately not a
                # second, different definition under the same name.
                relative_strength=_safe_float(rec.get("meta_score")),
                disagreement=float(rec["disagreement"]),
                technical_features=json.dumps(technical),
                sentiment_features=json.dumps(sentiment),
                feature_vector=json.dumps(vector),
                model_version=version,
                git_commit_hash=commit,
            )
        )

    published = _insert_new(session_factory, rows)
    if published.empty:
        logger.info(
            "All %d prediction_ids for %s/%s already published -- nothing new", len(rows), end, horizon
        )
        return published

    _export_files(published, end)
    logger.info("Published %d predictions for horizon=%s as of %s", len(published), horizon, end)
    return published
