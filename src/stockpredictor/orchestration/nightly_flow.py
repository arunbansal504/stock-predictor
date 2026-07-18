"""Nightly orchestration DAG (§12, §27 step 12): wires ingestion -> features
-> labels -> prediction -> ranking -> explanation into one Prefect flow.

Idempotent + resumable by construction, not by Prefect checkpointing: every
stage's storage write is an upsert-on-key (lake per-symbol Parquet files, DB
rows keyed by natural key -- see storage/lake.py, storage/models.py), so
re-running the whole flow for the same date range is always safe. Prefect's
job here is sequencing, logging, and retries -- not state management. A
`run_metadata` row per stage (orchestration/run_tracking.py) is the audit
trail (§13, §23), separate from and in addition to Prefect's own task state.
"""

from __future__ import annotations

import datetime as dt
import os

# We run Prefect with no separate server (its default ephemeral-local mode,
# matching the "no extra infra" MVP posture in §16/§17). Log-shipping to
# Prefect's own tracking API is therefore pointless overhead here -- it
# spins up a background async HTTP client whose shutdown can race process
# teardown in short-lived scripts/tests. Must be set before `prefect` is
# imported, since settings are read at import time.
os.environ.setdefault("PREFECT_LOGGING_TO_API_ENABLED", "false")

import pandas as pd
from prefect import flow, task

# Every task here takes a Lake/sessionmaker argument, neither of which is
# JSON/pickle-serializable -- Prefect's default cache policy tries to hash
# task inputs to key a result cache, which fails loudly (as a logged error,
# not a crash) on those objects. We don't need Prefect's own result caching
# anyway: idempotency already comes from each stage's upsert-on-key storage
# (see module docstring), not from re-using a cached task result.
from prefect.cache_policies import NO_CACHE

from stockpredictor.common.logging import get_logger
from stockpredictor.explain.registry import persist_explanations
from stockpredictor.explain.signals import explain_predictions
from stockpredictor.features.registry import build_technical_features_for_universe, persist_features
from stockpredictor.ingestion.corporate_actions import sync_corporate_actions
from stockpredictor.ingestion.macro import ingest_macro_series
from stockpredictor.ingestion.prices import ingest_symbol_prices
from stockpredictor.ingestion.universe import load_universe_csv, sync_universe, sync_universe_from_nse
from stockpredictor.labels.registry import build_labels_for_universe, persist_labels
from stockpredictor.monitoring.alerts import send_alert
from stockpredictor.monitoring.drift import check_and_update_baseline
from stockpredictor.monitoring.freshness import check_domain_freshness
from stockpredictor.monitoring.quality_gates import check_minimum_success_ratio, check_non_empty
from stockpredictor.orchestration.run_tracking import run_tracked_stage
from stockpredictor.prediction.predict import get_latest_feature_snapshot, train_production_model
from stockpredictor.prediction.registry import persist_predictions
from stockpredictor.ranking.engine import (
    apply_ranking_filters,
    compute_liquidity_and_anomaly_flags,
    rank_universe,
)
from stockpredictor.ranking.engine import top_n as top_n_filter
from stockpredictor.ranking.registry import persist_rankings
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake
from stockpredictor.common.types import DataLayer

logger = get_logger(__name__)

MIN_PRICE_SUCCESS_RATIO = 0.8
MAX_PRICE_STALENESS_DAYS = 5  # generous enough to cover a long weekend/holiday cluster
DRIFT_Z_THRESHOLD = 4.0  # deliberately conservative -- an alert, not a gate; false positives are costly to trust
DEFAULT_HORIZONS: dict[str, int] = {"5d": 5, "30d": 30, "90d": 90}
DEFAULT_BENCHMARK = "NIFTY500"


@task(name="sync_universe", cache_policy=NO_CACHE)
def task_sync_universe(sessionmaker) -> list[str]:
    """Prefer NSE's live, current NIFTY 500 membership; fall back to the
    bundled CSV seed if NSE is unreachable (§5: every free/unofficial
    source needs a documented fallback, not a hard pipeline failure)."""
    try:
        df = sync_universe_from_nse(sessionmaker)
        return sorted(df["symbol"].tolist())
    except Exception as exc:
        logger.warning("Live NSE universe fetch failed (%s) -- falling back to the bundled CSV seed", exc)
        send_alert(f"NSE universe fetch failed, using CSV fallback: {exc}", level="warning")
        sync_universe(sessionmaker)
        return sorted(load_universe_csv()["symbol"].tolist())


@task(name="ingest_prices_and_macro", retries=1, retry_delay_seconds=30, cache_policy=NO_CACHE)
def task_ingest_prices_and_macro(
    lake: Lake, symbols: list[str], start: dt.date, end: dt.date, benchmark: str
) -> int:
    succeeded = 0
    for symbol in symbols:
        if ingest_symbol_prices(lake, symbol, start, end) > 0:
            succeeded += 1
    ingest_macro_series(lake, [benchmark], start, end)
    check_minimum_success_ratio(succeeded, len(symbols), MIN_PRICE_SUCCESS_RATIO, stage="ingest_prices")
    return succeeded


@task(name="ingest_corporate_actions", retries=1, retry_delay_seconds=30, cache_policy=NO_CACHE)
def task_ingest_corporate_actions(sessionmaker, symbols: list[str]) -> int:
    return sync_corporate_actions(sessionmaker, symbols)


@task(name="check_freshness", cache_policy=NO_CACHE)
def task_check_freshness(lake: Lake) -> None:
    """Non-fatal by design (§23): a stale price feed can legitimately mean
    "market was closed" (weekend/holiday), not a broken source. Alert, don't
    abort -- an operator reviewing alerts can tell the difference; a hard
    gate here would false-positive-abort the pipeline every long weekend."""
    result = check_domain_freshness(lake, DataLayer.SILVER, "prices", MAX_PRICE_STALENESS_DAYS)
    if not result["ok"]:
        send_alert(
            f"Price data looks stale: latest={result['latest_date']}, "
            f"{result['staleness_days']} days old (threshold {MAX_PRICE_STALENESS_DAYS}).",
            level="warning",
        )
    else:
        logger.info("Freshness OK: latest price date %s", result["latest_date"])


@task(name="build_features", cache_policy=NO_CACHE)
def task_build_features(lake: Lake) -> int:
    from stockpredictor.features.registry import TECHNICAL_FEATURE_COLUMNS

    features = build_technical_features_for_universe(lake)
    check_non_empty(features, stage="build_features")
    rows = persist_features(lake, features)

    latest_date = features["date"].max()
    latest_snapshot = features[features["date"] == latest_date]
    drift = check_and_update_baseline(lake, latest_snapshot, TECHNICAL_FEATURE_COLUMNS, DRIFT_Z_THRESHOLD)
    if drift is not None:
        drifted = drift[drift["drifted"]]
        if not drifted.empty:
            send_alert(
                f"Feature drift detected in {len(drifted)} feature(s) vs. prior baseline: "
                f"{', '.join(drifted['feature'].tolist())}",
                level="warning",
            )
        else:
            logger.info("No feature drift detected (%d features checked)", len(drift))

    return rows


@task(name="build_labels", cache_policy=NO_CACHE)
def task_build_labels(lake: Lake, benchmark: str, horizons: dict[str, int]) -> int:
    labels = build_labels_for_universe(lake, benchmark_series=benchmark, horizons=horizons)
    check_non_empty(labels, stage="build_labels")
    return persist_labels(lake, labels)


@task(name="predict_rank_explain", cache_policy=NO_CACHE)
def task_predict_rank_explain(lake: Lake, horizon: str, top_k: int, top_n_explain: int) -> int:
    model, feature_cols = train_production_model(lake, horizon)
    snapshot = get_latest_feature_snapshot(lake)
    check_non_empty(snapshot, stage=f"predict[{horizon}]")

    X = snapshot[feature_cols]
    scored = pd.DataFrame(
        {
            "symbol": snapshot["symbol"].values,
            "date": snapshot["date"].values,
            "horizon": horizon,
            "score": model.predict_proba(X),
            "disagreement": model.disagreement(X),
        }
    )
    persist_predictions(lake, scored)

    flags = compute_liquidity_and_anomaly_flags(lake)
    filtered = apply_ranking_filters(scored, flags)
    check_non_empty(filtered, stage=f"rank[{horizon}]")
    ranked = rank_universe(filtered)
    persist_rankings(lake, ranked, horizon)

    top = top_n_filter(ranked, top_n_explain)
    mask = snapshot["symbol"].isin(set(top["symbol"]))
    explanations = explain_predictions(model, X.loc[mask], snapshot.loc[mask, "symbol"], n_signals=5)
    persist_explanations(lake, explanations, date=snapshot["date"].max(), horizon=horizon)

    logger.info(
        "horizon=%s: %d scored, %d passed filters, top-%d explained",
        horizon, len(scored), len(filtered), len(top),
    )
    return len(filtered)


@flow(name="nightly_pipeline")
def nightly_pipeline(
    years_of_history: int = 5,
    horizons: dict[str, int] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    top_k: int = 10,
    top_n_explain: int = 20,
) -> str:
    horizons = horizons or DEFAULT_HORIZONS
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")

    lake = Lake()
    engine = make_engine()
    init_db(engine)
    sessionmaker = make_sessionmaker(engine)

    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * years_of_history))

    try:
        symbols = run_tracked_stage(sessionmaker, run_id, "sync_universe", task_sync_universe, sessionmaker)
        run_tracked_stage(
            sessionmaker, run_id, "ingest_prices_and_macro",
            task_ingest_prices_and_macro, lake, symbols, start, end, benchmark,
        )
        run_tracked_stage(sessionmaker, run_id, "check_freshness", task_check_freshness, lake)
        run_tracked_stage(
            sessionmaker, run_id, "ingest_corporate_actions",
            task_ingest_corporate_actions, sessionmaker, symbols,
        )
        run_tracked_stage(sessionmaker, run_id, "build_features", task_build_features, lake)
        run_tracked_stage(sessionmaker, run_id, "build_labels", task_build_labels, lake, benchmark, horizons)

        for horizon in horizons:
            run_tracked_stage(
                sessionmaker, run_id, f"predict_rank_explain[{horizon}]",
                task_predict_rank_explain, lake, horizon, top_k, top_n_explain,
            )
    except Exception as exc:
        # run_tracked_stage already recorded which stage failed in
        # run_metadata (§13, §23) -- this alert is the "someone should look
        # at this" signal on top of that audit trail.
        send_alert(f"Nightly pipeline run {run_id} failed: {exc}", level="error")
        raise

    logger.info("Nightly pipeline run %s complete", run_id)
    return run_id


if __name__ == "__main__":
    nightly_pipeline()
