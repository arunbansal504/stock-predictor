"""First full historical run (§27 Phase 1 step 15, §29 "Definition of MVP
done"): wires universe -> prices -> macro benchmark -> technical features ->
labels -> walk-forward model training -> backtest into one script, against
real data.

Not a unit test -- a research script you run by hand to sanity-check the
whole loop end to end and eyeball whether the results are *plausible*
(§30: "a too-good result is treated as a bug, not a win"). Safe to re-run;
every step is idempotent (upsert-on-key in the lake and the DB).

Usage:  .venv/Scripts/python.exe scripts/run_phase1_smoke.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from stockpredictor.backtest.engine import select_rebalance_dates, simulate_top_k_strategy
from stockpredictor.backtest.registry import persist_backtest_result
from stockpredictor.common.logging import get_logger
from stockpredictor.features.registry import build_technical_features_for_universe, persist_features
from stockpredictor.ingestion.macro import ingest_macro_series
from stockpredictor.ingestion.prices import ingest_symbol_prices
from stockpredictor.ingestion.universe import load_universe_csv, sync_universe, sync_universe_from_nse
from stockpredictor.labels.registry import build_labels_for_universe, persist_labels
from stockpredictor.models.dataset import build_training_frame, get_feature_columns
from stockpredictor.models.ensemble import StackedRanker
from stockpredictor.models.walk_forward import generate_folds, split
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

logger = get_logger("run_phase1_smoke")

HORIZON_NAME = "5d"
HORIZON_DAYS = 5
BENCHMARK = "NIFTY500"
YEARS_OF_HISTORY = 5
MIN_TRAIN_DAYS = 250
TEST_WINDOW_DAYS = 63
STEP_DAYS = 63
TOP_K = 5


def main() -> None:
    lake = Lake()
    engine = make_engine()
    init_db(engine)
    sessionmaker = make_sessionmaker(engine)

    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * YEARS_OF_HISTORY))

    logger.info("Step 1/6: syncing universe")
    try:
        universe_df = sync_universe_from_nse(sessionmaker)
        symbols = sorted(universe_df["symbol"].tolist())
    except Exception as exc:
        logger.warning("Live NSE universe fetch failed (%s) -- falling back to the bundled CSV seed", exc)
        sync_universe(sessionmaker)
        symbols = sorted(load_universe_csv()["symbol"].tolist())
    logger.info("Universe: %d symbols", len(symbols))

    logger.info("Step 2/6: ingesting prices + benchmark (%s to %s)", start, end)
    ingested, failed = 0, []
    for symbol in symbols:
        rows = ingest_symbol_prices(lake, symbol, start, end)
        if rows == 0:
            failed.append(symbol)
        else:
            ingested += 1
    ingest_macro_series(lake, [BENCHMARK], start, end)
    logger.info("Prices ingested for %d/%d symbols (failed: %s)", ingested, len(symbols), failed)

    logger.info("Step 3/6: building technical features")
    features = build_technical_features_for_universe(lake)
    persist_features(lake, features)
    logger.info("Feature rows: %d", len(features))

    logger.info("Step 4/6: building labels (horizon=%s)", HORIZON_NAME)
    labels = build_labels_for_universe(lake, benchmark_series=BENCHMARK, horizons={HORIZON_NAME: HORIZON_DAYS})
    persist_labels(lake, labels)
    logger.info("Label rows: %d", len(labels))

    logger.info("Step 5/6: walk-forward training + out-of-fold scoring")
    training_frame = build_training_frame(lake, HORIZON_NAME)
    logger.info("Training frame rows: %d", len(training_frame))
    if training_frame.empty:
        logger.error("Training frame is empty -- aborting")
        return

    feature_cols = get_feature_columns(use_cross_sectional=True)
    folds = generate_folds(
        training_frame["date"], min_train_days=MIN_TRAIN_DAYS, test_window_days=TEST_WINDOW_DAYS, step_days=STEP_DAYS
    )
    logger.info("Generated %d walk-forward folds", len(folds))
    if not folds:
        logger.error("No walk-forward folds -- not enough history yet (need >= %d trading days)", MIN_TRAIN_DAYS)
        return

    fold_splits = split(training_frame, folds)
    scored_frames = []
    for fold, (train_idx, test_idx) in zip(folds, fold_splits):
        train_df = training_frame.loc[train_idx]
        test_df = training_frame.loc[test_idx]
        if train_df.empty or test_df.empty:
            continue

        model = StackedRanker(random_state=42)
        try:
            model.fit(train_df[feature_cols], train_df["outperform"], train_df["date"])
        except ValueError as exc:
            logger.warning("Fold %d skipped: %s", fold.fold_id, exc)
            continue

        test_df = test_df.copy()
        test_df["score"] = model.predict_proba(test_df[feature_cols])
        # Labels carry an OVERLAPPING forward-return window (recomputed at
        # every trading day, see labels/returns.py) -- subsample to
        # non-overlapping rebalance dates per fold before this feeds the
        # backtest engine's compounding, or the same price move gets
        # compounded into the equity curve up to HORIZON_DAYS times (see
        # backtest/engine.py:select_rebalance_dates).
        test_df = select_rebalance_dates(test_df, HORIZON_DAYS)
        scored_frames.append(test_df)
        logger.info(
            "Fold %d: train=%s..%s test=%s..%s train_rows=%d test_rows=%d",
            fold.fold_id, fold.train_start.date(), fold.train_end.date(),
            fold.test_start.date(), fold.test_end.date(), len(train_df), len(test_df),
        )

    if not scored_frames:
        logger.error("No fold produced scored predictions -- aborting")
        return
    scored = pd.concat(scored_frames, ignore_index=True)

    logger.info("Step 6/6: backtesting the out-of-fold Top-%d strategy", TOP_K)
    result = simulate_top_k_strategy(scored, horizon_days=HORIZON_DAYS, top_k=TOP_K)
    persist_backtest_result(lake, result, horizon=HORIZON_NAME, strategy_id="top_k_technical_v1")
    logger.info("Backtest result persisted for the Streamlit Backtest Lab / API to read")

    print("\n=== Strategy vs Benchmark (out-of-fold, net of estimated costs) ===")
    print(pd.DataFrame({"strategy": result.metrics, "benchmark": result.benchmark_metrics}))
    print(f"\nMean IC across test dates: {result.ic_by_date.mean():.4f}")
    print(f"Test periods: {len(result.per_period_returns)}")
    print(
        "\nReminder (§30): a too-good result here is a leakage bug, not a win. "
        "This is a small, short-history smoke run -- not a real performance claim."
    )


if __name__ == "__main__":
    main()
