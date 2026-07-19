"""Backtest every published horizon, not just 5d.

scripts/run_phase1_smoke.py validates only the 5d horizon end to end
(ingestion through backtest); 30d and 90d are computed and published
nightly by orchestration/nightly_flow.py but have never been backtested --
meaning two-thirds of what the live rankings publish had zero out-of-sample
evidence. This script fills that gap using data already in the lake (no
re-ingestion, so it's fast to iterate with): walk-forward CV + Top-K
backtest for every horizon, persisted under the SAME strategy_id
run_phase1_smoke.py uses, so the API/UI's existing
`read_latest_backtest_result(strategy_id, horizon)` immediately serves all
three horizons once this has run, not just 5d.

Prerequisite: prices/fundamentals/features/labels already in the lake --
run scripts/run_phase1_smoke.py or the nightly pipeline at least once first.

Usage:  .venv/Scripts/python.exe scripts/run_backtests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from stockpredictor.backtest.calibration_curve import compute_decile_return_calibration
from stockpredictor.backtest.engine import BacktestResult, select_rebalance_dates, simulate_top_k_strategy
from stockpredictor.backtest.registry import persist_backtest_result
from stockpredictor.common.logging import get_logger
from stockpredictor.models.dataset import build_training_frame, get_feature_columns
from stockpredictor.models.ensemble import StackedRanker
from stockpredictor.models.walk_forward import generate_folds, split
from stockpredictor.storage.lake import Lake

logger = get_logger("run_backtests")

# Kept in sync with labels/registry.py's DEFAULT_HORIZONS and
# orchestration/nightly_flow.py's DEFAULT_HORIZONS.
HORIZONS: dict[str, int] = {"5d": 5, "30d": 30, "90d": 90}
STRATEGY_ID = "top_k_technical_fundamental_v1"
MIN_TRAIN_DAYS = 250
TEST_WINDOW_DAYS = 63
STEP_DAYS = 63
TOP_K = 5


def backtest_horizon(lake: Lake, horizon_name: str, horizon_days: int) -> BacktestResult | None:
    logger.info("=== Horizon %s (%d trading days) ===", horizon_name, horizon_days)
    training_frame = build_training_frame(lake, horizon_name)
    logger.info("Training frame rows: %d", len(training_frame))
    if training_frame.empty:
        logger.warning(
            "No training data for horizon=%s -- skipping (run ingestion + "
            "build_features/build_labels for this horizon first)",
            horizon_name,
        )
        return None

    feature_cols = get_feature_columns(use_cross_sectional=True)
    folds = generate_folds(
        training_frame["date"],
        min_train_days=MIN_TRAIN_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_days=STEP_DAYS,
    )
    logger.info("Generated %d walk-forward folds", len(folds))
    if not folds:
        logger.warning(
            "Not enough history for horizon=%s (need >= %d trading days) -- skipping",
            horizon_name, MIN_TRAIN_DAYS,
        )
        return None

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
        # every trading day) -- subsample to non-overlapping rebalance
        # dates per fold before this feeds the backtest engine's
        # compounding, or the same price move gets compounded into the
        # equity curve up to horizon_days times.
        test_df = select_rebalance_dates(test_df, horizon_days)
        scored_frames.append(test_df)
        logger.info(
            "Fold %d: train=%s..%s test=%s..%s train_rows=%d test_rows=%d",
            fold.fold_id, fold.train_start.date(), fold.train_end.date(),
            fold.test_start.date(), fold.test_end.date(), len(train_df), len(test_df),
        )

    if not scored_frames:
        logger.warning("No fold produced scored predictions for horizon=%s -- skipping", horizon_name)
        return None
    scored = pd.concat(scored_frames, ignore_index=True)

    result = simulate_top_k_strategy(scored, horizon_days=horizon_days, top_k=TOP_K)
    return_calibration = compute_decile_return_calibration(scored["score"], scored["forward_return"])
    persist_backtest_result(
        lake, result, horizon=horizon_name, strategy_id=STRATEGY_ID, return_calibration=return_calibration,
    )

    print(f"\n=== {horizon_name}: strategy vs benchmark vs universe (out-of-fold, net of estimated costs) ===")
    print(
        pd.DataFrame(
            {
                "strategy": result.metrics,
                "benchmark": result.benchmark_metrics,
                "universe (hold-everything)": result.universe_metrics,
            }
        )
    )
    print(
        f"Mean IC: {result.ic_by_date.mean():.4f}  "
        f"Mean turnover: {result.turnover_by_date.mean():.2f}  "
        f"Periods: {len(result.per_period_returns)}"
    )
    return result


def main() -> None:
    lake = Lake()
    results = {}
    for name, days in HORIZONS.items():
        result = backtest_horizon(lake, name, days)
        if result is not None:
            results[name] = result

    if not results:
        logger.error(
            "No horizon produced a backtest result -- is the lake populated? "
            "Run scripts/run_phase1_smoke.py first."
        )
        return

    print(f"\n=== Backtested {len(results)}/{len(HORIZONS)} horizons: {sorted(results)} ===")
    print(
        "\nReminder (a too-good result is a leakage bug, not a win): only promote a "
        "horizon to the live UI once its net-of-cost CAGR/Sharpe and OOS IC look "
        "genuinely tradeable, not just positive -- and compare against the "
        "'universe (hold-everything)' baseline, not just the cap-weighted "
        "benchmark, to see whether the ranking itself is adding value."
    )


if __name__ == "__main__":
    main()
