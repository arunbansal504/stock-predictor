"""Standing prediction-pipeline audit tool.

Trains the production model for one horizon exactly as `prediction/predict.py`
does, scores the latest feature snapshot, and prints diagnostics for both the
whole universe and a sample of individual stocks -- so questions like "are
these predictions actually stock-specific?" or "why did X rank above Y?" can
be answered from real numbers instead of re-deriving the pipeline by hand.

Read-only w.r.t. the lake: the model is trained in memory (nothing is
pickled/persisted anywhere in this codebase -- see prediction/predict.py) and
this script writes nothing to Gold.

Usage:
    .venv/Scripts/python.exe scripts/run_prediction_diagnostics.py
    .venv/Scripts/python.exe scripts/run_prediction_diagnostics.py --horizon 30d --top 15
    .venv/Scripts/python.exe scripts/run_prediction_diagnostics.py --symbols RELIANCE,TCS,INFY
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# News titles can contain non-ASCII characters (currency symbols, Indian
# company names); Windows' default console encoding (cp1252) can't print
# them, so force UTF-8 stdout rather than crashing mid-report.
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.features.sentiment import latest_sentiment_snapshot
from stockpredictor.prediction.predict import get_latest_feature_snapshot, train_production_model
from stockpredictor.storage.lake import Lake

logger = get_logger("run_prediction_diagnostics")

# A representative slice of technical indicators for the per-stock report --
# not the full feature list, which is printed separately as the raw _xrank
# vector.
TECHNICAL_DISPLAY_COLUMNS = [
    "rsi_14",
    "macd_hist",
    "return_20d",
    "return_60d",
    "price_vs_sma50",
    "dist_from_52w_high",
    "realized_vol_20d",
    "volume_zscore_20d",
]

NEAR_CONSTANT_STD_THRESHOLD = 1e-6
HIGH_NAN_RATE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", default="90d", choices=["5d", "30d", "90d"])
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols to inspect; overrides --top sampling"
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top-ranked stocks to inspect (default 10)")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to the latest available date")
    return parser.parse_args()


def print_universe_diagnostics(
    model, X: pd.DataFrame, meta_score: np.ndarray, score: np.ndarray, feature_cols: list[str]
) -> None:
    print("\n" + "=" * 78)
    print("UNIVERSE-LEVEL DIAGNOSTICS")
    print("=" * 78)

    print(f"\nRows scored: {len(X)}")
    print(
        f"meta_score (pre-calibration): min={meta_score.min():.6f} max={meta_score.max():.6f} "
        f"std={meta_score.std():.6f} nunique={len(np.unique(meta_score))}"
    )
    print(
        f"score (calibrated):           min={score.min():.6f} max={score.max():.6f} "
        f"std={score.std():.6f} nunique={len(np.unique(score))}"
    )
    if len(np.unique(score)) < 0.5 * len(X):
        print(
            "NOTE: score has far fewer unique values than rows scored. Expected under "
            "centered-isotonic calibration when the underlying signal is weak (score is "
            "anchored to a small number of historical evidence bands) -- see the calibrator "
            "block table below and meta_score for the finer-grained pre-calibration signal, "
            "not necessarily a bug."
        )

    dup_count = int(X.duplicated().sum())
    print(f"\nDuplicate feature vectors: {dup_count} / {len(X)}")
    if dup_count:
        print(X[X.duplicated(keep=False)].head(10).to_string())

    print("\nNear-constant features (std < %.0e):" % NEAR_CONSTANT_STD_THRESHOLD)
    stds = X.std(numeric_only=True)
    near_constant = stds[stds < NEAR_CONSTANT_STD_THRESHOLD]
    if near_constant.empty:
        print("  none found")
    else:
        for col, std in near_constant.items():
            print(f"  {col}: std={std:.2e} nunique={X[col].nunique()}")

    print(f"\nHigh-NaN-rate features (> {HIGH_NAN_RATE_THRESHOLD:.0%}):")
    nan_rates = X.isna().mean()
    high_nan = nan_rates[nan_rates > HIGH_NAN_RATE_THRESHOLD]
    if high_nan.empty:
        print("  none found")
    else:
        for col, rate in high_nan.items():
            print(f"  {col}: NaN rate={rate:.1%}")

    print("\nLightGBM feature importances (top 15):")
    importances = pd.Series(model.lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(15).to_string())

    print("\nLinear-baseline coefficients, |coef| descending (top 15, post impute/scale):")
    linear_coef = pd.Series(model.linear.named_steps["clf"].coef_[0], index=feature_cols)
    print(linear_coef.reindex(linear_coef.abs().sort_values(ascending=False).index).head(15).to_string())

    print("\nMeta-learner coefficients (on [lgbm_score, linear_score]):")
    print(
        f"  lgbm={model.meta.coef_[0][0]:.4f}  linear={model.meta.coef_[0][1]:.4f}  "
        f"intercept={model.meta.intercept_[0]:.4f}"
    )

    print("\nCalibrator block table (PAVA blocks backing separation_info / score anchors):")
    print(model.calibrator.block_stats.to_string(index=False))
    print(f"base_rate: {model.calibrator.base_rate:.6f}")


def select_symbols(snapshot: pd.DataFrame, score: np.ndarray, meta_score: np.ndarray, top_n: int) -> list[str]:
    ranked = pd.DataFrame(
        {"symbol": snapshot["symbol"].to_numpy(), "score": score, "meta_score": meta_score}
    ).sort_values(["score", "meta_score"], ascending=False)
    return ranked["symbol"].head(top_n).tolist()


def print_stock_report(
    lake: Lake,
    symbol: str,
    snapshot_row: pd.Series,
    feature_cols: list[str],
    base_row: pd.Series,
    meta_score_i: float,
    score_i: float,
    separation_row: pd.Series,
) -> None:
    print("\n" + "-" * 78)
    print(f"STOCK: {symbol}")
    print("-" * 78)

    news = lake.read(DataLayer.SILVER, "news", symbol)
    print(f"\nNews articles: {len(news)}")
    if not news.empty:
        recent = news.sort_values("published_date", ascending=False).head(5)
        for _, row in recent.iterrows():
            print(f"  [{row['published_date']}] ({row['sentiment_label']}, {row['sentiment_score']:.2f}) {row['title']}")

    as_of_ts = pd.Timestamp(snapshot_row["date"])
    sentiment_summary = latest_sentiment_snapshot(news, as_of_ts)
    print("\nSentiment (EXCLUDED from model inputs -- see features/sentiment.py / registry.py):")
    print(f"  mean_sentiment={sentiment_summary['mean_sentiment']}  article_count={sentiment_summary['article_count']}")

    print("\nKey technical indicators (raw values):")
    for col in TECHNICAL_DISPLAY_COLUMNS:
        if col in snapshot_row.index:
            print(f"  {col}: {snapshot_row[col]}")

    print("\nFinal feature vector (_xrank columns fed to the model):")
    feature_vector = snapshot_row[feature_cols]
    nan_count = int(feature_vector.isna().sum())
    print(f"  NaN count: {nan_count}/{len(feature_cols)}")
    print(feature_vector.to_string())

    print("\nModel outputs:")
    print(f"  lgbm_proba={base_row['lgbm']:.6f}  linear_proba={base_row['linear']:.6f}")
    print(f"  meta_score (pre-calibration)={meta_score_i:.6f}")
    print(f"  score (calibrated)={score_i:.6f}")
    print(
        f"  separation: n={int(separation_row['n'])} empirical_rate={separation_row['empirical_rate']:.6f} "
        f"p_value={separation_row['p_value']:.4g} direction={separation_row['separation_direction']} "
        f"base_rate={separation_row['base_rate']:.6f}"
    )


def main() -> None:
    args = parse_args()
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else None

    lake = Lake()
    logger.info("Training production model for horizon=%s", args.horizon)
    model, feature_cols = train_production_model(lake, args.horizon)

    snapshot = get_latest_feature_snapshot(lake, as_of=as_of)
    if snapshot.empty:
        logger.error("No feature snapshot available -- has the pipeline been run at least once?")
        return

    X = snapshot[feature_cols]
    base = model.base_scores(X)
    meta_score = model.meta_score(X)
    score = model.calibrator.transform(meta_score)
    separation = model.calibrator.separation_info(meta_score)

    print_universe_diagnostics(model, X, meta_score, score, feature_cols)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = select_symbols(snapshot, score, meta_score, args.top)

    for symbol in symbols:
        matches = snapshot.index[snapshot["symbol"] == symbol]
        if len(matches) == 0:
            logger.warning("Symbol %s not found in latest snapshot -- skipping", symbol)
            continue
        i = matches[0]
        print_stock_report(
            lake,
            symbol,
            snapshot.loc[i],
            feature_cols,
            base.loc[i],
            float(meta_score[i]),
            float(score[i]),
            separation.loc[i],
        )


if __name__ == "__main__":
    main()
