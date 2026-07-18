from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.ranking.engine import (
    apply_ranking_filters,
    compute_liquidity_and_anomaly_flags,
    rank_universe,
    top_n,
)


def _seed_prices(tmp_lake, symbol: str, closes: list[float], volumes: list[int]) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "date": dates,
            "close_adj": closes,
            "volume": pd.array(volumes, dtype="int64"),
        }
    )
    tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])


def test_liquidity_flags_computes_median_turnover_and_no_anomaly(tmp_lake):
    closes = [100.0] * 25
    volumes = [10_000] * 25
    _seed_prices(tmp_lake, "STABLE", closes, volumes)

    flags = compute_liquidity_and_anomaly_flags(tmp_lake, window=20)
    row = flags[flags["symbol"] == "STABLE"].iloc[0]
    assert row["median_turnover_20d"] == 100.0 * 10_000
    assert row["is_price_anomaly"] is False or row["is_price_anomaly"] == False  # noqa: E712


def test_liquidity_flags_detects_large_single_day_move(tmp_lake):
    closes = [100.0] * 24 + [150.0]  # +50% on the last day
    volumes = [10_000] * 25
    _seed_prices(tmp_lake, "SPIKY", closes, volumes)

    flags = compute_liquidity_and_anomaly_flags(tmp_lake, window=20)
    row = flags[flags["symbol"] == "SPIKY"].iloc[0]
    assert row["is_price_anomaly"] == True  # noqa: E712


def test_liquidity_flags_handles_single_row_history_gracefully(tmp_lake):
    _seed_prices(tmp_lake, "NEWLISTING", [100.0], [1000])
    flags = compute_liquidity_and_anomaly_flags(tmp_lake, window=20)
    row = flags[flags["symbol"] == "NEWLISTING"].iloc[0]
    assert row["is_price_anomaly"] == False  # noqa: E712
    assert pd.isna(row["latest_daily_return"])


def test_apply_ranking_filters_drops_illiquid_anomalous_and_missing_symbols():
    scored = pd.DataFrame(
        {"symbol": ["GOOD", "ILLIQUID", "ANOMALOUS", "NODATA"], "score": [0.8, 0.9, 0.95, 0.99]}
    )
    flags = pd.DataFrame(
        {
            "symbol": ["GOOD", "ILLIQUID", "ANOMALOUS"],
            "median_turnover_20d": [5_000_000.0, 100.0, 5_000_000.0],
            "latest_daily_return": [0.01, 0.01, 0.30],
            "is_price_anomaly": [False, False, True],
        }
    )
    out = apply_ranking_filters(scored, flags, min_turnover=1_000_000.0)
    assert list(out["symbol"]) == ["GOOD"]


def test_rank_universe_assigns_rank_1_to_highest_score():
    df = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [0.5, 0.9, 0.1]})
    ranked = rank_universe(df)
    assert ranked.iloc[0]["symbol"] == "B"
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.iloc[-1]["symbol"] == "C"


def test_rank_universe_empty_input():
    out = rank_universe(pd.DataFrame(columns=["symbol", "score"]))
    assert out.empty


def test_top_n_filters_to_requested_count():
    df = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "score": [0.9, 0.7, 0.5, 0.1]})
    ranked = rank_universe(df)
    top2 = top_n(ranked, 2)
    assert list(top2["symbol"]) == ["A", "B"]


def test_rank_universe_breaks_ties_with_meta_score_not_row_order():
    """Reproduces the exact live bug: isotonic calibration collapses many
    stocks onto one identical calibrated score (a genuine step function --
    see models/ensemble.py's meta_score docstring), and the old
    row-order tie-break (pandas .rank(method="first")) produced a rank
    ordering that carried zero real information for the tied group. Ties
    must now follow meta_score (still continuous) instead."""
    df = pd.DataFrame(
        {
            "symbol": ["LOW_META", "HIGH_META", "MID_META"],
            "score": [0.50, 0.50, 0.50],  # all tied -- the collapse scenario
            "meta_score": [0.40, 0.60, 0.50],
        }
    )
    ranked = rank_universe(df)
    assert list(ranked["symbol"]) == ["HIGH_META", "MID_META", "LOW_META"]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_rank_universe_score_still_takes_priority_over_meta_score():
    """A genuinely better score must still win even against a worse
    meta_score -- meta_score is only a tiebreaker, not a replacement."""
    df = pd.DataFrame(
        {
            "symbol": ["BETTER_SCORE", "WORSE_SCORE"],
            "score": [0.60, 0.55],
            "meta_score": [0.10, 0.90],
        }
    )
    ranked = rank_universe(df)
    assert list(ranked["symbol"]) == ["BETTER_SCORE", "WORSE_SCORE"]


def test_rank_universe_falls_back_to_row_order_without_meta_score_column():
    """Backward compatible: callers that don't supply meta_score (e.g. a
    synthetic scenario with already-distinct scores) still work."""
    df = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [0.5, 0.9, 0.1]})
    ranked = rank_universe(df)
    assert list(ranked["symbol"]) == ["B", "A", "C"]


def test_rank_universe_stable_row_order_fallback_when_tied_and_no_meta_score():
    df = pd.DataFrame({"symbol": ["FIRST", "SECOND"], "score": [0.5, 0.5]})
    ranked = rank_universe(df)
    assert list(ranked["symbol"]) == ["FIRST", "SECOND"]  # original order preserved
