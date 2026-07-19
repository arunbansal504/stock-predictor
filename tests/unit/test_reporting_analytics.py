from __future__ import annotations

import json

import pandas as pd
import pytest

from stockpredictor.reporting.analytics import compute_performance_analytics
from stockpredictor.storage.models import PublishedPrediction, Security, ValidationResult


def _seed(db_sessionmaker, rows: list[dict]) -> None:
    session = db_sessionmaker()
    try:
        symbols = {r["symbol"] for r in rows}
        for symbol in symbols:
            session.add(Security(symbol=symbol, exchange="NSE", name=symbol, sector="Technology"))
        for i, r in enumerate(rows):
            pred = PublishedPrediction(
                prediction_id=f"pred-{i}",
                prediction_date=r["date"],
                prediction_horizon=r.get("horizon", "90d"),
                stock_symbol=r["symbol"],
                buy_price=100.0,
                prediction_probability=r.get("probability", 0.6),
                confidence=0.5,
                rank=r.get("rank", 1),
                relative_strength=0.5,
                disagreement=0.1,
                technical_features=json.dumps({}),
                sentiment_features=json.dumps({}),
                feature_vector=json.dumps({}),
                model_version="test-version",
                git_commit_hash="deadbeef",
            )
            session.add(pred)
            session.flush()
            session.add(
                ValidationResult(
                    prediction_id=pred.prediction_id,
                    actual_return=r["actual_return"],
                    benchmark_return=r.get("benchmark_return", 0.0),
                    alpha=r["alpha"],
                    hit_or_miss=r["alpha"] > 0,
                    maximum_drawdown=-0.05,
                    maximum_gain=0.10,
                    volatility=0.2,
                    sharpe_ratio=1.0,
                    information_ratio=0.5,
                )
            )
        session.commit()
    finally:
        session.close()


def test_compute_performance_analytics_empty(tmp_lake, db_sessionmaker):
    result = compute_performance_analytics(tmp_lake, db_sessionmaker)
    assert result["n_published"] == 0
    assert result["n_resolved"] == 0


def test_compute_performance_analytics_basic_aggregates(tmp_lake, db_sessionmaker):
    rows = [
        {"symbol": "AAA", "date": pd.Timestamp("2024-01-05").date(), "actual_return": 0.10, "alpha": 0.05, "rank": 1},
        {"symbol": "BBB", "date": pd.Timestamp("2024-01-05").date(), "actual_return": -0.02, "alpha": -0.03, "rank": 6},
        {"symbol": "CCC", "date": pd.Timestamp("2024-02-05").date(), "actual_return": 0.03, "alpha": 0.01, "rank": 2},
    ]
    _seed(db_sessionmaker, rows)

    result = compute_performance_analytics(tmp_lake, db_sessionmaker)
    assert result["n_published"] == 3
    assert result["n_resolved"] == 3
    assert result["overall_accuracy"] == pytest.approx(2 / 3)
    assert result["top_5_accuracy"] == pytest.approx(1.0)  # only rank<=5 rows: AAA (hit), CCC (hit)
    assert result["avg_alpha"] == pytest.approx((0.05 - 0.03 + 0.01) / 3)
    assert "2024-01" in result["monthly_stats"]
    assert "2024-02" in result["monthly_stats"]
    assert result["by_horizon_ratios"]["90d"]["n"] == 3
