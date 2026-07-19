from __future__ import annotations

import json

import pandas as pd
import pytest

from stockpredictor.reporting.review import generate_monthly_review
from stockpredictor.storage.models import PublishedPrediction, Security, ValidationResult


def _seed(db_sessionmaker, rows: list[dict]) -> None:
    session = db_sessionmaker()
    try:
        symbols = {r["symbol"]: r.get("sector", "Technology") for r in rows}
        for symbol, sector in symbols.items():
            session.add(Security(symbol=symbol, exchange="NSE", name=symbol, sector=sector))
        for i, r in enumerate(rows):
            pred = PublishedPrediction(
                prediction_id=f"pred-{i}",
                prediction_date=r["date"],
                prediction_horizon="90d",
                stock_symbol=r["symbol"],
                buy_price=100.0,
                prediction_probability=0.6,
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
                    benchmark_return=0.0,
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


def test_generate_monthly_review_writes_both_reports(tmp_lake, db_sessionmaker, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpredictor.reporting.review.REPORTS_DIR", tmp_path / "reports")
    rows = [
        {"symbol": "AAA", "date": pd.Timestamp("2024-03-05").date(), "actual_return": 0.10, "alpha": 0.05, "rank": 1},
        {"symbol": "BBB", "date": pd.Timestamp("2024-03-06").date(), "actual_return": -0.02, "alpha": -0.03, "rank": 2, "sector": "Energy"},
    ]
    _seed(db_sessionmaker, rows)

    review_path, proposal_path = generate_monthly_review(tmp_lake, db_sessionmaker, month="2024-03")

    review_text = (tmp_path / "reports" / "2024-03-ML-Review.md").read_text(encoding="utf-8")
    proposal_text = (tmp_path / "reports" / "2024-03-Improvement-Proposal.md").read_text(encoding="utf-8")
    assert "# ML Review -- 2024-03" in review_text
    assert "AAA" in review_text  # winning prediction shows up in Q1
    assert "BBB" in review_text  # losing prediction shows up in Q2
    assert "# Improvement Proposal -- 2024-03" in proposal_text
    assert "ANALYSIS: fill in during monthly review" in review_text  # placeholders present, not fabricated


def test_generate_monthly_review_handles_zero_resolved_predictions(tmp_lake, db_sessionmaker, tmp_path, monkeypatch):
    """Regression test: a freshly-deployed system has published predictions
    but none resolved yet (real state observed dry-running this against the
    live repo). `load_resolved_predictions` used to return a bare, columnless
    `pd.DataFrame()` in this case, and indexing `frame["prediction_date"]`
    on it raised KeyError before any `.empty` check ran."""
    monkeypatch.setattr("stockpredictor.reporting.review.REPORTS_DIR", tmp_path / "reports")
    session = db_sessionmaker()
    try:
        session.add(Security(symbol="AAA", exchange="NSE", name="AAA", sector="Technology"))
        session.add(
            PublishedPrediction(
                prediction_id="pred-unresolved",
                prediction_date=pd.Timestamp("2024-03-05").date(),
                prediction_horizon="90d",
                stock_symbol="AAA",
                buy_price=100.0,
                prediction_probability=0.6,
                confidence=0.5,
                rank=1,
                relative_strength=0.5,
                disagreement=0.1,
                technical_features=json.dumps({}),
                sentiment_features=json.dumps({}),
                feature_vector=json.dumps({}),
                model_version="test-version",
                git_commit_hash="deadbeef",
            )
        )
        session.commit()
    finally:
        session.close()

    review_path, proposal_path = generate_monthly_review(tmp_lake, db_sessionmaker, month="2024-03")
    review_text = (tmp_path / "reports" / "2024-03-ML-Review.md").read_text(encoding="utf-8")
    assert "# ML Review -- 2024-03" in review_text


def test_generate_monthly_review_refuses_to_overwrite(tmp_lake, db_sessionmaker, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpredictor.reporting.review.REPORTS_DIR", tmp_path / "reports")
    _seed(db_sessionmaker, [{"symbol": "AAA", "date": pd.Timestamp("2024-03-05").date(), "actual_return": 0.1, "alpha": 0.05}])

    generate_monthly_review(tmp_lake, db_sessionmaker, month="2024-03")
    with pytest.raises(FileExistsError):
        generate_monthly_review(tmp_lake, db_sessionmaker, month="2024-03")
