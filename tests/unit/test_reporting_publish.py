from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import DataLayer
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.reporting.publish import publish_weekly_predictions
from stockpredictor.storage.models import PublishedPrediction, Security


def _seed_prices(tmp_lake, symbol: str, n: int, seed: int) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    daily_returns = rng.normal(0.0003, 0.01, n)
    close = 100.0 * np.cumprod(1 + daily_returns)
    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.985,
            "close": close,
            "close_adj": close,
            "volume": 200_000,
        }
    )
    tmp_lake.write(df, DataLayer.SILVER, "prices", symbol, key_cols=["symbol", "date"])
    return dates


def _seed_labels(tmp_lake, symbol: str, dates: pd.DatetimeIndex, seed: int, horizon: str) -> None:
    rng = np.random.default_rng(seed + 100)
    n = len(dates)
    outperform = pd.array(rng.uniform(0, 1, n) < 0.5, dtype="boolean")
    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "horizon": horizon,
            "excess_return": rng.normal(0, 0.02, n),
            "outperform": outperform,
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(df, DataLayer.GOLD, LABELS_DOMAIN, symbol, key_cols=["symbol", "date", "horizon"])


@pytest.fixture
def seeded_universe(tmp_lake, db_sessionmaker):
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    session = db_sessionmaker()
    try:
        for symbol in symbols:
            session.add(Security(symbol=symbol, exchange="NSE", name=symbol, sector="Technology"))
        session.commit()
    finally:
        session.close()

    for i, symbol in enumerate(symbols):
        dates = _seed_prices(tmp_lake, symbol, n=260, seed=i)
        _seed_labels(tmp_lake, symbol, dates, seed=i, horizon="90d")
    return symbols


def test_publish_weekly_predictions_writes_db_rows_and_files(tmp_lake, db_sessionmaker, seeded_universe, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpredictor.reporting.publish.PREDICTIONS_DIR", tmp_path / "predictions")

    published = publish_weekly_predictions(tmp_lake, db_sessionmaker, horizon="90d", top_k=2)

    assert len(published) == 2
    assert set(published.columns) >= {
        "prediction_id", "prediction_date", "prediction_horizon", "stock_symbol",
        "buy_price", "prediction_probability", "confidence", "rank", "disagreement",
        "technical_features", "sentiment_features", "feature_vector", "model_version", "git_commit_hash",
    }
    assert sorted(published["rank"]) == [1, 2]
    # technical_features/feature_vector are JSON-serialized -- confirm they round-trip.
    technical = json.loads(published.iloc[0]["technical_features"])
    assert "return_5d" in technical
    vector = json.loads(published.iloc[0]["feature_vector"])
    assert all(k.endswith("_xrank") for k in vector)

    session = db_sessionmaker()
    try:
        rows = session.query(PublishedPrediction).all()
        assert len(rows) == 2
    finally:
        session.close()

    csv_files = list((tmp_path / "predictions").glob("*.csv"))
    json_files = list((tmp_path / "predictions").glob("*.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 1


def test_publish_weekly_predictions_is_idempotent_on_rerun(tmp_lake, db_sessionmaker, seeded_universe, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpredictor.reporting.publish.PREDICTIONS_DIR", tmp_path / "predictions")

    first = publish_weekly_predictions(tmp_lake, db_sessionmaker, horizon="90d", top_k=2)
    assert len(first) == 2

    second = publish_weekly_predictions(tmp_lake, db_sessionmaker, horizon="90d", top_k=2)
    assert second.empty  # same date/horizon already published -- safe no-op, no duplicate rows, no raise

    session = db_sessionmaker()
    try:
        assert session.query(PublishedPrediction).count() == 2
    finally:
        session.close()
