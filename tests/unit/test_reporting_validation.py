from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stockpredictor.common.types import DataLayer
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.reporting.validation import run_daily_validation
from stockpredictor.storage.models import PublishedPrediction, Security, ValidationResult

HORIZON = "5d"
HORIZON_DAYS = 5


def _seed_scenario(tmp_lake, db_sessionmaker, prediction_date: pd.Timestamp, alpha_sign: float):
    dates = pd.bdate_range(prediction_date, periods=HORIZON_DAYS + 1)
    resolution_date = dates[-1]

    rng = np.random.default_rng(0)
    stock_returns = rng.normal(0.001 + 0.01 * alpha_sign, 0.005, HORIZON_DAYS)
    bench_returns = rng.normal(0.001, 0.003, HORIZON_DAYS)
    stock_close = 100.0 * np.cumprod(1 + np.concatenate([[0], stock_returns]))
    bench_close = 20000.0 * np.cumprod(1 + np.concatenate([[0], bench_returns]))

    prices = pd.DataFrame(
        {
            "symbol": "AAA", "date": dates,
            "open": stock_close, "high": stock_close * 1.01, "low": stock_close * 0.99,
            "close": stock_close, "close_adj": stock_close, "volume": 200_000,
        }
    )
    tmp_lake.write(prices, DataLayer.SILVER, "prices", "AAA", key_cols=["symbol", "date"])

    macro = pd.DataFrame({"series": "NIFTY500", "date": dates, "close": bench_close})
    tmp_lake.write(macro, DataLayer.SILVER, "macro", "NIFTY500", key_cols=["series", "date"])

    forward_return = stock_close[-1] / stock_close[0] - 1.0
    benchmark_forward_return = bench_close[-1] / bench_close[0] - 1.0
    excess_return = forward_return - benchmark_forward_return

    labels = pd.DataFrame(
        [{
            "symbol": "AAA", "date": prediction_date, "horizon": HORIZON,
            "forward_return": forward_return, "benchmark_forward_return": benchmark_forward_return,
            "excess_return": excess_return, "outperform": excess_return > 0,
            "label_valid_date": resolution_date,
        }]
    )
    tmp_lake.write(labels, DataLayer.GOLD, LABELS_DOMAIN, "AAA", key_cols=["symbol", "date", "horizon"])

    session = db_sessionmaker()
    try:
        session.add(Security(symbol="AAA", exchange="NSE", name="AAA", sector="Technology"))
        session.add(
            PublishedPrediction(
                prediction_id=f"{prediction_date:%Y%m%d}-{HORIZON}-AAA",
                prediction_date=prediction_date.date(),
                prediction_horizon=HORIZON,
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

    return forward_return, benchmark_forward_return, excess_return


def test_run_daily_validation_writes_result_for_resolved_prediction(tmp_lake, db_sessionmaker):
    prediction_date = pd.Timestamp("2023-01-02")  # far in the past -- always "resolved" by test time
    forward_return, benchmark_forward_return, excess_return = _seed_scenario(
        tmp_lake, db_sessionmaker, prediction_date, alpha_sign=1.0
    )

    n_written = run_daily_validation(tmp_lake, db_sessionmaker)
    assert n_written == 1

    session = db_sessionmaker()
    try:
        result = session.query(ValidationResult).one()
        assert result.actual_return == pytest.approx(forward_return, rel=1e-6)
        assert result.benchmark_return == pytest.approx(benchmark_forward_return, rel=1e-6)
        assert result.alpha == pytest.approx(excess_return, rel=1e-6)
        assert result.hit_or_miss == (excess_return > 0)
        assert result.maximum_drawdown is not None and result.maximum_drawdown <= 0
        assert result.maximum_gain is not None and result.maximum_gain >= 0
    finally:
        session.close()


def test_run_daily_validation_is_idempotent(tmp_lake, db_sessionmaker):
    prediction_date = pd.Timestamp("2023-01-02")
    _seed_scenario(tmp_lake, db_sessionmaker, prediction_date, alpha_sign=1.0)

    assert run_daily_validation(tmp_lake, db_sessionmaker) == 1
    assert run_daily_validation(tmp_lake, db_sessionmaker) == 0  # already validated -- no duplicate row

    session = db_sessionmaker()
    try:
        assert session.query(ValidationResult).count() == 1
    finally:
        session.close()


def test_run_daily_validation_skips_unresolved_predictions(tmp_lake, db_sessionmaker):
    from stockpredictor.common.trading_calendar import last_completed_nse_session

    future_date = pd.Timestamp(last_completed_nse_session()) + pd.Timedelta(days=1)
    _seed_scenario(tmp_lake, db_sessionmaker, future_date, alpha_sign=1.0)

    assert run_daily_validation(tmp_lake, db_sessionmaker) == 0
