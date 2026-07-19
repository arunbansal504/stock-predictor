from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockpredictor.api.app import app
from stockpredictor.api.dependencies import get_db_sessionmaker, get_lake
from stockpredictor.backtest.engine import BacktestResult
from stockpredictor.backtest.registry import persist_backtest_result
from stockpredictor.explain.registry import persist_explanations
from stockpredictor.labels.registry import GOLD_DOMAIN as LABELS_DOMAIN
from stockpredictor.orchestration.run_tracking import finish_stage, start_stage
from stockpredictor.prediction.registry import persist_predictions
from stockpredictor.ranking.registry import persist_rankings
from stockpredictor.common.types import DataLayer


@pytest.fixture
def client(tmp_lake, db_sessionmaker):
    app.dependency_overrides[get_lake] = lambda: tmp_lake
    app.dependency_overrides[get_db_sessionmaker] = lambda: db_sessionmaker
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed_rankings(tmp_lake, horizon="5d", date="2024-01-01"):
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "date": pd.to_datetime([date] * 3),
            "horizon": [horizon] * 3,
            "score": [0.8, 0.6, 0.4],
            "rank": [1, 2, 3],
            "close_price": [2847.30, 3912.10, 512.50],
        }
    )
    persist_rankings(tmp_lake, df, horizon)


def _seed_explanations(tmp_lake, horizon="5d", date="2024-01-01"):
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "factor_blocks": {"Momentum/Trend": 0.1},
                "top_positive_signals": [{"feature": "return_5d", "block": "Momentum/Trend", "contribution": 0.1}],
                "top_negative_signals": [],
            }
        ]
    )
    persist_explanations(tmp_lake, df, date=pd.Timestamp(date), horizon=horizon)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_rankings_returns_top_n_with_disclaimer(client, tmp_lake):
    _seed_rankings(tmp_lake)
    resp = client.get("/rankings", params={"horizon": "5d", "top_n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "disclaimer" in body and "not investment advice" in body["disclaimer"].lower()
    assert body["data"]["count"] == 2
    assert [r["symbol"] for r in body["data"]["rankings"]] == ["AAA", "BBB"]


def test_rankings_includes_close_price(client, tmp_lake):
    """The share price a stock was ranked at must be visible in the API
    response, not just its score/rank -- see ranking/engine.py's
    compute_liquidity_and_anomaly_flags, which is where close_price is
    attached to every ranked row upstream of this endpoint."""
    _seed_rankings(tmp_lake)
    resp = client.get("/rankings", params={"horizon": "5d", "top_n": 3})
    rankings = resp.json()["data"]["rankings"]
    assert [r["close_price"] for r in rankings] == pytest.approx([2847.30, 3912.10, 512.50])


def test_get_stock_includes_close_price(client, tmp_lake):
    _seed_rankings(tmp_lake)
    resp = client.get("/stocks/AAA", params={"horizon": "5d"})
    assert resp.json()["data"]["close_price"] == pytest.approx(2847.30)


def test_rankings_404_when_no_data(client):
    resp = client.get("/rankings", params={"horizon": "30d"})
    assert resp.status_code == 404


def test_get_stock_includes_ranking_and_explanation(client, tmp_lake):
    _seed_rankings(tmp_lake)
    _seed_explanations(tmp_lake)
    resp = client.get("/stocks/AAA", params={"horizon": "5d"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["symbol"] == "AAA"
    assert data["rank"] == 1
    assert data["explanation"] is not None
    assert data["explanation"]["top_positive_signals"][0]["feature"] == "return_5d"


def test_get_stock_explanation_is_none_when_not_yet_computed(client, tmp_lake):
    _seed_rankings(tmp_lake)  # no explanations seeded
    resp = client.get("/stocks/BBB", params={"horizon": "5d"})
    assert resp.status_code == 200
    assert resp.json()["data"]["explanation"] is None


def test_get_stock_404_for_unranked_symbol(client, tmp_lake):
    _seed_rankings(tmp_lake)
    resp = client.get("/stocks/NOTREAL", params={"horizon": "5d"})
    assert resp.status_code == 404


def test_get_stock_explanation_endpoint(client, tmp_lake):
    _seed_explanations(tmp_lake)
    resp = client.get("/stocks/AAA/explanation", params={"horizon": "5d"})
    assert resp.status_code == 200
    assert resp.json()["data"]["symbol"] == "AAA"


def test_get_stock_explanation_404_when_missing(client, tmp_lake):
    resp = client.get("/stocks/AAA/explanation", params={"horizon": "5d"})
    assert resp.status_code == 404


def test_accuracy_computes_hit_rate_by_decile(client, tmp_lake):
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, n)
    outperform = pd.array(scores > 0.5, dtype="boolean")  # perfectly informative score

    predictions = pd.DataFrame(
        {"symbol": ["AAA"] * n, "date": dates, "horizon": ["5d"] * n, "score": scores, "disagreement": 0.1}
    )
    persist_predictions(tmp_lake, predictions)

    labels = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "date": dates,
            "horizon": ["5d"] * n,
            "excess_return": rng.normal(0, 0.01, n),
            "outperform": outperform,
            "label_valid_date": dates,
        }
    )
    tmp_lake.write(labels, DataLayer.GOLD, LABELS_DOMAIN, "AAA", key_cols=["symbol", "date", "horizon"])

    resp = client.get("/accuracy", params={"horizon": "5d"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["n_resolved_predictions"] == n
    deciles = data["hit_rate_by_score_decile"]
    # Top decile (9) should show a much higher hit rate than bottom (0) --
    # the score was constructed to be perfectly informative.
    assert deciles["9"] > deciles["0"]


def test_accuracy_404_without_history(client):
    resp = client.get("/accuracy", params={"horizon": "5d"})
    assert resp.status_code == 404


def test_get_backtest_returns_latest_result(client, tmp_lake):
    idx = pd.Index(["d1", "d2"], name="date")
    result = BacktestResult(
        per_period_returns=pd.Series([0.02, 0.01], index=idx),
        benchmark_returns=pd.Series([0.01, 0.005], index=idx),
        universe_returns=pd.Series([0.015, 0.008], index=idx),
        ic_by_date=pd.Series([0.1, 0.2], index=idx),
        turnover_by_date=pd.Series([1.0, 0.5], index=idx),
        metrics={"cagr": 0.15, "sharpe": 1.2, "n_periods": 2},
        benchmark_metrics={"cagr": 0.08, "sharpe": 0.6, "n_periods": 2},
        universe_metrics={"cagr": 0.10, "sharpe": 0.8, "n_periods": 2},
    )
    persist_backtest_result(tmp_lake, result, horizon="5d", strategy_id="top_k_v1")

    resp = client.get("/backtests/top_k_v1", params={"horizon": "5d"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["strategy_cagr"] == 0.15
    assert data["benchmark_cagr"] == 0.08
    assert len(data["equity_curve"]) == 2


def test_get_backtest_404_when_missing(client):
    resp = client.get("/backtests/nonexistent", params={"horizon": "5d"})
    assert resp.status_code == 404


def test_monitoring_runs_empty_when_no_history(client):
    resp = client.get("/monitoring/runs")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["latest_run"] is None
    assert data["recent_stages"] == []


def test_monitoring_runs_returns_latest_summary_and_history(client, db_sessionmaker):
    id1 = start_stage(db_sessionmaker, "run1", "sync_universe")
    finish_stage(db_sessionmaker, id1, "success", rows_processed=40)
    id2 = start_stage(db_sessionmaker, "run1", "build_features")
    finish_stage(db_sessionmaker, id2, "success", rows_processed=1000)

    resp = client.get("/monitoring/runs")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["latest_run"]["run_id"] == "run1"
    assert data["latest_run"]["overall_status"] == "success"
    assert len(data["recent_stages"]) == 2


def _seed_portfolio_scenario(tmp_lake, db_sessionmaker, n_symbols=10, horizon="5d"):
    from stockpredictor.storage.models import Security

    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(0)

    rankings = pd.DataFrame(
        {
            "symbol": symbols,
            "date": pd.to_datetime(["2024-06-01"] * n_symbols),
            "horizon": [horizon] * n_symbols,
            "score": np.linspace(0.7, 0.5, n_symbols),
            "rank": range(1, n_symbols + 1),
        }
    )
    persist_rankings(tmp_lake, rankings, horizon)

    dates = pd.bdate_range("2024-01-01", periods=110)
    for i, s in enumerate(symbols):
        closes = 100 + np.cumsum(rng.normal(0, 1 + i * 0.1, len(dates)))
        # ATR is now computed on demand from silver prices (see
        # portfolio/service.py's _latest_atr_by_symbol), not read from a
        # pre-computed gold/features snapshot -- needs real OHLC, not just
        # close_adj, to exercise the same code path as production.
        prices_df = pd.DataFrame(
            {
                "symbol": s,
                "date": dates,
                "open": closes + rng.normal(0, 0.3, len(dates)),
                "high": closes + rng.uniform(0.5, 1.5, len(dates)),
                "low": closes - rng.uniform(0.5, 1.5, len(dates)),
                "close": closes,
                "close_adj": closes,
                "volume": 100_000,
            }
        )
        tmp_lake.write(prices_df, DataLayer.SILVER, "prices", s, key_cols=["symbol", "date"])

    calibration = pd.DataFrame(
        {"decile": [0, 1], "score_min": [0.0, 0.5], "score_max": [0.49, 1.0], "mean_return": [0.01, 0.04], "median_return": [0.01, 0.04], "n_obs": [10, 10]}
    )
    persist_backtest_result(
        tmp_lake,
        BacktestResult(
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            {},
            {},
            {},
        ),
        horizon=horizon, strategy_id="top_k_technical_fundamental_v1", return_calibration=calibration,
    )

    session = db_sessionmaker()
    try:
        for i, s in enumerate(symbols):
            session.add(Security(symbol=s, exchange="NSE", name=f"{s} Ltd.", sector="IT" if i % 3 else "Financials"))
        session.commit()
    finally:
        session.close()

    return symbols


def test_portfolio_construct_returns_full_portfolio(client, tmp_lake, db_sessionmaker):
    _seed_portfolio_scenario(tmp_lake, db_sessionmaker, n_symbols=10)

    resp = client.post("/portfolio/construct", json={"horizon": "5d", "top_n": 10, "risk_profile": "balanced"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["risk_profile"] == "balanced"
    assert len(data["positions"]) == 10
    assert data["expected_volatility"] > 0
    assert "not investment advice" in data["disclaimer"].lower()
    total_weight = sum(p["weight"] for p in data["positions"])
    assert total_weight == pytest.approx(data["total_allocated_weight"], abs=1e-6)


def test_portfolio_construct_404_when_no_rankings(client):
    resp = client.post("/portfolio/construct", json={"horizon": "90d"})
    assert resp.status_code == 404


def test_portfolio_construct_default_risk_profile_is_balanced(client, tmp_lake, db_sessionmaker):
    _seed_portfolio_scenario(tmp_lake, db_sessionmaker, n_symbols=10)
    resp = client.post("/portfolio/construct", json={})
    assert resp.status_code == 200
    assert resp.json()["data"]["risk_profile"] == "balanced"


def test_portfolio_construct_conservative_flags_diversification_shortfall(client, tmp_lake, db_sessionmaker):
    _seed_portfolio_scenario(tmp_lake, db_sessionmaker, n_symbols=3)
    resp = client.post("/portfolio/construct", json={"top_n": 3, "risk_profile": "conservative"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["diversification_warning"] is not None
