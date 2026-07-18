"""Read-mostly API (§14): serves pre-computed rankings, predictions, and
explanations. Never triggers model inference on demand (§4: "The UI never
triggers model inference on demand; it reads yesterday-night's published
results") -- every route here is a lake read, nothing trains or predicts
inline. Every response carries the research/education disclaimer (§1, §15).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.api.dependencies import get_db_sessionmaker, get_lake
from stockpredictor.backtest.registry import read_latest_backtest_result
from stockpredictor.common.types import RiskProfile
from stockpredictor.explain.registry import read_explanations
from stockpredictor.monitoring.accuracy import compute_accuracy
from stockpredictor.monitoring.run_status import get_latest_run_summary, get_recent_runs
from stockpredictor.portfolio.service import DEFAULT_STRATEGY_ID, construct_portfolio_from_lake
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.lake import Lake

DISCLAIMER = (
    "For research/educational purposes only. Not investment advice. "
    "Markets carry risk; past performance does not guarantee future results."
)

app = FastAPI(
    title="Stock Predictor Research API",
    version="0.1.0",
    description="Research/education tool. Not investment advice.",
)


def _envelope(data: Any) -> dict:
    return {"data": data, "disclaimer": DISCLAIMER}


class PortfolioConstructRequest(BaseModel):
    horizon: str = "5d"
    top_n: int = 10
    risk_profile: RiskProfile = RiskProfile.BALANCED
    strategy_id: str = DEFAULT_STRATEGY_ID
    lookback_days: int = 90


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/rankings")
def get_rankings(
    horizon: str = "5d",
    top_n: int = Query(10, ge=1, le=500),
    lake: Lake = Depends(get_lake),
) -> dict:
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        raise HTTPException(status_code=404, detail=f"No rankings available for horizon={horizon}")

    top = ranked[ranked["rank"] <= top_n].copy()
    as_of = str(top["date"].max().date()) if not top.empty else None
    top["date"] = top["date"].astype(str)
    return _envelope(
        {
            "horizon": horizon,
            "as_of_date": as_of,
            "count": len(top),
            "rankings": top.to_dict(orient="records"),
        }
    )


@app.get("/stocks/{symbol}")
def get_stock(symbol: str, horizon: str = "5d", lake: Lake = Depends(get_lake)) -> dict:
    ranked = read_latest_rankings(lake, horizon)
    row = ranked[ranked["symbol"] == symbol]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No ranking for symbol={symbol}, horizon={horizon}")

    record = row.iloc[0].to_dict()
    record["date"] = str(record["date"])

    explanations = read_explanations(lake, horizon)
    exp_row = explanations[explanations["symbol"] == symbol] if not explanations.empty else explanations
    explanation = None
    if not exp_row.empty:
        explanation = exp_row.iloc[0].to_dict()
        explanation["date"] = str(explanation["date"])

    return _envelope({"symbol": symbol, "horizon": horizon, **record, "explanation": explanation})


@app.get("/stocks/{symbol}/explanation")
def get_stock_explanation(symbol: str, horizon: str = "5d", lake: Lake = Depends(get_lake)) -> dict:
    explanations = read_explanations(lake, horizon)
    if explanations.empty:
        raise HTTPException(status_code=404, detail=f"No explanations available for horizon={horizon}")
    row = explanations[explanations["symbol"] == symbol]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No explanation for symbol={symbol}, horizon={horizon}")
    record = row.iloc[0].to_dict()
    record["date"] = str(record["date"])
    return _envelope(record)


@app.get("/backtests/{strategy_id}")
def get_backtest(strategy_id: str, horizon: str = "5d", lake: Lake = Depends(get_lake)) -> dict:
    result = read_latest_backtest_result(lake, strategy_id, horizon)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No backtest result for strategy_id={strategy_id}, horizon={horizon}"
        )
    result["run_date"] = str(result["run_date"])
    return _envelope(result)


@app.post("/portfolio/construct")
def post_portfolio_construct(
    request: PortfolioConstructRequest,
    lake: Lake = Depends(get_lake),
    sessionmaker_: sessionmaker[Session] = Depends(get_db_sessionmaker),
) -> dict:
    """§12: turns the current Top-N ranking into an illustrative research
    portfolio (HRP allocation, risk-profile caps, stop-loss/target). Not
    model inference -- a deterministic optimization over already-published
    rankings, safe to compute on demand (see portfolio/constructor.py)."""
    portfolio = construct_portfolio_from_lake(
        lake, sessionmaker_, request.horizon, request.risk_profile, request.top_n,
        request.strategy_id, request.lookback_days,
    )
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"No rankings available for horizon={request.horizon}")
    return _envelope(dataclasses.asdict(portfolio))


@app.get("/monitoring/runs")
def get_monitoring_runs(
    limit: int = Query(20, ge=1, le=200),
    sessionmaker_: sessionmaker[Session] = Depends(get_db_sessionmaker),
) -> dict:
    """Recent pipeline stage history (§23), the read side of run_metadata."""
    latest = get_latest_run_summary(sessionmaker_)
    recent = get_recent_runs(sessionmaker_, limit=limit)
    return _envelope({"latest_run": latest, "recent_stages": recent})


@app.get("/accuracy")
def get_accuracy(horizon: str = "5d", lake: Lake = Depends(get_lake)) -> dict:
    """Historical calibration sanity check (§15 "Model Transparency" screen).
    See monitoring/accuracy.py for what "accuracy" means here and why."""
    result = compute_accuracy(lake, horizon)
    if result is None:
        raise HTTPException(status_code=404, detail="Not enough resolved history yet to compute accuracy")
    return _envelope(result)
