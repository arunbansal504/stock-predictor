"""Glue layer between `construct_portfolio` (pure) and the lake/DB (§12).

Shared by the API (api/app.py) and the Streamlit UI so both surfaces gather
data the same way -- same rationale as monitoring/accuracy.py: business
logic lives once, thin callers wrap it.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.backtest.registry import read_latest_return_calibration
from stockpredictor.common.types import DataLayer, RiskProfile
from stockpredictor.portfolio.constructor import ConstructedPortfolio, construct_portfolio
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import Security

DEFAULT_STRATEGY_ID = "top_k_technical_fundamental_v1"


def construct_portfolio_from_lake(
    lake: Lake,
    session_factory: sessionmaker[Session],
    horizon: str,
    risk_profile: RiskProfile,
    top_n: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    lookback_days: int = 90,
) -> ConstructedPortfolio | None:
    """Returns None if there are no published rankings for `horizon` yet --
    callers decide how to present that (404 in the API, an info message in
    the UI)."""
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        return None

    candidates = ranked.sort_values("rank").head(top_n)
    symbols = candidates["symbol"].tolist()

    prices = lake.read_all(DataLayer.SILVER, "prices")
    prices = prices[prices["symbol"].isin(symbols)] if not prices.empty else prices

    features = lake.read_all(DataLayer.GOLD, "features")
    if not features.empty:
        features = features[features["symbol"].isin(symbols)].sort_values("date").groupby("symbol").tail(1)
        atr_by_symbol = features.set_index("symbol")["atr_14"]
    else:
        atr_by_symbol = pd.Series(dtype="float64")

    session = session_factory()
    try:
        secs = session.execute(select(Security).where(Security.symbol.in_(symbols))).scalars().all()
        sector_by_symbol = pd.Series({s.symbol: s.sector for s in secs})
    finally:
        session.close()

    return_calibration = read_latest_return_calibration(lake, strategy_id, horizon)

    return construct_portfolio(
        ranked=candidates,
        prices=prices,
        atr_by_symbol=atr_by_symbol,
        sector_by_symbol=sector_by_symbol,
        return_calibration=return_calibration,
        risk_profile=risk_profile,
        horizon=horizon,
        top_n=top_n,
        lookback_days=lookback_days,
    )
