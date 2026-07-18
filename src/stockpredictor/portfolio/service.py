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


def _read_for_symbols(lake: Lake, layer: DataLayer, domain: str, symbols: list[str]) -> pd.DataFrame:
    """Read only the given symbols' per-symbol lake files, not the whole
    universe. The lake already stores one Parquet file per symbol (see
    storage/lake.py) -- `lake.read_all` (a DuckDB glob over every symbol's
    file) is the right tool for genuinely cross-sectional reads, but for a
    top-N candidate list (typically 10-50 symbols out of ~500) it means
    loading the *entire* universe's full history into memory only to
    immediately discard 90%+ of it. That's not just slow -- reading the
    full gold/features table (~250MB, every symbol's 5-year history) for
    every portfolio request is exactly the kind of memory spike that OOM-
    kills a request on a resource-constrained host (observed live on
    Streamlit Community Cloud's free tier)."""
    frames = [lake.read(layer, domain, s) for s in symbols]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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

    prices = _read_for_symbols(lake, DataLayer.SILVER, "prices", symbols)

    features = _read_for_symbols(lake, DataLayer.GOLD, "features", symbols)
    if not features.empty:
        features = features.sort_values("date").groupby("symbol").tail(1)
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
