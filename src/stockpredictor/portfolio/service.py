"""Glue layer between `construct_portfolio` (pure) and the lake/DB (§12).

Shared by the API (api/app.py) and the Streamlit UI so both surfaces gather
data the same way -- same rationale as monitoring/accuracy.py: business
logic lives once, thin callers wrap it.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.backtest.registry import read_latest_return_calibration
from stockpredictor.common.types import DataLayer, RiskProfile
from stockpredictor.features.technical import compute_atr
from stockpredictor.portfolio.constructor import ConstructedPortfolio, construct_portfolio
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import Security

DEFAULT_STRATEGY_ID = "top_k_technical_fundamental_v1"

# How far beyond a caller's requested `top_n` we'll pull additional
# ranked candidates to fully deploy `investment_amount` when the risk
# profile's position/sector caps mean top_n alone can't (see
# `construct_portfolio_from_lake`'s docstring). A ceiling, not a target --
# a large, genuinely diversified retail portfolio; if capital still can't
# fully deploy by 50 names, that's an honest constraint (see
# `construct_portfolio`'s shortfall_note), not something to paper over by
# reaching for rank #200.
MAX_AUTO_EXPAND_POOL = 50


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


def _latest_atr_by_symbol(prices: pd.DataFrame) -> pd.Series:
    """ATR-14 as of the latest available date per symbol, computed directly
    from silver prices via the same `compute_atr` the batch feature
    pipeline uses (features/technical.py) -- not read from a pre-computed
    gold/features snapshot. That snapshot is fully regenerated from silver
    every nightly run anyway (no incremental dependency on a prior copy),
    so persisting it in git just for this one column would mean committing
    ~250MB of mostly-redundant data every night for no benefit (see
    .gitignore). `compute_atr` isn't multi-symbol-safe on its own -- its
    rolling/ewm calculations would blend across symbol boundaries if fed a
    concatenated multi-symbol frame -- so it's called once per symbol's own
    date-sorted sub-frame, exactly as features/registry.py's
    build_technical_features_for_universe does."""
    if prices.empty:
        return pd.Series(dtype="float64")
    values: dict[str, float] = {}
    for symbol, group in prices.groupby("symbol"):
        atr = compute_atr(group.sort_values("date"))["atr_14"].dropna()
        if not atr.empty:
            values[symbol] = atr.iloc[-1]
    return pd.Series(values, dtype="float64")


def _build_portfolio_for_pool(
    lake: Lake,
    session_factory: sessionmaker[Session],
    ranked: pd.DataFrame,
    return_calibration: pd.DataFrame,
    horizon: str,
    risk_profile: RiskProfile,
    pool_size: int,
    lookback_days: int,
    investment_amount: float | None,
) -> ConstructedPortfolio:
    """One attempt at building a portfolio from the top `pool_size` ranked
    candidates -- fetches prices/ATR/sector data for exactly that many
    symbols (see `_read_for_symbols`'s docstring on why this stays scoped
    rather than reading the whole universe), then delegates the actual
    allocation math to the pure `construct_portfolio`."""
    candidates = ranked.sort_values("rank").head(pool_size)
    symbols = candidates["symbol"].tolist()

    prices = _read_for_symbols(lake, DataLayer.SILVER, "prices", symbols)
    atr_by_symbol = _latest_atr_by_symbol(prices)

    session = session_factory()
    try:
        secs = session.execute(select(Security).where(Security.symbol.in_(symbols))).scalars().all()
        sector_by_symbol = pd.Series({s.symbol: s.sector for s in secs})
    finally:
        session.close()

    return construct_portfolio(
        ranked=candidates,
        prices=prices,
        atr_by_symbol=atr_by_symbol,
        sector_by_symbol=sector_by_symbol,
        return_calibration=return_calibration,
        risk_profile=risk_profile,
        horizon=horizon,
        top_n=pool_size,
        lookback_days=lookback_days,
        investment_amount=investment_amount,
    )


def construct_portfolio_from_lake(
    lake: Lake,
    session_factory: sessionmaker[Session],
    horizon: str,
    risk_profile: RiskProfile,
    top_n: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    lookback_days: int = 90,
    investment_amount: float | None = None,
) -> ConstructedPortfolio | None:
    """Returns None if there are no published rankings for `horizon` yet --
    callers decide how to present that (404 in the API, an info message in
    the UI).

    `top_n` is a *floor* on candidates considered, not a hard target: a
    tight risk profile's position/sector caps (risk_profiles.py) create a
    hard ceiling on how much of `investment_amount` can go into just
    `top_n` names -- e.g. Conservative's 10% position cap means 5 names can
    never absorb more than 50% of capital, no matter how the optimizer
    weights them. That's a real, correct constraint on *concentration*, but
    it shouldn't silently double as a constraint on *how much of the
    investor's own stated amount gets deployed at all* -- those are
    different questions (see constructor.py's `construct_portfolio`
    docstring and the module-level MAX_AUTO_EXPAND_POOL comment). So: if
    `top_n` candidates can't fully deploy `investment_amount` under the
    chosen risk profile, this reaches further down the ranked list (more
    diversified, lower-conviction names) and retries, up to
    `MAX_AUTO_EXPAND_POOL`, before accepting a genuine shortfall. The
    returned portfolio's `diversification_warning` says explicitly when
    more names were used than requested, and `len(portfolio.positions)` --
    not the original `top_n` -- is the real count actually allocated
    across."""
    ranked = read_latest_rankings(lake, horizon)
    if ranked.empty:
        return None

    return_calibration = read_latest_return_calibration(lake, strategy_id, horizon)

    requested_top_n = top_n
    pool_size = top_n
    max_pool = min(len(ranked), max(top_n, MAX_AUTO_EXPAND_POOL))

    portfolio = _build_portfolio_for_pool(
        lake, session_factory, ranked, return_calibration, horizon, risk_profile,
        pool_size, lookback_days, investment_amount,
    )
    while portfolio.total_allocated_weight < 0.999 and pool_size < max_pool:
        pool_size = min(max_pool, max(pool_size + 1, pool_size * 2))
        portfolio = _build_portfolio_for_pool(
            lake, session_factory, ranked, return_calibration, horizon, risk_profile,
            pool_size, lookback_days, investment_amount,
        )

    if pool_size > requested_top_n and len(portfolio.positions) > requested_top_n:
        expansion_note = (
            f"Requested top {requested_top_n}; expanded to {len(portfolio.positions)} names to "
            f"more fully deploy the investment amount under the {risk_profile.value} profile's "
            "diversification requirements."
        )
        portfolio = replace(
            portfolio,
            diversification_warning=(
                f"{portfolio.diversification_warning} {expansion_note}"
                if portfolio.diversification_warning
                else expansion_note
            ),
        )

    return portfolio
