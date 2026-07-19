"""Universe loader (§27 Phase 1 step 3, §29; expanded per the roadmap to
pull NSE's real NIFTY 500 constituent list -- see connectors/universe_nse.py).

Two sources, same downstream schema (symbol, exchange, name, sector):

- `sync_universe_from_nse`: the live, current NIFTY 500 membership from
  NSE. This is what the nightly pipeline uses.
- `sync_universe` (CSV-based): reads config/universe_seed.csv, a small
  hand-picked set of liquid large-caps. Kept for offline/deterministic
  tests and as a documented fallback if NSE's feed is unreachable (§5:
  every free/unofficial source needs one) -- see orchestration/nightly_flow.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.common.config import REPO_ROOT, load_yaml_config
from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.connectors.universe_nse import fetch_nifty500_constituents
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import Security

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"symbol", "exchange", "name", "sector"}

MEMBERSHIP_GOLD_DOMAIN = "universe_membership"
MEMBERSHIP_KEY_COLS = ["date", "symbol"]
_MEMBERSHIP_FILE_KEY = "membership"


def load_universe_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """Read and validate the universe seed CSV. Raises if required columns are
    missing or symbols are duplicated (both would silently corrupt the
    securities master otherwise)."""
    if csv_path is None:
        cfg = load_yaml_config("universe.yaml")
        csv_path = REPO_ROOT / cfg["seed_file"]

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV {csv_path} missing required columns: {missing}")

    dupes = df[df.duplicated(subset="symbol", keep=False)]
    if not dupes.empty:
        raise ValueError(f"Universe CSV {csv_path} has duplicate symbols: {sorted(dupes['symbol'].unique())}")

    return df


def _upsert_securities(session_factory: sessionmaker[Session], df: pd.DataFrame) -> int:
    """Shared upsert core for both universe sources: update existing rows
    for a symbol in place (name/sector/exchange), insert new ones. Symbols
    that dropped out of the source (e.g. left the index) are left
    untouched, not deleted -- their historical data is still valid for
    backtests; tracking "no longer in the active universe" explicitly is a
    later-phase concern, not something to silently destroy data over now."""
    session = session_factory()
    try:
        existing = {s.symbol: s for s in session.execute(select(Security)).scalars()}
        for row in df.itertuples(index=False):
            sec = existing.get(row.symbol)
            if sec is None:
                sec = Security(
                    symbol=row.symbol,
                    exchange=row.exchange,
                    name=row.name,
                    sector=row.sector,
                )
                session.add(sec)
            else:
                sec.exchange = row.exchange
                sec.name = row.name
                sec.sector = row.sector
                sec.is_active = True
        session.commit()
        return len(df)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def sync_universe(
    session_factory: sessionmaker[Session],
    csv_path: Path | None = None,
) -> int:
    """Upsert the bundled seed CSV into `securities`. See module docstring
    -- this is the offline/fallback path; `sync_universe_from_nse` is what
    the nightly pipeline actually uses."""
    df = load_universe_csv(csv_path)
    n = _upsert_securities(session_factory, df)
    logger.info("Synced %d securities into universe (source=csv)", n)
    return n


def get_active_symbols(session_factory: sessionmaker[Session]) -> list[str]:
    """Every symbol currently on record in `securities`, regardless of
    source -- the last-known-good universe. Used as the middle rung of
    orchestration/nightly_flow.py's universe fallback: if today's live NSE
    fetch fails, reusing yesterday's already-synced ~500-symbol universe is
    a much smaller rank-composition shift than collapsing all the way down
    to the 40-symbol CSV seed (which would make cross-sectional `_xrank`
    features -- and therefore ranks -- swing sharply for reasons that have
    nothing to do with any stock's actual behavior)."""
    session = session_factory()
    try:
        rows = session.execute(select(Security.symbol)).scalars()
        return sorted(rows)
    finally:
        session.close()


def persist_universe_membership(lake: Lake, as_of: dt.date, symbols: list[str]) -> int:
    """Snapshot which symbols were in the tradable universe on `as_of`, one
    row per (date, symbol) -- accrues one snapshot per nightly run so a
    later backtest can restrict each historical rebalance date to the
    universe as it stood point-in-time, instead of (the prior behavior)
    applying TODAY's membership across 5 years of history, which silently
    excludes delisted/demoted names and includes recently-added winners
    (survivorship bias). History before this started recording remains
    biased -- an honest, documented gap; free data provides no retroactive
    fix for it."""
    if not symbols:
        return 0
    df = pd.DataFrame({"date": [pd.Timestamp(as_of)] * len(symbols), "symbol": symbols})
    return lake.write(df, DataLayer.GOLD, MEMBERSHIP_GOLD_DOMAIN, _MEMBERSHIP_FILE_KEY, key_cols=MEMBERSHIP_KEY_COLS)


def read_universe_membership(lake: Lake) -> pd.DataFrame:
    """Every recorded (date, symbol) universe-membership snapshot -- see
    persist_universe_membership. Empty before any nightly run has recorded
    one."""
    return lake.read(DataLayer.GOLD, MEMBERSHIP_GOLD_DOMAIN, _MEMBERSHIP_FILE_KEY)


def get_security_names(session_factory: sessionmaker[Session], symbols: list[str]) -> dict[str, str]:
    """symbol -> company name for the given symbols, from `securities`.
    Used by news ingestion (ingestion/news.py via orchestration/
    nightly_flow.py), since a bare ticker is too generic a search term on
    its own (see connectors/news_rss.py's docstring). Symbols with no
    matching row are simply absent from the result, not an error -- the
    caller skips news ingestion for those rather than guessing a name."""
    session = session_factory()
    try:
        rows = session.execute(select(Security).where(Security.symbol.in_(symbols))).scalars()
        return {s.symbol: s.name for s in rows}
    finally:
        session.close()


def sync_universe_from_nse(session_factory: sessionmaker[Session]) -> pd.DataFrame:
    """Fetch NSE's current NIFTY 500 constituent list live and upsert into
    `securities`. Returns the fetched DataFrame (not just a count) so
    callers can get the symbol list without a second read -- see
    orchestration/nightly_flow.py's task_sync_universe."""
    df = fetch_nifty500_constituents()
    n = _upsert_securities(session_factory, df[list(REQUIRED_COLUMNS)])
    logger.info("Synced %d securities into universe (source=nse_live)", n)
    return df
