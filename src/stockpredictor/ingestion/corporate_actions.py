"""Ingestion for corporate actions into the relational store (§13: small,
PIT-critical reference data — see storage/models.py:CorporateAction).

Requires the acting symbols to already exist in `securities` (foreign key) --
run ingestion.universe.sync_universe first.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.common.logging import get_logger
from stockpredictor.connectors.corporate_actions_yfinance import fetch_corporate_actions
from stockpredictor.storage.models import CorporateAction

logger = get_logger(__name__)


def sync_corporate_actions(
    session_factory: sessionmaker[Session],
    symbols: list[str],
    exchange: str = "NSE",
) -> int:
    """Fetch and upsert corporate actions for `symbols` on their natural key
    (symbol, action_type, ex_date). Returns the number of actions synced."""
    df = fetch_corporate_actions(symbols, exchange)
    if df.empty:
        return 0

    session = session_factory()
    try:
        existing = {
            (a.symbol, a.action_type, a.ex_date): a
            for a in session.execute(select(CorporateAction)).scalars()
        }
        for row in df.itertuples(index=False):
            key = (row.symbol, row.action_type, row.ex_date)
            action = existing.get(key)
            if action is None:
                session.add(
                    CorporateAction(
                        symbol=row.symbol,
                        action_type=row.action_type,
                        ex_date=row.ex_date,
                        knowable_date=row.knowable_date,
                        ratio=row.ratio,
                        value=row.value,
                    )
                )
            else:
                action.knowable_date = row.knowable_date
                action.ratio = row.ratio
                action.value = row.value
        session.commit()
        logger.info("Synced %d corporate actions", len(df))
        return len(df)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
