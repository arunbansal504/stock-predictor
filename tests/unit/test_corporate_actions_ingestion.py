from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select

from stockpredictor.ingestion import corporate_actions as ca_ingestion
from stockpredictor.storage.models import CorporateAction, Security


def _seed_security(db_sessionmaker, symbol: str = "TCS") -> None:
    session = db_sessionmaker()
    try:
        session.add(Security(symbol=symbol, exchange="NSE", name="Tata Consultancy Services", sector="IT"))
        session.commit()
    finally:
        session.close()


def _fake_actions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "TCS",
                "action_type": "split",
                "ex_date": dt.date(2018, 5, 31),
                "knowable_date": dt.date(2018, 5, 31),
                "ratio": 2.0,
                "value": None,
            }
        ],
        columns=["symbol", "action_type", "ex_date", "knowable_date", "ratio", "value"],
    )


def test_sync_corporate_actions_inserts_new_row(db_sessionmaker, monkeypatch):
    _seed_security(db_sessionmaker)
    monkeypatch.setattr(ca_ingestion, "fetch_corporate_actions", lambda symbols, exchange="NSE": _fake_actions_df())

    n = ca_ingestion.sync_corporate_actions(db_sessionmaker, ["TCS"])
    assert n == 1

    session = db_sessionmaker()
    try:
        rows = session.execute(select(CorporateAction)).scalars().all()
        assert len(rows) == 1
        assert rows[0].symbol == "TCS"
        assert rows[0].ratio == 2.0
    finally:
        session.close()


def test_sync_corporate_actions_upserts_on_natural_key(db_sessionmaker, monkeypatch):
    _seed_security(db_sessionmaker)

    df_v1 = _fake_actions_df()
    monkeypatch.setattr(ca_ingestion, "fetch_corporate_actions", lambda symbols, exchange="NSE": df_v1)
    ca_ingestion.sync_corporate_actions(db_sessionmaker, ["TCS"])

    df_v2 = df_v1.copy()
    df_v2.loc[0, "ratio"] = 3.0  # simulate a corrected ratio on re-ingest
    monkeypatch.setattr(ca_ingestion, "fetch_corporate_actions", lambda symbols, exchange="NSE": df_v2)
    ca_ingestion.sync_corporate_actions(db_sessionmaker, ["TCS"])

    session = db_sessionmaker()
    try:
        rows = session.execute(select(CorporateAction)).scalars().all()
        assert len(rows) == 1  # still one row -- updated, not duplicated
        assert rows[0].ratio == 3.0
    finally:
        session.close()


def test_sync_corporate_actions_empty_fetch_returns_zero(db_sessionmaker, monkeypatch):
    _seed_security(db_sessionmaker)
    monkeypatch.setattr(
        ca_ingestion,
        "fetch_corporate_actions",
        lambda symbols, exchange="NSE": pd.DataFrame(
            columns=["symbol", "action_type", "ex_date", "knowable_date", "ratio", "value"]
        ),
    )
    n = ca_ingestion.sync_corporate_actions(db_sessionmaker, ["TCS"])
    assert n == 0
