"""End-to-end smoke test on a small universe (§22): universe sync -> price
ingestion for every synced symbol -> data lands correctly in both the
relational store and the analytical lake. No network calls (fetch_prices is
monkeypatched) -- this is testing pipeline *wiring*, not the live data
source, which the contract tests already cover in isolation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select

from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion import prices as prices_ingestion
from stockpredictor.ingestion.universe import sync_universe
from stockpredictor.storage.models import Security


def _fake_prices_for(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol, symbol],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "adj_close": [10.2, 11.2],
            "volume": pd.array([1000, 1200], dtype="int64"),
            "source": ["yfinance", "yfinance"],
        }
    )


def test_nightly_smoke_universe_then_prices(tmp_lake, db_sessionmaker, monkeypatch):
    monkeypatch.setattr(
        prices_ingestion.prices_yfinance,
        "fetch_prices",
        lambda symbols, start, end, exchange="NSE": _fake_prices_for(symbols[0]),
    )

    # Step 1: sync the (real, bundled) universe seed into the relational store.
    synced = sync_universe(db_sessionmaker)
    assert synced > 0

    session = db_sessionmaker()
    try:
        symbols = [s.symbol for s in session.execute(select(Security)).scalars()]
    finally:
        session.close()

    # Step 2: ingest EOD prices for every symbol in the universe.
    total_rows = 0
    failures = []
    for symbol in symbols:
        try:
            rows = prices_ingestion.ingest_symbol_prices(
                tmp_lake, symbol, dt.date(2024, 1, 1), dt.date(2024, 1, 2)
            )
            total_rows += rows
        except Exception as exc:  # a single bad symbol must not kill the run (§3 NFR)
            failures.append((symbol, exc))

    assert not failures, f"unexpected per-symbol failures: {failures}"
    assert total_rows == 2 * len(symbols)

    # Step 3: the cross-sectional read path (what feature code will use) sees everyone.
    all_silver = tmp_lake.read_all(DataLayer.SILVER, "prices")
    assert set(all_silver["symbol"]) == set(symbols)
    assert len(all_silver) == 2 * len(symbols)
