from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from stockpredictor.ingestion import universe as universe_ingestion
from stockpredictor.ingestion.universe import (
    get_security_names,
    load_universe_csv,
    sync_universe,
    sync_universe_from_nse,
)
from stockpredictor.storage.models import Security


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "universe.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_universe_csv_reads_real_seed_file():
    # Exercises the actual bundled seed file, not just synthetic fixtures --
    # catches accidental corruption of config/universe_seed.csv.
    df = load_universe_csv()
    assert len(df) > 0
    assert {"symbol", "exchange", "name", "sector"}.issubset(df.columns)
    assert df["symbol"].is_unique


def test_load_universe_csv_rejects_missing_columns(tmp_path):
    path = _write_csv(tmp_path, [{"symbol": "AAA", "exchange": "NSE"}])
    with pytest.raises(ValueError, match="missing required columns"):
        load_universe_csv(path)


def test_load_universe_csv_rejects_duplicate_symbols(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {"symbol": "AAA", "exchange": "NSE", "name": "A Corp", "sector": "IT"},
            {"symbol": "AAA", "exchange": "NSE", "name": "A Corp Dup", "sector": "IT"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate symbols"):
        load_universe_csv(path)


def test_sync_universe_inserts_new_securities(tmp_path, db_sessionmaker):
    path = _write_csv(
        tmp_path,
        [
            {"symbol": "AAA", "exchange": "NSE", "name": "A Corp", "sector": "IT"},
            {"symbol": "BBB", "exchange": "NSE", "name": "B Corp", "sector": "Financials"},
        ],
    )
    n = sync_universe(db_sessionmaker, csv_path=path)
    assert n == 2

    session = db_sessionmaker()
    try:
        symbols = {s.symbol for s in session.execute(select(Security)).scalars()}
        assert symbols == {"AAA", "BBB"}
    finally:
        session.close()


def test_sync_universe_upserts_existing_symbol(tmp_path, db_sessionmaker):
    path1 = _write_csv(
        tmp_path, [{"symbol": "AAA", "exchange": "NSE", "name": "Old Name", "sector": "IT"}]
    )
    sync_universe(db_sessionmaker, csv_path=path1)

    path2 = _write_csv(
        tmp_path, [{"symbol": "AAA", "exchange": "NSE", "name": "New Name", "sector": "Pharma"}]
    )
    sync_universe(db_sessionmaker, csv_path=path2)

    session = db_sessionmaker()
    try:
        sec = session.get(Security, "AAA")
        assert sec.name == "New Name"
        assert sec.sector == "Pharma"
    finally:
        session.close()


def _fake_nse_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "RELIANCE", "exchange": "NSE", "name": "Reliance Industries Ltd.", "sector": "Energy", "isin": "INE002A01018"},
            {"symbol": "TCS", "exchange": "NSE", "name": "Tata Consultancy Services Ltd.", "sector": "IT", "isin": "INE467B01029"},
        ]
    )


def test_sync_universe_from_nse_inserts_and_returns_full_frame(db_sessionmaker, monkeypatch):
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_df)

    df = sync_universe_from_nse(db_sessionmaker)
    assert len(df) == 2
    assert "isin" in df.columns  # caller gets the full fetched frame, not just the DB-stored subset

    session = db_sessionmaker()
    try:
        symbols = {s.symbol for s in session.execute(select(Security)).scalars()}
        assert symbols == {"RELIANCE", "TCS"}
    finally:
        session.close()


def test_sync_universe_from_nse_upserts_existing_symbol(db_sessionmaker, monkeypatch):
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_df)
    sync_universe_from_nse(db_sessionmaker)

    updated = _fake_nse_df()
    updated.loc[updated["symbol"] == "TCS", "sector"] = "Technology"
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", lambda: updated)
    sync_universe_from_nse(db_sessionmaker)

    session = db_sessionmaker()
    try:
        sec = session.get(Security, "TCS")
        assert sec.sector == "Technology"
    finally:
        session.close()


def test_sync_universe_from_nse_propagates_fetch_failures(db_sessionmaker, monkeypatch):
    def fail():
        raise ValueError("NSE unreachable")

    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", fail)
    with pytest.raises(ValueError, match="NSE unreachable"):
        sync_universe_from_nse(db_sessionmaker)


def test_get_security_names_returns_symbol_to_name_mapping(db_sessionmaker, monkeypatch):
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_df)
    sync_universe_from_nse(db_sessionmaker)

    names = get_security_names(db_sessionmaker, ["RELIANCE", "TCS"])
    assert names == {"RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd."}


def test_get_security_names_omits_unknown_symbols(db_sessionmaker, monkeypatch):
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_df)
    sync_universe_from_nse(db_sessionmaker)

    names = get_security_names(db_sessionmaker, ["RELIANCE", "NOTREAL"])
    assert names == {"RELIANCE": "Reliance Industries Ltd."}
