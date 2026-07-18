"""FastAPI dependencies -- kept as thin, override-able functions so tests can
swap in a tmp_path-backed Lake via `app.dependency_overrides` instead of
touching the real data directory."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.storage.db import make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake


def get_lake() -> Lake:
    return Lake()


def get_db_sessionmaker() -> sessionmaker[Session]:
    return make_sessionmaker(make_engine())
