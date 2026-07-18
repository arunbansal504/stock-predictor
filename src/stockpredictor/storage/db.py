"""Relational (Postgres in prod / SQLite in dev+tests) engine + session setup.

Architecture doc §13: reference/dimension data (securities master, corporate
actions, run metadata, later: users/watchlists/alerts) lives here. Bulk
time-series (prices, features, predictions) lives in the Parquet lake
(storage/lake.py) instead — this DB is deliberately kept small.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.common.config import Settings, get_settings
from stockpredictor.storage.models import Base


def make_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite:///"):
        # Ensure the parent directory for a file-based SQLite DB exists.
        db_path = url.replace("sqlite:///", "", 1)
        from pathlib import Path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args, future=True)


def init_db(engine: Engine) -> None:
    """Create all tables that don't yet exist. Idempotent — safe to call on
    every process start. Real migrations (Alembic) are a Phase 2 concern once
    schema changes need to preserve production data."""
    Base.metadata.create_all(engine)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commits on clean exit, rolls back on exception."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
