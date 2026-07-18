from __future__ import annotations

from pathlib import Path

import pytest

from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake


@pytest.fixture
def tmp_lake(tmp_path: Path) -> Lake:
    return Lake(root=tmp_path / "lake")


@pytest.fixture
def db_sessionmaker(tmp_path: Path):
    """A fresh, isolated SQLite DB per test — no shared state, no need for a
    running Postgres to run the test suite (matches the MVP's zero-config
    dev/test posture, see common/config.py)."""
    from stockpredictor.common.config import Settings

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    engine = make_engine(settings)
    init_db(engine)
    return make_sessionmaker(engine)
