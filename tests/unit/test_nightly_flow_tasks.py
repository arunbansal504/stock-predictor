"""Unit-level tests for individual nightly_flow tasks, distinct from the
full end-to-end integration test (tests/integration/test_nightly_flow.py).
Focused here on task_sync_universe's live-NSE-first, CSV-fallback behavior.

Calls task functions via `.fn` (the raw undecorated function Prefect's
@task wraps) rather than invoking them as real Prefect tasks: calling a
task directly outside a @flow context spins up a full temporary Prefect
server (~10+ seconds, observed directly, not just theoretical) purely to
run one function -- unnecessary cost for testing task logic in isolation.
This only works because nightly_flow.py's tasks use the module-level
`logger` (common/logging.py) rather than Prefect's `get_run_logger()`,
which requires an active task/flow run context and would otherwise force
every task-level test through that same expensive path.
"""

from __future__ import annotations

import pandas as pd

import stockpredictor.ingestion.universe as universe_ingestion
from stockpredictor.orchestration import nightly_flow
from stockpredictor.storage.models import Security


def _fake_nse_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": "LIVEFAKE", "exchange": "NSE", "name": "Live Fake Ltd.", "sector": "IT", "isin": "INE000000001"}]
    )


def test_task_sync_universe_prefers_live_nse_when_available(db_sessionmaker, monkeypatch):
    # Patch at the connector level (as looked up inside ingestion/universe.py),
    # not sync_universe_from_nse itself -- its own upsert side effect is
    # exactly what this test needs to exercise, not bypass.
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", lambda: _fake_nse_df())

    symbols = nightly_flow.task_sync_universe.fn(db_sessionmaker)
    assert symbols == ["LIVEFAKE"]

    session = db_sessionmaker()
    try:
        assert session.get(Security, "LIVEFAKE") is not None
    finally:
        session.close()


def test_task_sync_universe_falls_back_to_csv_when_nse_fails(db_sessionmaker, monkeypatch):
    def fail():
        raise RuntimeError("simulated NSE outage")

    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", fail)
    monkeypatch.setattr(nightly_flow, "send_alert", lambda *a, **k: False)

    symbols = nightly_flow.task_sync_universe.fn(db_sessionmaker)
    # Falls back to the bundled CSV seed -- should contain the well-known
    # large-caps from config/universe_seed.csv, not the (failed) live fetch.
    assert "RELIANCE" in symbols
    assert "LIVEFAKE" not in symbols
