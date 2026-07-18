from __future__ import annotations

import pytest
from sqlalchemy import select

from stockpredictor.orchestration.run_tracking import finish_stage, run_tracked_stage, start_stage
from stockpredictor.storage.models import RunMetadata


def _get_row(db_sessionmaker, stage_id: int) -> RunMetadata:
    session = db_sessionmaker()
    try:
        return session.get(RunMetadata, stage_id)
    finally:
        session.close()


def test_start_stage_creates_running_row(db_sessionmaker):
    stage_id = start_stage(db_sessionmaker, run_id="run1", stage="ingest")
    row = _get_row(db_sessionmaker, stage_id)
    assert row.run_id == "run1"
    assert row.stage == "ingest"
    assert row.status == "running"
    assert row.finished_at is None


def test_finish_stage_updates_status_and_rows(db_sessionmaker):
    stage_id = start_stage(db_sessionmaker, run_id="run1", stage="ingest")
    finish_stage(db_sessionmaker, stage_id, status="success", rows_processed=42)
    row = _get_row(db_sessionmaker, stage_id)
    assert row.status == "success"
    assert row.rows_processed == 42
    assert row.finished_at is not None


def test_finish_stage_rejects_invalid_status(db_sessionmaker):
    stage_id = start_stage(db_sessionmaker, run_id="run1", stage="ingest")
    with pytest.raises(ValueError, match="status must be"):
        finish_stage(db_sessionmaker, stage_id, status="done")


def test_run_tracked_stage_records_success_and_returns_result(db_sessionmaker):
    def fn(x):
        return x * 2

    result = run_tracked_stage(db_sessionmaker, "run1", "double", fn, 21)
    assert result == 42

    session = db_sessionmaker()
    try:
        row = session.execute(select(RunMetadata).where(RunMetadata.stage == "double")).scalar_one()
    finally:
        session.close()
    assert row.status == "success"


def test_run_tracked_stage_records_failure_and_reraises(db_sessionmaker):
    def failing():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_tracked_stage(db_sessionmaker, "run1", "will_fail", failing)

    session = db_sessionmaker()
    try:
        row = session.execute(select(RunMetadata).where(RunMetadata.stage == "will_fail")).scalar_one()
    finally:
        session.close()
    assert row.status == "failed"
    assert "boom" in row.detail


def test_run_tracked_stage_records_int_result_as_rows_processed(db_sessionmaker):
    run_tracked_stage(db_sessionmaker, "run1", "counted", lambda: 7)
    session = db_sessionmaker()
    try:
        row = session.execute(select(RunMetadata).where(RunMetadata.stage == "counted")).scalar_one()
    finally:
        session.close()
    assert row.rows_processed == 7
