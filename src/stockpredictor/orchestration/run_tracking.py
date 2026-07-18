"""Run audit trail (§13 `run_metadata` table, §23 monitoring: "run
success/failure... run duration"). Every DAG stage gets a row so a failed
nightly run is inspectable without digging through logs.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.storage.models import RunMetadata


def start_stage(session_factory: sessionmaker[Session], run_id: str, stage: str) -> int:
    session = session_factory()
    try:
        row = RunMetadata(run_id=run_id, stage=stage, status="running")
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id
    finally:
        session.close()


def finish_stage(
    session_factory: sessionmaker[Session],
    stage_id: int,
    status: str,
    rows_processed: int | None = None,
    detail: str | None = None,
) -> None:
    if status not in ("success", "failed"):
        raise ValueError(f"status must be 'success' or 'failed', got {status!r}")
    session = session_factory()
    try:
        row = session.get(RunMetadata, stage_id)
        if row is None:
            raise ValueError(f"No run_metadata row with id={stage_id}")
        row.status = status
        row.finished_at = dt.datetime.now(dt.timezone.utc)
        row.rows_processed = rows_processed
        row.detail = detail[:2048] if detail else None
        session.commit()
    finally:
        session.close()


def run_tracked_stage(
    session_factory: sessionmaker[Session],
    run_id: str,
    stage: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run `fn(*args, **kwargs)`, recording a run_metadata row for `stage`
    (§13, §23: the audit trail behind "run success/failure... run
    duration"). Re-raises any exception after marking the stage failed --
    this is instrumentation layered on top of the pipeline, not error
    suppression; a failing stage must still halt the orchestration flow."""
    stage_id = start_stage(session_factory, run_id, stage)
    try:
        result = fn(*args, **kwargs)
        rows = result if isinstance(result, int) else None
        finish_stage(session_factory, stage_id, "success", rows_processed=rows)
        return result
    except Exception as exc:
        finish_stage(session_factory, stage_id, "failed", detail=str(exc))
        raise
