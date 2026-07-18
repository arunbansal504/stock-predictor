"""Run-status queries over `run_metadata` (§23: "run success/failure...
run duration"), the read side of orchestration/run_tracking.py's writes.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from stockpredictor.storage.models import RunMetadata


def get_recent_runs(session_factory: sessionmaker[Session], limit: int = 20) -> list[dict]:
    session = session_factory()
    try:
        rows = (
            session.execute(
                select(RunMetadata).order_by(RunMetadata.started_at.desc()).limit(limit)
            )
            .scalars()
            .all()
        )
        return [
            {
                "run_id": r.run_id,
                "stage": r.stage,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "rows_processed": r.rows_processed,
                "detail": r.detail,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_latest_run_summary(session_factory: sessionmaker[Session]) -> dict | None:
    """Status of the most recent complete run (grouped by run_id): overall
    status is "failed" if any stage failed, "running" if any stage hasn't
    finished, else "success"."""
    session = session_factory()
    try:
        latest_run_id = session.execute(
            select(RunMetadata.run_id).order_by(RunMetadata.started_at.desc()).limit(1)
        ).scalar_one_or_none()
        if latest_run_id is None:
            return None

        rows = (
            session.execute(select(RunMetadata).where(RunMetadata.run_id == latest_run_id))
            .scalars()
            .all()
        )
    finally:
        session.close()

    statuses = {r.stage: r.status for r in rows}
    if any(s == "failed" for s in statuses.values()):
        overall = "failed"
    elif any(s == "running" for s in statuses.values()):
        overall = "running"
    else:
        overall = "success"

    return {"run_id": latest_run_id, "overall_status": overall, "stages": statuses}
