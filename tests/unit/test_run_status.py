from __future__ import annotations

from stockpredictor.monitoring.run_status import get_latest_run_summary, get_recent_runs
from stockpredictor.orchestration.run_tracking import finish_stage, start_stage


def test_get_recent_runs_empty(db_sessionmaker):
    assert get_recent_runs(db_sessionmaker) == []


def test_get_recent_runs_returns_most_recent_first(db_sessionmaker):
    id1 = start_stage(db_sessionmaker, "run1", "stage_a")
    finish_stage(db_sessionmaker, id1, "success", rows_processed=5)
    id2 = start_stage(db_sessionmaker, "run1", "stage_b")
    finish_stage(db_sessionmaker, id2, "success", rows_processed=10)

    runs = get_recent_runs(db_sessionmaker, limit=10)
    assert len(runs) == 2
    assert runs[0]["stage"] == "stage_b"  # most recently started first
    assert runs[0]["rows_processed"] == 10


def test_get_recent_runs_respects_limit(db_sessionmaker):
    for i in range(5):
        stage_id = start_stage(db_sessionmaker, "run1", f"stage_{i}")
        finish_stage(db_sessionmaker, stage_id, "success")
    runs = get_recent_runs(db_sessionmaker, limit=2)
    assert len(runs) == 2


def test_get_latest_run_summary_none_when_no_runs(db_sessionmaker):
    assert get_latest_run_summary(db_sessionmaker) is None


def test_get_latest_run_summary_success_when_all_stages_succeed(db_sessionmaker):
    id1 = start_stage(db_sessionmaker, "run1", "stage_a")
    finish_stage(db_sessionmaker, id1, "success")
    id2 = start_stage(db_sessionmaker, "run1", "stage_b")
    finish_stage(db_sessionmaker, id2, "success")

    summary = get_latest_run_summary(db_sessionmaker)
    assert summary["run_id"] == "run1"
    assert summary["overall_status"] == "success"


def test_get_latest_run_summary_failed_when_any_stage_failed(db_sessionmaker):
    id1 = start_stage(db_sessionmaker, "run1", "stage_a")
    finish_stage(db_sessionmaker, id1, "success")
    id2 = start_stage(db_sessionmaker, "run1", "stage_b")
    finish_stage(db_sessionmaker, id2, "failed", detail="boom")

    summary = get_latest_run_summary(db_sessionmaker)
    assert summary["overall_status"] == "failed"


def test_get_latest_run_summary_running_when_stage_unfinished(db_sessionmaker):
    start_stage(db_sessionmaker, "run1", "stage_a")  # never finished

    summary = get_latest_run_summary(db_sessionmaker)
    assert summary["overall_status"] == "running"


def test_get_latest_run_summary_only_considers_most_recent_run_id(db_sessionmaker):
    id1 = start_stage(db_sessionmaker, "run1", "stage_a")
    finish_stage(db_sessionmaker, id1, "failed", detail="old failure")
    id2 = start_stage(db_sessionmaker, "run2", "stage_a")
    finish_stage(db_sessionmaker, id2, "success")

    summary = get_latest_run_summary(db_sessionmaker)
    assert summary["run_id"] == "run2"
    assert summary["overall_status"] == "success"
