"""Daily validation: resolve published predictions whose horizon has
completed, store outcomes + risk metrics, and regenerate the static
performance dashboard (ML Review Board spec Part 2 + Part 3). Intended to
run daily via .github/workflows/daily_validation.yml, after nightly's run.

Usage:  .venv/Scripts/python.exe scripts/run_daily_validation.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockpredictor.common.config import REPO_ROOT
from stockpredictor.common.logging import get_logger
from stockpredictor.monitoring.alerts import send_alert
from stockpredictor.orchestration.run_tracking import run_tracked_stage
from stockpredictor.reporting.analytics import compute_performance_analytics
from stockpredictor.reporting.dashboard import render_dashboard_html
from stockpredictor.reporting.validation import run_daily_validation
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

logger = get_logger("run_daily_validation")

DASHBOARD_PATH = REPO_ROOT / "reports" / "dashboard" / "index.html"


def main() -> None:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    lake = Lake()
    engine = make_engine()
    init_db(engine)
    session_factory = make_sessionmaker(engine)

    try:
        n_validated = run_tracked_stage(
            session_factory, run_id, "daily_validation", run_daily_validation, lake, session_factory
        )
        analytics = run_tracked_stage(
            session_factory, run_id, "compute_analytics", compute_performance_analytics, lake, session_factory
        )
        run_tracked_stage(
            session_factory, run_id, "render_dashboard", render_dashboard_html, analytics, DASHBOARD_PATH
        )
    except Exception as exc:
        send_alert(f"Daily validation run {run_id} failed: {exc}", level="error")
        raise

    logger.info("Daily validation run %s complete: %d predictions validated", run_id, n_validated)


if __name__ == "__main__":
    main()
