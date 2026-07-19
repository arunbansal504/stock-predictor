"""Monthly ML Review Board report: a fully data-driven ML-Review + Improvement-
Proposal Markdown pair (ML Review Board spec Parts 4-7). Intended to run on
the 1st of each month via .github/workflows/monthly_ml_review.yml, after
monthly_backtest.yml's own run that same day.

Usage:  .venv/Scripts/python.exe scripts/generate_monthly_review.py [YYYY-MM]
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockpredictor.common.logging import get_logger
from stockpredictor.monitoring.alerts import send_alert
from stockpredictor.orchestration.run_tracking import run_tracked_stage
from stockpredictor.reporting.review import generate_monthly_review
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

logger = get_logger("generate_monthly_review")


def main() -> None:
    month = sys.argv[1] if len(sys.argv) > 1 else None

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    lake = Lake()
    engine = make_engine()
    init_db(engine)
    session_factory = make_sessionmaker(engine)

    try:
        review_path, proposal_path = run_tracked_stage(
            session_factory, run_id, "generate_monthly_review",
            generate_monthly_review, lake, session_factory, month,
        )
    except Exception as exc:
        send_alert(f"Monthly ML review run {run_id} failed: {exc}", level="error")
        raise

    logger.info("Monthly review run %s complete: wrote %s and %s", run_id, review_path, proposal_path)
    # Printed (not logged) so the GH Actions workflow can capture these paths
    # for the follow-up `gh issue create` step without parsing log lines.
    print(f"REVIEW_PATH={review_path}")
    print(f"PROPOSAL_PATH={proposal_path}")


if __name__ == "__main__":
    main()
