"""Weekly publish: freeze this week's official Top-N recommendation set into
`published_predictions` + `predictions/YYYY-MM-DD.{csv,json}` (ML Review
Board spec Part 1). Intended to run every Friday via
.github/workflows/weekly_prediction.yml, after nightly's daily run for that
same date has already committed fresh silver data.

Usage:  .venv/Scripts/python.exe scripts/publish_weekly_predictions.py [horizon] [top_k]
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockpredictor.common.logging import get_logger
from stockpredictor.monitoring.alerts import send_alert
from stockpredictor.orchestration.run_tracking import run_tracked_stage
from stockpredictor.reporting.publish import publish_weekly_predictions
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

logger = get_logger("publish_weekly_predictions")

DEFAULT_HORIZON = "90d"
DEFAULT_TOP_K = 10


def main() -> None:
    horizon = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HORIZON
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    lake = Lake()
    engine = make_engine()
    init_db(engine)
    session_factory = make_sessionmaker(engine)

    try:
        published = run_tracked_stage(
            session_factory, run_id, f"publish_weekly[{horizon}]",
            publish_weekly_predictions, lake, session_factory, horizon, top_k,
        )
    except Exception as exc:
        send_alert(f"Weekly publish run {run_id} failed: {exc}", level="error")
        raise

    logger.info("Weekly publish run %s complete: %d new predictions published", run_id, len(published))


if __name__ == "__main__":
    main()
