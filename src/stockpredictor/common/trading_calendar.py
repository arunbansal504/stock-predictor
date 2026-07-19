"""Pins a pipeline run's data horizon to the last completed NSE session.

Without this, a manual run's `end` date is `date.today()` regardless of
wall-clock time -- an intraday rerun ingests today's partial/live bar (and
even a post-close rerun can pick up a yfinance revision to today's Adj
Close), so two runs on the same day see different "latest" data and
therefore retrain to different ranks. Pinning `end` to the last *completed*
session means every run before tomorrow's close sees the identical set of
resolved trading days -- see orchestration/nightly_flow.py's ingestion
`end` and the `as_of` threading through features/labels/prediction/ranking.

NSE trades 09:15-15:30 IST, Mon-Fri (holidays aside). No holiday calendar is
needed here: a holiday date simply has no price bar anywhere downstream, so
treating it as "the session" is harmless -- feature/label code already
operates on whatever dates are actually present in the data.
"""

from __future__ import annotations

import datetime as dt

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
# NSE closes 15:30 IST; a half-hour settle buffer before treating today's
# session as complete.
SESSION_COMPLETE_HOUR_IST = 16


def last_completed_nse_session(now: dt.datetime | None = None) -> dt.date:
    """The most recent NSE trading date whose session has fully closed, as
    of `now` (defaults to the current UTC time)."""
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(dt.timezone.utc).replace(tzinfo=None)

    ist = now + IST_OFFSET
    session_date = ist.date()
    if ist.hour < SESSION_COMPLETE_HOUR_IST:
        session_date -= dt.timedelta(days=1)

    while session_date.weekday() >= 5:  # Saturday=5, Sunday=6
        session_date -= dt.timedelta(days=1)

    return session_date
