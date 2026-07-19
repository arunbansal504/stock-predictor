from __future__ import annotations

import datetime as dt

from stockpredictor.common.trading_calendar import last_completed_nse_session


def test_before_close_returns_previous_weekday():
    # Wed 2026-07-15 09:00 IST = 2026-07-15 03:30 UTC -- before the 16:00 IST cutoff.
    now = dt.datetime(2026, 7, 15, 3, 30, tzinfo=dt.timezone.utc)
    assert last_completed_nse_session(now) == dt.date(2026, 7, 14)


def test_after_close_returns_same_weekday():
    # Wed 2026-07-15 19:00 IST = 2026-07-15 13:30 UTC -- after the 16:00 IST cutoff.
    now = dt.datetime(2026, 7, 15, 13, 30, tzinfo=dt.timezone.utc)
    assert last_completed_nse_session(now) == dt.date(2026, 7, 15)


def test_monday_before_close_rolls_back_to_friday():
    # Mon 2026-07-20 09:00 IST = Mon 2026-07-20 03:30 UTC -- before close, so
    # steps back to Sunday, then rolls back over the weekend to Friday.
    now = dt.datetime(2026, 7, 20, 3, 30, tzinfo=dt.timezone.utc)
    assert last_completed_nse_session(now) == dt.date(2026, 7, 17)


def test_saturday_rolls_back_to_friday():
    now = dt.datetime(2026, 7, 18, 13, 30, tzinfo=dt.timezone.utc)  # Sat, well after any cutoff
    assert last_completed_nse_session(now) == dt.date(2026, 7, 17)


def test_naive_datetime_treated_as_utc():
    now = dt.datetime(2026, 7, 15, 13, 30)  # naive, after-close in IST if treated as UTC
    assert last_completed_nse_session(now) == dt.date(2026, 7, 15)


def test_defaults_to_current_time_when_now_is_none():
    result = last_completed_nse_session()
    assert isinstance(result, dt.date)
    assert result <= dt.date.today()


def test_result_is_never_a_weekend():
    # Sweep a full week's worth of hourly ticks -- the result must always be
    # a Mon-Fri date regardless of when in the week `now` falls.
    start = dt.datetime(2026, 7, 13, 0, 0, tzinfo=dt.timezone.utc)  # Monday
    for hours in range(0, 24 * 7, 3):
        now = start + dt.timedelta(hours=hours)
        result = last_completed_nse_session(now)
        assert result.weekday() < 5, f"now={now} -> {result} ({result.strftime('%A')})"
