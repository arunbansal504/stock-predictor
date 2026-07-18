"""Point-in-time (PIT) correctness utilities.

Architecture doc §5 / §22: look-ahead bias is "the #1 silent killer of
backtests." The discipline enforced here: every fact has both an *event date*
(what period/day it's about) and a *knowable date* (the earliest date a
researcher could actually have known it). Features computed "as of" some date
must only use facts whose knowable_date <= as_of.

For daily OHLCV, event date == knowable date (a close price is knowable once
the market closes that day). For fundamentals, corporate actions, and news,
they diverge — e.g. a quarter ending 31-Mar might not be *announced* until
20-May, so knowable_date=2024-05-20 even though the period is 2024-03-31.
Every connector/ingestion module is responsible for stamping `knowable_date`
correctly; these helpers just enforce the filtering + provide a leakage guard
that tests (tests/leakage/) call directly.
"""

from __future__ import annotations

import pandas as pd


def filter_as_of(df: pd.DataFrame, as_of: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """Keep only rows whose `date_col` is on or before `as_of`.

    Use for series where the event date itself is the knowable date (e.g.
    daily prices: today's close is knowable from today onward).
    """
    ts = pd.to_datetime(df[date_col])
    return df.loc[ts <= pd.Timestamp(as_of)].copy()


def filter_knowable_as_of(
    df: pd.DataFrame, as_of: pd.Timestamp, knowable_col: str = "knowable_date"
) -> pd.DataFrame:
    """Keep only rows knowable on or before `as_of`.

    Use for fundamentals/corporate-actions/news, where `knowable_col` is the
    publication/announcement date, distinct from the underlying event/period.
    """
    if knowable_col not in df.columns:
        raise KeyError(
            f"'{knowable_col}' column missing — every PIT-sensitive dataset must "
            "stamp a knowable_date at ingestion time (see ingestion/ modules)."
        )
    ts = pd.to_datetime(df[knowable_col])
    return df.loc[ts <= pd.Timestamp(as_of)].copy()


def latest_knowable_as_of(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_col: str,
    knowable_col: str = "knowable_date",
) -> pd.DataFrame:
    """As-of join: for each `group_col` (e.g. symbol), return the single most
    recent record knowable on or before `as_of`. This is the standard pattern
    for pulling "current fundamentals as the market knew them on date X."
    """
    filtered = filter_knowable_as_of(df, as_of, knowable_col)
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values(knowable_col)
    return filtered.groupby(group_col, as_index=False).tail(1)


def assert_pit_safe(
    df: pd.DataFrame,
    as_of_col: str,
    knowable_col: str,
) -> None:
    """Leakage guard: raise if any row's knowable date is strictly after the
    as_of date under which it was used to build a feature/label.

    Intended to be called from tests/leakage/ and, cheaply, from the
    orchestration DAG as a data-quality gate (§22, §27) — a violation here
    means a feature was built with information from the future.
    """
    as_of = pd.to_datetime(df[as_of_col])
    knowable = pd.to_datetime(df[knowable_col])
    violations = df.loc[knowable > as_of]
    if not violations.empty:
        raise AssertionError(
            f"PIT violation: {len(violations)} row(s) use data not yet knowable "
            f"at their as_of date. First offending row:\n{violations.iloc[0]}"
        )
