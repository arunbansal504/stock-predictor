from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.common.pit import (
    assert_pit_safe,
    filter_as_of,
    filter_knowable_as_of,
    latest_knowable_as_of,
)


def test_filter_as_of_excludes_future_rows():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"])})
    out = filter_as_of(df, as_of=pd.Timestamp("2024-01-05"))
    assert list(out["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-05"]))


def test_filter_knowable_as_of_requires_column():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    with pytest.raises(KeyError):
        filter_knowable_as_of(df, as_of=pd.Timestamp("2024-01-01"))


def test_filter_knowable_as_of_uses_publication_not_period_date():
    # Q4 result for period ending 2024-03-31, but not announced until 2024-05-20.
    df = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2024-03-31"]),
            "knowable_date": pd.to_datetime(["2024-05-20"]),
        }
    )
    # As-of a date between period end and announcement: must NOT be visible yet.
    out = filter_knowable_as_of(df, as_of=pd.Timestamp("2024-04-15"))
    assert out.empty

    out2 = filter_knowable_as_of(df, as_of=pd.Timestamp("2024-05-20"))
    assert len(out2) == 1


def test_latest_knowable_as_of_picks_most_recent_per_group():
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "knowable_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-03-01"]),
            "value": [1, 2, 3],
        }
    )
    out = latest_knowable_as_of(df, as_of=pd.Timestamp("2024-12-31"), group_col="symbol")
    out = out.set_index("symbol")
    assert out.loc["A", "value"] == 2  # the later-knowable record for A wins
    assert out.loc["B", "value"] == 3


def test_latest_knowable_as_of_respects_as_of_cutoff():
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "knowable_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "value": [1, 2],
        }
    )
    # As-of a date before the second record was knowable, only the first is visible.
    out = latest_knowable_as_of(df, as_of=pd.Timestamp("2024-03-01"), group_col="symbol")
    assert out.set_index("symbol").loc["A", "value"] == 1


def test_assert_pit_safe_raises_on_violation():
    df = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-01-01"]),
            "knowable_date": pd.to_datetime(["2024-01-05"]),  # knowable AFTER as_of -> leakage
        }
    )
    with pytest.raises(AssertionError):
        assert_pit_safe(df, as_of_col="as_of", knowable_col="knowable_date")


def test_assert_pit_safe_passes_when_no_violation():
    df = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-01-05"]),
            "knowable_date": pd.to_datetime(["2024-01-01"]),
        }
    )
    assert_pit_safe(df, as_of_col="as_of", knowable_col="knowable_date")  # should not raise
