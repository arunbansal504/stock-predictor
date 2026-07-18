"""Leakage tests (§22: "shuffle/lag tests to prove no look-ahead; label-timing
assertions"). These are meant to grow alongside features/ and labels/ as they
land in later steps; for now they pin the PIT contract that all future
feature/label code must honor, using common/pit.py directly.

Framing note from the architecture doc (§30): a too-good backtest result is
treated as a leakage bug, not a win. These tests are the first line of
defense against that failure mode -- they must stay green as a hard CI gate
once the orchestration DAG exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.common.pit import assert_pit_safe, latest_knowable_as_of
from stockpredictor.ingestion.prices import bronze_to_silver
from stockpredictor.labels.returns import build_labels_for_symbol


def test_price_silver_never_produces_future_knowable_dates():
    """For prices, knowable_date must equal date exactly (same-day
    knowability) -- if a bug ever set it earlier or later, downstream feature
    joins could either leak future information or needlessly discard current
    data."""
    bronze = pd.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "adj_close": [1.0, 2.0, 3.0],
            "volume": pd.array([100, 100, 100], dtype="int64"),
            "source": ["yfinance"] * 3,
        }
    )
    silver = bronze_to_silver(bronze)
    # as_of == date for every row -> assert_pit_safe must pass.
    assert_pit_safe(silver, as_of_col="date", knowable_col="knowable_date")


def test_simulated_fundamental_join_leaks_future_quarter_without_pit_guard():
    """Demonstrates the exact bug class §25/§26 warn about: naively joining a
    fundamentals table on report *period* instead of *announcement* date
    silently leaks the future. `latest_knowable_as_of` is the correct
    primitive; a raw filter on period end is not.
    """
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "period_end": pd.to_datetime(["2023-12-31", "2024-03-31"]),
            "knowable_date": pd.to_datetime(["2024-02-10", "2024-05-20"]),
            "eps": [10.0, 12.0],
        }
    )
    as_of = pd.Timestamp("2024-04-01")  # after Q4 period end, before its announcement

    # WRONG approach (what a naive pipeline might do): filter on period_end <= as_of.
    naive_leaked = fundamentals.loc[fundamentals["period_end"] <= as_of]
    assert len(naive_leaked) == 2  # includes the not-yet-announced Q4 EPS -- a leak

    # CORRECT approach: PIT-safe as-of join.
    correct = latest_knowable_as_of(fundamentals, as_of=as_of, group_col="symbol")
    assert len(correct) == 1
    assert correct.iloc[0]["eps"] == 10.0  # only the Q3 figure was actually knowable


def test_assert_pit_safe_catches_a_deliberately_injected_leak():
    leaked = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-04-01"]),
            "knowable_date": pd.to_datetime(["2024-05-20"]),  # knowable AFTER as_of
        }
    )
    with pytest.raises(AssertionError, match="PIT violation"):
        assert_pit_safe(leaked, as_of_col="as_of", knowable_col="knowable_date")


def test_walk_forward_training_cutoff_must_respect_label_embargo():
    """Demonstrates the "embargo" bug class (§25, Lopez de Prado): a 10-day
    horizon label decided on day T is not *resolved* until T+10. A
    walk-forward split that includes decision rows up to the training cutoff
    without checking `label_valid_date` trains on partially-future
    information for every row within the last `horizon` days of the window.
    """
    dates = pd.bdate_range("2024-01-01", periods=20)
    stock = pd.DataFrame({"symbol": ["AAA"] * 20, "date": dates, "close_adj": range(100, 120)})
    bench = pd.DataFrame({"series": ["NIFTY500"] * 20, "date": dates, "close": range(1000, 1020)})

    horizon = 10
    labels = build_labels_for_symbol(stock, bench, horizons={"10d": horizon})

    train_cutoff = dates[14]  # a walk-forward fold boundary

    # WRONG: naively taking every decision row up to the cutoff includes rows
    # whose 10-day-forward label isn't resolved until after the cutoff.
    naive_train_set = labels[labels["date"] <= train_cutoff]
    still_unresolved = naive_train_set[naive_train_set["label_valid_date"] > train_cutoff]
    assert len(still_unresolved) > 0  # the leak this test exists to catch

    # CORRECT: embargo -- only train on labels whose resolution date has
    # already passed as of the cutoff. This also naturally drops rows whose
    # label never resolved within available history (label_valid_date is
    # NaT), which must be excluded from training regardless of embargo.
    embargoed_train_set = naive_train_set[naive_train_set["label_valid_date"] <= train_cutoff]
    assert len(embargoed_train_set) < len(naive_train_set)
    assert (embargoed_train_set["label_valid_date"] <= train_cutoff).all()
    assert embargoed_train_set["label_valid_date"].notna().all()
