from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer


def test_write_then_read_roundtrip(tmp_lake):
    df = pd.DataFrame(
        {
            "symbol": ["RELIANCE"] * 3,
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    n = tmp_lake.write(df, DataLayer.SILVER, "prices", "RELIANCE", key_cols=["symbol", "date"])
    assert n == 3

    out = tmp_lake.read(DataLayer.SILVER, "prices", "RELIANCE")
    assert len(out) == 3
    assert list(out["close"]) == [100.0, 101.0, 102.0]


def test_write_upserts_and_dedups_on_key(tmp_lake):
    df1 = pd.DataFrame(
        {
            "symbol": ["TCS", "TCS"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100.0, 101.0],
        }
    )
    tmp_lake.write(df1, DataLayer.SILVER, "prices", "TCS", key_cols=["symbol", "date"])

    # Re-ingest an overlapping range: 01-02 revised, 01-03 new.
    df2 = pd.DataFrame(
        {
            "symbol": ["TCS", "TCS"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [999.0, 103.0],
        }
    )
    n = tmp_lake.write(df2, DataLayer.SILVER, "prices", "TCS", key_cols=["symbol", "date"])

    out = tmp_lake.read(DataLayer.SILVER, "prices", "TCS").sort_values("date")
    assert n == 3  # 01-01, 01-02 (revised), 01-03 -- not 4
    assert list(out["close"]) == [100.0, 999.0, 103.0]  # last write wins, sorted by key


def test_read_all_combines_multiple_symbols(tmp_lake):
    a = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2024-01-01"]), "close": [10.0]})
    b = pd.DataFrame({"symbol": ["B"], "date": pd.to_datetime(["2024-01-01"]), "close": [20.0]})
    tmp_lake.write(a, DataLayer.SILVER, "prices", "A", key_cols=["symbol", "date"])
    tmp_lake.write(b, DataLayer.SILVER, "prices", "B", key_cols=["symbol", "date"])

    out = tmp_lake.read_all(DataLayer.SILVER, "prices")
    assert set(out["symbol"]) == {"A", "B"}
    assert len(out) == 2


def test_read_all_empty_domain_returns_empty_frame(tmp_lake):
    out = tmp_lake.read_all(DataLayer.SILVER, "nonexistent_domain")
    assert out.empty


def test_read_missing_symbol_returns_empty_frame(tmp_lake):
    out = tmp_lake.read(DataLayer.BRONZE, "prices", "NOPE")
    assert out.empty


def test_read_all_orders_by_symbol_and_date_by_default(tmp_lake):
    """DuckDB's parallel multi-file glob scan does not guarantee row order
    on its own -- read_all's default `order_by` is what makes downstream
    order-sensitive code (e.g. models/ensemble.py's chronological base/meta
    split) reproducible run to run. Writes symbol B before A, each with
    dates out of order, so a naive scan wouldn't happen to already be sorted."""
    b = pd.DataFrame(
        {"symbol": ["B", "B"], "date": pd.to_datetime(["2024-01-02", "2024-01-01"]), "close": [21.0, 20.0]}
    )
    a = pd.DataFrame(
        {"symbol": ["A", "A"], "date": pd.to_datetime(["2024-01-02", "2024-01-01"]), "close": [11.0, 10.0]}
    )
    tmp_lake.write(b, DataLayer.SILVER, "prices", "B", key_cols=["symbol", "date"])
    tmp_lake.write(a, DataLayer.SILVER, "prices", "A", key_cols=["symbol", "date"])

    out = tmp_lake.read_all(DataLayer.SILVER, "prices")
    assert list(out["symbol"]) == ["A", "A", "B", "B"]
    assert list(out["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]))


def test_read_all_skips_missing_order_columns_gracefully(tmp_lake):
    """A domain without "symbol"/"date" columns (e.g. backtest results) must
    not error just because the default order_by references columns that
    don't exist in that domain's schema."""
    df = pd.DataFrame({"id": [2, 1], "value": [20.0, 10.0]})
    tmp_lake.write(df, DataLayer.GOLD, "misc", "only", key_cols=["id"])

    out = tmp_lake.read_all(DataLayer.GOLD, "misc")
    assert len(out) == 2
