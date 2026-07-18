from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.labels.returns import build_labels_for_symbol, compute_forward_return


def _price_df(closes: list[float], symbol: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"symbol": [symbol] * len(closes), "date": dates, "close_adj": closes})


def _bench_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"series": ["NIFTY500"] * len(closes), "date": dates, "close": closes})


def test_compute_forward_return_matches_manual_shift():
    df = _price_df([100, 110, 121, 133.1])
    out = compute_forward_return(df, horizon_days=1, price_col="close_adj")
    # (110/100 - 1), (121/110 - 1), (133.1/121 - 1), NaN (no forward data for last row)
    assert out.iloc[:-1].tolist() == pytest.approx([0.10, 0.10, 0.10])
    assert pd.isna(out.iloc[-1])


def test_build_labels_for_symbol_computes_correct_excess_return():
    stock = _price_df([100, 105, 110, 115, 120, 125])  # +5% roughly per step
    bench = _bench_df([1000, 1010, 1020, 1030, 1040, 1050])  # ~+1% per step
    out = build_labels_for_symbol(stock, bench, horizons={"2d": 2})

    row0 = out.iloc[0]
    expected_stock_fwd = 110 / 100 - 1
    expected_bench_fwd = 1020 / 1000 - 1
    assert row0["forward_return"] == pytest.approx(expected_stock_fwd)
    assert row0["benchmark_forward_return"] == pytest.approx(expected_bench_fwd)
    assert row0["excess_return"] == pytest.approx(expected_stock_fwd - expected_bench_fwd)
    assert row0["outperform"] == (expected_stock_fwd > expected_bench_fwd)


def test_build_labels_label_valid_date_is_date_plus_horizon_trading_days():
    stock = _price_df([100, 101, 102, 103, 104])
    bench = _bench_df([1000, 1001, 1002, 1003, 1004])
    out = build_labels_for_symbol(stock, bench, horizons={"2d": 2})

    dates = stock["date"].tolist()
    assert out.iloc[0]["label_valid_date"] == dates[2]
    assert out.iloc[1]["label_valid_date"] == dates[3]


def test_build_labels_tail_rows_have_na_labels_not_fabricated_values():
    stock = _price_df([100, 101, 102, 103, 104])
    bench = _bench_df([1000, 1001, 1002, 1003, 1004])
    out = build_labels_for_symbol(stock, bench, horizons={"2d": 2})

    tail = out.iloc[-2:]  # last 2 rows can't resolve a 2-day-forward label
    assert tail["forward_return"].isna().all()
    assert tail["excess_return"].isna().all()
    assert tail["outperform"].isna().all()
    assert tail["label_valid_date"].isna().all()


def test_build_labels_multiple_horizons_stack_rows():
    stock = _price_df([100, 101, 102, 103, 104, 105, 106, 107])
    bench = _bench_df([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007])
    out = build_labels_for_symbol(stock, bench, horizons={"2d": 2, "3d": 3})

    assert len(out) == 2 * len(stock)
    assert set(out["horizon"]) == {"2d", "3d"}
