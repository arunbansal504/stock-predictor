from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.features.fundamental import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    build_fundamental_features_for_symbol,
    compute_fundamental_ratios,
)


def _fundamentals_df(symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "period_end": pd.Timestamp("2023-03-31"),
                "knowable_date": pd.Timestamp("2023-05-01"),
                "revenue": 1000.0,
                "net_income": 100.0,
                "eps": 10.0,
                "total_equity": 500.0,
                "total_debt": 200.0,
                "total_assets": 1000.0,
                "shares_outstanding": 10.0,
            },
            {
                "symbol": symbol,
                "period_end": pd.Timestamp("2024-03-31"),
                "knowable_date": pd.Timestamp("2024-05-01"),
                "revenue": 1200.0,  # +20% YoY
                "net_income": 150.0,
                "eps": 15.0,  # +50% YoY
                "total_equity": 600.0,
                "total_debt": 180.0,
                "total_assets": 1100.0,
                "shares_outstanding": 10.0,
            },
        ]
    )


def _price_series(symbol: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2023-04-01", "2024-06-01")
    return pd.DataFrame({"symbol": [symbol] * len(dates), "date": dates, "close_adj": 100.0})


def test_compute_fundamental_ratios_matches_manual_calc():
    out = compute_fundamental_ratios(_fundamentals_df()).set_index("period_end")

    row = out.loc[pd.Timestamp("2024-03-31")]
    assert row["roe"] == pytest.approx(150.0 / 600.0)
    assert row["roa"] == pytest.approx(150.0 / 1100.0)
    assert row["debt_to_equity"] == pytest.approx(180.0 / 600.0)
    assert row["net_margin"] == pytest.approx(150.0 / 1200.0)
    assert row["book_value_per_share"] == pytest.approx(600.0 / 10.0)
    assert row["revenue_growth_yoy"] == pytest.approx(0.20)
    assert row["eps_growth_yoy"] == pytest.approx(0.50)


def test_compute_fundamental_ratios_first_year_has_nan_growth():
    out = compute_fundamental_ratios(_fundamentals_df()).set_index("period_end")
    row = out.loc[pd.Timestamp("2023-03-31")]
    assert pd.isna(row["revenue_growth_yoy"])  # no prior year to compare against
    assert pd.isna(row["eps_growth_yoy"])


def test_build_fundamental_features_uses_correct_snapshot_before_and_after_knowable_date():
    prices = _price_series()
    fundamentals = _fundamentals_df()
    out = build_fundamental_features_for_symbol(prices, fundamentals).set_index("date")

    # The day before FY2024 becomes knowable (2024-05-01): still FY2023 figures.
    before = out.loc[pd.Timestamp("2024-04-30")]
    assert before["roe"] == pytest.approx(100.0 / 500.0)

    # On and after the knowable_date: FY2024 figures apply.
    on_date = out.loc[pd.Timestamp("2024-05-01")]
    assert on_date["roe"] == pytest.approx(150.0 / 600.0)


def test_build_fundamental_features_start_of_history_is_nan_not_fabricated():
    prices = _price_series()
    fundamentals = _fundamentals_df()
    out = build_fundamental_features_for_symbol(prices, fundamentals).set_index("date")

    # Before the first fundamental snapshot is knowable (2023-05-01).
    before_first = out.loc[pd.Timestamp("2023-04-03")]
    for col in FUNDAMENTAL_FEATURE_COLUMNS:
        assert pd.isna(before_first[col])


def test_build_fundamental_features_pe_pb_change_daily_with_price():
    prices = _price_series()
    last_date = prices["date"].max()  # last actual trading day in the series
    second_last_date = prices["date"].iloc[-2]
    prices.loc[prices["date"] == last_date, "close_adj"] = 200.0  # price doubles on the last day
    fundamentals = _fundamentals_df()
    out = build_fundamental_features_for_symbol(prices, fundamentals).set_index("date")

    normal_day = out.loc[second_last_date]
    spike_day = out.loc[last_date]
    # Same fundamental snapshot (FY2024, eps=15) underlies both days, but PE
    # must differ because price differs -- PE/PB are price-dependent daily,
    # not frozen at the fundamental snapshot's own date.
    assert spike_day["pe_ratio"] == pytest.approx(2 * normal_day["pe_ratio"])


def test_build_fundamental_features_empty_fundamentals_returns_all_nan():
    prices = _price_series()
    out = build_fundamental_features_for_symbol(prices, pd.DataFrame())
    assert len(out) == len(prices)
    for col in FUNDAMENTAL_FEATURE_COLUMNS:
        assert out[col].isna().all()


def test_build_fundamental_features_is_fast_for_a_large_price_history():
    """Performance regression guard: this must use a vectorized as-of join
    (pd.merge_asof), not a per-row Python loop -- a naive loop over ~1250
    daily rows x 500 symbols would take minutes-to-hours, not milliseconds.
    """
    import time

    dates = pd.bdate_range("2020-01-01", periods=1250)
    prices = pd.DataFrame({"symbol": ["AAA"] * len(dates), "date": dates, "close_adj": 100.0})
    fundamentals = _fundamentals_df()

    t0 = time.time()
    build_fundamental_features_for_symbol(prices, fundamentals)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"took {elapsed:.2f}s -- likely regressed to a non-vectorized join"
