from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpredictor.features.sentiment import (
    SENTIMENT_FEATURE_COLUMNS,
    build_sentiment_features_for_symbol,
    latest_sentiment_snapshot,
)


def _prices(symbol: str, start: str, n: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"symbol": symbol, "date": dates, "close_adj": 100 + np.arange(n)})


def _news(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_build_sentiment_features_no_news_returns_all_nan_columns():
    prices = _prices("AAA", "2026-06-01", 30)
    out = build_sentiment_features_for_symbol(prices, pd.DataFrame())
    assert list(out["date"]) == list(pd.to_datetime(prices["date"]))
    for col in SENTIMENT_FEATURE_COLUMNS:
        assert out[col].isna().all()


def test_build_sentiment_features_news_volume_is_zero_not_nan_when_no_articles():
    prices = _prices("AAA", "2026-06-01", 30)
    news = _news([{"published_date": pd.Timestamp("2026-06-05").date(), "sentiment_score": 0.5}])
    out = build_sentiment_features_for_symbol(prices, news)
    early_row = out.iloc[0]
    assert early_row["date"] < pd.Timestamp("2026-06-05")
    assert early_row["news_volume_5d"] == 0  # a real fact (no news yet), not undefined
    assert pd.isna(early_row["news_sentiment_5d"])  # undefined, not fabricated as neutral


def test_build_sentiment_features_is_pit_correct_no_future_leakage():
    """A trading day's rolling sentiment must never include an article
    published after that day -- the core leakage guard for this module."""
    prices = _prices("AAA", "2026-06-01", 10)
    trading_dates = pd.to_datetime(prices["date"])
    early_date = trading_dates.iloc[2]
    future_date = trading_dates.iloc[-1]  # published on the very last trading day

    news = _news(
        [
            {"published_date": future_date.date(), "sentiment_score": 1.0},
        ]
    )
    out = build_sentiment_features_for_symbol(prices, news)
    early_row = out[out["date"] == early_date].iloc[0]
    # The only article in the whole dataset was published after `early_date`
    # -- so as-of early_date there must be zero observed news.
    assert early_row["news_volume_5d"] == 0
    assert pd.isna(early_row["news_sentiment_5d"])


def test_build_sentiment_features_5d_mean_is_volume_weighted():
    prices = _prices("AAA", "2026-06-01", 10)
    as_of = pd.to_datetime(prices["date"]).iloc[4]
    news = _news(
        [
            {"published_date": as_of.date(), "sentiment_score": 1.0},
            {"published_date": as_of.date(), "sentiment_score": 1.0},
            {"published_date": as_of.date(), "sentiment_score": -1.0},
        ]
    )
    out = build_sentiment_features_for_symbol(prices, news)
    row = out[out["date"] == as_of].iloc[0]
    # 2 articles at +1.0, 1 article at -1.0 -> volume-weighted mean = 1/3, not
    # a naive mean-of-one-day (which would coincidentally also be 1/3 here,
    # so this test also checks volume via news_volume_5d directly).
    assert row["news_sentiment_5d"] == pytest.approx(1 / 3)
    assert row["news_volume_5d"] == 3


def test_build_sentiment_features_momentum_is_5d_minus_20d():
    prices = _prices("AAA", "2026-05-01", 40)
    dates = pd.to_datetime(prices["date"])
    recent = dates.iloc[-1]
    # The rolling windows are calendar-day based (one row per calendar day,
    # not per trading day -- see build_sentiment_features_for_symbol), so
    # the "old" article must be offset in calendar days, not trading-day
    # index positions, to land reliably inside the 20d window but outside
    # the 5d one.
    old = recent - pd.Timedelta(days=10)
    news = _news(
        [
            {"published_date": recent.date(), "sentiment_score": 1.0},
            {"published_date": old.date(), "sentiment_score": -1.0},
        ]
    )
    out = build_sentiment_features_for_symbol(prices, news)
    row = out[out["date"] == recent].iloc[0]
    assert row["news_sentiment_momentum"] == pytest.approx(row["news_sentiment_5d"] - row["news_sentiment_20d"])
    assert row["news_sentiment_momentum"] > 0  # recent positive news, older negative -> momentum up


def test_latest_sentiment_snapshot_empty_news_returns_zero_articles():
    snap = latest_sentiment_snapshot(pd.DataFrame(), pd.Timestamp("2026-07-18"))
    assert snap["article_count"] == 0
    assert snap["mean_sentiment"] is None
    assert snap["articles"] == []


def test_latest_sentiment_snapshot_respects_lookback_window():
    news = _news(
        [
            {
                "published_date": pd.Timestamp("2026-07-18").date(),
                "title": "In window",
                "url": "https://a",
                "source": "X",
                "sentiment_score": 0.8,
                "sentiment_label": "positive",
            },
            {
                "published_date": pd.Timestamp("2026-06-01").date(),
                "title": "Too old",
                "url": "https://b",
                "source": "X",
                "sentiment_score": -0.5,
                "sentiment_label": "negative",
            },
        ]
    )
    snap = latest_sentiment_snapshot(news, pd.Timestamp("2026-07-18"), lookback_days=5)
    assert snap["article_count"] == 1
    assert snap["mean_sentiment"] == pytest.approx(0.8)
    assert snap["articles"][0]["title"] == "In window"
