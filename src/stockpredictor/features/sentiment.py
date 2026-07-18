"""Sentiment/News feature block (§7, §9).

**Not currently wired into `ALL_FEATURE_COLUMNS`** (features/registry.py) --
see that module's docstring note and README for why. Short version:
connectors/news_rss.py has no historical backfill (Google News RSS only
returns current results), so real news history only starts accumulating
from whenever nightly ingestion first ran. Feeding a model columns that are
entirely NaN across nearly all of a multi-year walk-forward backtest window
already backfired once for fundamentals' growth ratios (see
features/fundamental.py's docstring: IC dropped from 0.037 to 0.015) --
same failure mode would apply here, worse, since news history starts at
zero rather than partial. This module is built and wired into nightly
ingestion now so real data accumulates going forward; revisit folding
SENTIMENT_FEATURE_COLUMNS into ALL_FEATURE_COLUMNS once enough calendar
time has passed to actually evaluate it out-of-sample, not before.

In the meantime, `latest_sentiment_snapshot` powers a live "current
sentiment" display (Streamlit's Stock Detail tab) -- real user-facing value
today, independent of whether/when this becomes a trained-model feature.

PIT correctness: a news article's `published_date` is its own knowable date
(see ingestion/news.py). Aggregating a *trailing* window as of each trading
date (backward-looking `.rolling()`) is therefore leakage-free by
construction -- no separate knowable_date filter step is needed the way
features/fundamental.py needs one for annual statements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTIMENT_FEATURE_COLUMNS: list[str] = [
    "news_sentiment_5d",
    "news_sentiment_20d",
    "news_sentiment_momentum",
    "news_volume_5d",
    "news_sentiment_dispersion_5d",
]

_SHORT_WINDOW = 5
_LONG_WINDOW = 20


def _daily_aggregates(news_df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date with any news, summed/counted -- the
    volume-weighted building block for rolling means (a day with 10
    articles should count 10x as much as a day with 1 in a trailing-window
    average, not be treated as one equally-weighted observation)."""
    daily = news_df.groupby("published_date")["sentiment_score"].agg(["sum", "count"]).reset_index()
    return daily.rename(columns={"sum": "daily_sum", "count": "daily_count"})


def build_sentiment_features_for_symbol(prices_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """For one symbol: a PIT-correct, trading-calendar-aligned sentiment
    feature frame. Trading dates with zero news in a window get
    `news_volume_5d=0` (a real fact) but NaN sentiment-mean columns
    (undefined, not fabricated as neutral 0) -- the same "honest gap, not a
    fabricated placeholder" convention as features/fundamental.py."""
    trading_dates = prices_df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    trading_dates["date"] = pd.to_datetime(trading_dates["date"]).dt.normalize()

    if news_df.empty:
        out = trading_dates.copy()
        for col in SENTIMENT_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    news = news_df.copy()
    news["published_date"] = pd.to_datetime(news["published_date"]).dt.normalize()
    daily = _daily_aggregates(news)

    calendar_start = min(daily["published_date"].min(), trading_dates["date"].min())
    calendar_end = trading_dates["date"].max()
    calendar = pd.DataFrame({"date": pd.date_range(calendar_start, calendar_end, freq="D")})
    calendar = calendar.merge(daily, left_on="date", right_on="published_date", how="left").drop(
        columns=["published_date"]
    )
    calendar["daily_count"] = calendar["daily_count"].fillna(0)

    sum_5d = calendar["daily_sum"].rolling(_SHORT_WINDOW, min_periods=1).sum()
    sum_20d = calendar["daily_sum"].rolling(_LONG_WINDOW, min_periods=1).sum()
    count_5d = calendar["daily_count"].rolling(_SHORT_WINDOW, min_periods=1).sum()
    count_20d = calendar["daily_count"].rolling(_LONG_WINDOW, min_periods=1).sum()

    calendar["news_sentiment_5d"] = sum_5d / count_5d.replace(0, np.nan)
    calendar["news_sentiment_20d"] = sum_20d / count_20d.replace(0, np.nan)
    calendar["news_sentiment_momentum"] = calendar["news_sentiment_5d"] - calendar["news_sentiment_20d"]
    calendar["news_volume_5d"] = count_5d
    # Day-to-day volatility of the daily mean sentiment, not of individual
    # article scores -- an approximation (documented, not hidden) that
    # avoids a per-row rolling join; still a legitimate "how noisy/uncertain
    # is the sentiment signal lately" measure.
    calendar["news_sentiment_dispersion_5d"] = (calendar["daily_sum"] / calendar["daily_count"].replace(0, np.nan)).rolling(
        _SHORT_WINDOW, min_periods=1
    ).std()

    merged = trading_dates.merge(calendar[["date"] + SENTIMENT_FEATURE_COLUMNS], on="date", how="left")
    return merged


def latest_sentiment_snapshot(news_df: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 5) -> dict:
    """Non-windowed convenience summary for the UI: mean sentiment, article
    count, and the underlying articles (title/url/source/date/score) for
    the trailing `lookback_days` as of `as_of`. Returns an "empty" summary
    (mean_sentiment=None, articles=[]) rather than raising when there's no
    news -- a stock legitimately having no recent coverage is normal, not
    an error."""
    if news_df.empty:
        return {"mean_sentiment": None, "article_count": 0, "articles": []}

    news = news_df.copy()
    news["published_date"] = pd.to_datetime(news["published_date"])
    as_of = pd.Timestamp(as_of).normalize()
    window_start = as_of - pd.Timedelta(days=lookback_days - 1)
    window = news[(news["published_date"] >= window_start) & (news["published_date"] <= as_of)]

    if window.empty:
        return {"mean_sentiment": None, "article_count": 0, "articles": []}

    window = window.sort_values("published_date", ascending=False)
    return {
        "mean_sentiment": float(window["sentiment_score"].mean()),
        "article_count": len(window),
        "articles": window[["published_date", "title", "url", "source", "sentiment_score", "sentiment_label"]].to_dict(
            "records"
        ),
    }
