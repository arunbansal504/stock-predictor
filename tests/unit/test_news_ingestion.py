from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion import news as news_ingestion
from stockpredictor.sentiment.classifier import MODEL_NAME


def _fake_raw_news(symbol: str, n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "published_date": pd.Timestamp("2026-07-15").date(),
                "title": f"{symbol} reports strong quarterly growth",
                "summary": "Details of the results.",
                "url": f"https://example.com/{symbol}-{i}",
                "source": "Moneycontrol",
            }
            for i in range(n)
        ]
    )


def _fake_scored(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    out = df.copy()
    out["sentiment_score"] = 0.5
    out["sentiment_label"] = "positive"
    out["model_version"] = MODEL_NAME
    return out


def _seed_silver_news(tmp_lake, symbol: str, url: str, model_version) -> None:
    """Pre-populate a symbol's silver news file as if a prior run already
    scored this article -- model_version=None simulates a row written
    before that field existed (no such column at all)."""
    row = {
        "symbol": symbol,
        "published_date": pd.Timestamp("2026-07-10").date(),
        "title": "Old headline",
        "summary": "Old summary",
        "url": url,
        "source": "Moneycontrol",
        "sentiment_score": 0.3,
        "sentiment_label": "positive",
    }
    if model_version is not None:
        row["model_version"] = model_version
    tmp_lake.write(pd.DataFrame([row]), DataLayer.SILVER, "news", symbol, key_cols=news_ingestion.KEY_COLS)


def test_ingest_symbol_news_writes_bronze_and_silver(tmp_lake, monkeypatch):
    raw = _fake_raw_news("RELIANCE")
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda symbol, name: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, symbol, name: df)
    monkeypatch.setattr(news_ingestion, "score_articles", _fake_scored)

    rows = news_ingestion.ingest_symbol_news(tmp_lake, "RELIANCE", "Reliance Industries Limited")
    assert rows == 2

    bronze = tmp_lake.read(DataLayer.BRONZE, "news", "RELIANCE")
    assert len(bronze) == 2  # unfiltered raw fetch, including anything relevance would later drop

    silver = tmp_lake.read(DataLayer.SILVER, "news", "RELIANCE")
    assert len(silver) == 2
    assert (silver["sentiment_score"] == 0.5).all()


def test_ingest_symbol_news_empty_fetch_returns_zero(tmp_lake, monkeypatch):
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda symbol, name: pd.DataFrame())
    rows = news_ingestion.ingest_symbol_news(tmp_lake, "NODATA", "No Data Corp")
    assert rows == 0


def test_ingest_symbol_news_nothing_relevant_returns_zero(tmp_lake, monkeypatch):
    raw = _fake_raw_news("RELIANCE")
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda symbol, name: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, symbol, name: df.iloc[0:0])

    rows = news_ingestion.ingest_symbol_news(tmp_lake, "RELIANCE", "Reliance Industries Limited")
    assert rows == 0
    # bronze still recorded the raw fetch even though nothing passed relevance
    bronze = tmp_lake.read(DataLayer.BRONZE, "news", "RELIANCE")
    assert len(bronze) == 2


def test_ingest_symbol_news_refetching_same_url_does_not_duplicate_or_overwrite(tmp_lake, monkeypatch):
    """A second run that re-fetches the same URL (Google News RSS has no
    "since last run" filter -- see ingestion/news.py's docstring) must not
    duplicate the row. It also should NOT pick up a changed title on the
    refetch: once a URL is scored under the current model, it's skipped
    entirely on future runs (see _urls_already_scored) rather than
    re-processed, so the original stored row is left untouched -- this is
    the intended, documented behavior of the skip-rescoring optimization,
    not upsert-on-every-refetch."""
    raw = _fake_raw_news("RELIANCE", n=1)
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda symbol, name: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, symbol, name: df)
    monkeypatch.setattr(news_ingestion, "score_articles", _fake_scored)
    news_ingestion.ingest_symbol_news(tmp_lake, "RELIANCE", "Reliance Industries Limited")

    revised = _fake_raw_news("RELIANCE", n=1)
    revised.loc[0, "title"] = "Updated headline for the same article"
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda symbol, name: revised)
    news_ingestion.ingest_symbol_news(tmp_lake, "RELIANCE", "Reliance Industries Limited")

    silver = tmp_lake.read(DataLayer.SILVER, "news", "RELIANCE")
    assert len(silver) == 1  # same URL -- not duplicated
    assert silver.iloc[0]["title"] == raw.iloc[0]["title"]  # original preserved, refetch was skipped


def test_ingest_symbol_news_skips_rescoring_already_scored_urls(tmp_lake, monkeypatch):
    symbol = "RELIANCE"
    already_scored_url = "https://example.com/old-article"
    new_url = "https://example.com/new-article"
    _seed_silver_news(tmp_lake, symbol, already_scored_url, MODEL_NAME)

    raw = pd.DataFrame(
        [
            {
                "symbol": symbol, "published_date": pd.Timestamp("2026-07-15").date(),
                "title": "Old headline (refetched)", "summary": "...", "url": already_scored_url,
                "source": "Moneycontrol",
            },
            {
                "symbol": symbol, "published_date": pd.Timestamp("2026-07-16").date(),
                "title": "Brand new headline", "summary": "...", "url": new_url, "source": "Moneycontrol",
            },
        ]
    )
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda s, n: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, s, n: df)

    scored_urls_seen: list[str] = []

    def _tracking_score(df, text_col="title"):
        scored_urls_seen.extend(df["url"].tolist())
        return _fake_scored(df)

    monkeypatch.setattr(news_ingestion, "score_articles", _tracking_score)
    news_ingestion.ingest_symbol_news(tmp_lake, symbol, "Reliance Industries Limited")

    assert already_scored_url not in scored_urls_seen  # skipped -- already scored under current model
    assert new_url in scored_urls_seen  # genuinely new -- scored


def test_ingest_symbol_news_rescores_stale_model_version(tmp_lake, monkeypatch):
    """A different model_version (simulating a future FinBERT upgrade) must
    force a re-score, not silently keep the old model's score."""
    symbol = "RELIANCE"
    url = "https://example.com/article"
    _seed_silver_news(tmp_lake, symbol, url, "ProsusAI/finbert-old-version")

    raw = pd.DataFrame(
        [{
            "symbol": symbol, "published_date": pd.Timestamp("2026-07-15").date(),
            "title": "Headline", "summary": "...", "url": url, "source": "Moneycontrol",
        }]
    )
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda s, n: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, s, n: df)

    scored_urls_seen: list[str] = []

    def _tracking_score(df, text_col="title"):
        scored_urls_seen.extend(df["url"].tolist())
        return _fake_scored(df)

    monkeypatch.setattr(news_ingestion, "score_articles", _tracking_score)
    news_ingestion.ingest_symbol_news(tmp_lake, symbol, "Reliance Industries Limited")

    assert url in scored_urls_seen


def test_ingest_symbol_news_rescores_rows_predating_model_version_field(tmp_lake, monkeypatch):
    """A silver row written before model_version existed (no such column at
    all) is conservatively treated as not-yet-scored, not as current."""
    symbol = "RELIANCE"
    url = "https://example.com/article"
    _seed_silver_news(tmp_lake, symbol, url, model_version=None)

    raw = pd.DataFrame(
        [{
            "symbol": symbol, "published_date": pd.Timestamp("2026-07-15").date(),
            "title": "Headline", "summary": "...", "url": url, "source": "Moneycontrol",
        }]
    )
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda s, n: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, s, n: df)

    scored_urls_seen: list[str] = []

    def _tracking_score(df, text_col="title"):
        scored_urls_seen.extend(df["url"].tolist())
        return _fake_scored(df)

    monkeypatch.setattr(news_ingestion, "score_articles", _tracking_score)
    news_ingestion.ingest_symbol_news(tmp_lake, symbol, "Reliance Industries Limited")

    assert url in scored_urls_seen


def test_ingest_symbol_news_no_finbert_call_when_nothing_new_to_score(tmp_lake, monkeypatch):
    """When every relevant article is already scored under the current
    model version, score_articles's empty-input branch must short-circuit
    without ever loading a real FinBERT model -- verified by deliberately
    NOT monkeypatching score_articles here (its empty branch never imports
    transformers, so this stays fast and needs no real model)."""
    symbol = "RELIANCE"
    url = "https://example.com/article"
    _seed_silver_news(tmp_lake, symbol, url, MODEL_NAME)

    raw = pd.DataFrame(
        [{
            "symbol": symbol, "published_date": pd.Timestamp("2026-07-15").date(),
            "title": "Headline (refetched, unchanged)", "summary": "...", "url": url, "source": "Moneycontrol",
        }]
    )
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", lambda s, n: raw)
    monkeypatch.setattr(news_ingestion, "filter_relevant", lambda df, s, n: df)

    rows = news_ingestion.ingest_symbol_news(tmp_lake, symbol, "Reliance Industries Limited")
    assert rows == 1  # the existing row, untouched
