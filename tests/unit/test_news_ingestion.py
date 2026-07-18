from __future__ import annotations

import pandas as pd

from stockpredictor.common.types import DataLayer
from stockpredictor.ingestion import news as news_ingestion


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
    return out


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


def test_ingest_symbol_news_upserts_on_url(tmp_lake, monkeypatch):
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
    assert len(silver) == 1  # same URL -- updated in place, not duplicated
    assert silver.iloc[0]["title"] == "Updated headline for the same article"
