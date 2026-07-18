"""Contract test for the news RSS connector (§5, §22). No network -- pins
normalization against a realistic Google News RSS feed shape."""

from __future__ import annotations

from stockpredictor.connectors import news_rss

_SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>"Reliance Industries" stock NSE - Google News</title>
<item>
  <title>Reliance Industries posts strong Q4 results</title>
  <link>https://example.com/article-1</link>
  <pubDate>Mon, 17 Jul 2026 09:30:00 GMT</pubDate>
  <source url="https://moneycontrol.com">Moneycontrol</source>
  <description>Reliance Industries reported a 15% jump in quarterly profit.</description>
</item>
<item>
  <title>Reliance shares dip amid broader market weakness</title>
  <link>https://example.com/article-2</link>
  <pubDate>Sun, 16 Jul 2026 14:00:00 GMT</pubDate>
  <source url="https://economictimes.com">Economic Times</source>
  <description>Reliance stock fell 2% in a weak session.</description>
</item>
<item>
  <title>An article with no pubDate</title>
  <link>https://example.com/article-3</link>
  <description>Should be dropped -- can't PIT-stamp it.</description>
</item>
</channel>
</rss>"""


def test_fetch_news_normalizes_schema_and_drops_undated_entries(monkeypatch):
    monkeypatch.setattr(news_rss, "_fetch_feed_bytes", lambda query: _SAMPLE_FEED)

    df = news_rss.fetch_news_for_symbol("RELIANCE", "Reliance Industries Limited")
    assert list(df.columns) == news_rss.NEWS_COLUMNS
    assert len(df) == 2  # the undated third entry is dropped
    assert set(df["url"]) == {"https://example.com/article-1", "https://example.com/article-2"}


def test_fetch_news_published_date_parsed_correctly(monkeypatch):
    monkeypatch.setattr(news_rss, "_fetch_feed_bytes", lambda query: _SAMPLE_FEED)

    df = news_rss.fetch_news_for_symbol("RELIANCE", "Reliance Industries Limited").set_index("url")
    row = df.loc["https://example.com/article-1"]
    assert str(row["published_date"]) == "2026-07-17"
    assert row["source"] == "Moneycontrol"


_FEED_WITH_DUPLICATE_URL = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<item>
  <title>Reliance Industries posts strong Q4 results</title>
  <link>https://example.com/article-1</link>
  <pubDate>Mon, 17 Jul 2026 09:30:00 GMT</pubDate>
  <source url="https://moneycontrol.com">Moneycontrol</source>
  <description>Reliance Industries reported a 15% jump in quarterly profit.</description>
</item>
<item>
  <title>Reliance Industries posts strong Q4 results (syndicated)</title>
  <link>https://example.com/article-1</link>
  <pubDate>Mon, 17 Jul 2026 10:00:00 GMT</pubDate>
  <source url="https://mirror.example.com">Mirror Site</source>
  <description>Same article, re-syndicated by an aggregator.</description>
</item>
</channel>
</rss>"""


def test_fetch_news_dedups_by_url(monkeypatch):
    monkeypatch.setattr(news_rss, "_fetch_feed_bytes", lambda query: _FEED_WITH_DUPLICATE_URL)
    df = news_rss.fetch_news_for_symbol("RELIANCE", "Reliance Industries Limited")
    assert df["url"].is_unique
    assert len(df) == 1


def test_fetch_news_empty_feed_returns_empty_frame(monkeypatch):
    empty_feed = b"""<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>"""
    monkeypatch.setattr(news_rss, "_fetch_feed_bytes", lambda query: empty_feed)
    df = news_rss.fetch_news_for_symbol("NODATA", "No Data Corp")
    assert df.empty
    assert list(df.columns) == news_rss.NEWS_COLUMNS


def test_fetch_news_fetch_exception_returns_empty_frame(monkeypatch):
    def fail(query):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(news_rss, "_fetch_feed_bytes", fail)
    df = news_rss.fetch_news_for_symbol("RELIANCE", "Reliance Industries Limited")
    assert df.empty
