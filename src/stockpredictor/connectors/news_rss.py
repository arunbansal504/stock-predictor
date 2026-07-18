"""News connector via Google News RSS search (§5 "News" row, §9 sentiment
pipeline's "collect" stage).

Deliberately no historical backfill capability, unlike prices_yfinance.py or
fundamentals_yfinance.py: Google News RSS search returns only its current
most-recent result set (in practice, roughly the last few weeks) for any
query -- there is no date-ranged pagination into the past. This is a real,
structural limitation of the free source, not an implementation gap: it
means news/sentiment data can only be ingested "as it happens," accumulating
one day at a time from whenever ingestion starts, exactly like a live
newspaper archive you started clipping today. See ingestion/news.py and
features/sentiment.py for how this constrains what the resulting features
can honestly be used for.

Query targets the company name (not just the bare ticker) because searching
a 3-6 letter NSE symbol alone returns mostly noise -- "ITC", "SBIN" etc. are
not distinctive search terms. sentiment/relevance.py provides a second,
independent relevance guard downstream, since even a name-based query can
surface loosely-related wire stories.
"""

from __future__ import annotations

import datetime as dt
import io
from urllib.parse import quote_plus

import feedparser
import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

NEWS_RSS_URL = "https://news.google.com/rss/search"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}

NEWS_COLUMNS: list[str] = ["symbol", "published_date", "title", "summary", "url", "source"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_feed_bytes(query: str) -> bytes:
    # Pre-encode the query with quote_plus rather than passing it via
    # httpx's `params=` -- Google News RSS's query syntax uses characters
    # (quotes, spaces) that need exactly one round of escaping, and letting
    # httpx re-encode an already-built query string risks double-escaping.
    url = f"{NEWS_RSS_URL}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    response = httpx.get(url, headers=_HEADERS, timeout=20.0)
    response.raise_for_status()
    return response.content


def _build_query(symbol: str, company_name: str) -> str:
    return f'"{company_name}" stock NSE'


def fetch_news_for_symbol(symbol: str, company_name: str) -> pd.DataFrame:
    """Fetch the current Google News RSS result set for one company.
    Returns one row per article (deduplicated by URL within this fetch);
    caller (ingestion/news.py) handles cross-run dedup via the lake's
    key-column upsert. Feed-parse failures degrade to an empty frame rather
    than raising -- this is a best-effort, free, unofficial source (§5), not
    something a whole nightly run should abort over."""
    query = _build_query(symbol, company_name)
    try:
        raw_bytes = _fetch_feed_bytes(query)
    except Exception:
        logger.exception("Failed to fetch news RSS for %s (%s)", symbol, company_name)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    parsed = feedparser.parse(io.BytesIO(raw_bytes))
    if parsed.bozo and not parsed.entries:
        logger.warning("News RSS feed for %s did not parse cleanly and had no entries", symbol)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []
    for entry in parsed.entries:
        published = getattr(entry, "published_parsed", None)
        if published is None:
            continue  # an article we can't PIT-stamp is not usable, not a fallback-to-today guess
        published_date = dt.date(*published[:3])
        source = getattr(getattr(entry, "source", None), "title", None) or ""
        rows.append(
            {
                "symbol": symbol,
                "published_date": published_date,
                "title": getattr(entry, "title", ""),
                "summary": getattr(entry, "summary", ""),
                "url": getattr(entry, "link", ""),
                "source": source,
            }
        )

    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = pd.DataFrame(rows)[NEWS_COLUMNS]
    df = df[df["url"] != ""].drop_duplicates(subset="url")
    logger.info("Fetched %d news article(s) for %s", len(df), symbol)
    return df
