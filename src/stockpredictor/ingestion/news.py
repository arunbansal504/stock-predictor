"""Bronze -> Silver ingestion for news + sentiment (§5, §9).

Bronze = the raw RSS fetch, as-is (audit trail; includes articles later
judged irrelevant, so a relevance-filter bug can be diagnosed against what
was actually fetched). Silver = relevance-filtered and FinBERT-scored --
what features/sentiment.py and the UI actually read.

`knowable_date` is not a separate stamped column here the way it is for
fundamentals: a news article's `published_date` *is* its knowable date (you
cannot know about an article before it was published) -- same event-date-
equals-knowable-date convention as daily prices (see ingestion/prices.py,
common/pit.py).
"""

from __future__ import annotations

from stockpredictor.common.logging import get_logger
from stockpredictor.common.types import DataLayer
from stockpredictor.connectors.news_rss import fetch_news_for_symbol
from stockpredictor.sentiment.classifier import score_articles
from stockpredictor.sentiment.relevance import filter_relevant
from stockpredictor.storage.lake import Lake

logger = get_logger(__name__)

DOMAIN = "news"
KEY_COLS = ["symbol", "url"]


def ingest_symbol_news(lake: Lake, symbol: str, company_name: str) -> int:
    """Fetch, bronze-write, relevance-filter, sentiment-score, and
    silver-write news for one symbol. Returns the resulting silver row
    count (0 if nothing was fetched or nothing survived relevance
    filtering) -- same per-symbol-gap convention as ingestion/prices.py and
    ingestion/fundamentals.py, so one symbol's empty result doesn't fail the
    whole run."""
    raw = fetch_news_for_symbol(symbol, company_name)
    if raw.empty:
        return 0
    lake.write(raw, DataLayer.BRONZE, DOMAIN, symbol, key_cols=KEY_COLS)

    relevant = filter_relevant(raw, symbol, company_name)
    if relevant.empty:
        logger.info("No relevant news survived filtering for %s (%d fetched)", symbol, len(raw))
        return 0

    scored = score_articles(relevant)
    rows = lake.write(scored, DataLayer.SILVER, DOMAIN, symbol, key_cols=KEY_COLS)
    logger.info("Ingested %d silver news rows for %s", rows, symbol)
    return rows
