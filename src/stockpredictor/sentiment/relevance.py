"""Relevance filtering (§9 sentiment pipeline: "dedup -> relevance-filter").

A second, independent guard on top of the RSS query itself
(connectors/news_rss.py already searches by company name, not just the bare
ticker) -- a quoted-name search can still surface wire stories that mention
the company only in passing (index roundups, sector-wide pieces, unrelated
companies with a similar name fragment). Keeping an irrelevant article would
silently corrupt the sentiment aggregate for a symbol on a day it had no
real news of its own.

Deliberately simple (substring/word-boundary matching, not an ML classifier)
-- per the architecture doc's Truth 3, a lightweight filter that's easy to
reason about earns its place before a heavier one is justified.
"""

from __future__ import annotations

import re

import pandas as pd

# Corporate suffixes carry no discriminating power for a relevance match and
# would make the primary check ("Reliance Industries Limited" appearing
# verbatim) fail on headlines that drop them (nearly all do).
_CORPORATE_SUFFIXES = re.compile(
    r"\b(limited|ltd\.?|inc\.?|corporation|corp\.?|company|co\.?|plc)\b", re.IGNORECASE
)


def _company_name_tokens(company_name: str) -> list[str]:
    """The distinctive leading words of a company name, suffixes stripped
    -- e.g. "Reliance Industries Limited" -> ["Reliance", "Industries"].
    A single generic token (e.g. just "India") would false-positive on
    unrelated national news, so short/common leading words are dropped."""
    cleaned = _CORPORATE_SUFFIXES.sub("", company_name).strip()
    tokens = [t for t in re.split(r"\s+", cleaned) if len(t) > 2]
    return tokens[:2] if tokens else [cleaned]


def is_relevant(text: str, symbol: str, company_name: str) -> bool:
    """True if `text` (title + summary, already concatenated by the caller)
    plausibly refers to this company: the bare symbol as a whole word, or
    all of the company name's distinctive leading tokens."""
    if not text:
        return False
    lowered = text.lower()

    if re.search(rf"\b{re.escape(symbol.lower())}\b", lowered):
        return True

    tokens = _company_name_tokens(company_name)
    return all(re.search(rf"\b{re.escape(t.lower())}\b", lowered) for t in tokens)


def filter_relevant(articles: pd.DataFrame, symbol: str, company_name: str) -> pd.DataFrame:
    """Keep only rows of `articles` (must have title/summary columns) that
    pass `is_relevant`. Returns an empty frame with the same columns, not a
    KeyError, when given an empty input."""
    if articles.empty:
        return articles

    combined_text = articles["title"].fillna("") + " " + articles["summary"].fillna("")
    mask = combined_text.apply(lambda t: is_relevant(t, symbol, company_name))
    return articles.loc[mask].reset_index(drop=True)
