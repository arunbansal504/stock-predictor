"""NSE index-constituent connector (§27: expanding the Phase 1 seed universe
from the 40-symbol scaffold toward the real NIFTY 500).

Fetches NSE's own published constituent list -- real, current index
membership, not a hand-maintained approximation. NSE's main site is
bot-protected in ways that vary over time (session cookies, UA checks); this
endpoint (the "archives" CSV mirror) has historically been the reliable,
unauthenticated way to get this data, but treat it with the same
"unofficial until proven otherwise" caution as every other free source in
§5 -- wrapped in retry/backoff, and callers (ingestion/universe.py) fall
back to the bundled CSV seed if it's unreachable rather than failing the
whole pipeline.
"""

from __future__ import annotations

import io

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# NSE's CDN/edge rejects requests that don't look like a browser.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

_SOURCE_COLUMNS = {"Company Name", "Industry", "Symbol", "Series", "ISIN Code"}
UNIVERSE_COLUMNS = ["symbol", "exchange", "name", "sector", "isin"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_csv_text(url: str) -> str:
    response = httpx.get(url, headers=_HEADERS, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    return response.text


def fetch_nifty500_constituents() -> pd.DataFrame:
    """Fetch NSE's current NIFTY 500 constituent list, normalized to the
    same schema as config/universe_seed.csv (symbol, exchange, name,
    sector) plus `isin` as bonus reference metadata.

    Raises if the source schema has drifted (§22: contract tests catch this
    class of bug before it silently corrupts the universe) or if duplicate
    symbols survive normalization.
    """
    raw_text = _fetch_csv_text(NIFTY500_URL)
    raw = pd.read_csv(io.StringIO(raw_text))

    missing = _SOURCE_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"NSE NIFTY 500 CSV schema changed -- missing columns: {missing}")

    df = raw.rename(
        columns={
            "Symbol": "symbol",
            "Company Name": "name",
            "Industry": "sector",
            "ISIN Code": "isin",
        }
    )
    df["exchange"] = "NSE"
    df = df[df["Series"] == "EQ"]  # ordinary equity only, in case other series appear
    df = df[UNIVERSE_COLUMNS].reset_index(drop=True)

    if df.empty:
        raise ValueError("NSE NIFTY 500 CSV parsed to zero rows -- treat as a fetch failure")
    # Surface duplicates rather than silently deduping (same philosophy as
    # ingestion/universe.py's load_universe_csv) -- a duplicate symbol in an
    # index membership list means something is wrong with the source, not
    # something to quietly paper over.
    if df["symbol"].duplicated().any():
        dupes = sorted(df.loc[df["symbol"].duplicated(keep=False), "symbol"].unique())
        raise ValueError(f"NSE NIFTY 500 CSV contains duplicate symbols: {dupes}")

    logger.info("Fetched %d NIFTY 500 constituents from NSE", len(df))
    return df
