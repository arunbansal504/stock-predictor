"""yfinance-based EOD price connector for NSE/BSE-listed symbols.

Free/unofficial data source (§5 sourcing map — "unofficial; can rate-limit or
break"), so this module leans hard on retry/backoff and strict schema
validation: upstream breakage must surface immediately as a connector error,
never propagate silently into features/predictions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from stockpredictor.common.logging import get_logger
from stockpredictor.connectors.base import PRICE_BRONZE_COLUMNS, validate_schema

logger = get_logger(__name__)

SOURCE_NAME = "yfinance"
_EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def to_provider_ticker(symbol: str, exchange: str = "NSE") -> str:
    suffix = _EXCHANGE_SUFFIX.get(exchange, ".NS")
    return f"{symbol}{suffix}"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
def _download_one(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    # auto_adjust=False so we get both raw OHLC and Adj Close explicitly,
    # rather than yfinance silently overwriting Close with an adjusted value —
    # ingestion/prices.py decides how adjustment is surfaced downstream.
    return yf.Ticker(ticker).history(
        start=start, end=end + dt.timedelta(days=1), auto_adjust=False, actions=False
    )


def fetch_prices(
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Fetch daily OHLCV for each symbol, normalized to PRICE_BRONZE_COLUMNS.

    Per-symbol failures are logged and skipped (architecture §3 NFR: "a
    failed source degrades gracefully") rather than aborting the whole batch
    — one delisted or mistyped ticker shouldn't kill a 500-symbol nightly run.
    """
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        ticker = to_provider_ticker(symbol, exchange)
        try:
            raw = _download_one(ticker, start, end)
        except Exception:
            logger.exception("Failed to fetch prices for %s (%s)", symbol, ticker)
            continue

        if raw is None or raw.empty:
            logger.warning("No price data returned for %s (%s)", symbol, ticker)
            continue

        frame = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frame["symbol"] = symbol
        frame["source"] = SOURCE_NAME
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame["volume"] = frame["volume"].fillna(0).astype("int64")
        frames.append(frame[list(PRICE_BRONZE_COLUMNS.keys())])

    if not frames:
        return pd.DataFrame(columns=list(PRICE_BRONZE_COLUMNS.keys()))

    result = pd.concat(frames, ignore_index=True)
    validate_schema(result, PRICE_BRONZE_COLUMNS, context="prices_yfinance")
    return result
