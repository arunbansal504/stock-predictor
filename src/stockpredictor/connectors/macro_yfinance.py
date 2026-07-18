"""Macro/benchmark index connector (§5 sourcing map: "VIX/USDINR/crude via
yfinance... fine for EOD context").

Provides both the benchmark indices the label builder and backtest engine
compare stocks against (NIFTY 50/500, BANKNIFTY, SENSEX) and macro context
features (India VIX, USDINR, crude) — one connector because they share the
same fetch shape (a single time series per ticker, no per-symbol universe).

Tickers verified live against Yahoo Finance before being hardcoded here;
still an unofficial source, so treat symbol availability as best-effort, not
guaranteed (§5 caveats).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

SOURCE_NAME = "yfinance"

MACRO_TICKERS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY500": "^CRSLDX",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "INDIA_VIX": "^INDIAVIX",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
}

MACRO_COLUMNS: list[str] = ["series", "date", "close", "source"]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
def _download_one(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return yf.Ticker(ticker).history(
        start=start, end=end + dt.timedelta(days=1), auto_adjust=False, actions=False
    )


def fetch_macro_series(
    series_names: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Fetch daily closes for each named macro/benchmark series. Unknown
    names or per-series fetch failures are logged and skipped (§3 NFR:
    graceful degradation) rather than aborting the batch."""
    frames: list[pd.DataFrame] = []
    for name in series_names:
        ticker = MACRO_TICKERS.get(name)
        if ticker is None:
            logger.warning("Unknown macro series requested: %s", name)
            continue
        try:
            raw = _download_one(ticker, start, end)
        except Exception:
            logger.exception("Failed to fetch macro series %s (%s)", name, ticker)
            continue

        if raw is None or raw.empty:
            logger.warning("No macro data returned for %s (%s)", name, ticker)
            continue

        frame = raw.reset_index().rename(columns={"Date": "date", "Close": "close"})
        frame["series"] = name
        frame["source"] = SOURCE_NAME
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frames.append(frame[MACRO_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=MACRO_COLUMNS)
    return pd.concat(frames, ignore_index=True)
