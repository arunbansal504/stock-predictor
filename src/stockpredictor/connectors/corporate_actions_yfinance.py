"""Corporate actions (splits/dividends) connector via yfinance.

Known limitation (documented rather than hidden, per the architecture doc's
honesty principle): yfinance exposes only the *ex-date* for splits and
dividends, not the true public-announcement date. We stamp `knowable_date =
ex_date`, which is conservative in the safe direction for PIT correctness —
announcements almost always precede the ex-date, so treating the action as
"knowable" only from ex_date onward can only make features *later* to react,
never leak future information. Bonus issues are not reliably distinguished
from splits by this source; both surface as `action_type="split"` with a
ratio. Buybacks/insider trades are out of scope for this connector (§5:
deferred to a dedicated announcements source in a later phase).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

CORPORATE_ACTION_COLUMNS: list[str] = [
    "symbol",
    "action_type",
    "ex_date",
    "knowable_date",
    "ratio",
    "value",
]


def _validate_actions(t: yf.Ticker, ticker: str) -> yf.Ticker:
    """Observed in practice (not just theoretical): under rapid sequential
    calls across a 40-symbol universe, yfinance sometimes returns None for
    `.splits`/`.dividends` instead of an empty Series -- not a raised
    exception, so @retry wouldn't otherwise see it as a failure worth
    retrying. Treat None as a transient failure rather than silently
    accepting it as "this ticker has no history". Split out from
    `_fetch_ticker_actions` so this logic is unit-testable without paying
    for tenacity's real backoff delay.
    """
    if t.splits is None or t.dividends is None:
        raise RuntimeError(f"yfinance returned None splits/dividends for {ticker} (likely transient)")
    return t


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_ticker_actions(ticker: str) -> yf.Ticker:
    # Touch both properties now so retry covers the actual network call --
    # yfinance fetches lazily on first access.
    t = yf.Ticker(ticker)
    _ = t.splits, t.dividends
    return _validate_actions(t, ticker)


def fetch_corporate_actions(symbols: list[str], exchange: str = "NSE") -> pd.DataFrame:
    """Fetch splits + dividends for each symbol, normalized to
    CORPORATE_ACTION_COLUMNS. Per-symbol failures are logged and skipped."""
    from stockpredictor.connectors.prices_yfinance import to_provider_ticker

    rows: list[dict] = []
    for symbol in symbols:
        ticker = to_provider_ticker(symbol, exchange)
        try:
            t = _fetch_ticker_actions(ticker)
        except Exception:
            logger.exception("Failed to fetch corporate actions for %s (%s)", symbol, ticker)
            continue

        for ex_date, ratio in t.splits.items():
            d = pd.Timestamp(ex_date).tz_localize(None).normalize()
            rows.append(
                {
                    "symbol": symbol,
                    "action_type": "split",
                    "ex_date": d.date(),
                    "knowable_date": d.date(),
                    "ratio": float(ratio),
                    "value": None,
                }
            )

        for ex_date, amount in t.dividends.items():
            d = pd.Timestamp(ex_date).tz_localize(None).normalize()
            rows.append(
                {
                    "symbol": symbol,
                    "action_type": "dividend",
                    "ex_date": d.date(),
                    "knowable_date": d.date(),
                    "ratio": None,
                    "value": float(amount),
                }
            )

    if not rows:
        return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    return pd.DataFrame(rows, columns=CORPORATE_ACTION_COLUMNS)
