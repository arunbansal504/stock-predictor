"""Contract test for the fundamentals connector (§5, §22). No network --
pins normalization + PIT-stamping against a realistic shape of yfinance's
`Ticker.financials` / `Ticker.balance_sheet` / `Ticker.earnings_dates`
(verified live against RELIANCE.NS/TCS.NS before this module was written:
annual columns are fiscal period-end Timestamps, most-recent-first; row
labels "Total Revenue", "Net Income", "Diluted EPS", "Stockholders Equity",
"Total Debt", "Total Assets", "Ordinary Shares Number").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.connectors import fundamentals_yfinance as fa


def _statements(include_latest_partial_year: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31"])
    financials = pd.DataFrame(
        {
            periods[0]: [np.nan if include_latest_partial_year else 8.0e11, 1.0e13, np.nan],
            periods[1]: [6.96e11, 9.6e12, 51.47],
            periods[2]: [6.96e11, 9.0e12, 51.45],
        },
        index=["Net Income", "Total Revenue", "Diluted EPS"],
    )
    balance_sheet = pd.DataFrame(
        {
            periods[0]: [8.4e12, 3.7e12, 2.18e13, 1.35e10],
            periods[1]: [8.4e12, 3.7e12, 1.95e13, 1.35e10],
            periods[2]: [7.9e12, 3.5e12, 1.76e13, 1.35e10],
        },
        index=["Stockholders Equity", "Total Debt", "Total Assets", "Ordinary Shares Number"],
    )
    return financials, balance_sheet


def _earnings_dates() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        ["2026-04-24 09:00:00-04:00", "2025-04-25 10:00:00-04:00", "2024-04-22 09:00:00-04:00"],
        name="Earnings Date",
    )
    return pd.DataFrame({"Reported EPS": [12.5, 8.3, 14.0]}, index=idx)


def test_fetch_fundamentals_normalizes_schema_and_drops_unreported_year(monkeypatch):
    financials, balance_sheet = _statements(include_latest_partial_year=True)
    monkeypatch.setattr(fa, "_fetch_statements", lambda ticker: (financials, balance_sheet, _earnings_dates()))

    df = fa.fetch_fundamentals("RELIANCE")
    assert list(df.columns) == fa.FUNDAMENTALS_COLUMNS
    # The 2026-03-31 column has NaN Net Income -> dropped (not fabricated as 0).
    assert len(df) == 2
    assert set(df["period_end"].astype(str)) == {"2025-03-31", "2024-03-31"}


def test_fetch_fundamentals_knowable_date_is_first_announcement_after_period_end(monkeypatch):
    financials, balance_sheet = _statements(include_latest_partial_year=True)
    monkeypatch.setattr(fa, "_fetch_statements", lambda ticker: (financials, balance_sheet, _earnings_dates()))

    df = fa.fetch_fundamentals("RELIANCE").set_index("period_end")
    # FY2025-03-31 results first announced 2025-04-25 (from the fake earnings calendar).
    row = df.loc[pd.Timestamp("2025-03-31").date()]
    assert row["knowable_date"] == pd.Timestamp("2025-04-25").date()


def test_fetch_fundamentals_knowable_date_always_after_period_end(monkeypatch):
    financials, balance_sheet = _statements(include_latest_partial_year=True)
    monkeypatch.setattr(fa, "_fetch_statements", lambda ticker: (financials, balance_sheet, _earnings_dates()))

    df = fa.fetch_fundamentals("RELIANCE")
    for _, row in df.iterrows():
        assert row["knowable_date"] > row["period_end"]


def test_fetch_fundamentals_falls_back_to_filing_deadline_without_earnings_calendar(monkeypatch):
    financials, balance_sheet = _statements(include_latest_partial_year=True)
    monkeypatch.setattr(fa, "_fetch_statements", lambda ticker: (financials, balance_sheet, None))

    df = fa.fetch_fundamentals("RELIANCE").set_index("period_end")
    row = df.loc[pd.Timestamp("2025-03-31").date()]
    expected = (pd.Timestamp("2025-03-31") + pd.Timedelta(days=fa.ANNUAL_FILING_DEADLINE_DAYS)).date()
    assert row["knowable_date"] == expected


def test_fetch_fundamentals_empty_statements_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(fa, "_fetch_statements", lambda ticker: (pd.DataFrame(), pd.DataFrame(), None))
    df = fa.fetch_fundamentals("NODATA")
    assert df.empty
    assert list(df.columns) == fa.FUNDAMENTALS_COLUMNS


def test_fetch_fundamentals_skips_symbol_on_fetch_exception(monkeypatch):
    def fail(ticker):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(fa, "_fetch_statements", fail)
    df = fa.fetch_fundamentals("BADSYMBOL")
    assert df.empty
