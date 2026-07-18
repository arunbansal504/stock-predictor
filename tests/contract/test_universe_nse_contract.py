"""Contract test for the NSE universe connector (§5, §22). No network --
pins the normalization contract against a realistic sample of NSE's actual
CSV shape (verified live against nsearchives.nseindia.com before this
module was written: 500 rows, columns "Company Name, Industry, Symbol,
Series, ISIN Code", all Series == "EQ", no nulls, no duplicate symbols).
"""

from __future__ import annotations

import pytest

from stockpredictor.connectors import universe_nse as ca

_SAMPLE_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
360 ONE WAM Ltd.,Financial Services,360ONE,EQ,INE466L01038
3M India Ltd.,Diversified,3MINDIA,EQ,INE470A01017
ABB India Ltd.,Capital Goods,ABB,EQ,INE117A01022
Mahindra & Mahindra Ltd.,Automobile and Auto Components,M&M,EQ,INE101A01026
"""


def test_fetch_nifty500_constituents_normalizes_schema(monkeypatch):
    monkeypatch.setattr(ca, "_fetch_csv_text", lambda url: _SAMPLE_CSV)

    df = ca.fetch_nifty500_constituents()
    assert list(df.columns) == ca.UNIVERSE_COLUMNS
    assert len(df) == 4
    assert (df["exchange"] == "NSE").all()

    row = df[df["symbol"] == "M&M"].iloc[0]
    assert row["name"] == "Mahindra & Mahindra Ltd."
    assert row["sector"] == "Automobile and Auto Components"
    assert row["isin"] == "INE101A01026"


def test_fetch_nifty500_constituents_filters_non_eq_series(monkeypatch):
    csv_with_other_series = _SAMPLE_CSV + "Some Trust,Trusts,SOMETR,ETF,INE000000000\n"
    monkeypatch.setattr(ca, "_fetch_csv_text", lambda url: csv_with_other_series)

    df = ca.fetch_nifty500_constituents()
    assert "SOMETR" not in set(df["symbol"])
    assert len(df) == 4


def test_fetch_nifty500_constituents_raises_on_schema_drift(monkeypatch):
    broken_csv = "Symbol,Name\nABB,ABB India\n"
    monkeypatch.setattr(ca, "_fetch_csv_text", lambda url: broken_csv)

    with pytest.raises(ValueError, match="schema changed"):
        ca.fetch_nifty500_constituents()


def test_fetch_nifty500_constituents_raises_on_empty_result(monkeypatch):
    empty_csv = "Company Name,Industry,Symbol,Series,ISIN Code\n"
    monkeypatch.setattr(ca, "_fetch_csv_text", lambda url: empty_csv)

    with pytest.raises(ValueError, match="zero rows"):
        ca.fetch_nifty500_constituents()


def test_fetch_nifty500_constituents_raises_on_duplicate_symbols(monkeypatch):
    dup_csv = _SAMPLE_CSV + "ABB India Ltd. Duplicate,Capital Goods,ABB,EQ,INE117A01099\n"
    monkeypatch.setattr(ca, "_fetch_csv_text", lambda url: dup_csv)

    with pytest.raises(ValueError, match="duplicate symbols"):
        ca.fetch_nifty500_constituents()
