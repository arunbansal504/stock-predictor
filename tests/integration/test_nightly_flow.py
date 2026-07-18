"""Integration test for the nightly Prefect flow (§12, §22: "end-to-end
nightly-run smoke test"). No network -- connector fetch functions are
monkeypatched with small synthetic data; storage points at a tmp_path lake
and a tmp_path SQLite DB via monkeypatching the flow module's top-level
`Lake`/`make_engine` names, since the flow's entrypoint intentionally reads
those from process-wide config (a real orchestration entrypoint should).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

import stockpredictor.ingestion.corporate_actions as ca_ingestion
import stockpredictor.ingestion.fundamentals as fundamentals_ingestion
import stockpredictor.ingestion.macro as macro_ingestion
import stockpredictor.ingestion.news as news_ingestion
import stockpredictor.ingestion.prices as prices_ingestion
import stockpredictor.ingestion.universe as universe_ingestion
from stockpredictor.common.config import Settings
from stockpredictor.orchestration import nightly_flow
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake
from stockpredictor.storage.models import RunMetadata


def _fake_nse_universe() -> pd.DataFrame:
    # A small fake universe, not the real 500 -- keeps this test fast while
    # still exercising the live-fetch code path (sync_universe_from_nse),
    # not just the CSV fallback.
    return pd.DataFrame(
        [
            {"symbol": f"FAKE{i}", "exchange": "NSE", "name": f"Fake Co {i}", "sector": "IT", "isin": f"INE{i:09d}"}
            for i in range(5)
        ]
    )


def _fake_prices(symbols, start, end, exchange="NSE"):
    symbol = symbols[0]
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(n, 100_000, dtype="int64"),
            "source": ["yfinance"] * n,
        }
    )


def _fake_macro(series_names, start, end):
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(7)
    frames = []
    for name in series_names:
        closes = 1000 + np.cumsum(rng.normal(0, 1, n))
        frames.append(pd.DataFrame({"series": [name] * n, "date": dates, "close": closes, "source": ["yfinance"] * n}))
    return pd.concat(frames, ignore_index=True)


def _fake_corporate_actions(symbols, exchange="NSE"):
    return pd.DataFrame(columns=["symbol", "action_type", "ex_date", "knowable_date", "ratio", "value"])


def _fake_fundamentals(symbol, exchange="NSE"):
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "period_end": pd.Timestamp("2023-03-31").date(),
                "knowable_date": pd.Timestamp("2023-05-01").date(),
                "revenue": 1000.0,
                "net_income": 100.0,
                "eps": 10.0,
                "total_equity": 500.0,
                "total_debt": 200.0,
                "total_assets": 1000.0,
                "shares_outstanding": 10.0,
            }
        ]
    )


def _fake_news(symbol, company_name):
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "published_date": pd.Timestamp("2026-07-17").date(),
                "title": f"{company_name} reports strong results",
                "summary": "Synthetic test article.",
                "url": f"https://example.com/{symbol}",
                "source": "Test Wire",
            }
        ]
    )


def _fake_score_articles(articles, text_col="title"):
    out = articles.copy()
    out["sentiment_score"] = 0.5
    out["sentiment_label"] = "positive"
    return out


def test_nightly_pipeline_runs_end_to_end_and_records_success(tmp_path, monkeypatch):
    lake = Lake(root=tmp_path / "lake")
    engine = make_engine(Settings(database_url=f"sqlite:///{tmp_path / 'app.db'}"))
    init_db(engine)
    sessionmaker = make_sessionmaker(engine)

    monkeypatch.setattr(nightly_flow, "Lake", lambda: lake)
    monkeypatch.setattr(nightly_flow, "make_engine", lambda: engine)
    monkeypatch.setattr(nightly_flow, "make_sessionmaker", lambda eng: sessionmaker)
    monkeypatch.setattr(nightly_flow, "init_db", lambda eng: None)

    monkeypatch.setattr(prices_ingestion.prices_yfinance, "fetch_prices", _fake_prices)
    monkeypatch.setattr(macro_ingestion.macro_yfinance, "fetch_macro_series", _fake_macro)
    monkeypatch.setattr(ca_ingestion, "fetch_corporate_actions", _fake_corporate_actions)
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_universe)
    monkeypatch.setattr(fundamentals_ingestion, "fetch_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", _fake_news)
    monkeypatch.setattr(news_ingestion, "score_articles", _fake_score_articles)

    run_id = nightly_flow.nightly_pipeline(
        years_of_history=1, horizons={"5d": 5}, top_k=3, top_n_explain=5
    )
    assert isinstance(run_id, str)

    session = sessionmaker()
    try:
        rows = session.execute(select(RunMetadata).where(RunMetadata.run_id == run_id)).scalars().all()
    finally:
        session.close()

    statuses = {r.stage: r.status for r in rows}
    assert statuses, "no run_metadata rows were recorded"
    assert all(status == "success" for status in statuses.values()), statuses
    assert "sync_universe" in statuses
    assert "ingest_fundamentals" in statuses
    assert "ingest_news" in statuses
    assert "build_features" in statuses
    assert "build_labels" in statuses
    assert "predict_rank_explain[5d]" in statuses

    # Verify the pipeline actually produced ranked, explained output --
    # not just that no stage raised.
    from stockpredictor.ranking.registry import read_latest_rankings
    from stockpredictor.explain.registry import read_explanations

    rankings = read_latest_rankings(lake, "5d")
    assert not rankings.empty
    assert rankings["rank"].min() == 1

    explanations = read_explanations(lake, "5d")
    assert not explanations.empty
    assert set(explanations["symbol"]).issubset(set(rankings["symbol"]))
