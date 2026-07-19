"""Regression test for the "different stock ranks every manual run" bug.

Reuses the exact synthetic fixtures from test_nightly_flow.py (no network)
so this exercises the real code path a live rerun takes, not a stripped-down
approximation of it.
"""

from __future__ import annotations

import pandas as pd

import stockpredictor.ingestion.corporate_actions as ca_ingestion
import stockpredictor.ingestion.fundamentals as fundamentals_ingestion
import stockpredictor.ingestion.macro as macro_ingestion
import stockpredictor.ingestion.news as news_ingestion
import stockpredictor.ingestion.prices as prices_ingestion
import stockpredictor.ingestion.universe as universe_ingestion
from stockpredictor.common.config import Settings
from stockpredictor.orchestration import nightly_flow
from stockpredictor.ranking.registry import read_latest_rankings
from stockpredictor.storage.db import init_db, make_engine, make_sessionmaker
from stockpredictor.storage.lake import Lake

from tests.integration.test_nightly_flow import (
    _fake_corporate_actions,
    _fake_fundamentals,
    _fake_macro,
    _fake_news,
    _fake_nse_universe,
    _fake_prices,
    _fake_score_articles,
)

RANK_COMPARISON_COLUMNS = ["symbol", "date", "rank", "score", "meta_score"]


def _patch_connectors(monkeypatch) -> None:
    monkeypatch.setattr(prices_ingestion.prices_yfinance, "fetch_prices", _fake_prices)
    monkeypatch.setattr(macro_ingestion.macro_yfinance, "fetch_macro_series", _fake_macro)
    monkeypatch.setattr(ca_ingestion, "fetch_corporate_actions", _fake_corporate_actions)
    monkeypatch.setattr(universe_ingestion, "fetch_nifty500_constituents", _fake_nse_universe)
    monkeypatch.setattr(fundamentals_ingestion, "fetch_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(news_ingestion, "fetch_news_for_symbol", _fake_news)
    monkeypatch.setattr(news_ingestion, "score_articles", _fake_score_articles)


def _run_pipeline(lake: Lake, sessionmaker, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(nightly_flow, "Lake", lambda: lake)
    monkeypatch.setattr(nightly_flow, "make_sessionmaker", lambda eng: sessionmaker)
    monkeypatch.setattr(nightly_flow, "init_db", lambda eng: None)
    _patch_connectors(monkeypatch)

    nightly_flow.nightly_pipeline(years_of_history=1, horizons={"5d": 5}, top_k=3, top_n_explain=5)
    return read_latest_rankings(lake, "5d")


def test_same_day_rerun_against_the_same_lake_produces_identical_rankings(tmp_path, monkeypatch):
    """The user's actual complaint: run the pipeline, then run it again the
    same day without anything changing upstream. Both runs must agree on
    every symbol's rank, score, and tiebreak -- not just be "close"."""
    lake = Lake(root=tmp_path / "lake")
    engine = make_engine(Settings(database_url=f"sqlite:///{tmp_path / 'app.db'}"))
    init_db(engine)
    sessionmaker = make_sessionmaker(engine)
    monkeypatch.setattr(nightly_flow, "make_engine", lambda: engine)

    ranked_run1 = _run_pipeline(lake, sessionmaker, monkeypatch)
    ranked_run2 = _run_pipeline(lake, sessionmaker, monkeypatch)

    pd.testing.assert_frame_equal(
        ranked_run1[RANK_COMPARISON_COLUMNS].sort_values("symbol").reset_index(drop=True),
        ranked_run2[RANK_COMPARISON_COLUMNS].sort_values("symbol").reset_index(drop=True),
    )


def test_two_independent_runs_with_identical_inputs_produce_identical_rankings(tmp_path, monkeypatch):
    """Same synthetic "day", but two entirely separate lakes/DBs -- isolates
    determinism of the compute path itself (model fit, ordering, ranking)
    from the idempotent-fetch-skip behavior the same-lake test above relies
    on."""
    lake_a = Lake(root=tmp_path / "lake_a")
    engine_a = make_engine(Settings(database_url=f"sqlite:///{tmp_path / 'app_a.db'}"))
    init_db(engine_a)
    sessionmaker_a = make_sessionmaker(engine_a)
    monkeypatch.setattr(nightly_flow, "make_engine", lambda: engine_a)
    ranked_a = _run_pipeline(lake_a, sessionmaker_a, monkeypatch)

    lake_b = Lake(root=tmp_path / "lake_b")
    engine_b = make_engine(Settings(database_url=f"sqlite:///{tmp_path / 'app_b.db'}"))
    init_db(engine_b)
    sessionmaker_b = make_sessionmaker(engine_b)
    monkeypatch.setattr(nightly_flow, "make_engine", lambda: engine_b)
    ranked_b = _run_pipeline(lake_b, sessionmaker_b, monkeypatch)

    pd.testing.assert_frame_equal(
        ranked_a[RANK_COMPARISON_COLUMNS].sort_values("symbol").reset_index(drop=True),
        ranked_b[RANK_COMPARISON_COLUMNS].sort_values("symbol").reset_index(drop=True),
    )
