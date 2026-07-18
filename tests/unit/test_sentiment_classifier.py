"""Unit tests for FinBERT score wiring. The actual model is never loaded
here -- `_get_pipeline` is monkeypatched with a deterministic fake so the
fast unit suite doesn't download/run a real transformer (see
sentiment/classifier.py's docstring)."""

from __future__ import annotations

import pandas as pd

from stockpredictor.sentiment import classifier


def _fake_pipeline(texts: list[str]) -> list[list[dict]]:
    outputs = []
    for text in texts:
        if "surge" in text or "record profit" in text:
            outputs.append(
                [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.05}]
            )
        elif "plunge" in text or "loss" in text:
            outputs.append(
                [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.9}, {"label": "neutral", "score": 0.05}]
            )
        else:
            outputs.append(
                [{"label": "positive", "score": 0.2}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.6}]
            )
    return outputs


def test_score_articles_positive_headline_gets_positive_score(monkeypatch):
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame({"title": ["Company profits surge on record profit quarter"]})
    out = classifier.score_articles(df)
    assert out.iloc[0]["sentiment_score"] > 0
    assert out.iloc[0]["sentiment_label"] == "positive"


def test_score_articles_negative_headline_gets_negative_score(monkeypatch):
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame({"title": ["Shares plunge after quarterly loss"]})
    out = classifier.score_articles(df)
    assert out.iloc[0]["sentiment_score"] < 0
    assert out.iloc[0]["sentiment_label"] == "negative"


def test_score_articles_neutral_headline_near_zero(monkeypatch):
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame({"title": ["Company holds annual general meeting"]})
    out = classifier.score_articles(df)
    assert out.iloc[0]["sentiment_score"] == 0.0
    assert out.iloc[0]["sentiment_label"] == "neutral"


def test_score_articles_preserves_row_count_and_original_columns(monkeypatch):
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame({"title": ["Profit surge", "Loss reported"], "url": ["https://a", "https://b"]})
    out = classifier.score_articles(df)
    assert len(out) == 2
    assert "url" in out.columns
    assert list(out["url"]) == ["https://a", "https://b"]


def test_score_articles_empty_input_returns_empty_with_score_columns(monkeypatch):
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame(columns=["title"])
    out = classifier.score_articles(df)
    assert out.empty
    for col in classifier.SCORED_COLUMNS_ADDED:
        assert col in out.columns


def test_score_articles_stamps_model_version(monkeypatch):
    """model_version is what lets ingestion/news.py skip re-scoring an
    already-seen article safely -- see that module's docstring."""
    monkeypatch.setattr(classifier, "_get_pipeline", lambda: _fake_pipeline)
    df = pd.DataFrame({"title": ["Profit surge"]})
    out = classifier.score_articles(df)
    assert out.iloc[0]["model_version"] == classifier.MODEL_NAME
