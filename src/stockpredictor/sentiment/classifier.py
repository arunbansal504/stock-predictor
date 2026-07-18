"""FinBERT polarity scoring (§9: "Sentiment/classification: FinBERT ... for
finance-tuned polarity").

`ProsusAI/finbert` is a BERT model fine-tuned on financial text, output as
three class probabilities (positive/negative/neutral). We reduce that to a
single signed `sentiment_score` in [-1, 1] = P(positive) - P(negative) --
the standard, simple way to turn a 3-class distribution into an orderable
polarity feature; neutral mass just shrinks the magnitude of both classes
rather than being dropped.

The model is loaded lazily and cached process-wide (`_get_pipeline`): the
nightly batch scores a few hundred articles across the whole universe in one
run, not per-request, so a one-time ~2-5s CPU load cost per process is
irrelevant next to that. Tests never trigger this -- they monkeypatch
`_get_pipeline` to avoid a real model download/inference in the fast unit
suite (see tests/unit/test_sentiment_classifier.py).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "ProsusAI/finbert"
SCORED_COLUMNS_ADDED = ["sentiment_score", "sentiment_label", "model_version"]


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:
    from transformers import pipeline

    logger.info("Loading FinBERT sentiment model (%s) -- one-time per process", MODEL_NAME)
    return pipeline("text-classification", model=MODEL_NAME, top_k=None)


def _polarity_from_scores(class_scores: list[dict[str, Any]]) -> tuple[float, str]:
    by_label = {item["label"].lower(): item["score"] for item in class_scores}
    positive = by_label.get("positive", 0.0)
    negative = by_label.get("negative", 0.0)
    top_label = max(class_scores, key=lambda item: item["score"])["label"].lower()
    return positive - negative, top_label


def score_articles(articles: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """Add `sentiment_score` (float, [-1, 1]), `sentiment_label`
    (positive/negative/neutral), and `model_version` columns, scored from
    `text_col`. Scores the headline rather than the RSS summary --
    summaries from aggregator feeds are often truncated mid-sentence or
    duplicate the title, and the headline alone is what FinBERT's training
    distribution (financial news headlines) most closely matches.

    `model_version` (stamped as `MODEL_NAME`) is what lets
    ingestion/news.py skip re-scoring an already-seen article safely: a
    published headline's score can't change since FinBERT is
    deterministic, but stamping the model identity means a future FinBERT
    upgrade naturally triggers a full re-score instead of silently mixing
    old- and new-model scores in the same column."""
    if articles.empty:
        out = articles.copy()
        for col in SCORED_COLUMNS_ADDED:
            out[col] = pd.Series(dtype="float64" if col == "sentiment_score" else "object")
        return out

    classifier = _get_pipeline()
    texts = articles[text_col].fillna("").tolist()
    predictions = classifier(texts)

    scores, labels = [], []
    for class_scores in predictions:
        score, label = _polarity_from_scores(class_scores)
        scores.append(score)
        labels.append(label)

    out = articles.copy()
    out["sentiment_score"] = scores
    out["sentiment_label"] = labels
    out["model_version"] = MODEL_NAME
    return out
