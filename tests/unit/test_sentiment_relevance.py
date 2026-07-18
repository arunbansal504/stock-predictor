from __future__ import annotations

import pandas as pd

from stockpredictor.sentiment.relevance import filter_relevant, is_relevant


def test_is_relevant_matches_bare_symbol_as_whole_word():
    assert is_relevant("RELIANCE hits new 52-week high", "RELIANCE", "Reliance Industries Limited")


def test_is_relevant_rejects_symbol_as_substring_of_another_word():
    # "ITC" should not match inside "STITCH" or similar -- word-boundary matters.
    assert not is_relevant("Local stitching industry sees growth", "ITC", "ITC Limited")


def test_is_relevant_matches_company_name_tokens_without_symbol():
    assert is_relevant(
        "Tata Consultancy Services wins major European contract", "TCS", "Tata Consultancy Services Limited"
    )


def test_is_relevant_rejects_unrelated_article():
    assert not is_relevant("Crude oil prices surge on Middle East tensions", "RELIANCE", "Reliance Industries Limited")


def test_is_relevant_false_for_empty_text():
    assert not is_relevant("", "RELIANCE", "Reliance Industries Limited")


def test_filter_relevant_keeps_only_matching_rows():
    df = pd.DataFrame(
        {
            "title": ["Reliance Industries Q4 profit jumps", "Unrelated market roundup"],
            "summary": ["Strong results for Reliance", "General index commentary"],
            "url": ["https://a", "https://b"],
        }
    )
    out = filter_relevant(df, "RELIANCE", "Reliance Industries Limited")
    assert list(out["url"]) == ["https://a"]


def test_filter_relevant_empty_input_returns_empty():
    df = pd.DataFrame(columns=["title", "summary", "url"])
    out = filter_relevant(df, "RELIANCE", "Reliance Industries Limited")
    assert out.empty
