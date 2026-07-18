from __future__ import annotations

import pandas as pd
import pytest

from stockpredictor.monitoring.quality_gates import (
    DataQualityError,
    check_minimum_success_ratio,
    check_non_empty,
)


def test_check_minimum_success_ratio_passes_above_threshold():
    check_minimum_success_ratio(succeeded=90, total=100, min_ratio=0.8, stage="test")  # no raise


def test_check_minimum_success_ratio_raises_below_threshold():
    with pytest.raises(DataQualityError, match="70/100"):
        check_minimum_success_ratio(succeeded=70, total=100, min_ratio=0.8, stage="test")


def test_check_minimum_success_ratio_raises_on_zero_total():
    with pytest.raises(DataQualityError, match="zero symbols"):
        check_minimum_success_ratio(succeeded=0, total=0, min_ratio=0.8, stage="test")


def test_check_non_empty_passes_for_populated_frame():
    check_non_empty(pd.DataFrame({"a": [1]}), stage="test")  # no raise


def test_check_non_empty_raises_for_empty_frame():
    with pytest.raises(DataQualityError, match="empty result"):
        check_non_empty(pd.DataFrame(), stage="test")


def test_check_non_empty_raises_for_none():
    with pytest.raises(DataQualityError, match="empty result"):
        check_non_empty(None, stage="test")
