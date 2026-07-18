"""Data-quality gates (§22, §27): checks that abort the run before bad or
incomplete data reaches feature/model code, rather than letting a partial
failure silently propagate into a misleading ranking.
"""

from __future__ import annotations

import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when a gate fails. The orchestration flow lets this propagate
    to halt downstream tasks (§3 NFR: a *single-symbol* source failure
    degrades gracefully, but a run-wide quality collapse must stop the run
    rather than publish a ranking built on mostly-missing data)."""


def check_minimum_success_ratio(succeeded: int, total: int, min_ratio: float, stage: str) -> None:
    if total == 0:
        raise DataQualityError(f"[{stage}] zero symbols in universe -- nothing to check")
    ratio = succeeded / total
    if ratio < min_ratio:
        raise DataQualityError(
            f"[{stage}] only {succeeded}/{total} symbols succeeded ({ratio:.1%}), "
            f"below the required {min_ratio:.1%} -- aborting run"
        )


def check_non_empty(df: pd.DataFrame | None, stage: str) -> None:
    if df is None or df.empty:
        raise DataQualityError(f"[{stage}] produced an empty result -- aborting run")
