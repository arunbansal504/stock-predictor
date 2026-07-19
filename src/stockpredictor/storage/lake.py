"""Analytical data lake: Bronze/Silver Parquet, queryable via DuckDB.

Architecture doc §5 (medallion architecture) and §13 (DuckDB+Parquet chosen
over a managed time-series DB for the personal-scale MVP — zero ops, trivial
backups, swap for Timescale/ClickHouse only if scale ever forces it).

Layout: ``{lake_root}/{layer}/{domain}/{symbol}.parquet`` — one file per
symbol per domain. This makes single-symbol reads O(1) file opens and keeps
appends cheap (read-modify-write one small file, not a giant partition).
Cross-symbol analytical queries go through DuckDB against a glob, which is
where the columnar format actually pays off.

Bronze is append-only in spirit but physically dedups on the natural key so
re-running an ingestion for a date range is idempotent (§5: "re-running a
date is safe"). Silver additionally guarantees sortedness and dtype
normalization; it's what feature code reads from.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from stockpredictor.common.config import get_settings
from stockpredictor.common.types import DataLayer


class Lake:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_settings().lake_root

    def _dir(self, layer: DataLayer, domain: str) -> Path:
        d = self.root / layer.value / domain
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, layer: DataLayer, domain: str, symbol: str) -> Path:
        return self._dir(layer, domain) / f"{symbol}.parquet"

    def write(
        self,
        df: pd.DataFrame,
        layer: DataLayer,
        domain: str,
        symbol: str,
        key_cols: list[str],
    ) -> int:
        """Upsert `df` into the (layer, domain, symbol) file, deduplicating on
        `key_cols` (last write wins) and keeping the file sorted by key_cols
        for efficient range reads. Returns the resulting row count.
        """
        path = self._path(layer, domain, symbol)
        if df.empty:
            return self._count(path)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()

        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined = combined.sort_values(key_cols).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        return len(combined)

    def read(
        self,
        layer: DataLayer,
        domain: str,
        symbol: str,
    ) -> pd.DataFrame:
        path = self._path(layer, domain, symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_all(
        self,
        layer: DataLayer,
        domain: str,
        order_by: tuple[str, ...] | None = ("symbol", "date"),
    ) -> pd.DataFrame:
        """Read every symbol's file for a (layer, domain) via DuckDB glob —
        the cross-sectional read path feature/ranking code uses (§7: rank
        features across the whole universe on each date).

        A parallel multi-file glob scan does not guarantee row order across
        runs; `order_by` (any of its columns present in this domain's
        schema) makes the result deterministic run-to-run — important
        because downstream code (e.g. models/ensemble.py's chronological
        base/meta split) is order-sensitive. Columns not present in this
        domain are silently skipped rather than erroring, so one shared
        default works across domains with different schemas."""
        d = self._dir(layer, domain)
        pattern = str(d / "*.parquet")
        if not any(d.glob("*.parquet")):
            return pd.DataFrame()
        con = duckdb.connect()
        try:
            query = f"SELECT * FROM read_parquet('{pattern}')"
            if order_by:
                schema_cols = {
                    row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pattern}')").fetchall()
                }
                cols = [c for c in order_by if c in schema_cols]
                if cols:
                    query += " ORDER BY " + ", ".join(cols)
            return con.execute(query).fetchdf()
        finally:
            con.close()

    def _count(self, path: Path) -> int:
        if not path.exists():
            return 0
        return len(pd.read_parquet(path))
