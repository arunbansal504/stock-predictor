"""Central settings + YAML config loading.

Two kinds of configuration on purpose:

- `.env` / environment variables -> `Settings` (pydantic-settings): secrets and
  per-environment knobs (DB URL, log level). Never committed.
- `config/*.yaml` -> plain dicts loaded by `load_yaml_config`: universe,
  sources, model, backtest definitions. Versioned in git, safe to review in a
  PR, no secrets allowed in them.

Defaults are chosen so the whole system runs with zero configuration on a
fresh checkout (SQLite app DB, local DuckDB/Parquet lake) — matching the
"near-zero budget" MVP posture from the architecture doc.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/stockpredictor/common/config.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="dev")

    # Defaults to a local SQLite file so `pytest` / a fresh dev machine needs
    # no running Postgres. Point at a real Postgres URL (see docker-compose.yml)
    # for a production-like run.
    database_url: str = Field(default=f"sqlite:///{REPO_ROOT / 'data' / 'app.db'}")

    lake_root: Path = Field(default=REPO_ROOT / "data")

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # Optional, unset by default (Phase 3+ paid sources / LLM / alerting).
    alpha_vantage_api_key: str | None = None
    polygon_api_key: str | None = None
    finnhub_api_key: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @property
    def bronze_dir(self) -> Path:
        return self.lake_root / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.lake_root / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.lake_root / "gold"


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings. Use `get_settings.cache_clear()` in tests
    that need to re-read environment variables."""
    return Settings()


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML file from config/ by name, e.g. load_yaml_config("universe.yaml")."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
