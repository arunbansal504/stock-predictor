"""Structured-ish logging setup.

Deliberately stdlib-only (no structlog dependency) to keep the MVP dependency
footprint small — see architecture doc §16's push-back on adding infra before
a metric forces it. `LOG_JSON=true` switches to single-line JSON records,
which is what you want once logs are shipped to a real aggregator later.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from stockpredictor.common.config import get_settings

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def configure_logging(force: bool = False) -> None:
    """Idempotent logging setup. Call once at process start (CLI/orchestration
    entrypoints); safe to call repeatedly (e.g. in tests)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
