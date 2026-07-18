"""Alerting (§14, §23: "Alerting to email/Telegram on failures or drift
breaches"). MVP scope is Telegram-only (email is a documented later
addition, same pattern) and degrades gracefully to a log line when no
credentials are configured -- so the rest of the pipeline behaves
identically whether or not alerting is set up, matching the near-zero-
config MVP posture (§16).
"""

from __future__ import annotations

import httpx

from stockpredictor.common.config import get_settings
from stockpredictor.common.logging import get_logger

logger = get_logger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_alert(message: str, level: str = "warning") -> bool:
    """Send an alert via Telegram if configured, otherwise log it. Returns
    True if an external channel actually delivered the message (useful for
    tests/callers that want to distinguish "sent" from "only logged")."""
    settings = get_settings()
    log_fn = logger.error if level == "error" else logger.warning
    log_fn("ALERT [%s]: %s", level, message)

    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.info("Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset) -- alert logged only")
        return False

    url = TELEGRAM_API_URL.format(token=settings.telegram_bot_token)
    try:
        response = httpx.post(
            url,
            json={"chat_id": settings.telegram_chat_id, "text": f"[{level.upper()}] {message}"},
            timeout=10.0,
        )
        response.raise_for_status()
        return True
    except Exception:
        logger.exception("Failed to deliver Telegram alert (message was still logged above)")
        return False
