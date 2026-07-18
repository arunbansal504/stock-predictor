from __future__ import annotations

import httpx
import pytest

from stockpredictor.common.config import Settings
from stockpredictor.monitoring import alerts


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    alerts.get_settings.cache_clear()
    yield
    alerts.get_settings.cache_clear()


def test_send_alert_logs_only_when_telegram_not_configured(monkeypatch):
    monkeypatch.setattr(alerts, "get_settings", lambda: Settings())
    sent = alerts.send_alert("something went wrong")
    assert sent is False


def test_send_alert_sends_via_telegram_when_configured(monkeypatch):
    configured = Settings(telegram_bot_token="TESTTOKEN", telegram_chat_id="12345")
    monkeypatch.setattr(alerts, "get_settings", lambda: configured)

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", fake_post)

    sent = alerts.send_alert("pipeline failed", level="error")
    assert sent is True
    assert "TESTTOKEN" in captured["url"]
    assert captured["json"]["chat_id"] == "12345"
    assert "pipeline failed" in captured["json"]["text"]


def test_send_alert_returns_false_when_telegram_call_fails(monkeypatch):
    configured = Settings(telegram_bot_token="TESTTOKEN", telegram_chat_id="12345")
    monkeypatch.setattr(alerts, "get_settings", lambda: configured)

    def fake_post(url, json, timeout):
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(alerts.httpx, "post", fake_post)

    sent = alerts.send_alert("pipeline failed")
    assert sent is False  # must not raise -- alerting failures shouldn't crash the caller
