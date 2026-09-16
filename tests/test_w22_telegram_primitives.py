"""W22: the raw Telegram HTTP primitives added to TelegramNotifier —
send_with_buttons, edit_message, answer_callback, get_updates.

`telegram.py` imports aiohttp LAZILY, inside each method body (`import
aiohttp`), rather than at module level — so there is no module-level
`ait.notifications.telegram.aiohttp` attribute to monkeypatch. These tests
instead patch `sys.modules["aiohttp"]` with a fake module before calling
into TelegramNotifier: the local `import aiohttp` statement finds the fake
already in sys.modules and binds it, exactly like the real import would.
"""
from __future__ import annotations

import sys
import types

import pytest

from ait.notifications.telegram import TelegramNotifier


class _FakeResponse:
    def __init__(self, status=200, json_body=None, text_body=""):
        self.status = status
        self._json_body = json_body if json_body is not None else {"ok": True, "result": {}}
        self._text_body = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json_body

    async def text(self):
        return self._text_body


class _FakeSession:
    def __init__(self, response_queue, calls):
        self._queue = response_queue
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        self._calls.append({"url": url, "json": json})
        return self._queue.pop(0) if self._queue else _FakeResponse()


@pytest.fixture
def fake_aiohttp(monkeypatch):
    """Yields (calls, responses). Append _FakeResponse instances to
    `responses` in the order the test expects them to be consumed."""
    calls: list[dict] = []
    responses: list[_FakeResponse] = []
    fake_mod = types.SimpleNamespace(
        ClientSession=lambda *a, **kw: _FakeSession(responses, calls),
        ClientTimeout=lambda total=None: None,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_mod)
    return calls, responses


def _notifier() -> TelegramNotifier:
    return TelegramNotifier(bot_token="TESTTOKEN", chat_id="123456")


class TestSendWithButtons:
    def test_sends_inline_keyboard_and_returns_message_id(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(json_body={"ok": True, "result": {"message_id": 777}}))
        tg = _notifier()

        import asyncio
        msg_id = asyncio.run(tg.send_with_buttons(
            "Is this yours?", [("Yes", "manual:yes:K"), ("No", "manual:no:K")]))

        assert msg_id == 777
        assert len(calls) == 1
        payload = calls[0]["json"]
        assert payload["text"] == "Is this yours?"
        assert payload["chat_id"] == "123456"
        buttons = payload["reply_markup"]["inline_keyboard"][0]
        assert buttons[0] == {"text": "Yes", "callback_data": "manual:yes:K"}
        assert buttons[1] == {"text": "No", "callback_data": "manual:no:K"}

    def test_failure_returns_none(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=400, json_body={"ok": False, "description": "bad"}))
        tg = _notifier()
        import asyncio
        msg_id = asyncio.run(tg.send_with_buttons("x", [("Y", "y")]))
        assert msg_id is None

    def test_disabled_notifier_skips_the_network_call(self, fake_aiohttp):
        calls, _ = fake_aiohttp
        tg = TelegramNotifier(bot_token="", chat_id="")
        import asyncio
        msg_id = asyncio.run(tg.send_with_buttons("x", [("Y", "y")]))
        assert msg_id is None
        assert calls == []


class TestEditMessage:
    def test_edits_text_and_buttons(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=200))
        tg = _notifier()
        import asyncio
        ok = asyncio.run(tg.edit_message(777, "updated", [("Y", "manual:yes:K")]))
        assert ok is True
        payload = calls[0]["json"]
        assert payload["message_id"] == 777
        assert payload["text"] == "updated"
        assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "manual:yes:K"

    def test_edit_without_buttons_omits_reply_markup(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=200))
        tg = _notifier()
        import asyncio
        asyncio.run(tg.edit_message(777, "updated"))
        assert "reply_markup" not in calls[0]["json"]

    def test_failure_returns_false(self, fake_aiohttp):
        _, responses = fake_aiohttp
        responses.append(_FakeResponse(status=400, text_body="bad request"))
        tg = _notifier()
        import asyncio
        assert asyncio.run(tg.edit_message(777, "x")) is False


class TestAnswerCallback:
    def test_answers_with_the_given_text(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=200))
        tg = _notifier()
        import asyncio
        ok = asyncio.run(tg.answer_callback("cbq1", "Recorded: yes"))
        assert ok is True
        assert calls[0]["json"] == {"callback_query_id": "cbq1", "text": "Recorded: yes"}


class TestGetUpdates:
    def test_returns_the_result_list(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        canned = [{"update_id": 5, "message": {"text": "yes"}}]
        responses.append(_FakeResponse(json_body={"ok": True, "result": canned}))
        tg = _notifier()
        import asyncio
        got = asyncio.run(tg.get_updates(offset=5, timeout=1))
        assert got == canned
        assert calls[0]["json"]["offset"] == 5
        assert calls[0]["json"]["timeout"] == 1
        assert set(calls[0]["json"]["allowed_updates"]) == {"message", "callback_query"}

    def test_failure_returns_empty_list_not_raises(self, fake_aiohttp):
        _, responses = fake_aiohttp
        responses.append(_FakeResponse(status=500, json_body={"ok": False}))
        tg = _notifier()
        import asyncio
        assert asyncio.run(tg.get_updates(offset=0)) == []

    def test_disabled_notifier_returns_empty_without_a_call(self, fake_aiohttp):
        calls, _ = fake_aiohttp
        tg = TelegramNotifier(bot_token="", chat_id="")
        import asyncio
        assert asyncio.run(tg.get_updates(offset=0)) == []
        assert calls == []


class TestTokenNeverLeaksOnError:
    """Mirrors the existing send() redaction contract: an exception message
    containing the bot token must have it masked before logging."""

    def test_bot_token_is_redacted_from_a_raised_error(self, monkeypatch):
        class _ExplodingSession:
            async def __aenter__(self):
                raise RuntimeError(
                    "connect failed to https://api.telegram.org/botTESTTOKEN/sendMessage")
            async def __aexit__(self, *exc):
                return False

        fake_mod = types.SimpleNamespace(
            ClientSession=lambda *a, **kw: _ExplodingSession(),
            ClientTimeout=lambda total=None: None,
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_mod)
        tg = _notifier()

        # structlog output isn't captured by caplog in this codebase (see
        # the W24/W21 tests' own pattern) — patch the logger directly.
        import ait.notifications.telegram as telegram_mod
        logged = []
        monkeypatch.setattr(telegram_mod.log, "warning",
                            lambda event, **kw: logged.append((event, kw)))

        import asyncio
        result = asyncio.run(tg.send_with_buttons("x", [("Y", "y")]))

        assert result is None
        assert len(logged) == 1
        error_text = logged[0][1]["error"]
        assert "TESTTOKEN" not in error_text
        assert "/bot***" in error_text
