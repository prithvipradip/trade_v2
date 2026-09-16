"""Telegram notifications for trade alerts and daily summaries.

Setup instructions:
1. Message @BotFather on Telegram
2. Send /newbot and follow prompts to create your bot
3. Copy the bot token (looks like: 123456789:ABCdefGhIjKlMnOpQrStUvWxYz)
4. Start a chat with your bot and send any message
5. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
6. Find your chat_id in the response JSON
7. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file
"""

from __future__ import annotations

from ait.utils.logging import get_logger

log = get_logger("notifications.telegram")


class TelegramNotifier:
    """Sends notifications via Telegram bot."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._bot = None

        if not self._enabled:
            log.warning(
                "telegram_not_configured",
                hint="Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env",
            )

    @property
    def enabled(self) -> bool:
        """W22: public read of configured-ness, so callers (e.g. the
        manual-positions poller) can skip work cleanly instead of reaching
        into the private _enabled attribute."""
        return self._enabled

    async def send(self, message: str) -> bool:
        """Send a message via Telegram.

        Returns True if sent successfully, False otherwise.
        Messages are truncated to 4096 chars (Telegram limit).
        """
        if not self._enabled:
            log.debug("telegram_disabled", message=message[:100])
            return False

        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            # Plain text (no parse_mode) to avoid Markdown escape errors on
            # special chars like $, (, ), _, *, ` in trade messages
            payload = {
                "chat_id": self._chat_id,
                "text": message[:4096],
                "disable_web_page_preview": True,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.debug("telegram_sent", chars=len(message))
                        return True
                    else:
                        body = await resp.text()
                        log.warning("telegram_send_failed", status=resp.status, body=body[:200])
                        return False

        except Exception as e:
            log.warning("telegram_error", error=__import__("re").sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])
            return False

    async def send_with_buttons(
        self, message: str, buttons: list[tuple[str, str]]
    ) -> int | None:
        """W22: send a message with tappable inline-keyboard buttons.

        `buttons` is [(label, callback_data), ...], rendered as one row.
        Returns the sent message's message_id (needed to edit it later and
        to resolve a reply-to-message back to this prompt), or None on
        failure — same disabled/error handling as send().
        """
        if not self._enabled:
            log.debug("telegram_disabled", message=message[:100])
            return None
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": message[:4096],
                "disable_web_page_preview": True,
                "reply_markup": {
                    "inline_keyboard": [
                        [{"text": label, "callback_data": data}
                         for label, data in buttons]
                    ]
                },
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    body = await resp.json()
                    if resp.status == 200 and body.get("ok"):
                        return body["result"]["message_id"]
                    log.warning("telegram_send_buttons_failed", status=resp.status,
                                body=str(body)[:200])
                    return None
        except Exception as e:
            log.warning("telegram_error", error=__import__("re").sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])
            return None

    async def edit_message(
        self, message_id: int, text: str, buttons: list[tuple[str, str]] | None = None
    ) -> bool:
        """W22: update a previously-sent message's text (and optionally its
        buttons) in place — used to show the operator which answer is
        currently recorded, including after a correction."""
        if not self._enabled:
            return False
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._token}/editMessageText"
            payload = {
                "chat_id": self._chat_id,
                "message_id": message_id,
                "text": text[:4096],
            }
            if buttons is not None:
                payload["reply_markup"] = {
                    "inline_keyboard": [
                        [{"text": label, "callback_data": data}
                         for label, data in buttons]
                    ]
                }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    log.warning("telegram_edit_failed", status=resp.status, body=body[:200])
                    return False
        except Exception as e:
            log.warning("telegram_error", error=__import__("re").sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])
            return False

    async def answer_callback(self, callback_query_id: str, text: str = "") -> bool:
        """W22: acknowledge a tapped button. Telegram requires this or the
        button shows a spinner on the operator's phone until it times out."""
        if not self._enabled:
            return False
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._token}/answerCallbackQuery"
            payload = {"callback_query_id": callback_query_id, "text": text[:200]}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200
        except Exception as e:
            log.warning("telegram_error", error=__import__("re").sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])
            return False

    async def get_updates(self, offset: int, timeout: int = 25) -> list[dict]:
        """W22: long-poll inbound updates (messages + button taps) since
        `offset`. `timeout` is Telegram's own long-poll wait (seconds) —
        the HTTP call blocks server-side until an update arrives or it
        elapses, so the poller isn't a tight loop. Returns [] on any
        failure (network blip, disabled) rather than raising, so a single
        bad poll can't kill the poller task; the caller just tries again.
        """
        if not self._enabled:
            return []
        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self._token}/getUpdates"
            params = {"offset": offset, "timeout": timeout,
                      "allowed_updates": ["message", "callback_query"]}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url, json=params,
                        timeout=aiohttp.ClientTimeout(total=timeout + 10)) as resp:
                    body = await resp.json()
                    if resp.status == 200 and body.get("ok"):
                        return body.get("result", [])
                    log.warning("telegram_get_updates_failed", status=resp.status,
                                body=str(body)[:200])
                    return []
        except Exception as e:
            log.warning("telegram_error", error=__import__("re").sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])
            return []

    async def send_trade_alert(
        self,
        action: str,
        symbol: str,
        strategy: str,
        contracts: int,
        price: float,
        confidence: float,
    ) -> bool:
        """Send a formatted trade alert."""
        emoji = "🟢" if action == "BUY" else "🔴"
        msg = (
            f"{emoji} {action} {contracts}x {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Price: ${price:.2f}\n"
            f"Confidence: {confidence:.0%}"
        )
        return await self.send(msg)

    async def send_error_alert(self, error: str) -> bool:
        """Send an error notification."""
        msg = f"⚠️ AIT ERROR\n{error[:500]}"
        return await self.send(msg)

    async def send_circuit_breaker_alert(self, reason: str) -> bool:
        """Send circuit breaker trigger notification."""
        msg = f"🛑 CIRCUIT BREAKER TRIPPED\nReason: {reason}"
        return await self.send(msg)

    async def send_daily_summary(self, summary: str) -> bool:
        """Send end-of-day summary."""
        msg = f"📊 Daily Summary\n{summary}"
        return await self.send(msg)
