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
            payload = {
                "chat_id": self._chat_id,
                "text": message[:4096],
                "parse_mode": "Markdown",
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
            log.warning("telegram_error", error=str(e))
            return False

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
            f"{emoji} *{action}* {contracts}x {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Price: ${price:.2f}\n"
            f"Confidence: {confidence:.0%}"
        )
        return await self.send(msg)

    async def send_error_alert(self, error: str) -> bool:
        """Send an error notification."""
        msg = f"⚠️ *AIT ERROR*\n```\n{error[:500]}\n```"
        return await self.send(msg)

    async def send_circuit_breaker_alert(self, reason: str) -> bool:
        """Send circuit breaker trigger notification."""
        msg = f"🛑 *CIRCUIT BREAKER TRIPPED*\nReason: {reason}"
        return await self.send(msg)

    async def send_daily_summary(self, summary: str) -> bool:
        """Send end-of-day summary."""
        msg = f"📊 *Daily Summary*\n{summary}"
        return await self.send(msg)
