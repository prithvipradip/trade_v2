"""W18: a stale IBKR_PORT must not cost another silent outage — and the
throttle-ordering fix must actually cover the CLASS this time.

CONTEXT. The operator moved from TWS to IB Gateway. `.env` still said
IBKR_PORT=7497 while Gateway listened on 4002, so the supervisor deferred the
bot for 9h47m behind a bare `bot_restart_deferred_gateway_down | port=7497`.
The mismatch was measurable the whole time; nothing looked.

AND A CORRECTION. W17's w17-01b fixed the "consume the throttle only on a
DELIVERED page" ordering and its commit claimed the class. There are FOUR
sites sharing `_last_gateway_alert`; it fixed two (both inside
_maybe_recover_gateway) and missed two — including the one that had been
firing every two minutes for ten hours. These tests pin all four.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import ait.orchestration.master as M


class TestPortHint:
    def test_names_the_sibling_port_that_is_actually_open(self):
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002):
            hint = M._port_hint(7497)
        assert "4002" in hint and "IB Gateway paper" in hint
        assert "IBKR_PORT" in hint, "the hint must name the knob to turn"

    def test_silent_when_the_configured_port_is_the_open_one(self):
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002):
            assert M._port_hint(4002) == ""

    def test_silent_when_nothing_is_listening_anywhere(self):
        with patch.object(M, "_gateway_listening", return_value=False):
            assert M._port_hint(7497) == ""

    def test_a_hint_must_never_break_the_alert_it_decorates(self):
        with patch.object(M, "_gateway_listening", side_effect=OSError("socket blew up")):
            assert M._port_hint(7497) == ""


def _mgr_down(*, last_alert=None):
    """A BotManager whose bot is NOT running, ready to hit the gateway check."""
    m = M.BotManager.__new__(M.BotManager)
    m._proc = None
    m._restarts = 0
    m._last_restart = None
    m._gateway_down_since = None
    m._last_gateway_alert = last_alert
    m._gateway_recoveries = 0
    m._last_gateway_recovery = None
    m._log_handle = None
    return m


class TestThrottleOrderingCoversAllFourSites:
    """A dropped page must not consume the 30-minute throttle — at EVERY site."""

    def test_health_check_deferral_site_retries_after_a_dropped_page(self):
        """master.py ~:771 — the site that fired for ten hours. W17 missed it."""
        m = _mgr_down()
        with patch.object(M, "_gateway_listening", return_value=False), \
             patch.object(M, "_port_hint", return_value=""), \
             patch.object(M, "_alert", return_value=False) as alert, \
             patch.object(M, "_in_post_market_window", return_value=False), \
             patch.object(M.Path, "exists", return_value=False):
            m.health_check()
            assert m._last_gateway_alert is None, "a dropped page must not burn the throttle"
            m.health_check()
        assert alert.call_count >= 2, "the next cycle must retry a page that never landed"

    def test_health_check_deferral_site_throttles_a_delivered_page(self):
        m = _mgr_down()
        with patch.object(M, "_gateway_listening", return_value=False), \
             patch.object(M, "_port_hint", return_value=""), \
             patch.object(M, "_alert", return_value=True) as alert, \
             patch.object(M, "_in_post_market_window", return_value=False), \
             patch.object(M.Path, "exists", return_value=False):
            m.health_check()
            assert m._last_gateway_alert is not None
            m.health_check()
        assert alert.call_count == 1, "a delivered page must still be throttled"

    def test_the_two_recovery_sites_are_still_gated(self):
        """W17 fixed these; pin them so a later edit cannot regress the class."""
        import inspect
        src = inspect.getsource(M.BotManager._maybe_recover_gateway)
        assert src.count("if _alert(") == 2, "both recovery pages must gate on delivery"
        assert "\n                self._last_gateway_alert = now\n                _alert(" not in src


class TestReadOnlyAlarmDoesNotFailOpen:
    """w17-08. verify_can_trade answered True after FAILING to place its probe.

    It exists to catch a READ-ONLY Gateway session — orders rejected with Error
    321 while the bot still logs trading_verified. Gateway is the app that
    actually has that failure, and it recurs on Gateway's daily restart.
    Answering True on an error restores the exact silence the probe prevents.

    Note the return value stays unused: both callers of _check_trading_enabled
    (:513, :900) discard it, and that is correct — the function's job is the
    ALERT, not a gate. W17 batch 2 corrected the register on that point."""

    def _client(self, side_effect):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from ait.broker.ibkr_client import IBKRClient
        c = IBKRClient.__new__(IBKRClient)
        c._connected = True
        ib = MagicMock()
        ib.qualifyContractsAsync = AsyncMock(side_effect=side_effect)
        ib.placeOrder = MagicMock()
        c._ib = ib
        return c, ib, asyncio

    def test_probe_failure_is_UNKNOWN_not_yes(self):
        for exc in (ConnectionError("dropped"), TimeoutError("no response"),
                    RuntimeError("boom")):
            c, ib, asyncio = self._client(exc)
            got = asyncio.run(c.verify_can_trade())
            assert got is None, f"{type(exc).__name__} must be UNKNOWN, not True"
            assert not ib.placeOrder.called, "nothing may reach the broker"

    def test_unqualifiable_contract_is_also_UNKNOWN(self):
        c, ib, asyncio = self._client(None)
        ib.qualifyContractsAsync.side_effect = None
        ib.qualifyContractsAsync.return_value = []
        assert asyncio.run(c.verify_can_trade()) is None

    def test_orchestrator_pages_on_UNKNOWN(self):
        """The whole point: UNKNOWN must reach a human, not pass silently."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from ait.bot.orchestrator import TradingOrchestrator
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._ibkr = MagicMock()
        o._ibkr.verify_can_trade = AsyncMock(return_value=None)
        o._send_notification = AsyncMock()
        o._alert_gate = lambda *a, **k: True
        asyncio.run(o._check_trading_enabled())
        o._send_notification.assert_awaited_once()
        assert "CANNOT VERIFY" in o._send_notification.await_args[0][0]


class TestCapitalBaseHasNoSilentFallback:
    """w17-11. Three copies each wrapped the declared authority in a bare
    unlogged `except` and fell back to a SECOND hardcoded 196000 — the exact
    pre-R16 bug the authority was created to fix. W17 batch 2 measured it:
    196,000 against a live 723,071, silently, then raised to the 12th power for
    the annualised return."""

    def test_no_module_carries_its_own_capital_literal(self):
        import pathlib, re
        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in ("src/ait/dashboard/app.py",
                    "src/ait/monitoring/analytics.py",
                    "src/ait/monitoring/duckdb_analytics.py"):
            src = (root / rel).read_text(encoding="utf-8")
            code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
            assert not re.search(r'["\']196000["\']|\b196000\.0\b', code), (
                f"{rel} still carries its own capital-base literal")

    def test_all_three_resolve_through_the_single_authority(self):
        from ait.config.runtime_env import capital_base
        from ait.monitoring import analytics as AN, duckdb_analytics as DK
        assert AN._capital_base is capital_base
        assert DK._capital_base is capital_base

    def test_dashboard_reads_a_key_that_is_actually_written(self):
        """app.py queried bot_state['account_value'], which is never written
        anywhere in src/. The live DB holds only 'last_net_liquidation'."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        app = (root / "src/ait/dashboard/app.py").read_text(encoding="utf-8")
        assert "key = 'last_net_liquidation'" in app
        writers = (root / "src/ait/bot/orchestrator.py").read_text(encoding="utf-8")
        assert 'set_state("last_net_liquidation"' in writers, "the key must have a writer"
