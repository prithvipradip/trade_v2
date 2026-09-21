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
firing every two minutes for ten hours.

W40 CORRECTION (R26 `w18-throttle-tests-getsource-and-three-of-four-sites`).
This docstring used to end "These tests pin all four." It was not true. The
class EXECUTED one site (health_check), pinned two by `inspect.getsource` —
hollow when written, and made a permanent no-op by W34, which moved those two
pages onto their own `_last_gateway_escalation` latch so the asserted string
vanished from the method entirely — and left the fourth, in `BotManager.start()`,
with no assertion coverage anywhere in the suite. R26 measured that: reverting
start() to consume-on-drop left all 308 supervisor tests green.
All four are now driven through the real methods, and the start() site's
mutation goes red.
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
    m._last_gateway_escalation = None   # W34: the escalation's own throttle
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

    def test_the_budget_exhausted_escalation_retries_after_a_dropped_page(self):
        """W40 (R26 `w18-throttle-tests-getsource-and-three-of-four-sites`).

        This replaces an `inspect.getsource` pin that counted `"if _alert("`
        occurrences in `_maybe_recover_gateway`. It was hollow when written —
        R26 added `or True` to both gates and the class still reported 3 passed
        — and W34 then made it a PERMANENT no-op by moving these pages onto
        their own `_last_gateway_escalation` latch, so the method no longer
        contains the string `_last_gateway_alert` at all and the second
        assertion became unconditionally true. A source-string assertion proves
        code was WRITTEN, never that it RUNS, and decays into silence on
        rename; this drives the real method instead.
        """
        m = _mgr_down()
        m._gateway_down_since = datetime.now() - timedelta(hours=3)
        m._gateway_recoveries = M.GATEWAY_MAX_RECOVERIES   # budget spent
        now = datetime.now()

        with patch.object(M, "_alert", return_value=False) as alert:
            m._maybe_recover_gateway(4002, now)
            assert m._last_gateway_escalation is None, (
                "a dropped MANUAL ACTION page burned the 30-minute throttle — "
                "the operator is told nothing for half an hour about an outage "
                "automation has already given up on"
            )
            m._maybe_recover_gateway(4002, now + timedelta(seconds=1))
        assert alert.call_count == 2, "the next cycle must retry a page that never landed"

    def test_the_budget_exhausted_escalation_throttles_a_delivered_page(self):
        """CONTROL: a page that DID land must still be throttled."""
        m = _mgr_down()
        m._gateway_down_since = datetime.now() - timedelta(hours=3)
        m._gateway_recoveries = M.GATEWAY_MAX_RECOVERIES
        now = datetime.now()

        with patch.object(M, "_alert", return_value=True) as alert:
            m._maybe_recover_gateway(4002, now)
            assert m._last_gateway_escalation is not None
            m._maybe_recover_gateway(4002, now + timedelta(seconds=1))
        assert alert.call_count == 1, "a delivered page must still be throttled"

    def test_the_start_site_retries_after_a_dropped_page(self):
        """THE FOURTH SITE — which had no assertion coverage anywhere.

        R26 measured it: reverting `BotManager.start()`'s throttle to
        consume-on-drop left all 308 supervisor tests green, and the whole
        suite unchanged at 8 failed / 1841 passed. The class this file is named
        for ("covers all four sites") executed ONE.

        Trap, and the reason this is not a copy of the health_check tests:
        `ensure_gateway` is imported INSIDE start(), so patching
        `M.ensure_gateway` silently does nothing and the real launcher runs.
        """
        m = _mgr_down()
        now_before = datetime.now()

        with patch("ait.orchestration.gateway.ensure_gateway", return_value=False), \
             patch.object(M, "_scan_foreign_bot_pids", return_value=[]), \
             patch.object(M, "_port_hint", return_value=""), \
             patch.object(M, "_alert", return_value=False) as alert:
            m.start()
            assert m._last_gateway_alert is None, (
                "a dropped 'cannot start bot — Gateway not available' page "
                "burned the throttle, so the next attempt stays silent"
            )
            m.start()

        assert alert.call_count == 2, "the next attempt must retry a dropped page"
        assert m._proc is None, "premise moved: start() launched a bot without a Gateway"
        assert now_before <= datetime.now()

    def test_the_start_site_throttles_a_delivered_page(self):
        """CONTROL: the same site must still throttle a page that landed."""
        m = _mgr_down()
        with patch("ait.orchestration.gateway.ensure_gateway", return_value=False), \
             patch.object(M, "_scan_foreign_bot_pids", return_value=[]), \
             patch.object(M, "_port_hint", return_value=""), \
             patch.object(M, "_alert", return_value=True) as alert:
            m.start()
            assert m._last_gateway_alert is not None
            m.start()
        assert alert.call_count == 1, "a delivered page must still be throttled"


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

    def test_each_module_resolves_the_base_from_the_authority(self, monkeypatch):
        """W40 (R26 `w18-throttle-tests-getsource-and-three-of-four-sites`).

        This replaces a `read_text()` regex for the literal `196000` across
        three files. That check could not observe whether any module EXECUTES
        a fallback, and a reintroduced one spelled `196_000`, `1.96e5` or via
        a named constant passed it cleanly.

        Executed instead: move the authority and require every module's
        resolved base to move with it. A module holding its own literal
        diverges here whatever its spelling. The sibling below pins identity;
        this pins the VALUE actually returned, which is what the annualised
        return is computed from.
        """
        from ait.config import runtime_env as RE
        from ait.monitoring import analytics as AN, duckdb_analytics as DK

        monkeypatch.setenv("AIT_CAPITAL_BASE", "424242")
        RE._capital_base_cache = None  # if one exists; harmless otherwise

        expected = RE.capital_base()
        assert expected == 424242, (
            f"premise moved: the authority no longer follows AIT_CAPITAL_BASE "
            f"(got {expected})"
        )
        for name, mod in (("analytics", AN), ("duckdb_analytics", DK)):
            assert mod._capital_base() == expected, (
                f"{name} resolved {mod._capital_base()} while the authority "
                f"says {expected} — it is carrying its own base"
            )

    def test_all_three_resolve_through_the_single_authority(self):
        from ait.config.runtime_env import capital_base
        from ait.monitoring import analytics as AN, duckdb_analytics as DK
        assert AN._capital_base is capital_base
        assert DK._capital_base is capital_base

    def test_dashboard_reads_a_key_that_is_actually_written(self, tmp_path, monkeypatch):
        """app.py queried bot_state['account_value'], which is never written
        anywhere in src/. The live DB holds only 'last_net_liquidation'.

        W40: this substring-matched `key = 'last_net_liquidation'` in app.py
        and `set_state("last_net_liquidation"` in orchestrator.py. It ALREADY
        broke once for no behavioural reason — W29 reformatted the `set_state`
        call across three lines, behaviour identical, and the test went red
        (PLAN.md:5073-5084 registered it for this sweep). Worse, it had pushed
        a "KEEP THIS CALL ON ONE LINE" comment into production code: the test
        constraining the source it was meant to observe.

        The INTENT — the dashboard's key must have a writer — is unchanged, and
        is now proved as the round trip it always was: drive a REAL trading
        cycle, then read it back with the dashboard's OWN query. A rename on
        either side breaks this; a reformat cannot.
        """
        import asyncio
        import sqlite3

        from tests.test_hot_path_smoke import build_smoke_orchestrator
        from ait.dashboard.app import _NLV_QUERY

        monkeypatch.chdir(tmp_path)          # the rig's caller contract
        rig = build_smoke_orchestrator()
        try:
            asyncio.run(rig.orch._trading_cycle())
            db_path = rig.orch._state._db_path
        finally:
            rig.restore()

        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(_NLV_QUERY).fetchone()
        finally:
            con.close()

        assert row is not None, (
            "the dashboard's account-value query found nothing after a real "
            "trading cycle — the key it reads has no writer again"
        )
        assert float(row[0]) > 0, f"the key was written but holds {row[0]!r}"
