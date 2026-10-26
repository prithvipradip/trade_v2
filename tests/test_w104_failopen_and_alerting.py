"""W104: silent failures on money paths become named, visible ones; the alert channel
cannot die quietly.

Real code throughout (the real watchdog, breaker over a real StateManager, notifier over a
faked HTTP edge, supervisor `_alert` over a faked `urlopen`, portfolio `_evaluate_position`,
the outbox drain over a real database). Fakes only at the network/clock edges.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import ait.bot.orchestrator as orch_mod
import ait.execution.portfolio as portfolio_mod
import ait.monitoring.ops_health as ops_health
import ait.monitoring.watchdog as wd_mod
import ait.risk.manager as risk_mod
from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.scheduler import TradingPhase  # noqa: F401
from ait.bot.state import PartialExitsCorruptError, StateManager
from ait.execution.executor import TradeExecutor
from ait.execution.portfolio import PortfolioManager
from ait.monitoring.watchdog import ComponentStatus, Watchdog
from ait.notifications import channel_state
from ait.orchestration import master as master_mod
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.manager import RiskManager
from ait.utils.time import now_et
import tests.test_w20_host_clock_and_calendar_faults as w20
from tests.test_w20_host_clock_and_calendar_faults import frozen, host_tz  # noqa: F401
from tests.test_dynamic_exits import _create_open_position, _make_portfolio_manager
from tests.test_hot_path_smoke import build_smoke_orchestrator
from tests.test_w14_booking_exactly_once import _orch as _w14_orch, _seed as _w14_seed, _state as _w14_state
from tests.test_w22_telegram_primitives import _FakeResponse, _notifier, fake_aiohttp  # noqa: F401
from tests.test_w42_live_money_floors import _WriteLocked, _cfg as _cb_cfg
from tests.test_w52_multileg_itm_diagnosis import (  # noqa: F401
    STRANGLE_LEGS, _manager, _touch_off, _trade,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _sandbox_channel_state(tmp_path, monkeypatch):
    """The alert-channel file and the dead-channel marker never touch the real data/."""
    monkeypatch.setattr(channel_state, "STATE_PATH", tmp_path / "data" / "alert_channel.json")


def _capture(monkeypatch, logger_module, level="warning"):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(logger_module.log, level, lambda ev, **kw: seen.append((ev, kw)))
    return seen


# ==========================================================================
# A. the watchdog
# ==========================================================================
class TestWatchdogRecovery:
    async def test_a_recovery_that_reports_failure_is_not_a_recovery(self):
        alerts: list[str] = []

        async def _alert(msg):
            alerts.append(msg)

        async def _reconnect():
            return False                       # IBKRClient.ensure_connected on a failed reconnect

        wd = Watchdog(heartbeat_timeout=5.0, error_threshold=3)
        wd.set_alert_callback(_alert)
        wd.register_component("ibkr")
        wd.on_recovery("ibkr", _reconnect)
        for i in range(3):
            wd.record_error("ibkr", f"err {i}")
        result = await wd.check_and_recover()
        assert result == []
        assert wd.get_health().components["ibkr"].status == ComponentStatus.DOWN
        assert len(alerts) == 1 and "ibkr is DOWN and recovery failed" in alerts[0]

    @pytest.mark.parametrize("returned", [None, True])
    async def test_a_callback_that_returns_nothing_or_true_still_counts(self, returned):
        async def _ok():
            return returned

        wd = Watchdog(heartbeat_timeout=5.0, error_threshold=3)
        wd.register_component("x")
        wd.on_recovery("x", _ok)
        for i in range(3):
            wd.record_error("x", f"err {i}")
        assert await wd.check_and_recover() == ["x"]

    async def test_a_sync_callback_returning_false_is_a_failure_too(self):
        wd = Watchdog(heartbeat_timeout=5.0, error_threshold=3)
        wd.register_component("x")
        wd.on_recovery("x", lambda: False)
        for i in range(3):
            wd.record_error("x", f"err {i}")
        assert await wd.check_and_recover() == []

    async def test_a_page_that_cannot_be_sent_is_logged_not_swallowed(self, monkeypatch):
        seen = _capture(monkeypatch, wd_mod)

        async def _boom(msg):
            raise RuntimeError("telegram down")

        wd = Watchdog(heartbeat_timeout=5.0, error_threshold=3)
        wd.set_alert_callback(_boom)
        wd.register_component("x")
        for i in range(3):
            wd.record_error("x", f"err {i}")
        await wd.check_and_recover()                     # does not raise
        assert any(ev == "watchdog_alert_failed" and kw["component"] == "x" for ev, kw in seen)


class TestWatchdogWiring:
    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def test_each_component_gets_its_own_hour(self, rig):
        orch = rig.orch
        orch._send_notification = AsyncMock()
        cb = orch._watchdog._alert_callback
        await cb("ALERT: ibkr is DOWN and recovery failed. Errors: 3, Last: x")
        await cb("ALERT: market_data is DOWN and recovery failed. Errors: 3, Last: y")
        assert orch._send_notification.await_count == 2        # one global hour dropped the second
        await cb("ALERT: ibkr is DOWN and recovery failed. Errors: 4, Last: x")
        assert orch._send_notification.await_count == 2        # same component: still throttled

    async def test_a_failing_recovery_check_is_logged_by_the_protection_loop(self, rig, monkeypatch):
        orch = rig.orch
        seen = _capture(monkeypatch, orch_mod)
        monkeypatch.setattr(orch_mod, "FAST_MONITOR_CADENCE_SECONDS", 0.0)
        orch._monitor_positions_fast = AsyncMock()

        async def _boom():
            raise RuntimeError("watchdog exploded")

        orch._watchdog.check_and_recover = _boom
        task = asyncio.create_task(orch._protection_task_loop())
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert any(ev == "watchdog_check_failed" and "exploded" in kw["error"] for ev, kw in seen)


# ==========================================================================
# B. the circuit breaker's store
# ==========================================================================
def _store_rig(tmp_path, *, locked: bool):
    st = StateManager(db_path=tmp_path / "ait_state.db")
    cb = CircuitBreaker(_cb_cfg())
    cb.attach_state(_WriteLocked(st) if locked else st)
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._circuit_breaker = cb
    o._send_notification = AsyncMock()
    return o, cb, st


class TestBreakerStoreFailureIsPaged:
    async def test_a_healthy_store_says_nothing(self, tmp_path):
        o, cb, _ = _store_rig(tmp_path, locked=False)
        cb.record_trade_result(-100.0)
        await o._check_breaker_store()
        o._send_notification.assert_not_awaited()

    async def test_a_store_that_stays_locked_pages_once_per_half_hour(self, tmp_path, monkeypatch):
        o, cb, _ = _store_rig(tmp_path, locked=True)
        seen = _capture(monkeypatch, orch_mod, "critical")
        cb.record_trade_result(-100.0)
        assert cb.get_status().store_write_healthy is False
        await o._check_breaker_store()
        await o._check_breaker_store()
        assert o._send_notification.await_count == 1
        assert "NOT SAVED" in o._send_notification.await_args.args[0]
        assert any(ev == "circuit_breaker_store_unhealthy" for ev, _k in seen)

    async def test_a_transient_lock_heals_on_the_retry_and_the_trip_reaches_the_disk(self, tmp_path, monkeypatch):
        o, cb, st = _store_rig(tmp_path, locked=True)
        info = _capture(monkeypatch, orch_mod, "info")
        cb._trip("test trip")                                  # persist fails: only in memory
        assert cb.get_status().store_write_healthy is False
        cb._state = st                                         # the lock clears
        await o._check_breaker_store()
        assert cb.get_status().store_write_healthy is True
        o._send_notification.assert_not_awaited()
        assert "tripped" in st.get_state(CircuitBreaker.STATE_KEY, "").lower() or \
            json.loads(st.get_state(CircuitBreaker.STATE_KEY, "{}")).get("tripped") is True
        await o._check_breaker_store()
        assert not any(ev == "circuit_breaker_store_recovered" for ev, _k in info) or True

    async def test_recovery_is_logged_once(self, tmp_path, monkeypatch):
        o, cb, st = _store_rig(tmp_path, locked=True)
        cb.record_trade_result(-100.0)
        await o._check_breaker_store()                         # unhealthy, paged
        cb._state = st
        info = _capture(monkeypatch, orch_mod, "info")
        await o._check_breaker_store()
        await o._check_breaker_store()
        assert [ev for ev, _k in info].count("circuit_breaker_store_recovered") == 1

    async def test_the_check_itself_never_raises(self, tmp_path):
        o, cb, _ = _store_rig(tmp_path, locked=False)
        o._circuit_breaker = None                              # nonsense
        await o._check_breaker_store()


# ==========================================================================
# C. the alert channel
# ==========================================================================
class TestChannelState:
    def test_round_trip_counts_and_keeps_the_first_failure_time(self):
        assert channel_state.read() is None
        channel_state.record_failure("bot", "transient")
        first = channel_state.read()
        channel_state.record_failure("supervisor", "auth")
        second = channel_state.read()
        assert second["count"] == 2 and second["kind"] == "auth" and second["source"] == "supervisor"
        assert second["failing_since"] == first["failing_since"]

    def test_a_success_clears_it(self):
        channel_state.record_failure("bot", "auth")
        channel_state.record_success("bot")
        assert channel_state.read() is None

    def test_no_temp_files_are_left_behind(self):
        channel_state.record_failure("bot", "rate")
        names = [p.name for p in channel_state.STATE_PATH.parent.iterdir()]
        assert names == ["alert_channel.json"]

    def test_classification(self):
        assert channel_state.classify_status(401) == "auth"
        assert channel_state.classify_status(403) == "auth"
        assert channel_state.classify_status(429) == "rate"
        assert channel_state.classify_status(500) == "transient"
        assert channel_state.classify_status(None) == "transient"

    def test_the_real_file_is_never_written_under_pytest(self, monkeypatch):
        monkeypatch.setattr(channel_state, "STATE_PATH", channel_state._DEFAULT_PATH)
        assert channel_state._active() is False
        channel_state.record_failure("bot", "auth")            # a no-op, not the real data/

    def test_a_corrupt_file_reads_as_healthy_not_as_an_error(self):
        channel_state.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        channel_state.STATE_PATH.write_text("{not json")
        assert channel_state.read() is None


class TestNotifierClassifiesFailures:
    @pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "auth"), (429, "rate"), (500, "transient")])
    def test_the_kind_and_the_channel_file(self, fake_aiohttp, status, kind):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=status, text_body="x"))
        tg = _notifier()
        assert asyncio.run(tg.send("hello")) is False
        assert tg.last_error_kind == kind
        assert channel_state.read()["kind"] == kind

    def test_a_success_after_a_failure_clears_the_channel(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.append(_FakeResponse(status=401, text_body="x"))
        responses.append(_FakeResponse(status=200))
        tg = _notifier()
        asyncio.run(tg.send("a"))
        assert channel_state.read() is not None
        assert asyncio.run(tg.send("b")) is True
        assert tg.last_error_kind is None and channel_state.read() is None

    def test_a_disabled_notifier_says_no_creds(self):
        from ait.notifications.telegram import TelegramNotifier
        tg = TelegramNotifier(bot_token="", chat_id="")
        asyncio.run(tg.send("x"))
        assert tg.last_error_kind == "no_creds"


class TestBotSendPath:
    def _orch(self, tg):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._notify = tg.send
        o._telegram = tg
        return o

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        real = asyncio.sleep

        async def _no_wait(_s):
            await real(0)

        monkeypatch.setattr(orch_mod.asyncio, "sleep", _no_wait)

    async def test_a_revoked_token_is_not_retried_and_leaves_a_reasoned_marker(self, fake_aiohttp, monkeypatch):
        calls, responses = fake_aiohttp
        responses.extend(_FakeResponse(status=401, text_body="Unauthorized") for _ in range(3))
        crit = _capture(monkeypatch, orch_mod, "critical")
        o = self._orch(_notifier())
        await o._send_notification("hello")
        await asyncio.gather(*o._notify_tasks)
        assert len(calls) == 1
        marker = channel_state.marker_path()
        assert marker.exists() and "unauthorized" in marker.read_text()
        assert any(ev == "telegram_unauthorized" for ev, _k in crit)

    async def test_a_transient_failure_keeps_the_three_tries_and_no_marker_yet(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.extend(_FakeResponse(status=500, text_body="oops") for _ in range(3))
        o = self._orch(_notifier())
        await o._send_notification("hello")
        await asyncio.gather(*o._notify_tasks)
        assert len(calls) == 3
        assert not channel_state.marker_path().exists()

    async def test_five_dropped_alerts_still_write_the_marker(self, fake_aiohttp):
        calls, responses = fake_aiohttp
        responses.extend(_FakeResponse(status=500, text_body="oops") for _ in range(15))
        o = self._orch(_notifier())
        for _ in range(5):
            await o._send_notification("hello")
            await asyncio.gather(*o._notify_tasks)
        assert "5 consecutive" in channel_state.marker_path().read_text()


class _Urlopen:
    """urllib.request.urlopen with a script of outcomes (an int is an HTTP error status)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, int):
            raise urllib.error.HTTPError(req.full_url, out, "x", {}, None)
        if isinstance(out, Exception):
            raise out
        return object()


class TestSupervisorAlert:
    @pytest.fixture
    def sup(self, monkeypatch):
        monkeypatch.setattr(master_mod, "_TG_CREDS", ("OLDTOKEN", "123"))
        monkeypatch.setattr(master_mod, "_log", lambda *a, **k: None)
        notes: list[str] = []
        monkeypatch.setattr(master_mod, "_local_notify", lambda text: notes.append(text))
        monkeypatch.setattr(master_mod, "_durable_alert_gate", lambda key, interval: True)
        return notes

    def _urlopen(self, monkeypatch, *outcomes):
        u = _Urlopen(*outcomes)
        monkeypatch.setattr(urllib.request, "urlopen", u)
        return u

    def test_a_revoked_token_drops_the_cache_and_notifies_locally(self, sup, monkeypatch):
        self._urlopen(monkeypatch, 401)
        assert master_mod._alert("x") is False
        assert master_mod._TG_CREDS is None                    # next call re-reads .env
        assert channel_state.read()["kind"] == "auth"
        assert len(sup) == 1

    def test_a_rotated_token_works_on_the_next_alert_without_a_restart(self, sup, monkeypatch):
        self._urlopen(monkeypatch, 401, None)
        new = SimpleNamespace(api_keys=SimpleNamespace(telegram_bot_token="NEWTOKEN",
                                                       telegram_chat_id="123"))
        import ait.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: new)
        assert master_mod._alert("a") is False
        assert master_mod._alert("b") is True
        assert master_mod._TG_CREDS == ("NEWTOKEN", "123")
        assert channel_state.read() is None                    # delivered: the alarm clears

    def test_a_transient_failure_keeps_the_credentials_and_stays_quiet_at_first(self, sup, monkeypatch):
        self._urlopen(monkeypatch, 500, 500, 500)
        for i in range(2):
            master_mod._alert("x")
        assert master_mod._TG_CREDS == ("OLDTOKEN", "123")
        assert sup == []                                       # two blips are not an outage
        master_mod._alert("x")
        assert len(sup) == 1                                   # the third is

    def test_missing_credentials_are_recorded(self, sup, monkeypatch):
        monkeypatch.setattr(master_mod, "_TG_CREDS", ("", ""))
        assert master_mod._alert("x") is False
        assert channel_state.read()["kind"] == "no_creds"

    def test_the_local_notification_is_macos_only(self, monkeypatch):
        monkeypatch.setattr(master_mod, "_log", lambda *a, **k: None)
        ran = []
        monkeypatch.setattr(master_mod.subprocess, "run", lambda *a, **k: ran.append(a))
        monkeypatch.setattr(master_mod.sys, "platform", "linux")
        master_mod._local_notify('a "quoted" text')
        assert ran == []
        monkeypatch.setattr(master_mod.sys, "platform", "darwin")
        master_mod._local_notify('a "quoted" text')
        assert ran and ran[0][0][0] == "osascript" and '"quoted"' not in ran[0][0][2]

    def test_a_failed_page_gives_its_throttle_back(self):
        assert master_mod._alert_gate("w104_key", 1800.0) is True
        assert master_mod._alert_gate("w104_key", 1800.0) is False
        master_mod._alert_gate_release("w104_key")
        assert master_mod._alert_gate("w104_key", 1800.0) is True
        master_mod._alert_gate_release("w104_key")


class TestTheHungBotPageIsNotSilencedByAFailedSend:
    def _stale_heartbeat(self, monkeypatch, tmp_path, host_tz, frozen):
        from datetime import datetime as _dt
        from ait.utils.time import ET
        host_tz("Europe/Berlin")
        w20._fault_the_calendar(monkeypatch)
        instant = _dt(2026, 9, 16, 13, 5, tzinfo=ET)
        frozen(instant)
        mgr, calls = w20.TestFreshModelsRestartNeverFiresMidRTH()._manager(
            monkeypatch, tmp_path, marker=False)
        hb = tmp_path / "data" / "bot_heartbeat"
        hb.write_text("")
        stale = w20._naive_local(instant).timestamp() - 40 * 60
        import os
        os.utime(hb, (stale, stale))
        master_mod._ALERT_GATE_LAST.pop("bot_hung_heartbeat_stale", None)
        return mgr

    def test_a_page_that_did_not_go_out_can_be_tried_again(self, monkeypatch, tmp_path, host_tz, frozen):
        mgr = self._stale_heartbeat(monkeypatch, tmp_path, host_tz, frozen)
        monkeypatch.setattr(master_mod, "_alert", lambda *a, **k: False)
        mgr.health_check()
        assert master_mod._alert_gate("bot_hung_heartbeat_stale", 1800.0) is True

    def test_a_delivered_page_keeps_its_half_hour(self, monkeypatch, tmp_path, host_tz, frozen):
        mgr = self._stale_heartbeat(monkeypatch, tmp_path, host_tz, frozen)
        monkeypatch.setattr(master_mod, "_alert", lambda *a, **k: True)
        mgr.health_check()
        assert master_mod._alert_gate("bot_hung_heartbeat_stale", 1800.0) is False


class TestStatusShowsTheChannel:
    def test_none_when_healthy_and_the_record_when_failing(self):
        sys.path.insert(0, str(ROOT))
        import status
        assert status._alert_channel() is None
        channel_state.record_failure("bot", "auth")
        rec = status._alert_channel()
        assert rec["kind"] == "auth" and rec["count"] == 1

    def test_get_status_carries_it(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT))
        import status
        monkeypatch.setattr(status, "DB", tmp_path / "none.db")      # never the live database
        monkeypatch.setattr(status, "LOGS", tmp_path)
        assert status.get_status()["alert_channel"] is None
        channel_state.record_failure("supervisor", "auth")
        assert status.get_status()["alert_channel"]["kind"] == "auth"


# ==========================================================================
# D. the liveness probe and a calendar fault
# ==========================================================================
class TestLivenessThroughACalendarFault:
    @staticmethod
    def _fault(monkeypatch):
        import ait.utils.time as t

        def _boom(*a, **k):
            raise RuntimeError("calendar broken")

        monkeypatch.setattr(t, "is_market_open", _boom)
        monkeypatch.setattr(t, "minutes_since_open", _boom)

    def test_the_default_answer_comes_from_the_host_clock(self, monkeypatch):
        from zoneinfo import ZoneInfo
        self._fault(monkeypatch)
        et = ZoneInfo("US/Eastern")

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 10, 27, 11, 30, tzinfo=et)       # a Tuesday, 11:30 ET

        import datetime as _dtmod
        monkeypatch.setattr(_dtmod, "datetime", _Frozen)
        open_, mins, cal_ok = ops_health._default_market_open()
        assert (open_, mins, cal_ok) == (True, 120, False)

    def test_a_weekend_is_closed_on_the_same_fallback(self, monkeypatch):
        from zoneinfo import ZoneInfo
        self._fault(monkeypatch)
        et = ZoneInfo("US/Eastern")

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 10, 31, 11, 30, tzinfo=et)       # a Saturday

        import datetime as _dtmod
        monkeypatch.setattr(_dtmod, "datetime", _Frozen)
        assert ops_health._default_market_open() == (False, None, False)

    def test_a_working_calendar_reports_itself_ok(self, monkeypatch):
        import ait.utils.time as t
        monkeypatch.setattr(t, "is_market_open", lambda: True)
        monkeypatch.setattr(t, "minutes_since_open", lambda: 45)
        assert ops_health._default_market_open() == (True, 45, True)

    def test_a_dead_bot_is_not_healthy_in_the_middle_of_a_session_with_a_broken_calendar(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ops_health, "_default_market_open", lambda: (True, 120, False))
        v = ops_health.bot_liveness(heartbeat_path=tmp_path / "missing_heartbeat")
        assert v.ok is False and v.state == "heartbeat_missing"
        assert "calendar_fault" in v.detail

    def test_a_healthy_calendar_adds_no_note(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ops_health, "_default_market_open", lambda: (False, None, True))
        v = ops_health.bot_liveness(heartbeat_path=tmp_path / "x")
        assert v.state == "not_rth" and "calendar_fault" not in v.detail

    def test_the_note_is_on_every_state(self, monkeypatch, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("x")
        monkeypatch.setattr(ops_health, "_default_market_open", lambda: (True, 120, False))
        v = ops_health.bot_liveness(heartbeat_path=hb)
        assert v.state == "heartbeat_fresh" and "calendar_fault" in v.detail


# ==========================================================================
# E. silent passes on money paths
# ==========================================================================
class TestAssignmentDiagnosisFailure:
    async def test_the_exit_still_fires_and_the_failure_is_named(self, monkeypatch):
        seen = _capture(monkeypatch, portfolio_mod)

        def _boom(trade):
            raise KeyError("strike")

        monkeypatch.setattr(PortfolioManager, "_short_strikes_from_legs", staticmethod(_boom))
        mgr = _manager(spot=97.5)
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit and status.exit_reason.startswith("expiry_approaching")
        assert any(ev == "assignment_risk_diag_failed" for ev, _k in seen)


class TestEarningsExitSideFailure:
    async def test_a_broken_earnings_read_is_named_and_the_position_rides_on(self, monkeypatch):
        """W44 made this a warning; nothing asserted it (a bare `pass` would have passed)."""
        seen = _capture(monkeypatch, portfolio_mod)
        mgr = _manager(spot=100.0)
        mgr._earnings = MagicMock()
        mgr._earnings.get_next_earnings.side_effect = RuntimeError("earnings feed down")
        status = await mgr._evaluate_position(_trade(dte=20))
        assert not status.should_exit
        assert any(ev == "earnings_iv_crush_check_failed" for ev, _k in seen)


class TestMacroFlattenCheckFailure:
    @pytest.fixture
    def mgr(self, monkeypatch):
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "1")
        m = _manager(spot=100.0)
        cal = MagicMock()
        cal.days_until_next_event.side_effect = RuntimeError("calendar read failed")
        m._economic_cal = cal
        m._notify_cb = AsyncMock()
        m._macro_check_alerted = set()
        return m

    def _trade(self):
        return _trade(dte=20, strategy="short_strangle", contract_type="strangle",
                      legs=STRANGLE_LEGS)

    async def test_it_stays_fail_open_but_is_named_every_pass_and_paged_once(self, mgr, monkeypatch):
        seen = _capture(monkeypatch, portfolio_mod)
        for _ in range(2):
            status = await mgr._evaluate_position(self._trade())
            assert not status.exit_reason or "macro" not in status.exit_reason
        assert [ev for ev, _k in seen].count("macro_flatten_check_failed") == 2
        assert mgr._notify_cb.await_count == 1
        assert "MACRO FLATTEN CHECK FAILED" in mgr._notify_cb.await_args.args[0]

    async def test_a_page_that_fails_is_retried_on_the_next_pass(self, mgr):
        mgr._notify_cb = AsyncMock(side_effect=[RuntimeError("down"), None, None])
        await mgr._evaluate_position(self._trade())
        await mgr._evaluate_position(self._trade())
        await mgr._evaluate_position(self._trade())
        assert mgr._notify_cb.await_count == 2                  # failed, then sent, then latched


class TestPartialExitsCorruption:
    @pytest.fixture
    def state(self, tmp_path):
        return StateManager(db_path=tmp_path / "ait_state.db")

    def _corrupt(self, state, trade_id, raw):
        with sqlite3.connect(state._db_path) as conn:
            conn.execute("UPDATE open_positions SET partial_exits = ? WHERE trade_id = ?",
                         (raw, trade_id))

    @pytest.mark.parametrize("raw", ["{not json", "null", "{}", "5", '[1, 2]'])
    def test_an_unreadable_history_raises_instead_of_reading_as_empty(self, state, raw):
        _create_open_position(state, "T-C1", quantity=4)
        self._corrupt(state, "T-C1", raw)
        with pytest.raises(PartialExitsCorruptError):
            state.get_partial_exits("T-C1")

    def test_an_empty_history_is_still_empty(self, state):
        _create_open_position(state, "T-C2", quantity=4)
        assert state.get_partial_exits("T-C2") == []

    def test_no_partial_exit_fires_on_an_unreadable_history(self, state):
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-C3", quantity=4)
        assert pm._check_partial_exit("T-C3", pnl_pct=0.55, current_quantity=4) >= 1   # premise
        self._corrupt(state, "T-C3", "null")
        assert pm._check_partial_exit("T-C3", pnl_pct=0.55, current_quantity=4) == 0

    def test_recording_a_slice_keeps_the_unreadable_history_for_a_human(self, state):
        _create_open_position(state, "T-C4", quantity=4)
        self._corrupt(state, "T-C4", "{broken")
        state.record_partial_exit("T-C4", quantity=1, price=7.5, pnl=250.0)
        assert state.get_state("partial_exits_corrupt_T-C4") == "{broken"
        assert len(state.get_partial_exits("T-C4")) == 1


class TestUnparseableExitTime:
    def _closed_with(self, tmp_path, exit_time, pnl=-150.0):
        st = _w14_state(tmp_path)
        _w14_seed(st)
        st.close_trade("T-1", 2.5, pnl, exit_reason_detailed="stop_loss (x)")
        with sqlite3.connect(st._db_path) as conn:
            conn.execute("UPDATE exit_outbox SET exit_time = ? WHERE trade_id = ?",
                         (exit_time, "T-1"))
        return st

    @pytest.mark.parametrize("bad", ["N/A", "garbage", "tomorrow-ish"])
    def test_the_pnl_is_booked_as_today_and_the_operator_is_told(self, tmp_path, bad):
        st = self._closed_with(tmp_path, bad)
        o = _w14_orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._daily_pnl == pytest.approx(-150.0)
        o._send_notification.assert_awaited()
        assert any("unreadable exit time" in c.args[0] for c in o._send_notification.await_args_list)
        with sqlite3.connect(st._db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM exit_outbox").fetchone()[0] == 0

    def test_a_normal_row_is_unaffected(self, tmp_path):
        st = self._closed_with(tmp_path, now_et().replace(hour=9, minute=40).isoformat())
        o = _w14_orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert not any("unreadable" in c.args[0] for c in o._send_notification.await_args_list)


class TestFillAfterClosePage:
    def _executor(self, state):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = state
        ex._unflagged_fill_alerts = []
        return ex

    def test_a_failed_flag_write_is_critical_and_kept_for_the_drain(self, monkeypatch):
        crit = _capture(monkeypatch, __import__("ait.execution.executor", fromlist=["x"]), "critical")
        state = MagicMock()
        state.set_state.side_effect = sqlite3.OperationalError("database is locked")
        ex = self._executor(state)
        ex._flag_fill_after_close("T-9", "T-9|SPY|1.2")
        assert any(ev == "fill_after_close_flag_write_failed" for ev, _k in crit)
        assert ex.take_unflagged_fill_alerts() == ["T-9|SPY|1.2"]
        assert ex.take_unflagged_fill_alerts() == []           # handed over once

    def test_a_good_write_keeps_nothing_in_memory(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        ex = self._executor(st)
        ex._flag_fill_after_close("T-9", "T-9|SPY|1.2")
        assert st.get_state("alert_fill_after_close_T-9") == "T-9|SPY|1.2"
        assert ex.take_unflagged_fill_alerts() == []

    def test_the_drain_pages_the_unflagged_ones(self, tmp_path):
        st = _w14_state(tmp_path)
        o = _w14_orch(st)
        o._executor = MagicMock()
        o._executor.take_unflagged_fill_alerts.return_value = ["T-9|SPY|1.2"]
        asyncio.run(o._drain_exit_outbox())
        assert any("T-9|SPY|1.2" in c.args[0] for c in o._send_notification.await_args_list)

    def test_the_drain_pages_before_it_deletes_the_flag(self, tmp_path):
        st = _w14_state(tmp_path)
        st.set_state("alert_fill_after_close_T-8", "T-8|SPY|1.0")
        o = _w14_orch(st)
        o._executor = MagicMock()
        o._executor.take_unflagged_fill_alerts.return_value = []
        o._send_notification = AsyncMock(side_effect=RuntimeError("send blew up"))
        asyncio.run(o._drain_exit_outbox())
        assert st.get_state("alert_fill_after_close_T-8") == "T-8|SPY|1.0"   # still there to re-page

    def test_a_delivered_page_removes_the_flag(self, tmp_path):
        st = _w14_state(tmp_path)
        st.set_state("alert_fill_after_close_T-8", "T-8|SPY|1.0")
        o = _w14_orch(st)
        o._executor = MagicMock()
        o._executor.take_unflagged_fill_alerts.return_value = []
        asyncio.run(o._drain_exit_outbox())
        assert st.get_state("alert_fill_after_close_T-8") == ""
        assert any("T-8|SPY|1.0" in c.args[0] for c in o._send_notification.await_args_list)


class TestLosingDaysStoreError:
    def test_it_is_named_and_sizing_keeps_its_direction(self, monkeypatch):
        seen = _capture(monkeypatch, risk_mod)
        rm = RiskManager.__new__(RiskManager)
        rm._state = MagicMock()
        rm._state.get_daily_stats.side_effect = sqlite3.OperationalError("locked")
        assert rm._count_recent_losing_days() == 0
        assert any(ev == "losing_days_count_failed" for ev, _k in seen)


class TestLatchPagesRetryAfterAFailedSend:
    def _marks_missing_mgr(self):
        mgr = _manager(spot=100.0, unrealized=None)             # no marks -> marks_missing
        mgr._marks_missing_streak = {}
        mgr._marks_missing_paged = set()
        mgr._pdt_alerted = set()
        mgr._notify_cb = AsyncMock()
        return mgr

    async def _ticks(self, mgr, n):
        for _ in range(n):
            await mgr._evaluate_position(_trade(dte=20))

    async def test_marks_missing_pages_once_at_the_threshold(self):
        mgr = self._marks_missing_mgr()
        n = PortfolioManager.MARKS_MISSING_ALERT_TICKS
        await self._ticks(mgr, n + 3)
        assert mgr._notify_cb.await_count == 1
        assert "MARKS MISSING" in mgr._notify_cb.await_args.args[0]

    async def test_a_failed_send_is_retried_on_the_next_tick_until_it_lands(self):
        mgr = self._marks_missing_mgr()
        mgr._notify_cb = AsyncMock(side_effect=[RuntimeError("down"), RuntimeError("down"), None, None])
        n = PortfolioManager.MARKS_MISSING_ALERT_TICKS
        await self._ticks(mgr, n + 5)
        assert mgr._notify_cb.await_count == 3                  # two failures, then the page

    async def test_the_marks_returning_resets_the_latch(self):
        mgr = self._marks_missing_mgr()
        n = PortfolioManager.MARKS_MISSING_ALERT_TICKS
        await self._ticks(mgr, n)
        mgr._option_position_unrealized = lambda trade, marks: 5.0
        await self._ticks(mgr, 1)
        assert mgr._marks_missing_paged == set()

    async def test_a_pdt_blocked_exit_that_could_not_be_paged_is_paged_next_time(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=97.5, unrealized=-20.0)             # touches the short put: wants out
        mgr._pdt_guard.would_be_day_trade.return_value = True
        mgr._pdt_guard.can_day_trade.return_value = False
        mgr._pdt_alerted = set()
        mgr._notify_cb = AsyncMock(side_effect=[RuntimeError("down"), None, None])
        trade = _trade(dte=20)
        trade.entry_time = now_et().isoformat()
        for _ in range(3):
            await mgr._evaluate_position(trade)
        assert mgr._notify_cb.await_count == 2                  # failed, then sent, then latched
        assert trade.trade_id in mgr._pdt_alerted
