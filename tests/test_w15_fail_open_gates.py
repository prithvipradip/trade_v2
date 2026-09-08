"""W15 (R24): two protective gates that failed OPEN.

time-calendar-01 — the pre-event blackout converted calendar days to sessions
by counting WEEKDAYS, so every NYSE holiday counted as a trading session. The
consumer blocks when ``sessions <= blackout``, so an INFLATED count means
FEWER blocks: the gate failed open across every holiday, in exactly the
direction its own docstring claimed was fail-safe ("erring toward MORE
sessions is the fail-safe direction" — backwards).

regress-live-07 / time-calendar-02 — the post-stop cooldown's caller swallowed
every exception with a bare `pass`. W4 had just given that gate a NYSE-calendar
dependency and a fresh sqlite connection, so a calendar fault or a busy DB
silently DISABLED the re-entry guard for every symbol on every scan, with no
log line and no page — the bot could re-enter the symbol it had just been
stopped out of (the 2026-08-24 QQQ shape).

Every test EXECUTES the real code with the clock pinned.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ait.bot.orchestrator as M
from ait.bot.orchestrator import TradingOrchestrator
from ait.utils.time import is_trading_day


def _pin(y, mo, d, h=10):
    """Pin orchestrator's `datetime.now()` without touching the calendar."""
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(y, mo, d, h, 0)
    return patch.object(M, "datetime", _DT)


class TestBlackoutCountsRealSessions:
    def test_labor_day_is_not_a_trading_session(self):
        """The premise: the fix depends on the calendar knowing this."""
        assert is_trading_day(date(2026, 9, 7)) is False
        assert is_trading_day(date(2026, 9, 8)) is True

    def test_holiday_no_longer_inflates_the_count(self):
        """Fri 09-04 -> Wed 09-09 with Labor Day between. Only ONE real
        session (Tue) separates entry from the event; the weekday count said
        2, so a blackout of 1 did not block."""
        with _pin(2026, 9, 4):
            assert TradingOrchestrator._sessions_until(5) == 1

    def test_the_gate_now_blocks_that_entry(self):
        blackout = 1
        with _pin(2026, 9, 4):
            sessions = TradingOrchestrator._sessions_until(5)
        assert sessions <= blackout, "credit entry would still slip through"

    @pytest.mark.parametrize("cal_days,expected", [
        (4, 0),   # Fri -> Tue 09-08: no session between
        (5, 1),   # Fri -> Wed 09-09: Tue only
        (6, 2),   # Fri -> Thu 09-10: Tue, Wed
    ])
    def test_holiday_week_matrix(self, cal_days, expected):
        with _pin(2026, 9, 4):
            assert TradingOrchestrator._sessions_until(cal_days) == expected

    def test_plain_weekend_still_reads_zero(self):
        """The R16 behaviour this gate was built for must be preserved:
        Friday entry, Monday event, no session in between."""
        with _pin(2026, 9, 11):            # a Friday, no holiday ahead
            assert TradingOrchestrator._sessions_until(3) == 0

    def test_none_passes_through(self):
        assert TradingOrchestrator._sessions_until(None) is None

    def test_calendar_fault_degrades_to_weekdays_without_crashing(self):
        with _pin(2026, 9, 4), \
             patch("ait.utils.time.is_trading_day", side_effect=RuntimeError("pmc down")):
            out = TradingOrchestrator._sessions_until(5)
        assert isinstance(out, int) and out >= 0


class TestCooldownFailsClosed:
    """The real helper, executed — no source-string assertions."""

    def _orch(self, side_effect=None, returns=None):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._post_stop_cooldown_until = MagicMock(
            side_effect=side_effect, return_value=returns)
        o._send_notification = AsyncMock()
        o._alert_gate = MagicMock(return_value=True)
        return o

    def _sig(self, strategy="iron_condor"):
        return SimpleNamespace(symbol="QQQ", strategy_name=strategy)

    def test_locked_db_blocks_a_credit_entry(self):
        """Pre-W15 a bare `pass` let the entry through with no trace."""
        import sqlite3
        o = self._orch(side_effect=sqlite3.OperationalError("database is locked"))
        assert asyncio.run(o._cooldown_blocks_entry(self._sig())) is True

    def test_calendar_fault_blocks_too(self):
        o = self._orch(side_effect=RuntimeError("pmc schedule failed"))
        assert asyncio.run(o._cooldown_blocks_entry(self._sig())) is True

    def test_the_operator_is_paged(self):
        o = self._orch(side_effect=RuntimeError("pmc down"))
        asyncio.run(o._cooldown_blocks_entry(self._sig()))
        o._send_notification.assert_awaited()
        assert "COOLDOWN CHECK FAILED" in o._send_notification.await_args[0][0]

    def test_a_failing_pager_does_not_unblock_the_gate(self):
        o = self._orch(side_effect=RuntimeError("pmc down"))
        o._send_notification = AsyncMock(side_effect=RuntimeError("telegram down"))
        assert asyncio.run(o._cooldown_blocks_entry(self._sig())) is True

    def test_debit_shapes_keep_the_permissive_behaviour(self):
        """The rule exists for the autocorrelated-loss case; a debit shape is
        not it, so a broken check should not halt those entries."""
        o = self._orch(side_effect=RuntimeError("pmc down"))
        assert asyncio.run(
            o._cooldown_blocks_entry(self._sig("long_call"))) is False

    def test_an_active_cooldown_still_blocks(self):
        o = self._orch(returns="2026-09-04T10:00:00")
        assert asyncio.run(o._cooldown_blocks_entry(self._sig())) is True

    def test_a_clear_cooldown_allows_the_entry(self):
        o = self._orch(returns=None)
        assert asyncio.run(o._cooldown_blocks_entry(self._sig())) is False
        o._send_notification.assert_not_awaited()
