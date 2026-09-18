"""W37 (register time-calendar-03, second half).

W36 moved the outbox drain above the no-new-entries `return`. What remained:
the drain's date rule. A close whose exit_time is not TODAY was claimed,
logged `stale_exit_booking_skipped` and dropped — never booked anywhere. That
is every reconciler/sweep/vanished close found by the 16:00-16:15 post-market
reconcile (the 120s grace means it cannot be drained until the next session),
every close booked while the bot was down overnight, and a Friday close found
on Monday. Its day's daily_stats stayed understated for good, Thompson never
saw the outcome, no ORPHAN CLOSE page fired, and the sizer's
`recent_losing_days` could read one fewer losing day.

The reason for the skip was sound and is kept: stats and the breaker are DAILY
quantities, so a prior-day loss must not enter TODAY's halt math. The fix is to
bucket by session instead of discarding: credit the close's OWN day, leave
today's breaker and PDT count alone, record the outcome for learning, and page.

Real StateManager, real circuit breaker, the real drain.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta

import pytest

from ait.utils.time import now_et
from tests.test_w14_booking_exactly_once import _orch, _seed, _state


def _close_in_a_prior_session(st, trade_id="T-1", pnl=-150.0, days_ago=1):
    st.close_trade(trade_id, 2.5, pnl, exit_reason_detailed="stop_loss (x)")
    when = (now_et() - timedelta(days=days_ago)).replace(hour=16, minute=5)
    with sqlite3.connect(st._db_path) as conn:
        conn.execute("UPDATE exit_outbox SET exit_time = ? WHERE trade_id = ?",
                     (when.isoformat(), trade_id))
    return when.date()


def _outbox_rows(st) -> int:
    with sqlite3.connect(st._db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM exit_outbox").fetchone()[0]


class TestAPriorSessionCloseIsBookedToItsOwnDay:
    def test_it_is_credited_to_the_day_it_closed(self, tmp_path):
        st = _state(tmp_path)
        _seed(st)
        day = _close_in_a_prior_session(st, pnl=-150.0)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())

        then = st.get_daily_stats(day)
        assert then.total_pnl == pytest.approx(-150.0), (
            "a 16:05 reconciler close drained the next morning was claimed and "
            "dropped: its day's ledger is permanently short by the loss")
        assert then.trades_lost == 1
        assert _outbox_rows(st) == 0

    def test_todays_halt_math_is_untouched(self, tmp_path):
        """The reason the old rule skipped these rows — kept."""
        st = _state(tmp_path)
        _seed(st)
        _close_in_a_prior_session(st, pnl=-150.0)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())

        today = st.get_daily_stats()
        assert today.total_pnl == pytest.approx(0.0) and today.trades_lost == 0
        assert o._circuit_breaker._daily_pnl == pytest.approx(0.0)
        assert o._circuit_breaker._consecutive_losses == 0
        o._pdt_guard.record_day_trade.assert_not_called()

    def test_learning_sees_the_outcome_and_the_operator_is_paged(self, tmp_path):
        st = _state(tmp_path)
        _seed(st)
        day = _close_in_a_prior_session(st, pnl=-150.0)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())

        o._thompson.record_outcome.assert_called_once()
        assert o._thompson.record_outcome.call_args.kwargs["won"] is False
        msg = o._send_notification.await_args.args[0]
        assert "ORPHAN CLOSE BOOKED" in msg and day.isoformat() in msg

    def test_it_is_booked_exactly_once(self, tmp_path):
        st = _state(tmp_path)
        _seed(st)
        day = _close_in_a_prior_session(st, pnl=-150.0)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats(day).total_pnl == pytest.approx(-150.0)
        assert o._thompson.record_outcome.call_count == 1

    def test_a_same_day_orphan_still_counts_against_today(self, tmp_path):
        """Control: the today path is unchanged."""
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        with sqlite3.connect(st._db_path) as conn:
            conn.execute("UPDATE exit_outbox SET exit_time = ?",
                         ((now_et() - timedelta(minutes=5)).isoformat(),))
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._consecutive_losses == 1


class TestPostMarketDrainsAfterItsReconcile:
    def test_the_drain_runs_after_the_post_market_reconcile(self, tmp_path, monkeypatch):
        """The 16:00-16:15 reconcile is where manual flattens and assignments
        are discovered; nothing drained after it."""
        from unittest.mock import AsyncMock
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            o = rig.orch
            order: list[str] = []
            real_reconcile = o._reconciler.reconcile

            async def _reconcile():
                order.append("reconcile")
                return await real_reconcile()

            async def _drain():
                order.append("drain")

            o._reconciler.reconcile = _reconcile
            o._drain_exit_outbox = _drain
            o._send_notification = AsyncMock()
            asyncio.run(o._post_market())
            assert "reconcile" in order and "drain" in order, order
            assert order.index("drain") > order.index("reconcile"), order
        finally:
            rig.restore()
