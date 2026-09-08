"""W14 (R24 booking-integrity cluster): the circuit breaker must see every
real loss exactly once, and nothing else.

Five findings, one design problem — the claim happened BEFORE the booking
rather than atomically with it, and nothing filtered what got booked:

  concurrency-01  the claim was a committed DELETE performed before any
                  booking write, so a crash in that window destroyed the only
                  record of the obligation while nothing had been booked.
  concurrency-02  a claim that RAISED set _claimed = True and booked anyway,
                  but the row survived — so the drain booked the same close a
                  second time (one -$150 stop read as -$300 to the breaker).
  regress-live-01 the reconciler's $0 bookkeeping rows booked as LOSSES, and
                  record_trade_result(0.0) RESETS an active loss streak: two
  pnl-money-01    real losses plus one swept never-filled entry meant the
                  third real loss no longer tripped the 3-loss pause.
  regress-live-04 the fail-closed breaker auto-resumed and then OVERWROTE the
                  still-intact persisted blob with zeros.
  concurrency-03  a CAS refusal was always read as "terminal row", discarding
                  the real fill price of a legitimately concurrent promotion.

Every test EXECUTES the real StateManager / CircuitBreaker / executor.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.executor import TradeExecutor
from ait.risk.circuit_breaker import CircuitBreaker


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed(st, trade_id="T-1", status="filled"):
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, "QQQ", "iron_condor", "short", status,
             datetime.now().isoformat(), 1.90))


def _backdate(st, trade_id, minutes):
    with sqlite3.connect(st._db_path) as conn:
        conn.execute("UPDATE exit_outbox SET exit_time = ? WHERE trade_id = ?",
                     ((datetime.now() - timedelta(minutes=minutes)).isoformat(),
                      trade_id))


def _cfg():
    return SimpleNamespace(max_daily_loss_pct=0.02, max_consecutive_losses=3,
                           pause_minutes_after_losses=30, max_api_failures=5)


def _orch(st):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._circuit_breaker = CircuitBreaker(_cfg())
    o._pdt_guard = MagicMock()
    o._thompson = MagicMock()
    o._trainer = MagicMock()
    o._send_notification = AsyncMock()
    o._find_trade_by_id = lambda tid: SimpleNamespace(
        trade_id=tid, symbol="QQQ", strategy="iron_condor",
        contract_type="iron_condor", quantity=1,
        entry_time=datetime.now().isoformat(), legs="[]")
    return o


# ------------------------------------------------- concurrency-01
class TestObligationSurvivesACrash:
    def test_claim_marks_rather_than_deletes(self, tmp_path):
        """The row must still exist after a claim — that is what makes the
        work recoverable if the booker dies before writing."""
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        assert st.claim_exit_booking("T-1") is True
        with sqlite3.connect(st._db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM exit_outbox "
                             "WHERE trade_id='T-1'").fetchone()[0]
        assert n == 1, "claim destroyed the obligation (the pre-W14 defect)"

    def test_completion_retires_it(self, tmp_path):
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        st.claim_exit_booking("T-1")
        st.complete_exit_booking("T-1")
        assert st.pending_exit_bookings(older_than_seconds=-1) == []

    def test_a_stale_claim_is_re_offered(self, tmp_path):
        """Booker claimed, then died. The obligation must come back."""
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        st.claim_exit_booking("T-1")
        _backdate(st, "T-1", minutes=10)
        with sqlite3.connect(st._db_path) as conn:  # simulate a dead claimer
            conn.execute(
                "UPDATE exit_outbox SET claimed_at = ? WHERE trade_id='T-1'",
                ((datetime.now() - timedelta(seconds=st.CLAIM_STALE_SECONDS + 60))
                 .isoformat(),))
        assert [r["trade_id"] for r in
                st.pending_exit_bookings(older_than_seconds=120.0)] == ["T-1"]

    def test_a_fresh_claim_is_not_re_offered(self, tmp_path):
        """A slow-but-live booker must never be overtaken."""
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        _backdate(st, "T-1", minutes=10)
        st.claim_exit_booking("T-1")
        assert st.pending_exit_bookings(older_than_seconds=120.0) == []

    def test_crash_between_claim_and_booking_is_recovered(self, tmp_path):
        """End-to-end: the loss reaches the breaker on the NEXT drain."""
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss (x)")
        st.claim_exit_booking("T-1")          # booker claims...
        # ...and dies here. Its claim goes stale:
        with sqlite3.connect(st._db_path) as conn:
            conn.execute(
                "UPDATE exit_outbox SET claimed_at = ?, exit_time = ? "
                "WHERE trade_id='T-1'",
                ((datetime.now() - timedelta(seconds=st.CLAIM_STALE_SECONDS + 60)).isoformat(),
                 (datetime.now() - timedelta(minutes=10)).isoformat()))
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert o._circuit_breaker._daily_pnl == pytest.approx(-150.0)


# ------------------------------------------------- regress-live-01 / pnl-money-01
class TestPhantomRowsAreNotTrades:
    @pytest.mark.parametrize("reason", [
        "stale_pending_never_filled",
        "reconciler_unknown_exit",
        "reconciler_unknown_exit_needs_review",
    ])
    def test_zero_dollar_sentinels_do_not_reset_the_loss_streak(
            self, tmp_path, reason):
        """THE DEFECT: record_trade_result(0.0) resets consecutive_losses, so
        a swept never-filled entry wiped a real 2-loss streak and the third
        real loss no longer tripped the 3-loss pause."""
        st = _state(tmp_path)
        _seed(st, "T-NF", status="pending")
        st.close_trade("T-NF", 0.0, 0.0, exit_reason_detailed=reason)
        _backdate(st, "T-NF", minutes=10)
        o = _orch(st)
        o._circuit_breaker._consecutive_losses = 2      # two REAL losses
        asyncio.run(o._drain_exit_outbox())
        assert o._circuit_breaker._consecutive_losses == 2, "streak was reset"
        assert st.get_daily_stats().trades_lost == 0
        o._thompson.record_outcome.assert_not_called()
        o._send_notification.assert_not_awaited()

    def test_sentinel_is_retired_not_left_to_retry_forever(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, "T-NF", status="pending")
        st.close_trade("T-NF", 0.0, 0.0,
                       exit_reason_detailed="stale_pending_never_filled")
        _backdate(st, "T-NF", minutes=10)
        asyncio.run(_orch(st)._drain_exit_outbox())
        assert st.pending_exit_bookings(older_than_seconds=-1) == []

    def test_a_real_loss_still_books(self, tmp_path):
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 8.76, -272.29,
                       exit_reason_detailed="short_strike_touch (x)")
        _backdate(st, "T-1", minutes=10)
        o = _orch(st)
        o._circuit_breaker._consecutive_losses = 2
        asyncio.run(o._drain_exit_outbox())
        assert o._circuit_breaker._consecutive_losses == 3
        assert st.get_daily_stats().total_pnl == pytest.approx(-272.29)


# ------------------------------------------------- regress-live-04
class TestBreakerNeverOverwritesAnIntactBlob:
    def test_unreadable_state_refuses_to_persist(self):
        b = CircuitBreaker(_cfg())
        bad = MagicMock()
        bad.get_state.side_effect = sqlite3.OperationalError("database is locked")
        b.attach_state(bad)
        assert b._tripped and b._state_unreadable
        bad.set_state.reset_mock()
        b._persist()
        bad.set_state.assert_not_called(), "would have zeroed a real streak"

    def test_a_good_read_restores_authority(self, tmp_path):
        b = CircuitBreaker(_cfg())
        b.attach_state(_state(tmp_path))
        assert b._state_unreadable is False
        b._persist()          # must not raise, must write

    def test_the_intact_streak_survives_a_locked_relaunch(self, tmp_path):
        """A real 2-loss streak must still be there for the NEXT process."""
        st = _state(tmp_path)
        good = CircuitBreaker(_cfg())
        good.attach_state(st)
        good.record_trade_result(-100.0)
        good.record_trade_result(-100.0)
        assert good._consecutive_losses == 2

        blocked = MagicMock()
        blocked.get_state.side_effect = sqlite3.OperationalError("locked")
        crashed = CircuitBreaker(_cfg())
        crashed.attach_state(blocked)
        crashed._persist()                     # pre-W14: wrote zeros

        recovered = CircuitBreaker(_cfg())
        recovered.attach_state(st)
        assert recovered._consecutive_losses == 2, "the streak was destroyed"


# ------------------------------------------------- concurrency-03
class TestCasRefusalDistinguishesPromotionFromTerminal:
    def _rig(self, st):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = st
        return ex, SimpleNamespace(
            trade_id="T-1", contracts=1,
            signal=SimpleNamespace(symbol="QQQ", legs=[], contract=None,
                                   strategy_name="iron_condor"))

    def test_concurrent_promotion_records_the_real_fill(self, tmp_path):
        """The sweep promoted the row to FILLED first. That is not terminal —
        discarding the fill left entry_price at the optimistic signal price."""
        st = _state(tmp_path)
        _seed(st, "T-1", status="filled")
        ex, pend = self._rig(st)
        ex._update_trade_filled(pend, 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price = conn.execute("SELECT entry_price FROM trades "
                                 "WHERE trade_id='T-1'").fetchone()[0]
        assert price == pytest.approx(1.62)
        assert st.get_state("alert_fill_after_close", "") == ""

    def test_genuinely_terminal_row_is_still_refused_and_paged(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, "T-2", status="closed")
        ex, _ = self._rig(st)
        pend = SimpleNamespace(
            trade_id="T-2", contracts=1,
            signal=SimpleNamespace(symbol="QQQ", legs=[], contract=None,
                                   strategy_name="iron_condor"))
        ex._update_trade_filled(pend, 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price = conn.execute("SELECT entry_price FROM trades "
                                 "WHERE trade_id='T-2'").fetchone()[0]
        assert price == pytest.approx(1.90), "wrote to a closed row"
        assert st.get_state("alert_fill_after_close", "").startswith("T-2|QQQ")
