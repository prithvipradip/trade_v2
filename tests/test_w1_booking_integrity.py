"""W1 (R23 breaker-bypass family + concurrency-1 + fail-direction-10):
booking integrity. Every test EXECUTES the real code paths.

Pre-fix failures by construction:
- exit_outbox table/claim/pending did not exist (AttributeError);
- reconciler-style closes (close_trade with no callback) never moved the
  breaker or daily stats — the drain did not exist;
- a fill whose CAS was refused still rewrote entry_price and re-inserted
  open_positions on the closed row;
- a breaker whose state store raised on read started FRESH (untripped),
  silently clearing an active pause.
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
from ait.risk.circuit_breaker import CircuitBreaker
from ait.execution.executor import TradeExecutor


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed_trade(st: StateManager, trade_id="T-W1", status="filled",
                entry_price=1.90, entry_time=None) -> None:
    entry_time = entry_time or datetime.now().isoformat()
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, "QQQ", "iron_condor", "neutral", status, entry_time,
             entry_price))


def _backdate_outbox(st: StateManager, trade_id: str, minutes: float) -> None:
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "UPDATE exit_outbox SET exit_time = ? WHERE trade_id = ?",
            ((datetime.now() - timedelta(minutes=minutes)).isoformat(), trade_id))


def _breaker_cfg():
    return SimpleNamespace(max_daily_loss_pct=0.02, max_consecutive_losses=3,
                           pause_minutes_after_losses=30, max_api_failures=5)


def _orch(st: StateManager):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._circuit_breaker = CircuitBreaker(_breaker_cfg())
    o._pdt_guard = MagicMock()
    o._thompson = MagicMock()
    o._trainer = MagicMock()
    o._send_notification = AsyncMock()
    o._find_trade_by_id = lambda tid: SimpleNamespace(
        trade_id=tid, symbol="QQQ", strategy="iron_condor",
        entry_time=datetime.now().isoformat(), legs="[]")
    return o


# ------------------------------------------------------------ outbox basics
class TestOutbox:
    def test_close_trade_enqueues_transactionally(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st)
        assert st.close_trade("T-W1", 2.5, -272.29,
                              exit_reason_detailed="short_strike_touch (x)")
        rows = st.pending_exit_bookings(older_than_seconds=-1)
        assert [r["trade_id"] for r in rows] == ["T-W1"]
        assert rows[0]["realized_pnl"] == pytest.approx(-272.29)

    def test_refused_close_does_not_enqueue(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, status="closed")
        assert not st.close_trade("T-W1", 2.5, -100.0)
        assert st.pending_exit_bookings(older_than_seconds=-1) == []

    def test_claim_is_exactly_once(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st)
        st.close_trade("T-W1", 2.5, -50.0)
        assert st.claim_exit_booking("T-W1") is True
        assert st.claim_exit_booking("T-W1") is False

    def test_grace_period_hides_fresh_rows(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st)
        st.close_trade("T-W1", 2.5, -50.0)
        assert st.pending_exit_bookings(older_than_seconds=120.0) == []
        _backdate_outbox(st, "T-W1", minutes=3)
        assert len(st.pending_exit_bookings(older_than_seconds=120.0)) == 1


# ------------------------------------------------- the drain books orphans
class TestDrainBooksOrphanCloses:
    def test_reconciler_style_close_reaches_breaker_and_stats(self, tmp_path):
        # THE R23 headline: pre-W1 this loss hit trades.realized_pnl and
        # nothing else — breaker/daily stats/Thompson never moved.
        st = _state(tmp_path)
        _seed_trade(st)
        st.close_trade("T-W1", 8.76, -272.29,
                       exit_reason_detailed="short_strike_touch (x)")
        _backdate_outbox(st, "T-W1", minutes=3)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        stats = st.get_daily_stats()
        assert stats.total_pnl == pytest.approx(-272.29)
        assert stats.trades_lost == 1
        assert o._circuit_breaker._daily_pnl == pytest.approx(-272.29)
        o._thompson.record_outcome.assert_called_once()
        assert st.pending_exit_bookings(older_than_seconds=-1) == []

    def test_claimed_row_not_double_booked_by_drain(self, tmp_path):
        # The executor callback claims first; the drain must find nothing.
        st = _state(tmp_path)
        _seed_trade(st)
        st.close_trade("T-W1", 2.5, -50.0)
        _backdate_outbox(st, "T-W1", minutes=3)
        assert st.claim_exit_booking("T-W1")
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(0.0)
        assert o._circuit_breaker._daily_pnl == pytest.approx(0.0)

    def test_prior_day_close_claimed_but_not_booked_into_today(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st)
        st.close_trade("T-W1", 2.5, -50.0)
        _backdate_outbox(st, "T-W1", minutes=60 * 26)
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(0.0)
        assert st.pending_exit_bookings(older_than_seconds=-1) == []

    def test_fill_after_close_flag_pages_and_clears(self, tmp_path):
        st = _state(tmp_path)
        st.set_state("alert_fill_after_close", "T-W1|QQQ|1.62")
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        o._send_notification.assert_awaited()
        assert "CRITICAL" in o._send_notification.await_args_list[0].args[0]
        assert st.get_state("alert_fill_after_close", "") == ""


# ------------------------------------------------------- CAS-gated fills
class TestFillCasGate:
    def _executor(self, st):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = st
        return ex

    def _pending(self, trade_id="T-W1"):
        return SimpleNamespace(
            trade_id=trade_id, contracts=1,
            signal=SimpleNamespace(symbol="QQQ", legs=[], contract=None,
                                   strategy_name="iron_condor"))

    def test_fill_on_closed_row_writes_nothing(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, status="filled", entry_price=1.90)
        st.close_trade("T-W1", 2.5, -50.0)  # sweep/reconciler closed it
        self._executor(st)._update_trade_filled(self._pending(), 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price, status = conn.execute(
                "SELECT entry_price, status FROM trades WHERE trade_id='T-W1'"
            ).fetchone()
            open_rows = conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE trade_id='T-W1'"
            ).fetchone()[0]
        # pre-fix: entry_price rewritten to 1.62 and an orphan open_positions
        # row re-inserted on the closed $0 row
        assert price == pytest.approx(1.90)
        assert status == "closed"
        assert open_rows == 0
        assert st.get_state("alert_fill_after_close", "").startswith("T-W1|QQQ")

    def test_partial_fill_on_closed_row_writes_nothing(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, status="filled", entry_price=1.90)
        st.close_trade("T-W1", 2.5, -50.0)
        self._executor(st)._update_trade_partial(self._pending(), 1, 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price, qty = conn.execute(
                "SELECT entry_price, quantity FROM trades WHERE trade_id='T-W1'"
            ).fetchone()
        assert price == pytest.approx(1.90)
        assert qty == 1

    def test_fill_on_open_row_still_books_normally(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, status="pending", entry_price=1.90)
        self._executor(st)._update_trade_filled(self._pending(), 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price, status = conn.execute(
                "SELECT entry_price, status FROM trades WHERE trade_id='T-W1'"
            ).fetchone()
        assert status == "filled"
        assert price == pytest.approx(1.62)


# ------------------------------------------- breaker load fails closed
class TestBreakerFailClosed:
    def test_unreadable_store_trips_conservatively(self):
        b = CircuitBreaker(_breaker_cfg())
        bad = MagicMock()
        bad.get_state.side_effect = sqlite3.OperationalError("database is locked")
        b.attach_state(bad)
        # pre-fix: load failure fell through to a FRESH untripped breaker,
        # silently clearing any active pause
        assert b._tripped is True
        assert b._trip_reason == "state_unreadable_fail_closed"
        assert b._resume_time > 0

    def test_empty_store_stays_fresh(self, tmp_path):
        b = CircuitBreaker(_breaker_cfg())
        b.attach_state(_state(tmp_path))
        assert b._tripped is False
