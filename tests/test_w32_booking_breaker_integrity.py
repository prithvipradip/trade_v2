"""W32 (R26 F2): booking and breaker integrity — three residuals W14 left.

R26 re-probed W14's "booking exactly-once" at HEAD and found the waves had
closed the headline mechanism and left the finding's own acceptance test
failing in three places:

  two-phase-claim-double-books (concurrency-01 residual)
      W14 made the claim a MARK so a crash before any write is recoverable.
      But the booking is several non-idempotent writes (breaker, daily_stats,
      PDT, Thompson), and a fault AFTER some of them left the claim to go
      stale — the drain then re-applied the whole close. Probe: one -$272.29
      stop read as -$544.58 / 2 losses to the breaker. At-most-once became
      at-least-once. The money writes (daily_stats + the breaker blob) now
      land in ONE sqlite transaction together with a core_booked marker, and
      every other step is flagged as it lands, so a retry resumes instead of
      repeating.

  breaker-unreadable-never-reloads (regress-live-04 residual)
      After one unreadable read the breaker ran the rest of the process on a
      zero placeholder: is_tripped lifted the fail-closed pause after one
      window WITHOUT re-reading the store, so a daily-loss halt or a real
      streak sitting intact on disk was ignored until the next restart, and
      persistence stayed disabled for the life of the process.

  _update_trade_partial CAS gate (concurrency-03 residual)
      W14 taught _update_trade_filled that a refused CAS can mean "the sweep
      promoted this row first", not "terminal". The partial-fill twin kept the
      old reading: the real partial quantity was discarded, the row stayed at
      the full signal size, and a false CRITICAL page fired — through a single
      KV key that a second refusal overwrote.

Every test EXECUTES the real StateManager / CircuitBreaker / orchestrator /
executor. None reads source text.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import timedelta
from types import SimpleNamespace


import pytest

from ait.bot.state import StateManager
from ait.execution.executor import TradeExecutor
from ait.risk.circuit_breaker import CircuitBreaker
from ait.utils.time import now_et

from tests.test_w14_booking_exactly_once import _backdate, _cfg, _orch, _seed, _state


def _close(st, trade_id="T-1", pnl=-150.0, reason="stop_loss (x)"):
    _seed(st, trade_id)
    assert st.close_trade(trade_id, 2.5, pnl, exit_reason_detailed=reason)


def _stale_the_claim(st, trade_id="T-1"):
    """The booker died (or gave up): its claim ages past CLAIM_STALE_SECONDS."""
    old = (now_et() - timedelta(seconds=st.CLAIM_STALE_SECONDS + 120)).isoformat()
    with sqlite3.connect(st._db_path) as conn:
        conn.execute("UPDATE exit_outbox SET claimed_at = ?, exit_time = ? "
                     "WHERE trade_id = ?", (old, old, trade_id))


def _exit(trade_id="T-1", pnl=-150.0):
    return [{"trade_id": trade_id, "realized_pnl": pnl,
             "exit_reason": "stop_loss (x)", "exit_price": 2.5}]


class _FlakyThompson:
    """Raises on the first record_outcome — a fault AFTER the money writes."""

    def __init__(self):
        self.calls = 0

    def record_outcome(self, **kw):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("thompson store blew up")


# ------------------------------------------------------------ exactly-once
class TestACloseIsBookedExactlyOnce:
    def test_a_fault_after_the_money_writes_does_not_double_book(self, tmp_path):
        """The R26 probe, as a test: -150 must never read as -300."""
        st = _state(tmp_path)
        _close(st)
        o = _orch(st)
        o._thompson = _FlakyThompson()
        asyncio.run(o._process_completed_exits(_exit()))
        _stale_the_claim(st)
        asyncio.run(o._drain_exit_outbox())
        stats = st.get_daily_stats()
        assert stats.total_pnl == pytest.approx(-150.0), f"double-booked: {stats.total_pnl}"
        assert stats.trades_lost == 1
        assert o._circuit_breaker._daily_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._consecutive_losses == 1

    def test_the_failed_step_is_retried_and_the_obligation_then_retires(self, tmp_path):
        st = _state(tmp_path)
        _close(st)
        o = _orch(st)
        o._thompson = _FlakyThompson()
        asyncio.run(o._process_completed_exits(_exit()))
        assert st.pending_exit_bookings(older_than_seconds=-1) == [], \
            "a fresh claim must not be re-offered"
        _stale_the_claim(st)
        asyncio.run(o._drain_exit_outbox())
        assert o._thompson.calls == 2, "the step that failed was never retried"
        with sqlite3.connect(st._db_path) as conn:
            left = conn.execute("SELECT COUNT(*) FROM exit_outbox").fetchone()[0]
        assert left == 0, "obligation never retired"

    def test_the_operator_is_paged_once_not_once_per_attempt(self, tmp_path):
        st = _state(tmp_path)
        _close(st)
        o = _orch(st)
        o._thompson = _FlakyThompson()
        asyncio.run(o._process_completed_exits(_exit()))
        _stale_the_claim(st)
        asyncio.run(o._drain_exit_outbox())
        pages = [c.args[0] for c in o._send_notification.await_args_list]
        assert len(pages) == 1, pages

    def test_a_kill_after_the_core_commit_is_not_rebooked_by_the_next_process(self, tmp_path):
        """Cross-process: the core landed, the process died before completing.
        A NEW process (fresh breaker restored from the blob) drains the row."""
        st = _state(tmp_path)
        _close(st)
        o = _orch(st)
        o._circuit_breaker.attach_state(st)
        o._thompson = _FlakyThompson()
        asyncio.run(o._process_completed_exits(_exit()))
        _stale_the_claim(st)

        o2 = _orch(_state(tmp_path))            # the relaunched process
        o2._circuit_breaker.attach_state(o2._state)
        assert o2._circuit_breaker._daily_pnl == pytest.approx(-150.0), \
            "the breaker blob did not land with the stats"
        asyncio.run(o2._drain_exit_outbox())
        assert o2._state.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o2._circuit_breaker._daily_pnl == pytest.approx(-150.0)
        assert o2._circuit_breaker._consecutive_losses == 1

    def test_a_failed_core_write_rolls_the_in_memory_breaker_back(self, tmp_path):
        """If the transaction fails, the process lives on — its breaker must
        not keep a loss that was never made durable, or the retry doubles it."""
        st = _state(tmp_path)
        _close(st)
        o = _orch(st)
        real = st.book_exit_core
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real(*a, **kw)

        st.book_exit_core = flaky
        asyncio.run(o._process_completed_exits(_exit()))
        assert o._circuit_breaker._daily_pnl == pytest.approx(0.0)
        assert o._circuit_breaker._consecutive_losses == 0
        _stale_the_claim(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._daily_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._consecutive_losses == 1

    def test_the_drain_path_alone_is_exactly_once_too(self, tmp_path):
        st = _state(tmp_path)
        _close(st)
        _backdate(st, "T-1", minutes=10)
        o = _orch(st)
        o._thompson = _FlakyThompson()
        asyncio.run(o._drain_exit_outbox())
        _stale_the_claim(st)
        asyncio.run(o._drain_exit_outbox())
        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._consecutive_losses == 1

    def test_an_existing_outbox_gains_the_new_columns(self, tmp_path):
        """A live DB was created before these columns existed."""
        db = tmp_path / "ait_state.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE exit_outbox (trade_id TEXT PRIMARY KEY, "
                         "realized_pnl REAL NOT NULL, exit_reason TEXT DEFAULT '', "
                         "exit_time TEXT NOT NULL, claimed_at TEXT)")
        StateManager(db_path=db)
        with sqlite3.connect(db) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(exit_outbox)")}
        assert {"core_booked", "steps_done"} <= cols, cols


# ------------------------------------------------------------ breaker re-read
class _FlakyStore:
    """First read raises (a transient lock); afterwards it is the real store."""

    def __init__(self, inner, failures=1):
        self.inner, self.failures, self.gets = inner, failures, 0

    def get_state(self, key, default=""):
        self.gets += 1
        if self.failures > 0:
            self.failures -= 1
            raise sqlite3.OperationalError("database is locked")
        return self.inner.get_state(key, default)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _store_with_a_real_halt(tmp_path):
    st = _state(tmp_path)
    st.set_state(CircuitBreaker.STATE_KEY, json.dumps({
        "date": now_et().date().isoformat(), "consecutive_losses": 2,
        "daily_pnl": -2100.0, "tripped": True,
        "reason": "daily_loss (2.1% of $100000)", "resume_time": 0.0}))
    return st


@pytest.fixture
def clock(monkeypatch):
    t = {"now": time.time()}
    monkeypatch.setattr(time, "time", lambda: t["now"])
    return t


class TestTheBreakerReReadsItsStore:
    def test_a_transient_failure_does_not_hide_a_real_halt_for_the_process(
            self, tmp_path, clock):
        flaky = _FlakyStore(_store_with_a_real_halt(tmp_path))
        b = CircuitBreaker(_cfg())
        b.attach_state(flaky)
        assert b._state_unreadable and b.is_tripped
        clock["now"] += 31 * 60                      # the fail-closed window ends
        assert b.is_tripped is True, "lifted a real daily-loss halt blind"
        assert "daily_loss" in b._trip_reason
        assert flaky.gets > 1, "never re-read the store"
        assert b._state_unreadable is False

    def test_the_re_read_happens_early_not_only_when_the_window_ends(
            self, tmp_path, clock):
        flaky = _FlakyStore(_store_with_a_real_halt(tmp_path))
        b = CircuitBreaker(_cfg())
        b.attach_state(flaky)
        clock["now"] += 60
        assert b.is_tripped is True
        assert b._state_unreadable is False and b._consecutive_losses == 2

    def test_results_recorded_while_blind_are_replayed_onto_the_truth(
            self, tmp_path, clock):
        inner = _store_with_a_real_halt(tmp_path)
        b = CircuitBreaker(_cfg())
        b.attach_state(_FlakyStore(inner, failures=2))   # attach AND the first retry fail
        b.record_trade_result(-100.0)                # still blind: queued, nothing durable
        clock["now"] += 60
        b.is_tripped                                 # noqa: B018 — triggers the re-read
        blob = json.loads(inner.get_state(CircuitBreaker.STATE_KEY))
        assert blob["consecutive_losses"] == 3, blob
        assert blob["daily_pnl"] == pytest.approx(-2200.0), blob

    def test_a_store_that_stays_unreadable_stays_fail_closed(self, tmp_path, clock):
        b = CircuitBreaker(_cfg())
        b.attach_state(_FlakyStore(_state(tmp_path), failures=10 ** 6))
        clock["now"] += 31 * 60
        assert b.is_tripped is True, "resumed on a placeholder it cannot verify"
        clock["now"] += 31 * 60
        assert b.is_tripped is True

    def test_re_reads_are_rate_limited(self, tmp_path, clock):
        flaky = _FlakyStore(_state(tmp_path), failures=10 ** 6)
        b = CircuitBreaker(_cfg())
        b.attach_state(flaky)
        for _ in range(200):
            b.is_tripped                             # noqa: B018
        assert flaky.gets <= 2, flaky.gets


# ------------------------------------------------------------ partial-fill gate
def _executor(st):
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = st
    return ex


def _pending(trade_id, contracts=2):
    return SimpleNamespace(
        trade_id=trade_id, contracts=contracts,
        signal=SimpleNamespace(symbol="QQQ", legs=[], contract=None,
                               strategy_name="iron_condor", entry_price=1.90))


class TestPartialFillAfterAConcurrentPromotion:
    def test_the_real_partial_quantity_is_written(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, "T-1", status="filled")            # the sweep promoted it first
        with sqlite3.connect(st._db_path) as conn:
            conn.execute("UPDATE trades SET quantity = 2 WHERE trade_id='T-1'")
        _executor(st)._update_trade_partial(_pending("T-1"), 1, 1.62)
        with sqlite3.connect(st._db_path) as conn:
            qty, price = conn.execute("SELECT quantity, entry_price FROM trades "
                                      "WHERE trade_id='T-1'").fetchone()
        assert (qty, price) == (1, pytest.approx(1.62)), \
            "exit would be sized for 2 with 1 live at the broker"
        assert st.state_keys_like("alert_fill_after_close%") == []

    def test_a_terminal_row_is_still_refused(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, "T-2", status="closed")
        _executor(st)._update_trade_partial(_pending("T-2"), 1, 1.62)
        with sqlite3.connect(st._db_path) as conn:
            price = conn.execute("SELECT entry_price FROM trades "
                                 "WHERE trade_id='T-2'").fetchone()[0]
        assert price == pytest.approx(1.90), "wrote to a closed row"

    def test_two_refusals_are_two_flags_not_one_overwritten(self, tmp_path):
        st = _state(tmp_path)
        for tid in ("T-2", "T-3"):
            _seed(st, tid, status="closed")
        ex = _executor(st)
        ex._update_trade_partial(_pending("T-2"), 1, 1.62)
        ex._update_trade_filled(_pending("T-3", contracts=1), 1.62)
        keys = sorted(st.state_keys_like("alert_fill_after_close%"))
        assert keys == ["alert_fill_after_close_T-2", "alert_fill_after_close_T-3"], keys

    def test_the_drain_pages_every_flag_and_clears_them(self, tmp_path):
        st = _state(tmp_path)
        st.set_state("alert_fill_after_close_T-2", "T-2|QQQ|partial:1")
        st.set_state("alert_fill_after_close_T-3", "T-3|QQQ|1.62")
        st.set_state("alert_fill_after_close", "T-0|SPY|2.10")     # pre-W32 key
        o = _orch(st)
        asyncio.run(o._drain_exit_outbox())
        pages = " ".join(c.args[0] for c in o._send_notification.await_args_list)
        assert all(t in pages for t in ("T-0", "T-2", "T-3")), pages
        assert st.state_keys_like("alert_fill_after_close%") == []
