"""W50 (R26 register Group F — the qty>1 ledger fork): money-flow-05,
trade-life-partial-slice-pnl-erased and trade-life-promotion-ignores-
broker-quantity.

Three findings from reports/blindspot_composition_hunt_20260825.md (lines
532, 540, 548), all ENFORCEMENT_GAP | live_money. Re-verified against HEAD
this session, not trusted from the original R23 audit text — every line
number it cites has moved since 2026-08-25, and the commission-estimate
mechanism it describes was itself relocated by W37/W48. Dormancy is real but
is TWO independent gates, not one: position_sizer.py's hard cap (keyed to
config.yaml's max_contracts_per_trade) bounds the position sizer's own
arithmetic, and paper_trading_mode separately pins the learning multiplier
at 1.0 (orchestrator.get_strategy_multiplier) — either alone keeps qty at 1
today.

This file lands in two parts, one commit each:

1. money-flow-05 + trade-life-partial-slice-pnl-erased (near-identical
   erasure, THIS commit): close_trade DELETEd open_positions.partial_exits
   before anyone folded it into realized_pnl, so a scaled-out trade's
   go-live PF input silently dropped every closed slice's P&L (and the
   executor's true-partial branch reached no ledger at all for a multi-leg
   fill). Fixed by folding the archived history into realized_pnl inside
   close_trade itself (one owner, atomic with the DELETE via BEGIN
   IMMEDIATE), feeding daily_stats/breaker per-slice in real time on BOTH
   partial-exit paths, and keeping the final booking's money delta to the
   slice NOT already fed in real time — while Thompson/drift/the go-live
   scorecard all read the FULL round trip. This is a deliberate departure
   from the register's own suggested fix (which said to stop the ladder
   path's real-time feed instead) — that alternative would regress BC-H3's
   real-time breaker visibility on a locked-in partial loss. The
   commission true-up's remembered estimate (W37) now ACCUMULATES across
   slices instead of being overwritten by the last one, matching what
   total_commission() (which sums every execution row) actually needs.

2. trade-life-promotion-ignores-broker-quantity (separate, more
   dangerous, next commit): reconciler promotion adopted the LOCAL
   quantity unconditionally; the later exit then reversed that quantity
   blind, which could open an inverted position on a restart that missed a
   partial fill. Fixed two ways: promotion now adopts the broker's own leg
   quantity when every leg agrees, and freezes new entries (the same
   frozen_keys/HALT_UNTRACKED path the untracked-position branches use)
   when legs disagree rather than guessing which number is real; and
   position_liveness — the gate the exit path already consults before
   every reverse combo — is now quantity-aware, so even an un-adopted
   mismatch is caught before the reversal, not just at promotion time.

Every test executes the real StateManager / CircuitBreaker / executor /
orchestrator / reconciler. No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeStatus
from ait.execution.executor import PendingExitOrder, TradeExecutor
from ait.execution.reconciler import PositionReconciler
from ait.risk.circuit_breaker import CircuitBreaker
from ait.utils.time import now_et
from tests.fakes import FakeIB, FakeIBKRClient, _option_position

TID = "T-W50"

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
]


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed(st, trade_id=TID, quantity=2, strategy="iron_condor",
         contract_type="iron_condor", legs=None, status="filled",
         entry_time=None) -> None:
    """One trade row, raw SQL (same idiom as test_w14/test_w37) — flexible
    enough for both the booking-pipeline tests (status=filled, needs an
    open_positions row) and the reconciler promotion tests (status=pending,
    no open_positions row until promoted)."""
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type, legs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trade_id, "SPY", strategy, "short", status,
             entry_time or now_et().isoformat(), 2.50, quantity,
             contract_type, json.dumps(legs if legs is not None else [])))
        if status in ("filled", "partial", "closing"):
            conn.execute(
                "INSERT INTO open_positions (position_id, trade_id, symbol, "
                "contract_type, quantity, entry_price, entry_time, "
                "partial_exits) VALUES (?,?,?,?,?,?,?,?)",
                (trade_id, trade_id, "SPY", contract_type, quantity, 2.50,
                 now_et().isoformat(), "[]"))


def _cfg():
    return SimpleNamespace(max_daily_loss_pct=0.02, max_consecutive_losses=3,
                           pause_minutes_after_losses=30, max_api_failures=5)


def _orch(st):
    """Real StateManager, real CircuitBreaker, real _find_trade_by_id (the
    W37-rig shape, not W14/W32's hardcoded-namespace one) — these tests need
    trade.partial_exits to actually round-trip through the DB."""
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._circuit_breaker = CircuitBreaker(_cfg())
    o._pdt_guard = MagicMock()
    o._thompson = MagicMock()
    o._trainer = MagicMock()
    o._send_notification = AsyncMock()
    o._find_trade_by_id = st.get_trade_by_id
    return o


def _executor(st, breaker=None):
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = st
    ex._circuit_breaker = breaker if breaker is not None else CircuitBreaker(_cfg())
    return ex


def _fake_fill(order_id, filled, avg_price):
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id),
        orderStatus=SimpleNamespace(filled=filled, avgFillPrice=avg_price),
        fills=[], contract=SimpleNamespace(secType="BAG"))


def _rec(st, ib=None) -> PositionReconciler:
    return PositionReconciler(FakeIBKRClient(ib or FakeIB()), st)


def _halt_text(tmp_path) -> str | None:
    p = tmp_path / "data" / "HALT_UNTRACKED"
    return p.read_text() if p.exists() else None


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    """Same fixture shape as test_w41_live_money_floors.py — the reconciler
    writes data/HALT_UNTRACKED relative to cwd."""
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Part 1: money-flow-05 + trade-life-partial-slice-pnl-erased
# ---------------------------------------------------------------------------

class TestCloseTradeFoldsPartialHistory:
    """The core fix: close_trade reads open_positions.partial_exits BEFORE
    the DELETE and folds the slices' P&L into realized_pnl."""

    def test_partial_slices_are_folded_into_the_final_realized_pnl(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, quantity=1)  # what's left after the earlier partial(s)
        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=60.0)
        st.record_partial_exit(TID, quantity=1, price=1.30, pnl=40.0)

        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-25.0,
                              exit_reason_detailed="stop_loss (x)")

        trade = st.get_trade_by_id(TID)
        assert trade.realized_pnl == pytest.approx(60.0 + 40.0 - 25.0), (
            "the two partial slices' P&L did not survive the close — "
            f"got {trade.realized_pnl}"
        )

    def test_the_partial_history_is_archived_onto_the_trades_row(self, tmp_path):
        """The erasure this closes: open_positions.partial_exits used to be
        DELETEd with nothing keeping a copy anywhere."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=60.0)

        st.close_trade(TID, exit_price=2.80, realized_pnl=-25.0)

        archived = json.loads(st.get_trade_by_id(TID).partial_exits)
        assert len(archived) == 1
        assert archived[0]["pnl"] == pytest.approx(60.0)

    def test_no_partial_history_is_byte_identical_to_pre_w50(self, tmp_path):
        """CONTROL: the common case (every trade in the current track
        record has no partial-exit history) must be unaffected — folded
        realized_pnl equals the remainder exactly, archive stays empty."""
        st = _state(tmp_path)
        _seed(st, quantity=1)

        st.close_trade(TID, exit_price=2.80, realized_pnl=-25.0)

        trade = st.get_trade_by_id(TID)
        assert trade.realized_pnl == pytest.approx(-25.0)
        assert json.loads(trade.partial_exits) == []

    def test_record_partial_exit_after_close_does_not_resurrect_or_corrupt(self, tmp_path):
        """Defensive check on the DELETE ordering: partial_exits is read and
        folded INSIDE close_trade's own transaction, before the
        open_positions row is deleted. A record_partial_exit call against
        the now-gone row must be a no-op, not a crash or a silent corruption
        of the already-folded P&L."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=60.0)
        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-25.0)

        st.record_partial_exit(TID, quantity=1, price=9.99, pnl=999.0)  # must not raise

        trade = st.get_trade_by_id(TID)
        assert trade.realized_pnl == pytest.approx(60.0 - 25.0), (
            "a record_partial_exit call after close changed the already-folded P&L"
        )

    def test_the_exit_outbox_row_also_carries_the_folded_total(self, tmp_path):
        """The crash-recovery drain (pending_exit_bookings) reads
        exit_outbox.realized_pnl as if it were the trade's whole P&L —
        it must agree with trades.realized_pnl, not the bare remainder."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=60.0)
        st.close_trade(TID, exit_price=2.80, realized_pnl=-25.0)

        with sqlite3.connect(st._db_path) as conn:
            row = conn.execute(
                "SELECT realized_pnl FROM exit_outbox WHERE trade_id = ?",
                (TID,)).fetchone()
        assert row[0] == pytest.approx(60.0 - 25.0)


class TestCommissionEstimateAccumulatesAcrossSlices:
    """W37 remembers the fill-time commission ESTIMATE so the true-up can
    reverse exactly what was subtracted. Before this wave, a SECOND
    _calculate_realized_pnl call for the same trade_id (the remainder,
    after an earlier partial) OVERWROTE the first slice's remembered
    estimate instead of adding to it — the true-up then diffed a
    single-slice estimate against total_commission()'s all-rows sum."""

    def _cost_model(self, monkeypatch, per_leg):
        monkeypatch.setattr(
            "ait.config.settings.load_settings",
            lambda *a, **k: SimpleNamespace(backtest=SimpleNamespace(
                commission_per_contract=per_leg, regulatory_fees_per_contract=0.0)))

    def test_two_slices_remember_the_sum_not_the_last_one(self, tmp_path, monkeypatch):
        st = _state(tmp_path)
        _seed(st, quantity=1)
        self._cost_model(monkeypatch, per_leg=0.65)
        ex = _executor(st)
        trade = st.get_trade_by_id(TID)

        ex._calculate_realized_pnl(trade, exit_price=1.00)   # slice: est 5.20
        ex._calculate_realized_pnl(trade, exit_price=1.00)   # remainder: est 5.20

        remembered = float(st.get_state(f"commission_est_{TID}"))
        assert remembered == pytest.approx(5.20 * 2), (
            f"expected the SUM of both calls' estimates (10.40), got "
            f"{remembered} — a second call overwrote the first instead of "
            "accumulating"
        )

    def test_the_true_up_reverses_the_full_accumulated_estimate(
            self, tmp_path, monkeypatch):
        """Reproduces the actual consequence: without accumulation, the
        true-up diffs a ONE-slice estimate (5.20) against a real ledger
        covering TWO slices' worth of legs, landing every partial-exit
        trade's true-up short by the earlier slice's estimated commission."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        self._cost_model(monkeypatch, per_leg=0.65)
        ex = _executor(st)
        trade = st.get_trade_by_id(TID)

        pnl_1 = ex._calculate_realized_pnl(trade, exit_price=1.00)
        pnl_2 = ex._calculate_realized_pnl(trade, exit_price=1.00)
        booked_total = round(pnl_1 + pnl_2, 2)
        accumulated_est = float(st.get_state(f"commission_est_{TID}"))

        # 4 legs x (entry + exit) x 2 "slices worth" of legs = 16 real rows.
        with sqlite3.connect(st._db_path) as conn:
            for i in range(16):
                conn.execute(
                    "INSERT INTO executions (exec_id, order_id, trade_id, "
                    "symbol, commission) VALUES (?,?,?,?,?)",
                    (f"E{i}", 1, TID, "SPY", 0.9168))
        real_total = 16 * 0.9168

        trued = _orch(st)._true_up_commission(TID, booked_total)

        expected_delta = round(accumulated_est - real_total, 2)
        expected = round(booked_total + expected_delta, 2)
        assert trued == pytest.approx(expected, abs=0.01)
        # And directly: the delta must reflect BOTH slices' estimate, not
        # just the last one's 5.20.
        assert accumulated_est == pytest.approx(10.40)


class TestFinalBookingDoesNotDoubleCountRealTimeFedSlices:
    """The money/outcome split: a partial slice already fed daily_stats and
    the breaker in real time (both partial-exit paths call
    record_partial_pnl as each slice closes) must not be counted AGAIN when
    the trade's folded total is booked at final close."""

    def test_a_partial_already_fed_in_real_time_is_counted_once(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, quantity=1)
        o = _orch(st)

        # The slice, exactly as both partial-exit paths now book it: history
        # in partial_exits AND daily_stats/breaker fed in real time.
        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=100.0)
        stats = st.get_daily_stats()
        stats.total_pnl += 100.0
        st.update_daily_stats(stats)
        o._circuit_breaker.record_partial_pnl(100.0)

        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-30.0)
        folded = st.get_trade_by_id(TID).realized_pnl
        assert folded == pytest.approx(70.0)

        asyncio.run(o._process_completed_exits([
            {"trade_id": TID, "realized_pnl": folded,
             "exit_reason": "stop_loss (x)", "exit_price": 2.80},
        ]))

        stats_after = st.get_daily_stats()
        assert stats_after.total_pnl == pytest.approx(70.0), (
            f"expected the slice's 100 counted ONCE plus the -30 remainder "
            f"= 70 total; got {stats_after.total_pnl} (170 would mean the "
            "slice was double-counted)"
        )
        assert o._circuit_breaker._daily_pnl == pytest.approx(70.0)

    def test_a_trade_with_no_partial_history_books_the_full_amount_once(self, tmp_path):
        """CONTROL: the ordinary case — no real-time feed happened, so the
        whole realized_pnl is the money delta, exactly as before this wave."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        o = _orch(st)

        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-150.0)
        asyncio.run(o._process_completed_exits([
            {"trade_id": TID, "realized_pnl": -150.0,
             "exit_reason": "stop_loss (x)", "exit_price": 2.80},
        ]))

        assert st.get_daily_stats().total_pnl == pytest.approx(-150.0)
        assert o._circuit_breaker._daily_pnl == pytest.approx(-150.0)


class TestWinLossUsesTheFullRoundTripNotJustTheRemainder:
    def test_a_net_positive_trade_resets_the_loss_streak_despite_a_losing_remainder(
            self, tmp_path):
        st = _state(tmp_path)
        _seed(st, quantity=1)
        o = _orch(st)
        o._circuit_breaker._consecutive_losses = 2  # two prior real losses

        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=100.0)
        stats = st.get_daily_stats()
        stats.total_pnl += 100.0
        st.update_daily_stats(stats)
        o._circuit_breaker.record_partial_pnl(100.0)

        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-30.0)  # remainder LOSS
        folded = st.get_trade_by_id(TID).realized_pnl
        assert folded > 0, "the trade must net positive overall for this test to mean anything"

        asyncio.run(o._process_completed_exits([
            {"trade_id": TID, "realized_pnl": folded,
             "exit_reason": "stop_loss (x)", "exit_price": 2.80},
        ]))

        assert o._circuit_breaker._consecutive_losses == 0, (
            "a trade that netted positive overall must reset the loss "
            "streak, even though its remainder slice alone was a loss"
        )
        assert st.get_daily_stats().trades_won == 1
        assert st.get_daily_stats().trades_lost == 0

    def test_a_net_negative_trade_still_counts_as_a_loss(self, tmp_path):
        """CONTROL: the streak must still extend for a genuinely losing
        trade — this is not "wins never lose" by accident."""
        st = _state(tmp_path)
        _seed(st, quantity=1)
        o = _orch(st)

        st.record_partial_exit(TID, quantity=1, price=1.10, pnl=10.0)
        stats = st.get_daily_stats()
        stats.total_pnl += 10.0
        st.update_daily_stats(stats)
        o._circuit_breaker.record_partial_pnl(10.0)

        assert st.close_trade(TID, exit_price=2.80, realized_pnl=-200.0)
        folded = st.get_trade_by_id(TID).realized_pnl
        assert folded < 0

        asyncio.run(o._process_completed_exits([
            {"trade_id": TID, "realized_pnl": folded,
             "exit_reason": "stop_loss (x)", "exit_price": 2.80},
        ]))

        assert o._circuit_breaker._consecutive_losses == 1
        assert st.get_daily_stats().trades_lost == 1


class TestExecutorTruePartialFeedsStatsInRealTime:
    """Before this wave, a multi-leg true-partial fill (a condor scaling
    out — the orchestrator's own single-leg ladder path can never take it,
    per money-flow-05's own correction) reached NO ledger at all until the
    eventual final close."""

    def test_a_multi_leg_true_partial_now_reaches_daily_stats_and_breaker(self, tmp_path):
        st = _state(tmp_path)
        _seed(st, quantity=2, legs=CONDOR_LEGS)
        breaker = CircuitBreaker(_cfg())
        ex = _executor(st, breaker=breaker)

        pending = PendingExitOrder(TID, "partial_take_profit", estimated_pnl=50.0)
        fill = _fake_fill(order_id=1, filled=1, avg_price=-1.00)  # closed 1 of 2
        completed_exits: list[dict] = []

        ex._book_partial_exit(1, pending, [fill], completed_exits)

        assert st.get_daily_stats().total_pnl != 0.0, (
            "the true-partial slice never reached daily_stats"
        )
        assert breaker._daily_pnl != 0.0, (
            "the true-partial slice never reached the circuit breaker"
        )
        assert completed_exits == [], (
            "a TRUE partial (not the whole remaining position) must not "
            "emit a completed-exit event — the trade isn't closed yet"
        )
        assert st.get_trade_by_id(TID).quantity == 1

    def test_the_slice_uses_record_partial_pnl_not_a_completed_trade_result(self, tmp_path):
        """CONTROL: a scale-out must not touch the consecutive-loss counter
        (record_partial_pnl's whole reason to exist, per BC-H3) even when
        the slice itself is a loss."""
        st = _state(tmp_path)
        _seed(st, quantity=2, legs=CONDOR_LEGS)
        breaker = CircuitBreaker(_cfg())
        breaker._consecutive_losses = 2
        ex = _executor(st, breaker=breaker)

        pending = PendingExitOrder(TID, "stop_loss", estimated_pnl=-50.0)
        fill = _fake_fill(order_id=1, filled=1, avg_price=5.00)  # a losing close price
        ex._book_partial_exit(1, pending, [fill], [])

        assert breaker._consecutive_losses == 2, (
            "a partial slice must never advance (or reset) the "
            "consecutive-loss counter — it is not a completed trade"
        )
