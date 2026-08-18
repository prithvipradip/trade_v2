"""R12 order-lifecycle tests — real executor + real state, fake broker.

The five scenarios the R12 test report specified, written against the
POST-hardening contracts of Tier A items 1-4 (CAS status transitions,
filled-qty-first partial handling, disconnect-safe check_fills, capped
reprice ladder, promotion-before-sweep reconcile):

1. place -> fill -> exit -> close for a credit iron condor (P&L booked from
   real fill prices, credit-membership formula, no resurrection after close)
2. partial-then-cancel keeps the filled contracts MANAGED (status PARTIAL,
   retained quantity, open_positions row) — never a clean CANCELLED
3. a disconnect blip (or post-reconnect broker amnesia) never cancels
   working orders
4. the entry reprice ladder escalates toward marketable, honors the
   risk-validated debit cap, and times out into a real cancel
5. startup reconcile promotes PENDING trades whose legs are live at the
   broker and sweeps dead PENDING orphans as never-filled

Pattern notes: see tests/fakes.py — real TradeExecutor/StateManager on a tmp
SQLite file; only the broker boundary is faked (SimpleNamespace shapes, no
MagicMock).
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from ait.bot.state import StateManager, TradeStatus
from ait.execution.executor import TradeExecutor
from ait.execution.reconciler import PositionReconciler
from ait.risk.circuit_breaker import CircuitBreaker
from ait.strategies.base import Signal, SignalDirection

from tests.fakes import FakeIB, FakeIBKRClient, _option_position

EXPIRY_ISO = "2026-08-21"
EXPIRY_IB = "20260821"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch, risk_config):
    """Real executor + state on a tmp DB; fake broker; guards relaxed."""
    monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
    monkeypatch.setenv("AIT_ENTRY_LADDER_START", "0.25")  # pin ladder step 0
    state = StateManager(tmp_path / "state.db")
    ib = FakeIB()
    client = FakeIBKRClient(ib)
    executor = TradeExecutor(
        ibkr_client=client,
        state=state,
        circuit_breaker=CircuitBreaker(risk_config),
        order_timeout=240,
    )
    return SimpleNamespace(state=state, ib=ib, client=client,
                           executor=executor, db=tmp_path / "state.db")


def _leg(strike: float, right: str, action: str) -> dict:
    return {
        "contract": SimpleNamespace(strike=strike, right=right, expiry=EXPIRY_ISO),
        "action": action,
        "ratio": 1,
    }


def _ic_signal(credit: float = 1.00, width: float = 3.0) -> Signal:
    """Defined-risk SPY iron condor collecting ``credit`` (2 SELL / 2 BUY)."""
    return Signal(
        symbol="SPY",
        strategy_name="iron_condor",
        direction=SignalDirection.NEUTRAL,
        confidence=0.70,
        legs=[
            _leg(95.0, "P", "BUY"),
            _leg(98.0, "P", "SELL"),
            _leg(102.0, "C", "SELL"),
            _leg(105.0, "C", "BUY"),
        ],
        entry_price=credit,
        max_loss=(width - credit) * 100,
        max_profit=credit * 100,
        expiry=date(2026, 8, 21),
    )


def _debit_spread_signal(debit: float = 2.00, max_loss: float = 210.0) -> Signal:
    """Bull call spread paying ``debit`` (1 BUY / 1 SELL)."""
    return Signal(
        symbol="SPY",
        strategy_name="bull_call_spread",
        direction=SignalDirection.BULLISH,
        confidence=0.70,
        legs=[
            _leg(100.0, "C", "BUY"),
            _leg(105.0, "C", "SELL"),
        ],
        entry_price=debit,
        max_loss=max_loss,
        max_profit=(5.0 - debit) * 100,
        expiry=date(2026, 8, 21),
    )


def _trade_row(env, trade_id: str):
    return env.state.get_trade_by_id(trade_id)


def _open_position_row(env, trade_id: str):
    with sqlite3.connect(env.db) as conn:
        return conn.execute(
            "SELECT quantity, entry_price FROM open_positions WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()


async def _place(env, signal: Signal, contracts: int = 1) -> tuple[str, int]:
    trade_id = await env.executor.execute_signal(signal, contracts)
    assert trade_id is not None, "execute_signal refused the order"
    order_id = next(iter(env.executor._pending_orders))
    return trade_id, order_id


# ---------------------------------------------------------------------------
# 1. place -> fill -> exit -> close (credit iron condor)
# ---------------------------------------------------------------------------

async def test_credit_ic_place_fill_exit_close(env):
    trade_id, order_id = await _place(env, _ic_signal(credit=1.00))

    # Placed: PENDING row exists, order is working at the broker.
    assert _trade_row(env, trade_id).status == TradeStatus.PENDING
    assert len(env.ib.openTrades()) == 1
    # Credit combo quotes a NEGATIVE limit (always-BUY convention).
    assert env.ib.openTrades()[0].order.lmtPrice < 0

    # Entry fills at -0.95 (collected $0.95, some slippage vs the $1.00 mid).
    env.ib.resolve(order_id, "Filled", avg_price=-0.95, filled=1)
    filled, exits = await env.executor.check_fills()
    assert filled == [trade_id] and exits == []

    row = _trade_row(env, trade_id)
    assert row.status == TradeStatus.FILLED
    assert row.entry_price == pytest.approx(0.95)  # REAL fill, unsigned premium
    pos = _open_position_row(env, trade_id)
    assert pos is not None and pos[0] == 1  # managed by the portfolio monitor

    # Exit: orchestrator flips FILLED -> CLOSING (CAS) and registers the order.
    assert env.state.update_trade_status(
        trade_id, TradeStatus.CLOSING,
        from_statuses=[TradeStatus.FILLED, TradeStatus.PARTIAL],
    )
    exit_order = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=0.55,
                                 action="BUY", orderType="LMT")
    exit_trade = env.ib.placeOrder(
        SimpleNamespace(secType="BAG", symbol="SPY"), exit_order)
    exit_oid = exit_trade.order.orderId
    env.executor.register_exit_order(exit_oid, trade_id, "take_profit_50pct",
                                     estimated_pnl=55.0)
    env.executor._pending_exit_orders[exit_oid].submitted_at -= 30  # age past grace

    # Exit fills: bought the condor back for $0.40.
    env.ib.resolve(exit_oid, "Filled", avg_price=0.40, filled=1)
    filled, exits = await env.executor.check_fills()

    assert len(exits) == 1
    booked = exits[0]
    assert booked["trade_id"] == trade_id
    assert booked["exit_price"] == pytest.approx(0.40)
    # Credit membership: (entry_credit - exit_debit)*100*qty - 8-leg commission
    expected_pnl = (0.95 - 0.40) * 100 - 0.65 * 4 * 1 * 2
    assert booked["realized_pnl"] == pytest.approx(expected_pnl)

    row = _trade_row(env, trade_id)
    assert row.status == TradeStatus.CLOSED
    assert row.realized_pnl == pytest.approx(expected_pnl)
    assert _open_position_row(env, trade_id) is None
    assert env.executor.pending_count == 0

    # Post-hardening: terminal state is terminal. No double close, no
    # CLOSING->FILLED resurrection (the position-REVERSAL edge).
    assert env.state.close_trade(trade_id, 0.0, 0.0) is False
    assert env.state.update_trade_status(
        trade_id, TradeStatus.FILLED, from_statuses=[TradeStatus.CLOSING]) is False
    assert _trade_row(env, trade_id).status == TradeStatus.CLOSED
    assert _trade_row(env, trade_id).realized_pnl == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# 2. partial fill then cancel of the remainder stays MANAGED
# ---------------------------------------------------------------------------

async def test_partial_then_cancel_keeps_filled_contracts_managed(env):
    trade_id, order_id = await _place(env, _ic_signal(credit=1.00), contracts=2)

    # 1 of 2 fills, then the order dies at IBKR (cancel/inactive) — the
    # filled contract is LIVE at the broker and must stay managed.
    env.ib.resolve(order_id, "Cancelled", avg_price=-0.90, filled=1, remaining=1)
    filled, _ = await env.executor.check_fills()

    row = _trade_row(env, trade_id)
    assert row.status == TradeStatus.PARTIAL, (
        "partial-then-cancel must book PARTIAL, not a clean CANCELLED — "
        "the filled contracts have no stop/TP/expiry handling otherwise"
    )
    assert row.quantity == 1  # retained qty = what actually filled
    pos = _open_position_row(env, trade_id)
    assert pos is not None and pos[0] == 1
    # Booking is done: the order is no longer tracked (no re-book every cycle),
    # and the trade surfaces in `filled` so the orchestrator manages it.
    assert order_id not in env.executor._pending_orders
    assert trade_id in filled

    # Post-hardening: PARTIAL must not silently become FILLED-at-full-qty.
    assert _trade_row(env, trade_id).quantity == 1


# ---------------------------------------------------------------------------
# 3. disconnect blip: working orders are NOT mass-cancelled
# ---------------------------------------------------------------------------

async def test_disconnect_blip_does_not_cancel_working_orders(env):
    trade_id, order_id = await _place(env, _ic_signal(credit=1.00))

    # Blip: connection drops mid-cycle. check_fills must skip the pass.
    env.client.connected = False
    filled, exits = await env.executor.check_fills()
    assert (filled, exits) == ([], [])
    assert _trade_row(env, trade_id).status == TradeStatus.PENDING
    assert order_id in env.executor._pending_orders
    assert env.client.cancelled == []

    # Reconnect amnesia: broker lists come back EMPTY for a moment. A young
    # unknown order is still pending, not cancelled.
    env.client.connected = True
    env.ib.open_trades.clear()
    env.ib.all_trades.clear()
    filled, exits = await env.executor.check_fills()
    assert (filled, exits) == ([], [])
    assert _trade_row(env, trade_id).status == TradeStatus.PENDING
    assert order_id in env.executor._pending_orders
    assert env.ib.cancelled_order_ids == []


# ---------------------------------------------------------------------------
# 4. reprice ladder: escalates, honors the debit cap, times out
# ---------------------------------------------------------------------------

async def test_ladder_escalates_honors_debit_cap_and_times_out(env):
    # Debit spread at $2.00; risk manager validated $210 => $2.10/contract cap.
    trade_id, order_id = await _place(env, _debit_spread_signal(2.00, 210.0))
    pending = env.executor._pending_orders[order_id]
    order = env.ib.openTrades()[0].order

    # Step 0: near mid — base + 25% of the 15% marketable offset.
    step0 = order.lmtPrice
    assert step0 == pytest.approx(2.08, abs=0.01)
    assert pending.debit_cap == pytest.approx(2.10)

    # 46s unfilled -> step 1 (60% of offset = 2.18) but the risk-validated
    # cap wins: never above 2.10.
    pending.submitted_at -= 46
    await env.executor._cancel_stale_orders()
    step1 = order.lmtPrice
    assert step1 >= step0  # debit ladder is monotonic toward marketable
    assert step1 == pytest.approx(2.10)

    # 92s -> step 2 (100% of offset = 2.30): still capped at 2.10.
    pending.submitted_at -= 46
    await env.executor._cancel_stale_orders()
    assert order.lmtPrice == pytest.approx(2.10)
    assert order.lmtPrice <= pending.debit_cap

    # Past the order timeout: the entry is cancelled for real and the
    # trade row goes CANCELLED (no phantom position).
    pending.submitted_at -= 300
    filled, _ = await env.executor.check_fills()
    assert order_id in env.client.cancelled
    assert filled == []
    assert _trade_row(env, trade_id).status == TradeStatus.CANCELLED
    assert order_id not in env.executor._pending_orders


async def test_multi_contract_debit_cap_stays_per_contract(env):
    """R17 (found in passing, not one of the original 20): signal.max_loss is
    ALWAYS a per-contract figure (every strategy builds its signal at
    quantity=1) and so is the limit price the cap bounds — contracts sized
    >1 must not shrink the cap. Pre-fix, `max_loss / (100 * contracts)`
    clamped a 2-contract $210/contract spread's cap to $1.05 instead of
    $2.10, pricing every escalation step below any real market and letting
    the order time out unfilled — silently killing every multi-contract
    debit entry once sizing scales past 1 lot."""
    trade_id, order_id = await _place(env, _debit_spread_signal(2.00, 210.0), contracts=3)
    pending = env.executor._pending_orders[order_id]

    # Cap must stay per-contract ($2.10), not shrink to 210/(100*3)=$0.70.
    assert pending.debit_cap == pytest.approx(2.10)

    order = env.ib.openTrades()[0].order
    pending.submitted_at -= 92  # step 2: 100% of the marketable offset
    await env.executor._cancel_stale_orders()
    assert order.lmtPrice == pytest.approx(2.10)


# ---------------------------------------------------------------------------
# 5. reconcile: promotes live PENDING, sweeps dead PENDING
# ---------------------------------------------------------------------------

def _pending_ic_record(state: StateManager, trade_id: str, age_minutes: int) -> None:
    """Insert an aged PENDING iron-condor row via the executor's own schema."""
    from ait.bot.state import TradeDirection, TradeRecord
    import json

    legs = [
        {"strike": 95.0, "right": "P", "action": "BUY", "expiry": EXPIRY_ISO},
        {"strike": 98.0, "right": "P", "action": "SELL", "expiry": EXPIRY_ISO},
        {"strike": 102.0, "right": "C", "action": "SELL", "expiry": EXPIRY_ISO},
        {"strike": 105.0, "right": "C", "action": "BUY", "expiry": EXPIRY_ISO},
    ]
    state.record_trade(TradeRecord(
        trade_id=trade_id,
        symbol="SPY",
        strategy="iron_condor",
        direction=TradeDirection.SHORT,
        status=TradeStatus.PENDING,
        entry_time=(datetime.now() - timedelta(minutes=age_minutes)).isoformat(),
        entry_price=1.00,
        quantity=1,
        contract_type="iron_condor",
        expiry=EXPIRY_ISO,
        legs=json.dumps(legs),
    ))


async def test_reconcile_promotes_live_pending_and_sweeps_dead(tmp_path):
    # --- A: legs LIVE at the broker => promotion to FILLED, never a sweep.
    state_a = StateManager(tmp_path / "a.db")
    _pending_ic_record(state_a, "T-LIVE", age_minutes=40)
    ib_a = FakeIB()
    ib_a.positions_list = [
        _option_position("SPY", 95.0, "P", EXPIRY_IB, qty=1),
        _option_position("SPY", 98.0, "P", EXPIRY_IB, qty=-1),
        _option_position("SPY", 102.0, "C", EXPIRY_IB, qty=-1),
        _option_position("SPY", 105.0, "C", EXPIRY_IB, qty=1),
    ]
    rec_a = PositionReconciler(FakeIBKRClient(ib_a), state_a)
    result_a = await rec_a.reconcile()

    assert result_a.promoted == 1
    promoted = state_a.get_trade_by_id("T-LIVE")
    assert promoted.status == TradeStatus.FILLED
    assert promoted.exit_reason_detailed == ""  # not closed, not swept

    # --- B: nothing at the broker => the aged PENDING orphan is swept as
    # never-filled bookkeeping ($0, excluded from real-close analytics), and
    # the zero-options guard refuses to mass-close anything as a stale local.
    state_b = StateManager(tmp_path / "b.db")
    _pending_ic_record(state_b, "T-DEAD", age_minutes=40)
    rec_b = PositionReconciler(FakeIBKRClient(FakeIB()), state_b)
    result_b = await rec_b.reconcile()

    assert result_b.promoted == 0
    dead = state_b.get_trade_by_id("T-DEAD")
    assert dead.status == TradeStatus.CLOSED
    assert dead.exit_reason_detailed == "stale_pending_never_filled"
    assert dead.realized_pnl == 0.0
    assert dead.exit_price == 0.0


# ---------------------------------------------------------------------------
# 6. INCIDENT 2026-07-13 replay: blind-window exit re-trigger cannot duplicate
# ---------------------------------------------------------------------------

async def test_incident_20260713_exit_retrigger_books_exactly_one_close(env):
    """Replay of the 07-13 macro-flatten triple-fill (PLAN.md INCIDENT):
    exit #1 filled at the broker while fill detection was blind; the next
    fast-monitor cycles re-placed the same exit twice, and the duplicate
    fills built untracked inverse positions (4 accidental reverse condors).
    Post-R12 contract under the same timeline:
      a) the FILLED->CLOSING CAS refuses a re-trigger while an exit is in
         flight (the orchestrator only places an exit after CAS success);
      b) the stale-pending path cancels but KEEPS TRACKING (no premature
         CLOSING->FILLED revert);
      c) when the cancel loses the race, the late-detected fill books
         exactly ONE close and the trade is terminal."""
    trade_id, order_id = await _place(env, _ic_signal(credit=1.00))
    env.ib.resolve(order_id, "Filled", avg_price=-0.95, filled=1)
    await env.executor.check_fills()
    assert _trade_row(env, trade_id).status == TradeStatus.FILLED

    # 13:30:44 — exit #1: CAS FILLED->CLOSING, place, register.
    assert env.state.update_trade_status(
        trade_id, TradeStatus.CLOSING,
        from_statuses=[TradeStatus.FILLED, TradeStatus.PARTIAL],
    )
    exit_order = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=0.55,
                                 action="BUY", orderType="LMT")
    exit_trade = env.ib.placeOrder(
        SimpleNamespace(secType="BAG", symbol="SPY"), exit_order)
    exit_oid = exit_trade.order.orderId
    env.executor.register_exit_order(exit_oid, trade_id, "macro_event_flatten",
                                     estimated_pnl=40.0)

    # 13:31:44 — the fast monitor comes around again while the bot is blind
    # to the fill. Pre-R12 this re-placed the same exit (order 261926).
    # The re-trigger gate is the CAS: it must refuse while CLOSING.
    assert env.state.update_trade_status(
        trade_id, TradeStatus.CLOSING,
        from_statuses=[TradeStatus.FILLED, TradeStatus.PARTIAL],
    ) is False, "second exit must be refused while one is in flight"
    assert len(env.ib.placed) == 2  # entry + ONE exit; Monday saw three exits

    # Exit looks stuck (>300s): the executor requests a cancel but keeps
    # tracking — no premature revert that would re-arm the portfolio monitor.
    env.executor._pending_exit_orders[exit_oid].submitted_at -= 301
    await env.executor.check_fills()
    assert exit_oid in env.ib.cancelled_order_ids
    assert exit_oid in env.executor._pending_exit_orders
    assert _trade_row(env, trade_id).status == TradeStatus.CLOSING

    # The cancel LOSES: IB reports the order actually filled at 13:30:44.
    env.ib.resolve(exit_oid, "Filled", avg_price=0.40, filled=1)
    filled, exits = await env.executor.check_fills()
    assert len(exits) == 1, "the late-detected fill books exactly one close"
    assert exits[0]["trade_id"] == trade_id
    assert exits[0]["exit_price"] == pytest.approx(0.40)
    assert _trade_row(env, trade_id).status == TradeStatus.CLOSED
    assert _open_position_row(env, trade_id) is None
    assert exit_oid not in env.executor._pending_exit_orders

    # Terminal is terminal: nothing can double-book or resurrect it.
    assert env.state.close_trade(trade_id, 0.0, 0.0) is False
    assert env.state.update_trade_status(
        trade_id, TradeStatus.FILLED, from_statuses=[TradeStatus.CLOSING]) is False
    _, exits_again = await env.executor.check_fills()
    assert exits_again == []
    assert len(env.ib.placed) == 2  # still: one entry, one exit, ever
