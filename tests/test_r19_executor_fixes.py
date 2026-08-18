"""R19 executor fixes — the audit findings whose file is executor.py.

Every test asserts the POST-fix contract and FAILS against the pre-fix code
by construction (verified by re-running this file against a scratch copy of
executor.py with each fix reverted):

[1] HIGH  an entry that FILLED during a disconnect was booked CANCELLED.
    Every "did this order die?" question was answered with
    `t.order.orderId == order_id` against THIS session's Trade cache, and
    ib_insync 0.9.86 wipes that cache on any disconnect
    (wrapper.connectionClosed -> reset()). Orders recovered afterwards via
    reqCompletedOrders routinely carry order.orderId = 0 (IBKR keys completed
    orders by permId), so the match failed either way: the PENDING row flipped
    to CANCELLED — a status neither the portfolio monitor nor the reconciler
    re-adopts — while four condor legs were LIVE at the broker.
[2] MED   the cancel CAS from-set included PARTIAL, so a partially filled
    entry could be flipped to CANCELLED, orphaning the filled contracts and
    their open_positions row.
[3] MED   _reconstruct_bag_price divided the net premium (summed over ACTUAL
    fills) by the ORDERED quantity, so a partial BAG fill reconstructed at
    filled/ordered of the true price — and that price is booked as
    entry_price / exit price, poisoning every downstream P&L.
[4] MED   the executions ledger attributed every fill via t.order.orderId, so
    reconnect-recovered orders (orderId 0) wrote rows with order_id 0 and
    trade_id '' — losing commission/P&L attribution for exactly the fills
    around a connection loss.
[5] MED   two cancel paths were DEAD: the 30s partial-remainder cancel and the
    >300s stale-pending exit cancel both searched `open_trades` for an order
    that, by their own branch precondition, is never in open_trades. They
    logged cancellations that were never transmitted.
[6] HIGH  the single-leg wide-spread reject hardcoded 0.15, shadowing the
    EXISTING options.max_bid_ask_spread_pct config field (config.yaml = 0.40).

Pattern notes: see tests/fakes.py — real TradeExecutor + real StateManager on
a tmp SQLite file, only the broker boundary faked with SimpleNamespace shapes
(never MagicMock for broker objects: it invents attributes and hides drift).
The fakes here EXTEND tests/fakes.py with the surface R19 needs: ib.fills()
(ib_insync's execution cache, which outlives the Trade cache) and the R16
all-clients open-order probe.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from types import SimpleNamespace

import pytest

from ait.bot.state import StateManager, TradeStatus
from ait.config.settings import OptionsConfig
from ait.data.options_chain import OptionContract
from ait.execution.executor import DEFAULT_MAX_SPREAD_PCT, TradeExecutor
from ait.risk.circuit_breaker import CircuitBreaker
from ait.strategies.base import Signal, SignalDirection

from tests.fakes import FakeIB, FakeIBKRClient, _broker_trade

EXPIRY_ISO = "2026-08-21"
BAG_CON_ID = 28812380  # IBKR's generic combo conId (protocol constant)


# ---------------------------------------------------------------------------
# Broker fakes: executions + the all-clients probe
# ---------------------------------------------------------------------------

def _execution(exec_id: str, order_id: int, perm_id: int, side: str,
               price: float, shares: float = 1.0, con_id: int = 700001,
               sec_type: str = "OPT", commission: float = 0.65):
    """An ib_insync ``Fill`` lookalike as ``ib.fills()`` returns it.

    Execution carries BOTH keys — orderId and permId — which is the whole
    point of R19 #1: they survive on the execution even when the parent
    (completed-order) Trade reports orderId 0.
    """
    return SimpleNamespace(
        execution=SimpleNamespace(
            execId=exec_id, orderId=order_id, permId=perm_id, side=side,
            shares=shares, price=price, time="2026-08-17T14:51:37+00:00"),
        commissionReport=SimpleNamespace(commission=commission, realizedPNL=0.0),
        contract=SimpleNamespace(conId=con_id, secType=sec_type, symbol="SPY"),
    )


def _condor_executions(order_id: int, perm_id: int, *, units: float = 1.0,
                       bag_price: float | None = -4.31) -> list:
    """The five executions IBKR reports for one filled SPY condor unit: four
    leg rows plus the BAG summary row carrying the net combo price."""
    fills = [
        _execution("R19.01", order_id, perm_id, "SLD", 2.09, units, 792205417),
        _execution("R19.02", order_id, perm_id, "SLD", 3.01, units, 792205363),
        _execution("R19.03", order_id, perm_id, "BOT", 0.19, units, 792205400),
        _execution("R19.04", order_id, perm_id, "BOT", 0.60, units, 792205377),
    ]
    if bag_price is not None:
        fills.append(_execution("R19.05", order_id, perm_id, "BOT", bag_price,
                                units, BAG_CON_ID, sec_type="BAG",
                                commission=0.0))
    return fills


class ExecutionAwareIB(FakeIB):
    """FakeIB + ib_insync's execution cache (``ib.fills()``).

    ``fills_list`` is populated independently of ``all_trades`` — that is
    exactly the post-reconnect state R19 #1 is about: wrapper.reset() empties
    the Trade cache, connectAsync then re-requests executions
    (reqExecutionsAsync), so the fills come back while the Trade that produced
    them does not.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fills_list: list = []
        self.cancel_raises = False

    def fills(self) -> list:
        return list(self.fills_list)

    def cancelOrder(self, order) -> None:  # noqa: N802 - ib_insync casing
        if self.cancel_raises:
            raise RuntimeError("FakeIB: cancel rejected by the gateway")
        # ib_insync only needs order.orderId on the wire; a bare Order with no
        # matching cached Trade is legal (it logs 'Unknown orderId' locally).
        super().cancelOrder(order)


class BrokerAwareClient(FakeIBKRClient):
    """FakeIBKRClient + the R16 authoritative all-clients open-order probe.

    ``broker_open_ids`` is the BROKER's own answer: a set means "these ids are
    working under some clientId"; None means the broker could not be asked.
    """

    def __init__(self, ib=None, connected: bool = True,
                 client_id: int = 105) -> None:
        super().__init__(ib, connected)
        self.client_id = client_id
        self.broker_open_ids: set[int] | None = set()

    async def get_all_open_order_ids(self, timeout: float = 8.0) -> set[int] | None:
        if self.broker_open_ids is None:
            return None
        return set(self.broker_open_ids)


@pytest.fixture
def env(tmp_path, monkeypatch, risk_config):
    """Real executor + real state on a tmp DB; broker boundary faked."""
    monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
    monkeypatch.setenv("AIT_ENTRY_LADDER_START", "0.25")
    state = StateManager(tmp_path / "state.db")
    ib = ExecutionAwareIB()
    client = BrokerAwareClient(ib)
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


def _ic_signal(credit: float = 4.31, width: float = 10.0) -> Signal:
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


def _long_call_signal(bid: float, ask: float) -> Signal:
    """Single-leg DEBIT entry — the only path through the spread gate."""
    contract = OptionContract(
        symbol="SPY", expiry=date(2026, 8, 21), strike=100.0, right="C",
        bid=bid, ask=ask, last=(bid + ask) / 2, volume=100, open_interest=100,
        implied_vol=0.20, con_id=700123,
    )
    return Signal(
        symbol="SPY",
        strategy_name="long_call",
        direction=SignalDirection.BULLISH,
        confidence=0.70,
        contract=contract,
        action="BUY",
        entry_price=(bid + ask) / 2,
        max_loss=ask * 100,
        max_profit=1000.0,
        expiry=date(2026, 8, 21),
    )


async def _place(env, signal: Signal, contracts: int = 1) -> tuple[str, int]:
    trade_id = await env.executor.execute_signal(signal, contracts)
    assert trade_id is not None, "execute_signal refused the order"
    order_id = next(iter(env.executor._pending_orders))
    return trade_id, order_id


def _row(env, trade_id: str):
    return env.state.get_trade_by_id(trade_id)


def _open_position_row(env, trade_id: str):
    with sqlite3.connect(env.db) as conn:
        return conn.execute(
            "SELECT quantity, entry_price FROM open_positions WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()


def _wipe_session_caches(env, keep_unrelated: bool = True) -> None:
    """Reproduce ib_insync's wrapper.connectionClosed() -> reset(): the Trade
    cache is emptied. `keep_unrelated` leaves ONE unrelated trade behind so
    the R16 empty-snapshot guard (which defers on a wholly empty cache) is
    satisfied and the pre-fix code reaches its cancel verdict — i.e. the
    single-order fill-during-outage variant R16 did not cover."""
    env.ib.open_trades.clear()
    env.ib.all_trades.clear()
    if keep_unrelated:
        env.ib.all_trades.append(
            _broker_trade(999999, "Filled", avg_price=1.23, filled=1))


# ===========================================================================
# [1] HIGH — an entry filled during a disconnect must never book CANCELLED
# ===========================================================================

class TestFillDuringDisconnectIsNotCancelled:

    async def test_fill_recovered_by_execution_order_id(self, env):
        trade_id, order_id = await _place(env, _ic_signal(credit=4.31))
        assert _row(env, trade_id).status == TradeStatus.PENDING

        # The condor FILLS while the connection is down. On reconnect the
        # Trade cache is empty but the executions came back.
        _wipe_session_caches(env)
        env.ib.fills_list = _condor_executions(order_id, perm_id=0)
        env.executor._pending_orders[order_id].submitted_at -= 60
        env.client.broker_open_ids = set()  # broker: not working (it filled)

        filled, _ = await env.executor.check_fills()

        row = _row(env, trade_id)
        assert row.status == TradeStatus.FILLED, (
            "an order with broker executions must never be booked CANCELLED — "
            "the four legs are LIVE and CANCELLED is a status nothing re-adopts"
        )
        assert trade_id in filled
        assert row.entry_price == pytest.approx(4.31), (
            "the fill price must be reconstructed from the executions, not "
            "guessed from the signal"
        )
        pos = _open_position_row(env, trade_id)
        assert pos is not None and pos[0] == 1, "the position must be MANAGED"
        assert order_id not in env.executor._pending_orders

    async def test_fill_recovered_by_perm_id_when_order_id_is_zero(self, env):
        """The IBKR completed-order quirk: reqCompletedOrders rows are keyed by
        permId and report orderId 0, so orderId matching can never succeed."""
        trade_id, order_id = await _place(env, _ic_signal(credit=4.31))

        # IBKR assigns the permId once the order is acknowledged; the executor
        # learns it on a normal pass while the order is still visible.
        env.ib.open_trades[0].order.permId = 777001
        await env.executor.check_fills()
        assert env.executor._order_perm_map.get(order_id) == 777001, (
            "the permId must be captured while the session can still see the "
            "order — after the cache wipe there is no other route back to it"
        )

        # Disconnect: cache wiped, and the recovered completed order carries
        # orderId 0 (so even the recovered Trade cannot be matched by id).
        _wipe_session_caches(env, keep_unrelated=False)
        recovered = _broker_trade(0, "Filled", avg_price=-4.31, filled=1)
        recovered.order.permId = 777001
        env.ib.all_trades.append(recovered)
        env.ib.fills_list = _condor_executions(0, perm_id=777001)
        env.executor._pending_orders[order_id].submitted_at -= 60
        env.client.broker_open_ids = set()

        filled, _ = await env.executor.check_fills()

        assert _row(env, trade_id).status == TradeStatus.FILLED
        assert trade_id in filled
        assert _open_position_row(env, trade_id) is not None

    async def test_confirm_order_not_working_refuses_when_fills_exist(self, env):
        """The R16 gate answers "is it working?" — but a FILLED order is also
        not working. With executions present it must refuse the cancel."""
        _, order_id = await _place(env, _ic_signal())
        env.ib.fills_list = _condor_executions(order_id, perm_id=0)
        env.client.broker_open_ids = set()

        assert await env.executor._confirm_order_not_working(
            order_id, env.ib.trades()) is False

    async def test_genuinely_dead_order_still_books_cancelled(self, env):
        """Positive control: no executions anywhere, broker answers, session
        cache healthy — the cancel verdict must still be issued."""
        trade_id, order_id = await _place(env, _ic_signal())
        _wipe_session_caches(env)  # order gone, one unrelated trade remains
        env.ib.fills_list = []
        env.executor._pending_orders[order_id].submitted_at -= 60
        env.client.broker_open_ids = set()

        filled, _ = await env.executor.check_fills()

        assert filled == []
        assert _row(env, trade_id).status == TradeStatus.CANCELLED
        assert _open_position_row(env, trade_id) is None


# ===========================================================================
# [2] MEDIUM — a cancel verdict must never flip a PARTIAL row
# ===========================================================================

class TestCancelVerdictNeverOrphansPartialContracts:

    async def _partial_then_amnesia(self, env) -> tuple[str, int]:
        trade_id, order_id = await _place(env, _ic_signal(credit=4.31),
                                          contracts=2)
        # Pass 1: 1 of 2 combo units filled, order still alive at IBKR
        # (PendingCancel is non-terminal, so the tracker is retained).
        env.ib.resolve(order_id, "PendingCancel", avg_price=-4.31,
                       filled=1, remaining=1)
        await env.executor.check_fills()
        assert _row(env, trade_id).status == TradeStatus.PARTIAL
        assert _open_position_row(env, trade_id)[0] == 1

        # Pass 2: connection blip — the order is now invisible in every cache
        # and the broker says it is no longer working.
        _wipe_session_caches(env)
        env.executor._pending_orders[order_id].submitted_at -= 60
        env.client.broker_open_ids = set()
        return trade_id, order_id

    async def test_partial_row_is_not_flipped_to_cancelled(self, env):
        trade_id, order_id = await self._partial_then_amnesia(env)

        filled, _ = await env.executor.check_fills()

        row = _row(env, trade_id)
        assert row.status != TradeStatus.CANCELLED, (
            "the partially filled contracts are LIVE at the broker; CANCELLED "
            "orphans them in a status nothing re-adopts"
        )
        assert row.status == TradeStatus.FILLED
        assert row.quantity == 1, "the retained quantity is what actually filled"
        assert _open_position_row(env, trade_id)[0] == 1
        assert trade_id in filled, "the live contracts must be handed back as a position"

    async def test_pure_pending_row_still_cancels(self, env):
        """Positive control: the narrowing must not stop a real cancel."""
        trade_id, order_id = await _place(env, _ic_signal())
        _wipe_session_caches(env)
        env.executor._pending_orders[order_id].submitted_at -= 60
        env.client.broker_open_ids = set()

        await env.executor.check_fills()

        assert _row(env, trade_id).status == TradeStatus.CANCELLED


# ===========================================================================
# [3] MEDIUM — _reconstruct_bag_price must divide by the FILLED quantity
# ===========================================================================

def _bag_trade(ordered: float, filled: float, leg_shares: float,
               avg_price: float = 0.0) -> SimpleNamespace:
    """A BAG trade whose per-leg executions net a 6.12 credit per combo unit
    (the audit's own reproduction: 2.93 + 4.03 sold, 0.72 + 0.12 bought)."""
    fills = [
        _execution("B.01", 4242, 0, "SLD", 2.93, leg_shares),
        _execution("B.02", 4242, 0, "SLD", 4.03, leg_shares),
        _execution("B.03", 4242, 0, "BOT", 0.72, leg_shares),
        _execution("B.04", 4242, 0, "BOT", 0.12, leg_shares),
    ]
    return SimpleNamespace(
        order=SimpleNamespace(orderId=4242, permId=0, totalQuantity=ordered),
        orderStatus=SimpleNamespace(status="Cancelled", filled=filled,
                                    remaining=ordered - filled,
                                    avgFillPrice=avg_price),
        contract=SimpleNamespace(secType="BAG", symbol="SPY"),
        fills=fills,
        log=[],
    )


class TestBagPriceUsesFilledQuantity:

    def test_partial_bag_fill_reconstructs_the_true_per_combo_price(self):
        # 1 of 2 combo units filled: the net premium was collected ONCE, so
        # the per-combo price is 6.12 — dividing by the ORDERED 2 reported
        # -3.06, exactly half the real credit.
        price = TradeExecutor._reconstruct_bag_price(
            _bag_trade(ordered=2, filled=1, leg_shares=1))
        assert price == pytest.approx(-6.12), (
            "net premium is summed over ACTUAL fills, so it must be divided "
            "by the FILLED quantity"
        )

    def test_full_fill_is_unchanged(self):
        # Regression guard: with nothing partial, ordered == filled.
        price = TradeExecutor._reconstruct_bag_price(
            _bag_trade(ordered=2, filled=2, leg_shares=2))
        assert price == pytest.approx(-6.12)

    def test_entry_fill_price_uses_the_corrected_reconstruction(self, env):
        # The value flows into trades.entry_price via _get_fill_price's
        # avgFillPrice-missing fallback, so the error would poison every
        # downstream P&L, stop and target.
        pending = SimpleNamespace(trade_id="T-BAG", contracts=2,
                                  signal=SimpleNamespace(entry_price=6.12))
        price = env.executor._get_fill_price(
            4242, [_bag_trade(ordered=2, filled=1, leg_shares=1)], pending)
        assert price == pytest.approx(-6.12)


# ===========================================================================
# [4] MEDIUM — ledger attribution must survive orderId 0
# ===========================================================================

def _recovered_bag_trade(perm_id: int, exec_order_id: int,
                         exec_perm_id: int) -> SimpleNamespace:
    """A completed order as ib_insync re-creates it after a reconnect: the
    Trade reports orderId 0 (IBKR keys completed orders by permId)."""
    return SimpleNamespace(
        order=SimpleNamespace(orderId=0, permId=perm_id, totalQuantity=1),
        orderStatus=SimpleNamespace(status="Filled", filled=1, remaining=0,
                                    avgFillPrice=-4.31),
        contract=SimpleNamespace(secType="BAG", symbol="SPY"),
        fills=_condor_executions(exec_order_id, exec_perm_id),
        log=[],
    )


def _sweep_rows(trade, *, order_trade_map: dict,
                perm_trade_map: dict | None = None) -> list[dict]:
    rows: list[dict] = []
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = SimpleNamespace(record_execution=lambda **kw: rows.append(kw))
    ex._ibkr = SimpleNamespace(ib=SimpleNamespace(trades=lambda: [trade]))
    ex._order_trade_map = dict(order_trade_map)
    ex._pending_orders = {}
    ex._pending_exit_orders = {}
    ex._order_ctx_map = {}
    ex._order_perm_map = {}
    ex._perm_trade_map = dict(perm_trade_map or {})
    ex._sweep_executions()
    return rows


class TestLedgerAttributionSurvivesReconnect:

    def test_execution_order_id_keys_the_row_when_the_trade_reports_zero(self):
        rows = _sweep_rows(
            _recovered_bag_trade(perm_id=777001, exec_order_id=4242,
                                 exec_perm_id=777001),
            order_trade_map={4242: "T-SPY"})

        assert len(rows) == 5
        assert {r["order_id"] for r in rows} == {4242}, (
            "the ledger must key fills by the execution's true orderId, not "
            "the recovered order's 0"
        )
        assert {r["trade_id"] for r in rows} == {"T-SPY"}, (
            "commission/P&L attribution is lost when trade_id comes back ''"
        )
        assert {r["perm_id"] for r in rows} == {777001}

    def test_perm_id_map_attributes_fills_with_no_usable_order_id(self):
        # Fully permId-keyed: neither the Trade nor the executions carry an
        # orderId. The placement-time permId map is the only link left.
        rows = _sweep_rows(
            _recovered_bag_trade(perm_id=777001, exec_order_id=0,
                                 exec_perm_id=777001),
            order_trade_map={}, perm_trade_map={777001: "T-SPY"})

        assert len(rows) == 5
        assert {r["trade_id"] for r in rows} == {"T-SPY"}
        assert {r["perm_id"] for r in rows} == {777001}

    def test_normal_orders_are_unchanged(self):
        # Regression guard for the R16 semantics: healthy trade, healthy ids.
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=4242, permId=777001, totalQuantity=1),
            orderStatus=SimpleNamespace(status="Filled", filled=1, remaining=0,
                                        avgFillPrice=-4.31),
            contract=SimpleNamespace(secType="BAG", symbol="SPY"),
            fills=_condor_executions(4242, 777001),
            log=[])
        rows = _sweep_rows(trade, order_trade_map={4242: "T-SPY"})
        assert {r["order_id"] for r in rows} == {4242}
        assert {r["trade_id"] for r in rows} == {"T-SPY"}
        # R16 semantics preserved: only the BAG row carries combo context and
        # it is stored as a magnitude.
        bag = [r for r in rows if r["con_id"] == BAG_CON_ID]
        assert len(bag) == 1 and bag[0]["price"] == pytest.approx(4.31)


# ===========================================================================
# [5] MEDIUM — the two dead cancel paths must transmit (or page loudly)
# ===========================================================================

class TestCancelPathsAreReal:

    async def test_partial_remainder_cancel_is_transmitted(self, env):
        trade_id, order_id = await _place(env, _ic_signal(), contracts=2)
        # Non-terminal partial, older than 30s. The branch is reachable ONLY
        # when the order is absent from open_trades — which is precisely why
        # the old open_trades loop could never find it.
        env.ib.resolve(order_id, "PendingCancel", avg_price=-4.31,
                       filled=1, remaining=1)
        env.executor._pending_orders[order_id].submitted_at -= 31

        await env.executor.check_fills()

        assert order_id in env.ib.cancelled_order_ids, (
            "the 30s remainder cancel logged a cancellation that was never "
            "transmitted — the remainder rode at a stale price to the 240s "
            "timeout instead"
        )

    async def test_stale_pending_exit_cancel_reaches_a_cache_invisible_order(self, env):
        # A close order still working at the broker but invisible locally
        # (post-reconnect amnesia): present in NO cache, so both the
        # open_trades loop and IBKRClient.cancel_order(int) are no-ops.
        env.executor.register_exit_order(88881, "T-EXIT", "take_profit_50pct",
                                         estimated_pnl=0.0)
        env.executor._pending_exit_orders[88881].submitted_at -= 301
        env.client.broker_open_ids = {88881}

        await env.executor.check_fills()

        assert 88881 in env.ib.cancelled_order_ids, (
            "a cancel must be sent by orderId on the wire; the trade cache "
            "cannot be a precondition for cancelling a live order"
        )
        assert 88881 in env.executor._pending_exit_orders  # still tracked

    async def test_untransmittable_cancel_pages_the_operator_once(self, env):
        env.ib.cancel_raises = True
        env.executor.register_exit_order(88882, "T-EXIT2", "stop_loss",
                                         estimated_pnl=0.0)
        env.executor._pending_exit_orders[88882].submitted_at -= 301

        await env.executor.check_fills()
        await env.executor.check_fills()

        assert (88882, "exit_stale_pending_cancel_unsent") in \
            env.executor._stuck_order_pages, (
            "a safety path that cannot act must say so loudly, once — never "
            "sit silently in the code doing nothing"
        )

    async def test_transmitted_cancel_is_sent_only_once(self, env):
        env.executor.register_exit_order(88883, "T-EXIT3", "stop_loss",
                                         estimated_pnl=0.0)
        env.executor._pending_exit_orders[88883].submitted_at -= 301
        env.client.broker_open_ids = {88883}

        await env.executor.check_fills()
        await env.executor.check_fills()

        assert env.ib.cancelled_order_ids.count(88883) == 1


# ===========================================================================
# [6] HIGH (config) — the spread ceiling comes from config, not a literal
# ===========================================================================

def _executor_with_settings(env, spread_pct: float | None) -> TradeExecutor:
    settings = None
    if spread_pct is not None:
        settings = SimpleNamespace(
            options=OptionsConfig(max_bid_ask_spread_pct=spread_pct))
    return TradeExecutor(
        ibkr_client=env.client, state=env.state,
        circuit_breaker=env.executor._circuit_breaker,
        order_timeout=240, settings=settings)


class TestSpreadGateReadsConfig:

    def test_no_settings_keeps_the_historical_threshold(self, env):
        # Nothing breaks for call sites that do not thread settings in.
        assert env.executor._max_spread_pct == DEFAULT_MAX_SPREAD_PCT == 0.15

    def test_config_value_is_what_the_gate_enforces(self, env):
        ex = _executor_with_settings(env, 0.40)
        assert ex._max_spread_pct == pytest.approx(0.40)

    async def test_loosened_config_admits_a_20pct_spread(self, env):
        # config.yaml sets 0.40 ("stale-quote spreads run wider" on this
        # delayed-data paper account) while the executor vetoed >15% with only
        # a WARNING — the scanner and the executor disagreeing in silence.
        ex = _executor_with_settings(env, 0.40)
        trade_id = await ex.execute_signal(_long_call_signal(bid=1.80, ask=2.20), 1)

        assert trade_id is not None, "a 20% spread is inside the configured 40%"
        assert _row(env, trade_id).status == TradeStatus.PENDING
        assert len(env.ib.placed) == 1

    async def test_tightened_config_rejects_a_12pct_spread(self, env):
        # The literal broke BOTH ways: tightening the config below 0.15 did
        # not tighten the executor either.
        ex = _executor_with_settings(env, 0.10)
        trade_id = await ex.execute_signal(_long_call_signal(bid=1.88, ask=2.12), 1)

        assert trade_id is None, "a 12% spread exceeds the configured 10%"
        assert env.ib.placed == []

    async def test_configured_ceiling_still_rejects_beyond_it(self, env):
        ex = _executor_with_settings(env, 0.40)
        trade_id = await ex.execute_signal(_long_call_signal(bid=1.00, ask=2.00), 1)

        assert trade_id is None, "a 67% spread exceeds even the loosened 40%"
        assert env.ib.placed == []
