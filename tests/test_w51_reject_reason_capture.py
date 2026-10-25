"""W51 (model-vs-reality-margin-model-unimplemented, reject-reason capture
half): entries rejected by the broker (a real margin reject, code 201; a
read-only-account reject, 321; no-security-definition, 200) used to collapse
into the same generic CANCELLED status as a routine timeout — indistinguishable
in the trades table. `docs/archive/GAP_AUDIT_R7.md:514` named this "SOON" six weeks
before this wave; the R23 register's `model-vs-reality-margin-model-unimplemented`
finding named it again.

Re-verified against HEAD this session (probe pass, see PLAN.md's W51 entry):
`_ORDER_ERROR_CODES`, `_on_error` and `order_error_messages` all already
existed before this wave (added W33, for a different purpose — the exit-side
price-band parser) — 201 was already tracked, 321/200 were not, and the only
accessor threw the code away and had exactly one caller, on the EXIT path
only. This wave adds 321/200 to the tracked set, a code-preserving accessor
(`order_error_codes`) alongside the existing text-only one, and wires it into
the entry-cancelled path (`_update_trade_cancelled`) plus two new `trades`
columns (`reject_code`, `reject_text`).

Drives the real ib_insync `Wrapper` -> `errorEvent` -> `IBKRClient` ->
`TradeExecutor.check_fills()` -> `StateManager`, the same end-to-end style
`test_w33_exit_pricing_corrective.py` uses for the exit side (real wire, not
a mock). No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ib_insync import Contract, Order, OrderStatus, Trade

from ait.bot.state import StateManager
from ait.broker.ibkr_client import IBKRClient
from ait.execution.executor import PendingOrder, TradeExecutor


def _client() -> IBKRClient:
    """Same idiom as test_w33_exit_pricing_corrective.py's _client()."""
    c = IBKRClient(SimpleNamespace(host="127.0.0.1", port=4002, client_id=1,
                                   account="DU0", timeout=5, readonly=False))
    c._connected = True
    c._ib.isConnected = lambda: True
    return c


def _working_order(client: IBKRClient, order_id: int, action: str = "BUY") -> Trade:
    """A Trade exactly as placeOrder leaves it in the wrapper's cache —
    same idiom as test_w33_exit_pricing_corrective.py's _working_order."""
    w = client.ib.wrapper
    order = Order(orderId=order_id, clientId=w.clientId, action=action,
                  orderType="LMT", lmtPrice=1.10, totalQuantity=1)
    trade = Trade(Contract(secType="BAG", symbol="SPY"), order,
                  OrderStatus(orderId=order_id, status="Submitted"), [], [])
    w.trades[w.orderKey(order.clientId, order.orderId, order.permId)] = trade
    return trade


def _broker_rejects(client: IBKRClient, order_id: int, code: int, text: str) -> None:
    """What the wire delivers for an order-killing reject: the error, then
    Cancelled — same idiom as test_w33's _broker_rejects, generalised to any
    code instead of hardcoded 202."""
    w = client.ib.wrapper
    w.error(order_id, code, text, "")
    w.orderStatus(order_id, "Cancelled", 0, 1, 0.0, 0, 0, 0.0, w.clientId,
                  "", 0.0)


def _seed_pending_trade(st: StateManager, trade_id: str) -> None:
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, "SPY", "iron_condor", "short", "pending",
             datetime.now().isoformat(), 1.90))


class TestOrderErrorCodesAccessor:
    """The direct proof: order_error_codes preserves the code that
    order_error_messages (unchanged, still used by the W33 exit path)
    already discarded."""

    def test_the_code_is_preserved_alongside_the_text(self):
        c = _client()
        _working_order(c, 77)
        _broker_rejects(c, 77, 201, "Order rejected - margin")
        assert c.order_error_codes(77) == [(201, "Order rejected - margin")]
        assert c.order_error_messages(77) == ["Order rejected - margin"], (
            "the existing text-only accessor must be unaffected"
        )

    @pytest.mark.parametrize("code,text", [
        (321, "Error validating request - read-only account"),
        (200, "No security definition has been found for the request"),
    ])
    def test_the_newly_tracked_codes_are_captured(self, code, text):
        """321 and 200 were not in _ORDER_ERROR_CODES before this wave —
        reproduced against pre-fix source via git stash (see PLAN.md)."""
        c = _client()
        _working_order(c, 78)
        _broker_rejects(c, 78, code, text)
        assert c.order_error_codes(78) == [(code, text)]

    def test_untracked_codes_are_still_not_captured(self):
        """CONTROL: an untracked code (feed-status chatter) must not start
        being captured just because the allowlist grew by two entries."""
        c = _client()
        c.ib.wrapper.error(-1, 2104, "Market data farm connection is OK", "")
        assert c.order_error_codes(-1) == []


class TestEntryRejectReasonReachesTheTradeRow:
    """wire -> Wrapper.error -> errorEvent -> client -> check_fills ->
    trades.reject_code/reject_text. Before this wave the entry-cancelled path
    consulted nothing — a 201 margin reject was indistinguishable from a
    plain timeout, both just 'trade_cancelled' with no reason."""

    def test_a_margin_reject_is_stamped_on_the_trade_row(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed_pending_trade(st, "T-201")
        c = _client()
        ex = TradeExecutor(c, st, MagicMock())
        _working_order(c, 77)
        ex._pending_orders[77] = PendingOrder(
            trade_id="T-201", signal=MagicMock(), contracts=1)
        _broker_rejects(c, 77, 201, "Order rejected - margin")

        asyncio.run(ex.check_fills())

        trade = st.get_trade_by_id("T-201")
        assert trade.status.value == "cancelled"
        assert trade.reject_code == 201
        assert trade.reject_text == "Order rejected - margin"

    def test_a_cancel_with_no_captured_error_leaves_the_defaults(self, tmp_path):
        """CONTROL: a plain timeout/cancel with nothing captured by
        errorEvent must not fabricate a reason — columns stay at default,
        matching every trade in the current track record."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed_pending_trade(st, "T-NOERR")
        c = _client()
        ex = TradeExecutor(c, st, MagicMock())
        _working_order(c, 79)
        ex._pending_orders[79] = PendingOrder(
            trade_id="T-NOERR", signal=MagicMock(), contracts=1)
        w = c.ib.wrapper
        w.orderStatus(79, "Cancelled", 0, 1, 0.0, 0, 0, 0.0, w.clientId, "", 0.0)

        asyncio.run(ex.check_fills())

        trade = st.get_trade_by_id("T-NOERR")
        assert trade.status.value == "cancelled"
        assert trade.reject_code == 0
        assert trade.reject_text == ""

    def test_a_fake_ibkr_client_without_the_accessor_does_not_break(self, tmp_path):
        """CONTROL: FakeIBKRClient (used throughout the rest of the executor
        suite) has no order_error_codes — the getattr+callable guard must
        degrade to a no-op, not raise, or every other executor test using it
        would need updating for this wave alone."""
        from tests.fakes import FakeIB, FakeIBKRClient

        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed_pending_trade(st, "T-FAKE")
        fake = FakeIBKRClient(FakeIB())
        ex = TradeExecutor(fake, st, MagicMock())

        assert ex._update_trade_cancelled(
            PendingOrder(trade_id="T-FAKE", signal=MagicMock(), contracts=1),
            order_id=999,
        ) is True
        trade = st.get_trade_by_id("T-FAKE")
        assert trade.status.value == "cancelled"
        assert trade.reject_code == 0
        assert trade.reject_text == ""
