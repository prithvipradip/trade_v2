"""Shape-faithful broker fakes for executor / reconciler / lifecycle tests.

R12 (test-maturity audit): the PROVEN lifecycle pattern is a REAL
TradeExecutor + REAL StateManager on a tmp SQLite file, with only the broker
boundary faked by plain ``SimpleNamespace`` objects that mirror ib_insync's
attribute shapes exactly (Trade.order.orderId, Trade.orderStatus.status,
Position.contract.lastTradeDateOrContractMonth, ...). Full suite runtime for
the lifecycle file is a few seconds.

WARNING — do NOT use MagicMock for broker shapes.
MagicMock auto-creates every attribute, so a typo'd or renamed field
(``orderStatus.stauts``, ``avgFilPrice``) returns a truthy child Mock instead
of raising AttributeError, numeric comparisons like ``filled > 0`` raise (or
worse, pass via __gt__ mocking), and a test can go green against an object
shape the real ib_insync API never produces. SimpleNamespace fails loudly on
any attribute the fake doesn't explicitly define — that loud failure is the
point: when ib_insync's shape and the code's expectation drift, the test
breaks instead of lying.

Conventions mirrored from ib_insync / the AIT execution layer:
- combo (BAG) prices are SIGNED as-defined: negative = the combo nets a
  credit (always-BUY convention).
- ``orderStatus.filled`` / ``remaining`` are CONTRACT counts for options.
- ``ib.trades()`` returns ALL trades this session (open + terminal);
  ``ib.openTrades()`` only the working subset.
"""

from __future__ import annotations

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Trade-shaped fakes
# ---------------------------------------------------------------------------

def _broker_trade(
    order_id: int,
    status: str,
    avg_price: float,
    filled: float,
    remaining: float = 0,
    *,
    total_qty: float | None = None,
    lmt_price: float = 0.0,
    sec_type: str = "BAG",
    symbol: str = "SPY",
    fills: list | None = None,
) -> SimpleNamespace:
    """An ib_insync ``Trade`` lookalike as reported by ``ib.trades()``.

    ``status`` uses IBKR's spelling: "Submitted", "PreSubmitted", "Filled",
    "Cancelled", "ApiCancelled", "Inactive". ``avg_price`` is the SIGNED
    as-defined price for BAGs (negative = credit collected).
    """
    total = total_qty if total_qty is not None else filled + remaining
    return SimpleNamespace(
        order=SimpleNamespace(
            orderId=order_id,
            permId=order_id * 1000 + 7,
            totalQuantity=total,
            lmtPrice=lmt_price,
            action="BUY",
            orderType="LMT",
        ),
        orderStatus=SimpleNamespace(
            status=status,
            filled=filled,
            remaining=remaining,
            avgFillPrice=avg_price,
        ),
        contract=SimpleNamespace(secType=sec_type, symbol=symbol),
        fills=list(fills or []),
        log=[],
    )


def _open_trade(
    order_id: int,
    *,
    total_qty: float = 1,
    lmt_price: float = 0.0,
    sec_type: str = "BAG",
    symbol: str = "SPY",
) -> SimpleNamespace:
    """A WORKING order (status Submitted, nothing filled) — the shape
    ``ib.openTrades()`` returns while an order rests on the book."""
    return _broker_trade(
        order_id,
        "Submitted",
        avg_price=0.0,
        filled=0,
        remaining=total_qty,
        total_qty=total_qty,
        lmt_price=lmt_price,
        sec_type=sec_type,
        symbol=symbol,
    )


def _option_position(
    symbol: str,
    strike: float,
    right: str,
    expiry: str,
    qty: float,
    avg_cost: float = 0.0,
) -> SimpleNamespace:
    """An ib_insync ``Position`` lookalike (``ib.positions()`` row) for an
    option leg. ``expiry`` uses IBKR's YYYYMMDD wire format."""
    return SimpleNamespace(
        contract=SimpleNamespace(
            secType="OPT",
            symbol=symbol,
            strike=strike,
            right=right,
            lastTradeDateOrContractMonth=expiry,
        ),
        position=qty,
        avgCost=avg_cost,
    )


# ---------------------------------------------------------------------------
# IB / IBKRClient fakes
# ---------------------------------------------------------------------------

class _Event:
    """Minimal ib_insync-style event: supports ``+=`` / ``-=`` / emit."""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        if handler in self.handlers:
            self.handlers.remove(handler)
        return self

    def emit(self, *args) -> None:
        for h in list(self.handlers):
            h(*args)


class FakeIB:
    """Stands in for ``ib_insync.IB`` behind IBKRClient.ib.

    State lives in three plain lists the test mutates directly:
    - ``open_trades``: what ``openTrades()`` returns (working orders)
    - ``all_trades``:  what ``trades()`` returns (session superset)
    - ``positions_list`` / ``portfolio_list``: broker positions/portfolio

    ``reqMktData`` RAISES by default so the executor's live-quote validation
    block (GOV-1, with its 1.5 s settle sleep) is skipped instantly; set
    ``quote = (bid, ask)`` to exercise it.
    """

    def __init__(self) -> None:
        self.open_trades: list = []
        self.all_trades: list = []
        self.positions_list: list = []
        self.portfolio_list: list = []
        self.cancelled_order_ids: list[int] = []
        self.placed: list = []          # every (contract, order) passed to placeOrder
        self.quote: tuple | None = None  # (bid, ask) to enable market data
        self.orderStatusEvent = _Event()
        self.disconnectedEvent = _Event()
        self._next_order_id = 1000

    # -- orders ------------------------------------------------------------
    def placeOrder(self, contract, order):  # noqa: N802 - ib_insync casing
        self.placed.append((contract, order))
        if not getattr(order, "orderId", 0):
            self._next_order_id += 1
            order.orderId = self._next_order_id
        # A modification (same orderId already working) keeps the existing
        # Trade object, mirroring ib_insync.
        for t in self.open_trades:
            if t.order.orderId == order.orderId:
                return t
        trade = _open_trade(
            order.orderId,
            total_qty=getattr(order, "totalQuantity", 1),
            lmt_price=getattr(order, "lmtPrice", 0.0),
            sec_type=getattr(contract, "secType", "BAG"),
            symbol=getattr(contract, "symbol", "SPY"),
        )
        # Keep the REAL order object attached so lmtPrice mutations by the
        # reprice ladder are observable on the same instance.
        trade.order = order
        self.open_trades.append(trade)
        self.all_trades.append(trade)
        return trade

    def cancelOrder(self, order) -> None:  # noqa: N802
        self.cancelled_order_ids.append(order.orderId)
        for t in list(self.open_trades):
            if t.order.orderId == order.orderId:
                t.orderStatus.status = "Cancelled"
                self.open_trades.remove(t)

    def trades(self) -> list:
        return list(self.all_trades)

    def openTrades(self) -> list:  # noqa: N802
        return list(self.open_trades)

    # -- market data (default: unavailable, so GOV-1 validation no-ops) -----
    def reqMktData(self, *a, **k):  # noqa: N802
        if self.quote is None:
            raise RuntimeError("FakeIB: no market data (set .quote to enable)")
        return SimpleNamespace(bid=self.quote[0], ask=self.quote[1])

    def ticker(self, contract):
        if self.quote is None:
            return None
        return SimpleNamespace(bid=self.quote[0], ask=self.quote[1])

    def cancelMktData(self, contract) -> None:  # noqa: N802
        pass

    # -- account -------------------------------------------------------------
    def positions(self) -> list:
        return list(self.positions_list)

    def portfolio(self) -> list:
        return list(self.portfolio_list)

    def isConnected(self) -> bool:  # noqa: N802
        return True

    # -- test helpers --------------------------------------------------------
    def resolve(self, order_id: int, status: str, avg_price: float,
                filled: float, remaining: float = 0) -> SimpleNamespace:
        """Move a working order to a terminal state: drop it from
        openTrades() and rewrite its trades() row in place."""
        for t in list(self.open_trades):
            if t.order.orderId == order_id:
                self.open_trades.remove(t)
        for i, t in enumerate(self.all_trades):
            if t.order.orderId == order_id:
                resolved = _broker_trade(
                    order_id, status, avg_price, filled, remaining,
                    total_qty=getattr(t.order, "totalQuantity", None),
                    lmt_price=getattr(t.order, "lmtPrice", 0.0),
                    sec_type=getattr(t.contract, "secType", "BAG"),
                    symbol=getattr(t.contract, "symbol", "SPY"),
                )
                self.all_trades[i] = resolved
                return resolved
        resolved = _broker_trade(order_id, status, avg_price, filled, remaining)
        self.all_trades.append(resolved)
        return resolved


class FakeIBKRClient:
    """Stands in for ``ait.broker.ibkr_client.IBKRClient``.

    Only the surface the executor/reconciler actually touches is defined —
    anything else raises AttributeError (see module docstring for why that
    is deliberate).
    """

    def __init__(self, ib: FakeIB | None = None, connected: bool = True) -> None:
        self.ib = ib or FakeIB()
        self.connected = connected
        self.cancelled: list[int] = []
        self._next_con_id = 100
        # R12 F3.1 shape parity: orderIds owned by another clientId session
        # (invisible to this session's trades()/openTrades()).
        self.foreign_open_order_ids: set[int] = set()
        # Set True to mirror a session that connected on a fallback clientId.
        self.on_fallback_client_id = False

    # -- connection ----------------------------------------------------------
    async def ensure_connected(self) -> bool:
        return self.connected

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

    # -- contracts -----------------------------------------------------------
    async def qualify_contract(self, contract):
        if not self.connected:
            return None
        self._next_con_id += 1
        contract.conId = self._next_con_id
        return contract

    async def qualify_contracts_batch(self, contracts: list) -> list:
        if not self.connected:
            return [None] * len(contracts)
        out = []
        for c in contracts:
            self._next_con_id += 1
            c.conId = self._next_con_id
            out.append(c)
        return out

    # -- orders ----------------------------------------------------------------
    async def place_order(self, contract, order):
        if not self.connected:
            return None
        return self.ib.placeOrder(contract, order)

    async def cancel_order(self, trade_or_id) -> bool:
        order_id = (
            trade_or_id if isinstance(trade_or_id, int)
            else trade_or_id.order.orderId
        )
        self.cancelled.append(order_id)
        for t in self.ib.openTrades():
            if t.order.orderId == order_id:
                self.ib.cancelOrder(t.order)
                return True
        return False

    # -- queries (mirror the real client: [] when disconnected) ----------------
    def get_all_trades(self) -> list:
        if not self.connected:
            return []
        return self.ib.trades()

    def get_open_orders(self) -> list:
        if not self.connected:
            return []
        return self.ib.openTrades()

    def get_positions(self) -> list:
        if not self.connected:
            return []
        return self.ib.positions()

    def get_portfolio(self) -> list:
        if not self.connected:
            return []
        return self.ib.portfolio()
