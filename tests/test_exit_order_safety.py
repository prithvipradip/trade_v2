"""R14 Tier-1 exit-order safety — the two live-exposure fixes.

1. Exit-price bound (_close_multi_leg): the exit path had NO price sanity
   bound — `ask + cross` with no ceiling, and a MARKET order on a 4-leg BAG
   as the no-quote fallback. Executed proof: a $2-wide condor with a garbage
   9.90 ask placed BUY LMT 10.00 (5x the structural max loss). Now: credit
   buybacks are capped at the wing width, the no-quote fallback prices AT the
   width instead of going to market, and shapes with no structural bound
   defer instead of market-ordering blind.

2. Broker-position liveness gate (_execute_exit / position_liveness): nothing
   checked the position still EXISTS at IBKR before reversing it. After a
   manual TWS flatten the monitor still demanded the exit and the reverse
   combo REBUILT the position inverted (the 07-13 incident end-state,
   reachable by an operator with no bug at all). The gate is three-state
   because each answer has a different catastrophe:
     - "gone"    → refuse AND book the trade. Refusing alone strands it FILLED
                   forever: reconcile()'s zero-options guard declines to
                   mass-close on exactly this state, so nothing else ever
                   closes the row.
     - "partial" → refuse and page. Reversing all legs of a half-flattened
                   structure OPENS inverted positions on the missing ones.
     - "unknown" → PROCEED. An empty position cache is indistinguishable from
                   a wedged feed, so "gone" is only ever returned after a fresh
                   authoritative broker re-query; anything less must not
                   disable a stop.

Tests drive the REAL methods via __new__-constructed objects with only
collaborators mocked — the same harness pattern as test_credit_exits.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import TradeStatus
from ait.execution.reconciler import PositionReconciler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONDOR_LEGS = [
    {"strike": 713.0, "right": "P", "action": "BUY", "expiry": "2026-07-24"},
    {"strike": 715.0, "right": "P", "action": "SELL", "expiry": "2026-07-24"},
    {"strike": 752.0, "right": "C", "action": "SELL", "expiry": "2026-07-24"},
    {"strike": 754.0, "right": "C", "action": "BUY", "expiry": "2026-07-24"},
]

STRADDLE_LEGS = [
    {"strike": 299.0, "right": "C", "action": "BUY", "expiry": "2026-07-24"},
    {"strike": 299.0, "right": "P", "action": "BUY", "expiry": "2026-07-24"},
]

STRANGLE_LEGS = [
    {"strike": 312.0, "right": "C", "action": "SELL", "expiry": "2026-07-24"},
    {"strike": 286.0, "right": "P", "action": "SELL", "expiry": "2026-07-24"},
]

PUT_SPREAD_LEGS = [
    {"strike": 297.0, "right": "P", "action": "BUY", "expiry": "2026-07-16"},
    {"strike": 292.0, "right": "P", "action": "SELL", "expiry": "2026-07-16"},
]


@dataclass
class FakeTrade:
    trade_id: str = "T-EXIT-SAFETY"
    symbol: str = "SPY"
    strategy: str = "iron_condor"
    contract_type: str = "iron_condor"
    quantity: int = 1
    entry_price: float = 1.00
    status: object = TradeStatus.FILLED
    legs: str = field(default_factory=lambda: json.dumps(CONDOR_LEGS))
    expiry: str | None = "2026-07-24"
    strike: float | None = None


def _quote(bid: float, ask: float) -> SimpleNamespace:
    return SimpleNamespace(bid=bid, ask=ask)


def _orchestrator(
    ticker: SimpleNamespace | None,
    exit_cross: float = 0.05,
) -> tuple[TradingOrchestrator, MagicMock, list[str]]:
    """Real TradingOrchestrator via __new__ with only collaborators mocked.

    Returns (orchestrator, place_order mock, sent notifications).
    """
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    notifications: list[str] = []

    ibkr = MagicMock()
    ibkr.qualify_contract = AsyncMock(
        side_effect=lambda c: SimpleNamespace(conId=1, contract=c)
    )
    ibkr.place_order = AsyncMock(
        return_value=SimpleNamespace(order=SimpleNamespace(orderId=42))
    )
    ibkr.ib.reqMktData = MagicMock()
    ibkr.ib.cancelMktData = MagicMock()
    ibkr.ib.ticker = MagicMock(return_value=ticker)
    orch._ibkr = ibkr

    orch._settings = SimpleNamespace(
        exit=SimpleNamespace(exit_cross_amount=exit_cross)
    )

    async def _notify(msg: str) -> None:
        notifications.append(msg)

    orch._send_notification = _notify
    return orch, ibkr.place_order, notifications


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """_close_multi_leg sleeps 0.5s waiting for the combo quote — skip it."""
    real_sleep = asyncio.sleep

    async def _instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr("ait.bot.orchestrator.asyncio.sleep", _instant)


# ---------------------------------------------------------------------------
# _defined_risk_width — the structural bound itself
# ---------------------------------------------------------------------------

class TestDefinedRiskWidth:
    def test_iron_condor_uses_wider_wing(self):
        legs = [
            {"strike": 710.0, "right": "P", "action": "BUY"},   # 5-wide puts
            {"strike": 715.0, "right": "P", "action": "SELL"},
            {"strike": 752.0, "right": "C", "action": "SELL"},  # 2-wide calls
            {"strike": 754.0, "right": "C", "action": "BUY"},
        ]
        assert TradingOrchestrator._defined_risk_width(legs) == 5.0

    def test_vertical_spread(self):
        assert TradingOrchestrator._defined_risk_width(PUT_SPREAD_LEGS) == 5.0

    def test_straddle_has_no_bound(self):
        assert TradingOrchestrator._defined_risk_width(STRADDLE_LEGS) is None

    def test_strangle_has_no_bound(self):
        assert TradingOrchestrator._defined_risk_width(STRANGLE_LEGS) is None

    def test_calendar_width_is_falsy(self):
        # Same strike twice on one right (calendar): width 0.0 — falsy, so
        # callers treat it exactly like "no structural bound".
        legs = [
            {"strike": 300.0, "right": "C", "action": "SELL", "expiry": "2026-07-17"},
            {"strike": 300.0, "right": "C", "action": "BUY", "expiry": "2026-08-21"},
        ]
        assert not TradingOrchestrator._defined_risk_width(legs)

    def test_malformed_leg_returns_none(self):
        assert TradingOrchestrator._defined_risk_width(
            [{"right": "P", "action": "BUY"}]  # no strike
        ) is None

    def test_three_legged_shape_returns_none(self):
        legs = [
            {"strike": 710.0, "right": "P", "action": "BUY"},
            {"strike": 715.0, "right": "P", "action": "SELL"},
            {"strike": 720.0, "right": "P", "action": "SELL"},
        ]
        assert TradingOrchestrator._defined_risk_width(legs) is None

    def test_empty_legs_returns_none(self):
        assert TradingOrchestrator._defined_risk_width([]) is None


# ---------------------------------------------------------------------------
# _close_multi_leg — pricing matrix {credit, debit, no-width} x {quote state}
# ---------------------------------------------------------------------------

class TestCloseMultiLegPricing:
    async def test_credit_condor_sane_quote_not_capped(self):
        """A normal buyback quote inside the wings passes through untouched."""
        orch, place_order, _ = _orchestrator(_quote(bid=0.85, ask=0.95))
        trade = FakeTrade()  # $2-wide condor
        await orch._close_multi_leg(trade, CONDOR_LEGS)
        order = place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice == pytest.approx(1.00)  # ask 0.95 + cross 0.05

    async def test_credit_condor_garbage_ask_capped_at_wing_width(self):
        """THE executed proof: $2-wide condor, garbage 9.90 ask. Pre-fix this
        placed BUY LMT 10.00 — 5x the structural max loss. Now: capped."""
        orch, place_order, _ = _orchestrator(_quote(bid=0.10, ask=9.90))
        trade = FakeTrade()
        await orch._close_multi_leg(trade, CONDOR_LEGS)
        order = place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice == pytest.approx(2.00)  # AT the wing width

    async def test_deep_itm_condor_cap_stays_marketable(self):
        """The cap must sit AT the width, not below it. A condor whose short
        leg is deep ITM is legitimately worth ~width; an earlier width-0.01
        cap was non-marketable exactly then, so the exit re-placed forever
        while the ITM short carried assignment risk."""
        orch, place_order, _ = _orchestrator(_quote(bid=1.98, ask=2.02))
        trade = FakeTrade()  # $2-wide
        await orch._close_multi_leg(trade, CONDOR_LEGS)
        order = place_order.call_args.args[1]
        assert order.lmtPrice == pytest.approx(2.00)
        assert order.lmtPrice >= 2.00  # marketable against a 2.00 fair value

    async def test_credit_condor_no_quote_prices_at_wing_width_not_market(self):
        """Pre-fix the no-quote fallback was a MARKET order on a 4-leg BAG —
        unbounded fill exactly when quotes evaporate. Now: LMT at the width
        (the max loss the wings already guarantee), operator paged."""
        orch, place_order, notes = _orchestrator(
            _quote(bid=float("nan"), ask=float("nan"))
        )
        trade = FakeTrade()
        result = await orch._close_multi_leg(trade, CONDOR_LEGS)
        assert result is not None
        order = place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice == pytest.approx(2.00)
        assert any("EXIT WITHOUT QUOTES" in n for n in notes)

    async def test_credit_condor_zero_quote_treated_as_no_quote(self):
        """IBKR uses 0 for 'no quote' on combos — must hit the width fallback,
        not place LMT 0.05 (0 + cross) that would rest forever."""
        orch, place_order, notes = _orchestrator(_quote(bid=0.0, ask=0.0))
        trade = FakeTrade()
        await orch._close_multi_leg(trade, CONDOR_LEGS)
        order = place_order.call_args.args[1]
        assert order.lmtPrice == pytest.approx(2.00)
        assert any("EXIT WITHOUT QUOTES" in n for n in notes)

    async def test_debit_spread_no_quote_defers_instead_of_market(self):
        """Debit shape, no quote: refuse to market-order the BAG; return None
        so the caller leaves the trade FILLED and the monitor retries."""
        orch, place_order, notes = _orchestrator(
            _quote(bid=float("nan"), ask=float("nan"))
        )
        trade = FakeTrade(
            strategy="bear_put_spread", contract_type="spread",
            legs=json.dumps(PUT_SPREAD_LEGS),
        )
        result = await orch._close_multi_leg(trade, PUT_SPREAD_LEGS)
        assert result is None
        place_order.assert_not_called()
        assert any("EXIT DEFERRED" in n for n in notes)

    async def test_long_straddle_no_quote_defers(self):
        """The shape of the live IWM position: debit, no structural width.
        No quote must mean defer — never a market BAG."""
        orch, place_order, notes = _orchestrator(
            _quote(bid=float("nan"), ask=float("nan"))
        )
        trade = FakeTrade(
            symbol="IWM", strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        result = await orch._close_multi_leg(trade, STRADDLE_LEGS)
        assert result is None
        place_order.assert_not_called()
        assert any("EXIT DEFERRED" in n for n in notes)

    async def test_credit_strangle_no_quote_defers(self):
        """Credit shape with NO structural bound (short strangle): both
        choices are bad; we defer with a page rather than market a BAG blind.
        Documents the decided trade-off."""
        orch, place_order, notes = _orchestrator(
            _quote(bid=float("nan"), ask=float("nan"))
        )
        trade = FakeTrade(
            symbol="IWM", strategy="short_strangle", contract_type="spread",
            legs=json.dumps(STRANGLE_LEGS),
        )
        result = await orch._close_multi_leg(trade, STRANGLE_LEGS)
        assert result is None
        place_order.assert_not_called()
        assert any("EXIT DEFERRED" in n for n in notes)

    async def test_credit_strangle_quote_not_capped(self):
        """No structural bound exists for a strangle — the cap must NOT
        apply, even to an ugly quote (there is no width to cap at)."""
        orch, place_order, _ = _orchestrator(_quote(bid=9.00, ask=9.90))
        trade = FakeTrade(
            symbol="IWM", strategy="short_strangle", contract_type="spread",
            legs=json.dumps(STRANGLE_LEGS),
        )
        await orch._close_multi_leg(trade, STRANGLE_LEGS)
        order = place_order.call_args.args[1]
        assert order.lmtPrice == pytest.approx(9.95)  # ask + cross, uncapped

    async def test_debit_close_negative_quote_passes_through(self):
        """Closing a debit position nets a CREDIT — combo quotes negative.
        The wing-width cap (credit-only) must not touch it."""
        orch, place_order, _ = _orchestrator(_quote(bid=-1.20, ask=-1.00))
        trade = FakeTrade(
            strategy="bear_put_spread", contract_type="spread",
            legs=json.dumps(PUT_SPREAD_LEGS),
        )
        await orch._close_multi_leg(trade, PUT_SPREAD_LEGS)
        order = place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice == pytest.approx(-0.95)  # ask -1.00 + cross

    async def test_debit_close_garbage_positive_ask_capped_at_cross(self):
        """The mirror of the condor incident, on the IWM straddle's exact
        path. Closing a long structure SELLS what we own, so the quote is
        negative; a sign-corrupted +9.90 ask made the bot place BUY LMT 9.95 —
        PAYING ~$995 to dispose of a position it owned, uncapped and unlogged
        (the credit-only cap never looked at debit shapes). Never pay more
        than the crossing buffer to get out of something you own."""
        orch, place_order, _ = _orchestrator(_quote(bid=9.00, ask=9.90))
        trade = FakeTrade(
            symbol="IWM", strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        await orch._close_multi_leg(trade, STRADDLE_LEGS)
        order = place_order.call_args.args[1]
        assert order.lmtPrice == pytest.approx(0.05)  # the EXIT_CROSS bound
        assert order.lmtPrice <= 0.05

    async def test_debit_close_zero_limit_is_placed_not_deferred(self):
        """A computed limit of exactly 0.00 is a REAL marketable price for a
        debit close (dying straddle: ask -0.05 + 0.05 cross = 0.00). The old
        `limit_price != 0` truthiness test called that 'no quote' and deferred
        a perfectly closeable position indefinitely, page-spamming each pass."""
        orch, place_order, notes = _orchestrator(_quote(bid=-0.10, ask=-0.05))
        trade = FakeTrade(
            symbol="IWM", strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        result = await orch._close_multi_leg(trade, STRADDLE_LEGS)
        assert result is not None
        order = place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice == pytest.approx(0.0)
        assert not any("EXIT DEFERRED" in n for n in notes)

    async def test_no_market_order_ever_leaves_close_multi_leg(self):
        """Property over the whole matrix: whatever the quote state, the exit
        must never place a MARKET order on a multi-leg BAG."""
        cases = [
            (FakeTrade(), CONDOR_LEGS, _quote(float("nan"), float("nan"))),
            (FakeTrade(), CONDOR_LEGS, _quote(0.0, 0.0)),
            (FakeTrade(), CONDOR_LEGS, _quote(0.10, 9.90)),
            (FakeTrade(strategy="long_straddle", contract_type="spread",
                       legs=json.dumps(STRADDLE_LEGS)),
             STRADDLE_LEGS, _quote(float("nan"), float("nan"))),
            (FakeTrade(strategy="short_strangle", contract_type="spread",
                       legs=json.dumps(STRANGLE_LEGS)),
             STRANGLE_LEGS, _quote(float("nan"), float("nan"))),
            (FakeTrade(strategy="bear_put_spread", contract_type="spread",
                       legs=json.dumps(PUT_SPREAD_LEGS)),
             PUT_SPREAD_LEGS, _quote(float("nan"), float("nan"))),
        ]
        for trade, legs, ticker in cases:
            orch, place_order, _ = _orchestrator(ticker)
            await orch._close_multi_leg(trade, legs)
            for call in place_order.call_args_list:
                order = call.args[1]
                assert order.orderType != "MKT", (
                    f"market order leaked for {trade.strategy}"
                )

    async def test_ticker_none_defers_for_debit(self):
        """qualify/ticker returning nothing at all (not just NaN) must land
        in the same safe branches."""
        orch, place_order, _ = _orchestrator(ticker=None)
        trade = FakeTrade(
            strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        result = await orch._close_multi_leg(trade, STRADDLE_LEGS)
        assert result is None
        place_order.assert_not_called()


# ---------------------------------------------------------------------------
# position_liveness — the reconciler-side liveness answer
# ---------------------------------------------------------------------------

_UNSET = object()


def _reconciler(
    ibkr_positions,
    connected: bool = True,
    fresh=_UNSET,
) -> PositionReconciler:
    """`fresh` is the AUTHORITATIVE re-query result. Defaults to agreeing with
    the cache; pass a different list to model a wedged/stale cache, or None to
    model a broker that won't answer."""
    rec = PositionReconciler.__new__(PositionReconciler)
    ibkr = MagicMock()
    ibkr.connected = connected
    ibkr.get_positions.return_value = ibkr_positions
    ibkr.get_positions_fresh = AsyncMock(
        return_value=ibkr_positions if fresh is _UNSET else fresh
    )
    rec._ibkr = ibkr
    rec._state = MagicMock()
    return rec


def _ibkr_pos(symbol: str, strike: float, right: str, expiry: str):
    """An IBKR position row: OPT contract, expiry in IBKR's YYYYMMDD form."""
    return SimpleNamespace(position=1.0, contract=SimpleNamespace(
        secType="OPT", symbol=symbol, strike=strike, right=right,
        lastTradeDateOrContractMonth=expiry,
    ))


CONDOR_AT_BROKER = [
    _ibkr_pos("SPY", 713.0, "P", "20260724"),
    _ibkr_pos("SPY", 715.0, "P", "20260724"),
    _ibkr_pos("SPY", 752.0, "C", "20260724"),
    _ibkr_pos("SPY", 754.0, "C", "20260724"),
]


class TestPositionLiveness:
    async def test_all_legs_at_broker_is_live(self):
        """Regression for the key-format catastrophe: local legs store
        YYYY-MM-DD, IBKR reports YYYYMMDD — normalization must make these
        intersect, or EVERY exit would be refused."""
        rec = _reconciler(CONDOR_AT_BROKER)
        assert await rec.position_liveness(FakeTrade()) == "live"

    async def test_manually_flattened_position_is_gone(self):
        """The 07-13 end-state: operator flattened in TWS, broker confirms it
        holds nothing. The exit must refuse to reverse."""
        rec = _reconciler([])
        assert await rec.position_liveness(FakeTrade()) == "gone"

    async def test_unrelated_positions_do_not_count(self):
        others = [
            _ibkr_pos("QQQ", 770.0, "C", "20260731"),
            SimpleNamespace(contract=SimpleNamespace(secType="STK", symbol="SPY")),
        ]
        rec = _reconciler(others)
        assert await rec.position_liveness(FakeTrade()) == "gone"

    async def test_wedged_position_stream_is_unknown_not_gone(self):
        """THE denial-of-exit trap. ib_insync's startup reqPositions can time
        out with `connected` still True, leaving an EMPTY cache — which looks
        exactly like a manual flatten. Believing it would refuse every exit,
        disabling stops precisely when the broker link is flaky. The fresh
        re-query is what tells the two apart: cache empty, broker says LIVE."""
        rec = _reconciler([], fresh=CONDOR_AT_BROKER)
        assert await rec.position_liveness(FakeTrade()) == "live"

    async def test_broker_wont_confirm_is_unknown_so_exit_proceeds(self):
        """Cache says gone, authoritative re-query fails. We must NOT retire
        the position on the cache's word — 'unknown' lets the exit go out."""
        rec = _reconciler([], fresh=None)
        assert await rec.position_liveness(FakeTrade()) == "unknown"

    async def test_partial_flatten_is_partial_not_live(self):
        """Operator buys back the condor's two shorts in TWS (a routine
        risk-off move), leaving the wings. ANY-leg semantics would call this
        'live' and let _close_multi_leg reverse ALL FOUR legs — SELLING the
        two already-flat shorts, i.e. OPENING new naked shorts. That is the
        rebuild-inverted incident this gate exists to prevent."""
        rec = _reconciler([
            _ibkr_pos("SPY", 713.0, "P", "20260724"),  # long wings only
            _ibkr_pos("SPY", 754.0, "C", "20260724"),
        ])
        assert await rec.position_liveness(FakeTrade()) == "partial"

    async def test_straddle_one_leg_manually_sold_is_partial(self):
        """The live position's shape: IWM 299C+299P, operator sells only the
        call. Reversing both legs would SELL the 299C we no longer own — a
        naked short call, unbounded risk."""
        rec = _reconciler([_ibkr_pos("IWM", 299.0, "P", "20260724")])
        trade = FakeTrade(
            symbol="IWM", strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        assert await rec.position_liveness(trade) == "partial"

    async def test_wedged_cache_plus_half_flattened_structure_is_partial(self):
        """Both failures at once, and the fresh query is the only thing that
        sees it: the cache is empty (wedged stream) AND the operator really did
        sell one leg. The re-query must classify this as partial — resolving it
        to 'live' would reverse all legs and open a naked short on the leg that
        is already gone."""
        rec = _reconciler([], fresh=[_ibkr_pos("SPY", 713.0, "P", "20260724")])
        assert await rec.position_liveness(FakeTrade()) == "partial"

    async def test_fresh_query_confirming_all_legs_is_live(self):
        rec = _reconciler([], fresh=CONDOR_AT_BROKER)
        assert await rec.position_liveness(FakeTrade()) == "live"

    async def test_disconnected_broker_is_unknown(self):
        """Never 'gone': a disconnect blip must not disable exits."""
        rec = _reconciler([], connected=False)
        assert await rec.position_liveness(FakeTrade()) == "unknown"

    async def test_get_positions_raising_is_unknown(self):
        rec = _reconciler([])
        rec._ibkr.get_positions.side_effect = RuntimeError("socket dropped")
        assert await rec.position_liveness(FakeTrade()) == "unknown"

    async def test_unkeyable_trade_is_unknown(self):
        """No legs, no strike → STK-fallback key → 'can't tell', not 'gone'."""
        rec = _reconciler([])
        trade = FakeTrade(legs="[]", strike=None, strategy="long_call")
        assert await rec.position_liveness(trade) == "unknown"

    async def test_single_leg_trade_via_strike_fallback(self):
        """Single-option trades key off trade.strike/expiry — both answers."""
        trade = FakeTrade(
            legs="[]", strike=700.0, expiry="2026-07-24",
            strategy="cash_secured_put", contract_type="put",
        )
        rec = _reconciler([_ibkr_pos("SPY", 700.0, "P", "20260724")])
        assert await rec.position_liveness(trade) == "live"
        assert await _reconciler([]).position_liveness(trade) == "gone"

    async def test_fresh_query_only_runs_when_cache_says_gone(self):
        """The re-query is a broker round-trip: it must not fire on every
        healthy exit, only when we are about to retire a position."""
        rec = _reconciler(CONDOR_AT_BROKER)
        assert await rec.position_liveness(FakeTrade()) == "live"
        rec._ibkr.get_positions_fresh.assert_not_called()


# ---------------------------------------------------------------------------
# _execute_exit — the gate wiring: refuse on False, proceed on None
# ---------------------------------------------------------------------------

def _exit_orchestrator(
    liveness,
    trade: FakeTrade,
) -> tuple[TradingOrchestrator, AsyncMock, MagicMock, list[str]]:
    """Real _execute_exit with everything downstream of the gate mocked."""
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    notifications: list[str] = []

    orch._find_trade_record = lambda tid: trade
    ibkr = MagicMock()
    ibkr.ensure_connected = AsyncMock(return_value=True)
    orch._ibkr = ibkr

    reconciler = MagicMock()
    if isinstance(liveness, Exception):
        reconciler.position_liveness = AsyncMock(side_effect=liveness)
    else:
        reconciler.position_liveness = AsyncMock(return_value=liveness)
    reconciler.book_vanished_trade = AsyncMock(return_value=True)
    orch._reconciler = reconciler

    orch._watchdog = MagicMock()
    orch._state = MagicMock()
    orch._state.update_trade_status.return_value = True
    orch._executor = MagicMock()

    close_multi_leg = AsyncMock(
        return_value=SimpleNamespace(order=SimpleNamespace(orderId=99))
    )
    orch._close_multi_leg = close_multi_leg

    async def _notify(msg: str) -> None:
        notifications.append(msg)

    orch._send_notification = _notify
    return orch, close_multi_leg, orch._state, notifications


def _pos_status(trade_id: str = "T-EXIT-SAFETY") -> SimpleNamespace:
    return SimpleNamespace(
        trade_id=trade_id, symbol="SPY", exit_reason="take_profit_short",
        unrealized_pnl=55.0,
    )


class TestExecuteExitLivenessGate:
    async def test_gone_at_broker_refuses_exit(self):
        """position confirmed absent: NO order, NO status change, loud page.
        Pre-fix the reverse combo re-opened the position inverted."""
        trade = FakeTrade(contract_type="spread")
        orch, close, state, notes = _exit_orchestrator("gone", trade)
        await orch._execute_exit(_pos_status())
        close.assert_not_called()
        state.update_trade_status.assert_not_called()
        assert any("EXIT REFUSED" in n for n in notes)

    async def test_gone_at_broker_BOOKS_the_trade(self):
        """THE strand bug. Refusing the exit is only half the job: reconcile()'s
        zero-options guard ('refusing to mass-close') fires on exactly the state
        a manual flatten of the LAST position leaves, so its stale-local loop
        never books the row. Without booking it HERE the trade sits FILLED
        forever — monitor re-demands the exit, gate refuses, every pass, and no
        code path ever closes it."""
        trade = FakeTrade(contract_type="spread")
        orch, close, _, notes = _exit_orchestrator("gone", trade)
        await orch._execute_exit(_pos_status())
        orch._reconciler.book_vanished_trade.assert_awaited_once_with(trade)
        close.assert_not_called()
        assert any("Booked closed" in n for n in notes)

    async def test_gone_but_booking_fails_says_so(self):
        """Never claim it was handled when it wasn't."""
        trade = FakeTrade(contract_type="spread")
        orch, _, _, notes = _exit_orchestrator("gone", trade)
        orch._reconciler.book_vanished_trade = AsyncMock(return_value=False)
        await orch._execute_exit(_pos_status())
        assert any("manual review" in n for n in notes)

    async def test_partial_structure_refuses_exit(self):
        """Some legs gone: reversing ALL of them would OPEN inverted positions
        on the missing ones (a naked short in the worst case). Refuse, page a
        human, and do NOT book — the remainder is still live at the broker."""
        trade = FakeTrade(contract_type="spread")
        orch, close, state, notes = _exit_orchestrator("partial", trade)
        await orch._execute_exit(_pos_status())
        close.assert_not_called()
        state.update_trade_status.assert_not_called()
        orch._reconciler.book_vanished_trade.assert_not_called()
        assert any("partial structure" in n for n in notes)

    async def test_unknown_liveness_proceeds(self):
        """Broker view unavailable: the exit MUST still go out. A wedged data
        feed must never silently disable a stop."""
        trade = FakeTrade(contract_type="spread")
        orch, close, state, _ = _exit_orchestrator("unknown", trade)
        await orch._execute_exit(_pos_status())
        close.assert_called_once()
        state.update_trade_status.assert_called_once()

    async def test_live_at_broker_proceeds(self):
        trade = FakeTrade(contract_type="spread")
        orch, close, _, _ = _exit_orchestrator("live", trade)
        await orch._execute_exit(_pos_status())
        close.assert_called_once()

    async def test_liveness_check_crash_never_blocks_exit(self):
        """The check is advisory: if it raises, the exit proceeds."""
        trade = FakeTrade(contract_type="spread")
        orch, close, _, _ = _exit_orchestrator(RuntimeError("recon broke"), trade)
        await orch._execute_exit(_pos_status())
        close.assert_called_once()

    async def test_deferred_close_leaves_trade_filled(self):
        """_close_multi_leg returning None (exit deferred, no quote) must NOT
        move the trade to CLOSING — it stays FILLED for the next pass."""
        trade = FakeTrade(contract_type="spread")
        orch, close, state, _ = _exit_orchestrator("live", trade)
        close.return_value = None
        await orch._execute_exit(_pos_status())
        state.update_trade_status.assert_not_called()


# ---------------------------------------------------------------------------
# _alert_gate — refused/deferred exits re-fire every monitor pass; the page
# must not (R13 human-factors: an alert storm is an ignored alert)
# ---------------------------------------------------------------------------

class TestAlertThrottle:
    async def test_repeat_refusals_page_once_per_window(self):
        """The gate refuses every pass until post-market reconcile books the
        trade — Telegram must see ONE page per window, not one per pass."""
        trade = FakeTrade(contract_type="spread")
        orch, _, _, notes = _exit_orchestrator("gone", trade)
        for _ in range(5):
            await orch._execute_exit(_pos_status())
        assert sum("EXIT REFUSED" in n for n in notes) == 1

    async def test_window_expiry_pages_again(self, monkeypatch):
        trade = FakeTrade(contract_type="spread")
        orch, _, _, notes = _exit_orchestrator("gone", trade)
        clock = {"t": 1000.0}
        monkeypatch.setattr(
            "ait.bot.orchestrator.time.monotonic", lambda: clock["t"]
        )
        await orch._execute_exit(_pos_status())
        clock["t"] += 899.0
        await orch._execute_exit(_pos_status())  # still inside the window
        clock["t"] += 2.0
        await orch._execute_exit(_pos_status())  # window expired
        assert sum("EXIT REFUSED" in n for n in notes) == 2

    async def test_distinct_trades_page_independently(self):
        t1 = FakeTrade(contract_type="spread", trade_id="T-A")
        orch, _, _, notes = _exit_orchestrator("gone", t1)
        await orch._execute_exit(_pos_status("T-A"))
        t2 = FakeTrade(contract_type="spread", trade_id="T-B")
        orch._find_trade_record = lambda tid: t2
        await orch._execute_exit(_pos_status("T-B"))
        assert sum("EXIT REFUSED" in n for n in notes) == 2

    async def test_deferred_exit_page_throttled(self):
        """The no-quote deferral loops every cycle too — same storm risk."""
        orch, place_order, notes = _orchestrator(
            _quote(bid=float("nan"), ask=float("nan"))
        )
        trade = FakeTrade(
            strategy="long_straddle", contract_type="spread",
            legs=json.dumps(STRADDLE_LEGS),
        )
        for _ in range(4):
            assert await orch._close_multi_leg(trade, STRADDLE_LEGS) is None
        place_order.assert_not_called()
        assert sum("EXIT DEFERRED" in n for n in notes) == 1
