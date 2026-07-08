"""Regression tests for the 2026-06-11 P&L accounting fixes.

Covers:
1. Credit/debit branching in TradeExecutor._calculate_realized_pnl
   (long straddles were sign-inverted via contract_type="spread")
2. _get_exit_fill_price None-vs-0.0 sentinel
3. PortfolioManager._option_position_unrealized (option marks, not underlying)
4. PositionReconciler leg-aware matching + mass-close guard
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from ait.execution.executor import TradeExecutor
from ait.strategies.base import CREDIT_STRATEGIES


# ---------------------------------------------------------------------------
# helpers


@dataclass
class FakeTrade:
    entry_price: float
    quantity: int
    contract_type: str
    strategy: str
    symbol: str = "SPY"
    trade_id: str = "T-TEST"
    direction: object = None
    legs: str = "[]"
    expiry: str | None = "2026-12-18"
    strike: float | None = None


class _CalcHost:
    """Minimal host exposing the real _calculate_realized_pnl."""

    CREDIT_STRATEGIES = CREDIT_STRATEGIES
    _calculate_realized_pnl = TradeExecutor._calculate_realized_pnl


calc = _CalcHost()._calculate_realized_pnl


# ---------------------------------------------------------------------------
# 1. credit/debit P&L branching


class TestRealizedPnlFormula:
    def test_long_straddle_expiring_worthless_is_a_loss(self):
        t = FakeTrade(24.03, 1, "spread", "long_straddle")
        pnl = calc(t, 0.0)
        assert pnl == pytest.approx(-2403 - 2.6, abs=0.01)

    def test_long_straddle_doubling_is_a_gain(self):
        t = FakeTrade(19.925, 1, "spread", "long_straddle")
        pnl = calc(t, 39.85)
        assert pnl == pytest.approx(+1992.5 - 2.6, abs=0.01)

    def test_event_straddle_treated_as_debit(self):
        t = FakeTrade(10.0, 2, "spread", "event_straddle")
        pnl = calc(t, 15.0)
        assert pnl == pytest.approx((15 - 10) * 100 * 2 - 0.65 * 2 * 2 * 2, abs=0.01)

    def test_iron_condor_expiring_worthless_keeps_credit(self):
        t = FakeTrade(0.80, 1, "iron_condor", "iron_condor")
        pnl = calc(t, 0.0)
        assert pnl == pytest.approx(+80 - 5.2, abs=0.01)

    def test_iron_condor_bought_back_above_credit_is_a_loss(self):
        t = FakeTrade(0.80, 1, "iron_condor", "iron_condor")
        assert calc(t, 1.50) < 0

    def test_short_strangle_expiring_worthless_is_a_win(self):
        t = FakeTrade(2.50, 1, "spread", "short_strangle")
        pnl = calc(t, 0.0)
        assert pnl == pytest.approx(+250 - 2.6, abs=0.01)

    def test_bull_call_spread_is_debit_despite_two_legs(self):
        t = FakeTrade(1.20, 1, "spread", "bull_call_spread")
        assert calc(t, 2.00) > 0   # debit spread that widened = profit
        assert calc(t, 0.0) < 0    # expired worthless = loss

    def test_cash_secured_put_cheap_buyback_is_a_win(self):
        t = FakeTrade(5.00, 1, "put", "cash_secured_put")
        assert calc(t, 1.00) == pytest.approx(+400 - 1.3, abs=0.01)

    def test_stock_no_multiplier(self):
        t = FakeTrade(100.0, 50, "stock", "stock")
        assert calc(t, 110.0) == pytest.approx(500 - 65.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. exit fill price sentinel


class TestExitFillPriceSentinel:
    def _executor(self):
        ex = TradeExecutor.__new__(TradeExecutor)
        return ex

    def _ib_trade(self, order_id=1, avg=0.0, fills=(), status="Filled", filled_qty=0):
        t = MagicMock()
        t.order.orderId = order_id
        t.orderStatus.avgFillPrice = avg
        t.orderStatus.status = status
        t.orderStatus.filled = filled_qty
        t.fills = list(fills)
        return t

    def test_missing_order_returns_none(self):
        ex = self._executor()
        assert ex._get_exit_fill_price(99, []) is None

    def test_avg_price_used_when_positive(self):
        ex = self._executor()
        assert ex._get_exit_fill_price(1, [self._ib_trade(avg=2.5)]) == 2.5

    def test_zero_fill_with_filled_qty_is_real_zero(self):
        ex = self._executor()
        # combo closed at ~0: avgFillPrice 0 but order shows filled quantity
        assert ex._get_exit_fill_price(1, [self._ib_trade(avg=0.0, filled_qty=1)]) == 0.0

    def test_zero_fill_without_quantity_is_none(self):
        ex = self._executor()
        assert ex._get_exit_fill_price(1, [self._ib_trade(avg=0.0, filled_qty=0)]) is None


# ---------------------------------------------------------------------------
# 3. option position pricing from IBKR marks


def _portfolio_item(symbol, expiry, strike, right, price):
    item = MagicMock()
    item.contract.secType = "OPT"
    item.contract.symbol = symbol
    item.contract.lastTradeDateOrContractMonth = expiry
    item.contract.strike = strike
    item.contract.right = right
    item.marketPrice = price
    return item


class TestOptionPositionUnrealized:
    def _manager(self, portfolio_items):
        from ait.execution.portfolio import PortfolioManager
        mgr = PortfolioManager.__new__(PortfolioManager)
        mgr._ibkr = MagicMock()
        mgr._ibkr.ib.portfolio.return_value = portfolio_items
        return mgr

    def _direction(self, value):
        d = MagicMock()
        d.value = value
        return d

    def test_long_straddle_unrealized_from_leg_marks(self):
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-04-17"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2026-04-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs)
        trade.direction = self._direction("long")
        mgr = self._manager([
            _portfolio_item("SPY", "20260417", 635.0, "C", 18.0),
            _portfolio_item("SPY", "20260417", 635.0, "P", 12.0),
        ])
        # net_liq = 30, entry debit 24 -> +$600
        assert mgr._option_position_unrealized(trade) == pytest.approx(600.0)

    def test_iron_condor_unrealized_sign(self):
        legs = json.dumps([
            {"strike": 590.0, "right": "P", "action": "BUY", "expiry": "2026-07-17"},
            {"strike": 600.0, "right": "P", "action": "SELL", "expiry": "2026-07-17"},
            {"strike": 660.0, "right": "C", "action": "SELL", "expiry": "2026-07-17"},
            {"strike": 670.0, "right": "C", "action": "BUY", "expiry": "2026-07-17"},
        ])
        trade = FakeTrade(0.80, 1, "iron_condor", "iron_condor", legs=legs)
        trade.direction = self._direction("short")
        mgr = self._manager([
            _portfolio_item("SPY", "20260717", 590.0, "P", 0.10),
            _portfolio_item("SPY", "20260717", 600.0, "P", 0.25),
            _portfolio_item("SPY", "20260717", 660.0, "C", 0.20),
            _portfolio_item("SPY", "20260717", 670.0, "C", 0.05),
        ])
        # net_liq = 0.10 - 0.25 - 0.20 + 0.05 = -0.30; credit 0.80 -> +$50
        assert mgr._option_position_unrealized(trade) == pytest.approx(50.0)

    def test_missing_leg_mark_returns_none(self):
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-04-17"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2026-04-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs)
        trade.direction = self._direction("long")
        mgr = self._manager([
            _portfolio_item("SPY", "20260417", 635.0, "C", 18.0),
            # put leg absent
        ])
        assert mgr._option_position_unrealized(trade) is None

    def test_single_leg_long_call(self):
        trade = FakeTrade(3.0, 1, "call", "long_call", legs="[]",
                          expiry="2026-07-17", strike=650.0)
        trade.direction = self._direction("long")  # BULLISH
        mgr = self._manager([
            _portfolio_item("SPY", "20260717", 650.0, "C", 4.5),
        ])
        # bought at 3.00, now worth 4.50 -> +$150
        assert mgr._option_position_unrealized(trade) == pytest.approx(150.0)

    def test_single_leg_long_put_winning_reads_as_gain(self):
        # long_put has direction=SHORT (BEARISH); the OLD code signed the
        # leg SELL and reported a winning put as a -$500 loss -> instant stop.
        trade = FakeTrade(2.0, 1, "put", "long_put", legs="[]",
                          expiry="2026-07-17", strike=600.0)
        trade.direction = self._direction("short")  # BEARISH
        mgr = self._manager([
            _portfolio_item("SPY", "20260717", 600.0, "P", 3.0),
        ])
        # bought at 2.00, now worth 3.00 -> +$100 (NOT -$500)
        assert mgr._option_position_unrealized(trade) == pytest.approx(100.0)

    def test_single_leg_cash_secured_put_signed_as_credit(self):
        # CSP has direction=LONG (BULLISH); the OLD code signed the leg BUY
        # and made the 50% stop mathematically unreachable.
        trade = FakeTrade(0.80, 1, "put", "cash_secured_put", legs="[]",
                          expiry="2026-07-17", strike=580.0)
        trade.direction = self._direction("long")  # BULLISH
        mgr = self._manager([
            _portfolio_item("SPY", "20260717", 580.0, "P", 0.40),
        ])
        # sold at 0.80, now worth 0.40 to buy back -> +$40 (half of max profit)
        assert mgr._option_position_unrealized(trade) == pytest.approx(40.0)

    def test_single_leg_cash_secured_put_losing_reads_as_loss(self):
        trade = FakeTrade(0.80, 1, "put", "cash_secured_put", legs="[]",
                          expiry="2026-07-17", strike=580.0)
        trade.direction = self._direction("long")
        mgr = self._manager([
            _portfolio_item("SPY", "20260717", 580.0, "P", 2.50),
        ])
        # sold 0.80, now costs 2.50 to close -> -$170 (NOT +$330)
        assert mgr._option_position_unrealized(trade) == pytest.approx(-170.0)


# ---------------------------------------------------------------------------
# 4. reconciler leg-aware matching + mass-close guard


class TestReconcilerMatching:
    def _reconciler(self, ibkr_positions, open_trades):
        from ait.execution.reconciler import PositionReconciler
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._ibkr = MagicMock()
        rec._ibkr.get_positions.return_value = ibkr_positions
        rec._ibkr.get_portfolio.return_value = []
        rec._state = MagicMock()
        rec._state.get_open_trades.return_value = open_trades
        return rec

    def _ibkr_pos(self, symbol, expiry, strike, right):
        p = MagicMock()
        p.contract.secType = "OPT"
        p.contract.symbol = symbol
        p.contract.lastTradeDateOrContractMonth = expiry
        p.contract.strike = strike
        p.contract.right = right
        p.position = 1
        p.avgCost = 100.0
        return p

    @pytest.mark.asyncio
    async def test_straddle_with_live_legs_not_closed(self):
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-07-17"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2026-07-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs)
        trade.status = MagicMock()
        rec = self._reconciler(
            [self._ibkr_pos("SPY", "20260717", 635.0, "C"),
             self._ibkr_pos("SPY", "20260717", 635.0, "P")],
            [trade],
        )
        result = await rec.reconcile()
        assert result.stale_local == 0
        rec._state.close_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_ibkr_list_refuses_mass_close(self):
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-07-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs)
        trade.status = MagicMock()
        rec = self._reconciler([], [trade])
        result = await rec.reconcile()
        assert result.stale_local == 0
        rec._state.close_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_trade_with_live_legs_promoted_to_filled(self):
        # Orphan recovery: a trade stuck PENDING after a restart whose legs
        # are live in IBKR must be promoted to FILLED so the monitor manages it.
        from ait.bot.state import TradeStatus
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-07-17"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2026-07-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs)
        trade.status = TradeStatus.PENDING
        rec = self._reconciler(
            [self._ibkr_pos("SPY", "20260717", 635.0, "C"),
             self._ibkr_pos("SPY", "20260717", 635.0, "P")],
            [trade],
        )
        result = await rec.reconcile()
        assert result.promoted == 1
        rec._state.update_trade_status.assert_called_once()
        args = rec._state.update_trade_status.call_args[0]
        assert args[1] == TradeStatus.FILLED

    @pytest.mark.asyncio
    async def test_truly_gone_position_closed_with_flag(self):
        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2020-01-17"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2020-01-17"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle",
                          legs=legs, expiry="2020-01-17")
        trade.status = MagicMock()
        # IBKR holds an UNRELATED position so the safety guard doesn't trip
        rec = self._reconciler(
            [self._ibkr_pos("QQQ", "20260821", 500.0, "C")],
            [trade],
        )
        # ITM-aware expiry booking (audit R2): the reconciler now values the
        # structure at expiry from intrinsic using the underlying price
        # instead of assuming "expired worthless". Settle exactly ATM (635):
        # both straddle legs are worthless -> full debit loss, same -2400 as
        # before, but computed honestly.
        async def _settle(_symbol):
            return 635.0
        rec._underlying_last = _settle
        result = await rec.reconcile()
        assert result.stale_local == 1
        kwargs = rec._state.close_trade.call_args.kwargs
        assert kwargs["realized_pnl"] == pytest.approx(-2400.0)
        assert kwargs["exit_reason_detailed"] == "reconciler_expired_intrinsic"


# ---------------------------------------------------------------------------
# 5. exit-path fixes (2026-06-11 round 2)


class TestExitFillPriceSigned:
    def _executor(self):
        ex = TradeExecutor.__new__(TradeExecutor)
        return ex

    def _bag_trade(self, order_id=1, avg=float("nan"), fills=(), qty=1):
        t = MagicMock()
        t.order.orderId = order_id
        t.order.totalQuantity = qty
        t.orderStatus.avgFillPrice = avg
        t.orderStatus.status = "Filled"
        t.orderStatus.filled = qty
        t.contract.secType = "BAG"
        t.fills = list(fills)
        return t

    def test_negative_avg_fill_price_accepted(self):
        # Closing a debit straddle via reversed legs: as-defined price is
        # negative (credit received). The old code rejected it.
        ex = self._executor()
        assert ex._get_exit_fill_price(1, [self._bag_trade(avg=-30.0)]) == -30.0

    def test_bag_fills_fallback_is_side_aware(self):
        # 2-leg combo close, 1 contract: SELL call @2.00, SELL put @0.80
        # net received = 2.80/contract -> as-defined price = -2.80
        def fill(side, price, shares):
            f = MagicMock()
            f.execution.side = side
            f.execution.price = price
            f.execution.shares = shares
            return f
        ex = self._executor()
        # ib_insync execution.shares for options = CONTRACT count (deep-audit
        # MP-F1): the old fixture used shares=100 for 1 contract, enshrining a
        # convention that made real reconstructions 100x too small.
        t = self._bag_trade(avg=float("nan"), qty=1,
                            fills=[fill("SLD", 2.00, 1), fill("SLD", 0.80, 1)])
        assert ex._get_exit_fill_price(1, [t]) == pytest.approx(-2.80)

    def test_bag_mixed_side_fills(self):
        # closing a debit vertical: SELL long leg @2.00, BUY short leg @0.80
        # net received = 1.20 -> as-defined -1.20 (NOT the naive VWAP 1.40)
        def fill(side, price, shares):
            f = MagicMock()
            f.execution.side = side
            f.execution.price = price
            f.execution.shares = shares
            return f
        ex = self._executor()
        t = self._bag_trade(avg=float("nan"), qty=1,
                            fills=[fill("SLD", 2.00, 1), fill("BOT", 0.80, 1)])
        assert ex._get_exit_fill_price(1, [t]) == pytest.approx(-1.20)


class TestCloseDirection:
    """The close action must reverse the POSITION (credit=buy back,
    debit=sell), never the market bias stored in trade.direction."""

    def test_credit_strategies_close_with_buy(self):
        for strat in ("cash_secured_put", "covered_call", "iron_condor", "short_strangle"):
            close_action = "BUY" if strat in CREDIT_STRATEGIES else "SELL"
            assert close_action == "BUY", strat

    def test_debit_strategies_close_with_sell(self):
        for strat in ("long_call", "long_put", "long_straddle", "event_straddle",
                      "bull_call_spread", "bear_put_spread", "calendar_spread"):
            close_action = "BUY" if strat in CREDIT_STRATEGIES else "SELL"
            assert close_action == "SELL", strat


class TestLegFields:
    def test_contract_shaped_leg(self):
        c = MagicMock()
        c.strike = 635.0
        c.right = "C"
        c.expiry = "2026-07-17"
        strike, right, expiry = TradeExecutor._leg_fields({"contract": c, "action": "BUY"})
        assert (strike, right, expiry) == (635.0, "C", "2026-07-17")

    def test_dict_shaped_leg_event_straddle(self):
        # event_straddle/calendar emit plain dicts — these used to KeyError
        leg = {"strike": 600.0, "right": "P", "action": "BUY", "expiry": "2026-06-17"}
        strike, right, expiry = TradeExecutor._leg_fields(leg)
        assert (strike, right, expiry) == (600.0, "P", "2026-06-17")


class TestMarksMissingSafetyExits:
    """Missing IBKR marks must not disable DTE-based safety exits."""

    @pytest.mark.asyncio
    async def test_expiring_position_still_closed_without_marks(self):
        from datetime import date as _date, timedelta
        from ait.execution.portfolio import PortfolioManager
        mgr = PortfolioManager.__new__(PortfolioManager)
        mgr._ibkr = MagicMock()
        mgr._ibkr.ib.portfolio.return_value = []   # no marks at all
        mgr._state = MagicMock()
        mgr._state.get_high_water_mark.return_value = 0.0
        mgr._market_data = MagicMock()
        async def _price(symbol):
            return 600.0
        mgr._market_data.get_current_price = _price
        from ait.config.settings import ExitConfig
        mgr._exit_config = ExitConfig()
        mgr._earnings = None
        mgr._economic_cal = None
        async def _vol_mult(symbol):
            return 1.0
        mgr._get_volatility_stop_multiplier = _vol_mult
        mgr._pdt_guard = MagicMock()
        mgr._pdt_guard.would_be_day_trade.return_value = False

        legs = json.dumps([
            {"strike": 635.0, "right": "C", "action": "BUY", "expiry": "2026-06-15"},
            {"strike": 635.0, "right": "P", "action": "BUY", "expiry": "2026-06-15"},
        ])
        trade = FakeTrade(24.0, 1, "spread", "long_straddle", legs=legs,
                          expiry=(_date.today() + timedelta(days=3)).isoformat())
        trade.direction = MagicMock()
        trade.direction.value = "long"
        trade.trade_id = "T-MARKS"
        trade.entry_time = "2026-06-01T10:00:00"

        status = await mgr._evaluate_position(trade)
        # Old behavior: returned None (position invisible / unexitable).
        # New behavior: still evaluated; DTE<=5 rule forces the close.
        assert status is not None
        assert status.should_exit
        assert "expiry_approaching" in status.exit_reason
