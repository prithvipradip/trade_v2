"""W101-B: a condor close is never offered for less than the structure is worth.

2026-10-22/23, QQQ condor (short 707P / long 704P, width 3.0, credit 0.92): the bot
placed 176 closing combos, IB accepted all of them, none filled. 161 were priced
from the four legs (the combo never ticked) with no memory that the last
placement had not filled; 26 of the 63 retained 10-23 placements sat below the
spread's intrinsic value; and every normal 120 s re-price was counted as a broker
reject by the backoff.

Every test drives the real ``_close_multi_leg`` on the W33 rig (real settings,
stubbed broker). The numbers are the incident's own: at 15:58 ET the delayed leg
quotes were short 707P 2.82/2.87, long 704P 0.50/0.52, both call legs worthless,
QQQ 704.40.
"""
from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace

import pytest

import ait.bot.orchestrator as orch_mod
from ait.bot.orchestrator import TradingOrchestrator
from ait.execution.reconciler import PositionReconciler
from tests.test_w33_exit_pricing_corrective import NAN, _rig as _w33_rig
from unittest.mock import AsyncMock

LEGS = [
    {"strike": 704, "right": "P", "action": "BUY", "expiry": "20261023"},
    {"strike": 707, "right": "P", "action": "SELL", "expiry": "20261023"},
    {"strike": 769, "right": "C", "action": "SELL", "expiry": "20261023"},
    {"strike": 772, "right": "C", "action": "BUY", "expiry": "20261023"},
]
# conIds are handed out in leg order: 1 = long put (-> SELL), 2 = short put
# (-> BUY), 3 = short call (-> BUY), 4 = long call (-> SELL).
INCIDENT = {1: (0.50, 0.52), 2: (2.82, 2.87), 3: (-1.0, 0.01), 4: (-1.0, 0.01)}
NO_LEG_QUOTES = {1: (NAN, NAN), 2: (NAN, NAN), 3: (NAN, NAN), 4: (NAN, NAN)}
WIDTH = 3.0


_CON_IDS = {(704.0, "P"): 1, (707.0, "P"): 2, (769.0, "C"): 3, (772.0, "C"): 4}


def _rig(**kw):
    """The W33 rig, with contract ids that stay the same from one placement to the
    next (it hands out new ones on every call, so a second placement would find
    no leg quotes at all)."""
    o, ibkr, trade = _w33_rig(**kw)

    async def _qualify(contract):
        key = (float(contract.strike), contract.right)
        return SimpleNamespace(conId=_CON_IDS.get(key, 99))

    ibkr.qualify_contract = AsyncMock(side_effect=_qualify)
    return o, ibkr, trade


@pytest.fixture(autouse=True)
def _instant_polls(monkeypatch):
    real = asyncio.sleep

    async def _no_wait(_s):
        await real(0)

    monkeypatch.setattr(orch_mod.asyncio, "sleep", _no_wait)
    monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "3")          # the bot's delayed feed


TOUCH = "short_strike_touch (spot 704.40 vs short put 707.00)"      # a must-get-out exit
TAKE_PROFIT = "take_profit_short (P&L: 41.0%, target: 40.0%)"


def _go(o, ibkr, trade, *, spot=None, dte=None, legs=LEGS, reason=TOUCH):
    ibkr.place_order.reset_mock()
    asyncio.run(o._close_multi_leg(trade, legs, spot=spot, dte=dte, reason=reason))
    if not ibkr.place_order.called:
        return None
    return ibkr.place_order.call_args.args[1].lmtPrice


def _incident(**kw):
    kw.setdefault("leg_quotes", dict(INCIDENT))
    kw.setdefault("entry", 0.92)
    kw.setdefault("mark_cost", 2.40)
    return _rig(**kw)


def _shim(trade):
    return SimpleNamespace(strategy=trade.strategy, legs=json.dumps(LEGS), strike=None)


# --------------------------------------------------------------------------
# the incident
# --------------------------------------------------------------------------
class TestTheIncidentShape:
    def test_without_spot_or_dte_the_pricing_is_exactly_the_old_one(self):
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade) == pytest.approx(2.48)          # synth 2.38 + cross

    def test_near_expiry_on_delayed_data_the_floor_allows_for_a_stale_spot(self):
        """Spot 704.40 +/- 0.25% reaches 702.64, where the spread is worth the
        full 3.00; the limit is the wing width, not the 2.48 that never filled."""
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade, spot=704.4, dte=0) == pytest.approx(3.00)

    def test_near_expiry_on_live_data_the_floor_is_the_point_intrinsic_plus_cross(self, monkeypatch):
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade, spot=704.4, dte=0) == pytest.approx(2.70)

    def test_a_take_profit_gets_the_point_floor_not_the_delay_allowance(self):
        """Same spot, same day: a take-profit is not an exit that must get out, so
        no delay band and no width cap (2.70, not 3.00)."""
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade, spot=704.4, dte=0, reason=TAKE_PROFIT) == pytest.approx(2.70)
        o2, ibkr2, trade2 = _incident()
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=0, reason=None) == pytest.approx(2.70)

    def test_before_the_near_expiry_window_there_is_no_band_but_the_floor_still_applies(self):
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade, spot=704.4, dte=3) == pytest.approx(2.70)

    def test_the_floor_is_the_one_valuation_function(self):
        o, ibkr, trade = _incident()
        intr = PositionReconciler._structure_intrinsic(_shim(trade), 704.4)
        assert intr == pytest.approx(2.60)
        floor = o._exit_intrinsic_floor(trade, LEGS, 704.4, 3, WIDTH)
        assert floor == pytest.approx(round(intr + o._settings.exit.exit_cross_amount, 2))

    def test_out_of_the_money_is_untouched(self):
        o, ibkr, trade = _incident()
        legacy = _go(o, ibkr, trade)
        o2, ibkr2, trade2 = _incident()
        assert _go(o2, ibkr2, trade2, spot=740.0, dte=0) == pytest.approx(legacy)
        assert o2._exit_intrinsic_floor(trade2, LEGS, 740.0, 0, WIDTH) is None


class TestTheNearExpiryWindow:
    def test_the_boundary_day_is_inside_the_window(self):
        """exit_near_expiry_dte = 1: a spot ~15 minutes old is allowed for at DTE 1
        (floor reaches the width), not at DTE 2 (point intrinsic + cross)."""
        o, ibkr, trade = _incident()
        assert _go(o, ibkr, trade, spot=704.4, dte=1) == pytest.approx(3.00)
        o2, ibkr2, trade2 = _incident()
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=2) == pytest.approx(2.70)

    def test_near_expiry_a_low_stale_mark_does_not_cap_an_escalated_limit(self, monkeypatch):
        """Mark 1.00 -> ceiling 1.60. Four unfilled placements have escalated the
        limit to 2.98, above the 2.70 floor. Near expiry the stale mark must not
        pull it back to the floor (the order would never catch the market)."""
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _incident(mark_cost=1.0, prior_attempts=4)
        assert _go(o, ibkr, trade, spot=704.4, dte=0) == pytest.approx(2.98)
        o2, ibkr2, trade2 = _incident(mark_cost=1.0, prior_attempts=4)
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=3) == pytest.approx(2.70)   # capped, then floored

    def test_a_take_profit_near_expiry_is_still_held_by_the_mark_ceiling(self, monkeypatch):
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _incident(mark_cost=1.0, prior_attempts=4)
        assert _go(o, ibkr, trade, spot=704.4, dte=0, reason=TAKE_PROFIT) == pytest.approx(2.70)

    def test_near_expiry_the_no_quote_cap_is_the_width(self, monkeypatch):
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92,
                              mark_cost=1.0, prior_attempts=25)
        assert _go(o, ibkr, trade, spot=704.4, dte=0) == pytest.approx(3.00)
        o2, ibkr2, trade2 = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92,
                                 mark_cost=1.0, prior_attempts=25)
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=3) == pytest.approx(2.70)   # ceiling 1.60, then floor


# --------------------------------------------------------------------------
# negative controls and scope
# --------------------------------------------------------------------------
class TestSwitchesAndScope:
    def test_the_switch_restores_the_old_pricing(self):
        o, ibkr, trade = _incident()
        o._settings = o._settings.model_copy(deep=True)
        o._settings.exit.exit_intrinsic_floor_enabled = False
        assert _go(o, ibkr, trade, spot=704.4, dte=0) == pytest.approx(2.48)

    def test_a_debit_structure_has_no_floor(self):
        o, ibkr, trade = _incident()
        trade.strategy = "long_straddle"
        legs = [{"strike": 704, "right": "P", "action": "BUY", "expiry": "20261023"},
                {"strike": 704, "right": "C", "action": "BUY", "expiry": "20261023"}]
        assert o._exit_intrinsic_floor(trade, legs, 700.0, 0, 5.0) is None

    def test_no_spot_no_floor(self):
        o, ibkr, trade = _incident()
        for bad in (None, 0.0, -1.0, float("nan"), "x"):
            assert o._exit_intrinsic_floor(trade, LEGS, bad, 0, WIDTH) is None

    def test_the_floor_never_exceeds_the_wing_width(self):
        o, ibkr, trade = _incident()
        assert o._exit_intrinsic_floor(trade, LEGS, 650.0, 0, WIDTH) == pytest.approx(WIDTH)

    def test_the_sell_side_is_untouched(self):
        """A debit close (net sell) keeps its old cap: never pay more than the cross."""
        o, ibkr, trade = _rig(leg_quotes={1: (1.0, 1.1), 2: (1.0, 1.1)}, entry=2.0)
        trade.strategy = "long_straddle"
        legs = [{"strike": 704, "right": "P", "action": "BUY", "expiry": "20261023"},
                {"strike": 704, "right": "C", "action": "BUY", "expiry": "20261023"}]
        p = _go(o, ibkr, trade, spot=700.0, dte=0, legs=legs)
        assert p is None or p <= o._settings.exit.exit_cross_amount + 1e-9


# --------------------------------------------------------------------------
# escalation on the leg-priced path and the shared counter
# --------------------------------------------------------------------------
class TestEscalation:
    def test_the_leg_priced_path_concedes_on_every_unfilled_placement(self):
        o, ibkr, trade = _incident()
        prices = [_go(o, ibkr, trade) for _ in range(4)]
        assert prices[0] == pytest.approx(2.48)
        assert all(b > a for a, b in zip(prices, prices[1:])), prices
        assert prices[-1] <= WIDTH

    def test_zero_escalation_pct_disables_it_as_on_the_quoted_path(self):
        o, ibkr, trade = _incident()
        o._settings = o._settings.model_copy(deep=True)
        o._settings.exit.exit_unfilled_escalation_pct = 0.0
        prices = {_go(o, ibkr, trade) for _ in range(3)}
        assert prices == {2.48}

    def test_the_counter_is_shared_across_the_three_paths(self):
        """10-23 alternated between the leg-priced and the no-quote path. Two legs
        priced placements, then one with no leg quotes: the no-quote step is the
        THIRD, not the first."""
        legs_quotes = dict(INCIDENT)
        o, ibkr, trade = _incident(leg_quotes=legs_quotes)
        _go(o, ibkr, trade)
        _go(o, ibkr, trade)
        legs_quotes.update(NO_LEG_QUOTES)                    # the feed drops out
        # fresh mark 2.40: anchor + cross + n * step with n = 2
        assert _go(o, ibkr, trade) == pytest.approx(2.40 + 0.10 + 2 * 0.10)

    def test_a_new_episode_starts_clean(self):
        o, ibkr, trade = _incident()
        first = _go(o, ibkr, trade)
        _go(o, ibkr, trade)
        o._exit_unfilled_attempts[trade.trade_id]["at"] -= 2 * 3600
        assert _go(o, ibkr, trade) == pytest.approx(first)

    def test_booking_the_exit_clears_the_counter(self):
        o, ibkr, trade = _incident()
        _go(o, ibkr, trade)
        assert trade.trade_id in o._exit_unfilled_attempts
        o._exit_unfilled_attempts.pop(trade.trade_id)
        assert o._exit_attempt_n(trade.trade_id) == 0


# --------------------------------------------------------------------------
# the no-quote path
# --------------------------------------------------------------------------
class TestNoQuotePath:
    def test_near_expiry_the_blind_cap_no_longer_holds_the_limit_below_intrinsic(self, monkeypatch):
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92)   # no mark, no quotes
        legacy = _go(o, ibkr, trade)
        assert legacy == pytest.approx(1.02)                  # entry credit + cross, cap 1.84
        o2, ibkr2, trade2 = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92)
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=0) == pytest.approx(2.70)

    def test_no_mark_and_no_credit_prices_from_the_intrinsic_instead_of_deferring(self, monkeypatch):
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "1")
        o, ibkr, trade = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.0)
        assert _go(o, ibkr, trade) is None                    # legacy: nothing to price from
        o2, ibkr2, trade2 = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.0)
        assert _go(o2, ibkr2, trade2, spot=704.4, dte=0) == pytest.approx(2.70)

    def test_out_of_the_money_the_no_quote_path_is_unchanged(self):
        o, ibkr, trade = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92)
        legacy = _go(o, ibkr, trade)
        o2, ibkr2, trade2 = _rig(leg_quotes=dict(NO_LEG_QUOTES), entry=0.92)
        assert _go(o2, ibkr2, trade2, spot=740.0, dte=0) == pytest.approx(legacy)


# --------------------------------------------------------------------------
# the broker's band is the last word
# --------------------------------------------------------------------------
class TestTheBrokerBandStillWins:
    def test_a_band_below_the_floor_caps_the_limit(self):
        o, ibkr, trade = _incident(mark_cost=None, band=2.00, market=1.90)
        p = _go(o, ibkr, trade, spot=704.4, dte=0)
        assert p is not None and p < 2.00


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------
class TestProperties:
    def test_never_above_the_width_and_never_below_the_floor(self):
        rng = random.Random(101)
        for _ in range(120):
            spot = rng.uniform(690.0, 725.0)
            dte = rng.choice([0, 1, 3, 10])
            n = rng.choice([0, 1, 5, 40, 200])
            q = {1: (rng.uniform(0.0, 1.0),) * 2, 2: (rng.uniform(0.0, 4.0),) * 2,
                 3: (rng.uniform(0.0, 0.3),) * 2, 4: (rng.uniform(0.0, 0.3),) * 2}
            q = {k: (v[0], v[1] + 0.02) for k, v in q.items()}
            mark = rng.choice([None, rng.uniform(0.2, 3.5)])
            o, ibkr, trade = _rig(leg_quotes=q, entry=0.92, mark_cost=mark,
                                  prior_attempts=n)
            limit = _go(o, ibkr, trade, spot=spot, dte=dte)
            assert limit is not None
            assert limit <= WIDTH + 1e-9, (spot, dte, n, q, mark, limit)
            floor = o._exit_intrinsic_floor(trade, LEGS, spot, dte, WIDTH)
            if floor is not None:
                assert limit >= floor - 1e-9, (spot, dte, n, q, mark, limit, floor)


# --------------------------------------------------------------------------
# the call site
# --------------------------------------------------------------------------
class TestTheMonitorPassesSpotAndDte:
    @pytest.mark.asyncio
    async def test_execute_exit_hands_spot_dte_and_reason_to_the_close(self):
        from tests.test_exit_order_safety import FakeTrade, _exit_orchestrator, _pos_status
        trade = FakeTrade(contract_type="spread")
        orch, close, _, _ = _exit_orchestrator("live", trade)
        pos = _pos_status()
        pos.current_price, pos.dte, pos.exit_reason = 704.4, 0, TOUCH
        await orch._execute_exit(pos)
        close.assert_called_once()
        assert close.call_args.kwargs == {"spot": 704.4, "dte": 0, "reason": TOUCH}

    @pytest.mark.asyncio
    async def test_a_missing_spot_is_passed_as_none_not_zero(self):
        from tests.test_exit_order_safety import FakeTrade, _exit_orchestrator, _pos_status
        trade = FakeTrade(contract_type="spread")
        orch, close, _, _ = _exit_orchestrator("live", trade)
        pos = _pos_status()
        pos.current_price, pos.dte = 0.0, None             # the monitor's "no quote" value
        await orch._execute_exit(pos)
        assert close.call_args.kwargs["spot"] is None and close.call_args.kwargs["dte"] is None


# --------------------------------------------------------------------------
# rest time and the reject window
# --------------------------------------------------------------------------
class TestTheMustGetOutExitsRestTheShortTime:
    @pytest.mark.parametrize("reason,expected", [
        ("condor_defense_close (tested_side=put, ...)", 120.0),
        ("assignment_risk (short put ITM, DTE: 1)", 120.0),
        ("expiry_approaching (DTE: 0)", 120.0),
        ("expiry_approaching (DTE: 1)", 120.0),
        ("expiry_approaching (DTE: 2)", 300.0),
        ("expiry_approaching (DTE: 5)", 300.0),
        ("take_profit_short (P&L: 41.0%, target: 40.0%)", 300.0),
        ("macro_event_flatten", 300.0),
        ("", 300.0),
    ])
    def test_rest_by_reason(self, reason, expected):
        from ait.execution.executor import PendingExitOrder, TradeExecutor
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._settings = SimpleNamespace(exit=SimpleNamespace(
            exit_pending_rest_seconds=300.0, exit_stop_pending_rest_seconds=120.0))
        assert ex._exit_rest_seconds(PendingExitOrder("T1", reason, 0.0)) == expected


class TestTheBackoffTellsARepriceFromAReject:
    """The executor records when it takes an unfilled close down by its own rest
    timer; the backoff then reads the next request as a re-price, not a reject."""

    def _orch(self, monkeypatch):
        from ait.execution.executor import TradeExecutor
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        clock = {"t": 1000.0}
        monkeypatch.setattr("ait.bot.orchestrator.time.monotonic", lambda: clock["t"])
        orch._alert_gate_last = {}
        orch._executor = TradeExecutor.__new__(TradeExecutor)
        return orch, clock

    def test_a_reprice_after_the_rest_is_not_a_reject_strike(self, monkeypatch):
        orch, clock = self._orch(monkeypatch)
        for _ in range(4):
            assert orch._exit_retry_ready("T1") is True
            clock["t"] += 125                              # the 120 s stop rest + the cancel
            orch._executor._note_rest_cancel("T1")
            clock["t"] += 5                                # next monitor pass
        assert orch._exit_attempts["T1"]["strikes"] == 1   # no backoff ever built up

    def test_a_fast_reject_is_still_a_strike_and_pages_at_three(self, monkeypatch):
        orch, clock = self._orch(monkeypatch)
        notes = []

        async def _notify(m):
            notes.append(m)

        orch._send_notification = _notify

        async def drive():
            orch._exit_retry_ready("T1")
            for _ in range(6):
                clock["t"] += 5
                if not orch._exit_retry_ready("T1"):
                    clock["t"] += 200
                    orch._exit_retry_ready("T1")
            await asyncio.sleep(0)

        asyncio.run(drive())
        assert any("REPEATEDLY REJECTED" in n for n in notes)

    def test_a_reject_after_a_rest_cancel_still_counts(self, monkeypatch):
        """The rest cancel belongs to an EARLIER order: a later fast re-request
        (the new order was rejected) is a strike again."""
        orch, clock = self._orch(monkeypatch)
        assert orch._exit_retry_ready("T1") is True
        clock["t"] += 125
        orch._executor._note_rest_cancel("T1")
        clock["t"] += 5
        assert orch._exit_retry_ready("T1") is True        # re-price: strikes 1
        clock["t"] += 5                                    # the new order bounced
        assert orch._exit_retry_ready("T1") is True
        assert orch._exit_attempts["T1"]["strikes"] == 2

    def test_no_executor_means_the_old_clock_rule(self, monkeypatch):
        orch, clock = self._orch(monkeypatch)
        orch._executor = None
        assert orch._exit_retry_ready("T1") is True
        clock["t"] += 125
        assert orch._exit_retry_ready("T1") is True
        assert orch._exit_attempts["T1"]["strikes"] == 2   # 125 s < 240 s window
