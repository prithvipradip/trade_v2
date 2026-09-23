"""W52 (rule-3 multi-leg ITM diagnosis, PLAN.md finding [2] in the 2026-09-22
ROADMAP entry): portfolio.py rule 3 ("Assignment risk — ITM shorts on expiry
day") is supposed to say which short leg is ITM at DTE<=1. It never could
for a multi-leg credit trade: executor.py's _get_contract_type returns
"iron_condor"/"spread" for any multi-leg signal, never "put"/"call", and
rule 3's ITM check branched on exactly those two strings. Every multi-leg
credit close at DTE<=1 fell through to the generic expiry_approaching
label, whether or not it was genuinely ITM.

should_exit was ALREADY True in this branch either way (the fallback
guarantees it) — this is a diagnosis fix, not a protection fix.
TestShouldExitInvariant pins that should_exit is unchanged by this wave.

Every test drives the real _evaluate_position / _short_strikes_from_legs.
No inspect.getsource.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.data.market_data import Quote
from ait.execution.executor import TradeExecutor
from ait.execution.portfolio import PortfolioManager
from ait.utils.time import today_et

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])

STRANGLE_LEGS = json.dumps([
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
])


@dataclass
class FakeTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "iron_condor"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-W52"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = 98.0
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, unrealized: float | None = 5.0) -> PortfolioManager:
    """Real PortfolioManager via __new__ with only collaborators mocked —
    same proven harness as test_credit_exits.py's _manager."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()

    tick = {"n": 0}

    async def _quote(symbol):
        tick["n"] += 1
        return Quote(
            symbol=symbol,
            bid=round(spot - 0.01, 2),
            ask=round(spot + 0.01, 2),
            last=spot,
            volume=1_000_000,
            timestamp=datetime.now() + timedelta(seconds=tick["n"]),
        )
    mgr._market_data.get_quote = _quote

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price
    mgr._exit_config = ExitConfig()
    mgr._earnings = None
    mgr._economic_cal = None

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks: unrealized
    mgr._get_position_delta = lambda trade: None
    return mgr


def _trade(dte: int, **overrides) -> FakeTrade:
    """A credit trade expiring `dte` ET calendar days out.

    Built from today_et(), not date.today(): the code under test computes
    dte against today_et(), and on a host whose LOCAL date differs from the
    ET date (e.g. late evening UTC, where the host date is already
    tomorrow), a date.today()-built fixture silently drifts by a day right
    at the dte<=1 boundary this file exists to exercise. Same flake class
    W52a #2 fixed in test_r17_live_fixes.py.
    """
    kwargs: dict = {"expiry": (today_et() + timedelta(days=dte)).isoformat()}
    kwargs.update(overrides)
    t = FakeTrade(**kwargs)
    t.direction = SimpleNamespace(value="neutral")
    return t


@pytest.fixture(autouse=True)
def _touch_off(monkeypatch):
    """Isolate rule 3 from rule 1b: both key off the same short strikes, but
    1b checks at ANY dte, so without this every ITM fixture below would
    touch-stop out before reaching rule 3's dte<=1 branch. Same isolation
    test_w51_ex_div_assignment.py uses to test rule 3e."""
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")


# ---------------------------------------------------------------------------
# The helper, tested directly
# ---------------------------------------------------------------------------

class TestShortStrikesFromLegs:
    def test_condor_gives_both_short_strikes(self):
        trade = _trade(dte=20)
        assert PortfolioManager._short_strikes_from_legs(trade) == (98.0, 102.0)

    def test_ties_resolve_closest_to_the_money(self):
        legs = json.dumps([
            {"strike": 96.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
            {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
            {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
            {"strike": 104.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
        ])
        trade = _trade(dte=20, legs=legs)
        assert PortfolioManager._short_strikes_from_legs(trade) == (98.0, 102.0)

    def test_empty_legs_gives_none_none(self):
        trade = _trade(dte=20, legs="[]")
        assert PortfolioManager._short_strikes_from_legs(trade) == (None, None)

    def test_malformed_json_gives_none_none(self):
        trade = _trade(dte=20, legs="not json")
        assert PortfolioManager._short_strikes_from_legs(trade) == (None, None)

    def test_no_legs_at_all_gives_none_none(self):
        trade = _trade(dte=20, legs=None)
        assert PortfolioManager._short_strikes_from_legs(trade) == (None, None)


# ---------------------------------------------------------------------------
# The bug, reproduced then closed: multi-leg credit trades at DTE<=1
# ---------------------------------------------------------------------------

class TestCondorNamesTheITMSide:
    async def test_put_side_itm_at_dte_1(self):
        mgr = _manager(spot=97.50)  # < short put 98.0
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.exit_reason.startswith("assignment_risk"), status.exit_reason
        assert "put" in status.exit_reason and "98.00" in status.exit_reason

    async def test_call_side_itm_at_dte_1(self):
        mgr = _manager(spot=102.50)  # > short call 102.0
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.exit_reason.startswith("assignment_risk"), status.exit_reason
        assert "call" in status.exit_reason and "102.00" in status.exit_reason

    async def test_put_side_itm_at_dte_0(self):
        mgr = _manager(spot=97.50)
        status = await mgr._evaluate_position(_trade(dte=0))
        assert status.should_exit
        assert status.exit_reason.startswith("assignment_risk"), status.exit_reason

    async def test_short_strangle_names_the_put_side(self):
        """Finding [2]: the same gap applies to short_strangle (2-leg,
        contract_type='spread'), not just iron_condor."""
        mgr = _manager(spot=97.50)
        trade = _trade(dte=1, contract_type="spread", strategy="short_strangle",
                        legs=STRANGLE_LEGS)
        status = await mgr._evaluate_position(trade)
        assert status.should_exit
        assert status.exit_reason.startswith("assignment_risk"), status.exit_reason
        assert "put" in status.exit_reason

    async def test_degraded_quote_with_touch_pending_still_gets_the_label(
            self, monkeypatch):
        """Realistic path under the LIVE default config (touch stop ON): a
        stale quote reads "degraded" on first look (below the validator's
        staleness bound), so rule 1b's touch stop waits for a second
        agreeing tick (ExitConfig.touch_confirm_ticks=2) instead of firing
        immediately. On dte<=1 that gap is exactly where rule 3 used to
        mislabel the close — proves the fix is reachable with touch stop
        enabled, not just in the isolated tests above."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        stale_ts = datetime.now() - timedelta(seconds=600)

        async def _stale_quote(symbol):
            return Quote(symbol=symbol, bid=97.49, ask=97.51, last=97.50,
                         volume=1_000_000, timestamp=stale_ts)

        mgr = _manager(spot=97.50)
        mgr._market_data.get_quote = _stale_quote
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.exit_reason.startswith("assignment_risk"), status.exit_reason


class TestSurvivesContractTypeChanges:
    """Class guard (CLAUDE.md house rule: catch the next instance of a bug
    class): rule 3's new branch keys on `contract_type not in ("put",
    "call")`, not on the specific strings _get_contract_type happens to
    return today — so a future leg shape that earns a new contract_type
    string is covered without another fix here. Proven against the REAL
    _get_contract_type, not a hand-copied literal."""

    @pytest.mark.parametrize("leg_count", [2, 3, 4])
    async def test_any_non_put_call_shape_gets_the_itm_label(self, leg_count):
        all_legs = [
            {"strike": 98.0, "right": "P", "action": "SELL"},
            {"strike": 102.0, "right": "C", "action": "SELL"},
            {"strike": 95.0, "right": "P", "action": "BUY"},
            {"strike": 105.0, "right": "C", "action": "BUY"},
        ][:leg_count]
        signal = SimpleNamespace(
            legs=[{"action": leg["action"]} for leg in all_legs], contract=None)
        contract_type = TradeExecutor._get_contract_type(signal)
        assert contract_type not in ("put", "call")  # guards the test itself

        legs_json = json.dumps([{**leg, "expiry": "2026-12-18"} for leg in all_legs])
        mgr = _manager(spot=97.50)
        trade = _trade(dte=1, contract_type=contract_type, legs=legs_json)
        status = await mgr._evaluate_position(trade)
        assert status.exit_reason.startswith("assignment_risk"), (
            contract_type, status.exit_reason)


class TestHostTimezoneIndependence:
    """W52a #2 was exactly this flake class: a fixture built from
    server-local date.today() at a boundary the code itself evaluates in
    ET. This file's _trade() helper uses today_et(); pin a far-from-ET host
    clock and confirm the dte<=1 boundary still lands correctly."""

    async def test_dte_boundary_correct_under_a_non_et_host_clock(self, monkeypatch):
        monkeypatch.setenv("TZ", "Pacific/Kiritimati")  # UTC+14
        if hasattr(_time, "tzset"):
            _time.tzset()
        try:
            mgr = _manager(spot=97.50)
            status = await mgr._evaluate_position(_trade(dte=1))
            assert status.exit_reason.startswith("assignment_risk"), status.exit_reason
        finally:
            monkeypatch.undo()
            if hasattr(_time, "tzset"):
                _time.tzset()


# ---------------------------------------------------------------------------
# Controls: must read the same before and after the fix
# ---------------------------------------------------------------------------

class TestControlsUnchanged:
    async def test_condor_inside_strikes_still_expiry_approaching(self):
        mgr = _manager(spot=100.0)
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.exit_reason == "expiry_approaching (DTE: 1)"

    async def test_spot_exactly_on_the_strike_is_not_itm(self):
        """Strict inequality preserved: touching the strike is not ITM."""
        mgr = _manager(spot=98.0)
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.exit_reason == "expiry_approaching (DTE: 1)"

    async def test_single_leg_cash_secured_put_unchanged(self):
        mgr = _manager(spot=97.0)
        trade = _trade(dte=1, contract_type="put", strategy="cash_secured_put",
                        legs="[]", strike=98.0)
        status = await mgr._evaluate_position(trade)
        assert status.exit_reason == (
            "assignment_risk (short put ITM, underlying=$97.00, strike=$98.00)")

    async def test_single_leg_covered_call_unchanged(self):
        mgr = _manager(spot=103.0)
        trade = _trade(dte=1, contract_type="call", strategy="covered_call",
                        legs="[]", strike=102.0)
        status = await mgr._evaluate_position(trade)
        assert status.exit_reason == (
            "assignment_risk (short call ITM, underlying=$103.00, strike=$102.00)")

    async def test_malformed_legs_falls_through_safely(self):
        """Fail-safe, not fail-exit: a legs-parse error must not crash
        check_positions — same guarantee rule 1b/3e give."""
        mgr = _manager(spot=97.50)
        status = await mgr._evaluate_position(_trade(dte=1, legs="not json"))
        assert status.should_exit
        assert status.exit_reason == "expiry_approaching (DTE: 1)"

    async def test_missing_underlying_quote_still_expiry_approaching(self):
        async def _no_quote(symbol):
            return None
        mgr = _manager(spot=97.50)
        mgr._market_data.get_quote = _no_quote
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert "expiry_approaching" in status.exit_reason


class TestShouldExitInvariant:
    """The fix changes the LABEL only. should_exit was already True in
    every dte<=1 credit-position case before this wave; it must still be."""

    @pytest.mark.parametrize("spot", [97.50, 100.0, 102.50])
    async def test_should_exit_true_regardless_of_itm(self, spot):
        mgr = _manager(spot=spot)
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit is True
