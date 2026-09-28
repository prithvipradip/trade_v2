"""check_positions non-boolean decision-output extension (shared design:
PLAN.md's 2026-09-22 roadmap entry, "design once, use for both" W54's
rule 3/3e roll-vs-close dispatch and W56's condor_defense). Purely
additive -- no existing exit rule's behavior changes; PositionStatus.action
is always derived from should_exit/partial_exit_quantity today
(_derive_position_action), and nothing produces PositionAction.ROLL yet.

_derive_position_action is tested in isolation (pure function, no
PortfolioManager needed). A representative sample is then driven through
the REAL _evaluate_position (same __new__-with-mocked-collaborators harness
test_credit_exits.py uses) to prove the wiring at the return site, not just
the pure function. A final illustrative test proves the shape is sufficient
for a future roll decision without needing a second structural change.

No inspect.getsource.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.data.market_data import Quote
from ait.execution.portfolio import (
    PortfolioManager,
    PositionAction,
    PositionStatus,
    _derive_position_action,
)

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


# ---------------------------------------------------------------------------
# _derive_position_action -- pure function, exhaustive
# ---------------------------------------------------------------------------

class TestDerivePositionAction:
    @pytest.mark.parametrize("should_exit,partial_qty,expected", [
        (True, 0, PositionAction.CLOSE),
        (True, 3, PositionAction.CLOSE),   # should_exit wins even if partial also set
        (False, 3, PositionAction.PARTIAL_CLOSE),
        (False, 0, PositionAction.HOLD),
    ])
    def test_all_four_combinations(self, should_exit, partial_qty, expected):
        assert _derive_position_action(should_exit, partial_qty) == expected

    def test_never_returns_roll(self):
        """No caller produces ROLL yet -- confirmed by construction, not by
        inspecting source: every combination of the two real inputs is
        covered above and none of them is ROLL."""
        for should_exit in (True, False):
            for partial_qty in (0, 1, 5):
                assert _derive_position_action(should_exit, partial_qty) != PositionAction.ROLL


# ---------------------------------------------------------------------------
# PositionStatus -- backward compatibility, the illustrative roll case
# ---------------------------------------------------------------------------

class TestPositionStatusShape:
    def _status(self, **overrides) -> PositionStatus:
        kwargs = dict(
            trade_id="T", symbol="SPY", strategy="iron_condor", quantity=1,
            entry_price=1.0, current_price=1.0, unrealized_pnl=0.0, pnl_pct=0.0,
            high_water_mark=0.0, dte=10, should_exit=False, exit_reason="",
        )
        kwargs.update(overrides)
        return PositionStatus(**kwargs)

    def test_defaults_are_backward_compatible(self):
        """Every existing call site constructs PositionStatus without the
        two new kwargs -- both must default to values that mean 'nothing
        new here'."""
        status = self._status()
        assert status.action == PositionAction.HOLD
        assert status.roll_side is None

    def test_illustrative_future_roll_decision_fits_without_a_second_change(self):
        """Not wiring anything real -- proves the shape a future rule-3/3e
        or condor_defense caller would use, so no second structural change
        is needed when that wiring actually happens."""
        status = self._status(
            should_exit=False, exit_reason="assignment_risk_roll",
            action=PositionAction.ROLL, roll_side="P",
        )
        assert status.action == PositionAction.ROLL
        assert status.roll_side == "P"
        # Every existing field stays independently readable alongside the
        # new ones -- no existing consumer's read of should_exit/exit_reason
        # needs to change to coexist with a ROLL decision.
        assert status.should_exit is False
        assert status.exit_reason == "assignment_risk_roll"


# ---------------------------------------------------------------------------
# A representative sample through the REAL _evaluate_position -- proves the
# wiring at the return site, not just _derive_position_action in isolation.
# Harness matches test_credit_exits.py's proven __new__-with-mocked-
# collaborators pattern.
# ---------------------------------------------------------------------------

@dataclass
class FakeCondorTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-ACTION"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"


@dataclass
class FakeLongCallTrade:
    """A DEBIT strategy -- take_profit_long's ladder targets sit ABOVE the
    default partial_exit_levels[0]=0.5, unlike credit's take_profit_short
    (which floors at 0.50 too, so a partial-only credit scenario without
    also tripping full take-profit doesn't exist under default config)."""
    entry_price: float = 2.00
    quantity: int = 4
    contract_type: str = "call"
    strategy: str = "long_call"
    symbol: str = "SPY"
    trade_id: str = "T-PARTIAL"
    direction: object = None
    legs: str | None = None
    expiry: str | None = None
    strike: float | None = 100.0
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, unrealized: float | None) -> PortfolioManager:
    """Real PortfolioManager via __new__ with only collaborators mocked --
    the proven harness test_credit_exits.py uses."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._state.get_partial_exits.return_value = []
    mgr._market_data = MagicMock()

    tick = {"n": 0}

    async def _quote(symbol):
        tick["n"] += 1
        return Quote(
            symbol=symbol, bid=round(spot - 0.01, 2), ask=round(spot + 0.01, 2),
            last=spot, volume=1_000_000,
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


def _condor(dte: int) -> FakeCondorTrade:
    t = FakeCondorTrade(expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="neutral")
    return t


def _long_call(dte: int) -> FakeLongCallTrade:
    t = FakeLongCallTrade(expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="long")
    return t


class TestActionMatchesDerivationAcrossRealRules:
    """For each scenario, action/roll_side must equal exactly what
    _derive_position_action independently computes from that SAME result's
    should_exit/partial_exit_quantity -- proving the wiring at the return
    site of the real _evaluate_position, not a hand reimplementation."""

    def _assert_consistent(self, status):
        expected = _derive_position_action(status.should_exit, status.partial_exit_quantity)
        assert status.action == expected
        assert status.roll_side is None  # nothing produces ROLL yet

    async def test_touch_stop_exit(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=97.5, unrealized=-20.0)
        status = await mgr._evaluate_position(_condor(dte=20))
        assert status.should_exit and "short_strike_touch" in status.exit_reason
        assert status.action == PositionAction.CLOSE
        self._assert_consistent(status)

    async def test_take_profit_exit(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=55.0)
        status = await mgr._evaluate_position(_condor(dte=20))
        assert status.should_exit and "take_profit_short" in status.exit_reason
        assert status.action == PositionAction.CLOSE
        self._assert_consistent(status)

    async def test_expiry_approaching_exit(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=5.0)
        status = await mgr._evaluate_position(_condor(dte=3))
        assert status.should_exit and "expiry_approaching" in status.exit_reason
        assert status.action == PositionAction.CLOSE
        self._assert_consistent(status)

    async def test_healthy_hold(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=10.0)
        status = await mgr._evaluate_position(_condor(dte=20))
        assert not status.should_exit and status.partial_exit_quantity == 0
        assert status.action == PositionAction.HOLD
        self._assert_consistent(status)

    async def test_partial_exit_only_no_full_close(self, monkeypatch):
        """A debit long_call at +50% (>= partial_exit_levels[0]=0.5, below
        take_profit_long's DTE>20 target of 1.00) -- partial fires alone."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=400.0)  # 400 / (2.00*4*100) = 0.50
        status = await mgr._evaluate_position(_long_call(dte=25))
        assert not status.should_exit
        assert status.partial_exit_quantity > 0
        assert status.action == PositionAction.PARTIAL_CLOSE
        self._assert_consistent(status)

    async def test_pdt_blocked_exit_reflects_post_veto_state(self, monkeypatch):
        """The PDT veto sets should_exit=False AFTER a touch stop wanted to
        fire -- action must reflect that POST-veto HOLD, not the pre-veto
        CLOSE intent (portfolio.py's own rule 5 does this via should_exit=
        False before the return; _derive_position_action just needs to see
        the already-vetoed value, which it does since it runs at the same
        return site)."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=97.5, unrealized=-20.0)  # would touch-stop
        mgr._pdt_guard.would_be_day_trade.return_value = True
        mgr._pdt_guard.can_day_trade.return_value = False
        status = await mgr._evaluate_position(_condor(dte=20))
        assert not status.should_exit
        assert status.exit_reason == "exit_blocked_by_pdt"
        assert status.action == PositionAction.HOLD
        self._assert_consistent(status)
