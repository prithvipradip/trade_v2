"""Wires condor_defense.py (W56) into _evaluate_position -- DECISION-ONLY,
per the operator's explicit choice this wave: a CLOSE or ROLL verdict from
evaluate_condor_defense is reported through PositionStatus.action/roll_side/
action_reason and NEVER touches should_exit. Flipping condor_defense.enabled
to True changes nothing about what a live executor actually does -- only
what gets reported. That decision-only property is the centerpiece this
file proves, not just the mapping mechanics (test_w56_condor_defense.py
already covers evaluate_condor_defense itself in isolation).

Reuses test_w56_condor_defense.py's strike-for-delta/pricing/condor-legs
helpers (same brentq-against-py_vollib-delta convention, so each fixture's
solved |delta| provably lands in its intended band) rather than duplicating
them -- an established cross-test-file convention in this codebase
(test_r16_tail_fixes.py imports from test_r20_research_validity.py).

No inspect.getsource.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

pytest.importorskip("py_vollib")
pytest.importorskip("scipy")

from ait.config.settings import CondorDefenseConfig  # noqa: E402
from ait.data.market_data import Quote  # noqa: E402
from ait.execution.condor_defense import DefenseAction  # noqa: E402
from ait.execution.portfolio import (  # noqa: E402
    PortfolioManager,
    PositionAction,
    _map_condor_defense_action,
)
from tests.test_w56_condor_defense import (  # noqa: E402 — shared fixtures
    SPOT,
    _expiry_for_dte,
    _put_tested_condor,
)


class TestMapCondorDefenseAction:
    @pytest.mark.parametrize("defense_action,expected", [
        (DefenseAction.HOLD, PositionAction.HOLD),
        (DefenseAction.CLOSE, PositionAction.CLOSE),
        (DefenseAction.ROLL_UNTESTED, PositionAction.ROLL),
        (DefenseAction.ROLL_TESTED, PositionAction.ROLL),
    ])
    def test_exhaustive(self, defense_action, expected):
        assert _map_condor_defense_action(defense_action) == expected


def _manager(spot: float, unrealized: float, *, condor_defense_config=None) -> PortfolioManager:
    """Real PortfolioManager via __new__ with only collaborators mocked --
    same proven harness test_credit_exits.py/test_position_action_output.py
    use. `unrealized` is a fixed, deliberately small P&L that never trips
    stop/take-profit -- condor_defense's own solve is driven by the REAL
    option_marks passed straight into _evaluate_position, decoupled from
    the P&L override."""
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
            symbol=symbol, bid=round(spot - 0.01, 2), ask=round(spot + 0.01, 2),
            last=spot, volume=1_000_000,
            timestamp=datetime.now() + timedelta(seconds=tick["n"]),
        )
    mgr._market_data.get_quote = _quote

    async def _price(symbol):
        return spot
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
    mgr._option_position_unrealized = lambda trade, marks: unrealized
    mgr._get_position_delta = lambda trade: None
    if condor_defense_config is not None:
        mgr._condor_defense_config = condor_defense_config
    return mgr


class TestCloseIsActionable:
    """CLOSE is now folded into should_exit -- reuses the existing,
    already-audited close path, exactly like every other exit rule. ROLL
    stays decision-only (TestDecisionOnlyWiring below)."""

    async def test_close_verdict_sets_should_exit(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.45, 18)  # >= close_delta_threshold
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.should_exit is True
        assert "condor_defense_close" in status.exit_reason
        assert "tested_side=P" in status.exit_reason
        assert status.action == PositionAction.CLOSE
        assert status.roll_side is None
        assert status.action_reason is None, \
            "action_reason is reserved for the decision-only ROLL case now"

    async def test_close_verdict_vetoed_by_pdt_reports_hold_not_close(self, monkeypatch):
        """The direct regression guard for a real design issue found while
        implementing this: folding CLOSE into should_exit means rule 5's
        PDT veto (further down _evaluate_position) can reset should_exit
        back to False for THIS close too -- the return site must reflect
        that (action=HOLD), not report a stale CLOSE that didn't actually
        happen."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.45, 18)
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True))
        mgr._pdt_guard.would_be_day_trade.return_value = True
        mgr._pdt_guard.can_day_trade.return_value = False
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.should_exit is False
        assert status.exit_reason == "exit_blocked_by_pdt"
        assert status.action == PositionAction.HOLD
        assert status.action_reason is None


class TestDecisionOnlyWiring:
    """ROLL stays decision-only -- it has no executor yet (W57)."""

    async def test_roll_untested_verdict(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.27, 18)  # untested band + window
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True, side="both"))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.should_exit is False
        assert status.action == PositionAction.ROLL
        assert status.roll_side == "P"  # _put_tested_condor's tested side
        assert status.action_reason == "condor_defense_roll_untested_band"

    async def test_roll_tested_verdict(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.35, 25)  # tested band, dte>=22
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True, side="both"))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.should_exit is False
        assert status.action == PositionAction.ROLL
        assert status.roll_side == "P"
        assert status.action_reason == "condor_defense_roll_tested_band"


class TestFallsThroughToPlainDerivation:
    """Cases where condor_defense's own verdict is HOLD (or it's never
    computed at all) must be byte-identical to the previous wave's
    behavior: action_reason stays None."""

    async def test_below_monitor_band_falls_through(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.15, 18)  # below monitor_delta_max
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.action == PositionAction.HOLD
        assert status.action_reason is None
        assert status.roll_side is None

    async def test_disabled_config_falls_through(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.45, 18)  # would CLOSE if enabled
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=False))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.action == PositionAction.HOLD
        assert status.action_reason is None

    async def test_config_never_set_on_manager_falls_through(self, monkeypatch):
        """The regression proof: every one of the 311+13 pre-existing
        tests constructs PortfolioManager via __new__ without ever setting
        _condor_defense_config -- this must keep behaving exactly as
        before this wave, not raise AttributeError."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.45, 18)
        mgr = _manager(SPOT, unrealized=5.0, condor_defense_config=None)
        assert not hasattr(mgr, "_condor_defense_config")
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.action == PositionAction.HOLD
        assert status.action_reason is None

    async def test_non_iron_condor_strategy_falls_through(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        trade, marks, dte = _put_tested_condor(0.45, 18)
        trade.strategy = "short_strangle"
        trade.contract_type = "short_strangle"
        mgr = _manager(SPOT, unrealized=5.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.action == PositionAction.HOLD
        assert status.action_reason is None

    async def test_earlier_rule_already_exiting_blocks_condor_defense(self, monkeypatch):
        """A touch-stop exit (rule 1b) must WIN outright -- condor_defense
        must not even be computed (the not-should_exit gate), so no
        action_reason leaks in alongside an unrelated should_exit=True."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        # A 0.45-delta short put is close to, but not necessarily past, the
        # strike -- force an unambiguous touch by driving spot itself
        # through the short strike after building the fixture at a stable
        # reference spot for pricing/legs.
        trade, marks, dte = _put_tested_condor(0.45, 18)
        import json
        legs = json.loads(trade.legs)
        short_put_strike = max(
            leg["strike"] for leg in legs
            if leg["right"] == "P" and leg["action"] == "SELL")
        touch_spot = short_put_strike - 5.0  # comfortably through the short put
        mgr = _manager(touch_spot, unrealized=-20.0,
                       condor_defense_config=CondorDefenseConfig(enabled=True))
        trade.expiry = _expiry_for_dte(dte)
        status = await mgr._evaluate_position(trade, marks)
        assert status.should_exit is True
        assert "short_strike_touch" in status.exit_reason
        assert status.action == PositionAction.CLOSE
        assert status.action_reason is None, \
            "condor_defense must not even run once should_exit is already True"
