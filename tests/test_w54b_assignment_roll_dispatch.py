"""W54b: wire rule 3/rule 3e to the close-or-roll comparison.

Part A drives portfolio.py's real _evaluate_position (rule 3/rule 3e's new
assignment_roll_side signal, the PDT-veto reset, the arbitration with
condor_defense) -- same PortfolioManager.__new__ harness test_w52_multileg_
itm_diagnosis.py/test_w51_ex_div_assignment.py already use.

Part B drives orchestrator.py's new _compare_close_or_roll/_gather_ev_leg_
inputs directly, and confirms _execute_roll's prepared/roll_target reuse
(no second broker round-trip).

No inspect.getsource.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.config.settings import (
    AssignmentResponseConfig,
    CondorDefenseConfig,
    ExitConfig,
)
from ait.data.market_data import Quote
from ait.data.options_chain import OptionContract
from ait.execution.portfolio import PortfolioManager, PositionAction, PositionStatus
from ait.execution.roll_selector import RollTarget
from ait.utils.time import today_et

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


@dataclass
class FakeTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "iron_condor"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-W54B"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = 98.0
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, unrealized: float | None = 5.0,
            assignment_response_config=None, condor_defense_config=None) -> PortfolioManager:
    """Real PortfolioManager via __new__ -- same proven harness test_w52_
    multileg_itm_diagnosis.py's own _manager uses."""
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
    mgr._assignment_response_config = assignment_response_config or AssignmentResponseConfig()
    mgr._condor_defense_config = condor_defense_config
    return mgr


def _trade(dte: int, **overrides) -> FakeTrade:
    kwargs: dict = {"expiry": (today_et() + timedelta(days=dte)).isoformat()}
    kwargs.update(overrides)
    t = FakeTrade(**kwargs)
    t.direction = SimpleNamespace(value="neutral")
    return t


@pytest.fixture(autouse=True)
def _touch_off(monkeypatch):
    """Isolate rule 3 from rule 1b -- same isolation test_w52_multileg_itm_
    diagnosis.py/test_w51_ex_div_assignment.py already use."""
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")


# ---------------------------------------------------------------------------
# Part A: portfolio.py rule 3/rule 3e dispatch
# ---------------------------------------------------------------------------

class TestRule3ByteIdenticalWhenCloseOnly:
    async def test_default_policy_reports_no_roll_candidate(self):
        mgr = _manager(spot=97.50)  # ITM put
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.roll_side is None
        assert status.action != PositionAction.ROLL
        assert status.exit_reason.startswith("assignment_risk")


class TestRule3ReportsRollCandidate:
    async def test_roll_only_reports_put_side(self):
        cfg = AssignmentResponseConfig(policy="roll_only")
        mgr = _manager(spot=97.50, assignment_response_config=cfg)  # ITM put
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit  # unchanged -- still a close candidate
        assert status.roll_side == "P"
        assert status.action == PositionAction.ROLL
        assert status.action_reason == "assignment_risk_roll_only"

    async def test_close_or_roll_reports_call_side(self):
        cfg = AssignmentResponseConfig(policy="close_or_roll")
        mgr = _manager(spot=102.50, assignment_response_config=cfg)  # ITM call
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.roll_side == "C"
        assert status.action_reason == "assignment_risk_close_or_roll"

    async def test_no_config_at_all_behaves_like_close_only(self):
        """getattr-guard: a PortfolioManager built without setting the
        attribute (e.g. an older test fixture) must not crash and must not
        report a roll candidate."""
        mgr = _manager(spot=97.50)
        del mgr._assignment_response_config
        status = await mgr._evaluate_position(_trade(dte=1))
        assert status.should_exit
        assert status.roll_side is None


class TestRule3ePreExDivReportsRollCandidate:
    async def test_roll_only_always_reports_call_side(self):
        """Rule 3e is calls-only by construction -- roll_side must always
        be 'C', no translation needed (unlike rule 3's put/call)."""
        cfg = AssignmentResponseConfig(policy="roll_only")
        mgr = _manager(spot=101.0, assignment_response_config=cfg)
        mgr._dividends = MagicMock()
        mgr._dividends.get_next_ex_div.return_value = SimpleNamespace(
            next_ex_div_date=today_et() + timedelta(days=1), expected_amount=1.00)
        marks = {("SPY", "20261218", 102.0, "C"): 0.10}  # extrinsic well below the dividend
        status = await mgr._evaluate_position(_trade(dte=20), option_marks=marks)
        assert status.should_exit
        assert status.exit_reason.startswith("pre_ex_div_assignment_risk")
        assert status.roll_side == "C"
        assert status.action_reason == "assignment_risk_roll_only"


class TestPdtVetoResetsRollCandidate:
    async def test_pdt_vetoed_position_reports_no_roll_candidate(self):
        """The real finding this wave's own research surfaced: without the
        reset, a PDT-vetoed position would report should_exit=False +
        roll_side=set -- EXACTLY the shape _partition_roll_positions (W61)
        treats as a condor_defense-style roll ready to place, bypassing
        the PDT protection entirely."""
        cfg = AssignmentResponseConfig(policy="roll_only")
        mgr = _manager(spot=97.50, assignment_response_config=cfg)  # ITM put
        mgr._pdt_guard.would_be_day_trade.return_value = True
        mgr._pdt_guard.can_day_trade.return_value = False

        status = await mgr._evaluate_position(_trade(dte=1))

        assert status.should_exit is False
        assert status.roll_side is None
        assert status.action != PositionAction.ROLL
        assert status.exit_reason == "exit_blocked_by_pdt"


class TestArbitrationWithCondorDefense:
    async def test_rule_3_wins_condor_defense_never_evaluated(self, monkeypatch):
        """Proves the CLAIM directly, not just the outcome: condor_defense
        would ALSO produce a ROLL verdict on this exact tick if it were
        ever consulted (forced via a monkeypatch, since making the real
        evaluate_condor_defense fire needs realistic synthetic-delta market
        data unrelated to what this test is proving) -- confirm it is
        NEVER CALLED AT ALL once rule 3 has already claimed the tick, and
        that rule 3's own assignment_roll_side is what's reported."""
        from ait.execution.condor_defense import DefenseAction, DefenseDecision

        call_count = {"n": 0}

        def _fake_evaluate(*args, **kwargs):
            call_count["n"] += 1
            return DefenseDecision(DefenseAction.ROLL_UNTESTED, "forced_for_test",
                                   tested_side="C", tested_short_delta=0.30, dte=10)
        monkeypatch.setattr("ait.execution.portfolio.evaluate_condor_defense", _fake_evaluate)

        assign_cfg = AssignmentResponseConfig(policy="roll_only")
        condor_cfg = CondorDefenseConfig(enabled=True)
        mgr = _manager(spot=97.50, assignment_response_config=assign_cfg,
                       condor_defense_config=condor_cfg)

        status = await mgr._evaluate_position(_trade(dte=1))

        assert call_count["n"] == 0, (
            "condor_defense must never be evaluated once rule 3 already "
            "claimed should_exit=True on this tick")
        assert status.roll_side == "P"
        assert status.action_reason == "assignment_risk_roll_only"
        assert not status.action_reason.startswith("condor_defense")


# ---------------------------------------------------------------------------
# Part B: orchestrator.py _compare_close_or_roll / _gather_ev_leg_inputs
# ---------------------------------------------------------------------------

def _orch(state, ibkr=None, options_chain=None, settings=None) -> TradingOrchestrator:
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = state
    o._ibkr = ibkr or MagicMock()
    o._options_chain = options_chain or MagicMock()
    o._executor = MagicMock()
    o._executor.place_roll_combo = AsyncMock(return_value=True)
    o._settings = settings or SimpleNamespace(
        assignment_response=AssignmentResponseConfig(policy="close_or_roll", execute_rolls=True),
        options=SimpleNamespace(risk_free_rate=0.05),
    )
    o._watchdog = MagicMock()
    return o


def _trade_record(trade_id="T-A") -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
        contract_type="iron_condor", legs=CONDOR_LEGS,
    )


def _pos(trade_id="T-A", roll_side="P", current_price=97.50) -> PositionStatus:
    return PositionStatus(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor", quantity=1,
        entry_price=1.50, current_price=current_price, unrealized_pnl=0.0, pnl_pct=0.0,
        high_water_mark=0.0, dte=1, should_exit=True,
        exit_reason="assignment_risk (short put ITM)",
        roll_side=roll_side, action=PositionAction.ROLL,
    )


def _roll_target():
    return RollTarget(
        new_short=OptionContract(symbol="SPY", expiry="2027-01-15", strike=90.0, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry="2027-01-15", strike=85.0, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry="2027-01-15", iv_is_guessed=False,
    )


def _prepared_combo(mid_net=-0.75, closing_net=0.80, opening_net=-1.55, carried_value=0.0):
    closing_legs = [
        {"conId": 1, "strike": 98.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
        {"conId": 2, "strike": 95.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    ]
    opening_legs = [
        {"conId": 3, "strike": 90.0, "right": "P", "action": "SELL", "expiry": "2027-01-15"},
        {"conId": 4, "strike": 85.0, "right": "P", "action": "BUY", "expiry": "2027-01-15"},
    ]
    quotes = {1: (1.00, 1.10), 2: (0.20, 0.30), 3: (1.60, 1.70), 4: (0.10, 0.20)}
    return SimpleNamespace(
        bag=SimpleNamespace(), order=SimpleNamespace(orderRef=""),
        closing_legs=closing_legs, opening_legs=opening_legs, carried_legs=[],
        carried_value=carried_value, mid_net=mid_net, natural_net=mid_net,
        closing_net=closing_net, opening_net=opening_net, quotes=quotes,
    )


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


class TestCompareCloseOrRollGate:
    async def test_execute_rolls_off_returns_false_no_calls(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(policy="close_or_roll",
                                                          execute_rolls=False),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        select_spy = AsyncMock(return_value=None)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target", select_spy)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        result = await o._compare_close_or_roll(_pos())

        assert result is False
        select_spy.assert_not_awaited()


class TestCompareCloseOrRollPolicyRollOnly:
    async def test_always_rolls_once_prepared_no_comparison_call(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(policy="roll_only", execute_rolls=True),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        prepared = _prepared_combo(closing_net=0.80, carried_value=0.0)  # a losing position
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))
        heuristic_spy = MagicMock(return_value=False)
        monkeypatch.setattr("ait.bot.orchestrator.heuristic_should_roll", heuristic_spy)

        result = await o._compare_close_or_roll(_pos())

        assert result is True
        heuristic_spy.assert_not_called()  # roll_only skips the comparison entirely
        o._executor.place_roll_combo.assert_awaited_once()


class TestCompareCloseOrRollHeuristic:
    async def test_losing_position_rolls(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(
                policy="close_or_roll", execute_rolls=True, comparison_model="heuristic"),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        # closing_net=0.80 (a debit -> a loss if closed now) -> heuristic says roll.
        prepared = _prepared_combo(closing_net=0.80, carried_value=0.0)
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))

        result = await o._compare_close_or_roll(_pos())

        assert result is True
        o._executor.place_roll_combo.assert_awaited_once()

    async def test_winning_position_closes(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(
                policy="close_or_roll", execute_rolls=True, comparison_model="heuristic"),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        # closing_net=-0.30 (a credit -> a win if closed now) -> heuristic says close.
        prepared = _prepared_combo(closing_net=-0.30, carried_value=0.0)
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))

        result = await o._compare_close_or_roll(_pos())

        assert result is False
        o._executor.place_roll_combo.assert_not_awaited()


class TestCompareCloseOrRollEv:
    async def test_iv_guessed_falls_back_to_heuristic(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(
                policy="close_or_roll", execute_rolls=True, comparison_model="ev"),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        guessed_target = RollTarget(
            new_short=_roll_target().new_short, new_long=_roll_target().new_long,
            expiry="2027-01-15", iv_is_guessed=True)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=guessed_target))
        prepared = _prepared_combo(closing_net=0.80, carried_value=0.0)
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))
        ev_spy = MagicMock(return_value=False)
        monkeypatch.setattr("ait.bot.orchestrator.ev_should_roll", ev_spy)

        result = await o._compare_close_or_roll(_pos())

        ev_spy.assert_not_called()
        assert result is True  # heuristic (closing_net=0.80, a loss) says roll

    async def test_unsolvable_iv_falls_back_to_heuristic(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(
                policy="close_or_roll", execute_rolls=True, comparison_model="ev"),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        prepared = _prepared_combo(closing_net=0.80, carried_value=0.0)
        prepared.quotes = {}  # every leg's quote missing -> _gather_ev_leg_inputs -> None
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))
        ev_spy = MagicMock(return_value=False)
        monkeypatch.setattr("ait.bot.orchestrator.ev_should_roll", ev_spy)

        result = await o._compare_close_or_roll(_pos())

        ev_spy.assert_not_called()
        assert result is True  # heuristic fallback (closing_net=0.80, a loss) says roll

    async def test_real_ev_call_with_solvable_legs_reuses_prepared_no_second_fetch(
            self, tmp_path, monkeypatch):
        """The real proof of the no-second-broker-round-trip claim: select_
        roll_target/prepare_roll_combo are each called EXACTLY once, even
        though the comparison ALSO consumes prepared's own quotes."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.record_trade(_trade_record())
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(
                policy="close_or_roll", execute_rolls=True, comparison_model="ev"),
            options=SimpleNamespace(risk_free_rate=0.05))
        o = _orch(st, settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)
        select_spy = AsyncMock(return_value=_roll_target())
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target", select_spy)
        prepared = _prepared_combo(closing_net=0.80, carried_value=0.0)
        prepare_spy = AsyncMock(return_value=prepared)
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo", prepare_spy)

        result = await o._compare_close_or_roll(_pos())

        select_spy.assert_awaited_once()
        prepare_spy.assert_awaited_once()
        assert result in (True, False)  # a real verdict was reached, not an exception
        if result:
            # _execute_roll must have reused prepared/roll_target -- proven
            # by select/prepare still only having been called once each,
            # not a second time inside _execute_roll.
            select_spy.assert_awaited_once()
            prepare_spy.assert_awaited_once()


class TestGatherEvLegInputs:
    def test_missing_quote_returns_none(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        o = _orch(st)
        prepared = _prepared_combo()
        prepared.quotes = {1: (None, None), 2: (0.2, 0.3), 3: (1.6, 1.7), 4: (0.1, 0.2)}
        assert o._gather_ev_leg_inputs(prepared, spot=97.50) is None

    def test_a_carried_legs_own_failure_just_drops_it(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        o = _orch(st)
        prepared = _prepared_combo()
        prepared.carried_legs = [
            {"conId": 99, "strike": 105.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
        ]  # no quote for conId 99 -- not in prepared.quotes
        result = o._gather_ev_leg_inputs(prepared, spot=97.50)
        assert result is not None
        _closing, _opening, carried = result
        assert carried == []

    def test_happy_path_builds_all_three_lists(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        o = _orch(st)
        prepared = _prepared_combo()
        result = o._gather_ev_leg_inputs(prepared, spot=97.50)
        assert result is not None
        closing, opening, carried = result
        assert len(closing) == 2 and len(opening) == 2 and carried == []
        for leg in closing + opening:
            assert leg["iv"] > 0
            assert leg["flag"] == "p"
