"""W106 item 2: an inactive StrategyAdaptor must not bank adaptations for later.

With `learning.paper_trading_mode` the adaptor is built `active=False` and every
consult is a no-op, but `apply_insights` still mutated its overrides and
persisted them. A disable / boost / override learned in paper then took effect
the day the adaptor was built `active=True` (go-live), from data the 50-close
verdict was meant to measure under ONE fixed ruleset.
"""
from __future__ import annotations

from ait.bot.state import StateManager
from ait.learning.adaptor import StrategyAdaptor
from ait.learning.analyzer import TradeInsight

_INSIGHTS = [
    TradeInsight(category="strategy", insight="long_put losing",
                 action="disable_long_put", confidence=0.8),
    TradeInsight(category="strategy", insight="long_call winning",
                 action="boost_long_call", confidence=0.8),
]


def test_inactive_adaptor_applies_and_persists_nothing(tmp_path):
    state = StateManager(tmp_path / "s.db")
    paper = StrategyAdaptor(state, active=False)

    assert paper.apply_insights(_INSIGHTS) == []
    assert not state.get_state(StrategyAdaptor.STATE_KEY)          # never written

    live = StrategyAdaptor(state, active=True)          # go-live: same database
    assert live.is_strategy_enabled("long_put")
    assert live.get_strategy_multiplier("long_call") == 1.0
    assert live.get_all_overrides()["disabled_strategies"] == []


def test_active_adaptor_still_applies(tmp_path):
    """Negative control: the same insights do adapt an active adaptor."""
    state = StateManager(tmp_path / "s.db")
    live = StrategyAdaptor(state, active=True)
    assert len(live.apply_insights(_INSIGHTS)) == 2
    assert not live.is_strategy_enabled("long_put")
    assert state.get_state(StrategyAdaptor.STATE_KEY)
