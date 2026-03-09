"""Tests for the self-learning engine."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ait.bot.state import StateManager, TradeRecord, TradeDirection, TradeStatus
from ait.learning.analyzer import TradeAnalyzer, TradeInsight
from ait.learning.adaptor import AdaptationLimits, StrategyAdaptor
from ait.learning.engine import LearningEngine


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with sample trades."""
    db_path = tmp_path / "test_state.db"
    state = StateManager(db_path)

    # Insert some closed trades
    trades = [
        # Winning long_call trades
        _make_trade("T-001", "AAPL", "long_call", "long", "closed", 5.00, 7.50, 250.0, 0.75),
        _make_trade("T-002", "AAPL", "long_call", "long", "closed", 3.00, 4.80, 180.0, 0.72),
        _make_trade("T-003", "SPY", "long_call", "long", "closed", 4.00, 5.50, 150.0, 0.80),
        _make_trade("T-004", "MSFT", "long_call", "long", "closed", 6.00, 8.00, 200.0, 0.70),
        _make_trade("T-005", "NVDA", "long_call", "long", "closed", 7.00, 9.00, 200.0, 0.68),
        # Losing iron_condor trades
        _make_trade("T-006", "SPY", "iron_condor", "short", "closed", 2.00, 0.50, -150.0, 0.65),
        _make_trade("T-007", "QQQ", "iron_condor", "short", "closed", 1.80, 0.30, -150.0, 0.62),
        _make_trade("T-008", "SPY", "iron_condor", "short", "closed", 2.20, 0.60, -160.0, 0.63),
        _make_trade("T-009", "AAPL", "iron_condor", "short", "closed", 1.50, 0.20, -130.0, 0.60),
        _make_trade("T-010", "MSFT", "iron_condor", "short", "closed", 1.70, 0.40, -130.0, 0.58),
        # Low confidence losers
        _make_trade("T-011", "TSLA", "long_put", "long", "closed", 4.00, 2.00, -200.0, 0.55),
        _make_trade("T-012", "TSLA", "long_put", "long", "closed", 3.50, 1.50, -200.0, 0.52),
        _make_trade("T-013", "TSLA", "long_put", "long", "closed", 5.00, 3.00, -200.0, 0.54),
    ]

    for t in trades:
        state.record_trade(t)
        if t.status == TradeStatus.CLOSED:
            state.close_trade(t.trade_id, t.exit_price or 0, t.realized_pnl)

    return db_path, state


def _make_trade(
    trade_id, symbol, strategy, direction, status,
    entry_price, exit_price, pnl, confidence,
):
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        strategy=strategy,
        direction=TradeDirection(direction),
        status=TradeStatus(status),
        entry_time="2026-03-01T10:00:00",
        entry_price=entry_price,
        quantity=1,
        contract_type="call" if "call" in strategy else "put" if "put" in strategy else "spread",
        exit_time="2026-03-01T14:00:00",
        exit_price=exit_price,
        realized_pnl=pnl,
        ml_confidence=confidence,
    )


class TestTradeAnalyzer:
    """Test trade outcome analysis."""

    def test_analyze_finds_losing_strategy(self, tmp_db):
        db_path, _ = tmp_db
        analyzer = TradeAnalyzer(db_path)
        insights = analyzer.analyze_all(lookback_days=30)

        # Should find iron_condor as a losing strategy
        strategy_insights = [i for i in insights if i.category == "strategy"]
        losing = [i for i in strategy_insights if "disable" in i.action]
        assert any("iron_condor" in i.action for i in losing)

    def test_analyze_finds_winning_strategy(self, tmp_db):
        db_path, _ = tmp_db
        analyzer = TradeAnalyzer(db_path)
        insights = analyzer.analyze_all(lookback_days=30)

        strategy_insights = [i for i in insights if i.category == "strategy"]
        winning = [i for i in strategy_insights if "boost" in i.action]
        assert any("long_call" in i.action for i in winning)

    def test_get_strategy_stats(self, tmp_db):
        db_path, _ = tmp_db
        analyzer = TradeAnalyzer(db_path)
        stats = analyzer.get_strategy_stats(lookback_days=60)

        assert "long_call" in stats
        assert stats["long_call"].win_rate == 1.0  # All wins
        assert stats["iron_condor"].win_rate == 0.0  # All losses

    def test_get_symbol_stats(self, tmp_db):
        db_path, _ = tmp_db
        analyzer = TradeAnalyzer(db_path)
        stats = analyzer.get_symbol_stats(lookback_days=60)

        assert "AAPL" in stats
        assert "TSLA" in stats

    def test_no_trades_returns_empty(self, tmp_path):
        db_path = tmp_path / "empty.db"
        StateManager(db_path)
        analyzer = TradeAnalyzer(db_path)
        insights = analyzer.analyze_all(lookback_days=30)
        assert insights == []


class TestStrategyAdaptor:
    """Test strategy adaptation from insights."""

    def test_disable_losing_strategy(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        insights = [TradeInsight(
            category="strategy",
            insight="iron_condor losing",
            action="disable_iron_condor",
            confidence=0.8,
        )]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 1
        assert not adaptor.is_strategy_enabled("iron_condor")
        assert adaptor.get_strategy_multiplier("iron_condor") == 0.0

    def test_boost_winning_strategy(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        insights = [TradeInsight(
            category="strategy",
            insight="long_call winning",
            action="boost_long_call",
            confidence=0.8,
        )]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 1
        assert adaptor.get_strategy_multiplier("long_call") > 1.0

    def test_raise_confidence(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        insights = [TradeInsight(
            category="confidence",
            insight="Low confidence trades lose",
            action="raise_min_confidence_to_0.70",
            confidence=0.75,
        )]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 1
        assert adaptor.get_confidence_override() == 0.70

    def test_remove_symbol(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        insights = [TradeInsight(
            category="symbol",
            insight="TSLA losing",
            action="remove_symbol_TSLA",
            confidence=0.7,
        )]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 1
        assert not adaptor.is_symbol_allowed("TSLA")
        assert adaptor.is_symbol_allowed("AAPL")

    def test_max_adaptations_per_cycle(self, tmp_db):
        _, state = tmp_db
        limits = AdaptationLimits(max_adaptations_per_cycle=2)
        adaptor = StrategyAdaptor(state, limits)

        insights = [
            TradeInsight(category="strategy", insight="a", action="disable_iron_condor", confidence=0.9),
            TradeInsight(category="strategy", insight="b", action="boost_long_call", confidence=0.8),
            TradeInsight(category="symbol", insight="c", action="remove_symbol_TSLA", confidence=0.7),
        ]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 2

    def test_low_confidence_insights_skipped(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        insights = [TradeInsight(
            category="strategy",
            insight="weak signal",
            action="disable_long_put",
            confidence=0.3,
        )]

        adaptations = adaptor.apply_insights(insights)
        assert len(adaptations) == 0

    def test_safety_bounds_on_confidence(self, tmp_db):
        _, state = tmp_db
        limits = AdaptationLimits(min_confidence_ceiling=0.85)
        adaptor = StrategyAdaptor(state, limits)

        insights = [TradeInsight(
            category="confidence",
            insight="raise to 0.95",
            action="raise_min_confidence_to_0.95",
            confidence=0.8,
        )]

        adaptor.apply_insights(insights)
        assert adaptor.get_confidence_override() == 0.85

    def test_state_persists_and_loads(self, tmp_db):
        _, state = tmp_db

        # Apply an adaptation
        adaptor1 = StrategyAdaptor(state)
        adaptor1.apply_insights([TradeInsight(
            category="strategy", insight="x", action="disable_iron_condor", confidence=0.8,
        )])

        # New instance should load persisted state
        adaptor2 = StrategyAdaptor(state)
        assert not adaptor2.is_strategy_enabled("iron_condor")

    def test_reset_clears_all(self, tmp_db):
        _, state = tmp_db
        adaptor = StrategyAdaptor(state)

        adaptor.apply_insights([TradeInsight(
            category="strategy", insight="x", action="disable_iron_condor", confidence=0.8,
        )])
        adaptor.reset()

        assert adaptor.is_strategy_enabled("iron_condor")
        assert adaptor.get_confidence_override() is None


class TestLearningEngine:
    """Test the full learning cycle."""

    def test_learning_cycle_with_data(self, tmp_db):
        db_path, state = tmp_db
        analyzer = TradeAnalyzer(db_path)
        adaptor = StrategyAdaptor(state)
        engine = LearningEngine(state, analyzer, adaptor)

        result = engine.run_learning_cycle(lookback_days=30)

        assert result["insights"] > 0
        assert isinstance(result["details"], list)

    def test_learning_cycle_no_data(self, tmp_path):
        db_path = tmp_path / "empty.db"
        state = StateManager(db_path)
        engine = LearningEngine(state)

        result = engine.run_learning_cycle(lookback_days=30)
        assert result["insights"] == 0
        assert result["adaptations"] == 0

    def test_learning_history_recorded(self, tmp_db):
        db_path, state = tmp_db
        analyzer = TradeAnalyzer(db_path)
        engine = LearningEngine(state, analyzer)

        engine.run_learning_cycle(lookback_days=30)
        history = engine.get_learning_history()

        assert len(history) >= 1
        assert "timestamp" in history[0]

    def test_reset_all_learning(self, tmp_db):
        db_path, state = tmp_db
        analyzer = TradeAnalyzer(db_path)
        engine = LearningEngine(state, analyzer)

        engine.run_learning_cycle(lookback_days=30)
        engine.reset_all_learning()

        assert engine.get_learning_history() == []
        assert engine.get_current_adaptations()["disabled_strategies"] == []
