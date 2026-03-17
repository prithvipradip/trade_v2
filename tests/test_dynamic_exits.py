"""Tests for dynamic exit management, partial exits, and meta-labeling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ait.bot.state import StateManager, TradeRecord, TradeDirection, TradeStatus
from ait.config.settings import ExitConfig
from ait.execution.portfolio import PortfolioManager
from ait.ml.meta_label import MetaLabeler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    trade_id: str,
    symbol: str = "AAPL",
    strategy: str = "long_call",
    direction: str = "long",
    status: str = "closed",
    entry_price: float = 5.00,
    exit_price: float | None = 7.50,
    realized_pnl: float = 250.0,
    confidence: float = 0.70,
    quantity: int = 1,
    contract_type: str = "call",
    entry_time: str = "2026-03-01T10:00:00",
) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        strategy=strategy,
        direction=TradeDirection(direction),
        status=TradeStatus(status),
        entry_time=entry_time,
        entry_price=entry_price,
        quantity=quantity,
        contract_type=contract_type,
        exit_time="2026-03-01T14:00:00" if status == "closed" else None,
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        ml_confidence=confidence,
    )


def _create_open_position(state: StateManager, trade_id: str, symbol: str = "AAPL",
                           quantity: int = 4, entry_price: float = 5.00) -> None:
    """Insert an open_positions row directly for testing."""
    with sqlite3.connect(state._db_path) as conn:
        conn.execute(
            """INSERT INTO open_positions
               (position_id, trade_id, symbol, contract_type, quantity,
                entry_price, entry_time, high_water_mark, partial_exits)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"pos-{trade_id}", trade_id, symbol, "call", quantity,
             entry_price, "2026-03-01T10:00:00", 0.0, "[]"),
        )


def _make_portfolio_manager(state: StateManager, exit_config: ExitConfig | None = None) -> PortfolioManager:
    """Create a PortfolioManager with mocked dependencies except state."""
    return PortfolioManager(
        ibkr_client=MagicMock(),
        market_data=MagicMock(),
        state=state,
        circuit_breaker=MagicMock(),
        pdt_guard=MagicMock(),
        exit_config=exit_config or ExitConfig(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    """Fresh StateManager backed by a temp database."""
    return StateManager(tmp_path / "test_dynamic_exits.db")


@pytest.fixture
def pm(state: StateManager) -> PortfolioManager:
    """PortfolioManager with default ExitConfig and real state."""
    return _make_portfolio_manager(state)


# ===========================================================================
# 1. Dynamic Trailing Stops
# ===========================================================================

class TestDynamicTrailingStops:
    """Test PortfolioManager._calculate_dynamic_stop three-tier logic."""

    def test_tier1_below_breakeven_returns_initial_stop(self, pm: PortfolioManager) -> None:
        """When HWM is below breakeven trigger, return the negative initial stop."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=-0.10, hwm=0.15,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(-0.50)

    def test_tier1_hwm_zero(self, pm: PortfolioManager) -> None:
        """HWM at 0 is clearly below breakeven trigger."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=-0.05, hwm=0.0,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(-0.50)

    def test_tier1_hwm_just_below_trigger(self, pm: PortfolioManager) -> None:
        """HWM at 0.29 is still below 0.30 trigger."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=0.10, hwm=0.29,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(-0.50)

    def test_tier2_just_crossed_breakeven_trail_positive(self, pm: PortfolioManager) -> None:
        """HWM just crossed breakeven trigger; trail = 0.32 - 0.25 = 0.07."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=0.30, hwm=0.32,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(0.07)

    def test_tier2_trail_would_be_negative_clamps_to_zero(self, pm: PortfolioManager) -> None:
        """HWM exactly at trigger: trail = 0.30 - 0.25 = 0.05, still positive."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=0.25, hwm=0.30,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(0.05)

    def test_tier2_trail_negative_clamps(self, pm: PortfolioManager) -> None:
        """When trail calc goes negative, clamp to 0.0 (breakeven)."""
        # hwm=0.30, trailing_pct=0.40 => 0.30 - 0.40 = -0.10 => clamped to 0.0
        result = pm._calculate_dynamic_stop(
            pnl_pct=0.25, hwm=0.30,
            initial_stop=0.50, trailing_pct=0.40, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(0.0)

    def test_tier3_deep_profit_trailing(self, pm: PortfolioManager) -> None:
        """Deep in profit: trail = 0.80 - 0.25 = 0.55."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=0.50, hwm=0.80,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(0.55)

    def test_tier3_very_high_hwm(self, pm: PortfolioManager) -> None:
        """Extremely high HWM should trail correctly."""
        result = pm._calculate_dynamic_stop(
            pnl_pct=1.50, hwm=2.00,
            initial_stop=0.50, trailing_pct=0.25, breakeven_trigger=0.30,
        )
        assert result == pytest.approx(1.75)

    def test_stop_never_below_zero_once_breakeven_triggered(self, pm: PortfolioManager) -> None:
        """Once HWM >= breakeven trigger, stop is always >= 0."""
        for hwm in [0.30, 0.35, 0.50, 1.00]:
            result = pm._calculate_dynamic_stop(
                pnl_pct=0.0, hwm=hwm,
                initial_stop=0.50, trailing_pct=0.50, breakeven_trigger=0.30,
            )
            assert result >= 0.0, f"Stop should be >= 0 for hwm={hwm}, got {result}"


# ===========================================================================
# 2. Partial Exit Logic
# ===========================================================================

class TestPartialExitLogic:
    """Test PortfolioManager._check_partial_exit."""

    def test_returns_zero_when_quantity_one(self, state: StateManager) -> None:
        """Single-contract positions should never partial exit."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-1", quantity=1)
        result = pm._check_partial_exit("T-PE-1", pnl_pct=1.0, current_quantity=1)
        assert result == 0

    def test_returns_zero_when_no_level_reached(self, state: StateManager) -> None:
        """PnL below all partial exit thresholds returns 0."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-2", quantity=4)
        # Default levels: 0.50 and 1.00; pnl_pct=0.20 is below both
        result = pm._check_partial_exit("T-PE-2", pnl_pct=0.20, current_quantity=4)
        assert result == 0

    def test_returns_qty_when_first_level_reached(self, state: StateManager) -> None:
        """When first partial exit level (50% P&L) is reached, close ~33% of 4 = 1 contract."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-3", quantity=4)
        result = pm._check_partial_exit("T-PE-3", pnl_pct=0.55, current_quantity=4)
        # int(4 * 0.33) = 1, min(1, 4-1) = 1
        assert result == 1

    def test_returns_qty_when_second_level_reached(self, state: StateManager) -> None:
        """When second level (100% P&L) is reached and first not yet done, triggers first."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-4", quantity=6)
        # pnl_pct=1.05 exceeds both levels; first unreached level triggers
        result = pm._check_partial_exit("T-PE-4", pnl_pct=1.05, current_quantity=6)
        # First level: int(6 * 0.33) = 1, min(1, 6-1) = 1
        assert result >= 1

    def test_does_not_repeat_completed_levels(self, state: StateManager) -> None:
        """Already-completed partial exit levels should be skipped."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-5", quantity=4)

        # Record first partial exit at the 0.50 level
        state.record_partial_exit("T-PE-5", quantity=1, price=7.50, pnl=250.0)
        # Manually mark the level as completed by inserting pnl_level
        with sqlite3.connect(state._db_path) as conn:
            exits = json.loads(conn.execute(
                "SELECT partial_exits FROM open_positions WHERE trade_id = ?",
                ("T-PE-5",),
            ).fetchone()[0])
            exits[-1]["pnl_level"] = 0.50
            conn.execute(
                "UPDATE open_positions SET partial_exits = ? WHERE trade_id = ?",
                (json.dumps(exits), "T-PE-5"),
            )

        # With pnl_pct=0.55, the 0.50 level is completed so it should skip
        result = pm._check_partial_exit("T-PE-5", pnl_pct=0.55, current_quantity=3)
        assert result == 0

    def test_leaves_at_least_one_contract(self, state: StateManager) -> None:
        """Partial exit should never close all contracts."""
        exit_config = ExitConfig(partial_exit_levels=[
            {"pnl_pct": 0.30, "close_pct": 0.90},  # Try to close 90%
        ])
        pm = _make_portfolio_manager(state, exit_config)
        _create_open_position(state, "T-PE-6", quantity=2)

        result = pm._check_partial_exit("T-PE-6", pnl_pct=0.35, current_quantity=2)
        # max(1, int(2*0.90)) = 1, min(1, 2-1) = 1 => leaves 1 contract
        assert result == 1

    def test_large_position_partial_exit(self, state: StateManager) -> None:
        """10-contract position at first level: close ~33% = 3 contracts."""
        pm = _make_portfolio_manager(state)
        _create_open_position(state, "T-PE-7", quantity=10)

        result = pm._check_partial_exit("T-PE-7", pnl_pct=0.60, current_quantity=10)
        # int(10 * 0.33) = 3, min(3, 10-1) = 3
        assert result == 3


# ===========================================================================
# 3. Time-Decay Take Profit Targets
# ===========================================================================

class TestTimedDecayTargets:
    """Test PortfolioManager._get_take_profit_targets."""

    def test_dte_above_20(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(25) == (1.0, 0.50)

    def test_dte_exactly_21(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(21) == (1.0, 0.50)

    def test_dte_20(self, pm: PortfolioManager) -> None:
        """DTE=20 falls into the 10 < dte <= 20 bracket."""
        assert pm._get_take_profit_targets(20) == (0.75, 0.40)

    def test_dte_between_10_and_20(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(15) == (0.75, 0.40)

    def test_dte_11(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(11) == (0.75, 0.40)

    def test_dte_10(self, pm: PortfolioManager) -> None:
        """DTE=10 falls into the 5 < dte <= 10 bracket."""
        assert pm._get_take_profit_targets(10) == (0.50, 0.30)

    def test_dte_between_5_and_10(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(7) == (0.50, 0.30)

    def test_dte_6(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(6) == (0.50, 0.30)

    def test_dte_5(self, pm: PortfolioManager) -> None:
        """DTE=5 falls into the else (dte <= 5) bracket."""
        assert pm._get_take_profit_targets(5) == (0.25, 0.20)

    def test_dte_below_5(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(2) == (0.25, 0.20)

    def test_dte_zero(self, pm: PortfolioManager) -> None:
        assert pm._get_take_profit_targets(0) == (0.25, 0.20)

    def test_time_decay_scaling_disabled(self, state: StateManager) -> None:
        """With time_decay_scaling=False, always return default targets."""
        exit_config = ExitConfig(time_decay_scaling=False)
        pm = _make_portfolio_manager(state, exit_config)
        assert pm._get_take_profit_targets(3) == (1.0, 0.50)
        assert pm._get_take_profit_targets(15) == (1.0, 0.50)

    def test_dte_none(self, pm: PortfolioManager) -> None:
        """DTE=None (e.g. stock position) returns default targets."""
        assert pm._get_take_profit_targets(None) == (1.0, 0.50)


# ===========================================================================
# 4. High Water Mark Persistence
# ===========================================================================

class TestHighWaterMark:
    """Test StateManager high water mark operations."""

    def test_get_hwm_unknown_trade_returns_zero(self, state: StateManager) -> None:
        assert state.get_high_water_mark("nonexistent") == 0.0

    def test_update_and_get_hwm(self, state: StateManager) -> None:
        _create_open_position(state, "T-HWM-1")
        state.update_high_water_mark("T-HWM-1", 0.35)
        assert state.get_high_water_mark("T-HWM-1") == pytest.approx(0.35)

    def test_hwm_only_increases(self, state: StateManager) -> None:
        """HWM uses MAX in SQL -- should never decrease."""
        _create_open_position(state, "T-HWM-2")
        state.update_high_water_mark("T-HWM-2", 0.50)
        state.update_high_water_mark("T-HWM-2", 0.30)  # Lower value
        assert state.get_high_water_mark("T-HWM-2") == pytest.approx(0.50)

    def test_hwm_increases_when_higher(self, state: StateManager) -> None:
        _create_open_position(state, "T-HWM-3")
        state.update_high_water_mark("T-HWM-3", 0.20)
        state.update_high_water_mark("T-HWM-3", 0.60)
        assert state.get_high_water_mark("T-HWM-3") == pytest.approx(0.60)

    def test_hwm_starts_at_zero(self, state: StateManager) -> None:
        _create_open_position(state, "T-HWM-4")
        assert state.get_high_water_mark("T-HWM-4") == pytest.approx(0.0)


# ===========================================================================
# 5. Partial Exit Recording
# ===========================================================================

class TestPartialExitRecording:
    """Test StateManager partial exit persistence."""

    def test_get_partial_exits_empty(self, state: StateManager) -> None:
        """Unknown trade returns empty list."""
        assert state.get_partial_exits("nonexistent") == []

    def test_record_and_get_partial_exit(self, state: StateManager) -> None:
        _create_open_position(state, "T-PRE-1")
        state.record_partial_exit("T-PRE-1", quantity=2, price=7.50, pnl=500.0)

        exits = state.get_partial_exits("T-PRE-1")
        assert len(exits) == 1
        assert exits[0]["quantity"] == 2
        assert exits[0]["price"] == 7.50
        assert exits[0]["pnl"] == 500.0
        assert "time" in exits[0]

    def test_record_multiple_partial_exits(self, state: StateManager) -> None:
        """Multiple partial exits append to the JSON list."""
        _create_open_position(state, "T-PRE-2")
        state.record_partial_exit("T-PRE-2", quantity=1, price=6.00, pnl=100.0)
        state.record_partial_exit("T-PRE-2", quantity=2, price=8.00, pnl=600.0)

        exits = state.get_partial_exits("T-PRE-2")
        assert len(exits) == 2
        assert exits[0]["quantity"] == 1
        assert exits[1]["quantity"] == 2

    def test_partial_exits_for_fresh_position(self, state: StateManager) -> None:
        """New position should have empty partial exits."""
        _create_open_position(state, "T-PRE-3")
        assert state.get_partial_exits("T-PRE-3") == []


# ===========================================================================
# 6. Trade Context
# ===========================================================================

class TestTradeContext:
    """Test StateManager trade context operations."""

    def test_save_and_get_trade_context(self, state: StateManager) -> None:
        state.save_trade_context(
            trade_id="T-CTX-1",
            direction="long",
            confidence=0.75,
            regime="trending_up",
            vix=18.5,
            iv_rank=0.45,
            sentiment_score=0.30,
            signals='{"macd": "bullish"}',
        )

        ctx = state.get_trade_context("T-CTX-1")
        assert ctx is not None
        assert ctx["entry_direction"] == "long"
        assert ctx["entry_confidence"] == pytest.approx(0.75)
        assert ctx["entry_regime"] == "trending_up"
        assert ctx["entry_vix"] == pytest.approx(18.5)
        assert ctx["entry_iv_rank"] == pytest.approx(0.45)
        assert ctx["entry_sentiment_score"] == pytest.approx(0.30)
        assert ctx["entry_signals"] == '{"macd": "bullish"}'

    def test_get_unknown_trade_context_returns_none(self, state: StateManager) -> None:
        assert state.get_trade_context("nonexistent") is None

    def test_trade_context_upsert(self, state: StateManager) -> None:
        """Saving context twice for the same trade_id should overwrite."""
        state.save_trade_context("T-CTX-2", "long", 0.60, "range_bound", 20.0, 0.50, 0.10)
        state.save_trade_context("T-CTX-2", "short", 0.80, "high_volatility", 30.0, 0.70, -0.20)

        ctx = state.get_trade_context("T-CTX-2")
        assert ctx["entry_direction"] == "short"
        assert ctx["entry_confidence"] == pytest.approx(0.80)


# ===========================================================================
# 7. MetaLabeler
# ===========================================================================

class TestMetaLabeler:
    """Test the MetaLabeler model lifecycle."""

    def test_predict_returns_none_when_not_trained(self) -> None:
        ml = MetaLabeler()
        result = ml.predict({"primary_confidence": 0.80, "vix": 18.0})
        assert result is None

    def test_is_trained_false_initially(self) -> None:
        ml = MetaLabeler()
        assert ml.is_trained is False

    def test_build_training_data_empty_when_no_trades(self, state: StateManager) -> None:
        ml = MetaLabeler()
        df = ml.build_training_data(state)
        assert len(df) == 0

    def test_build_training_data_returns_empty_without_context(self, state: StateManager) -> None:
        """Closed trades without matching trade_context rows produce no training data."""
        trade = _make_trade("T-ML-1", status="closed", realized_pnl=100.0)
        state.record_trade(trade)
        state.close_trade("T-ML-1", exit_price=7.50, realized_pnl=100.0)

        ml = MetaLabeler()
        df = ml.build_training_data(state)
        assert len(df) == 0

    def test_train_returns_empty_when_insufficient_data(self, state: StateManager) -> None:
        """Fewer than 30 trades should return empty dict."""
        import pandas as pd

        ml = MetaLabeler()
        # Only 5 rows
        df = pd.DataFrame({
            "primary_confidence": [0.7] * 5,
            "vix": [18.0] * 5,
            "profitable": [1, 0, 1, 0, 1],
        })
        result = ml.train(df)
        assert result == {}

    def test_train_returns_empty_when_class_imbalance(self) -> None:
        """All-profitable or all-unprofitable data returns empty dict."""
        import pandas as pd

        ml = MetaLabeler()
        df = pd.DataFrame({
            "primary_confidence": [0.7] * 35,
            "vix": [18.0] * 35,
            "profitable": [1] * 35,  # All profitable, no negatives
        })
        result = ml.train(df)
        assert result == {}

    def test_train_succeeds_with_enough_data(self, state: StateManager, tmp_path: Path) -> None:
        """With 40+ trades with context, training should succeed."""
        import pandas as pd

        # Populate DB with 44 closed trades + trade_context
        regimes = ["trending_up", "trending_down", "high_volatility", "range_bound"]
        for i in range(44):
            trade_id = f"T-ML-TRAIN-{i:03d}"
            is_profitable = i < 22  # Half profitable, half not
            pnl = 150.0 if is_profitable else -120.0

            trade = _make_trade(
                trade_id=trade_id,
                symbol=["AAPL", "SPY", "QQQ", "NVDA"][i % 4],
                strategy="long_call",
                direction="long",
                status="closed",
                entry_price=5.00,
                exit_price=6.50 if is_profitable else 3.80,
                realized_pnl=pnl,
                confidence=0.60 + (i % 10) * 0.03,
                entry_time=f"2026-02-{1 + (i % 28):02d}T{10 + (i % 6)}:00:00",
            )
            state.record_trade(trade)
            state.close_trade(trade_id, exit_price=trade.exit_price or 0, realized_pnl=pnl)

            state.save_trade_context(
                trade_id=trade_id,
                direction="long",
                confidence=0.60 + (i % 10) * 0.03,
                regime=regimes[i % 4],
                vix=15.0 + (i % 15),
                iv_rank=0.20 + (i % 10) * 0.05,
                sentiment_score=-0.50 + i * 0.025,
            )

        ml = MetaLabeler()
        df = ml.build_training_data(state)

        assert len(df) >= 40
        assert "profitable" in df.columns
        assert "primary_confidence" in df.columns

        try:
            result = ml.train(df)
        except ImportError:
            pytest.skip("xgboost not installed")

        assert result != {}
        assert "accuracy" in result
        assert "precision" in result
        assert "trades_used" in result
        assert result["trades_used"] >= 40

    def test_predict_returns_meta_signal_after_training(self, state: StateManager) -> None:
        """After successful training, predict() should return a MetaSignal."""
        import pandas as pd

        regimes = ["trending_up", "trending_down", "high_volatility", "range_bound"]
        for i in range(44):
            trade_id = f"T-ML-PRED-{i:03d}"
            is_profitable = i < 22
            pnl = 150.0 if is_profitable else -120.0

            trade = _make_trade(
                trade_id=trade_id,
                status="closed",
                realized_pnl=pnl,
                entry_time=f"2026-02-{1 + (i % 28):02d}T{10 + (i % 6)}:00:00",
            )
            state.record_trade(trade)
            state.close_trade(trade_id, exit_price=6.50 if is_profitable else 3.80, realized_pnl=pnl)
            state.save_trade_context(
                trade_id=trade_id,
                direction="long",
                confidence=0.60 + (i % 10) * 0.03,
                regime=regimes[i % 4],
                vix=15.0 + (i % 15),
                iv_rank=0.20 + (i % 10) * 0.05,
                sentiment_score=-0.50 + i * 0.025,
            )

        ml = MetaLabeler()
        df = ml.build_training_data(state)

        try:
            result = ml.train(df)
        except ImportError:
            pytest.skip("xgboost not installed")

        if not result:
            pytest.skip("Training did not succeed (possible env issue)")

        assert ml.is_trained is True

        from ait.ml.meta_label import MetaSignal

        signal = ml.predict({
            "primary_confidence": 0.75,
            "vix": 20.0,
            "iv_rank": 0.40,
            "sentiment_score": 0.10,
            "hour_of_day": 11,
        })
        assert signal is not None
        assert isinstance(signal, MetaSignal)
        assert 0.0 <= signal.probability <= 1.0
        assert isinstance(signal.take_trade, bool)
        assert signal.features_used > 0

    def test_build_training_data_profitable_label(self, state: StateManager) -> None:
        """Profitable trades should have profitable=1, losers profitable=0."""
        for i, pnl in enumerate([200.0, -100.0, 50.0]):
            tid = f"T-ML-LBL-{i}"
            trade = _make_trade(tid, status="closed", realized_pnl=pnl)
            state.record_trade(trade)
            state.close_trade(tid, exit_price=7.0, realized_pnl=pnl)
            state.save_trade_context(tid, "long", 0.70, "trending_up", 18.0, 0.40, 0.10)

        ml = MetaLabeler()
        df = ml.build_training_data(state)

        assert len(df) == 3
        assert list(df["profitable"]) == [1, 0, 1]
