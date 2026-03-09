"""Tests for trade analytics."""

from __future__ import annotations

import pytest

from ait.monitoring.analytics import TradeAnalytics, PerformanceMetrics


class TestDrawdownCalculation:
    """Test max drawdown computation."""

    def test_no_drawdown(self):
        pnls = [100, 200, 300]
        dd_pct, dd_dollars = TradeAnalytics._calculate_drawdown(pnls)
        assert dd_dollars == 0

    def test_simple_drawdown(self):
        pnls = [100, 200, -500, 100]
        dd_pct, dd_dollars = TradeAnalytics._calculate_drawdown(pnls)
        # Peak is 300 (100+200), then drops to -200, drawdown = 500
        assert dd_dollars == 500

    def test_empty_pnls(self):
        dd_pct, dd_dollars = TradeAnalytics._calculate_drawdown([])
        assert dd_dollars == 0
        assert dd_pct == 0


class TestStreakCalculation:
    """Test win/loss streak computation."""

    def test_all_wins(self):
        max_w, max_l, current = TradeAnalytics._calculate_streaks([100, 200, 50])
        assert max_w == 3
        assert max_l == 0
        assert current == 3

    def test_all_losses(self):
        max_w, max_l, current = TradeAnalytics._calculate_streaks([-100, -200, -50])
        assert max_w == 0
        assert max_l == 3
        assert current == -3

    def test_mixed(self):
        max_w, max_l, current = TradeAnalytics._calculate_streaks([100, 200, -50, -100, 300])
        assert max_w == 2
        assert max_l == 2
        assert current == 1

    def test_empty(self):
        max_w, max_l, current = TradeAnalytics._calculate_streaks([])
        assert max_w == 0


class TestHoldTime:
    """Test hold time calculation."""

    def test_avg_hold_time(self):
        trades = [
            {"entry_time": "2026-03-01T10:00:00", "exit_time": "2026-03-01T14:00:00"},
            {"entry_time": "2026-03-02T10:00:00", "exit_time": "2026-03-02T16:00:00"},
        ]
        avg = TradeAnalytics._calculate_avg_hold_time(trades)
        assert avg == 5.0  # (4 + 6) / 2

    def test_missing_times(self):
        trades = [{"entry_time": "2026-03-01T10:00:00", "exit_time": None}]
        avg = TradeAnalytics._calculate_avg_hold_time(trades)
        assert avg == 0.0


class TestPerformanceMetrics:
    """Test the full performance calculation."""

    def test_default_metrics(self):
        m = PerformanceMetrics()
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0
