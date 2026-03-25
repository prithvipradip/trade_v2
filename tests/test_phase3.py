"""Tests for Phase 3 speed & execution improvements.

Tests batch contract qualification, passive order pricing,
and walk-forward validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from ait.broker.orders import OrderBuilder
from ait.ml.ensemble import DirectionPredictor


class TestPassivePricing:
    """Test passive limit order pricing."""

    def test_passive_buy_below_mid(self) -> None:
        order = OrderBuilder.passive_limit(
            action="BUY", quantity=1, bid=1.00, ask=1.20,
        )
        mid = (1.00 + 1.20) / 2  # 1.10
        assert order.lmtPrice < mid
        assert order.lmtPrice >= 1.00
        assert order.action == "BUY"
        assert order.totalQuantity == 1

    def test_passive_sell_above_mid(self) -> None:
        order = OrderBuilder.passive_limit(
            action="SELL", quantity=2, bid=2.00, ask=2.40,
        )
        mid = (2.00 + 2.40) / 2  # 2.20
        assert order.lmtPrice > mid
        assert order.lmtPrice <= 2.40
        assert order.action == "SELL"

    def test_passive_at_zero_improvement(self) -> None:
        """With 0% improvement, should be at mid-price."""
        order = OrderBuilder.passive_limit(
            action="BUY", quantity=1, bid=1.00, ask=1.20, improvement_pct=0.0,
        )
        assert order.lmtPrice == pytest.approx(1.10)

    def test_passive_at_full_improvement(self) -> None:
        """With 100% improvement, should be at natural side (bid for BUY)."""
        order = OrderBuilder.passive_limit(
            action="BUY", quantity=1, bid=1.00, ask=1.20, improvement_pct=1.0,
        )
        assert order.lmtPrice == pytest.approx(1.00)

    def test_passive_sell_full_improvement(self) -> None:
        """100% improvement for SELL should be at ask."""
        order = OrderBuilder.passive_limit(
            action="SELL", quantity=1, bid=1.00, ask=1.20, improvement_pct=1.0,
        )
        assert order.lmtPrice == pytest.approx(1.20)

    def test_passive_price_within_spread(self) -> None:
        """Price should always stay within the bid-ask spread."""
        for _ in range(50):
            bid = 2.0
            ask = 2.5
            imp = np.random.uniform(0, 1)
            order = OrderBuilder.passive_limit("BUY", 1, bid, ask, imp)
            assert bid <= order.lmtPrice <= ask

    def test_passive_narrow_spread(self) -> None:
        """With a very narrow spread, price should still be valid."""
        order = OrderBuilder.passive_limit(
            action="BUY", quantity=1, bid=5.00, ask=5.01,
        )
        assert 5.00 <= order.lmtPrice <= 5.01

    def test_passive_tif_is_day(self) -> None:
        order = OrderBuilder.passive_limit("BUY", 1, 1.00, 1.20)
        assert order.tif == "DAY"


class TestWalkForwardValidation:
    """Test walk-forward split logic in ensemble."""

    def test_walk_forward_split_produces_splits(self) -> None:
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        splits = predictor._walk_forward_split(n_samples=200, n_splits=5, gap=5)
        assert len(splits) >= 2

    def test_walk_forward_no_overlap(self) -> None:
        """Training and validation sets should not overlap."""
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        splits = predictor._walk_forward_split(n_samples=300, n_splits=5, gap=5)

        for train_idx, val_idx in splits:
            # No index should be in both
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0

    def test_walk_forward_gap_respected(self) -> None:
        """There should be a gap between end of training and start of validation."""
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        gap = 10
        splits = predictor._walk_forward_split(n_samples=300, n_splits=5, gap=gap)

        for train_idx, val_idx in splits:
            train_max = train_idx.max()
            val_min = val_idx.min()
            assert val_min - train_max >= gap

    def test_walk_forward_expanding_window(self) -> None:
        """Each successive fold should have more training data."""
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        splits = predictor._walk_forward_split(n_samples=300, n_splits=5, gap=5)

        if len(splits) >= 2:
            train_sizes = [len(train_idx) for train_idx, _ in splits]
            # Training sets should be increasing (expanding window)
            for i in range(1, len(train_sizes)):
                assert train_sizes[i] >= train_sizes[i - 1]

    def test_walk_forward_fallback_on_small_data(self) -> None:
        """With too little data, should fall back to TimeSeriesSplit."""
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        splits = predictor._walk_forward_split(n_samples=30, n_splits=5, gap=5)
        # Should still produce some splits (fallback)
        assert len(splits) >= 2

    def test_walk_forward_val_indices_in_range(self) -> None:
        n_samples = 200
        predictor = DirectionPredictor.__new__(DirectionPredictor)
        splits = predictor._walk_forward_split(n_samples=n_samples, n_splits=5, gap=5)

        for train_idx, val_idx in splits:
            assert train_idx.min() >= 0
            assert val_idx.max() <= n_samples
