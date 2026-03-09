"""Tests for PositionSizer risk management component."""

from __future__ import annotations

import pytest

from ait.config.settings import PositionConfig, RiskConfig
from ait.risk.position_sizer import PositionSize, PositionSizer


@pytest.fixture
def sizer(position_config: PositionConfig, risk_config: RiskConfig) -> PositionSizer:
    return PositionSizer(position_config, risk_config)


class TestBasicSizing:
    """Test basic sizing calculation."""

    def test_basic_calculation(self, sizer: PositionSizer) -> None:
        """With standard inputs, should return a valid PositionSize."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=2.50,
            confidence=0.85,
            implied_vol=0.25,
            strategy="long_call",
            underlying_price=450.0,
        )

        assert isinstance(result, PositionSize)
        assert result.contracts >= 1
        assert result.contracts <= 10
        assert result.max_risk_dollars > 0

    def test_zero_account_returns_zero(self, sizer: PositionSizer) -> None:
        result = sizer.calculate(
            account_value=0,
            option_price=2.50,
            confidence=0.85,
            implied_vol=0.25,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.contracts == 0

    def test_zero_option_price_returns_zero(self, sizer: PositionSizer) -> None:
        result = sizer.calculate(
            account_value=100_000,
            option_price=0,
            confidence=0.85,
            implied_vol=0.25,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.contracts == 0

    def test_max_risk_equals_contracts_times_cost(
        self, sizer: PositionSizer
    ) -> None:
        """max_risk_dollars should be contracts * option_price * 100."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=3.00,
            confidence=0.90,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.max_risk_dollars == result.contracts * 3.00 * 100


class TestConfidenceAdjustment:
    """Test confidence adjustment."""

    def test_high_confidence_larger_size(self, sizer: PositionSizer) -> None:
        """Higher confidence should yield more contracts (or equal)."""
        high = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.95,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        low = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.66,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert high.confidence_adjustment >= low.confidence_adjustment

    def test_min_confidence_gets_half_size(self, sizer: PositionSizer) -> None:
        """At min_confidence (0.65), conf_adj should be ~0.5."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.65,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert abs(result.confidence_adjustment - 0.5) < 0.05

    def test_confidence_1_gets_full_size(self, sizer: PositionSizer) -> None:
        """At confidence=1.0, conf_adj should be 1.0."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=1.0,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.confidence_adjustment == 1.0

    def test_conf_adj_clamped_to_min_0_3(self, sizer: PositionSizer) -> None:
        """Even very low confidence should not go below 0.3."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.50,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.confidence_adjustment >= 0.3


class TestVolatilityAdjustment:
    """Test volatility adjustment (high IV -> smaller size)."""

    def test_low_vol_full_size(self, sizer: PositionSizer) -> None:
        """IV <= 20% should give vol_adj = 1.0."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.15,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.volatility_adjustment == 1.0

    def test_medium_vol_reduced(self, sizer: PositionSizer) -> None:
        """IV ~35% should give vol_adj = 0.85."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.35,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.volatility_adjustment == 0.85

    def test_high_vol_half_size(self, sizer: PositionSizer) -> None:
        """IV > 60% should give vol_adj = 0.5."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.65,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.volatility_adjustment == 0.5

    def test_vol_40_to_60_gives_0_7(self, sizer: PositionSizer) -> None:
        """IV in (40%, 60%] should give vol_adj = 0.7."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.50,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert result.volatility_adjustment == 0.7

    def test_higher_vol_means_fewer_contracts(self, sizer: PositionSizer) -> None:
        """Higher IV should result in fewer or equal contracts."""
        low_vol = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.15,
            strategy="long_call",
            underlying_price=450.0,
        )
        high_vol = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.65,
            strategy="long_call",
            underlying_price=450.0,
        )
        assert low_vol.contracts >= high_vol.contracts


class TestStrategyRiskMultipliers:
    """Test strategy risk multipliers."""

    def test_iron_condor_smaller_than_long_call(
        self, sizer: PositionSizer
    ) -> None:
        """Iron condor (0.4) should size smaller than long_call (1.0)."""
        long_call = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        iron_condor = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.20,
            strategy="iron_condor",
            underlying_price=450.0,
        )
        # Iron condor has lower multiplier so fewer contracts
        assert iron_condor.contracts <= long_call.contracts

    def test_unknown_strategy_uses_1_0(self, sizer: PositionSizer) -> None:
        """Unknown strategy should default to multiplier of 1.0."""
        result = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.20,
            strategy="unknown_strategy",
            underlying_price=450.0,
        )
        long_call = sizer.calculate(
            account_value=100_000,
            option_price=1.00,
            confidence=0.85,
            implied_vol=0.20,
            strategy="long_call",
            underlying_price=450.0,
        )
        # Both use 1.0 multiplier, so should match
        assert result.contracts == long_call.contracts

    @pytest.mark.parametrize(
        "strategy,expected_mult",
        [
            ("long_call", 1.0),
            ("long_put", 1.0),
            ("covered_call", 0.5),
            ("cash_secured_put", 0.5),
            ("bull_call_spread", 0.6),
            ("bear_put_spread", 0.6),
            ("iron_condor", 0.4),
            ("long_straddle", 1.2),
            ("short_strangle", 0.8),
        ],
    )
    def test_strategy_multipliers_match(
        self, strategy: str, expected_mult: float
    ) -> None:
        """Each strategy should have the documented risk multiplier."""
        assert PositionSizer.STRATEGY_RISK[strategy] == expected_mult


class TestFloorAndCap:
    """Test floor of 1 contract, cap of 10."""

    def test_floor_of_1(self, sizer: PositionSizer) -> None:
        """Even with tiny account, should get at least 1 contract."""
        result = sizer.calculate(
            account_value=1_000,
            option_price=50.0,  # $5000/contract > account
            confidence=0.66,
            implied_vol=0.65,
            strategy="iron_condor",
            underlying_price=450.0,
        )
        assert result.contracts == 1

    def test_cap_of_10(self, sizer: PositionSizer) -> None:
        """Even with huge account and cheap options, cap at 10."""
        result = sizer.calculate(
            account_value=10_000_000,
            option_price=0.10,  # Very cheap
            confidence=1.0,
            implied_vol=0.10,
            strategy="long_straddle",  # 1.2x multiplier
            underlying_price=450.0,
        )
        assert result.contracts == 10

    def test_max_contracts_for_budget(self, sizer: PositionSizer) -> None:
        """max_contracts_for_budget should be budget / (price * 100)."""
        assert sizer.max_contracts_for_budget(5000, 2.50) == 20
        assert sizer.max_contracts_for_budget(5000, 0) == 0
        assert sizer.max_contracts_for_budget(249, 2.50) == 0
