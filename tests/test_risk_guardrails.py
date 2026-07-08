"""Tests for the 2026-06-24 risk guardrails.

Focus: the aggregate capital-at-risk cap (#6) and that pending positions are
counted by the duplicate/limit guards (#1). The orchestrator-side guards
(cooldown #2, working-orders #3, settling #5) are integration-level and
covered by live behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.risk.manager import RiskManager, TradeRequest


def _manager(net_liq=200_000.0, open_positions=None):
    pos_config = MagicMock(max_open_positions=5, max_position_pct=0.05,
                           max_portfolio_delta=1.0, max_portfolio_risk_pct=0.20)
    risk_config = MagicMock(min_confidence=0.50, max_position_risk_pct=0.10,
                            max_credit_positions=6, credit_vix_halt=28.0)
    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=net_liq)
    account.can_afford = AsyncMock(return_value=True)
    account.get_buying_power = AsyncMock(return_value=net_liq)
    cb = MagicMock(is_tripped=False)
    pdt = MagicMock()
    pdt.can_open_position = MagicMock(return_value=(True, ""))
    sizer = MagicMock()
    sizer.calculate = MagicMock(return_value=MagicMock(contracts=1, max_risk_dollars=500))
    corr = MagicMock()
    corr.check_correlation = MagicMock(return_value=(True, ""))

    rm = RiskManager(pos_config, risk_config, account, cb, pdt, sizer,
                     correlation_guard=corr, state=None)
    rm._open_positions = open_positions or []
    return rm


def _req(symbol="SPY", strategy="iron_condor", max_loss=2000.0):
    return TradeRequest(
        symbol=symbol, strategy=strategy, direction="neutral",
        contracts=1, entry_price=1.0, confidence=0.95, max_loss=max_loss,
    )


class TestAggregateRiskCap:
    @pytest.mark.asyncio
    async def test_single_small_trade_passes(self):
        rm = _manager(net_liq=200_000)
        v = await rm.validate_trade(_req(max_loss=2000))  # 1% of account
        assert v.approved, v.reason

    @pytest.mark.asyncio
    async def test_aggregate_exceeds_cap_rejected(self):
        # Cap raised 10% -> 20% on 2026-06-30 (higher trade volume for
        # learning). Open book at $39k risk on $200k; +$2k pushes past $40k.
        existing = [
            {"symbol": "QQQ", "strategy": "iron_condor", "max_loss": 19500},
            {"symbol": "IWM", "strategy": "iron_condor", "max_loss": 19500},
        ]
        rm = _manager(net_liq=200_000, open_positions=existing)
        v = await rm.validate_trade(_req(symbol="DIA", max_loss=2000))
        assert not v.approved
        assert "aggregate risk" in v.reason

    @pytest.mark.asyncio
    async def test_aggregate_under_cap_passes(self):
        existing = [{"symbol": "QQQ", "strategy": "iron_condor", "max_loss": 5000}]
        rm = _manager(net_liq=200_000, open_positions=existing)
        v = await rm.validate_trade(_req(symbol="DIA", max_loss=2000))  # 5k+2k=7k < 20k
        assert v.approved, v.reason


class TestPendingCountedByGuards:
    @pytest.mark.asyncio
    async def test_duplicate_pending_position_blocks_reentry(self):
        # A PENDING iron_condor on SPY is in _open_positions -> duplicate guard fires.
        existing = [{"symbol": "SPY", "strategy": "iron_condor", "max_loss": 2000}]
        rm = _manager(open_positions=existing)
        v = await rm.validate_trade(_req(symbol="SPY", strategy="iron_condor"))
        assert not v.approved
        assert "duplicate" in v.reason

    @pytest.mark.asyncio
    async def test_max_positions_counts_pending(self):
        existing = [{"symbol": s, "strategy": "iron_condor", "max_loss": 500}
                    for s in ("SPY", "QQQ", "IWM", "DIA", "NVDA")]  # 5 = cap
        rm = _manager(open_positions=existing)
        v = await rm.validate_trade(_req(symbol="AMD"))
        assert not v.approved
        assert "max positions" in v.reason
