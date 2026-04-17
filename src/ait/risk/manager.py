"""Portfolio-level risk management.

Enforces position limits, portfolio Greeks limits, concentration rules,
and validates every trade before execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from ait.broker.account import AccountManager
from ait.config.settings import PositionConfig, RiskConfig
from ait.data.options_chain import OptionContract
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.correlation import CorrelationGuard
from ait.risk.pdt_guard import PDTGuard
from ait.risk.position_sizer import PositionSizer
from ait.utils.logging import get_logger

log = get_logger("risk.manager")


@dataclass
class PortfolioGreeks:
    """Aggregate portfolio Greeks."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Daily theta (dollars)
    vega: float = 0.0


@dataclass
class TradeValidation:
    """Result of trade validation."""

    approved: bool
    reason: str
    position_size: int = 0
    max_risk: float = 0.0


@dataclass
class TradeRequest:
    """A proposed trade to validate."""

    symbol: str
    strategy: str
    direction: str  # "long" or "short"
    contracts: int
    entry_price: float
    option: OptionContract | None = None
    confidence: float = 0.0
    implied_vol: float = 0.30
    # For multi-leg strategies
    max_loss: float | None = None  # Defined risk strategies


class RiskManager:
    """Central risk management — validates all trades before execution."""

    def __init__(
        self,
        position_config: PositionConfig,
        risk_config: RiskConfig,
        account_manager: AccountManager,
        circuit_breaker: CircuitBreaker,
        pdt_guard: PDTGuard,
        position_sizer: PositionSizer,
        correlation_guard: CorrelationGuard | None = None,
        state=None,
    ) -> None:
        self._pos_config = position_config
        self._risk_config = risk_config
        self._account = account_manager
        self._circuit_breaker = circuit_breaker
        self._pdt_guard = pdt_guard
        self._sizer = position_sizer
        self._correlation = correlation_guard or CorrelationGuard()
        self._state = state

        # Track current positions for limit checks
        self._open_positions: list[dict] = []
        self._portfolio_greeks = PortfolioGreeks()

    def _count_correlated_positions(self, new_symbol: str, open_symbols: list[str]) -> int:
        """Count open positions that are highly correlated with a new symbol.

        Uses CorrelationGuard's sector groups as the quick heuristic.
        """
        from ait.risk.correlation import SECTOR_GROUPS
        count = 0
        new_groups = {g for g, syms in SECTOR_GROUPS.items() if new_symbol in syms}
        for sym in open_symbols:
            if not sym or sym == new_symbol:
                continue
            sym_groups = {g for g, syms in SECTOR_GROUPS.items() if sym in syms}
            if new_groups & sym_groups:
                count += 1
        return count

    def _count_recent_losing_days(self) -> int:
        """Count consecutive losing days in recent trading history."""
        if not self._state:
            return 0
        try:
            from datetime import date, timedelta
            today = date.today()
            losing_streak = 0
            for days_back in range(1, 11):  # Look back up to 10 days
                d = today - timedelta(days=days_back)
                stats = self._state.get_daily_stats(d)
                if stats.trades_taken == 0:
                    continue  # Skip non-trading days
                if stats.total_pnl < 0:
                    losing_streak += 1
                else:
                    break  # Streak ended
            return losing_streak
        except Exception:
            return 0

    def update_positions(self, positions: list[dict]) -> None:
        """Update current position list from IBKR or state."""
        self._open_positions = positions
        self._recalculate_portfolio_greeks()

    async def validate_trade(self, request: TradeRequest) -> TradeValidation:
        """Validate a proposed trade against all risk rules.

        Checks in order (fastest rejections first):
        1. Circuit breaker
        2. Confidence threshold
        3. Position count limit
        4. Duplicate position check
        5. Correlation check (prevent correlated stacking)
        6. Buying power
        7. Portfolio delta limit
        8. Daily loss check
        9. Position sizing
        """
        # 1. Circuit breaker
        if self._circuit_breaker.is_tripped:
            status = self._circuit_breaker.get_status()
            return TradeValidation(False, f"circuit breaker: {status.reason}")

        # 2. Confidence threshold
        if request.confidence < self._risk_config.min_confidence:
            return TradeValidation(
                False,
                f"confidence {request.confidence:.2f} < min {self._risk_config.min_confidence}",
            )

        # 3a. Daily trade limit
        if hasattr(request, 'daily_trades_taken') and hasattr(self._risk_config, 'max_daily_trades'):
            max_daily = getattr(self._risk_config, 'max_daily_trades', 5)
            if request.daily_trades_taken >= max_daily:
                return TradeValidation(
                    False,
                    f"max daily trades reached ({request.daily_trades_taken}/{max_daily})",
                )

        # 3b. Position count limit
        if len(self._open_positions) >= self._pos_config.max_open_positions:
            return TradeValidation(
                False,
                f"max positions reached ({len(self._open_positions)}/{self._pos_config.max_open_positions})",
            )

        # 4. Duplicate position check (same symbol + same strategy)
        for pos in self._open_positions:
            if pos.get("symbol") == request.symbol and pos.get("strategy") == request.strategy:
                return TradeValidation(
                    False,
                    f"duplicate position: {request.symbol} {request.strategy} already open",
                )

        # 5. Correlation check (prevent stacking correlated positions)
        open_symbols = [p.get("symbol") for p in self._open_positions if p.get("symbol")]
        corr_allowed, corr_reason = self._correlation.check_correlation(
            request.symbol, open_symbols
        )
        if not corr_allowed:
            return TradeValidation(False, f"correlation block: {corr_reason}")

        # 6. Buying power check
        account_value = await self._account.get_net_liquidation()
        estimated_cost = request.entry_price * request.contracts * 100
        if not await self._account.can_afford(estimated_cost):
            return TradeValidation(False, "insufficient buying power")

        # 6b. Per-position max risk — no single trade should risk more than 3% of account
        max_risk_per_trade = account_value * 0.03
        if hasattr(request, 'max_loss') and request.max_loss and request.max_loss > max_risk_per_trade:
            return TradeValidation(
                False,
                f"position risk ${request.max_loss:.0f} exceeds 3% account limit ${max_risk_per_trade:.0f}",
            )

        # 6c. Concentration limit — no more than 20% of account in one symbol
        symbol_exposure = sum(
            abs(p.get("market_value", 0))
            for p in self._open_positions
            if p.get("symbol") == request.symbol
        )
        if (symbol_exposure + estimated_cost) > account_value * 0.20:
            return TradeValidation(
                False,
                f"symbol concentration: {request.symbol} exposure "
                f"${symbol_exposure + estimated_cost:.0f} exceeds 20% of ${account_value:.0f}",
            )

        # 7. Portfolio delta limit
        if request.option:
            new_delta = abs(
                self._portfolio_greeks.delta
                + request.option.delta * request.contracts * 100
            )
            max_delta_value = account_value * self._pos_config.max_portfolio_delta
            if new_delta > max_delta_value:
                return TradeValidation(
                    False,
                    f"portfolio delta {new_delta:.0f} would exceed limit {max_delta_value:.0f}",
                )

        # 8. Daily loss check
        if not self._circuit_breaker.check_daily_loss(account_value):
            return TradeValidation(False, "daily loss limit reached")

        # Count recent losing days for drawdown throttle
        recent_losing_days = self._count_recent_losing_days()

        # Correlation-adjusted sizing: reduce size when concentrated in correlated symbols
        correlated_count = self._count_correlated_positions(request.symbol, open_symbols)
        if correlated_count > 0:
            # Pre-reduce entry_price effectively by trimming contracts later
            # Log it so it's visible
            log.info("correlation_size_reduction",
                     symbol=request.symbol,
                     correlated_open=correlated_count)

        # 9. Position sizing
        size = self._sizer.calculate(
            account_value=account_value,
            option_price=request.entry_price,
            confidence=request.confidence,
            implied_vol=request.implied_vol,
            strategy=request.strategy,
            underlying_price=0,  # Not needed for final sizing
            recent_losing_days=recent_losing_days,
        )

        # Use the smaller of requested and recommended size
        final_contracts = min(request.contracts, size.contracts)

        # Correlation haircut: reduce size by 30% per correlated open position,
        # capped at 70% reduction (floor at 30% of calculated size)
        if correlated_count > 0:
            haircut = max(0.30, 1.0 - 0.30 * correlated_count)
            final_contracts = max(1, int(final_contracts * haircut))

        if final_contracts <= 0:
            return TradeValidation(False, "position sizer returned 0 contracts")

        log.info(
            "trade_validated",
            symbol=request.symbol,
            strategy=request.strategy,
            contracts=final_contracts,
            max_risk=size.max_risk_dollars,
            confidence=request.confidence,
        )

        return TradeValidation(
            approved=True,
            reason="all checks passed",
            position_size=final_contracts,
            max_risk=size.max_risk_dollars,
        )

    def _recalculate_portfolio_greeks(self) -> None:
        """Recalculate aggregate portfolio Greeks from open positions."""
        greeks = PortfolioGreeks()
        for pos in self._open_positions:
            qty = pos.get("quantity", 0)
            greeks.delta += pos.get("delta", 0) * qty * 100
            greeks.gamma += pos.get("gamma", 0) * qty * 100
            greeks.theta += pos.get("theta", 0) * qty * 100
            greeks.vega += pos.get("vega", 0) * qty * 100

        self._portfolio_greeks = greeks
        log.info(
            "portfolio_greeks",
            delta=f"{greeks.delta:.1f}",
            gamma=f"{greeks.gamma:.2f}",
            theta=f"{greeks.theta:.2f}",
            vega=f"{greeks.vega:.2f}",
            positions=len(self._open_positions),
        )
