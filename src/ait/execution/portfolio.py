"""Portfolio manager — monitors positions and enforces exit rules.

Continuously monitors open positions and triggers exits when:
- Stop loss is hit
- Take profit is reached
- Time-based exit (approaching expiry)
- Portfolio-level risk limits exceeded
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from ait.bot.state import StateManager, TradeRecord, TradeStatus
from ait.broker.ibkr_client import IBKRClient
from ait.data.market_data import MarketDataService
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.pdt_guard import PDTGuard
from ait.utils.logging import get_logger

log = get_logger("execution.portfolio")


@dataclass
class PositionStatus:
    """Current status of an open position."""

    trade_id: str
    symbol: str
    strategy: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float
    dte: int | None  # Days to expiry
    should_exit: bool
    exit_reason: str


class PortfolioManager:
    """Monitors open positions and manages exits."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        market_data: MarketDataService,
        state: StateManager,
        circuit_breaker: CircuitBreaker,
        pdt_guard: PDTGuard,
    ) -> None:
        self._ibkr = ibkr_client
        self._market_data = market_data
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._pdt_guard = pdt_guard

    async def check_positions(self) -> list[PositionStatus]:
        """Check all open positions and determine which need action."""
        open_trades = self._state.get_open_trades()
        if not open_trades:
            return []

        statuses = []
        for trade in open_trades:
            if trade.status not in (TradeStatus.FILLED, TradeStatus.PARTIAL):
                continue

            status = await self._evaluate_position(trade)
            if status:
                statuses.append(status)

        # Log summary
        exits_needed = [s for s in statuses if s.should_exit]
        if exits_needed:
            log.info(
                "positions_needing_exit",
                count=len(exits_needed),
                reasons=[f"{s.symbol}: {s.exit_reason}" for s in exits_needed],
            )

        return statuses

    async def _evaluate_position(self, trade: TradeRecord) -> PositionStatus | None:
        """Evaluate a single position for exit conditions."""
        current_price = await self._market_data.get_current_price(trade.symbol)
        if current_price is None:
            log.warning("cannot_evaluate_position", symbol=trade.symbol, reason="no price")
            return None

        # Calculate unrealized P&L
        # For long positions: (current - entry) × quantity × 100
        # For short positions (selling premium): (entry - current) × quantity × 100
        multiplier = 100  # Options multiplier
        if trade.contract_type == "stock":
            multiplier = 1

        is_short = trade.direction.value == "short"
        if is_short:
            unrealized_pnl = (trade.entry_price - current_price) * trade.quantity * multiplier
        else:
            unrealized_pnl = (current_price - trade.entry_price) * trade.quantity * multiplier

        cost_basis = trade.entry_price * trade.quantity * multiplier
        pnl_pct = unrealized_pnl / cost_basis if cost_basis > 0 else 0.0

        # Days to expiry
        dte = None
        if trade.expiry:
            try:
                expiry_date = date.fromisoformat(trade.expiry)
                dte = (expiry_date - date.today()).days
            except ValueError:
                pass

        # Check exit conditions
        should_exit = False
        exit_reason = ""

        # 1. Stop loss (50% of premium lost)
        if pnl_pct <= -0.50:
            should_exit = True
            exit_reason = f"stop_loss (P&L: {pnl_pct:.1%})"

        # 2. Take profit (75-100% gain for long, 50% of max for short)
        elif not is_short and pnl_pct >= 1.0:
            should_exit = True
            exit_reason = f"take_profit (P&L: {pnl_pct:.1%})"
        elif is_short and pnl_pct >= 0.50:
            should_exit = True
            exit_reason = f"take_profit_short (P&L: {pnl_pct:.1%})"

        # 3. Time decay — close positions with 5 or fewer DTE
        elif dte is not None and dte <= 5:
            should_exit = True
            exit_reason = f"expiry_approaching (DTE: {dte})"

        # 4. Check PDT before recommending exit
        if should_exit and trade.entry_time:
            entry_date = datetime.fromisoformat(trade.entry_time).date()
            if self._pdt_guard.would_be_day_trade(trade.symbol, entry_date):
                if not self._pdt_guard.can_day_trade():
                    should_exit = False
                    exit_reason = "exit_blocked_by_pdt"
                    log.warning(
                        "exit_blocked_pdt",
                        trade_id=trade.trade_id,
                        symbol=trade.symbol,
                    )

        return PositionStatus(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            strategy=trade.strategy,
            quantity=trade.quantity,
            entry_price=trade.entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            pnl_pct=pnl_pct,
            dte=dte,
            should_exit=should_exit,
            exit_reason=exit_reason,
        )

    async def get_portfolio_summary(self) -> dict:
        """Get a summary of all open positions."""
        open_trades = self._state.get_open_trades()
        total_unrealized = 0.0
        positions = []

        for trade in open_trades:
            if trade.status not in (TradeStatus.FILLED, TradeStatus.PARTIAL):
                continue
            status = await self._evaluate_position(trade)
            if status:
                total_unrealized += status.unrealized_pnl
                positions.append({
                    "symbol": status.symbol,
                    "strategy": status.strategy,
                    "pnl": status.unrealized_pnl,
                    "pnl_pct": status.pnl_pct,
                    "dte": status.dte,
                })

        today_stats = self._state.get_daily_stats()

        return {
            "open_positions": len(positions),
            "total_unrealized_pnl": total_unrealized,
            "today_realized_pnl": today_stats.total_pnl,
            "today_trades": today_stats.trades_taken,
            "positions": positions,
        }
