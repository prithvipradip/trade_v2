"""Portfolio manager — monitors positions and enforces exit rules.

Continuously monitors open positions and triggers exits when:
- Trailing stop is hit (dynamic, tightens as profit grows)
- Breakeven stop activated (locks in entry after profit threshold)
- Partial profit targets reached (scale out at milestones)
- Take profit is reached (time-decay adjusted)
- Time-based exit (approaching expiry)
- Portfolio-level risk limits exceeded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ait.bot.state import StateManager, TradeRecord, TradeStatus
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import ExitConfig
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
    high_water_mark: float
    dte: int | None  # Days to expiry
    should_exit: bool
    exit_reason: str
    partial_exit_quantity: int = 0  # Non-zero if partial exit needed


class PortfolioManager:
    """Monitors open positions and manages exits with dynamic stop management."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        market_data: MarketDataService,
        state: StateManager,
        circuit_breaker: CircuitBreaker,
        pdt_guard: PDTGuard,
        exit_config: ExitConfig | None = None,
        earnings_calendar=None,
    ) -> None:
        self._ibkr = ibkr_client
        self._market_data = market_data
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._pdt_guard = pdt_guard
        self._exit_config = exit_config or ExitConfig()
        self._earnings = earnings_calendar

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
        exits_needed = [s for s in statuses if s.should_exit or s.partial_exit_quantity > 0]
        if exits_needed:
            log.info(
                "positions_needing_action",
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

        # Update high water mark — always persist so journaling data stays accurate
        prev_hwm = self._state.get_high_water_mark(trade.trade_id)
        hwm = max(prev_hwm, pnl_pct)
        self._state.update_high_water_mark(trade.trade_id, hwm)

        # Get learning overrides for stop/take-profit (if any)
        stop_loss_pct = self._exit_config.initial_stop_loss_pct
        trailing_stop_pct = self._exit_config.trailing_stop_pct
        breakeven_trigger = self._exit_config.breakeven_trigger_pct

        # Volatility-adjusted stops: widen stops for high-volatility underlyings
        if self._exit_config.volatility_adjusted_stops:
            vol_multiplier = await self._get_volatility_stop_multiplier(trade.symbol)
            stop_loss_pct = min(0.50, stop_loss_pct * vol_multiplier)   # Cap at 50% max loss
            trailing_stop_pct = min(0.35, trailing_stop_pct * vol_multiplier)

        # Determine dynamic stop level
        effective_stop = self._calculate_dynamic_stop(
            pnl_pct, hwm, stop_loss_pct, trailing_stop_pct, breakeven_trigger,
        )

        # Determine take profit target (time-decay adjusted)
        take_profit_long, take_profit_short = self._get_take_profit_targets(dte)

        # Check exit conditions
        should_exit = False
        exit_reason = ""
        partial_exit_quantity = 0

        # 1. Dynamic stop loss (trailing/breakeven)
        if pnl_pct <= effective_stop:
            should_exit = True
            if hwm >= breakeven_trigger and effective_stop >= 0:
                exit_reason = f"trailing_stop (P&L: {pnl_pct:.1%}, peak: {hwm:.1%}, stop: {effective_stop:.1%})"
            elif hwm >= breakeven_trigger:
                exit_reason = f"breakeven_stop (P&L: {pnl_pct:.1%}, peak: {hwm:.1%})"
            else:
                exit_reason = f"stop_loss (P&L: {pnl_pct:.1%})"

        # 2. Take profit (time-decay adjusted)
        elif not is_short and pnl_pct >= take_profit_long:
            should_exit = True
            exit_reason = f"take_profit (P&L: {pnl_pct:.1%}, target: {take_profit_long:.1%})"
        elif is_short and pnl_pct >= take_profit_short:
            should_exit = True
            exit_reason = f"take_profit_short (P&L: {pnl_pct:.1%})"

        # 3. Time decay — close positions with 5 or fewer DTE
        elif dte is not None and dte <= 5:
            should_exit = True
            exit_reason = f"expiry_approaching (DTE: {dte})"

        # 3b. Delta breach — close if directional risk ballooned
        # For neutral strategies (iron condor, straddle), delta should stay small
        # If abs(delta) > 0.50, position has taken on large directional exposure
        elif trade.strategy in ("iron_condor", "short_strangle", "long_straddle",
                                 "cash_secured_put", "covered_call"):
            pos_delta = self._get_position_delta(trade)
            if pos_delta is not None and abs(pos_delta) > 0.50:
                should_exit = True
                exit_reason = f"delta_breach (|Δ|={abs(pos_delta):.2f} > 0.50)"

        # 3c. IV crush pre-close — for SHORT premium strategies, close 2 days
        # before earnings to capture theta without eating the earnings IV crush
        elif self._earnings and trade.strategy in (
            "iron_condor", "short_strangle", "cash_secured_put", "covered_call",
        ):
            try:
                info = self._earnings.get_next_earnings(trade.symbol)
                if info and info.next_earnings_date:
                    from datetime import date as _d
                    days_to_earnings = (info.next_earnings_date - _d.today()).days
                    if 0 <= days_to_earnings <= 2 and pnl_pct > 0:
                        should_exit = True
                        exit_reason = f"pre_earnings_iv_crush (days={days_to_earnings}, pnl={pnl_pct:.1%})"
            except Exception:
                pass

        # 4. Check for partial exit milestones (only if not already exiting fully)
        if not should_exit and trade.quantity > 1:
            partial_exit_quantity = self._check_partial_exit(
                trade.trade_id, pnl_pct, trade.quantity,
            )
            if partial_exit_quantity > 0:
                exit_reason = f"partial_take_profit (P&L: {pnl_pct:.1%}, closing {partial_exit_quantity} contracts)"

        # 5. Check PDT before recommending exit
        if (should_exit or partial_exit_quantity > 0) and trade.entry_time:
            entry_date = datetime.fromisoformat(trade.entry_time).date()
            if self._pdt_guard.would_be_day_trade(trade.symbol, entry_date):
                if not self._pdt_guard.can_day_trade():
                    should_exit = False
                    partial_exit_quantity = 0
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
            high_water_mark=hwm,
            dte=dte,
            should_exit=should_exit,
            exit_reason=exit_reason,
            partial_exit_quantity=partial_exit_quantity,
        )

    def _calculate_dynamic_stop(
        self,
        pnl_pct: float,
        hwm: float,
        initial_stop: float,
        trailing_pct: float,
        breakeven_trigger: float,
    ) -> float:
        """Calculate the effective stop level based on profit history.

        Three tiers:
        1. Below breakeven trigger: use initial fixed stop (e.g., -50%)
        2. Crossed breakeven trigger: stop moves to 0% (breakeven)
        3. Above breakeven trigger: trail behind high water mark
        """
        if hwm < breakeven_trigger:
            # Tier 1: No profit lock — use initial stop
            return -initial_stop

        # Tier 2+: At minimum, protect entry (breakeven)
        # Tier 3: Trail behind HWM
        trailing_stop = hwm - trailing_pct
        return max(0.0, trailing_stop)  # Never worse than breakeven once triggered

    def _get_take_profit_targets(self, dte: int | None) -> tuple[float, float]:
        """Get time-decay adjusted take profit targets.

        As DTE decreases, we lower the target to capture profit
        before theta accelerates.
        """
        if not self._exit_config.time_decay_scaling or dte is None:
            return 1.0, 0.50  # Default: +100% long, +50% short

        if dte > 20:
            return 1.0, 0.50
        elif dte > 10:
            return 0.75, 0.40
        elif dte > 5:
            return 0.50, 0.30
        else:
            return 0.25, 0.20  # Very aggressive — grab what you can

    def _check_partial_exit(
        self,
        trade_id: str,
        pnl_pct: float,
        current_quantity: int,
    ) -> int:
        """Check if a partial exit milestone has been reached.

        Returns the number of contracts to close, or 0 if no partial exit needed.
        """
        if current_quantity <= 1:
            return 0

        prior_partials = self._state.get_partial_exits(trade_id)
        completed_levels = {p.get("pnl_level") for p in prior_partials}

        for level in self._exit_config.partial_exit_levels:
            level_pnl = level["pnl_pct"]
            close_pct = level["close_pct"]

            if level_pnl in completed_levels:
                continue

            if pnl_pct >= level_pnl:
                qty_to_close = max(1, int(current_quantity * close_pct))
                # Don't close everything — leave at least 1 contract
                qty_to_close = min(qty_to_close, current_quantity - 1)
                if qty_to_close > 0:
                    log.info(
                        "partial_exit_triggered",
                        trade_id=trade_id,
                        pnl_pct=pnl_pct,
                        level=level_pnl,
                        closing=qty_to_close,
                        remaining=current_quantity - qty_to_close,
                    )
                    return qty_to_close

        return 0

    def _get_position_delta(self, trade) -> float | None:
        """Fetch current aggregate delta for a position from IBKR.

        Returns None if delta can't be determined (single strike, missing data).
        For multi-leg strategies, sums delta across all legs.
        """
        try:
            if not self._ibkr or not self._ibkr.connected:
                return None
            total_delta = 0.0
            found_any = False
            for item in self._ibkr.ib.portfolio():
                if item.contract.symbol != trade.symbol:
                    continue
                if item.position == 0:
                    continue
                ticker = self._ibkr.ib.ticker(item.contract)
                if ticker and ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                    total_delta += ticker.modelGreeks.delta * item.position
                    found_any = True
            return total_delta if found_any else None
        except Exception:
            return None

    async def _get_volatility_stop_multiplier(self, symbol: str) -> float:
        """Calculate a stop width multiplier based on the underlying's volatility.

        High-vol stocks (TSLA, NVDA) get wider stops; low-vol (SPY) get tighter.
        Returns a multiplier around 1.0 (0.7 for low vol, up to 1.5 for high vol).
        """
        try:
            import numpy as np
            hist = await self._market_data.get_historical(symbol, days=30)
            if hist is None or len(hist) < 10 or "Close" not in hist.columns:
                return 1.0

            close = hist["Close"]
            log_returns = np.log(close / close.shift(1)).dropna()
            annualized_vol = float(log_returns.std() * np.sqrt(252))

            # Baseline: SPY ~16% annualized vol
            # Scale: 0.7x at 10% vol, 1.0x at 20%, 1.5x at 40%
            if annualized_vol <= 0:
                return 1.0
            multiplier = max(0.7, min(1.5, annualized_vol / 0.20))
            return multiplier
        except Exception:
            return 1.0

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
                    "hwm": status.high_water_mark,
                })

        today_stats = self._state.get_daily_stats()

        return {
            "open_positions": len(positions),
            "total_unrealized_pnl": total_unrealized,
            "today_realized_pnl": today_stats.total_pnl,
            "today_trades": today_stats.trades_taken,
            "positions": positions,
        }
