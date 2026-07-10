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
    vix: float = 0.0  # Current VIX level for regime-aware sizing


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

        # 2b. Weekend gap risk — require higher confidence after 2:30 PM ET Friday
        # for undefined-risk short strategies (short_strangle). Iron condors are
        # defined-risk so they're OK. Long straddles benefit from weekend vol.
        from datetime import datetime as _dt
        now = _dt.now()
        is_friday_late = now.weekday() == 4 and (now.hour > 14 or
                                                   (now.hour == 14 and now.minute >= 30))
        if is_friday_late and request.strategy in ("short_strangle",):
            if request.confidence < 0.90:
                return TradeValidation(
                    False,
                    f"weekend_gap_risk: {request.strategy} requires 90% conf on Fri PM "
                    f"(got {request.confidence:.2f})",
                )

        # (Dead daily-trade gate removed — audit 2026-07-07 item 3.3: it
        # checked a field TradeRequest never had AND a config key that lives
        # on TradingConfig, so it could never fire. The real daily budget is
        # enforced in orchestrator._get_trade_budget.)

        # 2c. Short-vol guardrails (audit R2 goal-align): the portfolio-delta
        # gate is dead (no greeks) and the daily breaker only sees REALIZED
        # P&L, so nothing else stops the whole book being short premium into
        # a gap. Two cheap brakes for credit strategies:
        from ait.strategies.base import CREDIT_STRATEGIES
        is_credit = request.strategy in CREDIT_STRATEGIES
        if is_credit:
            #  - vol-regime halt: no NEW short premium in a high-VIX regime
            # R7: FAIL CLOSED — a missing/zero VIX used to skip this gate
            # entirely (`request.vix` falsy), i.e. the vol-regime brake was
            # off exactly when the data layer was struggling.
            if not request.vix or request.vix <= 0:
                return TradeValidation(
                    False,
                    "credit entry halted: VIX unavailable (fail-closed)",
                )
            if request.vix >= self._risk_config.credit_vix_halt:
                return TradeValidation(
                    False,
                    f"credit entry halted: VIX {request.vix:.1f} >= "
                    f"{self._risk_config.credit_vix_halt:.0f}",
                )
            #  - concentration brake: cap simultaneous credit positions
            n_credit = sum(
                1 for p in self._open_positions
                if p.get("strategy") in CREDIT_STRATEGIES
            )
            if n_credit >= self._risk_config.max_credit_positions:
                return TradeValidation(
                    False,
                    f"short-premium cap: {n_credit}/"
                    f"{self._risk_config.max_credit_positions} credit positions open",
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

        # 6. Buying power check.
        # For CREDIT strategies entry_price is money RECEIVED — the old
        # `entry_price*contracts*100` check asked "can we afford income?"
        # (always yes) and never measured the capital the position consumes
        # (audit 2026-07-07 item 2.1). Capital consumed for credit trades is
        # approximated by the max_loss/tail estimate (now stress-based for
        # strangles); debit trades still pay the premium up front.
        account_value = await self._account.get_net_liquidation()
        # (is_credit computed at gate 2c above)
        if is_credit and getattr(request, "max_loss", None):
            estimated_cost = float(request.max_loss)
        else:
            estimated_cost = request.entry_price * request.contracts * 100
        if not await self._account.can_afford(estimated_cost):
            return TradeValidation(False, "insufficient buying power")

        # 6b. Per-position max risk (config: risk.max_position_risk_pct)
        _per_trade_pct = self._risk_config.max_position_risk_pct
        max_risk_per_trade = account_value * _per_trade_pct
        if hasattr(request, 'max_loss') and request.max_loss and request.max_loss > max_risk_per_trade:
            return TradeValidation(
                False,
                f"position risk ${request.max_loss:.0f} exceeds "
                f"{_per_trade_pct:.0%} account limit ${max_risk_per_trade:.0f}",
            )

        # 6b-2. AGGREGATE capital-at-risk — the sum of defined-risk across ALL
        # open positions plus this one must stay under a portfolio cap. The
        # per-trade 3% check alone lets the account stack many trades into a
        # large concentrated bet; this bounds the whole book.
        # Config-wired (audit item 3.3): positions.max_portfolio_risk_pct was
        # a phantom knob while this literal ruled. Default 0.20 (2026-06-30).
        PORTFOLIO_RISK_CAP_PCT = self._pos_config.max_portfolio_risk_pct
        portfolio_cap = account_value * PORTFOLIO_RISK_CAP_PCT
        open_risk = sum(float(p.get("max_loss", 0) or 0) for p in self._open_positions)
        new_risk = request.max_loss if (getattr(request, "max_loss", None) and request.max_loss) else estimated_cost
        if open_risk + new_risk > portfolio_cap:
            return TradeValidation(
                False,
                f"aggregate risk ${open_risk + new_risk:.0f} (open ${open_risk:.0f} "
                f"+ new ${new_risk:.0f}) exceeds {PORTFOLIO_RISK_CAP_PCT:.0%} "
                f"portfolio cap ${portfolio_cap:.0f}",
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
            # Deep-audit SR-H3: for credit strategies entry_price is the
            # CREDIT — sizing on it treats a $400-risk condor as a $100
            # position (~4-5x risk understatement). Size on per-contract
            # capital-at-risk when known.
            option_price=(
                request.max_loss / (max(1, request.contracts) * 100)
                if (is_credit and getattr(request, "max_loss", None))
                else request.entry_price
            ),
            confidence=request.confidence,
            implied_vol=request.implied_vol,
            strategy=request.strategy,
            underlying_price=0,  # Not needed for final sizing
            recent_losing_days=recent_losing_days,
            vix=request.vix,
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
