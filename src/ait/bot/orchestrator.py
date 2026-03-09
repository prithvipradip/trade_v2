"""Main trading orchestrator — the bot's brain.

Runs the complete trading loop:
1. Pre-market: prepare data, train models, reconcile positions
2. Market hours: scan → predict → generate signals → validate → execute → monitor
3. Post-market: reconcile, generate daily report

This is the single entry point for all trading logic.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import date

from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.bot.state import DailyStats, StateManager
from ait.broker.account import AccountManager
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import Settings
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.data.options_chain import OptionsChainService
from ait.execution.executor import TradeExecutor
from ait.execution.portfolio import PortfolioManager
from ait.execution.reconciler import PositionReconciler
from ait.ml.ensemble import DirectionPredictor
from ait.ml.regime import RegimeDetector
from ait.ml.trainer import ModelTrainer
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.manager import RiskManager, TradeRequest
from ait.risk.pdt_guard import PDTGuard
from ait.risk.position_sizer import PositionSizer
from ait.sentiment.engine import SentimentEngine
from ait.strategies.base import SignalDirection
from ait.strategies.selector import StrategySelector
from ait.utils.logging import get_logger

log = get_logger("bot.orchestrator")


class TradingOrchestrator:
    """Orchestrates the complete autonomous trading lifecycle."""

    def __init__(
        self,
        settings: Settings,
        ibkr_client: IBKRClient,
    ) -> None:
        self._settings = settings
        self._ibkr = ibkr_client

        # Core services
        self._account = AccountManager(ibkr_client)
        self._market_data = MarketDataService(
            ibkr_client,
            polygon_api_key=settings.api_keys.polygon_api_key,
        )
        self._options_chain = OptionsChainService(
            ibkr_client, self._market_data, settings.options,
        )
        self._historical = HistoricalDataStore()

        # State
        self._state = StateManager()

        # Risk
        self._circuit_breaker = CircuitBreaker(settings.risk)
        self._pdt_guard = PDTGuard(settings.account, self._state)
        self._position_sizer = PositionSizer(settings.positions, settings.risk)
        self._risk_manager = RiskManager(
            settings.positions, settings.risk,
            self._account, self._circuit_breaker,
            self._pdt_guard, self._position_sizer,
        )

        # ML
        self._predictor = DirectionPredictor(settings.ml)
        self._regime_detector = RegimeDetector()
        self._trainer = ModelTrainer(
            settings.ml, self._predictor, self._market_data, self._historical,
        )

        # Sentiment
        self._sentiment = SentimentEngine(
            settings.sentiment, self._market_data,
            finnhub_api_key=settings.api_keys.finnhub_api_key,
        )

        # Trading
        self._strategy_selector = StrategySelector(settings.options)
        self._executor = TradeExecutor(ibkr_client, self._state, self._circuit_breaker)
        self._portfolio = PortfolioManager(
            ibkr_client, self._market_data, self._state,
            self._circuit_breaker, self._pdt_guard,
        )

        # Scheduling
        self._scheduler = MarketScheduler()
        self._reconciler = PositionReconciler(ibkr_client, self._state)

        # Notification callback (set by main.py)
        self._notify = None

        self._running = False

    def set_notification_callback(self, callback) -> None:
        """Set async callback for sending notifications."""
        self._notify = callback

    async def run(self) -> None:
        """Main loop — runs continuously, trading during market hours."""
        self._running = True
        log.info("orchestrator_starting", mode=self._settings.trading.mode)

        while self._running:
            try:
                phase = self._scheduler.get_current_phase()

                if phase == TradingPhase.PRE_MARKET:
                    await self._pre_market()
                    await self._scheduler.wait_for_phase(TradingPhase.MARKET_OPEN)

                elif phase == TradingPhase.MARKET_OPEN:
                    await self._trading_loop()

                elif phase == TradingPhase.POST_MARKET:
                    await self._post_market()
                    await self._scheduler.wait_for_phase(TradingPhase.PRE_MARKET)

                elif phase == TradingPhase.OFF_HOURS:
                    log.info("off_hours", next_open=str(self._scheduler))
                    await self._scheduler.wait_for_phase(TradingPhase.PRE_MARKET)

            except KeyboardInterrupt:
                log.info("shutdown_requested")
                break
            except Exception as e:
                log.error("orchestrator_error", error=str(e), traceback=traceback.format_exc())
                await asyncio.sleep(60)  # Brief pause on error

        await self._shutdown()

    async def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        self._running = False

    # --- Trading Phases ---

    async def _pre_market(self) -> None:
        """Pre-market preparation (runs 30 min before open)."""
        log.info("pre_market_starting")

        # 1. Reconcile positions with IBKR
        result = await self._reconciler.reconcile()
        if result.discrepancies:
            await self._send_notification(
                f"Position discrepancies found:\n" +
                "\n".join(result.discrepancies[:5])
            )

        # 2. Reset daily counters
        self._circuit_breaker.check_daily_reset()

        # 3. Ensure ML models are trained
        await self._trainer.ensure_models_ready(self._settings.trading.universe)

        # 4. Get VIX and market regime
        vix = await self._market_data.get_vix()
        spy_data = await self._market_data.get_historical("SPY", days=60)
        if spy_data is not None:
            regime = self._regime_detector.analyze(spy_data, vix)
            log.info("pre_market_regime", regime=regime.regime.value, vix=vix)

        log.info("pre_market_complete")

    async def _trading_loop(self) -> None:
        """Main trading loop during market hours."""
        log.info("trading_loop_starting")
        scan_interval = self._settings.trading.scan_interval_seconds

        while self._running and self._scheduler.get_current_phase() == TradingPhase.MARKET_OPEN:
            try:
                await self._trading_cycle()
            except Exception as e:
                log.error("trading_cycle_error", error=str(e))
                self._circuit_breaker.record_api_failure()

            # Wait for next scan
            await asyncio.sleep(scan_interval)

    async def _trading_cycle(self) -> None:
        """Single trading cycle: scan all symbols, check positions, execute signals."""
        # 1. Check circuit breaker
        if self._circuit_breaker.is_tripped:
            log.warning("trading_halted", reason=self._circuit_breaker.get_status().reason)
            return

        # 2. Check and manage existing positions
        positions = await self._portfolio.check_positions()
        exits_needed = [p for p in positions if p.should_exit]
        for pos in exits_needed:
            log.info("position_exit_triggered", symbol=pos.symbol, reason=pos.exit_reason)
            # TODO: Execute exit order via executor

        # 3. Check if we should avoid new trades (last 15 min)
        if self._scheduler.should_avoid_new_trades():
            log.debug("skipping_new_trades", reason="close_to_market_close")
            return

        # 4. Check fill status of pending orders
        await self._executor.check_fills()

        # 5. Scan universe for new opportunities
        vix = await self._market_data.get_vix()

        for symbol in self._settings.trading.universe:
            try:
                await self._scan_symbol(symbol, vix)
            except Exception as e:
                log.warning("symbol_scan_failed", symbol=symbol, error=str(e))

    async def _scan_symbol(self, symbol: str, vix: float | None) -> None:
        """Analyze a single symbol for trading opportunities."""
        # Get historical data for features
        hist = await self._market_data.get_historical(symbol, days=60)
        if hist is None or hist.empty:
            return

        # ML prediction
        prediction = self._predictor.predict(hist)
        if prediction is None:
            return

        # Check minimum confidence
        if prediction.confidence < self._settings.risk.min_confidence:
            log.debug(
                "low_confidence_skip",
                symbol=symbol,
                confidence=prediction.confidence,
            )
            return

        # Market regime
        regime = self._regime_detector.analyze(hist, vix)

        # Sentiment (optional — graceful degradation)
        sentiment = await self._sentiment.get_sentiment(symbol)

        # Adjust confidence with sentiment
        final_confidence = prediction.confidence
        if sentiment.sources_available > 0:
            sentiment_adj = sentiment.composite_score * self._sentiment.weight
            final_confidence = max(0, min(1, final_confidence + sentiment_adj))

        # Map ML direction to signal direction
        direction = prediction.direction

        # Override direction if regime strongly disagrees
        if regime.confidence > 0.8:
            from ait.ml.regime import MarketRegime
            if regime.regime == MarketRegime.HIGH_VOLATILITY:
                direction = SignalDirection.NEUTRAL  # Play defensive in high vol

        # Get IV rank for strategy selection
        iv_rank = await self._estimate_iv_rank(symbol)

        # Get options chains
        chains = await self._options_chain.get_chain(symbol)
        if not chains:
            return

        # Filter liquid options
        filtered_chains = [
            c.filter_liquid(self._settings.options) for c in chains
        ]

        # Generate signals across all strategies
        for chain in filtered_chains:
            if not chain.calls and not chain.puts:
                continue

            signals = self._strategy_selector.generate_all_signals(
                symbol=symbol,
                chain=chain,
                market_direction=direction,
                confidence=final_confidence,
                iv_rank=iv_rank,
                historical_data=hist,
            )

            # Try to execute the best signal
            for signal in signals[:1]:  # Only the top signal per chain
                await self._try_execute(signal, final_confidence, sentiment)

    async def _try_execute(self, signal, confidence: float, sentiment) -> None:
        """Validate and execute a signal."""
        # Build trade request for risk validation
        request = TradeRequest(
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            direction=signal.direction.value,
            contracts=signal.quantity,
            entry_price=signal.entry_price,
            option=signal.contract,
            confidence=confidence,
            implied_vol=signal.contract.implied_vol if signal.contract else 0.30,
            max_loss=signal.max_loss if signal.is_defined_risk else None,
        )

        validation = await self._risk_manager.validate_trade(request)

        if not validation.approved:
            log.info(
                "trade_rejected",
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                reason=validation.reason,
            )
            return

        # Execute
        trade_id = await self._executor.execute_signal(signal, validation.position_size)

        if trade_id:
            msg = (
                f"TRADE: {signal.action} {validation.position_size}x "
                f"{signal.symbol} {signal.strategy_name}\n"
                f"Entry: ${signal.entry_price:.2f} | "
                f"Max Loss: ${signal.max_loss:.0f} | "
                f"Confidence: {confidence:.0%}"
            )
            await self._send_notification(msg)

            # Update daily stats
            stats = self._state.get_daily_stats()
            stats.trades_taken += 1
            self._state.update_daily_stats(stats)

    async def _post_market(self) -> None:
        """Post-market reconciliation and reporting."""
        log.info("post_market_starting")

        # Reconcile with IBKR
        await self._reconciler.reconcile()

        # Generate daily summary
        summary = await self._portfolio.get_portfolio_summary()
        stats = self._state.get_daily_stats()

        report = (
            f"DAILY SUMMARY ({date.today().isoformat()})\n"
            f"Trades: {stats.trades_taken} | "
            f"Won: {stats.trades_won} | Lost: {stats.trades_lost}\n"
            f"Realized P&L: ${stats.total_pnl:.2f}\n"
            f"Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}\n"
            f"Open Positions: {summary['open_positions']}"
        )
        await self._send_notification(report)

        log.info("post_market_complete", summary=summary)

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("orchestrator_shutting_down")
        await self._send_notification("Bot shutting down")

    # --- Helpers ---

    async def _estimate_iv_rank(self, symbol: str) -> float:
        """Estimate IV rank (0-100) from recent volatility data."""
        hist = await self._market_data.get_historical(symbol, days=252)
        if hist is None or len(hist) < 60:
            return 50.0  # Default mid-range

        import numpy as np
        close = hist["Close"]
        log_returns = np.log(close / close.shift(1)).dropna()

        # Current 20-day vol
        current_vol = float(log_returns.tail(20).std() * np.sqrt(252))

        # 1-year vol range
        rolling_vol = log_returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 2:
            return 50.0

        vol_min = float(rolling_vol.min())
        vol_max = float(rolling_vol.max())
        vol_range = vol_max - vol_min

        if vol_range <= 0:
            return 50.0

        iv_rank = ((current_vol - vol_min) / vol_range) * 100
        return max(0, min(100, iv_rank))

    async def _send_notification(self, message: str) -> None:
        """Send notification via configured channel."""
        if self._notify:
            try:
                await self._notify(message)
            except Exception as e:
                log.warning("notification_failed", error=str(e))
