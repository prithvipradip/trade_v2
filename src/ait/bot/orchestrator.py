"""Main trading orchestrator — the bot's brain.

Runs the complete trading loop:
1. Pre-market: prepare data, train models, reconcile positions, run learning cycle
2. Market hours: scan → predict → generate signals → validate → execute → monitor
3. Post-market: reconcile, learn from trades, generate daily report

This is the single entry point for all trading logic.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from datetime import date

from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.bot.state import DailyStats, StateManager, TradeRecord, TradeStatus
from ait.broker.account import AccountManager
from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.config.settings import Settings
from ait.data.earnings import EarningsCalendar
from ait.data.economic_calendar import EconomicCalendar
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.data.options_chain import OptionsChainService
from ait.data.multi_timeframe import MultiTimeframeAnalyzer
from ait.data.options_flow import OptionsFlowDetector
from ait.data.quality import DataQualityValidator
from ait.execution.executor import TradeExecutor
from ait.execution.portfolio import PortfolioManager, PositionStatus
from ait.execution.reconciler import PositionReconciler
from ait.learning.engine import LearningEngine
from ait.ml.ensemble import DirectionPredictor
from ait.ml.meta_label import MetaLabeler
from ait.ml.regime import RegimeDetector
from ait.ml.trainer import ModelTrainer
from ait.monitoring.analytics import TradeAnalytics
from ait.monitoring.watchdog import Watchdog
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.correlation import CorrelationGuard
from ait.risk.hedging import DeltaHedger
from ait.risk.manager import RiskManager, TradeRequest
from ait.risk.pdt_guard import PDTGuard
from ait.risk.position_sizer import PositionSizer
from ait.risk.capital_tiers import CapitalTierManager
from ait.sentiment.engine import SentimentEngine
from ait.learning.counterfactual import CounterfactualTracker
from ait.strategies.base import SignalDirection
from ait.strategies.selector import StrategySelector
from ait.strategies.thompson import ThompsonSampler
from ait.utils.logging import get_logger
from ait.utils.time import next_market_open

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
        self._correlation_guard = CorrelationGuard()
        self._risk_manager = RiskManager(
            settings.positions, settings.risk,
            self._account, self._circuit_breaker,
            self._pdt_guard, self._position_sizer,
            correlation_guard=self._correlation_guard,
        )
        self._delta_hedger = DeltaHedger()

        # ML
        self._predictor = DirectionPredictor(settings.ml)
        self._regime_detector = RegimeDetector()
        self._trainer = ModelTrainer(
            settings.ml, self._predictor, self._market_data, self._historical,
        )
        self._meta_labeler = MetaLabeler(
            min_probability=settings.meta_label.min_probability,
        ) if settings.meta_label.enabled else None

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
            exit_config=settings.exit,
        )

        # Scheduling
        self._scheduler = MarketScheduler()
        self._reconciler = PositionReconciler(ibkr_client, self._state)

        # Self-learning
        self._learning = LearningEngine(self._state)

        # Data quality & market intelligence
        self._data_quality = DataQualityValidator()
        self._earnings = EarningsCalendar()
        self._economic_cal = EconomicCalendar()
        self._flow_detector = OptionsFlowDetector()
        self._mtf_analyzer = MultiTimeframeAnalyzer()

        # Health monitoring
        self._watchdog = Watchdog()
        self._watchdog.register_component("trading_loop")
        self._watchdog.register_component("ibkr")
        self._watchdog.register_component("market_data")
        self._watchdog.on_recovery("ibkr", self._ibkr.ensure_connected)

        # Analytics
        self._analytics = TradeAnalytics()

        # Thompson sampling for strategy selection
        self._thompson = ThompsonSampler()

        # Counterfactual tracking for skipped trades
        self._counterfactual = CounterfactualTracker()

        # Capital tier manager — auto-selects strategies based on account size
        self._capital_tiers = CapitalTierManager()

        # Notification callback (set by main.py)
        self._notify = None

        self._running = False

        # Signal queue for entry timing optimization
        # Signals wait here until timing conditions are met (max 3 cycles)
        self._signal_queue: dict[str, dict] = {}  # symbol → {signal, confidence, sentiment, regime, age}

    def set_notification_callback(self, callback) -> None:
        """Set async callback for sending notifications."""
        self._notify = callback
        self._watchdog.set_alert_callback(callback)

    async def run(self) -> None:
        """Main loop — runs continuously, trading during market hours."""
        self._running = True
        log.info("orchestrator_starting", mode=self._settings.trading.mode)

        # Train models on startup if not yet trained (handles case where bot starts during market hours)
        await self._trainer.ensure_models_ready(self._settings.trading.universe)

        await self._send_notification(f"BOT STARTED | Mode: {self._settings.trading.mode} | {len(self._settings.trading.universe)} symbols")

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
                    log.info("off_hours", next_open=str(next_market_open()))
                    await self._scheduler.wait_for_phase(TradingPhase.PRE_MARKET)

            except KeyboardInterrupt:
                log.info("shutdown_requested")
                break
            except Exception as e:
                log.error("orchestrator_error", error=str(e), traceback=traceback.format_exc())
                await self._send_notification(f"ERROR: {str(e)}")
                self._watchdog.record_error("trading_loop", str(e))
                await self._watchdog.check_and_recover()
                await asyncio.sleep(60)

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
        self._data_quality.reset_tracking()

        # 3. Run self-learning cycle (analyze yesterday's trades)
        if self._settings.learning.enabled:
            learning_result = self._learning.run_learning_cycle(
                lookback_days=self._settings.learning.lookback_days
            )
            if learning_result["adaptations"] > 0:
                msg = (
                    f"LEARNING: {learning_result['adaptations']} adaptations applied\n"
                    + "\n".join(
                        f"  • {d['parameter']}: {d['old']} → {d['new']}"
                        for d in learning_result["details"]
                    )
                )
                await self._send_notification(msg)
                log.info("learning_adaptations_applied", summary=learning_result)

        # 4. Update correlation guard with recent price data (parallelized)
        async def _fetch_and_update_corr(sym: str) -> None:
            hist = await self._market_data.get_historical(sym, days=60)
            if hist is not None and "Close" in hist.columns:
                self._correlation_guard.update_price_data(sym, hist["Close"])

        await asyncio.gather(
            *[_fetch_and_update_corr(s) for s in self._settings.trading.universe],
            return_exceptions=True,
        )

        # 5. Ensure ML models are trained
        await self._trainer.ensure_models_ready(self._settings.trading.universe)

        # 6. Train/load meta-labeler (learns which signals to take vs skip)
        if self._meta_labeler is not None:
            if not self._meta_labeler.load_model():
                training_data = self._meta_labeler.build_training_data(self._state)
                if not training_data.empty:
                    stats = self._meta_labeler.train(training_data)
                    if stats:
                        log.info("meta_labeler_trained", stats=stats)
                else:
                    log.info("meta_labeler_skipped", reason="insufficient trade history")

        # 7. Get VIX and market regime
        vix = await self._market_data.get_vix()
        spy_data = await self._market_data.get_historical("SPY", days=60)
        if spy_data is not None:
            regime = self._regime_detector.analyze(spy_data, vix)
            log.info("pre_market_regime", regime=regime.regime.value, vix=vix)

        # 8. Apply Thompson sampling decay (forget old outcomes gradually)
        self._thompson.apply_decay()

        log.info(
            "pre_market_complete",
            learning_overrides=self._learning.get_current_adaptations(),
            model_version=self._predictor.model_version,
        )

    async def _monitor_positions_fast(self) -> None:
        """Fast position monitor — checks stops/exits every 30 seconds.

        Runs between full scan cycles so positions aren't unprotected
        for the full 5-minute scan interval.
        """
        try:
            positions = await self._portfolio.check_positions()

            # Skip positions already in CLOSING state (exit order already placed)
            closing_ids = {
                pe.trade_id for pe in self._executor._pending_exit_orders.values()
            }
            exits_needed = [
                p for p in positions
                if p.should_exit and p.trade_id not in closing_ids
            ]

            for pos in exits_needed:
                # Double-check trade isn't already CLOSING in DB
                trade = self._find_trade_record(pos.trade_id)
                if trade and trade.status == TradeStatus.CLOSING:
                    continue
                log.info("fast_monitor_exit", symbol=pos.symbol, reason=pos.exit_reason)
                await self._execute_exit(pos)

            # Also check pending order fills
            await self._executor.check_fills()
        except Exception as e:
            log.debug("fast_monitor_error", error=str(e))

    async def _trading_loop(self) -> None:
        """Main trading loop during market hours.

        Full scan every 5 minutes, but positions monitored every 30 seconds.
        """
        log.info("trading_loop_starting")
        scan_interval = self._settings.trading.scan_interval_seconds
        monitor_interval = 30  # Check positions every 30 seconds
        time_since_scan = 0

        while self._running and self._scheduler.get_current_phase() == TradingPhase.MARKET_OPEN:
            try:
                self._watchdog.heartbeat("trading_loop")

                if time_since_scan >= scan_interval:
                    # Full cycle: scan for new trades + check positions
                    await self._trading_cycle()
                    time_since_scan = 0
                else:
                    # Fast check: only monitor existing positions and fills
                    await self._monitor_positions_fast()
            except Exception as e:
                log.error("trading_cycle_error", error=str(e))
                self._circuit_breaker.record_api_failure()
                self._watchdog.record_error("trading_loop", str(e))

            await asyncio.sleep(monitor_interval)
            time_since_scan += monitor_interval

    async def _trading_cycle(self) -> None:
        """Single trading cycle: scan all symbols, check positions, execute signals."""
        # 1. Check circuit breaker
        if self._circuit_breaker.is_tripped:
            log.warning("trading_halted", reason=self._circuit_breaker.get_status().reason)
            await self._send_notification(f"CIRCUIT BREAKER: Trading halted - {self._circuit_breaker.get_status().reason}")
            return

        # 2. Sync risk manager with live positions (fixes stale position tracking)
        await self._sync_risk_manager_positions()

        # 3. Check and manage existing positions
        positions = await self._portfolio.check_positions()

        # Handle full exits
        exits_needed = [p for p in positions if p.should_exit]
        for pos in exits_needed:
            log.info("position_exit_triggered", symbol=pos.symbol, reason=pos.exit_reason)
            await self._execute_exit(pos)

        # Handle partial exits (scale-out at profit milestones)
        partial_exits = [p for p in positions if not p.should_exit and p.partial_exit_quantity > 0]
        for pos in partial_exits:
            log.info("partial_exit_triggered", symbol=pos.symbol, reason=pos.exit_reason,
                     quantity=pos.partial_exit_quantity)
            await self._execute_partial_exit(pos)

        # 3b. Thesis re-evaluation — exit early if original thesis invalidated
        remaining_positions = [p for p in positions if not p.should_exit and p.partial_exit_quantity == 0]
        for pos in remaining_positions:
            invalidated, reason = await self._check_thesis_valid(pos)
            if invalidated:
                log.info("thesis_invalidated", symbol=pos.symbol, reason=reason)
                pos.should_exit = True
                pos.exit_reason = f"thesis_invalidated: {reason}"
                await self._execute_exit(pos)

        # 4. Check portfolio delta hedging
        await self._check_hedging()

        # 5. Check if we should avoid new trades (last 15 min)
        if self._scheduler.should_avoid_new_trades():
            log.debug("skipping_new_trades", reason="close_to_market_close")
            return

        # 6. Check fill status of pending orders (entry + exit)
        _filled_entries, completed_exits = await self._executor.check_fills()

        # Process completed exits — update daily stats, Thompson, drift
        for ex in completed_exits:
            realized_pnl = ex["realized_pnl"]
            trade_id = ex["trade_id"]

            stats = self._state.get_daily_stats()
            stats.total_pnl += realized_pnl
            if realized_pnl > 0:
                stats.trades_won += 1
            else:
                stats.trades_lost += 1
                self._circuit_breaker.record_trade_result(realized_pnl)
            self._state.update_daily_stats(stats)

            trade = self._find_trade_by_id(trade_id)
            if trade:
                self._thompson.record_outcome(
                    strategy=trade.strategy,
                    won=realized_pnl > 0,
                    pnl=realized_pnl,
                )
                actual_dir = "bullish" if realized_pnl > 0 else "bearish"
                self._trainer.drift_detector.record_outcome(
                    trade_id=trade_id,
                    actual_direction=actual_dir,
                )

            msg = (
                f"EXIT FILLED: {trade.symbol if trade else trade_id}\n"
                f"Reason: {ex['exit_reason']}\n"
                f"Exit price: {ex['exit_price']:.2f}\n"
                f"P&L: ${realized_pnl:.2f}"
            )
            await self._send_notification(msg)
            log.info(
                "exit_fill_processed",
                trade_id=trade_id,
                pnl=realized_pnl,
                exit_price=ex["exit_price"],
            )

        # 7. Get effective universe (learning + capital tier filtering)
        adaptor = self._learning.adaptor

        # Get current account value for capital tier decisions
        account_snapshot = await self._account.get_snapshot()
        current_capital = account_snapshot.net_liquidation if account_snapshot else 10_000.0
        tier_config = self._capital_tiers.get_config(current_capital)

        # Filter universe: learning restrictions + capital tier affordability
        universe = [
            s for s in self._settings.trading.universe
            if adaptor.is_symbol_allowed(s)
        ]
        universe = self._capital_tiers.filter_universe(universe, current_capital)

        log.debug("capital_tier_active",
                  tier=tier_config.tier.value,
                  capital=f"${current_capital:,.0f}",
                  strategies=tier_config.allowed_strategies,
                  max_positions=tier_config.max_positions,
                  universe=universe)

        # 7b. Check if current hour is blocked by learning
        from datetime import datetime as dt_now
        current_hour = dt_now.now().hour
        if not adaptor.is_hour_allowed(current_hour):
            log.debug("hour_blocked_by_learning", hour=current_hour)
            return

        # 7c. Skip trading on/before major macro events (FOMC, CPI, NFP)
        if self._economic_cal.should_skip_trading():
            log.info("economic_event_skip", events=str(self._economic_cal.get_upcoming_events(days=2)))
            return

        # 8. Process queued signals first (entry timing optimization)
        await self._process_signal_queue()

        # 9. Scan universe for new opportunities
        vix = await self._market_data.get_vix()

        # Fetch cross-asset context once per scan cycle (VIX + SPY history for ML)
        market_context = await self._build_market_context()

        for symbol in universe:
            try:
                await self._scan_symbol(symbol, vix, market_context)
            except Exception as e:
                log.warning("symbol_scan_failed", symbol=symbol, error=str(e))

    async def _build_market_context(self) -> dict:
        """Fetch VIX and SPY historical data for ML cross-asset features."""
        context = {}
        try:
            vix_hist = await self._market_data.get_historical("^VIX", days=120)
            if vix_hist is not None and len(vix_hist) > 20:
                context["vix"] = vix_hist
        except Exception:
            pass
        try:
            spy_hist = await self._market_data.get_historical("SPY", days=120)
            if spy_hist is not None and len(spy_hist) > 20:
                context["spy"] = spy_hist
        except Exception:
            pass
        return context

    async def _scan_symbol(self, symbol: str, vix: float | None, market_context: dict | None = None) -> None:
        """Analyze a single symbol for trading opportunities.

        Parallelizes data fetches: historical, sentiment, and IV rank
        are fetched concurrently with asyncio.gather.
        """
        adaptor = self._learning.adaptor

        # Parallel fetch: historical data, sentiment, IV rank, intraday
        # 2 years of history for robust features (iv_rank, vol percentiles, trend)
        hist_task = self._market_data.get_historical(symbol, days=504)
        sentiment_task = self._sentiment.get_sentiment(symbol)
        iv_rank_task = self._estimate_iv_rank(symbol)
        intraday_task = self._market_data.get_intraday(symbol, interval="5m", days=1)

        hist, sentiment, iv_rank, intraday = await asyncio.gather(
            hist_task, sentiment_task, iv_rank_task, intraday_task,
            return_exceptions=True,
        )

        # Handle exceptions from parallel fetches
        if isinstance(hist, Exception):
            log.warning("hist_data_error", symbol=symbol, error=str(hist))
            return
        if hist is None or hist.empty:
            log.warning("hist_data_empty", symbol=symbol,
                        hint="No historical data — check market data subscription or data source")
            return
        if isinstance(sentiment, Exception):
            sentiment = None
        if isinstance(iv_rank, Exception):
            iv_rank = 50.0
        if isinstance(intraday, Exception):
            intraday = None

        # Data quality check on historical data
        if "Close" in hist.columns:
            prices = hist["Close"].tolist()
            if not self._data_quality.validate_historical(symbol, prices):
                return

        # ML prediction (with cross-asset context)
        prediction = self._predictor.predict(
            hist, symbol=symbol, market_context=market_context
        )
        if prediction is None:
            log.warning("ml_prediction_none", symbol=symbol)
            return

        log.info("ml_prediction", symbol=symbol,
                 direction=prediction.direction.value,
                 confidence=f"{prediction.confidence:.3f}")

        # Record prediction in drift detector for accuracy tracking
        drift = self._trainer.drift_detector
        drift.record_prediction(
            trade_id=f"{symbol}-{prediction.direction.value}",
            direction=prediction.direction.value,
            confidence=prediction.confidence,
        )

        # Use learning-adjusted confidence threshold
        # Use standard confidence threshold — per-symbol models are now accurate
        min_confidence = adaptor.get_confidence_override() or self._settings.risk.min_confidence
        if prediction.confidence < min_confidence:
            log.debug(
                "low_confidence_skip",
                symbol=symbol,
                confidence=prediction.confidence,
                threshold=min_confidence,
            )
            # Record counterfactual for low-confidence skips
            self._counterfactual.record_skip(
                symbol=symbol,
                strategy="unknown",
                direction=prediction.direction.value,
                confidence=prediction.confidence,
                entry_price=float(hist["Close"].iloc[-1]) if "Close" in hist.columns else 0,
                reject_reason="low_confidence",
            )
            return

        # Market regime
        regime = self._regime_detector.analyze(hist, vix)

        # Adjust confidence with sentiment (already fetched in parallel)
        final_confidence = prediction.confidence
        if sentiment and hasattr(sentiment, "sources_available") and sentiment.sources_available > 0:
            sentiment_adj = sentiment.composite_score * self._sentiment.weight
            final_confidence = max(0, min(1, final_confidence + sentiment_adj))

        # Multi-timeframe analysis: boost/penalize confidence based on alignment
        mtf = self._mtf_analyzer.analyze(hist, intraday)
        final_confidence = max(0, min(1, final_confidence + mtf.confidence_boost))

        # Map ML direction to signal direction
        direction = prediction.direction

        # Override direction if regime strongly disagrees
        if regime.confidence > 0.8:
            from ait.ml.regime import MarketRegime
            if regime.regime == MarketRegime.HIGH_VOLATILITY:
                direction = SignalDirection.NEUTRAL

        # Check earnings proximity — skip if too close to earnings
        if self._earnings.is_near_earnings(symbol):
            log.info("skipping_near_earnings", symbol=symbol)
            return

        # Pre-compute features once for meta-labeler (Phase 3.3)
        from ait.ml.features import FeatureEngine
        features_df = FeatureEngine().compute(hist)

        # Meta-label gate: ask secondary model if this signal is worth taking.
        # Uses primary confidence + market context to filter false positives.
        if self._meta_labeler is not None and self._meta_labeler.is_trained:
            last_features = {}
            if not features_df.empty:
                last_row = features_df.iloc[-1]
                last_features = {
                    "rsi_14": float(last_row.get("rsi_14", 0)),
                    "rsi_7": float(last_row.get("rsi_7", 0)),
                    "bb_position": float(last_row.get("bb_position", 0)),
                    "volume_sma_20_ratio": float(last_row.get("volume_sma_20_ratio", 0)),
                    "realized_vol_20": float(last_row.get("realized_vol_20", 0)),
                    "atr_pct": float(last_row.get("atr_pct", 0)),
                    "weekly_trend_aligned": float(last_row.get("weekly_trend_aligned", 0)),
                    "volume_confirmation": float(last_row.get("volume_confirmation", 0)),
                    "macd_hist": float(last_row.get("macd_hist", 0)),
                    "price_vs_sma_20": float(last_row.get("price_vs_sma_20", 0)),
                    "sma_10_20_cross": float(last_row.get("sma_10_20_cross", 0)),
                }

            from datetime import datetime as dt
            meta_context = {
                "primary_confidence": final_confidence,
                "regime_trending_up": 1.0 if regime and regime.regime.value == "trending_up" else 0.0,
                "regime_trending_down": 1.0 if regime and regime.regime.value == "trending_down" else 0.0,
                "regime_high_vol": 1.0 if regime and regime.regime.value == "high_volatility" else 0.0,
                "regime_range_bound": 1.0 if regime and regime.regime.value == "range_bound" else 0.0,
                "vix": vix or 0,
                "iv_rank": iv_rank,
                "sentiment_score": sentiment.composite_score if sentiment and hasattr(sentiment, "composite_score") else 0,
                "hour_of_day": dt.now().hour,
                **last_features,
            }

            meta_signal = self._meta_labeler.predict(meta_context)
            if meta_signal is not None and not meta_signal.take_trade:
                log.info(
                    "meta_label_reject",
                    symbol=symbol,
                    probability=f"{meta_signal.probability:.3f}",
                    direction=direction.value,
                    confidence=f"{final_confidence:.3f}",
                )
                # Record counterfactual for meta-label rejects
                self._counterfactual.record_skip(
                    symbol=symbol,
                    strategy="unknown",
                    direction=direction.value,
                    confidence=final_confidence,
                    entry_price=float(hist["Close"].iloc[-1]) if "Close" in hist.columns else 0,
                    reject_reason="meta_label_reject",
                )
                return

        # IV rank already fetched in parallel above

        # Get options chains
        chains = await self._options_chain.get_chain(symbol)
        if not chains:
            return

        # Filter liquid options
        filtered_chains = [
            c.filter_liquid(self._settings.options) for c in chains
        ]

        # Options flow analysis — detect unusual activity
        underlying_price = hist["Close"].iloc[-1] if "Close" in hist.columns else 0
        for chain in filtered_chains:
            if chain.calls or chain.puts:
                call_data = [
                    {"strike": c.strike, "volume": c.volume, "open_interest": c.open_interest,
                     "last_price": c.last, "delta": c.delta}
                    for c in (chain.calls or [])
                ]
                put_data = [
                    {"strike": p.strike, "volume": p.volume, "open_interest": p.open_interest,
                     "last_price": p.last, "delta": p.delta}
                    for p in (chain.puts or [])
                ]
                flow = self._flow_detector.analyze_chain(
                    symbol, call_data, put_data, underlying_price
                )
                # Hard gate: if strong flow disagrees with ML direction, reject entirely
                if flow.bias_strength > 0.7:
                    flow_disagrees = (
                        (flow.overall_bias == "bearish" and direction == SignalDirection.BULLISH) or
                        (flow.overall_bias == "bullish" and direction == SignalDirection.BEARISH)
                    )
                    if flow_disagrees:
                        log.info("flow_hard_gate_reject", symbol=symbol,
                                 flow_bias=flow.overall_bias, ml_direction=direction.value,
                                 bias_strength=flow.bias_strength)
                        return  # Skip this symbol entirely

                # Boost confidence if flow agrees with prediction direction
                if flow.bias_strength > 0.5:
                    if (flow.overall_bias == "bullish" and direction == SignalDirection.BULLISH) or \
                       (flow.overall_bias == "bearish" and direction == SignalDirection.BEARISH):
                        final_confidence = min(1.0, final_confidence + flow.bias_strength * 0.1)
                        log.info("flow_confidence_boost", symbol=symbol, bias=flow.overall_bias,
                                 boost=flow.bias_strength * 0.1)
                break  # Only analyze first chain for flow

        # Generate signals across all enabled strategies (learning may have disabled some)
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

            # Filter out signals from disabled strategies
            signals = [
                s for s in signals
                if adaptor.is_strategy_enabled(s.strategy_name)
            ]

            # Re-rank signals using Thompson sampling (exploration/exploitation)
            if signals:
                strategy_names = [s.strategy_name for s in signals]
                ranked_names = self._thompson.rank_strategies(strategy_names)
                # Build a name→signal map and reorder
                sig_map = {s.strategy_name: s for s in signals}
                signals = [sig_map[n] for n in ranked_names if n in sig_map]

            # Try to execute the best signal (with entry timing check)
            for signal in signals[:1]:
                if self._should_queue_signal(signal, hist):
                    self._signal_queue[signal.symbol] = {
                        "signal": signal,
                        "confidence": final_confidence,
                        "sentiment": sentiment,
                        "regime": regime,
                        "age": 0,
                    }
                    log.info("signal_queued", symbol=signal.symbol,
                             strategy=signal.strategy_name, reason="entry_timing")
                else:
                    await self._try_execute(signal, final_confidence, sentiment, regime)

    def _get_trade_budget(self) -> int:
        """Return how many trades are allowed this scan based on time of day.

        Reserves trades across the session instead of firing all at open:
          - First hour (9:30-10:30):  2 trades max (high-conviction only)
          - Mid-day (10:30-14:00):    up to 2 more
          - Power hour (14:00-16:00): remaining budget
        This way the bot can act on opportunities throughout the day.
        """
        from datetime import datetime as dt
        max_trades = self._settings.trading.max_daily_trades
        daily_stats = self._state.get_daily_stats(date.today())
        taken = daily_stats.trades_taken

        now = dt.now()
        hour_min = now.hour + now.minute / 60.0

        if hour_min < 10.5:      # 9:30-10:30 — first hour
            budget = 2
        elif hour_min < 14.0:    # 10:30-2:00 — mid-day
            budget = 4
        else:                    # 2:00-4:00 — power hour, release all
            budget = max_trades

        return min(budget, max_trades) - taken

    async def _try_execute(self, signal, confidence: float, sentiment, regime) -> None:
        """Validate and execute a signal."""
        adaptor = self._learning.adaptor

        # Check trade budget (time-based pacing)
        remaining = self._get_trade_budget()
        if remaining <= 0:
            daily_stats = self._state.get_daily_stats(date.today())
            log.info("trade_budget_exhausted",
                     trades_taken=daily_stats.trades_taken,
                     max=self._settings.trading.max_daily_trades,
                     budget_remaining=remaining)
            return

        # First hour requires higher confidence (only take the best setups)
        from datetime import datetime as dt
        hour_min = dt.now().hour + dt.now().minute / 60.0
        if hour_min < 10.5 and confidence < 0.85:
            log.info("first_hour_confidence_gate",
                     symbol=signal.symbol, confidence=f"{confidence:.2f}",
                     required=0.85)
            return

        # Check if this symbol already has a pending order
        pending_symbols = set()
        for oid, pending in self._executor._pending_orders.items():
            if hasattr(pending, 'signal') and hasattr(pending.signal, 'symbol'):
                pending_symbols.add(pending.signal.symbol)
        if signal.symbol in pending_symbols:
            log.debug("symbol_has_pending_order", symbol=signal.symbol)
            return

        # Check if position would hold through earnings (IV crush risk)
        if signal.expiry:
            from datetime import date as date_cls
            expiry_date = signal.expiry if isinstance(signal.expiry, date_cls) else date_cls.fromisoformat(str(signal.expiry))
            if self._earnings.would_hold_through_earnings(signal.symbol, date.today(), expiry_date):
                log.info("trade_blocked_earnings", symbol=signal.symbol, expiry=str(signal.expiry))
                return

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
            # Record counterfactual for risk-rejected trades
            self._counterfactual.record_skip(
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                direction=signal.direction.value,
                confidence=confidence,
                entry_price=signal.entry_price,
                reject_reason=f"risk_{validation.reason}",
            )
            return

        # Apply learning-based sizing multiplier
        strategy_mult = adaptor.get_strategy_multiplier(signal.strategy_name)
        adjusted_size = max(1, int(validation.position_size * strategy_mult))

        # Execute
        trade_id = await self._executor.execute_signal(signal, adjusted_size)

        if trade_id:
            # Store full entry context for thesis re-evaluation and learning
            regime_str = regime.regime.value if regime else ""
            self._state.set_state(f"trade_regime_{trade_id}", regime_str)
            self._state.save_trade_context(
                trade_id=trade_id,
                direction=signal.direction.value,
                confidence=confidence,
                regime=regime_str,
                vix=await self._market_data.get_vix() or 0,
                iv_rank=signal.iv_rank if hasattr(signal, "iv_rank") else 0,
                sentiment_score=sentiment.composite_score if sentiment and hasattr(sentiment, "composite_score") else 0,
            )

            msg = (
                f"TRADE: {signal.action} {adjusted_size}x "
                f"{signal.symbol} {signal.strategy_name}\n"
                f"Entry: ${signal.entry_price:.2f} | "
                f"Max Loss: ${signal.max_loss:.0f} | "
                f"Confidence: {confidence:.0%}"
            )
            if strategy_mult != 1.0:
                msg += f"\nLearning multiplier: {strategy_mult:.2f}x"
            await self._send_notification(msg)

            # Update daily stats
            stats = self._state.get_daily_stats()
            stats.trades_taken += 1
            self._state.update_daily_stats(stats)

    async def _execute_exit(self, pos: PositionStatus) -> None:
        """Execute an exit order for a position that needs to be closed."""
        trade = self._find_trade_record(pos.trade_id)
        if not trade:
            log.error("exit_trade_not_found", trade_id=pos.trade_id)
            return

        if not await self._ibkr.ensure_connected():
            log.error("exit_failed_no_connection", trade_id=pos.trade_id)
            self._watchdog.record_error("ibkr", "disconnected during exit")
            return

        try:
            exit_trade = None

            # Build the closing order (reverse of entry)
            if trade.contract_type in ("call", "put"):
                contract = ContractBuilder.option(
                    symbol=trade.symbol,
                    expiry=trade.expiry,
                    strike=trade.strike,
                    right="C" if trade.contract_type == "call" else "P",
                )
                qualified = await self._ibkr.qualify_contract(contract)
                if not qualified:
                    log.error("exit_qualification_failed", trade_id=pos.trade_id)
                    return

                # Reverse direction: if we bought, now sell; if we sold, now buy
                close_action = "SELL" if trade.direction.value == "long" else "BUY"
                order = OrderBuilder.market(action=close_action, quantity=trade.quantity)
                exit_trade = await self._ibkr.place_order(qualified, order)

            elif trade.contract_type in ("spread", "iron_condor"):
                # Multi-leg close: submit reverse combo order
                import json
                legs = json.loads(trade.legs)
                if legs:
                    exit_trade = await self._close_multi_leg(trade, legs)
                else:
                    log.warning("no_legs_data_for_close", trade_id=pos.trade_id)
                    return

            if exit_trade is None:
                log.error("exit_order_not_placed", trade_id=pos.trade_id)
                return

            # Mark trade as CLOSING — actual close happens when the exit
            # order fills (detected in executor.check_fills).
            self._state.update_trade_status(pos.trade_id, TradeStatus.CLOSING)

            # Register the exit order with the executor for fill tracking
            if exit_trade.order.orderId:
                self._executor.register_exit_order(
                    order_id=exit_trade.order.orderId,
                    trade_id=pos.trade_id,
                    exit_reason=pos.exit_reason,
                    estimated_pnl=pos.unrealized_pnl,
                )

            log.info(
                "exit_order_placed",
                trade_id=pos.trade_id,
                symbol=pos.symbol,
                reason=pos.exit_reason,
                status="CLOSING",
            )

        except Exception as e:
            log.error("exit_execution_failed", trade_id=pos.trade_id, error=str(e))
            self._watchdog.record_error("trading_loop", f"exit failed: {e}")

    async def _execute_partial_exit(self, pos: PositionStatus) -> None:
        """Execute a partial exit — close some contracts while keeping the rest open."""
        trade = self._find_trade_record(pos.trade_id)
        if not trade:
            log.error("partial_exit_trade_not_found", trade_id=pos.trade_id)
            return

        if not await self._ibkr.ensure_connected():
            log.error("partial_exit_failed_no_connection", trade_id=pos.trade_id)
            return

        try:
            qty_to_close = pos.partial_exit_quantity

            if trade.contract_type in ("call", "put"):
                contract = ContractBuilder.option(
                    symbol=trade.symbol,
                    expiry=trade.expiry,
                    strike=trade.strike,
                    right="C" if trade.contract_type == "call" else "P",
                )
                qualified = await self._ibkr.qualify_contract(contract)
                if not qualified:
                    log.error("partial_exit_qualification_failed", trade_id=pos.trade_id)
                    return

                close_action = "SELL" if trade.direction.value == "long" else "BUY"
                order = OrderBuilder.market(action=close_action, quantity=qty_to_close)
                await self._ibkr.place_order(qualified, order)

            # Record partial exit
            partial_pnl = pos.unrealized_pnl * (qty_to_close / trade.quantity)
            # Find which level was triggered
            pnl_level = pos.pnl_pct
            for level in self._settings.exit.partial_exit_levels:
                if pos.pnl_pct >= level["pnl_pct"]:
                    pnl_level = level["pnl_pct"]

            self._state.record_partial_exit(
                trade_id=pos.trade_id,
                quantity=qty_to_close,
                price=pos.current_price,
                pnl=partial_pnl,
                pnl_level=pnl_level,
            )

            # Update remaining quantity
            new_qty = trade.quantity - qty_to_close
            self._state.update_trade_quantity(pos.trade_id, new_qty)

            # Update daily stats with realized portion
            stats = self._state.get_daily_stats()
            stats.total_pnl += partial_pnl
            if partial_pnl > 0:
                stats.trades_won += 1
            self._state.update_daily_stats(stats)

            msg = (
                f"PARTIAL EXIT: {trade.symbol} {trade.strategy}\n"
                f"Closed {qty_to_close}/{trade.quantity} contracts\n"
                f"P&L: ${partial_pnl:.2f} ({pos.pnl_pct:.1%})\n"
                f"Remaining: {new_qty} contracts (trailing stop active)"
            )
            await self._send_notification(msg)

            log.info(
                "partial_exit_executed",
                trade_id=pos.trade_id,
                symbol=pos.symbol,
                closed=qty_to_close,
                remaining=new_qty,
                pnl=partial_pnl,
            )

        except Exception as e:
            log.error("partial_exit_failed", trade_id=pos.trade_id, error=str(e))

    async def _close_multi_leg(self, trade: TradeRecord, legs: list[dict]):
        """Close a multi-leg position by reversing the combo."""
        qualified_legs = []
        for leg in legs:
            contract = ContractBuilder.option(
                symbol=trade.symbol,
                expiry=leg.get("expiry", trade.expiry),
                strike=leg["strike"],
                right=leg["right"],
            )
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                log.error("leg_exit_qualification_failed", trade_id=trade.trade_id, leg=leg)
                return

            # Reverse the action
            original_action = leg.get("action", "BUY")
            close_action = "SELL" if original_action == "BUY" else "BUY"
            qualified_legs.append({
                "conId": qualified.conId,
                "action": close_action,
                "ratio": leg.get("ratio", 1),
            })

        combo = ContractBuilder.combo(symbol=trade.symbol, legs=qualified_legs)

        # Determine if we're closing a debit (long) or credit (short) position
        # Debit close (straddles, long spreads) → we SELL the combo
        # Credit close (iron condors, short spreads) → we BUY the combo back
        sell_legs = sum(1 for lg in qualified_legs if lg["action"] == "SELL")
        buy_legs = len(qualified_legs) - sell_legs
        combo_action = "SELL" if sell_legs > buy_legs else "BUY"

        # Try to get mid-price from IBKR for a limit order
        limit_price = None
        try:
            qualified_combo = await self._ibkr.qualify_contract(combo)
            if qualified_combo:
                self._ibkr.ib.reqMktData(qualified_combo, "", False, False)
                await asyncio.sleep(0.5)
                ticker = self._ibkr.ib.ticker(qualified_combo)
                if ticker:
                    import math
                    bid = ticker.bid if not math.isnan(ticker.bid) else None
                    ask = ticker.ask if not math.isnan(ticker.ask) else None
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        limit_price = round((bid + ask) / 2, 2)
                    elif bid is not None and bid > 0:
                        limit_price = round(bid, 2)
                    elif ask is not None and ask > 0:
                        limit_price = round(ask, 2)
        except Exception as e:
            log.warning("combo_mid_price_failed", error=str(e))

        # Use market order as fallback if we couldn't get a limit price
        if limit_price and limit_price > 0:
            log.info("combo_exit_limit", symbol=trade.symbol, action=combo_action,
                     limit_price=limit_price)
            order = OrderBuilder.combo_limit(
                action=combo_action, quantity=trade.quantity, limit_price=limit_price
            )
        else:
            log.info("combo_exit_market_fallback", symbol=trade.symbol, action=combo_action)
            order = OrderBuilder.market(action=combo_action, quantity=trade.quantity)

        return await self._ibkr.place_order(combo, order)

    async def _post_market(self) -> None:
        """Post-market reconciliation, learning, and reporting."""
        log.info("post_market_starting")

        # 1. Reconcile with IBKR
        await self._reconciler.reconcile()

        # 2. Run self-learning cycle
        if self._settings.learning.enabled:
            learning_result = self._learning.run_learning_cycle(
                lookback_days=self._settings.learning.lookback_days
            )
            log.info("post_market_learning", result=learning_result)

        # 3. Evaluate counterfactual outcomes (what would skipped trades have done?)
        if self._counterfactual.pending_count > 0:
            prices = {}
            for sym in self._settings.trading.universe:
                price = await self._market_data.get_current_price(sym)
                if price:
                    prices[sym] = price
            evaluated = self._counterfactual.evaluate_outcomes(prices)
            if evaluated > 0:
                cf_analysis = self._counterfactual.get_analysis()
                log.info(
                    "counterfactual_analysis",
                    evaluated=evaluated,
                    filter_accuracy=f"{cf_analysis['filter_accuracy']:.0%}",
                    missed=cf_analysis.get("missed_opportunities", 0),
                )

        # 4. Check drift status
        drift_report = self._trainer.drift_detector.check_drift()
        if drift_report.is_drifting:
            log.warning("post_market_drift", accuracy=f"{drift_report.accuracy:.2%}", reason=drift_report.reason)

        # 5. Generate analytics
        metrics = self._analytics.get_performance(lookback_days=30)

        # 6. Generate daily summary
        summary = await self._portfolio.get_portfolio_summary()
        stats = self._state.get_daily_stats()
        health = self._watchdog.get_health()

        report = (
            f"DAILY SUMMARY ({date.today().isoformat()})\n"
            f"Trades: {stats.trades_taken} | "
            f"Won: {stats.trades_won} | Lost: {stats.trades_lost}\n"
            f"Realized P&L: ${stats.total_pnl:.2f}\n"
            f"Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}\n"
            f"Open Positions: {summary['open_positions']}\n"
            f"\n30-Day Stats:\n"
            f"  Win Rate: {metrics.win_rate:.0%} | "
            f"Sharpe: {metrics.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: ${metrics.max_drawdown_dollars:.0f} | "
            f"Profit Factor: {metrics.profit_factor:.1f}\n"
            f"  Model: {self._predictor.model_version}\n"
            f"  System: {health.status.value} | "
            f"Memory: {health.memory_mb:.0f}MB"
        )

        overrides = self._learning.get_current_adaptations()
        if overrides.get("disabled_strategies"):
            report += f"\n  Learning disabled: {', '.join(overrides['disabled_strategies'])}"
        if overrides.get("removed_symbols"):
            report += f"\n  Learning removed: {', '.join(overrides['removed_symbols'])}"

        # Add drift and Thompson stats
        if drift_report.samples > 0:
            report += f"\n  Drift: accuracy={drift_report.accuracy:.0%}, samples={drift_report.samples}"
        thompson_stats = self._thompson.get_stats()
        if thompson_stats:
            top = thompson_stats[0]
            report += f"\n  Top strategy (Thompson): {top['strategy']} ({top['win_rate']:.0%} over {top['observations']} trades)"

        # Counterfactual summary
        cf = self._counterfactual.get_analysis()
        if cf["evaluated"] > 0:
            report += f"\n  Filters: {cf['filter_accuracy']:.0%} accurate, {cf.get('missed_opportunities', 0)} missed"

        await self._send_notification(report)
        log.info("post_market_complete", summary=summary, metrics=vars(metrics))

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("orchestrator_shutting_down")
        health = self._watchdog.get_summary()
        await self._send_notification(f"Bot shutting down\n\n{health}")

    async def _check_hedging(self) -> None:
        """Check if portfolio delta needs hedging with SPY."""
        try:
            portfolio_greeks = self._risk_manager._portfolio_greeks
            account_value = await self._account.get_net_liquidation()
            spy_price = await self._market_data.get_current_price("SPY")

            if not spy_price or account_value <= 0:
                return

            recommendation = self._delta_hedger.check_hedge_needed(
                portfolio_greeks, account_value, spy_price
            )

            if recommendation:
                cost = self._delta_hedger.calculate_hedge_cost(recommendation, spy_price)
                msg = (
                    f"HEDGE: {recommendation.action} {recommendation.quantity}x SPY\n"
                    f"Reason: {recommendation.reason}\n"
                    f"Estimated cost: ${cost:,.0f}"
                )

                # Auto-execute hedge if enabled
                if self._settings.exit.auto_hedge:
                    try:
                        contract = ContractBuilder.stock("SPY")
                        qualified = await self._ibkr.qualify_contract(contract)
                        if qualified:
                            order = OrderBuilder.market(
                                action=recommendation.action,
                                quantity=recommendation.quantity,
                            )
                            await self._ibkr.place_order(qualified, order)
                            msg += "\nStatus: AUTO-EXECUTED"
                            log.info(
                                "hedge_auto_executed",
                                action=recommendation.action,
                                shares=recommendation.quantity,
                            )
                        else:
                            msg += "\nStatus: FAILED (contract qualification)"
                    except Exception as he:
                        msg += f"\nStatus: FAILED ({he})"
                        log.error("hedge_execution_failed", error=str(he))
                else:
                    msg += "\nStatus: MANUAL (auto_hedge disabled)"

                await self._send_notification(msg)
                log.info(
                    "hedge_recommendation",
                    action=recommendation.action,
                    shares=recommendation.quantity,
                    delta=recommendation.current_delta,
                    auto_executed=self._settings.exit.auto_hedge,
                )
        except Exception as e:
            log.warning("hedging_check_failed", error=str(e))

    # --- Helpers ---

    def _should_queue_signal(self, signal, hist) -> bool:
        """Check if a signal should be queued for better entry timing.

        Queue bullish signals when RSI is high (overbought) — wait for pullback.
        Queue bearish signals when RSI is low (oversold) — wait for bounce.
        """
        if signal.symbol in self._signal_queue:
            return False  # Already queued

        if hist is None or hist.empty or "Close" not in hist.columns:
            return False

        try:
            close = hist["Close"]
            # Calculate RSI-14
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])

            # For bullish trades, prefer entry on pullback (RSI < 60)
            if signal.direction == SignalDirection.BULLISH and current_rsi > 65:
                return True

            # For bearish trades, prefer entry on bounce (RSI > 40)
            if signal.direction == SignalDirection.BEARISH and current_rsi < 35:
                return True

        except Exception:
            pass

        return False

    async def _process_signal_queue(self) -> None:
        """Process queued signals — execute if timing improved or expire."""
        expired = []

        for symbol, entry in list(self._signal_queue.items()):
            entry["age"] += 1

            # Expire after 3 cycles (15 min at 5-min intervals)
            if entry["age"] > 3:
                expired.append(symbol)
                log.info("signal_expired", symbol=symbol, strategy=entry["signal"].strategy_name)
                continue

            # Check if timing has improved
            hist = await self._market_data.get_historical(symbol, days=60)
            if not self._should_queue_signal(entry["signal"], hist):
                # Timing improved — execute now
                log.info("queued_signal_executing", symbol=symbol, age=entry["age"])
                await self._try_execute(
                    entry["signal"],
                    entry["confidence"],
                    entry["sentiment"],
                    entry["regime"],
                )
                expired.append(symbol)

        for symbol in expired:
            self._signal_queue.pop(symbol, None)

    async def _check_thesis_valid(self, pos: PositionStatus) -> tuple[bool, str]:
        """Re-evaluate whether the original trade thesis still holds.

        Returns (invalidated: bool, reason: str).
        Checks: direction flip, regime shift, VIX spike.
        """
        try:
            context = self._state.get_trade_context(pos.trade_id)
            if not context:
                return False, ""

            entry_direction = context.get("entry_direction", "")
            entry_regime = context.get("entry_regime", "")
            entry_vix = context.get("entry_vix", 0)

            # 1. Re-run ML prediction on fresh data (no market_context here — lightweight check)
            hist = await self._market_data.get_historical(pos.symbol, days=60)
            if hist is not None and not hist.empty:
                prediction = self._predictor.predict(hist, symbol=symbol, market_context=None)
                if prediction and prediction.confidence > 0.70:
                    # Direction has flipped with high confidence
                    if entry_direction == "bullish" and prediction.direction == SignalDirection.BEARISH:
                        return True, f"direction_flipped_to_bearish (conf={prediction.confidence:.0%})"
                    if entry_direction == "bearish" and prediction.direction == SignalDirection.BULLISH:
                        return True, f"direction_flipped_to_bullish (conf={prediction.confidence:.0%})"

            # 2. Check regime shift
            vix = await self._market_data.get_vix()
            if hist is not None and not hist.empty:
                regime = self._regime_detector.analyze(hist, vix)
                if regime and regime.confidence > 0.70:
                    from ait.ml.regime import MarketRegime
                    current_regime = regime.regime.value
                    # If entered during trending and now high volatility, thesis is suspect
                    if entry_regime in ("trending_up", "trending_down") and current_regime == "high_volatility":
                        return True, f"regime_shift ({entry_regime} → {current_regime})"

            # 3. Check VIX spike (>30% increase since entry)
            if vix and entry_vix and entry_vix > 0:
                vix_change = (vix - entry_vix) / entry_vix
                if vix_change > 0.30:
                    return True, f"vix_spike ({entry_vix:.1f} → {vix:.1f}, +{vix_change:.0%})"

        except Exception as e:
            log.warning("thesis_check_failed", trade_id=pos.trade_id, error=str(e))

        return False, ""

    async def _sync_risk_manager_positions(self) -> None:
        """Sync risk manager with current open positions and live Greeks.

        Without this, position count limits, delta checks, duplicate
        detection, and correlation guards all operate on stale data.
        """
        try:
            open_trades = self._state.get_open_trades()
            filled_trades = [t for t in open_trades if t.status in (TradeStatus.FILLED, TradeStatus.PARTIAL)]

            # Try to get live Greeks from IBKR portfolio
            ibkr_greeks: dict[str, dict] = {}
            portfolio_items = self._ibkr.get_portfolio() or []
            for item in portfolio_items:
                key = item.contract.symbol
                ibkr_greeks[key] = {
                    "delta": getattr(item, "delta", 0) or 0,
                    "gamma": getattr(item, "gamma", 0) or 0,
                    "theta": getattr(item, "theta", 0) or 0,
                    "vega": getattr(item, "vega", 0) or 0,
                }

            positions_for_risk = []
            for trade in filled_trades:
                greeks = ibkr_greeks.get(trade.symbol, {})
                positions_for_risk.append({
                    "symbol": trade.symbol,
                    "strategy": trade.strategy,
                    "quantity": trade.quantity,
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                })

            self._risk_manager.update_positions(positions_for_risk)
            log.debug("risk_manager_synced", position_count=len(positions_for_risk))
        except Exception as e:
            log.warning("risk_manager_sync_failed", error=str(e))

    def _find_trade_record(self, trade_id: str) -> TradeRecord | None:
        """Find a trade record by ID (open trades only)."""
        open_trades = self._state.get_open_trades()
        for t in open_trades:
            if t.trade_id == trade_id:
                return t
        return None

    def _find_trade_by_id(self, trade_id: str) -> TradeRecord | None:
        """Find any trade record by ID (including closed)."""
        return self._state.get_trade_by_id(trade_id)

    async def _estimate_iv_rank(self, symbol: str) -> float:
        """Estimate IV rank (0-100) from recent volatility data."""
        hist = await self._market_data.get_historical(symbol, days=252)
        if hist is None or len(hist) < 60:
            return 50.0

        import numpy as np
        close = hist["Close"]
        log_returns = np.log(close / close.shift(1)).dropna()

        current_vol = float(log_returns.tail(20).std() * np.sqrt(252))
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
