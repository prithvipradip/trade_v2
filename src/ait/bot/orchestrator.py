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
from datetime import date, datetime, timedelta

from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.bot.state import DailyStats, StateManager, TradeRecord, TradeStatus
from ait.broker.account import AccountManager
from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.config.settings import Settings
from ait.data.earnings import EarningsCalendar
from ait.data.economic_calendar import EconomicCalendar
from ait.data.edgar_filings import EDGARMonitor
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
from ait.strategies.base import CREDIT_STRATEGIES, SignalDirection
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
        # Config-wired (audit item 3.3) — was CorrelationGuard() with code
        # defaults, untunable without a code edit.
        self._correlation_guard = CorrelationGuard(
            max_correlation=settings.risk.max_correlation,
            max_correlated_positions=settings.risk.max_correlated_positions,
        )
        self._risk_manager = RiskManager(
            settings.positions, settings.risk,
            self._account, self._circuit_breaker,
            self._pdt_guard, self._position_sizer,
            correlation_guard=self._correlation_guard,
            state=self._state,
        )
        self._delta_hedger = DeltaHedger()

        # ML
        self._predictor = DirectionPredictor(settings.ml)
        self._regime_detector = RegimeDetector()

        # Range predictor (Tier 1) — used for iron condor confidence
        from ait.ml.range_predictor import RangePredictor
        self._range_predictor = RangePredictor(
            threshold_pct=0.05,  # ±5%
            horizon_days=30,
        )
        self._range_predictor.load_models()  # silent if no model yet

        # Vol-magnitude predictor (Tier 1) — used for long straddle confidence
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        self._vol_mag_predictor = VolMagnitudePredictor(
            threshold_pct=0.07,  # ±7% (bigger than range threshold)
            horizon_days=30,
        )
        self._vol_mag_predictor.load_models()

        self._trainer = ModelTrainer(
            settings.ml, self._predictor, self._market_data, self._historical,
            range_predictor=self._range_predictor,
            vol_mag_predictor=self._vol_mag_predictor,
        )
        self._meta_labeler = MetaLabeler(
            min_probability=settings.meta_label.min_probability,
        ) if settings.meta_label.enabled else None

        # Sentiment
        self._sentiment = SentimentEngine(
            settings.sentiment, self._market_data,
            finnhub_api_key=settings.api_keys.finnhub_api_key,
        )

        # Calendars — needed by portfolio for event-driven exits
        self._earnings = EarningsCalendar()
        self._economic_cal = EconomicCalendar()

        # SEC EDGAR 8-K monitor — flatten positions on material events
        self._edgar = EDGARMonitor(tracked_symbols=settings.trading.universe)
        self._edgar_check_count = 0

        # Trading
        self._strategy_selector = StrategySelector(settings.options)
        self._executor = TradeExecutor(ibkr_client, self._state, self._circuit_breaker)
        self._portfolio = PortfolioManager(
            ibkr_client, self._market_data, self._state,
            self._circuit_breaker, self._pdt_guard,
            exit_config=settings.exit,
            earnings_calendar=self._earnings,
            economic_calendar=self._economic_cal,
        )
        # Wire alerts for silently-unprotected states (marks outage, PDT-
        # blocked stop) into the same Telegram channel as everything else.
        self._portfolio._notify_cb = self._send_notification

        # Scheduling
        self._scheduler = MarketScheduler()
        self._reconciler = PositionReconciler(ibkr_client, self._state)

        # Self-learning
        self._learning = LearningEngine(self._state)

        # Data quality & market intelligence
        self._data_quality = DataQualityValidator()
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
        from datetime import datetime as _dt
        self._started_at = _dt.now()  # for the post-restart settling guard
        log.info("orchestrator_starting", mode=self._settings.trading.mode)

        # Train models on startup if not yet trained (handles case where bot starts during market hours)
        await self._trainer.ensure_models_ready(self._settings.trading.universe)

        # Reconcile on EVERY startup, not just at pre/post-market. A restart
        # mid-day (common — Gateway drops, machine wake) otherwise leaves
        # orders placed before the death stuck in PENDING forever: the
        # in-memory fill tracker is gone, so check_fills never advances them,
        # and they're never monitored for exits. Startup reconcile recovers
        # them against IBKR's actual positions.
        try:
            recon = await self._reconciler.reconcile()
            log.info("startup_reconcile_done",
                     matched=recon.matched, promoted=recon.promoted,
                     stale_closed=recon.stale_local, new_from_ibkr=recon.new_from_ibkr)
            await self._alert_reconcile_anomalies(recon)
        except Exception as e:
            log.error("startup_reconcile_failed", error=str(e))

        await self._send_notification(f"BOT STARTED | Mode: {self._settings.trading.mode} | {len(self._settings.trading.universe)} symbols")

        # Read-only guardrail: verify the Gateway session can actually place
        # orders. Read-only sessions silently reject every order (Error 321)
        # while the bot logs 'trade_executed', so nothing fills and it's
        # invisible for a whole day. Alert the moment it's detected.
        self._readonly_alerted = False
        await self._check_trading_enabled()

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

    async def _alert_reconcile_anomalies(self, recon) -> None:
        """Telegram-alert reconcile findings that need a human (assignment)."""
        try:
            urgent = [d for d in (recon.discrepancies or [])
                      if "ASSIGNMENT" in d or "PARTIAL legs missing" in d]
            if urgent:
                await self._send_notification(
                    "RECONCILE NEEDS ATTENTION:\n" + "\n".join(urgent[:5])
                )
        except Exception as e:  # noqa: BLE001
            log.warning("reconcile_alert_failed", error=str(e))

    async def _process_completed_exits(self, completed_exits: list[dict]) -> None:
        """Book completed exits: daily stats, circuit breaker, Thompson, drift.

        Must be called from EVERY check_fills() call site — most exits fill
        during the 30s fast monitor, and discarding its return used to mean
        the breaker/learning never saw them.
        """
        for ex in completed_exits:
            realized_pnl = ex["realized_pnl"]
            trade_id = ex["trade_id"]

            stats = self._state.get_daily_stats()
            stats.total_pnl += realized_pnl
            if realized_pnl > 0:
                stats.trades_won += 1
            else:
                stats.trades_lost += 1
            # Feed ALL results to the breaker, not just losses — its daily
            # P&L must net wins against losses or it trips on any losing
            # streak within an otherwise green day.
            self._circuit_breaker.record_trade_result(realized_pnl)
            self._state.update_daily_stats(stats)

            # Drop the per-trade risk key so the aggregate cap's namespace
            # doesn't accumulate dead entries (28 stale keys found in the
            # 2026-07-07 forensic audit).
            try:
                self._state.delete_state(f"trade_maxloss_{trade_id}")
            except Exception:  # noqa: BLE001
                pass

            # Deep-audit SR-H1: the PDT guard was fully built but NOTHING
            # ever called record_day_trade — it always reported 0/3 used.
            # A same-ET-day open+close is a day trade; record it.
            try:
                _t = self._find_trade_by_id(trade_id)
                if _t and _t.entry_time:
                    from ait.utils.time import now_et as _now_et
                    _entry_d = datetime.fromisoformat(_t.entry_time).date()
                    if _entry_d == _now_et().date():
                        self._pdt_guard.record_day_trade(_t.symbol)
            except Exception as _e:  # noqa: BLE001
                log.debug("pdt_record_failed", error=str(_e))

            trade = self._find_trade_by_id(trade_id)
            if trade:
                self._thompson.record_outcome(
                    strategy=trade.strategy,
                    won=realized_pnl > 0,
                    pnl=realized_pnl,
                )
                # Deep-audit BC-L2: deriving direction from P&L sign is
                # meaningless for market-neutral strategies (a strangle win
                # is NOT "bullish") and poisons drift accuracy tracking.
                if trade.strategy not in ("iron_condor", "short_strangle"):
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

    async def _monitor_positions_fast(self) -> None:
        """Fast position monitor — checks stops/exits every 30 seconds.

        Runs between full scan cycles so positions aren't unprotected
        for the full 5-minute scan interval.
        """
        try:
            # Read-only guardrail: re-probe at most every 20 min so mid-session
            # read-only (Gateway daily restart relogging read-only) is caught
            # and alerted, not just at startup.
            now = datetime.now()
            last = getattr(self, "_last_readonly_check", None)
            if last is None or (now - last).total_seconds() >= 1200:
                self._last_readonly_check = now
                await self._check_trading_enabled()

            positions = await self._portfolio.check_positions()
            # A5: a successful portfolio read == IBKR is genuinely alive.
            try:
                self._watchdog.heartbeat("ibkr")
            except Exception:
                pass

            # A1: mark-to-market daily-loss brake (the realized-only breaker
            # is entry-gated and blind to unrealized gap-day bleeding).
            # mtm_day = realized_today + (unrealized_now - unrealized_at_SOD).
            try:
                from ait.utils.time import now_et as _net
                _unreal_now = sum((p.unrealized_pnl or 0.0) for p in positions)
                _sod_key = f"mtm_sod_{_net().date().isoformat()}"
                _sod_raw = self._state.get_state(_sod_key, "")
                if _sod_raw == "":
                    self._state.set_state(_sod_key, str(_unreal_now))
                    _sod = _unreal_now
                else:
                    _sod = float(_sod_raw)
                _realized_today = self._state.get_daily_stats().total_pnl
                _mtm_day = _realized_today + (_unreal_now - _sod)
                _nlv = await self._account.get_net_liquidation()
                if _nlv > 0 and self._circuit_breaker.check_daily_loss_mtm(_mtm_day, _nlv):
                    if not getattr(self, "_mtm_halt_alerted", False):
                        self._mtm_halt_alerted = True
                        await self._send_notification(
                            f"DAILY MTM LOSS HALT: day P&L ${_mtm_day:,.0f} "
                            f"(realized ${_realized_today:,.0f} + unrealized move) "
                            f"breached the daily-loss cap. New entries blocked; "
                            f"exits still active."
                        )
            except Exception as _e:  # noqa: BLE001
                log.debug("mtm_check_failed", error=str(_e))

            # Skip positions already in CLOSING state (exit order already placed)
            closing_ids = {
                pe.trade_id for pe in self._executor._pending_exit_orders.values()
            }
            exits_needed = [
                p for p in positions
                if p.should_exit and p.trade_id not in closing_ids
            ]

            for pos in exits_needed:
                log.info("fast_monitor_exit", symbol=pos.symbol, reason=pos.exit_reason)
                await self._execute_exit(pos)

            # Also check pending order fills — and BOOK the completed exits
            # (stats/breaker/Thompson); they used to be silently discarded.
            _entries, completed_exits = await self._executor.check_fills()
            await self._process_completed_exits(completed_exits)

            # SEC 8-K material-event check every ~5 min (every 10th fast cycle)
            self._edgar_check_count += 1
            if self._edgar_check_count >= 10:
                self._edgar_check_count = 0
                await self._check_material_events()
        except Exception as e:
            # Deep-audit BC-H1: this was log.debug — the 30s stop/TP engine
            # could silently no-op for HOURS with a green heartbeat. Surface
            # loudly and feed the watchdog error counter.
            log.warning("fast_monitor_error", error=str(e))
            try:
                self._watchdog.record_error("trading_loop", f"fast_monitor: {e}")
            except Exception:
                pass

    async def _check_material_events(self) -> None:
        """Check for new SEC 8-K filings; flatten any held positions on the symbol."""
        try:
            # Only check symbols we currently hold positions in
            open_trades = self._state.get_open_trades()
            held_symbols = list({t.symbol for t in open_trades if t.symbol})
            if not held_symbols:
                return

            new_filings = await self._edgar.check_for_material_events(held_symbols)
            for filing in new_filings:
                log.warning(
                    "material_event_flatten",
                    symbol=filing.symbol,
                    accession=filing.accession_number,
                )
                await self._send_notification(
                    f"⚠️ SEC 8-K filed for {filing.symbol} — flattening positions"
                )
                # Force-close all positions in that symbol
                positions = await self._portfolio.check_positions()
                for pos in positions:
                    if pos.symbol == filing.symbol:
                        await self._execute_exit(pos)
        except Exception as e:
            log.debug("material_event_check_failed", error=str(e))

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
        await self._process_completed_exits(completed_exits)

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
        # TEMPORARILY DISABLED — env var AIT_SKIP_MACRO_EVENTS=0 bypasses this
        # to gather more paper trade data. Re-enable once we have 30+ trades.
        import os
        if os.environ.get("AIT_SKIP_MACRO_EVENTS", "0") == "1":
            if self._economic_cal.should_skip_trading():
                log.info("economic_event_skip",
                         events=str(self._economic_cal.get_upcoming_events(days=2)))
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
        """Fetch VIX, SPY, and macro data for ML cross-asset features."""
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
        # Macro data (FRED: yield curve, DXY)
        try:
            from ait.data.macro import MacroDataFetcher
            if not hasattr(self, "_macro_fetcher"):
                self._macro_fetcher = MacroDataFetcher()
            macros = await self._macro_fetcher.fetch_all(lookback_days=365)
            if macros:
                context["macros"] = macros
        except Exception:
            pass
        return context

    async def _scan_symbol(self, symbol: str, vix: float | None, market_context: dict | None = None) -> None:
        """Analyze a single symbol for trading opportunities.

        Parallelizes data fetches: historical, sentiment, and IV rank
        are fetched concurrently with asyncio.gather.
        """
        adaptor = self._learning.adaptor

        # Parallel fetch: historical data, sentiment, IV rank
        # 2 years of history for robust features (iv_rank, vol percentiles, trend)
        hist_task = self._market_data.get_historical(symbol, days=504)
        sentiment_task = self._sentiment.get_sentiment(symbol)
        iv_rank_task = self._estimate_iv_rank(symbol)

        hist, sentiment, iv_rank = await asyncio.gather(
            hist_task, sentiment_task, iv_rank_task,
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

        # Incremental intraday fetch: only request bars not yet stored in SQLite.
        # On first run (no stored data) fetches a full 7-day seed window.
        # Extends retention to 2 years so MFDFA and walk-forward VLMC analysis
        # have the maximum available history.
        try:
            last_ts = self._historical.get_latest_intraday_timestamp(symbol)
            if last_ts is None:
                intraday = await self._market_data.get_intraday(symbol, interval="5m", days=7)
            else:
                intraday = await self._market_data.get_intraday_since(symbol, since=last_ts)
        except Exception as e:
            log.debug("intraday_fetch_error", symbol=symbol, error=str(e))
            intraday = None

        # Persist new bars; reload full 2-year window for fractal feature computation.
        if intraday is not None and not intraday.empty:
            self._historical.save_intraday(symbol, intraday)
            self._historical.cleanup_old_intraday(keep_days=730)
        intraday_full = self._historical.load_intraday(symbol, days=730)

        # Data quality check on historical data
        if "Close" in hist.columns:
            prices = hist["Close"].tolist()
            if not self._data_quality.validate_historical(symbol, prices):
                return

        # Build live-signals dict for ML model (sentiment + options flow)
        live_signals = {}
        if sentiment:
            live_signals["sentiment_composite"] = float(getattr(sentiment, "composite_score", 0))
            live_signals["sentiment_news"] = float(getattr(sentiment, "news_score", 0) or 0)
            live_signals["sentiment_finbert"] = float(getattr(sentiment, "finbert_score", 0) or 0)
            live_signals["fear_greed"] = float(getattr(sentiment, "fear_greed_score", 0) or 0)

        # Intraday fractal + VLMC features (Fix 2c): pass intraday_store directly to
        # predict() so the predictor computes VLMC features via the same
        # _merge_intraday_features() code path used during training. This ensures
        # session-tiering and feature alignment are consistent between training and live.
        # The old compute_intraday_features() → live_signals path is removed to eliminate
        # the training/inference VLMC feature mismatch identified in Gap B.
        # Sentinel fractal features (hurst_scale_spread, multifractal_width) needed for
        # the fractal penalty below are still accessible via features_df or live_signals.

        # ML prediction (with cross-asset context, live signals, and intraday_store)
        prediction = self._predictor.predict(
            hist, symbol=symbol,
            market_context=market_context,
            live_signals=live_signals,
            intraday_store=self._historical,
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

        # Use learning-adjusted confidence threshold.
        # In paper_trading_mode, bypass adaptor override so the live confidence
        # threshold matches the backtest's fixed value (Fix 6a / Gap Z8).
        paper_mode = self._settings.learning.paper_trading_mode
        min_confidence = (
            (adaptor.get_confidence_override() or self._settings.risk.min_confidence)
            if not paper_mode
            else self._settings.risk.min_confidence
        )
        # Market-neutral strategies (iron_condor, short_strangle) must NOT be
        # vetoed by low DIRECTIONAL confidence: low directional conviction =
        # range-bound = their ideal regime, and high directional confidence =
        # trending = their worst. The backtest already skips this gate for
        # neutral strategies (engine.py _neutral_only); live didn't — audit
        # 2026-07-07 item 1.2 (documented cause of entering the tariff crash).
        # On low directional confidence we now continue in neutral-only mode:
        # directional strategies are dropped downstream and neutral ones are
        # gated solely by the range model + RANGE_MIN_CONFIDENCE floor.
        neutral_only = prediction.confidence < min_confidence
        if neutral_only:
            log.debug(
                "low_confidence_neutral_only",
                symbol=symbol,
                confidence=prediction.confidence,
                threshold=min_confidence,
            )
            # Record counterfactual for the directional skip (we may still
            # trade a neutral strategy on this symbol via the range model)
            self._counterfactual.record_skip(
                symbol=symbol,
                strategy="unknown",
                direction=prediction.direction.value,
                confidence=prediction.confidence,
                entry_price=float(hist["Close"].iloc[-1]) if "Close" in hist.columns else 0,
                reject_reason="low_confidence_directional",
            )

        # A5: a successful history fetch == the market-data path is alive.
        try:
            self._watchdog.heartbeat("market_data")
        except Exception:
            pass

        # Market regime
        regime = self._regime_detector.analyze(hist, vix)

        # Adjust confidence with sentiment (already fetched in parallel)
        final_confidence = prediction.confidence
        if sentiment and hasattr(sentiment, "sources_available") and sentiment.sources_available > 0:
            sentiment_adj = sentiment.composite_score * self._sentiment.weight
            final_confidence = max(0, min(1, final_confidence + sentiment_adj))

        # Multi-timeframe analysis: boost/penalize confidence based on alignment
        # Use intraday_full (full SQLite history) not the incremental fetch — the
        # MTF analyser needs ≥20 bars and the incremental fetch is often just a few.
        mtf = self._mtf_analyzer.analyze(hist, intraday_full)
        final_confidence = max(0, min(1, final_confidence + mtf.confidence_boost))

        # Pre-compute daily features once — used for fractal penalty and meta-labeler below.
        from ait.ml.features import FeatureEngine
        features_df = FeatureEngine().compute(hist)

        # Fractal regime confidence penalty (Gap Z5): mirrors backtest engine logic.
        # If hurst_scale_spread or multifractal_width indicate chaotic fractal regime,
        # penalise confidence by hurst_regime_penalty so the live bot is as conservative
        # as the backtest during chaotic periods.
        # hurst_scale_spread and multifractal_width are daily fractal features computed
        # from the pre-ML features_df (they are NOT intraday-only features).
        hurst_scale_spread = float(features_df.iloc[-1].get("hurst_scale_spread", 0.0)) if not features_df.empty else 0.0
        multifractal_width = float(features_df.iloc[-1].get("multifractal_width", 0.0)) if not features_df.empty else 0.0
        bc = self._settings.backtest
        if (hurst_scale_spread > bc.hurst_regime_threshold
                or multifractal_width > bc.multifractal_max_width):
            penalty = bc.hurst_regime_penalty
            final_confidence = max(0.0, final_confidence - penalty)
            log.debug(
                "fractal_penalty_applied",
                symbol=symbol,
                hurst_scale_spread=f"{hurst_scale_spread:.3f}",
                multifractal_width=f"{multifractal_width:.3f}",
                penalty=penalty,
                final_confidence=f"{final_confidence:.3f}",
            )

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

        # Meta-label gate: ask secondary model if this signal is worth taking.
        # Bypassed in paper_trading_mode so backtest and paper P&L are comparable (Gap Z1 interim).
        if not paper_mode and self._meta_labeler is not None and self._meta_labeler.is_trained:
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
                # Hard gate: if strong flow disagrees with ML direction, reject entirely.
                # Bypassed in paper_trading_mode — backtest has no flow data (Gap Z7).
                if not paper_mode and flow.bias_strength > 0.7:
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

        # Calendar spreads need multiple expiry chains — generate once per symbol.
        # Best in LOW IV (cheap premium to buy that may rise).
        calendar_signals = self._strategy_selector.generate_calendar_signals(
            symbol=symbol,
            chains=filtered_chains,
            market_direction=direction,
            confidence=final_confidence,
            iv_rank=iv_rank,
        )

        # Event-driven long straddles — fire 0-1 days before macro events.
        # Profit from IV expansion + directional move when our short-vol
        # strategies would be wrong-sided.
        event_straddle_signals = self._strategy_selector.generate_event_straddle_signals(
            symbol=symbol,
            chains=filtered_chains,
            market_direction=direction,
            confidence=final_confidence,
            iv_rank=iv_rank,
            economic_cal=self._economic_cal,
        )

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

            # Inject calendar signals on the FIRST chain iteration only
            if calendar_signals:
                signals = list(signals) + calendar_signals
                calendar_signals = []  # consume so we don't add twice

            # Inject event-straddle signals on the FIRST chain iteration only
            if event_straddle_signals:
                signals = list(signals) + event_straddle_signals
                event_straddle_signals = []  # consume so we don't add twice

            # Filter out signals from disabled strategies
            signals = [
                s for s in signals
                if adaptor.is_strategy_enabled(s.strategy_name)
            ]

            # Boost iron-condor confidence with range prediction (Tier 1 model)
            # P(stays in ±5% over 30 days) is what iron condors actually need.
            # 0.65 chosen via parameter sweep — beat 0.55 across every metric:
            # Sharpe 1.36 vs 0.64, Sortino 2.20 vs 0.97, RAROC 604% vs 276%.
            RANGE_MIN_CONFIDENCE = self._settings.ml.range_min_confidence
            model_overridden: set[str] = set()  # strategies whose confidence came from a payoff-matched model
            if self._range_predictor and self._range_predictor.is_trained:
                range_pred = self._range_predictor.predict(
                    hist, symbol=symbol,
                    market_context=market_context,
                    live_signals=live_signals,
                    intraday_store=self._historical,
                )
                if range_pred:
                    log.info("range_prediction", symbol=symbol,
                             p_in_range=f"{range_pred.probability_in_range:.3f}",
                             threshold=range_pred.threshold_pct,
                             horizon_days=range_pred.horizon_days)
                    # Override confidence for iron condors with range probability
                    for s in signals:
                        if s.strategy_name in ("iron_condor", "short_strangle"):
                            s.confidence = range_pred.probability_in_range
                            model_overridden.add(s.strategy_name)

            # Vol-magnitude model (Tier 1) for long straddles — predicts P(big move)
            # Direct signal for "buy volatility" strategies, more accurate than
            # the previous "1 - range probability" hack.
            if self._vol_mag_predictor and self._vol_mag_predictor.is_trained:
                vm_pred = self._vol_mag_predictor.predict(
                    hist, symbol=symbol,
                    market_context=market_context,
                    live_signals=live_signals,
                )
                if vm_pred:
                    log.info("vol_mag_prediction", symbol=symbol,
                             p_big_move=f"{vm_pred.probability_big_move:.3f}",
                             threshold=vm_pred.threshold_pct,
                             horizon_days=vm_pred.horizon_days)
                    for s in signals:
                        if s.strategy_name == "long_straddle":
                            s.confidence = vm_pred.probability_big_move
                            model_overridden.add(s.strategy_name)

            # Model-confidence floor — applies whenever a payoff-matched model
            # overrode a signal's confidence, REGARDLESS of which models are
            # trained. (Was nested inside the vol-mag block, so an untrained
            # vol-mag model silently skipped the iron-condor floor — audit
            # 2026-07-07, quick-win under item 1.2.)
            signals = [
                s for s in signals
                if s.strategy_name not in model_overridden
                or s.confidence >= RANGE_MIN_CONFIDENCE
            ]

            # Neutral-only mode (directional confidence below threshold):
            #  - drop directional strategies — no directional basis to trade;
            #  - drop neutral strategies that were NOT range-model-overridden —
            #    without the range model there is no basis at all.
            if neutral_only:
                signals = [
                    s for s in signals
                    if s.strategy_name in ("iron_condor", "short_strangle")
                    and s.strategy_name in model_overridden
                ]
                if signals:
                    log.info("neutral_only_candidates", symbol=symbol,
                             strategies=[s.strategy_name for s in signals])

            # Re-rank signals using Thompson sampling (exploration/exploitation).
            # Bypassed in paper_trading_mode — backtest uses deterministic strategy priority (Gap Z2).
            if signals and not paper_mode:
                strategy_names = [s.strategy_name for s in signals]
                ranked_names = self._thompson.rank_strategies(strategy_names)
                # Build a name→signal map and reorder
                sig_map = {s.strategy_name: s for s in signals}
                signals = [sig_map[n] for n in ranked_names if n in sig_map]

            # Defined-risk bias: float capped-loss strategies (iron_condor,
            # spreads) ahead of undefined-risk ones (short_strangle), keeping
            # the learned/ranked order WITHIN each group. This is a stable
            # partition, so Thompson still explores among defined-risk
            # strategies; undefined-risk stays only as a last-resort fallback.
            # Serves: capital efficiency + bounded downside ("don't run out of
            # money") without starving volume.
            if signals:
                signals = ([s for s in signals if s.is_defined_risk]
                           + [s for s in signals if not s.is_defined_risk])

            # Try the best signal; if it's risk-REJECTED, fall through to the
            # next-ranked strategy rather than abandoning the symbol. The old
            # code attempted only signals[:1], so a single un-executable top
            # signal (e.g. CC/CSP whose assignment notional exceeds the 3% cap)
            # meant zero trades even when a viable iron_condor ranked just
            # behind it. Cap the fall-through so we never place more than one
            # trade per symbol per scan (we break on the first handled signal).
            for signal in signals[:4]:
                # Confidence divergence fix (audit 2026-07-07): when a
                # payoff-matched model (range / vol-mag) overrode this signal's
                # confidence, THAT number — not the directional final_confidence
                # — is what justifies the trade, so it must be what risk
                # validation, sizing, and trade_context see. Previously the
                # range probability never reached risk sizing, and in
                # neutral-only mode the low directional confidence would have
                # auto-failed the risk manager's own confidence gate.
                eff_conf = (
                    signal.confidence
                    if signal.strategy_name in model_overridden
                    else final_confidence
                )
                if self._should_queue_signal(signal, hist):
                    self._signal_queue[signal.symbol] = {
                        "signal": signal,
                        "confidence": eff_conf,
                        "sentiment": sentiment,
                        "regime": regime,
                        "age": 0,
                    }
                    log.info("signal_queued", symbol=signal.symbol,
                             strategy=signal.strategy_name, reason="entry_timing")
                    break
                handled = await self._try_execute(signal, eff_conf, sentiment, regime)
                if handled:
                    break  # executed or symbol-level block — done with this symbol
                # else: risk-rejected — try the next-ranked strategy

    def _get_trade_budget(self) -> int:
        """Return how many trades are allowed this scan based on time of day.

        Reserves trades across the session instead of firing all at open:
          - First hour (9:30-10:30):  2 trades max (high-conviction only)
          - Mid-day (10:30-14:00):    up to 2 more
          - Power hour (14:00-16:00): remaining budget
        This way the bot can act on opportunities throughout the day.
        """
        from ait.utils.time import now_et
        max_trades = self._settings.trading.max_daily_trades
        daily_stats = self._state.get_daily_stats(now_et().date())
        taken = daily_stats.trades_taken

        now = now_et()  # ET-pinned (A3): budget tiers were wall-clock local
        hour_min = now.hour + now.minute / 60.0

        if hour_min < 10.5:      # 9:30-10:30 — first hour
            budget = 2
        elif hour_min < 14.0:    # 10:30-2:00 — mid-day
            budget = 4
        else:                    # 2:00-4:00 — power hour, release all
            budget = max_trades

        return min(budget, max_trades) - taken

    async def _check_trading_enabled(self) -> bool:
        """Probe whether orders can actually be placed; alert loudly if not.

        Guardrail against the silent read-only lock. Called at startup and
        periodically. On read-only it fires a single Telegram alert (until it
        recovers) so the user knows immediately to restart Gateway — instead
        of finding out at end of day that nothing traded.
        """
        try:
            can_trade = await self._ibkr.verify_can_trade()
        except Exception:
            return True  # never let the probe itself break the loop
        if not can_trade:
            if not getattr(self, "_readonly_alerted", False):
                self._readonly_alerted = True
                await self._send_notification(
                    "READ-ONLY: IB Gateway is rejecting all orders (Error 321). "
                    "Nothing can trade or exit until you RESTART GATEWAY and log "
                    "in as IB API. The bot will keep checking and tell you when "
                    "it clears."
                )
                log.critical("trading_disabled_alerted")
        else:
            if getattr(self, "_readonly_alerted", False):
                self._readonly_alerted = False
                await self._send_notification("Trading ENABLED again — Gateway accepts orders. Resuming.")
                log.info("trading_reenabled")
        return can_trade

    async def _try_execute(self, signal, confidence: float, sentiment, regime) -> bool:
        """Validate and execute a signal.

        Returns True when the symbol is "handled" (executed, or blocked for a
        symbol-level reason like budget/pending/earnings) and the caller
        should stop. Returns False ONLY when the signal was risk-REJECTED, so
        the caller can fall through to the next-ranked strategy — otherwise a
        single un-executable top signal (e.g. a CC/CSP whose assignment
        notional exceeds the per-trade cap) abandons the symbol even though a
        viable iron_condor was sitting right behind it.
        """
        adaptor = self._learning.adaptor

        # INST-2/3 (institutional audit): manual + automatic ENTRY freeze.
        #  - data/HALT: the operator's kill switch that isn't `kill` — blocks
        #    all NEW entries while exits keep being managed. Delete to resume.
        #  - data/HALT_UNTRACKED: set automatically when reconcile finds a
        #    live option position with no local record (mid-placement crash);
        #    trading blind next to an unmanaged position is not acceptable.
        from pathlib import Path as _P
        for _flag, _why in ((_P("data/HALT"), "manual halt file present"),
                            (_P("data/HALT_UNTRACKED"), "untracked live position needs review")):
            if _flag.exists():
                log.warning("entries_halted", reason=_why, flag=str(_flag))
                return True  # symbol handled: no new entries while halted

        # Check trade budget (time-based pacing)
        remaining = self._get_trade_budget()
        if remaining <= 0:
            daily_stats = self._state.get_daily_stats(__import__('ait.utils.time', fromlist=['now_et']).now_et().date())
            log.info("trade_budget_exhausted",
                     trades_taken=daily_stats.trades_taken,
                     max=self._settings.trading.max_daily_trades,
                     budget_remaining=remaining)
            return True

        from datetime import datetime as dt

        # GUARD #5 — post-restart settling period. Don't trade in the first few
        # minutes after startup: the bot trades on its very first scan, and with
        # frequent restarts that means re-evaluating and firing immediately each
        # time. Let positions/state settle and reconcile first.
        SETTLE_SECONDS = 300  # 5 min
        started = getattr(self, "_started_at", None)
        if started is not None and (dt.now() - started).total_seconds() < SETTLE_SECONDS:
            log.info("settling_period_skip", symbol=signal.symbol,
                     seconds_since_start=int((dt.now() - started).total_seconds()))
            return True  # whole-bot pause — not a per-signal reject

        # First hour requires higher confidence (only take the best setups)
        from ait.utils.time import now_et
        _et = now_et()
        hour_min = _et.hour + _et.minute / 60.0
        # Deep-audit BC-M2: neutral strategies carry RANGE probability as
        # confidence (~0.65-0.75 scale) — requiring 0.85 structurally banned
        # iron condors for the whole first hour. Gate directional signals
        # only. (Also ET-pinned per BC-M3.)
        _neutral = signal.strategy_name in ("iron_condor", "short_strangle")
        if hour_min < 10.5 and not _neutral and confidence < 0.85:
            log.info("first_hour_confidence_gate",
                     symbol=signal.symbol, confidence=f"{confidence:.2f}",
                     required=0.85)
            return True  # same confidence for all signals — trying another won't help

        # GUARD #2 — per-symbol+strategy cooldown. Refuse re-entry on the same
        # symbol+strategy within the window regardless of fill status; a simple
        # DB-backed backstop that survives restarts and doesn't depend on the
        # (fragile) in-memory pending tracker.
        COOLDOWN_MINUTES = 120
        try:
            cutoff = dt.now() - timedelta(minutes=COOLDOWN_MINUTES)
            for rt in self._state.get_recent_trades(n=30):
                if rt.symbol == signal.symbol and rt.strategy == signal.strategy_name:
                    try:
                        entered = datetime.fromisoformat(rt.entry_time)
                    except (ValueError, TypeError):
                        continue
                    if entered >= cutoff:
                        log.info("cooldown_skip", symbol=signal.symbol,
                                 strategy=signal.strategy_name,
                                 minutes_ago=int((dt.now() - entered).total_seconds() / 60))
                        return True  # symbol+strategy recently traded
        except Exception as e:  # noqa: BLE001
            log.debug("cooldown_check_failed", error=str(e))

        # GUARD #3 — working IBKR orders. Catch in-flight orders the local DB
        # hasn't caught up on (e.g. placed but not yet recorded/filled).
        try:
            for t in (self._ibkr.get_open_orders() or []):
                if getattr(t.contract, "symbol", None) == signal.symbol:
                    log.info("working_order_skip", symbol=signal.symbol)
                    return True
        except Exception as e:  # noqa: BLE001
            log.debug("working_order_check_failed", error=str(e))

        # Check if this symbol already has a pending order (in-memory tracker)
        pending_symbols = set()
        for oid, pending in self._executor._pending_orders.items():
            if hasattr(pending, 'signal') and hasattr(pending.signal, 'symbol'):
                pending_symbols.add(pending.signal.symbol)
        if signal.symbol in pending_symbols:
            log.debug("symbol_has_pending_order", symbol=signal.symbol)
            return True  # symbol already busy

        # Check if position would hold through earnings (IV crush risk)
        if signal.expiry:
            from datetime import date as date_cls
            expiry_date = signal.expiry if isinstance(signal.expiry, date_cls) else date_cls.fromisoformat(str(signal.expiry))
            if self._earnings.would_hold_through_earnings(signal.symbol, date.today(), expiry_date):
                log.info("trade_blocked_earnings", symbol=signal.symbol, expiry=str(signal.expiry))
                return True  # symbol blocked through earnings

        # Build trade request for risk validation
        current_vix = await self._market_data.get_vix() or 0.0
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
            vix=current_vix,
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
            return False  # rejected — let the caller try the next-ranked strategy

        # Apply learning-based sizing multiplier.
        # GATE-BYPASS FIX (audit 2026-07-07 item 2.3): the multiplier used to
        # inflate size AFTER validate_trade, so every dollar-based gate
        # (buying power, 3% per-trade, 20% aggregate, delta) had checked a
        # smaller trade than the one executed. If the multiplier scales size
        # up, re-validate at the actual size; on rejection fall back to the
        # gate-approved size.
        strategy_mult = adaptor.get_strategy_multiplier(signal.strategy_name)
        adjusted_size = max(1, int(validation.position_size * strategy_mult))
        if adjusted_size > request.contracts:
            scale = adjusted_size / max(1, request.contracts)
            revalidation = await self._risk_manager.validate_trade(TradeRequest(
                symbol=request.symbol,
                strategy=request.strategy,
                direction=request.direction,
                contracts=adjusted_size,
                entry_price=request.entry_price,
                option=request.option,
                confidence=request.confidence,
                implied_vol=request.implied_vol,
                max_loss=(request.max_loss * scale) if request.max_loss else None,
                vix=request.vix,
            ))
            if not revalidation.approved:
                log.info("multiplier_size_rejected_using_base",
                         symbol=signal.symbol, strategy=signal.strategy_name,
                         wanted=adjusted_size, using=validation.position_size,
                         reason=revalidation.reason)
                adjusted_size = max(1, validation.position_size)

        # Apply adaptor exit overrides to the signal (Gap Z8 / Fix 6a).
        # Bypassed in paper_trading_mode so exit params match the backtest's fixed values.
        _paper_mode = self._settings.learning.paper_trading_mode
        if not _paper_mode:
            sl_override = adaptor.get_stop_loss_override()
            if sl_override is not None and signal.stop_loss is not None:
                signal.stop_loss = signal.entry_price * (1 - sl_override) if signal.entry_price else signal.stop_loss
            ts_override = adaptor.get_trailing_stop_override(signal.strategy_name)
            if ts_override is not None:
                signal.trailing_stop_pct = ts_override  # stored on signal for executor
            tp_override = adaptor.get_take_profit_override(signal.strategy_name)
            if tp_override is not None:
                signal.profit_target_pct = tp_override  # stored on signal for executor

        # Execute
        trade_id = await self._executor.execute_signal(signal, adjusted_size)

        if trade_id:
            # Store full entry context for thesis re-evaluation and learning
            regime_str = regime.regime.value if regime else ""
            self._state.set_state(f"trade_regime_{trade_id}", regime_str)
            # Persist max_loss so the aggregate capital-at-risk guard can sum
            # it across open positions (survives restarts via the state DB).
            self._state.set_state(
                f"trade_maxloss_{trade_id}",
                str(signal.max_loss if signal.is_defined_risk else 0.0),
            )
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

        if not trade_id:
            # Risk passed but the order couldn't be built/placed (e.g. combo
            # leg qualification failed — the long-standing reason iron_condor
            # silently never fills). Treat like a reject so the caller falls
            # through to the next-ranked strategy instead of abandoning the
            # symbol. Logged distinctly so the build failure is visible.
            log.info("execution_attempt_failed_fallthrough",
                     symbol=signal.symbol, strategy=signal.strategy_name)
            return False

        # Risk passed and the order was placed — the symbol is handled
        # for this scan whether or not the order ultimately fills.
        return True

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

                # Reverse the POSITION, not the market bias: trade.direction
                # stores the signal's view (CSP is "long"/bullish but we SOLD
                # the put). Credit strategies buy back; debit strategies sell.
                close_action = "BUY" if trade.strategy in CREDIT_STRATEGIES else "SELL"
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

                close_action = "BUY" if trade.strategy in CREDIT_STRATEGIES else "SELL"
                order = OrderBuilder.market(action=close_action, quantity=qty_to_close)
                exit_trade = await self._ibkr.place_order(qualified, order)

                # PHANTOM-P&L FIX (audit 2026-07-07 item 2.2): the old code
                # booked an ESTIMATED partial P&L (pos.unrealized_pnl pro-rata)
                # with the UNDERLYING's price recorded as the option fill —
                # before the order even filled. Book on the real fill only.
                fill_price = None
                if exit_trade is not None:
                    # Bounded at ~4s (audit R2): this await runs inside the
                    # trading cycle, and while it sleeps NO fast-monitor pass
                    # runs — every other position's stop/TP check is delayed.
                    # Market orders on liquid options fill sub-second; 4s is
                    # plenty, and an unfilled order is cancelled + retried
                    # next cycle rather than waited on.
                    for _ in range(8):
                        await asyncio.sleep(0.5)
                        st = exit_trade.orderStatus
                        if st.status == "Filled" and st.avgFillPrice:
                            fill_price = float(st.avgFillPrice)
                            break
                        if st.status in ("Cancelled", "Inactive", "ApiCancelled"):
                            break
                if fill_price is None:
                    # Not filled in time: cancel so a re-trigger next cycle
                    # can't double-sell, and book NOTHING.
                    try:
                        if exit_trade is not None:
                            self._ibkr.ib.cancelOrder(exit_trade.order)
                    except Exception:
                        pass
                    log.warning("partial_exit_unfilled_not_booked",
                                trade_id=pos.trade_id, symbol=pos.symbol)
                    return

                # Realized P&L from the actual option fill (0.65/contract
                # closing commission, matching executor convention).
                if trade.strategy in CREDIT_STRATEGIES:
                    partial_pnl = (trade.entry_price - fill_price) * qty_to_close * 100
                else:
                    partial_pnl = (fill_price - trade.entry_price) * qty_to_close * 100
                partial_pnl -= 0.65 * qty_to_close
            else:
                # Multi-leg partials are NOT supported: the old code fell
                # through here without placing ANY order, then recorded the
                # "partial exit" anyway — booking phantom realized P&L and
                # decrementing quantity while IBKR still held everything.
                # Refuse loudly; the position stays managed as a whole.
                log.warning(
                    "partial_exit_unsupported_multi_leg",
                    trade_id=pos.trade_id,
                    strategy=trade.strategy,
                    contract_type=trade.contract_type,
                )
                return

            # Record partial exit (real fill price + fill-derived P&L; the
            # estimate-based booking was removed — item 2.2)
            pnl_level = pos.pnl_pct
            for level in self._settings.exit.partial_exit_levels:
                if pos.pnl_pct >= level["pnl_pct"]:
                    pnl_level = level["pnl_pct"]

            self._state.record_partial_exit(
                trade_id=pos.trade_id,
                quantity=qty_to_close,
                price=fill_price,
                pnl=partial_pnl,
                pnl_level=pnl_level,
            )

            # Update remaining quantity
            new_qty = trade.quantity - qty_to_close
            self._state.update_trade_quantity(pos.trade_id, new_qty)

            # Update daily stats with realized portion. Deep-audit BC-H3:
            # do NOT bump trades_won here — the final close books the win/
            # loss; counting the partial too made trades_won exceed
            # trades_taken. DO feed the circuit breaker's daily P&L, which
            # previously never saw partial-exit money.
            stats = self._state.get_daily_stats()
            stats.total_pnl += partial_pnl
            self._state.update_daily_stats(stats)
            try:
                self._circuit_breaker.record_partial_pnl(partial_pnl)
            except Exception:
                pass

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

        # The legs above are ALREADY reversed, so the combo order must be
        # BUY — IBKR executes a BUY combo's legs exactly as defined, while a
        # SELL combo flips every leg. The old "SELL if mostly-sell-legs"
        # logic double-reversed all-BUY entries (long/event straddles): the
        # "exit" bought the straddle AGAIN instead of closing it. Entry
        # combos use the same always-BUY convention (executor.py).
        combo_action = "BUY"

        # Try to get mid-price from IBKR for a limit order
        limit_price = None
        try:
            qualified_combo = await self._ibkr.qualify_contract(combo)
            if qualified_combo:
                self._ibkr.ib.reqMktData(qualified_combo, "", False, False)
                ticker = None
                try:
                    await asyncio.sleep(0.5)
                    ticker = self._ibkr.ib.ticker(qualified_combo)
                finally:
                    # ALWAYS cancel the streaming subscription. A leaked
                    # snapshot=False sub streams (and, if unentitled, errors
                    # 10091) forever; repeated exit-pricing attempts pile these
                    # up into a market-data flood that crashes the ib_insync
                    # message thread (native access violation). Cancelling here
                    # keeps req/cancel paired 1:1.
                    try:
                        self._ibkr.ib.cancelMktData(qualified_combo)
                    except Exception:
                        pass
                if ticker:
                    import math
                    # Combo quotes are SIGNED: closing a debit position
                    # (reversed legs = net sell) quotes NEGATIVE — we receive
                    # credit. Only 0/NaN means "no quote".
                    bid = ticker.bid if not math.isnan(ticker.bid) and ticker.bid != 0 else None
                    ask = ticker.ask if not math.isnan(ticker.ask) and ticker.ask != 0 else None
                    # An EXIT must FILL — a take-profit/stop that sits unfilled
                    # lets a winner reverse or a loss deepen. Price at the
                    # MARKETABLE side and cross the spread by a buffer, instead
                    # of the mid (which never crosses, so the order timed out
                    # and re-placed every 30s forever — the IWM strangle sat
                    # on a +54% gain it couldn't take). BUY pays up to the ask,
                    # SELL accepts down to the bid.
                    EXIT_CROSS = self._settings.exit.exit_cross_amount
                    if combo_action == "BUY":
                        if ask is not None:
                            limit_price = round(ask + EXIT_CROSS, 2)
                        elif bid is not None:
                            limit_price = round(bid + EXIT_CROSS, 2)
                    else:  # SELL
                        if bid is not None:
                            limit_price = round(bid - EXIT_CROSS, 2)
                        elif ask is not None:
                            limit_price = round(ask - EXIT_CROSS, 2)
        except Exception as e:
            log.warning("combo_mid_price_failed", error=str(e))

        # Use market order as fallback if we couldn't get a limit price.
        # Negative limits are valid (closing a debit position nets a credit).
        if limit_price is not None and limit_price != 0:
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
        recon = await self._reconciler.reconcile()
        await self._alert_reconcile_anomalies(recon)

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
            report += f"\n  Top strategy (Thompson): {top['strategy']} ({top['win_rate']:.0%} over {top['observations']:.0f} trades)"

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
                # NOTE: was `symbol=symbol` (undefined) — a swallowed NameError
                # that killed this entire thesis check (incl. regime/VIX branches
                # below) on every call since it was written. Audit 2026-07-07 item 1.1.
                prediction = self._predictor.predict(hist, symbol=pos.symbol, market_context=None)
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
            # First clear long-dead PENDING orphans so they don't keep
            # blocking new trades via the pending-aware dedup/correlation
            # guards (orders stuck pending never resolve on their own once the
            # in-memory fill tracker is lost). Runs every cycle, age-gated.
            swept = self._reconciler._sweep_stale_pending()
            if swept:
                log.info("stale_pending_swept", count=swept)

            open_trades = self._state.get_open_trades()
            # Include PENDING/CLOSING, not just FILLED: orders sitting pending
            # (fill not yet confirmed on delayed data) must still count toward
            # the duplicate, max-position, and aggregate-risk guards — otherwise
            # the bot re-places the same setup every cycle until the daily cap
            # (observed 2026-06-23: 2x SPY + 2x IWM iron_condor duplicates).
            filled_trades = [
                t for t in open_trades
                if t.status in (TradeStatus.FILLED, TradeStatus.PARTIAL,
                                TradeStatus.PENDING, TradeStatus.CLOSING)
            ]

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
                ml = self._state.get_state(f"trade_maxloss_{trade.trade_id}", "")
                positions_for_risk.append({
                    "symbol": trade.symbol,
                    "strategy": trade.strategy,
                    "quantity": trade.quantity,
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                    "max_loss": float(ml) if ml else 0.0,
                    # Concentration-gate repair (audit item 3.3): the 20%
                    # symbol-concentration gate reads market_value, which was
                    # never populated → exposure always 0 → gate dead. Entry
                    # notional is a reasonable standing proxy.
                    "market_value": abs(trade.entry_price) * trade.quantity * 100,
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
        """Send notification via configured channel.

        A11 (deep-audit OPS-M3): the send used to be awaited INLINE in the
        trading loop -- a slow/hung Telegram POST delayed fill-checks and
        exits by up to its 10s timeout, and a failed send was dropped with
        one log line. Now fire-and-forget with 3 attempts + backoff; the
        loop never waits on the network.
        """
        if not self._notify:
            return

        async def _send_with_retry(msg: str) -> None:
            for attempt in range(3):
                try:
                    await self._notify(msg)
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("notification_attempt_failed",
                                attempt=attempt + 1, error=str(e)[:200])
                    await asyncio.sleep(2 * (attempt + 1))
            log.error("notification_dropped_after_retries", preview=msg[:80])

        try:
            asyncio.get_running_loop().create_task(_send_with_retry(message))
        except RuntimeError:  # no loop (sync/test context) -- best-effort inline
            try:
                await self._notify(message)
            except Exception as e:  # noqa: BLE001
                log.warning("notification_failed", error=str(e)[:200])
