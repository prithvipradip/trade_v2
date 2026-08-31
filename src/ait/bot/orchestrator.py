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
from ait.config.runtime_env import capital_base, capital_base_source, contract_flag, contract_float
from ait.config.settings import Settings
from ait.data.earnings import EarningsCalendar
from ait.data.economic_calendar import EconomicCalendar
from ait.data.edgar_filings import EDGARMonitor
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.data.options_chain import OptionsChainService
from ait.data.multi_timeframe import MultiTimeframeAnalyzer
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
from ait.risk.manager import RiskManager, TradeRequest
from ait.risk.pdt_guard import PDTGuard
from ait.risk.position_sizer import PositionSizer
from ait.risk.capital_tiers import CapitalTierManager
from ait.learning.counterfactual import CounterfactualTracker
from ait.strategies.base import CREDIT_STRATEGIES, SignalDirection
from ait.strategies.selector import StrategySelector
from ait.strategies.thompson import ThompsonSampler
from ait.utils.logging import get_logger
from ait.utils.time import next_market_open

log = get_logger("bot.orchestrator")

# Strategies whose ENTRY confidence must come from the payoff-matched range
# model rather than the directional one — P(price stays inside the wings) is
# what a short-premium neutral structure actually gets paid for.
# fail-direction-02 (R22): declared once here so the fail-closed drop below
# cannot drift from the membership it protects. (The two pre-existing literal
# copies at the override and the neutral-only filter are set-membership-6, a
# separate register entry, and are left untouched by this wave.)
RANGE_GATED_STRATEGIES: frozenset[str] = frozenset({"iron_condor", "short_strangle"})

def _cooldown_now() -> datetime:
    """Reference instant for the post-stop cooldown, naive-LOCAL.

    numeric-pairs-02 (R22): kept naive-local on purpose — trades.exit_time is
    written with datetime.now() (state.py) and the host runs ET, so the DB
    rows and this clock share one convention. Extracted as a module function
    so tests can pin it and stay deterministic on any weekday.
    """
    return datetime.now()


# Cutoff depends only on the reference DATE, and each lookup costs an NYSE
# calendar query; memoise so a scan over the universe pays it once a day.
_COOLDOWN_CUTOFF_CACHE: dict[date, datetime] = {}


def _post_stop_cooldown_cutoff(now: datetime | None = None) -> datetime:
    """09:30 (naive-local == ET on this host) start of the PREVIOUS session.

    numeric-pairs-02 (R22): the R12-B4 spec (portfolio.py ~line 495) is a
    duration of ONE TRADING DAY — "exit_time's next trading session must have
    STARTED before re-entry" — with weekends/holidays resolved by the market
    calendar. The implementation was a flat `now - exit_time < 30h` wall-clock
    window, which agrees mid-week and collapses across a weekend: a Friday
    10:00 touch stop expired Saturday 16:00, so the bot re-entered that symbol
    from Monday's open into the same move R12-B4 exists to sit out (the 08-24
    QQQ re-entry is the live instance of the family).

    The rule is uniform: block while exit_time >= 09:30 on the latest trading
    day STRICTLY BEFORE the reference date. On a Monday that cutoff is
    Friday 09:30 (a Friday stop blocks all of Monday; a Thursday stop does
    not), mid-week it is yesterday 09:30, and after a market holiday the
    calendar walks back to the last real session on its own.
    """
    from ait.utils.time import MARKET_OPEN, is_trading_day

    now = now or _cooldown_now()
    ref = now.date()
    cached = _COOLDOWN_CUTOFF_CACHE.get(ref)
    if cached is not None:
        return cached

    prev = ref - timedelta(days=1)
    for _ in range(14):  # 14 calendar days covers any NYSE holiday cluster
        if is_trading_day(prev):
            break
        prev -= timedelta(days=1)
    cutoff = datetime.combine(prev, MARKET_OPEN)

    if len(_COOLDOWN_CUTOFF_CACHE) > 8:
        _COOLDOWN_CUTOFF_CACHE.clear()
    _COOLDOWN_CUTOFF_CACHE[ref] = cutoff
    return cutoff


RESTRICTED_LIST_PATH = "data/RESTRICTED.txt"
# fail-direction-11: sentinel returned when the ban file EXISTS but cannot be
# parsed. It is deliberately NOT a set — a caller that forgets to check it
# raises instead of silently reading "no bans".
RESTRICTED_UNREADABLE = None


def read_restricted_symbols(path: str | None = None) -> set[str] | None:
    """Read the operator's hard-ban list (data/RESTRICTED.txt, one symbol/line).

    fail-direction-11 (blindspot_composition_hunt_20260825): this was an
    inline `read_text()` inside a bare `except: pass`. A UTF-8 BOM (Notepad,
    `Set-Content`) parsed to {'\ufeffSPY'} and a UTF-16 file (PowerShell's
    default `>` / Out-File) to NUL-riddled garbage — no exception raised, no
    log written, and the banned symbol traded on regardless.

    Returns:
        set[str]  — the banned symbols (upper-cased); empty set when the file
                    is ABSENT, which means "no restrictions" exactly as before;
        None      — RESTRICTED_UNREADABLE: the file exists but could not be
                    parsed. The caller MUST refuse new entries until it reads
                    cleanly: an operator who dropped a ban file gets
                    protection, not silence.
    """
    from pathlib import Path as _Path

    p = _Path(path or RESTRICTED_LIST_PATH)
    try:
        if not p.exists():
            return set()
    except OSError as e:
        log.error("restricted_list_unreadable_fail_closed",
                  path=str(p), error=str(e), stage="exists")
        return RESTRICTED_UNREADABLE

    text = None
    try:
        # utf-8-sig strips the BOM that made 'SPY' read as '\ufeffSPY'.
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # PowerShell 5.1's `echo SPY > data\RESTRICTED.txt` writes UTF-16LE;
        # a real ban file in the operator's default encoding is readable, not
        # a failure. Anything else still fails closed below.
        try:
            text = p.read_text(encoding="utf-16")
            log.warning("restricted_list_decoded_utf16", path=str(p))
        except Exception as e:  # noqa: BLE001
            log.error("restricted_list_unreadable_fail_closed",
                      path=str(p), error=str(e), stage="decode")
            return RESTRICTED_UNREADABLE
    except Exception as e:  # noqa: BLE001 — OSError, AV/permission lock, ...
        log.error("restricted_list_unreadable_fail_closed",
                  path=str(p), error=str(e), stage="read")
        return RESTRICTED_UNREADABLE

    if "\x00" in text:
        log.error("restricted_list_unreadable_fail_closed",
                  path=str(p), error="NUL bytes in decoded text",
                  stage="decode")
        return RESTRICTED_UNREADABLE

    banned = {ln.strip().upper() for ln in text.splitlines() if ln.strip()}
    log.info("restricted_list_loaded", path=str(p), symbols=sorted(banned))
    return banned


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
        # R16: hand the breaker the state handle directly so its consecutive-
        # loss count and any active trip SURVIVE a restart. Pre-fix, the
        # keeper's 90s relaunch reset the loss streak to zero — the protection
        # against a losing streak was erased by the very crash-loop most
        # likely to accompany one.
        self._circuit_breaker = CircuitBreaker(settings.risk, state=self._state)
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

        # ML
        self._predictor = DirectionPredictor(settings.ml)
        self._regime_detector = RegimeDetector()

        # Range predictor (Tier 1) — used for iron condor confidence.
        # units-scale-05 (R22): these were bare literals (0.05, 30) — the LIVE
        # half of a must-agree pair whose RESEARCH half (walkforward's
        # _range_label_horizon) had already moved to the reachable trade
        # horizon (dte_range[0] - EXPIRY_APPROACHING_DTE = 9d). The live 0.65
        # floor is justified by a sweep of the research pipeline, so a 30-day
        # containment probability was being judged against a floor calibrated
        # on the strictly easier 9-day question — vetoing IC entries the
        # validated gate would take. Both sides now read one authority.
        from ait.ml.range_predictor import RangePredictor
        from ait.ml.range_spec import live_range_spec
        _range_threshold, _range_horizon = live_range_spec(settings)
        self._range_predictor = RangePredictor(
            threshold_pct=_range_threshold,
            horizon_days=_range_horizon,
        )
        self._range_predictor.load_models()  # silent if no model yet
        self._check_range_model_spec(_range_threshold, _range_horizon)

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

        # Calendars — needed by portfolio for event-driven exits
        self._earnings = EarningsCalendar()
        self._economic_cal = EconomicCalendar()

        # SEC EDGAR 8-K monitor — flatten positions on material events
        self._edgar = EDGARMonitor(tracked_symbols=settings.trading.universe)
        self._edgar_check_count = 0

        # Trading
        self._strategy_selector = StrategySelector(settings.options)
        # R20 (register): hand the executor the loaded settings so its
        # spread-reject gate reads options.max_bid_ask_spread_pct instead of
        # its 0.15 fallback — the last dormant piece of the R19 config
        # binding. Dormant today (IC-only), armed the day a single-leg
        # strategy is re-enabled.
        self._executor = TradeExecutor(ibkr_client, self._state,
                                       self._circuit_breaker, settings=settings)
        self._portfolio = PortfolioManager(
            ibkr_client, self._market_data, self._state,
            self._circuit_breaker, self._pdt_guard,
            exit_config=settings.exit,
            earnings_calendar=self._earnings,
            economic_calendar=self._economic_cal,
        )
        # GOV-5 (governance audit): live-profile assertion — the liquidity
        # gates were deliberately relaxed for paper/delayed data
        # (spread 40%, min volume 0). Shipping those to live guts the
        # illiquid-chain protection exactly when a bad fill costs money.
        # Refuse to start in live mode with paper-relaxed gates.
        import os as _os_g5
        if str(getattr(settings.trading, "mode", "paper")).lower() != "paper":
            _spread_env = float(_os_g5.environ.get("AIT_LIQ_MAX_SPREAD", "0.15"))
            _vol_env = int(_os_g5.environ.get("AIT_LIQ_MIN_VOLUME", "10"))
            _spread_cfg = float(getattr(settings.options, "max_bid_ask_spread_pct", 0.15) or 0.15)
            if _spread_env > 0.15 or _vol_env < 1 or _spread_cfg > 0.15:
                raise RuntimeError(
                    "LIVE-PROFILE ASSERTION FAILED: liquidity gates are still "
                    f"paper-relaxed (AIT_LIQ_MAX_SPREAD={_spread_env}, "
                    f"AIT_LIQ_MIN_VOLUME={_vol_env}, "
                    f"config max_bid_ask_spread_pct={_spread_cfg}). Tighten "
                    "them before live (see docs/RUNBOOK.md go-live checklist)."
                )

        # GOV-3 (governance audit): defined-risk-only keyed to TRADING MODE,
        # not just an env default. In any non-paper mode the undefined-risk
        # allowance is force-stripped — the executor then refuses strangles
        # regardless of what the environment claims.
        import os as _os_g3
        if str(getattr(settings.trading, "mode", "paper")).lower() != "paper":
            _os_g3.environ.pop("AIT_ALLOW_UNDEFINED_RISK", None)
            log.info("undefined_risk_disabled_live_mode")

        # Wire alerts for silently-unprotected states (marks outage, PDT-
        # blocked stop) into the same Telegram channel as everything else.
        self._portfolio._notify_cb = self._send_notification
        # R17: stale account data used to only ever log — wire it to the same
        # notify channel and the existing circuit-breaker halt lever.
        self._account._notify_cb = self._send_notification
        self._account._circuit_breaker = self._circuit_breaker

        # Scheduling
        self._scheduler = MarketScheduler()
        self._reconciler = PositionReconciler(ibkr_client, self._state)

        # Self-learning
        self._learning = LearningEngine(
            self._state,
            # R7: in paper mode the learning layer observes but never steers
            apply_adaptations=not self._settings.learning.paper_trading_mode,
        )

        # Data quality & market intelligence
        self._data_quality = DataQualityValidator()
        self._mtf_analyzer = MultiTimeframeAnalyzer()

        # Health monitoring
        self._watchdog = Watchdog()
        self._watchdog.register_component("trading_loop")
        self._watchdog.register_component("ibkr")
        # R8 (incident 2026-07-10): the watchdog's alert path existed but was
        # NEVER wired — 661 component-down states paged nobody while stops
        # were dead for a full session. Route to Telegram, 1/hour cooldown.
        async def _wd_alert(msg: str) -> None:
            # R16: cooldown keyed PER COMPONENT — one global timestamp meant a
            # NEW, different critical arriving inside another component's hour
            # was silently swallowed. Key = first token of the message.
            _component = (msg.split() or ["unknown"])[0]
            if self._alert_gate(f"watchdog:{_component}", interval_s=3600):
                await self._send_notification(f"WATCHDOG {msg}")
        self._watchdog.set_alert_callback(_wd_alert)

        # R11: event-driven fill detection (fills were detected ~33s late on
        # the polling loop). The executor calls back so event-detected exits
        # get booked identically to polled ones.
        try:
            # type-ignore justified: deliberate duck-typed slot. TradeExecutor
            # never declares _completed_exits_cb — it reads it back with
            # getattr(self, "_completed_exits_cb", None) (executor.py:700), so
            # mypy is correct that the attribute is undeclared and wrong that
            # this is a bug. Declaring it on TradeExecutor would be the real
            # fix; that is a src change outside this CI-hardening scope.
            self._executor._completed_exits_cb = (  # type: ignore[attr-defined]
                self._process_completed_exits
            )
            self._executor.attach_fill_events()
        except Exception as _e:  # noqa: BLE001
            log.warning("fill_events_wiring_failed", error=str(_e))
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

    def _check_range_model_spec(self, threshold: float, horizon: int) -> bool:
        """units-scale-05: MODEL-ARTIFACT HONESTY at startup.

        A loaded range model keeps the spec it was TRAINED at — its
        probabilities only answer that question, and silently re-labelling
        them with the live spec would be a lie (RangePredictor.load_models is
        deliberately built that way). But a stale artifact must not be
        invisible either: a model trained at 30d/±5% gating ~9-day trades is
        the exact divergence units-scale-05 registers, and the operator has to
        be able to see it in the startup log.

        Returns True when a mismatch was found and logged. Nothing is deleted
        or retrained here — ModelTrainer.needs_training already forces a
        rebuild at the designed spec via RangePredictor.spec_mismatch.
        """
        rp = getattr(self, "_range_predictor", None)
        if rp is None or not getattr(rp, "is_trained", False):
            return False
        trained_threshold = getattr(rp, "_threshold", None)
        trained_horizon = getattr(rp, "_horizon", None)
        if (trained_threshold is None or trained_horizon is None
                or (abs(float(trained_threshold) - float(threshold)) <= 1e-9
                    and int(trained_horizon) == int(horizon))):
            return False
        log.warning(
            "range_model_spec_mismatch",
            live_threshold=threshold,
            live_horizon_days=horizon,
            model_threshold=trained_threshold,
            model_horizon_days=trained_horizon,
            model_version=getattr(rp, "model_version", ""),
            trained_at=getattr(rp, "trained_at", None),
            note="loaded range model answers a DIFFERENT question than the "
                 "live spec; it keeps serving its trained spec until the "
                 "next retrain rebuilds it (see ait.ml.range_spec)",
        )
        return True

    def set_notification_callback(self, callback) -> None:
        """Set async callback for sending notifications."""
        self._notify = callback
        # R8: do NOT overwrite the cooldown-wrapped watchdog callback wired
        # in __init__ — it routes through _send_notification (which uses
        # self._notify) with a 1/hour cooldown; the raw callback would page
        # on every check pass of a flapping component.

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

            # R12-A (F4.1): re-adopt working exit orders the reconciler found
            # for kept-CLOSING trades — without a tracker, a restart-surviving
            # exit order is invisible and a duplicate close (position
            # REVERSAL) follows.
            for _tid, _oid in getattr(recon, "closing_exit_orders", {}).items():
                try:
                    self._executor.adopt_exit_order(
                        _oid, _tid, reason="reconcile_readopt")
                except Exception as _e:  # noqa: BLE001
                    log.warning("exit_order_adopt_failed",
                                trade_id=_tid, error=str(_e))
            log.info("startup_reconcile_done",
                     matched=recon.matched, promoted=recon.promoted,
                     stale_closed=recon.stale_local, new_from_ibkr=recon.new_from_ibkr)
            await self._alert_reconcile_anomalies(recon)
        except Exception as e:
            log.error("startup_reconcile_failed", error=str(e))

        await self._send_notification(f"BOT STARTED | Mode: {self._settings.trading.mode} | {len(self._settings.trading.universe)} symbols")

        # R17: capital_base() (drawdown%/return% denominator across the
        # dashboard/analytics) silently falls back to a hardcoded default if
        # AIT_CAPITAL_BASE is unset and no live NLV has been cached yet — a
        # silently-wrong-denominator risk if that happens at real go-live.
        # One-time startup check, not per-render; paper mode never pages.
        if (self._settings.trading.mode == "live"
                and capital_base_source() == "default"):
            log.critical("capital_base_unset_in_live_mode", value=capital_base())
            await self._send_notification(
                f"WARNING: live-mode capital base is the HARDCODED DEFAULT "
                f"(${capital_base():,.0f}) — set AIT_CAPITAL_BASE or every "
                f"risk %/drawdown reading is wrong."
            )

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
        # R16: the MTM-halt page latch is set-once-per-PROCESS, so after the
        # first halt (or, pre-fix, after a false one) a genuine second-day
        # halt paged nobody. Clear it with the other daily counters.
        self._mtm_halt_alerted = False

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

    async def _drain_exit_outbox(self) -> None:
        """W1 (R23 breaker-bypass family): book closes nobody booked.

        Pre-W1, every close the RECONCILER discovered (a stop that filled
        while the bot was down, a manual TWS flatten, an ITM expiry) landed
        in trades.realized_pnl but NEVER reached the circuit breaker, daily
        stats, PDT counter, or Thompson — the daily-loss halt and 3-loss
        pause undercounted exactly the chaotic-day losses they exist for.
        Same for a booking lost to a crash between close_trade's commit and
        the fill callback. close_trade now enqueues transactionally; this
        drain claims and books the orphans. The 120s grace keeps it from
        racing an in-flight executor callback (which claims within seconds).
        """
        # W1 (concurrency-1): page the fill-after-close flag the executor sets
        # when a fill's CAS is refused — a real position may be unmanaged.
        try:
            _fac = self._state.get_state("alert_fill_after_close", "")
            if _fac:
                self._state.delete_state("alert_fill_after_close")
                await self._send_notification(
                    f"CRITICAL: fill landed on a TERMINAL trade row ({_fac}). "
                    "A REAL position may be live at the broker with no "
                    "managing row — check TWS and the untracked-option freeze.")
        except Exception:  # noqa: BLE001
            pass

        try:
            pending = self._state.pending_exit_bookings(older_than_seconds=120.0)
        except Exception as e:  # noqa: BLE001
            log.warning("exit_outbox_read_failed", error=str(e))
            return
        if not pending:
            return
        from ait.utils.time import now_et as _now_et
        today = _now_et().date()
        for row in pending:
            trade_id = row["trade_id"]
            try:
                if not self._state.claim_exit_booking(trade_id):
                    continue  # lost the race to a concurrent booker — correct
                pnl = float(row["realized_pnl"] or 0.0)
                try:
                    exit_d = datetime.fromisoformat(row["exit_time"]).date()
                except Exception:  # noqa: BLE001
                    exit_d = None
                if exit_d != today:
                    # Stats/breaker are DAILY quantities — booking a prior-day
                    # close into today's halt math errs the other direction.
                    # Claimed (never resurfaces), logged for the human ledger.
                    log.warning("stale_exit_booking_skipped", trade_id=trade_id,
                                exit_time=row["exit_time"], pnl=pnl)
                    continue
                stats = self._state.get_daily_stats()
                stats.total_pnl += pnl
                if pnl > 0:
                    stats.trades_won += 1
                else:
                    stats.trades_lost += 1
                self._circuit_breaker.record_trade_result(pnl)
                self._state.update_daily_stats(stats)
                try:
                    self._state.delete_state(f"trade_maxloss_{trade_id}")
                except Exception:  # noqa: BLE001
                    pass
                _t = self._find_trade_by_id(trade_id)
                if _t and _t.entry_time:
                    try:
                        if datetime.fromisoformat(_t.entry_time).date() == today:
                            self._pdt_guard.record_day_trade(_t.symbol)
                    except Exception as _e:  # noqa: BLE001
                        log.debug("pdt_record_failed", error=str(_e))
                if _t:
                    self._thompson.record_outcome(
                        strategy=_t.strategy, won=pnl > 0, pnl=pnl)
                log.warning("orphan_close_booked", trade_id=trade_id, pnl=pnl,
                            reason=row.get("exit_reason", ""))
                await self._send_notification(
                    f"ORPHAN CLOSE BOOKED: {(_t.symbol if _t else trade_id)} "
                    f"P&L ${pnl:.2f} ({row.get('exit_reason') or 'unknown'}) — "
                    "found by reconcile; breaker/stats now counted it.")
            except Exception as e:  # noqa: BLE001 — one bad row must not stop the drain
                log.error("exit_outbox_booking_failed", trade_id=trade_id,
                          error=str(e))

    async def _process_completed_exits(self, completed_exits: list[dict]) -> None:
        """Book completed exits: daily stats, circuit breaker, Thompson, drift.

        Must be called from EVERY check_fills() call site — most exits fill
        during the 30s fast monitor, and discarding its return used to mean
        the breaker/learning never saw them.
        """
        for ex in completed_exits:
            trade_id = ex["trade_id"]

            # W1 (R23 concurrency-2): claim the outbox row before booking.
            # Exactly one booker wins the DELETE — if the cycle drain (or a
            # concurrent pass) already booked this close, do NOT double-count
            # it into stats/breaker/Thompson.
            try:
                _claimed = self._state.claim_exit_booking(trade_id)
            except Exception:  # noqa: BLE001 — outbox unavailable: book as before
                _claimed = True
            if not _claimed:
                log.warning("exit_booking_already_claimed", trade_id=trade_id)
                continue

            # R15 #8 (Tier-2 #4 for real): the fill-time booking embedded a
            # FLAT $0.65/leg/side commission estimate inside realized_pnl —
            # real commissions only ever reached the commission COLUMN, so
            # every new close drifted from the broker again (the exact class
            # D1 just restated away). True up BEFORE stats/breaker/Thompson
            # see the number: swap the estimate out, the ledger truth in.
            try:
                _real_comm0 = self._state.total_commission(trade_id)
                _t0 = self._find_trade_by_id(trade_id)
                # R16: commissionReports for the 4 exit legs arrive ASYNC —
                # truing up against a partial ledger (e.g. entry legs only)
                # overstates realized_pnl permanently. Require a complete
                # ledger (>= 2x leg count: entry side + exit side) before
                # swapping; else defer to the post-market re-true-up.
                _legs_n = 4
                try:
                    import json as _json
                    _legs_n = max(1, len(_json.loads(_t0.legs))) if _t0 and _t0.legs else 4
                except Exception:  # noqa: BLE001
                    pass
                _exec_n = self._state.count_executions(trade_id)
                if _real_comm0 > 0 and _t0 and _exec_n >= 2 * _legs_n:
                    from ait.execution.executor import TradeExecutor as _TE
                    _delta = round(_TE.commission_estimate(_t0) - _real_comm0, 2)
                    if abs(_delta) >= 0.01:
                        ex["realized_pnl"] = round(ex["realized_pnl"] + _delta, 2)
                        self._state.update_trade_realized_pnl(
                            trade_id, ex["realized_pnl"])
                        log.info("realized_pnl_commission_trueup",
                                 trade_id=trade_id, delta=_delta,
                                 real_comm=round(_real_comm0, 2))
                elif _t0:
                    self._state.set_state(f"trueup_pending_{trade_id}",
                                          datetime.now().date().isoformat())
                    log.info("commission_trueup_deferred_partial_ledger",
                             trade_id=trade_id, executions=_exec_n,
                             expected=2 * _legs_n)
            except Exception as _e:  # noqa: BLE001 — booking must never die here
                log.warning("commission_trueup_failed",
                            trade_id=trade_id, error=str(_e))

            realized_pnl = ex["realized_pnl"]

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

            # R14 #9: the exit completed — clear its reject-backoff record so a
            # future, unrelated exit on this trade_id starts with a clean slate.
            getattr(self, "_exit_attempts", {}).pop(trade_id, None)

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
            # R7: stamp the REAL round-trip commission (from the executions
            # ledger) onto the closed trade row for the PF verdict.
            try:
                _real_comm = self._state.total_commission(trade_id)
                if _real_comm > 0:
                    self._state.update_trade_commission(trade_id, _real_comm)
            except Exception:  # noqa: BLE001
                pass
            if trade:
                self._thompson.record_outcome(
                    strategy=trade.strategy,
                    won=realized_pnl > 0,
                    pnl=realized_pnl,
                )
                # Deep-audit BC-L2: deriving direction from P&L sign is
                # meaningless for market-neutral strategies (a strangle win
                # is NOT "bullish") and poisons drift accuracy tracking.
                # R17: long_straddle is equally direction-agnostic (buys
                # both a call and a put) and was missing from this list.
                if trade.strategy not in ("iron_condor", "short_strangle", "long_straddle"):
                    actual_dir = "bullish" if realized_pnl > 0 else "bearish"
                    # R5 audit CRITICAL: predictions are recorded under
                    # "{symbol}-{direction}" (see _scan_symbol) but outcomes
                    # were recorded under the real trade id — keys never
                    # matched, record_outcome silently no-opped, and drift
                    # detection had NEVER completed a single sample. Use the
                    # same key the prediction was recorded under (predictions
                    # key on bullish/bearish; TradeDirection is long/short).
                    _tdir = str(getattr(getattr(trade, "direction", ""), "value",
                                        getattr(trade, "direction", "")) or "")
                    _dir = {"long": "bullish", "short": "bearish"}.get(_tdir, _tdir)
                    self._trainer.drift_detector.record_outcome(
                        trade_id=f"{trade.symbol}-{_dir}",
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

            await self._check_economic_calendar_exhaustion()

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
                # R16: was DEBUG — the gap-day daily-loss brake could fail on
                # every tick and be invisible at the configured log level, so
                # the one control that stops a bleeding day would look healthy
                # while doing nothing. WARNING + a page (gated) instead.
                log.warning("mtm_check_failed", error=str(_e),
                            note="daily MTM loss brake did not evaluate this "
                                 "tick — the gap-day halt is not protecting "
                                 "the book until this clears")
                if self._alert_gate("mtm_brake_broken", interval_s=3600):
                    await self._send_notification(
                        f"MTM BRAKE NOT EVALUATING: {type(_e).__name__}: {_e}. "
                        f"The daily-loss halt is inactive until fixed; entries "
                        f"are NOT being blocked by it."
                    )

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
            _entries, completed_exits = await self._executor.check_fills_safe()
            await self._process_completed_exits(completed_exits)

            # SEC 8-K material-event check every ~5 min (every 10th fast cycle)
            self._edgar_check_count += 1
            if self._edgar_check_count >= 10:
                self._edgar_check_count = 0
                await self._check_material_events()
            self._clear_loop_error("fast_monitor_error")  # R8: success resets streak
            self._watchdog.note_success("trading_loop")
        except Exception as e:
            # Deep-audit BC-H1: this was log.debug — the 30s stop/TP engine
            # could silently no-op for HOURS with a green heartbeat. Surface
            # loudly and feed the watchdog error counter.
            log.warning("fast_monitor_error", error=str(e))
            try:
                self._watchdog.record_error("trading_loop", f"fast_monitor: {e}")
            except Exception:
                pass
            await self._note_loop_error("fast_monitor_error", str(e))

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
            # R5 audit: was log.debug — a permanently-failing 8-K check (e.g.
            # SEC blocking the UA) looked identical to "no filings" for months.
            log.warning("material_event_check_failed", error=str(e))

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
                # R6: file heartbeat for the SUPERVISOR's hang detection —
                # the in-process watchdog dies with a hung event loop, and
                # all process checks only prove existence, not liveness.
                try:
                    # R8: only refresh the heartbeat when the loop is actually
                    # HEALTHY — during the 07-10 error loop the top-of-loop
                    # touch kept the supervisor's hang detection blind while
                    # stops were dead. Any active error streak lets it go
                    # stale, so the 15-min supervisor restart+alert fires.
                    _streaks = getattr(self, "_err_streaks", {})
                    if not any(n for n, _ in _streaks.values()):
                        from pathlib import Path as _PHB
                        _PHB("data/bot_heartbeat").write_text(datetime.now().isoformat())
                except Exception:  # noqa: BLE001
                    pass


                if time_since_scan >= scan_interval:
                    # Full cycle: scan for new trades + check positions.
                    # R18 AMPLIFIER FIX: time_since_scan used to reset ONLY on
                    # success. When _trading_cycle raised, the counter kept
                    # growing, so EVERY subsequent 30s iteration re-took this
                    # branch and _monitor_positions_fast never ran again — the
                    # 2026-08-11 outage killed the MTM daily-loss brake, the
                    # 8-K material-event check and the read-only re-probe as
                    # collateral, not just entries. The scan cadence must
                    # advance whether or not the cycle succeeded, so a broken
                    # scan can never starve the fast monitor.
                    try:
                        await self._trading_cycle()
                        self._clear_loop_error("trading_cycle_error")  # R8: success resets streak
                        self._watchdog.note_success("trading_loop")
                    finally:
                        time_since_scan = 0
                else:
                    # Fast check: only monitor existing positions and fills
                    await self._monitor_positions_fast()
                    # R8: check_and_recover was only reachable from the outer
                    # exception handler (i.e. never) — run it periodically so
                    # component-down states can actually page.
                    self._wd_tick = getattr(self, "_wd_tick", 0) + 1
                    if self._wd_tick % 10 == 0:
                        try:
                            await self._watchdog.check_and_recover()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as e:
                log.error("trading_cycle_error", error=str(e))
                await self._note_loop_error("trading_cycle_error", str(e))
                self._circuit_breaker.record_api_failure()
                self._watchdog.record_error("trading_loop", str(e))

            await asyncio.sleep(monitor_interval)
            time_since_scan += monitor_interval

    async def _trading_cycle(self) -> None:
        """Single trading cycle: scan all symbols, check positions, execute signals."""
        # R13 (human-factors): make the entry-freeze state OBSERVABLE. The
        # halt check lives deep in _try_execute, so on a gated day (e.g. CPI
        # economic_event_skip) `entries_halted` never logged — 90 min of
        # frozen RTH on 07-14 produced ZERO greppable evidence and the
        # RUNBOOK kill-switch drill greps for exactly that. One line per
        # scan cycle, only while a halt file exists.
        try:
            from pathlib import Path as _P13
            _halts = [p.name for p in _P13("data").glob("HALT*") if p.is_file()]
            if _halts:
                log.warning("entries_frozen", halt_files=_halts)
        except Exception:  # noqa: BLE001
            pass

        # 1. Check circuit breaker
        if self._circuit_breaker.is_tripped:
            log.warning("trading_halted", reason=self._circuit_breaker.get_status().reason)
            # R13 (human-factors): this notification was UNTHROTTLED — on
            # 07-10 it sent the same message 62 times in one day (every
            # 5-min cycle while tripped), burying the signal. Notify once
            # per trip, then at most hourly while it stays tripped. The
            # per-cycle log line above is unchanged.
            _now = time.time()
            _last = getattr(self, "_cb_last_notify", 0.0)
            _was_tripped = getattr(self, "_cb_was_tripped", False)
            if not _was_tripped or (_now - _last) >= 3600:
                await self._send_notification(
                    f"CIRCUIT BREAKER: Trading halted - "
                    f"{self._circuit_breaker.get_status().reason}")
                self._cb_last_notify = _now
            self._cb_was_tripped = True
            return
        self._cb_was_tripped = False

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

        # R12-C: delta-hedging check REMOVED — dead since inception (greeks
        # feed ~0) and dangerous: its auto-execute path placed SPY stock
        # MARKET orders directly via the broker client, bypassing every
        # executor guardrail (rate limit, INST-5 defined-risk, GOV-1 NBBO,
        # market-hours). Module retired to deprecated/src/.

        # 5. Check if we should avoid new trades (last 15 min)
        if self._scheduler.should_avoid_new_trades():
            log.debug("skipping_new_trades", reason="close_to_market_close")
            return

        # 6. Check fill status of pending orders (entry + exit)
        _filled_entries, completed_exits = await self._executor.check_fills_safe()
        await self._process_completed_exits(completed_exits)

        # 6b. W1: book closes nobody booked — reconciler/sweep closes and
        # bookings lost to a crash between the sqlite close and the callback.
        await self._drain_exit_outbox()

        # 7. Get effective universe (learning + capital tier filtering)
        adaptor = self._learning.adaptor

        # Get current account value for capital tier decisions
        account_snapshot = await self._account.get_snapshot()
        current_capital = account_snapshot.net_liquidation if account_snapshot else 10_000.0
        # R16: publish NLV so return/drawdown percentages stop being computed
        # off a hardcoded 196000 that had already drifted from the real book
        # (and would be ~65x wrong at the planned $3k go-live scale).
        if account_snapshot and current_capital > 0:
            try:
                self._state.set_state("last_net_liquidation", str(current_capital))
            except Exception:  # noqa: BLE001
                pass
        tier_config = self._capital_tiers.get_config(current_capital)

        # Filter universe: learning restrictions + capital tier affordability
        universe = [
            s for s in self._settings.trading.universe
            if adaptor.is_symbol_allowed(s)
        ]
        universe = self._capital_tiers.filter_universe(universe, current_capital)

        # R16: log what is ACTUALLY tradeable, not the tier's menu. The tier
        # advertised 7 strategies (incl. short_strangle/long_straddle) while
        # config.strategies has been [iron_condor] since 07-22 — the log read
        # as if undefined-risk shapes were live and cost real diagnostic time
        # during the audit. Intersection is the truth; both are shown.
        _tradeable = [s for s in tier_config.allowed_strategies
                      if s in set(self._settings.options.strategies)]
        log.debug("capital_tier_active",
                  tier=tier_config.tier.value,
                  capital=f"${current_capital:,.0f}",
                  strategies=_tradeable,
                  tier_menu=tier_config.allowed_strategies,
                  config_enabled=list(self._settings.options.strategies),
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
        if contract_flag("AIT_SKIP_MACRO_EVENTS"):
            if self._economic_cal.should_skip_trading():
                log.info("economic_event_skip",
                         events=str(self._economic_cal.get_upcoming_events(days=2)))
                return


        # 9. Scan universe for new opportunities
        # R7-SOON: refresh the per-trade dollar budget once per cycle so
        # strategy construction fits the CURRENT account (paper $197k today,
        # $2.1k at Phase 2 via AIT_SIMULATED_CAPITAL) — and run the launch-
        # size coherence self-test whenever NLV moves >20%.
        try:
            _nlv = await self._account.get_net_liquidation()
            if _nlv and _nlv > 0:
                self._risk_budget = _nlv * self._settings.risk.max_position_risk_pct
                _last = getattr(self, "_coherence_checked_nlv", 0.0)
                if not _last or abs(_nlv - _last) / _last > 0.20:
                    self._coherence_checked_nlv = _nlv
                    await self._launch_size_coherence_check(_nlv)
        except Exception as _e:  # noqa: BLE001
            log.debug("risk_budget_refresh_failed", error=str(_e))

        vix = await self._market_data.get_vix()

        # Fetch cross-asset context once per scan cycle (VIX + SPY history for ML)
        market_context = await self._build_market_context()

        # R7 (gap audit): two-phase scan — collect candidates across the
        # WHOLE universe, then execute best-first. Slots were previously
        # allocated by config-file order (first acceptable condor won), so
        # the track record measured second-best trades whenever slots were
        # scarce. Score = eff_conf + 0.3 x credit/width (credit structures).
        candidates: list = []
        for symbol in universe:
            try:
                await self._scan_symbol(symbol, vix, market_context,
                                        collect=candidates)
            except Exception as e:
                log.warning("symbol_scan_failed", symbol=symbol, error=str(e))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            log.info("candidate_ranking",
                     ranked=[(f"{c[1].symbol}:{c[1].strategy_name}",
                              round(c[0], 3)) for c in candidates[:8]])
            handled_symbols: set[str] = set()
            for score, sig, conf, sent, reg in candidates:
                if sig.symbol in handled_symbols:
                    continue
                try:
                    handled = await self._try_execute(sig, conf, sent, reg)
                except Exception as e:  # noqa: BLE001
                    log.warning("candidate_execute_failed",
                                symbol=sig.symbol, error=str(e))
                    handled = True
                if handled:
                    handled_symbols.add(sig.symbol)

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

    async def _scan_symbol(self, symbol: str, vix: float | None, market_context: dict | None = None, collect: list | None = None) -> None:
        """Analyze a single symbol for trading opportunities.

        Parallelizes data fetches: historical, sentiment, and IV rank
        are fetched concurrently with asyncio.gather.
        """
        adaptor = self._learning.adaptor
        # R11 (R9 instrumentation): ~95% of per-symbol scan time was dark,
        # untimed I/O (DIA/GLD ~48s each, cause unknown). Coarse stage clocks.
        import time as _tt
        _scan_t0 = _tt.monotonic()
        _stage_ms: dict = {}

        # Parallel fetch: historical data, sentiment, IV rank
        # 2 years of history for robust features (iv_rank, vol percentiles, trend)
        hist_task = self._market_data.get_historical(symbol, days=504)
        iv_rank_task = self._estimate_iv_rank(symbol)

        # R12-C: sentiment retired — R7 traced its contribution to IC
        # decisions to exactly zero (range override replaces confidence
        # before sentiment could reach any decision; its ML features were
        # constant at train time). Downstream sites are None-tolerant.
        sentiment = None
        hist, iv_rank = await asyncio.gather(
            hist_task, iv_rank_task,
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
        # fail-direction-05 (R22): this used to be `iv_rank = 50.0` with NO
        # log — and 50 is the one value that passes BOTH IV gates
        # (iron_condor's >=15 floor and the risk manager's <=85 VRP cap), so
        # a broken IV layer made every symbol scan as "perfectly average IV"
        # and journalled that fabrication into entry_iv_rank as a real
        # measurement. Unknown now propagates as None.
        if isinstance(iv_rank, Exception):
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason=f"scan_gather_raised:{type(iv_rank).__name__}",
                        detail=str(iv_rank)[:160])
            iv_rank = None
        if iv_rank is None:
            # FAIL CLOSED for NEW entries only. Every downstream consumer
            # (iron_condor's IV floor, the VRP cap, the journalled
            # entry_iv_rank, meta-label training features) treats iv_rank as
            # a measurement; with none there is no basis to sell premium on
            # this symbol this scan. Exits/monitoring never run through
            # _scan_symbol, so open positions keep being managed and exited
            # normally — this only refuses to OPEN.
            log.warning("iv_rank_unavailable_entry_skipped", symbol=symbol)
            if self._alert_gate(f"iv_rank_unavailable:{symbol}", interval_s=3600):
                await self._send_notification(
                    f"IV DATA OUTAGE — {symbol}: IV rank could not be measured, "
                    f"so NEW entries on {symbol} are being skipped (fail-closed). "
                    f"Open positions are still monitored and exited normally."
                )
            return

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
            live_signals=None,  # R12-C: retired features; kwarg ignored
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

        # R12-C: sentiment confidence adjustment removed with the sentiment
        # stack — it never reached an IC decision (range override wins) and
        # carried a wrong-signed adjustment on the directional path.
        final_confidence = prediction.confidence

        # Multi-timeframe analysis: boost/penalize confidence based on alignment
        # Use intraday_full (full SQLite history) not the incremental fetch — the
        # MTF analyser needs ≥20 bars and the incremental fetch is often just a few.
        mtf = await self._mtf_analyzer.analyze_async(hist, intraday_full)  # R11: off-loop
        final_confidence = max(0, min(1, final_confidence + mtf.confidence_boost))

        # Pre-compute daily features once — used for fractal penalty and meta-labeler below.
        from ait.ml.features import FeatureEngine
        features_df = FeatureEngine().compute(hist)

        # R20: stash the entry-time feature row per symbol so _try_execute can
        # persist it into trade_context.entry_signals. That column has been
        # "{}" for EVERY trade ever taken, which means 11 of the meta-labeler's
        # 20 features were never captured — the "meta-labeler at 50 closes"
        # milestone was unreachable and nobody knew until the R19 coverage
        # guard refused to train on 9/20 features. Every close without this
        # is training data lost forever.
        try:
            _snap = getattr(self, "_entry_feature_snap", None)
            if _snap is None:
                _snap = self._entry_feature_snap = {}
            _snap.pop(symbol, None)
            if not features_df.empty:
                _snap[symbol] = features_df.iloc[-1]
        except Exception:  # noqa: BLE001 — telemetry must never break a scan
            pass

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
        _stage_ms["pre_chain"] = round((_tt.monotonic() - _scan_t0) * 1000)
        chains = await self._options_chain.get_chain(symbol)
        _stage_ms["chain"] = round((_tt.monotonic() - _scan_t0) * 1000) - _stage_ms["pre_chain"]
        if not chains:
            return

        # R16: SELF-HEALING IV STORE. The daily IV series' only writer was a
        # manual backfill script nobody scheduled — frozen since 07-09, so
        # every iv_rank ran on a month-old snapshot. Persist today's chain
        # ATM IV once per symbol per day; the store stays fresh as a side
        # effect of scanning.
        self._persist_daily_iv(symbol, chains)

        # Filter liquid options
        filtered_chains = [
            c.filter_liquid(self._settings.options) for c in chains
        ]

        # R12-C: options-flow analysis removed — gate/boost were
        # paper-bypassed, its ML features were constant at train time, and
        # bias_strength was a proportion (one small sweep read as 1.0).
        # Module retired to deprecated/src/.
        underlying_price = hist["Close"].iloc[-1] if "Close" in hist.columns else 0

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
                risk_budget=getattr(self, "_risk_budget", None),
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
            range_pred = None  # fail-direction-02: stays None when the gate has no evidence
            if self._range_predictor and self._range_predictor.is_trained:
                range_pred = self._range_predictor.predict(
                    hist, symbol=symbol,
                    market_context=market_context,
                    live_signals=None,  # R12-C: retired features; kwarg ignored
                    intraday_store=self._historical,
                )
                if range_pred:
                    log.info("range_prediction", symbol=symbol,
                             p_in_range=f"{range_pred.probability_in_range:.3f}",
                             threshold=range_pred.threshold_pct,
                             horizon_days=range_pred.horizon_days)
                    # Override confidence for iron condors with range probability.
                    # R16: the override itself is now gated on entry_gates_enabled —
                    # with gates OFF the prediction is LOGGED (line above, for future
                    # studies) but must not flow into eff_conf, where the risk
                    # manager's min_confidence=0.50 and the Friday >=0.90 gate were
                    # still vetoing entries. With gates off, ML observes; never vetoes.
                    if self._settings.ml.entry_gates_enabled:
                        for s in signals:
                            if s.strategy_name in ("iron_condor", "short_strangle"):
                                s.confidence = range_pred.probability_in_range
                                model_overridden.add(s.strategy_name)

            # fail-direction-02 (R22): ONE fail direction when the payoff-
            # matched gate has no evidence. Pre-fix, an unavailable range
            # model (missing/corrupt range.pkl — a documented incident class,
            # untrained after a caught training failure, or a routine
            # predict()->None) left iron_condor/short_strangle carrying the
            # DIRECTIONAL model's confidence, so the armed gate INVERTED by
            # regime: on neutral days the neutral-only filter below dropped
            # every IC silently (a sample stop), on directional days ICs were
            # admitted on directional evidence — precisely the wrong-payoff
            # basis this override exists to replace. Live mirror of
            # walkforward.py:1376 (range_min_confidence=1.0 on a training
            # failure). With gates OFF the model stays observe-only by design
            # (R16), so nothing is dropped there.
            if self._settings.ml.entry_gates_enabled and range_pred is None:
                _ungated = [s for s in signals
                            if s.strategy_name in RANGE_GATED_STRATEGIES]
                if _ungated:
                    signals = [s for s in signals
                               if s.strategy_name not in RANGE_GATED_STRATEGIES]
                    log.warning(
                        "range_gate_unavailable_entries_blocked",
                        symbol=symbol,
                        blocked=sorted({s.strategy_name for s in _ungated}),
                        model_trained=bool(self._range_predictor
                                           and self._range_predictor.is_trained),
                    )
                    if self._alert_gate("range_gate_unavailable", interval_s=3600):
                        await self._send_notification(
                            "RANGE GATE UNAVAILABLE — the range model produced no "
                            "prediction (missing/untrained artifact or predict() "
                            "returned nothing), so iron condor / short strangle "
                            "entries are being refused FAIL-CLOSED on every symbol "
                            "until it recovers. Exits are unaffected."
                        )

            # Vol-magnitude model (Tier 1) for long straddles — predicts P(big move)
            # Direct signal for "buy volatility" strategies, more accurate than
            # the previous "1 - range probability" hack.
            if self._vol_mag_predictor and self._vol_mag_predictor.is_trained:
                vm_pred = self._vol_mag_predictor.predict(
                    hist, symbol=symbol,
                    market_context=market_context,
                    live_signals=None,  # R12-C: retired features; kwarg ignored
                )
                if vm_pred:
                    log.info("vol_mag_prediction", symbol=symbol,
                             p_big_move=f"{vm_pred.probability_big_move:.3f}",
                             threshold=vm_pred.threshold_pct,
                             horizon_days=vm_pred.horizon_days)
                    # R17: unlike the range-model override above, this had no
                    # entry_gates_enabled check at all — it applied
                    # unconditionally whenever the model was trained,
                    # regardless of the master ML-gate switch.
                    if self._settings.ml.entry_gates_enabled:
                        for s in signals:
                            if s.strategy_name == "long_straddle":
                                s.confidence = vm_pred.probability_big_move
                                model_overridden.add(s.strategy_name)

            # Model-confidence floor — applies whenever a payoff-matched model
            # overrode a signal's confidence, REGARDLESS of which models are
            # trained. (Was nested inside the vol-mag block, so an untrained
            # vol-mag model silently skipped the iron-condor floor — audit
            # 2026-07-07, quick-win under item 1.2.)
            if self._settings.ml.entry_gates_enabled:
                signals = [
                    s for s in signals
                    if s.strategy_name not in model_overridden
                    or s.confidence >= RANGE_MIN_CONFIDENCE
                ]
            # else: ABLATION VERDICT 2026-08-03 — the floor vetoed nothing in
            # 3 identical walk-forward runs; gate stack alone decides entries.

            # Neutral-only mode (directional confidence below threshold):
            #  - drop directional strategies — no directional basis to trade;
            #  - drop neutral strategies that were NOT range-model-overridden —
            #    without the range model there is no basis at all.
            if neutral_only:
                signals = [
                    s for s in signals
                    if s.strategy_name in ("iron_condor", "short_strangle")
                    and (s.strategy_name in model_overridden
                         or not self._settings.ml.entry_gates_enabled)
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

            _stage_ms["total"] = round((_tt.monotonic() - _scan_t0) * 1000)
            log.info("scan_symbol_timing", symbol=symbol, **_stage_ms)

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
                # R7 two-phase scan: in collect mode, defer execution to the
                # cross-symbol ranking pass instead of executing in-place.
                if collect is not None:
                    _cw = 0.0
                    try:
                        from ait.strategies.base import CREDIT_STRATEGIES as _CS7
                        if (signal.strategy_name in _CS7 and signal.max_loss
                                and signal.entry_price):
                            _gross = abs(signal.entry_price) * 100
                            _cw = _gross / (signal.max_loss + _gross)
                    except Exception:  # noqa: BLE001
                        pass
                    collect.append((eff_conf + 0.3 * _cw, signal, eff_conf,
                                    sentiment, regime))
                    continue
                handled = await self._try_execute(signal, eff_conf, sentiment, regime)
                if handled:
                    break  # executed or symbol-level block — done with this symbol
                # else: risk-rejected — try the next-ranked strategy

    def _prestamp_mtm_baseline(self, unrealized_at_close: float) -> None:
        """R14 #8: stamp the NEXT trading day's MTM baseline with the PRIOR
        CLOSE's unrealized P&L, so a gap-open shows up as the full move from it.

        The mark-to-market brake computes mtm_day = realized + (unrealized_now
        - unrealized_at_SOD). Its SOD baseline used to be captured lazily on the
        first fast-monitor tick — AFTER the open — so on a -8% gap the already-
        collapsed unrealized became the baseline and mtm_day registered ~0: the
        brake was blind to exactly the overnight gap it exists to catch. The
        monitor's lazy write only fires when the key is empty, so a value
        pre-stamped here under the upcoming date wins.
        """
        try:
            key = f"mtm_sod_{next_market_open().date().isoformat()}"
            self._state.set_state(key, str(float(unrealized_at_close)))
            log.info("mtm_baseline_prestamped", key=key,
                     unrealized_at_close=round(float(unrealized_at_close), 2))
        except Exception as _e:  # noqa: BLE001
            log.warning("mtm_baseline_prestamp_failed", error=str(_e))

    @staticmethod
    def _position_capital_at_risk(signal, quantity: int) -> float:
        """TOTAL dollars at risk for an executed position = per-contract
        max_loss x contract count.

        SR-M6 (2026-07-15): every strategy builds its signal with quantity=1
        (iron_condor.py, spreads.py, long_options.py, straddles.py all set
        max_loss for ONE contract), but the trade executes `quantity`
        contracts. Both consumers of this number want the position TOTAL:
          - the aggregate capital-at-risk guard sums trade_maxloss_* across the
            book (risk/manager.py) — per-contract values let an N-lot book pass
            a portfolio cap it actually blows through by Nx;
          - capital_at_risk is the retained denominator for the go-live verdict
            (PF-per-unit-risk, DD-vs-deployed-risk) — D2's whole question is
            what that denominator is, so it must be real dollars at risk, not a
            per-lot slice.
        Previously both were stored per-contract (the quantity=1 value).
        Dormant while small-account sizing pins every trade to 1 lot; wrong the
        moment sizing scales past 1 (NLV>$10k, or a learning multiplier >1) —
        i.e. exactly at Phase-2, on the go-live path.

        R17: this used to return 0.0 whenever `is_defined_risk` was False —
        but that flag governs trade-RANKING eligibility (base.py), not
        whether a dollar-risk number exists. short_strangle (undefined-risk)
        still carries a real, positive, stress-estimated max_loss
        (straddles.py's `estimated_max_loss`), so it was being recorded as
        $0 exposure — invisible to the aggregate capital-at-risk cap. Any
        positive max_loss is real risk regardless of defined/undefined
        status; only a genuinely unset max_loss floors to 0.0.
        """
        max_loss = float(signal.max_loss or 0.0)
        if max_loss > 0:
            return max_loss * max(1, int(quantity))
        return 0.0

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

    async def _launch_size_coherence_check(self, nlv: float) -> None:
        """R7-SOON: assert the cap grid is mutually satisfiable at this NLV.

        The Phase-2 launch account bricked silently before: sizer budget <
        one minimum-viable condor, and the daily-loss breaker smaller than a
        single trade's stop. Runs at startup and whenever NLV moves >20%;
        alerts LOUDLY instead of letting the account sit structurally unable
        to trade (or unable to survive one normal stop-out).
        """
        try:
            budget = nlv * self._settings.risk.max_position_risk_pct
            import os as _os_c
            min_credit = contract_float("AIT_IC_MIN_CREDIT")
            min_ratio = contract_float("AIT_IC_MIN_CREDIT_WIDTH")
            # Minimum viable condor given the economics floors: the narrowest
            # width whose floor credit is even possible (credit <= ~0.45x
            # width in practice at 0.20 delta), then its max loss.
            min_width = max(1.0, min_credit / 0.45)
            min_viable_cost = (min_width - min_credit) * 100
            # Typical stopped-trade loss under the 1.25x credit limit
            loss_mult = contract_float("AIT_CREDIT_LOSS_LIMIT")
            typical_stop_loss = loss_mult * max(min_credit, min_ratio * min_width) * 100
            breaker_pct = float(getattr(self._settings.risk, "max_daily_loss_pct", 0.02) or 0.02)  # R8: real field name
            breaker_usd = nlv * breaker_pct
            problems = []
            if budget < min_viable_cost:
                problems.append(
                    f"per-trade budget ${budget:.0f} < minimum viable condor "
                    f"~${min_viable_cost:.0f} (width ${min_width:.1f}, credit "
                    f"floor ${min_credit:.2f}) — NO condor can pass sizing")
            if breaker_usd < 1.5 * typical_stop_loss:
                problems.append(
                    f"daily-loss breaker ${breaker_usd:.0f} < 1.5x one normal "
                    f"stop-out ~${typical_stop_loss:.0f} — a single stopped "
                    f"trade halts the whole day")
            log.info("launch_size_coherence",
                     nlv=round(nlv), budget=round(budget),
                     min_viable_ic=round(min_viable_cost),
                     breaker=round(breaker_usd),
                     problems=len(problems))
            if problems:
                await self._send_notification(
                    "CAP-GRID INCOHERENT at NLV ${:,.0f}:\n- ".format(nlv)
                    + "\n- ".join(problems)
                    + "\nFix knobs (risk.max_position_risk_pct / breaker / "
                      "credit floors) before expecting trades."
                )
        except Exception as _e:  # noqa: BLE001
            log.warning("coherence_check_failed", error=str(_e))

    async def _note_loop_error(self, kind: str, error: str) -> None:
        """R8 (incident 2026-07-10): 516 fast-monitor + 145 trading-cycle
        errors produced ZERO alerts — per-iteration try/excepts logged a
        WARNING and kept looping while stops/TPs were dead all session. Any
        streak of >=5 consecutive failures at one site pages Telegram once,
        re-pages hourly while it persists; the streak resets on the first
        clean iteration (_clear_loop_errors)."""
        import time as _t
        st = getattr(self, "_err_streaks", None)
        if st is None:
            st = self._err_streaks = {}
        n, last_alert = st.get(kind, (0, 0.0))
        n += 1
        if n >= 5 and (_t.time() - last_alert) > 3600:
            last_alert = _t.time()
            try:
                await self._send_notification(
                    f"LOOP IMPAIRED — {kind}: {n} consecutive failures. "
                    f"Position protection may be OFF. Latest: {error[:200]}"
                )
            except Exception:  # noqa: BLE001 — alerting must never crash the loop
                pass
        st[kind] = (n, last_alert)

    def _clear_loop_error(self, kind: str) -> None:
        """Reset one streak — call ONLY from that path's success point."""
        st = getattr(self, "_err_streaks", None)
        if st and kind in st:
            st[kind] = (0, st[kind][1])

    async def _get_vix_lkg(self) -> float | None:
        """VIX with a 45-minute last-known-good cache (R7 fail-closed)."""
        import time as _t
        try:
            v = await self._market_data.get_vix()
        except Exception:  # noqa: BLE001
            v = None
        if v and v > 0:
            self._vix_lkg = (float(v), _t.time())
            self._vix_fail_streak = 0
            return float(v)
        lkg = getattr(self, "_vix_lkg", None)
        if lkg and (_t.time() - lkg[1]) <= 2700:
            log.warning("vix_using_last_known_good", vix=lkg[0],
                        age_s=int(_t.time() - lkg[1]))
            self._vix_fail_streak = 0
            return lkg[0]
        log.error("vix_unavailable_fail_closed")
        # fail-direction-01 (R22): the fail-closed stop itself is correct —
        # but it was SILENT (log.error only; grep found no notification on
        # any VIX path). With config strategies = [iron_condor] this refuses
        # 100% of entries for the whole outage and the operator's only clue
        # is a day with zero trades: the same silent-full-stop signature the
        # blackout gate was hardened against in R18. Reaching here already
        # means BOTH sources failed AND the last known good is >45 min old,
        # so no failure streak is needed to establish this is an outage.
        # This never fires on a genuinely high VIX — that returns a real
        # number above and is refused downstream by the risk manager, which
        # is a decision, not a data failure.
        _n = int(getattr(self, "_vix_fail_streak", 0)) + 1
        self._vix_fail_streak = _n
        if self._alert_gate("vix_gate_unavailable", interval_s=3600):
            await self._send_notification(
                f"VIX DATA OUTAGE — no VIX print from either source and the "
                f"last known good is older than 45 min ({_n} consecutive "
                f"checks). Credit entries are being refused FAIL-CLOSED, so "
                f"the bot is NOT opening trades until this clears. Exits are "
                f"unaffected."
            )
        return None

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

    async def _check_economic_calendar_exhaustion(self) -> None:
        """R17: page once when the hardcoded macro-event calendar is nearly
        exhausted. EconomicCalendar._warn_if_nearly_exhausted only ever
        logged (log.critical, no notification route) -- the exact "macro
        guards went blind and nobody noticed" failure it exists to catch.
        """
        if self._economic_cal.exhausted_warned and not getattr(
            self, "_econ_cal_exhausted_notified", False
        ):
            self._econ_cal_exhausted_notified = True
            await self._send_notification(
                "ECONOMIC CALENDAR NEARLY EXHAUSTED: macro-event guards "
                "are about to go blind. Extend the FOMC/CPI/NFP/GDP/PCE "
                "tables in economic_calendar.py — see logs for details."
            )

    def _post_stop_cooldown_until(self, symbol: str) -> str | None:
        """R12-B4 query, extracted so tests can execute it against a real DB.

        Returns the exit_time of the most recent qualifying stop close inside
        the cooldown window, else None.

        2026-08-25 defect: the query matched only '%stop_loss%', but the
        short-strike touch stop — live's PRIMARY loss exit since R12-B1 —
        writes 'short_strike_touch (spot ...)'. Executed proof from the live
        book: QQQ touch-stopped 08-24 09:47:33 (−$272.29) and re-entered at
        10:00:40, 13 minutes later, straight back into the same move the rule
        exists to avoid. Both spellings of a forced adverse-move exit now
        match. trailing_stop/breakeven_stop stay excluded on purpose: those
        fire at/above breakeven, so re-entry after them is not the
        autocorrelated-loss sequence R12-B4 targets.

        numeric-pairs-02 (R22): the window is ONE TRADING DAY (the spec), not
        a flat 30 wall-clock hours — see _post_stop_cooldown_cutoff.
        """
        import sqlite3 as _sq12
        _con = _sq12.connect("file:data/ait_state.db?mode=ro", uri=True)
        try:
            _row = _con.execute(
                "SELECT exit_time FROM trades WHERE symbol=? AND status='closed' "
                "AND (COALESCE(exit_reason_detailed,'') LIKE '%stop_loss%' "
                "OR COALESCE(exit_reason_detailed,'') LIKE '%short_strike_touch%') "
                "ORDER BY exit_time DESC LIMIT 1", (symbol,)).fetchone()
        finally:
            _con.close()
        if _row and _row[0]:
            from datetime import datetime as _dt12
            _exit_t = _dt12.fromisoformat(_row[0])
            # numeric-pairs-02 (R22): was `_dt12.now() - _exit_t < 30h` — a
            # flat wall-clock window with no calendar, so every Friday and
            # pre-holiday stop bought ZERO post-stop sessions of cooldown.
            # The spec is ONE TRADING day; see _post_stop_cooldown_cutoff.
            if _exit_t >= _post_stop_cooldown_cutoff():
                return _row[0]
        return None

    def _duplicate_guard_verdict(self, signal) -> str:
        """fail-direction-08: verdict of the three duplicate-order layers.

        Returns:
            "clear"      — every layer answered and none found a duplicate;
            "duplicate"  — a layer found one (it logged which);
            "unverified" — a layer could not answer (locked DB, disconnected
                            broker). The old code logged that at DEBUG and
                            carried on to placement, so a locked-DB window
                            plus a reconnect blip could put a SECOND live
                            order on the same symbol with two DEBUG lines as
                            the only evidence.
        """
        from datetime import datetime as dt

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
                        return "duplicate"  # symbol+strategy recently traded
        except Exception as e:  # noqa: BLE001 — fail CLOSED (was log.debug + place)
            log.warning("cooldown_check_failed_entry_refused",
                        symbol=signal.symbol, error=str(e))
            return "unverified"

        # GUARD #3 — working IBKR orders. Catch in-flight orders the local DB
        # hasn't caught up on (e.g. placed but not yet recorded/filled).
        # get_open_orders() returns [] when DISCONNECTED, which is
        # indistinguishable from a verified-empty book — so check the
        # connection explicitly and treat a blind read as unverified.
        try:
            if not getattr(self._ibkr, "connected", True):
                log.warning("working_order_check_disconnected_entry_refused",
                            symbol=signal.symbol)
                return "unverified"
            for t in (self._ibkr.get_open_orders() or []):
                if getattr(t.contract, "symbol", None) == signal.symbol:
                    log.info("working_order_skip", symbol=signal.symbol)
                    return "duplicate"
        except Exception as e:  # noqa: BLE001 — fail CLOSED (was log.debug + place)
            log.warning("working_order_check_failed_entry_refused",
                        symbol=signal.symbol, error=str(e))
            return "unverified"

        # GUARD #4 — in-memory pending tracker (empty after every restart,
        # which is why the two DB/broker layers above must stay honest).
        try:
            pending_symbols = set()
            for _oid, pending in self._executor._pending_orders.items():
                if hasattr(pending, 'signal') and hasattr(pending.signal, 'symbol'):
                    pending_symbols.add(pending.signal.symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("pending_tracker_check_failed_entry_refused",
                        symbol=signal.symbol, error=str(e))
            return "unverified"
        if signal.symbol in pending_symbols:
            log.info("symbol_has_pending_order", symbol=signal.symbol)
            return "duplicate"

        return "clear"

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
        # GOV (governance audit): restricted list — hard-ban a symbol without
        # a config edit + restart (post-incident control). One symbol per line.
        # fail-direction-11: a present-but-unreadable ban file must never mean
        # "no bans" — it blocks NEW entries entirely until it reads cleanly.
        _banned = read_restricted_symbols()
        if _banned is RESTRICTED_UNREADABLE:
            log.warning("entries_halted_restricted_list_unreadable",
                        symbol=signal.symbol, path=RESTRICTED_LIST_PATH)
            return True  # symbol handled: no new entries while the ban file
            #              cannot be read (exits are untouched)
        if signal.symbol.upper() in _banned:
            log.warning("symbol_restricted", symbol=signal.symbol)
            return True

        # R6 (user-approved): don't OPEN credit structures the macro flatten
        # will force-close within days — paying full round-trip costs for
        # 1-2 days of theta is systematically negative EV, and it recurs at
        # every CPI/NFP/FOMC. Entry-side mirror of the exit-engine flatten.
        # Returns False so the caller can still try a debit strategy.
        try:
            from ait.strategies.base import CREDIT_STRATEGIES as _CS6
            if signal.strategy_name in _CS6 and self._economic_cal:
                _d2e = self._economic_cal.days_until_next_event()
                # PLAN 2026-08-03 (user-approved): 4 -> config, default 1.
                # The <=4 window blocked ~half of all trading days (NFP+CPI+
                # PCE lead-outs) and refused exactly the elevated pre-event
                # premium a seller is paid to take; every 14-30 DTE hold
                # spans events regardless, and wings cap the surprise.
                _blackout = self._settings.risk.pre_event_blackout_days
                # R16: days_until_next_event counts CALENDAR days, so a Friday
                # entry ahead of a Monday event reads d2e=3 and sails past a
                # 1-day window with ZERO intervening sessions. Convert to
                # TRADING days so "1 day before the event" means one session.
                self._blackout_fail_streak = 0  # R18: healthy pass resets
                _sessions = self._sessions_until(_d2e)
                if _sessions is not None and _sessions <= _blackout:
                    log.info("credit_entry_skipped_pre_event",
                             symbol=signal.symbol,
                             strategy=signal.strategy_name,
                             days_to_event=_d2e, sessions_to_event=_sessions)
                    return False
        except Exception as _e:  # noqa: BLE001
            # R16: this was a blanket except-pass — a calendar failure made the
            # gate FAIL OPEN silently, entering exactly into the event it
            # exists to avoid. Fail CLOSED for credit structures and say so.
            log.error("pre_event_blackout_check_failed_failing_closed",
                      symbol=signal.symbol, strategy=signal.strategy_name,
                      error=str(_e))
            # R18: fail-closed is correct, but SILENT fail-closed is the 08-11
            # outage signature again. iron_condor is the only enabled strategy
            # and it IS a credit structure, so a persistent fault here blocks
            # 100% of entries indefinitely with nothing but a log line. Count
            # consecutive failures and PAGE — a gate that stops all trading
            # must announce itself.
            _n = getattr(self, "_blackout_fail_streak", 0) + 1
            self._blackout_fail_streak = _n
            if _n >= 3 and self._alert_gate("blackout_check_broken", interval_s=3600):
                await self._send_notification(
                    f"ENTRY GATE BROKEN — the pre-event blackout check has "
                    f"failed {_n} times in a row ({type(_e).__name__}: "
                    f"{str(_e)[:120]}). Credit entries are being refused "
                    f"fail-closed, so the bot is NOT opening trades until this "
                    f"clears. Exits are unaffected."
                )
            try:
                from ait.strategies.base import CREDIT_STRATEGIES as _CS6b
                if signal.strategy_name in _CS6b:
                    return True
            except Exception:  # noqa: BLE001
                return True

        # R12-B4 (user-approved): post-stop re-entry discipline. The old
        # cooldown keyed on ENTRY time, so a symbol stopped out on day 3 of a
        # trend could re-enter on the very next scan into the same move —
        # the most autocorrelated loss sequence condors produce. A stop-loss
        # close now blocks NEW entries on that symbol for 1 trading day.
        try:
            _stopped_at = self._post_stop_cooldown_until(signal.symbol)
            if _stopped_at:
                log.info("entry_blocked_post_stop_cooldown",
                         symbol=signal.symbol, stopped_at=_stopped_at)
                return False
        except Exception:  # noqa: BLE001 — cooldown must never block the loop
            pass

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
            # R15 minor: per-signal reject, NOT symbol-handled. Neutral
            # candidates (IC/strangle) are EXEMPT from this gate and carry
            # their own model-overridden confidence — returning True here
            # suppressed every first-hour condor whenever a directional
            # signal outranked it, throttling exactly the sample we need.
            return False

        # GUARDS #2-#4 — duplicate-order layers. fail-direction-08: each layer
        # used to swallow its own errors at DEBUG and fall through to PLACE.
        _dup = self._duplicate_guard_verdict(signal)
        if _dup == "duplicate":
            return True  # symbol already busy — the layer logged which one
        if _dup == "unverified":
            # A layer could not answer (locked DB, disconnected broker read).
            # "No duplicate found" and "could not check" are different facts:
            # place on the first, refuse this pass on the second.
            log.warning("duplicate_guard_unverified_entry_refused",
                        symbol=signal.symbol, strategy=signal.strategy_name,
                        detail=("a duplicate-order check failed; refusing the "
                                "entry this pass rather than risking a second "
                                "live order for the same symbol"))
            return True

        # Check if position would hold through earnings (IV crush risk)
        if signal.expiry:
            from datetime import date as date_cls
            expiry_date = signal.expiry if isinstance(signal.expiry, date_cls) else date_cls.fromisoformat(str(signal.expiry))
            if self._earnings.would_hold_through_earnings(signal.symbol, date.today(), expiry_date):
                log.info("trade_blocked_earnings", symbol=signal.symbol, expiry=str(signal.expiry))
                return True  # symbol blocked through earnings

        # Build trade request for risk validation
        # R7: VIX with last-known-good fallback. `or 0.0` made the VIX-28
        # credit halt FAIL OPEN on any fetch hiccup (0 is falsy, gate skipped).
        # A fresh failure now reuses the last good print (<=45 min old);
        # otherwise vix stays None and the risk manager fails CLOSED for
        # credit entries.
        current_vix = await self._get_vix_lkg()
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
            # R12-B5c: IV-percentile cap (VRP top bucket = -2.3 vol pts)
            iv_rank=signal.iv_rank if hasattr(signal, "iv_rank") else 0.0,
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
            # R5 audit CRITICAL: this site stored the OPTION premium as
            # entry_price while evaluate_outcomes() compares against the
            # UNDERLYING spot — a skipped $3.50 call on $620 SPY scored as a
            # +17,600% missed win. Record the underlying reference price,
            # same units as the other two record sites.
            _under_px = 0.0
            try:
                _under_px = float(signal.underlying_price or 0)
            except Exception:
                pass
            self._counterfactual.record_skip(
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                direction=signal.direction.value,
                confidence=confidence,
                entry_price=_under_px,
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
                iv_rank=request.iv_rank,  # R12-B5c: carry through revalidation
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
            # R6 (user-approved): re-sync the risk manager IMMEDIATELY so a
            # position placed earlier in THIS scan cycle counts toward the
            # cluster/count/aggregate guards for the rest of the cycle
            # (QQQ IC and IWM IC entered 5 seconds apart on 07-06 proved
            # positions were invisible until the next cycle's sync).
            try:
                await self._sync_risk_manager_positions()
            except Exception as _e:  # noqa: BLE001
                log.warning("post_execute_risk_sync_failed", error=str(_e))
            # Store full entry context for thesis re-evaluation and learning
            regime_str = regime.regime.value if regime else ""
            self._state.set_state(f"trade_regime_{trade_id}", regime_str)
            # Persist TOTAL capital-at-risk for this position = per-contract
            # max_loss x the executed contract count. (See _position_capital_at_risk.)
            _total_risk = self._position_capital_at_risk(signal, adjusted_size)
            self._state.set_state(
                f"trade_maxloss_{trade_id}",
                str(_total_risk),
            )
            # R7: capital-at-risk RETAINED on the trade row (the KV above is
            # deleted on close) so PF-per-unit-risk and DD-vs-deployed-risk
            # stay reconstructable for the go-live verdict.
            try:
                self._state.set_trade_capital_at_risk(trade_id, _total_risk)
            except Exception:  # noqa: BLE001
                pass
            # trade-life-entry-vix-refetch-defeats-lkg (R22): this line was
            # `vix=await self._market_data.get_vix() or 0` — a SECOND,
            # independent network fetch (get_vix is uncached) seconds after
            # the validated one, with `current_vix` sitting unused in scope,
            # and the exact `or 0` pattern R7 removed from the risk path. One
            # transient hiccup wrote entry_vix=0 PERMANENTLY: the row poisons
            # trade_context/meta-label training as a false zero for a real
            # feature, and _check_thesis_valid's `entry_vix > 0` guard
            # disarms the vix_spike exit for that position's whole life.
            # Reuse the value the risk gates already validated; fall back to
            # the same last-known-good cache _get_vix_lkg maintains; write
            # NULL (None) if genuinely nothing is known — never a fake 0.
            _entry_vix = current_vix
            if _entry_vix is None:
                _lkg_ctx = getattr(self, "_vix_lkg", None)
                if _lkg_ctx and (time.time() - _lkg_ctx[1]) <= 2700:
                    _entry_vix = float(_lkg_ctx[0])
            if _entry_vix is None:
                log.warning("entry_vix_unknown_stored_null", trade_id=trade_id)
            self._state.save_trade_context(
                trade_id=trade_id,
                direction=signal.direction.value,
                confidence=confidence,
                regime=regime_str,
                vix=_entry_vix,
                iv_rank=signal.iv_rank if hasattr(signal, "iv_rank") else 0,
                sentiment_score=sentiment.composite_score if sentiment and hasattr(sentiment, "composite_score") else 0,
                # R20: the 11 technical META_FEATURES, captured at entry. This
                # column was "{}" on every trade ever taken — the meta-labeler
                # could never train (9/20 features) and every close without it
                # is training data lost. Snapshot comes from this scan cycle's
                # feature frame (stashed in _scan_symbol minutes earlier).
                signals=self._entry_signals_json(signal.symbol),
                model_version=getattr(self._predictor, "model_version", "") or "",
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

        # R12-A: refuse to build an exit for anything not FILLED/PARTIAL —
        # a CLOSED/CANCELLED row must never be resurrected into the exit
        # path (F4.5). Checked BEFORE any order is placed.
        if trade.status not in (TradeStatus.FILLED, TradeStatus.PARTIAL):
            log.warning("exit_refused_bad_status",
                        trade_id=pos.trade_id, status=str(trade.status))
            return

        if not await self._ibkr.ensure_connected():
            log.error("exit_failed_no_connection", trade_id=pos.trade_id)
            self._watchdog.record_error("ibkr", "disconnected during exit")
            return

        # R13 (human-factors #2) / R14: the position must still exist at the
        # broker, WHOLE, before we reverse it. Nothing here used to check:
        # after a manual TWS flatten (or any close the bot missed) the monitor
        # still demanded the exit, and the reverse combo REBUILT the position
        # inverted — the 07-13 incident's end-state, reachable by an operator
        # with no bug at all.
        #
        # "unknown" (wedged feed, unkeyable shape) must always PROCEED: a data
        # problem must never silently disable a stop.
        try:
            liveness = await self._reconciler.position_liveness(trade)
        except Exception as _e:  # noqa: BLE001 — never let the check kill an exit
            log.warning("exit_liveness_check_failed",
                        trade_id=pos.trade_id, error=str(_e))
            liveness = "unknown"

        if liveness == "gone":
            # Confirmed absent against a FRESH broker query. Refuse the
            # reverse combo — and BOOK the trade here. reconcile()'s
            # stale-local loop cannot: its zero-options guard fires on exactly
            # this state (the last open position manually flattened), so
            # without this the row would sit FILLED forever, re-demanding an
            # exit that is refused every pass.
            log.critical(
                "exit_refused_position_not_at_broker",
                trade_id=pos.trade_id, symbol=pos.symbol,
                reason=pos.exit_reason,
                note="legs confirmed absent at IBKR (manual close? missed "
                     "fill?) — reversing would OPEN an inverse position",
            )
            booked = await self._reconciler.book_vanished_trade(trade)
            if self._alert_gate(f"exit_refused:{pos.trade_id}"):
                _tail = ("Booked closed from broker records."
                         if booked else
                         "NOTE: booking it failed — the row needs manual review.")
                await self._send_notification(
                    f"EXIT REFUSED (position not at broker): {pos.symbol} "
                    f"{trade.strategy} — the bot wanted to exit but its legs "
                    f"are gone at IBKR (confirmed by a fresh query). No order "
                    f"placed. {_tail}"
                )
            return

        if liveness == "partial":
            # SOME legs gone (manual half-flatten, assignment). The reverse
            # combo reverses ALL stored legs, so the already-flat ones would be
            # OPENED as new, inverted, unmanaged positions — a naked short in
            # the worst case. That is the very incident class this gate exists
            # to prevent, so refuse and escalate: we never auto-close half a
            # structure (reconcile() takes the same line).
            log.critical(
                "exit_refused_partial_structure",
                trade_id=pos.trade_id, symbol=pos.symbol,
                reason=pos.exit_reason,
                note="only SOME legs live at IBKR — a full reverse combo would "
                     "OPEN inverted positions on the missing legs; needs a human",
            )
            if self._alert_gate(f"exit_partial:{pos.trade_id}"):
                await self._send_notification(
                    f"EXIT REFUSED (partial structure): {pos.symbol} "
                    f"{trade.strategy} — only SOME legs are still at IBKR "
                    f"(manual close? assignment?). Reversing all legs would "
                    f"open new inverted positions, so no order was placed. "
                    f"This needs you: close the remainder at IBKR."
                )
            return

        # R14 #9: reject backoff. A trade only re-reaches this method after its
        # PRIOR exit order died — while an exit is live the trade sits in
        # closing_ids and the monitor skips it. So a re-request IS the reject
        # signal: the last close was rejected/cancelled (illiquid, halted, a
        # crossed/again-stale quote) and reverted CLOSING->FILLED. Left alone
        # the monitor re-fires it every 30s forever, unbounded, paging nobody.
        # Throttle with an escalating backoff and escalate to a page.
        if not self._exit_retry_ready(pos.trade_id):
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
                exit_trade = await self._close_single_leg(
                    trade, qualified, close_action)
                if exit_trade is None:
                    # Deferred (short buyback with no quote — see helper). Leave
                    # the trade FILLED so the monitor retries next cycle.
                    return

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
            # R12-A: CAS — the last blind status write; only FILLED/PARTIAL
            # positions may move to CLOSING (a CLOSED/CANCELLED row must not
            # be resurrected into the exit path).
            if not self._state.update_trade_status(
                    pos.trade_id, TradeStatus.CLOSING,
                    from_statuses=(TradeStatus.FILLED, TradeStatus.PARTIAL)):
                # Race window only (pre-check passed above): another path
                # closed the trade between order placement and here. A live
                # close order now exists on a closed position — cancel it.
                log.critical("closing_transition_refused_cancelling_order",
                             trade_id=pos.trade_id)
                try:
                    if exit_trade is not None and getattr(exit_trade, "order", None):
                        self._ibkr.ib.cancelOrder(exit_trade.order)
                except Exception as _e:  # noqa: BLE001
                    log.error("race_exit_cancel_failed",
                              trade_id=pos.trade_id, error=str(_e))
                return

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

    @staticmethod
    def _defined_risk_width(legs: list[dict]) -> float | None:
        """R13: max vertical width (per share) of a defined-risk structure —
        the wider wing of an iron condor, or a vertical's strike distance.
        This is the STRUCTURAL bound on what closing a credit position can
        ever be worth: the wings cap the loss, so paying more than the width
        to buy it back is, by construction, a bad quote rather than a real
        price. None when the shape isn't a clean vertical set (straddles,
        strangles) — then there is no structural bound to enforce."""
        by_right: dict[str, list[float]] = {}
        for leg in legs:
            try:
                right = str(leg["right"]).upper()[:1]
                by_right.setdefault(right, []).append(float(leg["strike"]))
            except (KeyError, TypeError, ValueError):
                return None
        widths = []
        for strikes in by_right.values():
            if len(strikes) != 2:
                return None  # not a vertical pair on this side
            widths.append(abs(strikes[0] - strikes[1]))
        return max(widths) if widths else None

    async def _option_nbbo(self, qualified) -> tuple[float | None, float | None]:
        """(bid, ask) for a single qualified option, or (None, None). Mirrors
        the req/cancel pairing discipline of the combo path: a leaked
        snapshot=False subscription streams (and, if unentitled, errors 10091)
        forever and eventually crashes the ib_insync thread."""
        try:
            self._ibkr.ib.reqMktData(qualified, "", False, False)
            ticker = None
            try:
                await asyncio.sleep(0.5)
                ticker = self._ibkr.ib.ticker(qualified)
            finally:
                try:
                    self._ibkr.ib.cancelMktData(qualified)
                except Exception:
                    pass
            if ticker:
                import math
                # R18: same None-safety as the combo path — a single-leg
                # buyback with a one-sided quote must not raise here.
                bid = (ticker.bid if ticker.bid is not None
                       and not math.isnan(ticker.bid) and ticker.bid > 0 else None)
                ask = (ticker.ask if ticker.ask is not None
                       and not math.isnan(ticker.ask) and ticker.ask > 0 else None)
                return bid, ask
        except Exception as e:  # noqa: BLE001
            log.warning("single_leg_nbbo_failed", error=str(e))
        return None, None

    async def _close_single_leg(self, trade: TradeRecord, qualified, close_action: str):
        """Close a single-leg option with a PRICE-BOUNDED order.

        R14 item 3b: the single-leg exit path placed a raw MARKET order. For a
        SELL-to-close (a LONG option) that is bounded — the worst fill is $0
        and the premium is already sunk — so a market fallback is acceptable.
        For a BUY-to-close (a SHORT buyback) it is the same unbounded-fill
        catastrophe the multi-leg path was hardened against: a short CALL bought
        back at market in a no-quote tape has NO ceiling. So a buyback is a
        LIMIT, never a market order, and:
          - short PUT: capped at the strike — the most it can ever be worth is
            full intrinsic (underlying -> 0), so paying more is a bad quote;
          - short CALL: no structural cap exists, so price at the marketable ask
            plus the cross; a genuinely high ask is a genuinely expensive
            buyback, but the LIMIT means a lone garbage print can't fill beyond
            it. No quote at all -> defer + page rather than market a short.
        """
        EXIT_CROSS = self._settings.exit.exit_cross_amount
        bid, ask = await self._option_nbbo(qualified)

        if close_action == "SELL":
            # Closing a long option: receive credit. Marketable limit at the
            # bid; market fallback is bounded (>= 0), so it is acceptable.
            if bid is not None:
                limit_price = max(0.01, round(bid - EXIT_CROSS, 2))
                order = OrderBuilder.limit(action="SELL", quantity=trade.quantity,
                                           limit_price=limit_price)
            elif ask is not None:
                limit_price = max(0.01, round(ask - EXIT_CROSS, 2))
                order = OrderBuilder.limit(action="SELL", quantity=trade.quantity,
                                           limit_price=limit_price)
            else:
                log.warning("single_leg_sell_no_quote_market_fallback",
                            trade_id=trade.trade_id, symbol=trade.symbol,
                            note="long-option close; market fill is bounded at $0")
                order = OrderBuilder.market(action="SELL", quantity=trade.quantity)
            # R17: tag with trade_id for exact reconciler matching.
            order.orderRef = trade.trade_id
            return await self._ibkr.place_order(qualified, order)

        # BUY-to-close a short: unbounded risk if done at market. LIMIT only.
        is_put = trade.contract_type == "put"
        if ask is not None:
            limit_price = round(ask + EXIT_CROSS, 2)
        elif bid is not None:
            limit_price = round(bid + EXIT_CROSS, 2)
        else:
            # No quote on a SHORT buyback: never market it. Defer + page.
            log.critical(
                "single_leg_buyback_deferred_no_quote",
                trade_id=trade.trade_id, symbol=trade.symbol,
                strategy=trade.strategy,
                note="short-option buyback with no NBBO — refusing a market "
                     "order (unbounded fill on a short); will retry next cycle",
            )
            if self._alert_gate(f"exit_single_noquote:{trade.trade_id}"):
                await self._send_notification(
                    f"EXIT DEFERRED (no quotes): {trade.symbol} {trade.strategy} "
                    f"— short-option buyback with no NBBO. Refusing to market "
                    f"order a short (unbounded). Retrying; check the data feed."
                )
            return None

        # Cap a short PUT at its strike — full intrinsic is the ceiling on what
        # closing it can ever cost. A short CALL has no such structural cap.
        if is_put and trade.strike:
            cap = round(float(trade.strike), 2)
            if limit_price > cap:
                log.warning("single_leg_buyback_capped_at_strike",
                            trade_id=trade.trade_id, symbol=trade.symbol,
                            quoted_limit=limit_price, strike=cap)
                limit_price = cap

        log.info("single_leg_buyback_limit", trade_id=trade.trade_id,
                 symbol=trade.symbol, limit_price=limit_price)
        order = OrderBuilder.limit(action="BUY", quantity=trade.quantity,
                                   limit_price=limit_price)
        # R17: tag with trade_id for exact reconciler matching.
        order.orderRef = trade.trade_id
        return await self._ibkr.place_order(qualified, order)

    def _entry_signals_json(self, symbol: str) -> str:
        """R20: serialize the 11 technical META_FEATURES for trade_context.

        Values come from the feature row stashed by _scan_symbol this cycle
        (same frame the ML gates read), plus hour_of_day stamped here. Only
        finite numerics are written; a missing/stale snapshot degrades to
        "{}" exactly as before — never blocks an entry.
        """
        import json as _json
        import math as _math
        try:
            row = (getattr(self, "_entry_feature_snap", None) or {}).get(symbol)
            if row is None:
                return "{}"
            from ait.ml.meta_label import META_FEATURES
            out: dict[str, float] = {}
            for name in META_FEATURES:
                if name == "hour_of_day":
                    out[name] = float(datetime.now().hour)
                    continue
                if name in getattr(row, "index", ()):
                    try:
                        v = float(row[name])
                        if _math.isfinite(v):
                            out[name] = round(v, 6)
                    except (TypeError, ValueError):
                        pass
            return _json.dumps(out) if out else "{}"
        except Exception:  # noqa: BLE001 — telemetry must never block an entry
            return "{}"

    def _marked_cost_to_close(self, trade: TradeRecord) -> float | None:
        """Per-share cost to close a CREDIT structure per the monitor's marks.

        unrealized = (entry_credit - current_cost) * 100 * qty, so
        current_cost = entry_credit - unrealized/(100*qty). Returns None when
        the mark is absent or stale (>15 min) — callers fall back to the
        structural wing-width bound (R16 exit-pricing mark anchor).
        """
        try:
            mark = self._state.get_position_mark(trade.trade_id)
            if not mark or not trade.entry_price or not trade.quantity:
                return None
            age = (datetime.now()
                   - datetime.fromisoformat(mark["mark_time"])).total_seconds()
            if age > 900:
                return None
            cost = float(trade.entry_price) - (
                float(mark["unrealized_pnl"]) / (100.0 * trade.quantity))
            return max(0.0, cost)
        except Exception:  # noqa: BLE001 — a broken mark must never block an exit
            return None

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
        EXIT_CROSS = self._settings.exit.exit_cross_amount
        try:
            # R16: the live-quote path was DEAD CODE. It gated the NBBO fetch
            # behind qualify_contract(combo), but a BAG cannot be qualified
            # (reqContractDetails always returns None for a synthetic combo),
            # so limit_price stayed None on EVERY multi-leg exit and each one
            # fell through to the no-quote branch — pricing at the full wing
            # width plus a CRITICAL page, on routine take-profits. A raw Bag
            # IS valid for reqMktData and for order placement (the ENTRY path
            # already relies on exactly that), so request on `combo` directly.
            quote_contract = combo
            if True:
                self._ibkr.ib.reqMktData(quote_contract, "", False, False)
                ticker = None
                try:
                    # 0.5s was also too short for a 4-leg BAG to tick; the
                    # entry path waits longer. Poll up to ~2.5s, breaking as
                    # soon as a usable quote lands.
                    import math as _m
                    for _ in range(5):
                        await asyncio.sleep(0.5)
                        ticker = self._ibkr.ib.ticker(quote_contract)
                        if ticker and (
                                (ticker.bid is not None and not _m.isnan(ticker.bid)
                                 and ticker.bid != 0)
                                or (ticker.ask is not None and not _m.isnan(ticker.ask)
                                    and ticker.ask != 0)):
                            break
                finally:
                    # ALWAYS cancel the streaming subscription. A leaked
                    # snapshot=False sub streams (and, if unentitled, errors
                    # 10091) forever; repeated exit-pricing attempts pile these
                    # up into a market-data flood that crashes the ib_insync
                    # message thread (native access violation). Cancelling here
                    # keeps req/cancel paired 1:1.
                    try:
                        self._ibkr.ib.cancelMktData(quote_contract)
                    except Exception:
                        pass
                if ticker:
                    import math
                    # Combo quotes are SIGNED: closing a debit position
                    # (reversed legs = net sell) quotes NEGATIVE — we receive
                    # credit. Only 0/NaN means "no quote".
                    # R18: `is not None` FIRST — the poll loop above breaks on a
                    # one-sided quote, and math.isnan(None) raises TypeError,
                    # which silently reverted the R16 fix (exit priced at the
                    # full wing width + a CRITICAL page on a routine close).
                    bid = (ticker.bid if ticker.bid is not None
                           and not math.isnan(ticker.bid) and ticker.bid != 0 else None)
                    ask = (ticker.ask if ticker.ask is not None
                           and not math.isnan(ticker.ask) and ticker.ask != 0 else None)
                    # An EXIT must FILL — a take-profit/stop that sits unfilled
                    # lets a winner reverse or a loss deepen. Price at the
                    # MARKETABLE side and cross the spread by a buffer, instead
                    # of the mid (which never crosses, so the order timed out
                    # and re-placed every 30s forever — the IWM strangle sat
                    # on a +54% gain it couldn't take). BUY pays up to the ask,
                    # SELL accepts down to the bid.
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

        # R13 (market-catastrophe lens): the exit had NO price sanity bound —
        # `ask + cross` with no ceiling, and a MARKET order on a 4-leg BAG
        # exactly when quotes evaporate (executed proof: a $2-wide condor with
        # a garbage 9.90 ask placed BUY LMT 10.00 — 5x the wing width, i.e.
        # 5x the structural max loss). The entry's GOV-1 fat-finger guard
        # never covered this path.
        is_credit = trade.strategy in CREDIT_STRATEGIES
        width = self._defined_risk_width(legs)

        # A credit buyback (positive as-defined price = cost to close) can
        # never rationally exceed the wing width — that IS the max loss the
        # structure was sold to cap. Cap AT the width, not below it: a
        # deep-ITM condor is legitimately worth ~width, and a cap of
        # width-0.01 would be non-marketable exactly then, leaving the order
        # to re-place forever while the ITM short carries assignment risk.
        if is_credit and width and limit_price is not None and limit_price > width:
            cap = round(width, 2)
            log.warning(
                "combo_exit_limit_capped_at_wing_width",
                symbol=trade.symbol, trade_id=trade.trade_id,
                quoted_limit=limit_price, wing_width=width, capped_to=cap,
                note="quote implies paying more than the structural max "
                     "loss to close — capping at the wings",
            )
            limit_price = cap

        # R16: MARK-ANCHORED bound alongside the structural one. The width cap
        # was calibrated for $2-5 wings; at the promoted $30-35 wings R13's own
        # worked incident (garbage 9.90 ask -> BUY LMT 10.00 on a structure
        # fairly worth ~$1) passes UNCAPPED. Anchor to the monitor's persisted
        # leg-mark cost-to-close: never pay more than max(2x the marked cost,
        # entry credit + 25% of the wings) unless the mark is stale/absent —
        # then the structural cap is all we have (unchanged behavior).
        mark_cost = self._marked_cost_to_close(trade)
        if is_credit and width and limit_price is not None and mark_cost is not None:
            anchor = round(min(width, max(2.0 * mark_cost,
                                          float(trade.entry_price or 0) + 0.25 * width)), 2)
            if limit_price > anchor:
                log.warning(
                    "combo_exit_limit_capped_at_mark_anchor",
                    symbol=trade.symbol, trade_id=trade.trade_id,
                    quoted_limit=limit_price, mark_cost=round(mark_cost, 2),
                    wing_width=width, capped_to=anchor,
                )
                limit_price = anchor

        # The mirror bound, which the credit-only cap above missed entirely:
        # closing a DEBIT structure SELLS what we own, so the as-defined quote
        # is NEGATIVE (we receive credit) and the limit can only be positive by
        # the spread-crossing buffer on a near-worthless structure. A positive
        # limit beyond that means the feed handed us a sign-corrupted quote —
        # the same garbage-quote class as the condor incident — and placing it
        # would PAY, uncapped, to dispose of a long position. You never pay
        # more than the cross to get out of something you own.
        if not is_credit and limit_price is not None and limit_price > EXIT_CROSS:
            cap = round(EXIT_CROSS, 2)
            log.warning(
                "combo_exit_limit_capped_debit_close",
                symbol=trade.symbol, trade_id=trade.trade_id,
                quoted_limit=limit_price, capped_to=cap,
                note="closing a long structure should RECEIVE credit; a "
                     "positive limit beyond the cross implies a bad quote",
            )
            limit_price = cap

        # NOTE: `is not None`, NOT truthiness. A computed limit of exactly
        # 0.00 is a legitimate, marketable price for a debit close (a dying
        # straddle bid at the cross amount quotes ask=-0.10, +0.10 cross = 0.00
        # — BUY LMT 0.00 still fills against any negative offer). Treating it
        # as "no quote" deferred a perfectly closeable position indefinitely.
        if limit_price is not None:
            log.info("combo_exit_limit", symbol=trade.symbol, action=combo_action,
                     limit_price=limit_price)
            order = OrderBuilder.combo_limit(
                action=combo_action, quantity=trade.quantity, limit_price=limit_price
            )
        elif is_credit and width:
            # No usable quote on a credit structure. A MARKET order here is
            # the worst possible choice (unbounded fill on 4 illiquid legs in
            # exactly the tape where marks vanished). R16: with $30-35 wings,
            # pricing AT the width means offering ~$3,000-3,500/lot for a
            # structure whose marked cost may be ~$300 — so anchor to the
            # persisted leg marks when fresh; the raw width is the LAST
            # resort (marks gone too), unchanged from R13.
            if mark_cost is not None:
                fallback = round(min(width, max(2.0 * mark_cost,
                                                float(trade.entry_price or 0) + 0.25 * width)), 2)
            else:
                fallback = round(width, 2)
            log.critical(
                "combo_exit_no_quote_pricing_at_wing_width",
                symbol=trade.symbol, trade_id=trade.trade_id,
                wing_width=width, mark_cost=mark_cost, limit_price=fallback,
            )
            if self._alert_gate(f"exit_noquote:{trade.trade_id}"):
                await self._send_notification(
                    f"EXIT WITHOUT QUOTES: {trade.symbol} {trade.strategy} — "
                    f"no combo NBBO, closing at the wing width "
                    f"(${fallback:.2f}, the structural max loss). Check data "
                    f"feed."
                )
            order = OrderBuilder.combo_limit(
                action=combo_action, quantity=trade.quantity, limit_price=fallback
            )
        else:
            # No quote and no structural bound (debit shapes, or credit
            # strangles whose loss ISN'T capped): do NOT dump a multi-leg BAG
            # at market. For debit shapes deferral is strictly safer (loss
            # already capped at premium paid). For an unbounded credit shape
            # both options are bad — a blind market BAG in a no-quote tape is
            # the known catastrophe (unbounded FILL price), deferral keeps
            # unbounded MARKET exposure for one more cycle. We defer: the
            # monitor re-demands the exit every pass, and the operator is
            # paged below. Retry next cycle.
            log.critical(
                "combo_exit_deferred_no_quote",
                symbol=trade.symbol, trade_id=trade.trade_id,
                strategy=trade.strategy,
                note="no combo quote and no structural bound — refusing a "
                     "market order on a multi-leg BAG; will retry next cycle",
            )
            if self._alert_gate(f"exit_deferred:{trade.trade_id}"):
                await self._send_notification(
                    f"EXIT DEFERRED (no quotes): {trade.symbol} "
                    f"{trade.strategy} — refusing to market-order a multi-leg "
                    f"combo blind. Retrying every cycle. If this repeats, "
                    f"check the market-data feed."
                )
            return None

        # R17: tag with trade_id for exact reconciler matching.
        order.orderRef = trade.trade_id
        return await self._ibkr.place_order(combo, order)

    async def _post_market(self) -> None:
        """Post-market reconciliation, learning, and reporting."""
        log.info("post_market_starting")

        # R16: sweep orphaned per-trade risk keys. delete_state only runs on
        # the _process_completed_exits path, so trades that CANCELLED, were
        # manually flattened, or were restated leave trade_maxloss_* behind —
        # 28 such keys were found in the 07-07 forensics, and each one
        # permanently inflates the aggregate capital-at-risk denominator,
        # silently shrinking how much the bot is allowed to deploy.
        try:
            _open_ids = {t.trade_id for t in self._state.get_open_trades()}
            _swept = 0
            for _k in self._state.state_keys_like("trade_maxloss_%"):
                if _k[len("trade_maxloss_"):] not in _open_ids:
                    self._state.delete_state(_k)
                    _swept += 1
            if _swept:
                log.info("orphan_maxloss_keys_swept", count=_swept)
        except Exception as _e:  # noqa: BLE001
            log.warning("maxloss_key_sweep_failed", error=str(_e))

        # R16: settle deferred commission true-ups now that the day's
        # commissionReports have all landed (booking-time ledger was partial).
        for _tid in self._state.pending_trueup_trade_ids():
            try:
                _t = self._find_trade_by_id(_tid)
                _real = self._state.total_commission(_tid)
                if _t is not None and _real > 0 and _t.realized_pnl is not None:
                    from ait.execution.executor import TradeExecutor as _TE
                    _delta = round(_TE.commission_estimate(_t) - _real, 2)
                    if abs(_delta) >= 0.01:
                        _new_pnl = round(float(_t.realized_pnl) + _delta, 2)
                        self._state.update_trade_realized_pnl(_tid, _new_pnl)
                        _st = self._state.get_daily_stats()
                        _st.total_pnl = round(_st.total_pnl + _delta, 2)
                        self._state.update_daily_stats(_st)
                        log.info("commission_trueup_settled_post_market",
                                 trade_id=_tid, delta=_delta)
                self._state.delete_state(f"trueup_pending_{_tid}")
            except Exception as _e:  # noqa: BLE001
                log.warning("post_market_trueup_failed", trade_id=_tid,
                            error=str(_e))

        # 1. Reconcile with IBKR
        recon = await self._reconciler.reconcile()

        # R12-A (F4.1): re-adopt working exit orders the reconciler found
        # for kept-CLOSING trades — without a tracker, a restart-surviving
        # exit order is invisible and a duplicate close (position
        # REVERSAL) follows.
        for _tid, _oid in getattr(recon, "closing_exit_orders", {}).items():
            try:
                self._executor.adopt_exit_order(
                    _oid, _tid, reason="reconcile_readopt")
            except Exception as _e:  # noqa: BLE001
                log.warning("exit_order_adopt_failed",
                            trade_id=_tid, error=str(_e))
        await self._alert_reconcile_anomalies(recon)

        # GOV-4 (governance audit): EOD BREAK REPORT — reconcile used to run
        # startup/pre-market only and breaks were log lines nobody read. One
        # Telegram line per day: books-vs-broker status + NLV + the book's
        # -10% gap stress (a number that previously existed NOWHERE).
        try:
            _nlv = await self._account.get_net_liquidation()
            _stress = 0.0
            for _t in self._state.get_open_trades():
                if _t.status.value not in ("filled", "partial"):
                    continue
                _spot = await self._market_data.get_current_price(_t.symbol)
                if not _spot:
                    continue
                _settle = _spot * 0.90
                _iv = self._reconciler._structure_intrinsic(_t, _settle)
                if _iv is None:
                    continue
                from ait.strategies.base import CREDIT_STRATEGIES as _CS
                if _t.strategy in _CS:
                    _stress += (_t.entry_price - _iv) * 100 * _t.quantity
                else:
                    _stress += (_iv - _t.entry_price) * 100 * _t.quantity
            _breaks = len(recon.discrepancies or [])
            await self._send_notification(
                f"EOD RECON: {'CLEAN' if _breaks == 0 else f'{_breaks} BREAK(S)'} | "
                f"matched {recon.matched} | NLV ${_nlv:,.0f} | "
                f"book -10% gap stress: ${_stress:,.0f}"
                + ("" if _breaks == 0 else "\n" + "\n".join((recon.discrepancies or [])[:3]))
            )
        except Exception as _e:  # noqa: BLE001
            log.warning("eod_break_report_failed", error=str(_e))

        # 2. Run self-learning cycle
        if self._settings.learning.enabled:
            learning_result = self._learning.run_learning_cycle(
                lookback_days=self._settings.learning.lookback_days
            )
            log.info("post_market_learning", result=learning_result)

        # 3. Evaluate counterfactual outcomes (what would skipped trades have done?)
        # R12-C: counterfactual evaluation loop removed (analysis retired;
        # record_skip rows remain the taken-vs-vetoed record).
        drift_report = self._trainer.drift_detector.check_drift()
        if drift_report.is_drifting:
            log.warning("post_market_drift", accuracy=f"{drift_report.accuracy:.2%}", reason=drift_report.reason)

        # 5. Generate analytics
        metrics = self._analytics.get_performance(lookback_days=30)

        # 6. Generate daily summary
        summary = await self._portfolio.get_portfolio_summary()
        stats = self._state.get_daily_stats()
        health = self._watchdog.get_health()

        # A1-followup (R14 #8): PRE-STAMP tomorrow's MTM baseline with the PRIOR
        # CLOSE's unrealized P&L. The mark-to-market daily-loss brake computes
        # mtm_day = realized + (unrealized_now - unrealized_at_SOD); its SOD
        # baseline used to be captured lazily on the first fast-monitor tick of
        # the day — which runs AFTER the open, so on a -8% gap the already-
        # collapsed unrealized became the baseline and mtm_day registered ~0.
        # The brake was blind to exactly the overnight gap risk it exists to
        # catch. Stamp the close's figure under the NEXT trading day's key; the
        # monitor's lazy capture only writes when the key is empty, so a
        # pre-stamped value wins and the baseline becomes the prior close (a
        # genuine gap then shows up as the full unrealized move from it).
        self._prestamp_mtm_baseline(float(summary["total_unrealized_pnl"]))

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
        # R12-C: counterfactual analysis section removed from the report

        await self._send_notification(report)
        log.info("post_market_complete", summary=summary, metrics=vars(metrics))

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("orchestrator_shutting_down")
        health = self._watchdog.get_summary()
        await self._send_notification(f"Bot shutting down\n\n{health}")
        # R16: drain in-flight notification tasks. The shutdown page is
        # fire-and-forget like every other alert, so the loop used to close
        # out from under it — the one message that says "the bot is gone" was
        # the message most likely to be lost.
        _tasks = getattr(self, "_notify_tasks", None)
        if _tasks:
            try:
                await asyncio.wait(set(_tasks), timeout=15)
            except Exception as _e:  # noqa: BLE001
                log.warning("notify_drain_failed", error=str(_e))




    async def _check_thesis_valid(self, pos: PositionStatus) -> tuple[bool, str]:
        """Re-evaluate whether the original trade thesis still holds.

        Returns (invalidated: bool, reason: str).
        Checks: direction flip, regime shift, VIX spike.
        """
        try:
            # R16 (2026-08-07): defined-risk NEUTRAL credit structures are
            # EXEMPT from the whole thesis check. A condor profits in either
            # direction inside its range, so a direction "flip" is meaningless;
            # and the vix_spike branch here was the last remaining path that
            # force-flattened condors into events — undoing the user-approved
            # hold-through decision (rule-3d exemption) at maximum IV expansion.
            # Condor exits are TP/touch-stop/DTE only (R6 evidence: measured
            # best policy). Directional strategies keep the full check.
            if getattr(pos, "strategy", "") in ("iron_condor", "iron_butterfly"):
                return False, ""

            context = self._state.get_trade_context(pos.trade_id)
            if not context:
                return False, ""

            entry_direction = context.get("entry_direction", "")
            entry_regime = context.get("entry_regime", "")
            entry_vix = context.get("entry_vix", 0)

            # 1. Re-run ML prediction on fresh data (no market_context here — lightweight check)
            # R16: 60 days left vol_ratio/vol_mean_reversion/momentum_divergence
            # all-NaN (they need a 60-bar rolling window + warmup) — the zero-fill
            # fed the model systematically distorted inputs. 180 covers warmup.
            hist = await self._market_data.get_historical(pos.symbol, days=180)
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
                    "expiry": trade.expiry,  # R12-B5a: per-expiry cap
                    "quantity": trade.quantity,
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                    # R16: fall back to the trade row's capital_at_risk when the
                    # per-trade KV is missing (it leaks on any non-exit close
                    # path). Reporting 0.0 made every reader — aggregate cap,
                    # concentration, digests — treat the position as risk-free.
                    # R18: converted defensively. A non-numeric KV used to raise
                    # ValueError out of the whole loop, so update_positions() was
                    # never called and EVERY risk cap then validated against a
                    # STALE position set — a silent, unbounded failure.
                    "max_loss": self._position_max_loss(trade, ml),
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

    @staticmethod
    def _position_max_loss(trade, ml_kv: str) -> float:
        """R18: per-position max-loss resolution that CANNOT raise.

        Order: the trade_maxloss_* KV, else the trade row's capital_at_risk,
        else 0.0. A malformed KV degrades this ONE position to its row value
        instead of aborting the entire risk sync (which left the risk manager
        holding a stale book while every cap kept 'passing').
        """
        for candidate in (ml_kv, getattr(trade, "capital_at_risk", None)):
            if candidate in (None, ""):
                continue
            try:
                return abs(float(candidate))
            except (TypeError, ValueError):
                log.warning("position_max_loss_unparseable",
                            trade_id=getattr(trade, "trade_id", "?"),
                            value=str(candidate)[:40])
        return 0.0

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

    @staticmethod
    def _sessions_until(calendar_days: int | None) -> int | None:
        """R16: convert calendar days-to-event into TRADING sessions.

        days_until_next_event() counts calendar days, so a Friday entry with a
        Monday event reads 3 and clears a 1-day blackout despite there being
        NO session in between — the gate could never see the weekend. Counts
        weekdays strictly between today and the event (holidays ignored:
        erring toward MORE sessions is the fail-safe direction for a gate that
        blocks entries).
        """
        if calendar_days is None:
            return None
        today = datetime.now().date()
        sessions = 0
        for i in range(1, max(1, calendar_days) + 1):
            if (today + timedelta(days=i)).weekday() < 5:
                sessions += 1
        return max(0, sessions - 1)  # exclude the event day itself

    def _persist_daily_iv(self, symbol: str, chains) -> None:
        """R16: write today's chain ATM IV into the daily IV store (once per
        symbol per day) so iv_rank never runs on a frozen series again."""
        try:
            today = datetime.now().date().isoformat()
            marker = f"iv_saved_{symbol}"
            if self._state.get_state(marker, "") == today:
                return
            # R18: `atm_iv` is a dataclass FIELD (float), not a method. Calling
            # it raised TypeError into this function's own `except`, so the R16
            # "self-healing IV store" never wrote a single row — proven by the
            # store still ending 2026-07-09 for every symbol. Consequence: the
            # freshness gate in _estimate_iv_rank kept finding a stale series,
            # so EVERY iv_rank silently fell back to the realized-vol proxy —
            # the exact failure R16 added this function to fix.
            ivs = [c.atm_iv for c in chains if getattr(c, "atm_iv", None)]
            ivs = [v for v in ivs if v and v > 0.005]
            if not ivs:
                return
            import pandas as pd
            atm = sorted(ivs)[len(ivs) // 2]  # median across expiries
            n = self._historical.save_daily_iv(
                symbol, pd.Series([atm], index=[pd.Timestamp(today)]))
            if n:
                self._state.set_state(marker, today)
                log.info("daily_iv_persisted", symbol=symbol,
                         atm_iv=round(atm, 4))
        except Exception as e:  # noqa: BLE001 — never let bookkeeping kill a scan
            log.debug("daily_iv_persist_failed", symbol=symbol, error=str(e))

    async def _estimate_iv_rank(self, symbol: str) -> float | None:
        """IV rank (0-100), or None when it cannot be measured.

        R7: TRUE percentile of today's implied vol in the
        stored daily IV series when >=60 IV observations exist; otherwise the
        old realized-vol min-max proxy, tagged in the logs. Sell-when-IV-is-
        rich is THE premium-seller signal — the proxy measured realized, not
        implied, and the two diverge exactly when the edge is largest.

        fail-direction-05 (R22): every failure path here used to return a
        fabricated 50.0 — the one value that passes BOTH IV gates — and the
        proxy's own get_historical call sat outside any try, so its exception
        was laundered into 50.0 by _scan_symbol's gather. The triggers are
        normal-op (historical.db locked/stale/corrupt, a yfinance outage),
        i.e. the bot sold premium as if IV were average during exactly the
        outages when IV was unknown, and stored 50 as a measurement. Unknown
        is now None; the caller fails closed for new entries and the journal
        records NULL instead of a fabricated rank.
        """
        try:
            import sqlite3 as _sq
            con = _sq.connect("file:data/historical.db?mode=ro", uri=True)
            pairs = con.execute(
                "SELECT date, implied_vol FROM daily_prices WHERE symbol=? "
                "AND implied_vol IS NOT NULL AND implied_vol > 0 "
                "ORDER BY date DESC LIMIT 252", (symbol,)).fetchall()
            con.close()
            rows = [p[1] for p in pairs]
            # R16: FRESHNESS GATE — the series froze on 07-09 and every gate
            # ran a month on the stale snapshot, unnoticed. A stale head means
            # "cur" is not today's IV: fall through to the realized proxy
            # (tagged in logs) rather than serve a fictitious percentile.
            if pairs:
                _age_days = (datetime.now().date()
                             - datetime.fromisoformat(pairs[0][0][:10]).date()).days
                if _age_days > 5:
                    log.warning("iv_rank_store_stale", symbol=symbol,
                                latest=pairs[0][0], age_days=_age_days,
                                note="using realized-vol proxy; store should "
                                     "self-heal via daily_iv_persisted")
                    rows = []
            if len(rows) >= 60:
                cur = rows[0]
                below = sum(1 for v in rows if v < cur)
                rank = below / len(rows) * 100
                log.debug("iv_rank_true_percentile", symbol=symbol,
                          rank=round(rank, 1), n=len(rows))
                return max(0.0, min(100.0, rank))
        except Exception as _e:  # noqa: BLE001
            log.debug("iv_rank_store_read_failed", symbol=symbol, error=str(_e))
        log.debug("iv_rank_realized_proxy", symbol=symbol,
                  note="stored IV series <60 obs — run IV backfill")
        # fail-direction-05: this fetch used to sit OUTSIDE any try, so its
        # exception reached the caller's asyncio.gather and became 50.0 there.
        # Own the failure at the source and report it honestly as "unknown".
        try:
            hist = await self._market_data.get_historical(symbol, days=252)
        except Exception as _e:  # noqa: BLE001
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason=f"proxy_history_raised:{type(_e).__name__}",
                        detail=str(_e)[:160])
            return None
        if hist is None or len(hist) < 60:
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason="proxy_history_too_short",
                        rows=0 if hist is None else len(hist))
            return None

        try:
            import numpy as np
            close = hist["Close"]
            log_returns = np.log(close / close.shift(1)).dropna()

            current_vol = float(log_returns.tail(20).std() * np.sqrt(252))
            rolling_vol = log_returns.rolling(20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
        except Exception as _e:  # noqa: BLE001
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason=f"proxy_compute_raised:{type(_e).__name__}",
                        detail=str(_e)[:160])
            return None

        if len(rolling_vol) < 2:
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason="proxy_vol_history_too_short",
                        rows=len(rolling_vol))
            return None

        vol_min = float(rolling_vol.min())
        vol_max = float(rolling_vol.max())
        vol_range = vol_max - vol_min

        if vol_range <= 0:
            # A flat vol series is a FROZEN/degenerate feed, not "average IV".
            # Returning 50 here was the same fabrication as the exception
            # path — it passed both IV gates on data that says nothing.
            log.warning("iv_rank_unavailable", symbol=symbol,
                        reason="proxy_degenerate_vol_range",
                        vol_min=round(vol_min, 6), vol_max=round(vol_max, 6))
            return None

        iv_rank = ((current_vol - vol_min) / vol_range) * 100
        return max(0.0, min(100.0, float(iv_rank)))

    # R14 #9 exit-reject backoff tuning. A fast broker reject reverts the exit
    # order CLOSING->FILLED within seconds, so the monitor re-requests it on the
    # next 30s pass; the NORMAL working-order re-price reverts only at the ~300s
    # stale-order timeout. RAPID_WINDOW sits between those so consecutive fast
    # rejects escalate while the slow re-price cadence resets.
    _EXIT_REJECT_BASE_S = 30.0
    _EXIT_REJECT_CAP_S = 180.0
    _EXIT_REJECT_RAPID_WINDOW_S = 240.0
    _EXIT_REJECT_PAGE_STRIKES = 3

    def _exit_retry_ready(self, trade_id: str) -> bool:
        """R14 #9: escalating backoff for a repeatedly-rejected exit.

        Returns True if an exit attempt may proceed now, False to skip this
        pass. The FIRST exit for a trade always proceeds. A re-request that
        arrives within RAPID_WINDOW of the last attempt is treated as a reject
        (the previous close died fast); each such strike widens the backoff
        (30/60/120/180s) and, at PAGE_STRIKES, pages once per window. A
        re-request that arrives only after the slow working-order re-price
        cadence resets the strike count — normal price-chasing must not look
        like a reject storm.
        """
        now = time.monotonic()
        attempts: dict = getattr(self, "_exit_attempts", None)
        if attempts is None:
            attempts = self._exit_attempts = {}
        rec = attempts.get(trade_id)

        if rec is not None and now < rec["next_allowed_at"]:
            return False  # inside the backoff window — skip silently

        if rec is not None and (now - rec["last_attempt_at"]) < self._EXIT_REJECT_RAPID_WINDOW_S:
            strikes = rec["strikes"] + 1  # fast re-request => prior exit rejected
        else:
            strikes = 1  # first attempt, or the slow re-price cadence — reset

        backoff = (min(self._EXIT_REJECT_CAP_S,
                       self._EXIT_REJECT_BASE_S * (2 ** (strikes - 2)))
                   if strikes > 1 else 0.0)
        attempts[trade_id] = {
            "strikes": strikes,
            "last_attempt_at": now,
            "next_allowed_at": now + backoff,
        }
        if strikes > 1:
            log.warning("exit_retry_backoff", trade_id=trade_id, strikes=strikes,
                        backoff_s=round(backoff, 0),
                        note="prior exit rejected/cancelled; escalating backoff")
        if strikes >= self._EXIT_REJECT_PAGE_STRIKES and self._alert_gate(
                f"exit_reject:{trade_id}", interval_s=1800.0):
            # Fire-and-forget page (this method is sync). The repeated fast
            # reject is the signature of an illiquid or HALTED contract or a
            # persistently un-marketable quote — a human needs to look.
            try:
                asyncio.get_running_loop().create_task(self._send_notification(
                    f"EXIT REPEATEDLY REJECTED: {trade_id} — {strikes} fast "
                    f"rejects in a row. The contract may be halted or too "
                    f"illiquid to close at the quoted price. Backing off to "
                    f"{round(backoff)}s between tries; check it manually."
                ))
            except RuntimeError:
                pass
        return True

    def _alert_gate(self, key: str, interval_s: float = 900.0) -> bool:
        """R14: rate-limit a repeating page. True = send now, False = still
        inside the window. The exit paths re-demand a refused/deferred exit
        EVERY monitor pass (nothing enters closing_ids), so an unthrottled
        page repeats ~every 30s until post-market reconcile books the trade —
        an alert storm the operator learns to ignore (the R13 human-factors
        failure). Logs still fire every pass; only the page is gated."""
        now = time.monotonic()
        gates: dict[str, float] = getattr(self, "_alert_gate_last", None) or {}
        self._alert_gate_last = gates
        last = gates.get(key)
        if last is not None and (now - last) < interval_s:
            return False
        gates[key] = now
        return True

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
            # R8: TelegramNotifier.send returns False on failure without
            # raising — the retry loop treated that as success, so the A11
            # 3-attempt logic was unreachable and a dead Telegram channel was
            # invisible. False now retries; 5 consecutive definitive failures
            # write data/TELEGRAM_DEAD (surfaced by the supervisor digest).
            for attempt in range(3):
                try:
                    ok = await self._notify(msg)
                    if ok is not False:
                        self._tg_fail_streak = 0
                        return
                    log.warning("notification_attempt_returned_false",
                                attempt=attempt + 1)
                except Exception as e:  # noqa: BLE001
                    log.warning("notification_attempt_failed",
                                attempt=attempt + 1, error=str(e)[:200])
                await asyncio.sleep(2 * (attempt + 1))
            log.error("notification_dropped_after_retries", preview=msg[:80])
            self._tg_fail_streak = getattr(self, "_tg_fail_streak", 0) + 1
            if self._tg_fail_streak >= 5:
                try:
                    from pathlib import Path as _PTD
                    _PTD("data/TELEGRAM_DEAD").write_text(
                        f"{self._tg_fail_streak} consecutive dropped alerts")
                    log.critical("telegram_channel_dead",
                                 streak=self._tg_fail_streak)
                except Exception:  # noqa: BLE001
                    pass

        try:
            # R16: hold a strong reference. An unreferenced create_task can be
            # garbage-collected mid-flight (CPython gives the loop only a weak
            # ref), so an alert could vanish before its first await — and the
            # shutdown page in particular raced the loop closing. Tracked in a
            # set with a done-callback discard; _shutdown drains it.
            _tasks = getattr(self, "_notify_tasks", None)
            if _tasks is None:
                _tasks = self._notify_tasks = set()
            _t = asyncio.get_running_loop().create_task(_send_with_retry(message))
            _tasks.add(_t)
            _t.add_done_callback(_tasks.discard)
        except RuntimeError:  # no loop (sync/test context) -- best-effort inline
            try:
                await self._notify(message)
            except Exception as e:  # noqa: BLE001
                log.warning("notification_failed", error=str(e)[:200])
