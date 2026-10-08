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
from typing import NamedTuple

from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.bot.state import DailyStats, StateManager, TradeRecord, TradeStatus
from ait.broker.account import AccountManager
from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.config.runtime_env import capital_base, capital_base_source, contract_flag, contract_float
from ait.config.settings import Settings
from ait.data.dividends import DividendCalendar
from ait.data.earnings import EarningsCalendar
from ait.data.economic_calendar import EconomicCalendar
from ait.data.edgar_filings import EDGARMonitor
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.data.options_chain import OptionsChainService, solve_iv_from_price
from ait.data.multi_timeframe import MultiTimeframeAnalyzer
from ait.data.quality import DataQualityValidator
from ait.execution.executor import BAG_CON_ID, TradeExecutor
from ait.execution.portfolio import PortfolioManager, PositionAction, PositionStatus
from ait.execution.reconciler import PositionReconciler
from ait.execution.roll_booking import cas_roll_closing_trade, roll_order_ref, seed_roll_opening_trade
from ait.execution.roll_comparison import ev_should_roll, heuristic_should_roll
from ait.execution.roll_executor import prepare_roll_combo
from ait.execution.roll_selector import RollTarget, select_roll_target
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
from ait.utils.cadence import FAST_MONITOR_CADENCE_SECONDS
from ait.utils.logging import get_logger
from ait.utils.time import next_market_open, today_et
from ait.utils.time import now_et, parse_trade_ts

log = get_logger("bot.orchestrator")

# Strategies whose ENTRY confidence must come from the payoff-matched range
# model rather than the directional one — P(price stays inside the wings) is
# what a short-premium neutral structure actually gets paid for.
# fail-direction-02 (R22): declared once here so the fail-closed drop below
# cannot drift from the membership it protects. (The two pre-existing literal
# copies at the override and the neutral-only filter are set-membership-6, a
# separate register entry, and are left untouched by this wave.)
RANGE_GATED_STRATEGIES: frozenset[str] = frozenset({"iron_condor", "short_strangle"})

# W31: the condor builder's own rejection tokens, in operator English, for the
# launch cap-grid self-test. The RAW token is always reported beside the
# phrase — the phrase is for reading, the token is for grepping the builder.
# An unmapped reason falls through as its token rather than being dropped, so
# a new rejection path cannot silently become "no reason given".
_PROBE_REASON_PHRASE: dict[str, str] = {
    "credit_floor": "credit floor UNREACHABLE — the wing this budget affords "
                    "cannot yield the minimum credit",
    "credit_width_ratio": "credit/width ratio UNREACHABLE at the wing this "
                          "budget affords",
    "max_loss_over_budget": "the structure exceeds the per-trade budget once "
                            "strikes are SNAPPED to the real grid (the wing "
                            "the width formula asks for is not on the ladder)",
    "iv_rank_floor": "IV rank below backtest.ic_iv_rank_floor — the probe "
                     "never reached the cap grid",
    "insufficient_liquid_contracts": "too few liquid contracts in the probe "
                                     "chain (this is a probe defect, not an "
                                     "account finding)",
    "degraded_greeks": "the probe chain was judged to have guessed IV (this "
                       "is a probe defect, not an account finding)",
}


class CooldownBlock(NamedTuple):
    """W26: which stop is blocking, and when it closed.

    The post-stop cooldown used to return a bare exit_time, which was enough
    while it could only ever block the symbol it was asked about. Now that the
    window is cluster-scoped (logic-exit-risk-03) the blocking symbol is often
    NOT the blocked one, and an operator reading "QQQ entry blocked" has no way
    to know a SPY stop caused it unless the verdict carries it.
    """

    symbol: str
    exit_time: str


def _cooldown_now() -> datetime:
    """Reference instant for the post-stop cooldown, AWARE ET.

    numeric-pairs-02 (R22) kept this naive-local on the stated premise that
    "trades.exit_time is written with datetime.now() and the host runs ET, so
    the DB rows and this clock share one convention". W21 broke the first half
    of that premise (exit_time is now aware ET) and the CEST host move had
    already broken the second: a naive-local reference rolls its DATE over six
    hours early here, moving the cooldown to the wrong session. Both halves are
    fixed by reading ET explicitly. Still a module function so tests can pin
    it; a naive pin is tolerated because only .date() is taken from it.
    """
    return now_et()


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
    from ait.utils.time import ET, MARKET_OPEN, is_trading_day

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
    # W21: aware ET, so it can be compared against an aware exit_time.
    cutoff = datetime.combine(prev, MARKET_OPEN, tzinfo=ET)

    if len(_COOLDOWN_CUTOFF_CACHE) > 8:
        _COOLDOWN_CUTOFF_CACHE.clear()
    _COOLDOWN_CUTOFF_CACHE[ref] = cutoff
    return cutoff


RESTRICTED_LIST_PATH = "data/RESTRICTED.txt"
# fail-direction-11: sentinel returned when the ban file EXISTS but cannot be
# parsed. It is deliberately NOT a set — a caller that forgets to check it
# raises instead of silently reading "no bans".
RESTRICTED_UNREADABLE = None

# bot-day-09 (W47 registered, not fixed -- closed here): bound on
# _snapshot_marks_on_shutdown's own check_positions() call, leaving room
# inside main.py's 30s GRACEFUL_SHUTDOWN_TIMEOUT alongside the notify-drain's
# own 15s sub-budget.
SHUTDOWN_MARKS_SNAPSHOT_TIMEOUT_SECONDS = 10


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
        self._earnings = EarningsCalendar(
            buffer_days_before=settings.earnings.buffer_days_before,
            buffer_days_after=settings.earnings.buffer_days_after,
            refuse_entries_when_data_unavailable=(
                settings.earnings.refuse_entries_when_data_unavailable),
            lkg_max_age_hours=settings.earnings.lkg_max_age_hours,
            persist=self._state,  # W52a: LKG survives a restart
        )
        # W51 (model-vs-reality-ex-div-assignment-void): EarningsCalendar's
        # twin, protecting against early-assignment risk around ex-dividend
        # dates instead of IV crush around earnings.
        self._dividends = DividendCalendar(
            refuse_entries_when_data_unavailable=(
                settings.dividends.refuse_entries_when_data_unavailable),
            lkg_max_age_hours=settings.dividends.lkg_max_age_hours,
            persist=self._state,  # W52a: LKG survives a restart
        )
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
            dividend_calendar=self._dividends,
            condor_defense_config=settings.condor_defense,
            assignment_response_config=settings.assignment_response,
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
            # W42 (R26 policy-vs-impl-4): docs/RUNBOOK.md:153 documents this
            # gate as "AIT_LIQ_MIN_VOLUME >= 10" and claims the live-mode
            # startup assertion enforces it. It was `< 1`, so any value 1-9
            # passed — the volume half of the documented floor was never
            # actually asserted; only the spread half was.
            if _spread_env > 0.15 or _vol_env < 10 or _spread_cfg > 0.15:
                raise RuntimeError(
                    "LIVE-PROFILE ASSERTION FAILED: liquidity gates are still "
                    f"paper-relaxed (AIT_LIQ_MAX_SPREAD={_spread_env}, "
                    f"AIT_LIQ_MIN_VOLUME={_vol_env}, "
                    f"config max_bid_ask_spread_pct={_spread_cfg}). Tighten "
                    "them before live (see docs/RUNBOOK.md go-live checklist)."
                )

        # GOV-6 retired (W50): W41 added a startup refusal here for
        # max_contracts_per_trade > 1, "until the qty>1 booking path is
        # fixed (PLAN.md W50; R26 register: money-flow-05, trade-life-
        # partial-slice-pnl-erased, trade-life-promotion-ignores-broker-
        # quantity)". All three are fixed as of W50 (see PLAN.md) — close_
        # trade now folds partial-exit history into realized_pnl instead of
        # discarding it, the commission true-up accumulates across slices
        # instead of overwriting, and reconciler promotion adopts the
        # broker's leg quantity (or freezes new entries on a genuinely
        # inconsistent one) instead of trusting a stale local figure. A
        # guard-rail that outlives its bug is a false statement about the
        # system, so it comes down rather than staying as permanent
        # defense-in-depth. config.yaml's own max_contracts_per_trade: 1
        # pin is UNCHANGED by this — raising it is a separate trading
        # decision (Phase-2 learning volume), not a consequence of this fix.

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
        self._scheduler = MarketScheduler(settings)
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
        # Live incident 2026-09-17: registered with the instance-wide 120s
        # default, but heartbeat("market_data") below fires at most once per
        # SYMBOL inside a full scan cycle (scan_interval_seconds, 300s) --
        # never on the 30s fast-loop tick that keeps trading_loop/ibkr fresh.
        # Measured on the live book: consecutive heartbeats land 201-434s
        # apart, so EVERY gap exceeded 120s and check_and_recover (running
        # every ~5th fast-tick) paged "market_data is DOWN and recovery
        # failed" on an otherwise perfectly healthy scan loop -- repeatedly,
        # because market_data has no recovery callback (see below) to ever
        # clear it. 3x the scan interval comfortably covers the measured
        # variance (up to ~1.4x one cycle) with margin for a slow cycle,
        # while still paging within 2-3 missed cycles of a genuine stall.
        self._watchdog.register_component(
            "market_data",
            heartbeat_timeout=3 * self._settings.trading.scan_interval_seconds)
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

    def set_telegram(self, telegram) -> None:
        """W22: wire the full TelegramNotifier (buttons/edit/callbacks, not
        just .send) down to the reconciler, which needs it to prompt on an
        untracked position. Mirrors set_notification_callback's shape —
        main.py calls orchestrator.set_telegram(...) rather than reaching
        into the private _reconciler attribute directly."""
        self._reconciler.set_telegram(telegram)

    async def reconcile_now(self) -> None:
        """W22: run one reconcile() pass on demand, outside the normal
        startup/pre-market/post-market cadence — specifically so answering a
        manual-position Telegram prompt lifts HALT_UNTRACKED immediately
        instead of waiting for the next natural pass (or a restart). Passed
        into the poller task as a callback (main.py) rather than the poller
        holding a reference to the reconciler/IBKR client directly — the
        poller is deliberately lightweight (DB + Telegram only) so a slow or
        failed broker call can never block answering a prompt. Errors are
        logged, not raised: a failed on-demand reconcile must not crash the
        poller loop that triggered it; the next natural pass still runs.

        W30 (R26 F3, finding #6): this runs DETACHED (manual_positions.
        _fire_and_forget) from an operator's Telegram reply at ANY time,
        including mid-shutdown after IBKR has already disconnected —
        reconcile() against a dead client can only read an empty/stale
        snapshot, so skip rather than run it for no benefit. It also used to
        discard reconcile()'s result: a kept-CLOSING trade's working exit
        order was never re-adopted, exactly the gap the startup/post-market
        call sites already close.
        """
        if not getattr(self._ibkr, "connected", False):
            log.info("reconcile_now_skipped_disconnected")
            return
        try:
            recon = await self._reconciler.reconcile()
            for _tid, _oid in getattr(recon, "closing_exit_orders", {}).items():
                try:
                    self._executor.adopt_exit_order(
                        _oid, _tid, reason="reconcile_readopt")
                except Exception as _e:  # noqa: BLE001
                    log.warning("exit_order_adopt_failed",
                                trade_id=_tid, error=str(_e))
        except Exception as e:  # noqa: BLE001
            log.warning("reconcile_now_failed", error=str(e))

    def _launch_background_training(self) -> None:
        """W46 (bot-day-04): fire-and-forget so reconcile/exit-adoption/the
        trading loop start immediately instead of waiting up to ~7.5 min for
        a synchronous boot retrain. Guarded against double-launch -- run()'s
        startup call and _pre_market()'s call can both reach this in the
        same boot. Entry-scanning needs no new gate here: DirectionPredictor.
        predict / RangePredictor.predict already return None while untrained
        (ensemble.py, range_predictor.py), and _scan_symbol already treats a
        None prediction as skip-this-symbol -- the identical shape to the
        existing fail-direction-02 gate for the range predictor. Exits don't
        consult these predictors at all, so they are never gated by this.
        """
        existing = self._training_task
        if existing is not None and not existing.done():
            return
        if self._models_ready:
            return

        async def _train() -> None:
            try:
                ok = await self._trainer.ensure_models_ready(
                    self._settings.trading.universe)
                self._models_ready = bool(ok)
            except Exception as e:  # noqa: BLE001 — must never kill the startup sequence
                log.error("background_training_failed", error=str(e))
            finally:
                log.info("background_training_finished", ready=self._models_ready)

        self._training_task = asyncio.create_task(_train())

    async def run(self) -> None:
        """Main loop — runs continuously, trading during market hours."""
        self._running = True
        from datetime import datetime as _dt
        self._started_at = _dt.now()  # for the post-restart settling guard
        self._training_task: asyncio.Task | None = None
        self._models_ready = False
        log.info("orchestrator_starting", mode=self._settings.trading.mode)

        # W46 (bot-day-04): was a blocking `await ensure_models_ready(...)`
        # here -- training ran to completion BEFORE reconcile, exit-order
        # re-adoption, and the trading loop, leaving the book with zero
        # working loss exits (AIT_CREDIT_LOSS_LIMIT defaults to 0, so the
        # touch stop is the ONLY credit loss exit, and it only runs inside
        # the trading loop) for the full training duration on every slow
        # boot. Now fire-and-forget; see _launch_background_training.
        self._launch_background_training()

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

    async def _reconcile_with_broker(self, phase: str) -> bool:
        """W34 (2026-09-17 incident): reconcile only against a CONNECTED broker.

        The Gateway was down from 07:59 ET. ibkr_client.get_positions()
        returns [] when disconnected, so the 09:00 pre-market reconcile
        compared the book against NOTHING and reported clean — the one check
        that exists to catch a book/broker mismatch, passing because it could
        not see. The bot reconnects lazily (only client calls that go through
        ensure_connected), and _pre_market made none before reconciling.
        Returns False, pages, and leaves the reconcile OWED when the broker
        cannot be reached; _trading_cycle settles the debt.
        """
        if not await self._ibkr.ensure_connected():
            self._reconcile_pending = True
            log.error("reconcile_skipped_broker_disconnected", phase=phase)
            await self._send_notification(
                f"{phase.upper().replace('_', '-')}: broker NOT connected — "
                f"positions were NOT reconciled. This is not a clean result. "
                f"I will reconcile as soon as the connection returns; check "
                f"the Gateway."
            )
            return False
        self._reconcile_pending = False
        result = await self._reconciler.reconcile()
        if result.discrepancies:
            await self._send_notification(
                f"Position discrepancies found:\n" +
                "\n".join(result.discrepancies[:5])
            )
        return True

    async def _reconcile_if_owed(self) -> None:
        """W34: run the reconcile a disconnected phase could not."""
        if not getattr(self, "_reconcile_pending", False):
            return
        if not await self._ibkr.ensure_connected():
            return
        log.warning("reconcile_owed_running_now")
        await self._reconcile_with_broker("deferred")

    async def _pre_market(self) -> None:
        """Pre-market preparation (runs 30 min before open)."""
        log.info("pre_market_starting")

        # 1. Reconcile positions with IBKR
        await self._reconcile_with_broker("pre_market")

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

        # 5. Ensure ML models are trained. W46 (bot-day-04): was a blocking
        # await here too -- gated steps 6+ and the transition to MARKET_OPEN
        # on the same up-to-~7.5min training run. Fire-and-forget, guarded
        # against double-launch with run()'s own call to the same helper.
        self._launch_background_training()

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
        # W32 (concurrency-03 residual): the flag was ONE key, so a second
        # refused fill overwrote the first and only the last was ever paged.
        # The executor now writes alert_fill_after_close_<trade_id>; drain
        # every such key (and the pre-W32 bare key, if one is still stored).
        try:
            for _k in list(self._state.state_keys_like("alert_fill_after_close%")):
                _fac = self._state.get_state(_k, "")
                self._state.delete_state(_k)
                if _fac:
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
                # W14 (R24 regress-live-01 / pnl-money-01): the reconciler's
                # $0 bookkeeping rows are NOT trades. Booking them called
                # record_trade_result(0.0), which RESETS an active
                # consecutive-loss streak — two real losses plus one swept
                # never-filled entry meant the third real loss no longer
                # tripped the 3-loss pause. W3 built this authority for
                # exactly these rows; the drain never consulted it.
                from ait.reporting.go_live import is_real_close
                if not is_real_close(row.get("exit_reason")):
                    self._state.complete_exit_booking(trade_id)
                    log.info("not_real_close_outbox_discarded",
                             trade_id=trade_id,
                             reason=row.get("exit_reason", ""))
                    continue
                pnl = float(row["realized_pnl"] or 0.0)
                # R24 regress-live-09: the drain used to book the fill-time
                # ESTIMATE forever. Same correction the callback path applies.
                pnl = self._true_up_commission(trade_id, pnl)
                try:
                    exit_d = parse_trade_ts(row["exit_time"]).date()
                except Exception:  # noqa: BLE001
                    exit_d = None
                if exit_d is None:
                    # No usable exit_time: there is no day to credit. Claimed
                    # (never resurfaces), logged for the human ledger.
                    self._state.complete_exit_booking(trade_id)
                    log.warning("stale_exit_booking_skipped", trade_id=trade_id,
                                exit_time=row["exit_time"], pnl=pnl)
                    continue
                if exit_d != today:
                    # W37 (register time-calendar-03): stats and the breaker
                    # are DAILY quantities, so a prior-day close must not enter
                    # TODAY's halt math — that much of the old rule was right.
                    # But it DISCARDED the row: every close found by the
                    # 16:00-16:15 reconcile (undrainable until the next session
                    # because of the 120s grace), every overnight close, every
                    # Friday close found on Monday was booked nowhere — its
                    # day's ledger permanently short, Thompson blind to it, no
                    # page. Bucket by session instead: credit the day it
                    # closed, leave today's breaker and PDT count alone.
                    # W50 (money-flow-05): fetch before booking (not after)
                    # so the money/outcome split has trade.partial_exits —
                    # see _archived_partial_pnl. A trade with no partial
                    # history has _archived == 0.0, so _money_delta == pnl:
                    # byte-identical to pre-W50 behaviour.
                    _t = self._find_trade_by_id(trade_id)
                    _archived = self._archived_partial_pnl(_t)
                    _money_delta = round(pnl - _archived, 2)
                    _booked = self._book_prior_session_core(
                        trade_id, _money_delta, exit_d.isoformat(),
                        outcome_pnl=pnl)
                    try:
                        self._state.delete_state(f"trade_maxloss_{trade_id}")
                    except Exception:  # noqa: BLE001
                        pass
                    _done = self._exit_steps_done(trade_id)

                    def _thompson_prior(_t=_t, pnl=pnl):
                        if _t:
                            self._thompson.record_outcome(
                                strategy=_t.strategy, won=pnl > 0, pnl=pnl)

                    _ok = self._exit_step(trade_id, _done, "thompson",
                                          _thompson_prior)
                    _ok = await self._exit_notify_once(
                        trade_id, _done,
                        f"ORPHAN CLOSE BOOKED (prior session {exit_d.isoformat()}): "
                        f"{(_t.symbol if _t else trade_id)} P&L ${pnl:.2f} "
                        f"({row.get('exit_reason') or 'unknown'}) — credited to "
                        f"that day's stats; today's breaker/PDT not touched."
                    ) and _ok
                    log.warning("prior_session_close_booked", trade_id=trade_id,
                                exit_time=row["exit_time"], pnl=pnl,
                                stats_date=exit_d.isoformat(),
                                core_booked_now=_booked, steps_complete=_ok)
                    if _ok:
                        self._state.complete_exit_booking(trade_id)
                    continue
                # W32: money effects exactly once (one transaction + marker),
                # every other effect flagged as it lands — a retry resumes.
                # W50: fetch before booking — see _archived_partial_pnl.
                _t = self._find_trade_by_id(trade_id)
                _archived = self._archived_partial_pnl(_t)
                _money_delta = round(pnl - _archived, 2)
                _booked = self._book_exit_core(trade_id, _money_delta, outcome_pnl=pnl)
                try:
                    self._state.delete_state(f"trade_maxloss_{trade_id}")
                except Exception:  # noqa: BLE001
                    pass
                _done = self._exit_steps_done(trade_id)

                def _pdt(_t=_t):
                    if _t and _t.entry_time and \
                            parse_trade_ts(_t.entry_time).date() == today:
                        self._pdt_guard.record_day_trade(_t.symbol)

                def _thompson(_t=_t, pnl=pnl):
                    if _t:
                        self._thompson.record_outcome(
                            strategy=_t.strategy, won=pnl > 0, pnl=pnl)

                _ok = self._exit_step(trade_id, _done, "pdt", _pdt)
                _ok = self._exit_step(trade_id, _done, "thompson", _thompson) and _ok
                _ok = await self._exit_notify_once(
                    trade_id, _done,
                    f"ORPHAN CLOSE BOOKED: {(_t.symbol if _t else trade_id)} "
                    f"P&L ${pnl:.2f} ({row.get('exit_reason') or 'unknown'}) — "
                    "found by reconcile; breaker/stats now counted it.") and _ok
                log.warning("orphan_close_booked", trade_id=trade_id, pnl=pnl,
                            reason=row.get("exit_reason", ""),
                            core_booked_now=_booked, steps_complete=_ok)
                if _ok:
                    # W14 (concurrency-01): retired only after everything landed.
                    self._state.complete_exit_booking(trade_id)
            except Exception as e:  # noqa: BLE001 — one bad row must not stop the drain
                log.error("exit_outbox_booking_failed", trade_id=trade_id,
                          error=str(e))

    def _book_prior_session_core(self, trade_id: str, pnl: float,
                                 stats_date: str,
                                 outcome_pnl: float | None = None) -> bool:
        """W37: book a close into the daily_stats row of the day it CLOSED.

        The circuit breaker is deliberately not touched — it is a daily
        quantity and this loss is not today's. Same exactly-once marker as
        _book_exit_core. Returns True iff this call booked it.

        W50: `outcome_pnl` passthrough — see _archived_partial_pnl.
        """
        if self._state.exit_core_booked(trade_id) is True:
            return False
        return bool(self._state.book_exit_core(
            trade_id, pnl, None, None, stats_date=stats_date,
            outcome_pnl=outcome_pnl))

    def _book_exit_core(self, trade_id: str, pnl: float,
                        outcome_pnl: float | None = None) -> bool:
        """W32: apply a close's stats + breaker effects EXACTLY once.

        The breaker is updated in memory, then its blob, today's daily_stats
        delta and the outbox's core_booked marker are written in one sqlite
        transaction (StateManager.book_exit_core). If that write fails the
        in-memory breaker is rolled back — this process lives on, and a loss
        it kept without making durable would be counted again by the retry.
        Returns True iff this call booked it.

        W50: see _archived_partial_pnl — `pnl` (the money delta) and
        `outcome_pnl` (the win/loss verdict) can differ for a trade with
        partial-exit history.
        """
        if self._state.exit_core_booked(trade_id) is True:
            return False
        cb = self._circuit_breaker
        snap = cb.snapshot()
        cb.record_trade_result(pnl, persist=False, outcome_pnl=outcome_pnl)
        try:
            booked = bool(self._state.book_exit_core(
                trade_id, pnl, cb.STATE_KEY, cb.persistable_blob(),
                outcome_pnl=outcome_pnl))
        except Exception:
            cb.restore(snap)
            raise
        if not booked:
            cb.restore(snap)  # another pass booked it between the check and the write
        return booked

    def _exit_steps_done(self, trade_id: str) -> set:
        try:
            return set(self._state.exit_steps_done(trade_id))
        except Exception:  # noqa: BLE001 — unknown = nothing proven done
            return set()

    def _exit_step(self, trade_id: str, done: set, step: str, fn) -> bool:
        """W32: run one non-money side effect of a close at most once per
        success. A failure is contained (one broken learning store must not
        stop the page or the other steps) and left unflagged for the drain."""
        if step in done:
            return True
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.warning("exit_booking_step_failed", trade_id=trade_id,
                        step=step, error=str(e),
                        note="left unflagged; the drain retries this step")
            return False
        try:
            self._state.mark_exit_step(trade_id, step)
        except Exception as e:  # noqa: BLE001
            log.warning("exit_booking_step_mark_failed", trade_id=trade_id,
                        step=step, error=str(e))
        return True

    async def _exit_notify_once(self, trade_id: str, done: set, msg: str) -> bool:
        if "notify" in done:
            return True
        try:
            await self._send_notification(msg)
        except Exception as e:  # noqa: BLE001
            log.warning("exit_booking_step_failed", trade_id=trade_id,
                        step="notify", error=str(e))
            return False
        try:
            self._state.mark_exit_step(trade_id, "notify")
        except Exception as e:  # noqa: BLE001
            log.warning("exit_booking_step_mark_failed", trade_id=trade_id,
                        step="notify", error=str(e))
        return True

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
            except Exception as _ce:  # noqa: BLE001
                # W14 (R24 concurrency-02 / regress-live-02): this used to set
                # _claimed = True and book anyway. But the row survives a
                # failed claim, so the cycle drain claimed it minutes later
                # and booked the SAME close a second time — one -$150 stop
                # read as -$300 to the breaker, and two such exits tripped the
                # 3-loss pause on a two-loss day. Skip instead: the outbox is
                # durable, so the drain books it exactly once.
                log.error("exit_booking_claim_failed_deferred_to_drain",
                          trade_id=trade_id, error=str(_ce))
                continue
            if not _claimed:
                log.warning("exit_booking_already_claimed", trade_id=trade_id)
                continue

            # R15 #8: swap the fill-time ESTIMATE out and the ledger truth
            # in BEFORE stats/breaker/Thompson see the number. W11: extracted
            # so the outbox drain applies the identical correction.
            ex["realized_pnl"] = self._true_up_commission(
                trade_id, ex["realized_pnl"])

            realized_pnl = ex["realized_pnl"]  # FULL round trip (folded by close_trade)

            # W50 (money-flow-05): fetched here (not later) so the money/
            # outcome split below has it. A trade with no partial-exit
            # history has _archived_partial_pnl == 0.0, so money_delta ==
            # realized_pnl and every write below is byte-identical to before
            # this wave — the regression control for the no-partial case.
            trade = self._find_trade_by_id(trade_id)
            _archived = self._archived_partial_pnl(trade)
            money_delta = round(realized_pnl - _archived, 2)

            # W32 (R26 two-phase-claim-double-books): daily_stats and the
            # breaker used to be written one after the other, outside any
            # transaction, with the obligation retired at the very end — so a
            # fault after these writes left the claim to go stale and the drain
            # re-applied the whole close (-272.29 read as -544.58 / 2 losses).
            # They now land together with a core_booked marker, exactly once.
            # ALL results reach the breaker, not just losses: its daily P&L
            # must net wins against losses.
            #
            # W50: `money_delta` excludes partial slices already fed to
            # daily_stats/breaker in real time (see _archived_partial_pnl);
            # `outcome_pnl=realized_pnl` keeps the win/loss verdict reading
            # the trade's FULL round trip.
            try:
                self._book_exit_core(trade_id, money_delta, outcome_pnl=realized_pnl)
            except Exception as _be:  # noqa: BLE001
                log.error("exit_booking_core_failed_deferred_to_drain",
                          trade_id=trade_id, error=str(_be),
                          note="breaker rolled back; the row stays claimed and "
                               "the drain books it once the claim goes stale")
                continue
            _done = self._exit_steps_done(trade_id)

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
            getattr(self, "_exit_noquote_attempts", {}).pop(trade_id, None)
            getattr(self, "_exit_quoted_attempts", {}).pop(trade_id, None)

            # Deep-audit SR-H1: the PDT guard was fully built but NOTHING
            # ever called record_day_trade — it always reported 0/3 used.
            # A same-ET-day open+close is a day trade; record it.
            def _pdt(trade_id=trade_id):
                _t = self._find_trade_by_id(trade_id)
                if _t and _t.entry_time:
                    from ait.utils.time import now_et as _now_et
                    if parse_trade_ts(_t.entry_time).date() == _now_et().date():
                        self._pdt_guard.record_day_trade(_t.symbol)

            _ok = self._exit_step(trade_id, _done, "pdt", _pdt)

            # W50: `trade` was already fetched above (before _book_exit_core)
            # to compute the money/outcome split — no need to re-fetch here.
            # R7: stamp the REAL round-trip commission (from the executions
            # ledger) onto the closed trade row for the PF verdict.
            try:
                _real_comm = self._state.total_commission(trade_id)
                if _real_comm > 0:
                    self._state.update_trade_commission(trade_id, _real_comm)
            except Exception:  # noqa: BLE001
                pass
            if trade:
                def _thompson(trade=trade, realized_pnl=realized_pnl):
                    self._thompson.record_outcome(
                        strategy=trade.strategy,
                        won=realized_pnl > 0,
                        pnl=realized_pnl,
                    )

                _ok = self._exit_step(trade_id, _done, "thompson", _thompson) and _ok
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

                    def _drift(trade=trade, _dir=_dir, actual_dir=actual_dir):
                        self._trainer.drift_detector.record_outcome(
                            trade_id=f"{trade.symbol}-{_dir}",
                            actual_direction=actual_dir,
                        )

                    _ok = self._exit_step(trade_id, _done, "drift", _drift) and _ok

            msg = (
                f"EXIT FILLED: {trade.symbol if trade else trade_id}\n"
                f"Reason: {ex['exit_reason']}\n"
                f"Exit price: {ex['exit_price']:.2f}\n"
                f"P&L: ${realized_pnl:.2f}"
            )
            _ok = await self._exit_notify_once(trade_id, _done, msg) and _ok
            # W14 (concurrency-01): retire the obligation only now, after
            # every booking write above has landed. Claiming merely MARKS the
            # row, so a crash anywhere in this block leaves work for the drain
            # instead of destroying the record of a real close. W32: a step
            # that FAILED leaves the row too — the drain retries just that
            # step, since everything that landed is flagged.
            if _ok:
                try:
                    self._state.complete_exit_booking(trade_id)
                except Exception as _e:  # noqa: BLE001
                    log.warning("exit_booking_complete_failed",
                                trade_id=trade_id, error=str(_e),
                                note="row stays claimed; the drain retries it "
                                     "once the claim goes stale")
            else:
                log.warning("exit_booking_incomplete_left_for_drain",
                            trade_id=trade_id)
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
            # W47 bot-day-10: the whole tick is one position-protection pass
            # -- locked against _trading_cycle's own position-check block so
            # the two never interleave (see _get_position_lock). This task
            # now runs on its own guaranteed cadence regardless of scan
            # duration, so what used to be "mutually exclusive with the
            # scan" is now "serialized against the scan's position-check
            # specifically".
            async with self._get_position_lock():
                # Read-only guardrail: re-probe at most every 20 min so
                # mid-session read-only (Gateway daily restart relogging
                # read-only) is caught and alerted, not just at startup.
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

                # A1: mark-to-market daily-loss brake (the realized-only
                # breaker is entry-gated and blind to unrealized gap-day
                # bleeding). mtm_day = realized_today + (unrealized_now -
                # unrealized_at_SOD).
                try:
                    from ait.utils.time import now_et as _net
                    # W47 money-flow-02: a marks-missing position returns
                    # unrealized_pnl=0.0 (portfolio.py), indistinguishable from a
                    # real zero to everything below. Summing those in is a
                    # PHANTOM GAIN equal to the book's actual carry -- the one
                    # brake built to halt a bleeding gap day structurally cannot
                    # trip while it is reading fabricated numbers, and (if the
                    # SOD key happens to be empty this tick) would also LOCK that
                    # phantom in as today's baseline. Skip the evaluation AND the
                    # lazy stamp entirely; check_daily_loss_mtm is a pure per-tick
                    # evaluate-and-maybe-trip, so a skipped tick costs nothing --
                    # the next clean tick (real marks) re-evaluates for real.
                    _missing_ids = [p.trade_id for p in positions if p.marks_missing]
                    if _missing_ids:
                        log.warning(
                            "mtm_check_skipped_marks_missing",
                            count=len(_missing_ids), trade_ids=_missing_ids,
                            note="one or more positions have no IBKR marks this "
                                 "tick -- the MTM daily-loss brake did not "
                                 "evaluate, rather than evaluate on a fabricated "
                                 "zero. Per-position stop/TP paging is unchanged "
                                 "(portfolio.py's own marks-missing alert).",
                        )
                    else:
                        _unreal_now = sum((p.unrealized_pnl or 0.0) for p in positions)
                        _sod_key = f"mtm_sod_{_net().date().isoformat()}"
                        _sod_raw = self._state.get_state(_sod_key, "")
                        if _sod_raw == "":
                            _sod = await self._capture_mtm_baseline(positions, _unreal_now)
                            self._state.set_state(_sod_key, str(_sod))
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

    def _get_position_lock(self) -> asyncio.Lock:
        """W47 bot-day-10: serializes every check_positions()-and-exit pass.

        `_monitor_positions_fast` now runs on its OWN independent task
        (`_trading_loop` below), concurrently with `_trading_cycle`'s own
        position-check block (steps 3/3b/5/5b) rather than mutually
        exclusive with it. Two overlapping `check_positions()` calls are not
        merely redundant: `PortfolioManager._pass_quote_verdicts` is
        per-pass instance state (set at the top of `check_positions`, nulled
        in its `finally`) — an overlapping second pass would null the
        first's memo mid-flight, and `_spot_quote`'s per-symbol verdict
        (portfolio.py) is exactly the mechanism R19 added to stop a healthy
        feed from being misclassified "frozen" (which downgrades the touch
        stop to a 2-tick confirmation — the same +30-60s cost the R16/R19
        incident record already prices as a live loss). This lock makes the
        two callers take turns instead.

        Deliberately NOT held across `_trading_cycle`'s entry-scan tail
        (steps 6+): that part is minutes long and touches no position state,
        which is the entire point of running the fast monitor concurrently
        with it rather than after it.

        Lazy getattr-or-create, matching this class's existing pattern for
        state that post-dates `__init__` (see `PortfolioManager.
        _quality_validator`) — many existing tests build an orchestrator via
        `TradingOrchestrator.__new__` and none of them set this, so a plain
        `self._position_pass_lock` would raise AttributeError in all of them.
        """
        lock = getattr(self, "_position_pass_lock", None)
        if lock is None:
            lock = self._position_pass_lock = asyncio.Lock()
        return lock

    async def _trading_loop(self) -> None:
        """Main trading loop during market hours.

        W47 bot-day-10: position protection and the entry scan run as two
        INDEPENDENT concurrent tasks, not one sequential loop alternating
        between them. Before this, _trading_cycle's entry-scan tail
        (per-symbol chain fetches, NBBO waits, partial-fill waits — unbounded,
        no asyncio.wait_for anywhere in it) ran to completion before the next
        position check, so the real inter-check gap was `30s + scan
        duration`, not 30s — multi-minute exposure on the touch stop, the
        only credit-loss exit, during exactly the slow scans a volatile tape
        produces (the 2026-09-21 live exit log already shows this: "heartbeat
        stale 6.5 min" while a scan held the loop).

        Now `_protection_task_loop` runs `_monitor_positions_fast` on its own
        guaranteed FAST_MONITOR_CADENCE_SECONDS cadence no matter how long a
        scan takes, and also owns the supervisor file-heartbeat and the
        periodic watchdog recovery check — both used to run once per
        iteration of THIS loop, so a long scan silently stretched them
        exactly alongside the position-check gap this restructure closes.
        `_get_position_lock` (shared with `_trading_cycle`'s own position
        check) is the only thing genuine concurrency here needs: it stops
        the two from ever evaluating positions at the same instant, which
        would otherwise reintroduce R19's per-pass quote-verdict memo race
        (see that method's docstring). Nothing else needs guarding — a
        duplicated `_execute_exit` cannot double-place (W45's
        `_cas_to_closing` refuses the loser before `place_order`) and
        `check_fills_safe` already carries its own reentrancy guard (R11).
        """
        log.info("trading_loop_starting")
        scan_interval = self._settings.trading.scan_interval_seconds

        protect_task = asyncio.create_task(self._protection_task_loop())
        try:
            await self._scan_task_loop(scan_interval)
        finally:
            # Scoped to this phase: created on entry above, torn down here on
            # any exit (market close, bot stop, or an exception) — nothing
            # outlives the phase that started it.
            protect_task.cancel()
            try:
                await protect_task
            except asyncio.CancelledError:
                pass

    async def _protection_task_loop(self) -> None:
        """Task A: position protection, every FAST_MONITOR_CADENCE_SECONDS,
        unconditionally — regardless of any scan in flight. Runs until
        cancelled by `_trading_loop`; it does not check phase/`_running`
        itself, matching the "supervisor decides the lifetime" shape the
        entry-scan task (`_scan_task_loop`) hands control back through.
        """
        wd_tick = 0
        while True:
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

                await self._monitor_positions_fast()

                # R8: check_and_recover was only reachable from the outer
                # exception handler (i.e. never) — run it periodically so
                # component-down states can actually page.
                wd_tick += 1
                if wd_tick % 10 == 0:
                    try:
                        await self._watchdog.check_and_recover()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as e:
                log.error("trading_cycle_error", error=str(e))
                await self._note_loop_error("trading_cycle_error", str(e))
                self._circuit_breaker.record_api_failure()
                self._watchdog.record_error("trading_loop", str(e))

            await asyncio.sleep(FAST_MONITOR_CADENCE_SECONDS)

    async def _scan_task_loop(self, scan_interval: float) -> None:
        """Task B: the entry scan, every scan_interval seconds.

        Self-terminating (unlike the protection task): this loop's own
        while-condition is what lets `_trading_loop` return once the market
        closes or the bot stops, so it can cancel the protection task and
        hand control back to `run()`'s phase state machine.

        Cadence matches the original single-loop version exactly: the first
        scan fires after `scan_interval` seconds of accumulated sleep, and
        each subsequent one `scan_interval` seconds after the PREVIOUS scan
        completes (not from when it started) — a slow scan pushes the next
        one back by its own duration, same as before.
        """
        time_since_scan = 0.0
        while self._running and self._scheduler.get_current_phase() == TradingPhase.MARKET_OPEN:
            await asyncio.sleep(FAST_MONITOR_CADENCE_SECONDS)
            time_since_scan += FAST_MONITOR_CADENCE_SECONDS
            if time_since_scan < scan_interval:
                continue
            # R18 AMPLIFIER FIX: this reset used to happen only on success.
            # When _trading_cycle raised, the counter kept growing, so this
            # branch never re-armed and _monitor_positions_fast (now the
            # independent protection task) never ran again — the 2026-08-11
            # outage killed the MTM daily-loss brake, the 8-K material-event
            # check and the read-only re-probe as collateral, not just
            # entries. Reset BEFORE the call, unconditionally, so a raising
            # scan can never starve anything ever again — including, now,
            # itself: the next scan is still exactly scan_interval away.
            time_since_scan = 0.0
            try:
                await self._trading_cycle()
                self._clear_loop_error("trading_cycle_error")  # R8: success resets streak
                self._watchdog.note_success("trading_loop")
            except Exception as e:
                log.error("trading_cycle_error", error=str(e))
                await self._note_loop_error("trading_cycle_error", str(e))
                self._circuit_breaker.record_api_failure()
                self._watchdog.record_error("trading_loop", str(e))

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

        # W34: a phase that found the broker disconnected still owes a reconcile.
        await self._reconcile_if_owed()

        # 2. Sync risk manager with live positions (fixes stale position tracking)
        await self._sync_risk_manager_positions()

        # 3/3b/5/5b: position protection, locked against _monitor_positions_
        # fast's now-independent task (W47 bot-day-10 — see
        # _get_position_lock). Deliberately NOT wrapping steps 1/1b/2 above
        # (circuit breaker / reconcile_if_owed / sync_risk_manager_positions)
        # or step 6+ below (the entry scan) — those touch no position state,
        # and the entry scan is exactly the minutes-long tail that must stay
        # free to run without blocking the fast monitor.
        async with self._get_position_lock():
            # 3. Check and manage existing positions
            positions = await self._portfolio.check_positions()

            # Handle full exits
            exits_needed = [p for p in positions if p.should_exit]
            for pos in exits_needed:
                # W54b: rule 3/rule 3e's dual should_exit=True + roll_side
                # shape (never condor_defense's, which is should_exit=False
                # -- see PositionStatus/_evaluate_position's own docstrings)
                # gets a chance to roll instead of closing, via the
                # close-or-roll comparison. pos.roll_side alone is a safe,
                # sufficient gate: _compare_close_or_roll's OWN execute_
                # rolls/policy checks handle everything else, including
                # being a correct no-op when this wave's config stays at
                # its defaults.
                if pos.roll_side and await self._compare_close_or_roll(pos):
                    continue
                log.info("position_exit_triggered", symbol=pos.symbol, reason=pos.exit_reason)
                await self._execute_exit(pos)

            # Handle partial exits (scale-out at profit milestones)
            partial_exits = [p for p in positions if not p.should_exit and p.partial_exit_quantity > 0]
            for pos in partial_exits:
                log.info("partial_exit_triggered", symbol=pos.symbol, reason=pos.exit_reason,
                         quantity=pos.partial_exit_quantity)
                await self._execute_partial_exit(pos)

            # 3c. W61: roll dispatch — condor_defense's ROLL verdict, made
            # actionable ONLY when execute_rolls is ALSO on (a separate
            # opt-in from `enabled`, which alone still just reports the
            # verdict — see CondorDefenseConfig.execute_rolls). Mirrors
            # exactly how portfolio.py:1024 itself reads this config; there
            # is no separate orchestrator-level copy of it. When rolls_
            # enabled is False (today's only reachable state), ROLL
            # positions fall through to thesis re-evaluation below exactly
            # as they do today — byte-identical, unchanged behavior.
            _condor_cfg = getattr(self._portfolio, "_condor_defense_config", None)
            rolls_enabled = bool(_condor_cfg and _condor_cfg.enabled and _condor_cfg.execute_rolls)
            roll_positions, remaining_positions = self._partition_roll_positions(
                positions, rolls_enabled)
            for pos in roll_positions:
                log.info("roll_triggered", symbol=pos.symbol, side=pos.roll_side,
                         reason=pos.action_reason)
                await self._execute_roll(pos)

            # 3b. Thesis re-evaluation — exit early if original thesis invalidated
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

            # 5. Check fill status of pending orders (entry + exit)
            _filled_entries, completed_exits = await self._executor.check_fills_safe()
            await self._process_completed_exits(completed_exits)

            # 5b. W1: book closes nobody booked — reconciler/sweep closes and
            # bookings lost to a crash between the sqlite close and the callback.
            await self._drain_exit_outbox()

        # 6. Check if we should avoid new trades (the no-new-entries window).
        # W36 (register time-calendar-03): this `return` used to sit ABOVE
        # steps 5/5b, and 5b is the ONLY call site of the outbox drain — so an
        # orphan close landing in the last minutes of the session was not
        # booked until the next one. Booking is not an entry; only what
        # follows this line is.
        if self._scheduler.should_avoid_new_trades():
            log.debug("skipping_new_trades", reason="close_to_market_close")
            return

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
                # W29 (R26 F5 #6): persist the UNSCALED NLV. current_capital is
                # the snapshot's net_liquidation, which account.capital
                # (W-capital; env AIT_SIMULATED_CAPITAL overrides)
                # scales on purpose so the RISK STACK sizes as a small account.
                # This row is not risk input -- capital_base() resolves it as
                # the PERFORMANCE DENOMINATOR ahead of the default, so writing
                # the scaled figure re-scores the whole track record: the same
                # $863 drawdown reads 0.43% against $201k and 8.29% against
                # $10k, crossing the project's own <8% go-live gate, while
                # go_live.py (which divides by max_concurrent_car) stays right
                # and the dashboard disagrees with the scorecard ~72x.
                # The back-compat fallback lives in _reporting_nlv().
                # W40: a "KEEP THIS CALL ON ONE LINE" instruction used to sit
                # here, because a test grepped the literal
                # `set_state("last_net_liquidation"` out of this file to prove
                # the dashboard's key had a writer — so a W29 reformat with
                # identical behaviour failed it. The test now drives a real
                # trading cycle and reads the key back through the dashboard's
                # own query, so this line is free to be written however it
                # reads best.
                _reporting = self._reporting_nlv(account_snapshot)
                self._state.set_state("last_net_liquidation", str(_reporting))
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
        # W20a, THIRD instance of the class — found by the class-guard test,
        # not by me reading the two I already knew about. This one is a LIVE
        # ENTRY GATE, not telemetry. The set it compares against is built in
        # analyzer.py:459 from trades.entry_time, and adaptor._block_hour
        # accepts only `9 <= hour <= 16` — both ET-RTH conventions. Reading
        # the naive CEST clock here meant: an ET-learned block (say 10:00 ET)
        # could only ever match at 10:00 CEST = 04:00 ET, outside RTH, so it
        # NEVER fired; and CEST hours 17-22 (= 11:00-16:00 ET) fall outside
        # _block_hour's guard, so most of the session was unblockable. Latent
        # today only because no hour has ever been blocked (no persisted
        # blocked_hours anywhere on disk) — it fires the first time one is.
        from ait.utils.time import now_et as _now_et
        current_hour = _now_et().hour
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
        # $2.1k at Phase 2 via account.capital / AIT_SIMULATED_CAPITAL) — and run
        # the launch-size coherence self-test whenever NLV moves >20%.
        try:
            _nlv = await self._account.get_net_liquidation()
            if _nlv and _nlv > 0:
                self._risk_budget = _nlv * self._settings.risk.max_position_risk_pct
                _last = getattr(self, "_coherence_checked_nlv", 0.0)
                if not _last or abs(_nlv - _last) / _last > 0.20:
                    # W31: record it as checked only AFTER it reports success.
                    # This assignment used to sit ABOVE the await, so a check
                    # that threw was booked as done and never retried until
                    # NLV moved 20% — a self-test that failed once was off for
                    # the rest of the process.
                    if await self._launch_size_coherence_check(_nlv):
                        self._coherence_checked_nlv = _nlv
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

    def _meta_label_arm_sufficient(self) -> bool:
        """W49 (policy-vs-impl-7): is the real track record long enough to
        TRUST the meta-labeler's opinion for a live entry?

        `enabled=true` + `is_trained` alone proved nothing about this --
        three documents disagreed on the number (config.yaml said 50,
        PLAN.md said 100+, the code enforced neither at arm time). Now one
        adjustable number, `settings.meta_label.min_closes_to_enable`,
        checked against the SAME real-close population the go-live
        scorecard itself grades (`reporting.go_live.real_close_count` --
        real closes, IC-only, the same "how long is our track record"
        question, not a second counting convention).

        Fails CLOSED on any error (an unreadable count must not silently
        arm an unvalidated extra filter) -- for THIS gate specifically,
        "closed" means falling back to today's baseline of no meta-label
        opinion at all (skip the reject check below), not a blanket
        fail-closed-always rule: the meta-labeler is an ADDITIONAL filter
        that can only make entries MORE conservative, so not consulting it
        reverts to the already-proven primary-model-only behaviour rather
        than blocking anything.
        """
        try:
            from ait.reporting.go_live import real_close_count
            n = real_close_count(self._state._db_path)
            required = self._settings.meta_label.min_closes_to_enable
            sufficient = n >= required
            if not sufficient:
                log.info("meta_label_gate_insufficient_closes",
                         real_closes=n, required=required)
            return sufficient
        except Exception as e:  # noqa: BLE001
            log.warning("meta_label_close_count_check_failed", error=str(e))
            return False

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
            # W44 (fail-direction-03): is_near_earnings() also returns True
            # when the symbol's earnings data is unknown (refuse-on-unknown
            # policy) -- distinguish that from a normal earnings skip so an
            # outage pages instead of logging as routine.
            info = self._earnings.get_next_earnings(symbol)
            if not info.data_available:
                log.warning("earnings_data_unavailable_entry_refused", symbol=symbol)
                if self._alert_gate("earnings_data_unavailable", interval_s=3600):
                    await self._send_notification(
                        f"EARNINGS DATA UNAVAILABLE for {symbol} — no confirmed "
                        f"earnings answer and no usable last-known-good. New "
                        f"entries for this symbol are being refused FAIL-CLOSED "
                        f"until this clears. Exits are unaffected."
                    )
            else:
                log.info("skipping_near_earnings", symbol=symbol)
            return

        # Meta-label gate: ask secondary model if this signal is worth taking.
        # Bypassed in paper_trading_mode so backtest and paper P&L are comparable (Gap Z1 interim).
        # W49 (policy-vs-impl-7): `enabled`+`is_trained` alone proved nothing about
        # whether the live track record is long enough to TRUST what it says --
        # _meta_label_arm_sufficient() is the real-close-count floor, re-checked
        # every scan (not just once at startup) so crossing it mid-day takes
        # effect immediately.
        if (not paper_mode and self._meta_labeler is not None and self._meta_labeler.is_trained
                and self._meta_label_arm_sufficient()):
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

            # W20a (R25 tz-01): hour_of_day is an ML FEATURE and was stamped
            # from the naive LOCAL clock. This host runs CEST = ET+6, so the
            # same market hours that read {9..16} on the old ET box read
            # {15..22} here — a model trained on the mix learns a split that
            # encodes WHICH MACHINE WROTE THE ROW, not time of day. now_et()
            # is the repo's canonical helper and is already correct everywhere
            # else it is used.
            from ait.utils.time import now_et as _now_et
            meta_context = {
                "primary_confidence": final_confidence,
                "regime_trending_up": 1.0 if regime and regime.regime.value == "trending_up" else 0.0,
                "regime_trending_down": 1.0 if regime and regime.regime.value == "trending_down" else 0.0,
                "regime_high_vol": 1.0 if regime and regime.regime.value == "high_volatility" else 0.0,
                "regime_range_bound": 1.0 if regime and regime.regime.value == "range_bound" else 0.0,
                "vix": vix or 0,
                "iv_rank": iv_rank,
                "sentiment_score": sentiment.composite_score if sentiment and hasattr(sentiment, "composite_score") else 0,
                "hour_of_day": _now_et().hour,
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
                            # W42 (R26 set-membership-6): was a bare literal —
                            # a fourth independent copy of the range-gated set,
                            # alongside RANGE_GATED_STRATEGIES (:68) and the
                            # neutral-only filter below, which can only be
                            # non-empty because THIS check populated
                            # model_overridden from the SAME set. A drift here
                            # silently breaks that agreement.
                            if s.strategy_name in RANGE_GATED_STRATEGIES:
                                s.confidence = range_pred.probability_in_range
                                model_overridden.add(s.strategy_name)

            # W10 (R24): stash the prediction — INCLUDING its absence — so
            # _try_execute can persist it on the trade row. Logging it was not
            # enough: the pre-registered observe-mode rule ("compare closes
            # where p_in_range was above vs below 0.65") is scored months
            # later, and log lines rotate out in ~9 trading days. A symbol
            # whose model failed its edge gate produces None, and None is the
            # honest answer for that trade — never 0.0.
            try:
                _rp = getattr(self, "_entry_range_pred", None)
                if _rp is None:
                    _rp = self._entry_range_pred = {}
                _rp[symbol] = range_pred
            except Exception:  # noqa: BLE001 — telemetry must never block a scan
                pass

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
                    # W42 (R26 set-membership-6): the finding's own "must
                    # agree" pair — this filter's set has to match the one
                    # above that populates model_overridden, or a neutral
                    # strategy added to one but not the other silently loses
                    # every signal here.
                    if s.strategy_name in RANGE_GATED_STRATEGIES
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
                # trade-life-gatesoff-reintroduces-neutral-autoreject
                # (2026-09-01, PLAN pre-registration): in OBSERVE MODE nothing
                # writes model_overridden, so this used to fall back to the
                # DIRECTIONAL confidence for a market-neutral structure — and
                # manager.py:351 rejects < risk.min_confidence. In the neutral
                # regime a condor is built FOR, that number is below the gate,
                # so only trending-aligned days survived: condors entering
                # exclusively in their worst regime. The min_confidence gate is
                # a DIRECTIONAL gate; direction is not what a condor is paid
                # for, so validate direction-neutral structures on the
                # config-homed neutral baseline instead. Gates ON is unchanged:
                # the payoff-matched model still owns the number.
                if signal.strategy_name in model_overridden:
                    eff_conf = signal.confidence
                elif (not self._settings.ml.entry_gates_enabled
                        and signal.strategy_name in RANGE_GATED_STRATEGIES):
                    eff_conf = float(
                        self._settings.ml.observe_mode_neutral_confidence)
                    log.debug("observe_mode_neutral_confidence_applied",
                              symbol=signal.symbol,
                              strategy=signal.strategy_name,
                              directional=round(final_confidence, 3),
                              used=eff_conf)
                else:
                    eff_conf = final_confidence
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

    async def _capture_mtm_baseline(
        self, positions: list[PositionStatus], fallback_unrealized: float
    ) -> float:
        """W47 bot-day-09: the lazy SOD-key capture (the ONLY path this runs
        on, since `_prestamp_mtm_baseline` already claimed the key when it
        ran) used to stamp whatever the current tick's unrealized P&L was,
        with no log and no page -- indistinguishable from the outside from a
        genuinely fresh baseline. A bot down across the whole POST_MARKET
        window (nightly shutdown, or restarted after OFF_HOURS/PRE_MARKET),
        or simply never reaching `_post_market` the evening before, arrives
        here with no prestamp, and the next gap-down morning silently ran on
        this blind baseline -- the exact blindness the prestamp exists to
        prevent, reintroduced whenever the prestamp itself doesn't run.

        Prefer the last REAL persisted mark per position
        (StateManager.get_position_mark -- the same numbers master.py's
        digest already trusts, since the DB mark writer is itself guarded
        against marks-missing ticks, portfolio.py's `not marks_missing and
        persist` gate) over the current tick's number. Only when EVERY open
        position has one does this fully replace the fallback; a genuinely
        new position (no tick has ever marked it) has nothing to reconstruct
        from, so the fallback is used for it too -- but that case pages,
        because a baseline built from any current-tick number is a degraded
        one and the prior silence was itself the defect.
        """
        if not positions:
            return fallback_unrealized

        marks = [self._state.get_position_mark(p.trade_id) for p in positions]
        missing = [p.trade_id for p, m in zip(positions, marks) if m is None]
        if not missing:
            reconstructed = sum(m["unrealized_pnl"] for m in marks if m is not None)
            log.info(
                "mtm_baseline_reconstructed_from_persisted_marks",
                baseline=round(reconstructed, 2), positions=len(positions),
            )
            return reconstructed

        log.warning(
            "mtm_baseline_degraded_post_open",
            positions=len(positions), missing_persisted_marks=len(missing),
            missing_trade_ids=missing, fallback_baseline=round(fallback_unrealized, 2),
            note="today's MTM baseline was captured post-open instead of "
                 "prestamped at the prior close -- the bot likely missed the "
                 "_post_market window. A real overnight gap may be "
                 "under-counted today.",
        )
        if self._alert_gate("mtm_baseline_degraded", interval_s=3600):
            await self._send_notification(
                f"MTM BASELINE DEGRADED: today's daily-loss brake baseline was "
                f"captured post-open ({len(missing)}/{len(positions)} open "
                f"positions had no prior mark) instead of pre-stamped at the "
                f"prior close. A real overnight gap may be under-counted "
                f"until tomorrow's clean prestamp."
            )
        return fallback_unrealized

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

    @staticmethod
    def _reporting_nlv(snapshot) -> float:
        """W29 (R26 F5 #6): the NLV that belongs in the PERFORMANCE denominator.

        `snapshot.net_liquidation` is deliberately SCALED when
        account.capital simulation is on (W-capital; was env-only), so the whole
        risk stack sizes as a small account. That value must never reach `bot_state.last_net_liquidation`,
        because `capital_base()` resolves that row ahead of its default and every
        drawdown/return percentage divides by it. Persisting the scaled figure
        re-scores the entire track record: the same $863 drawdown reads 0.43%
        against $201k and 8.29% against $10k — across the project's own <8%
        go-live gate — while `reporting/go_live.py` (which divides by
        `max_concurrent_car`) stays correct, so the scorecard and the dashboard
        disagree ~72x about one book.

        A separate method rather than an inline expression so a test can drive
        the real decision instead of re-implementing it (R26 F7:
        "negative controls reimplement the line under test").

        `or net_liquidation` keeps snapshots built before this field existed
        working, and makes the unset-AIT_SIMULATED_CAPITAL path a no-op.
        """
        return (getattr(snapshot, "real_net_liquidation", 0.0) or 0.0) or float(
            getattr(snapshot, "net_liquidation", 0.0) or 0.0)

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

    def _build_probe_chains(self) -> list:
        """W31: one synthetic chain per configured strike grid."""
        from ait.data.synthetic_chain import build_probe_chain

        cfg = self._settings.sizing_probe
        return [
            build_probe_chain(
                underlying_price=cfg.underlying_price,
                implied_vol=cfg.implied_vol,
                dte=cfg.dte,
                strike_grid=g,
                risk_free_rate=self._settings.options.risk_free_rate,
            )
            for g in cfg.strike_grids
        ]

    def _run_probe_builder(self, chains, budget: float) -> list[dict]:
        """W31: run the REAL condor builder at `budget`, once per chain.

        Each result is what the builder DID: the structure it emitted, or its
        own stated reason for emitting nothing. Asking the builder is the
        whole point — a model of it cannot observe it refusing.
        """
        from ait.strategies.iron_condor import IronCondor as _IC

        results = []
        for chain in chains:
            # A FRESH instance per probe, never the shared one the selector
            # mutates (base.py:142 — not coroutine-safe, and this runs off
            # the trading loop).
            probe = _IC()
            probe.risk_budget = budget
            signals = probe.generate_signals(
                chain.symbol, chain, SignalDirection.NEUTRAL,
                self._settings.risk.min_confidence,
                self._settings.sizing_probe.iv_rank, None,
            )
            sig = signals[0] if signals else None
            results.append({
                "grid": self._probe_grid_of(chain),
                "max_loss": round(float(sig.max_loss), 2) if sig else None,
                "reason": None if sig else (probe.last_entry_reason or "unknown"),
            })
        return results

    @staticmethod
    def _probe_grid_of(chain) -> float:
        """Smallest gap between adjacent strikes — the ladder the builder had."""
        strikes = sorted({c.strike for c in chain.calls})
        gaps = [round(b - a, 2) for a, b in zip(strikes, strikes[1:])]
        return min(gaps) if gaps else 0.0

    def _probe_cap_grid(self, budget: float, chains=None) -> dict:
        """W31: the whole CPU-bound half of the self-test, in one call.

        Synchronous on purpose — the caller hands it to a thread once, rather
        than hopping per probe. Returns the per-grid verdicts and, when NOTHING
        was emitted, the budget that would fix it.
        """
        chains = chains if chains is not None else self._build_probe_chains()
        probes = self._run_probe_builder(chains, budget)
        advised = None
        if not any(p["max_loss"] is not None for p in probes):
            # MEASURED, not derived. W25 advised $86 where the account needed
            # $140 (PLAN.md W25 [C1]) because it inverted the width formula by
            # hand; asking the builder cannot drift from the builder. Reuses
            # the chains already built — the ladder is 20 builder runs, not 20
            # chain builds.
            ladder = max(budget, 1.0)
            for _ in range(20):
                ladder *= 1.25
                if any(p["max_loss"] is not None
                       for p in self._run_probe_builder(chains, ladder)):
                    advised = ladder
                    break
        return {"probes": probes, "advised_budget": advised}

    async def _launch_size_coherence_check(
        self, nlv: float, *, chains=None,
    ) -> bool:
        """R7-SOON: assert the cap grid is mutually satisfiable at this NLV.

        The Phase-2 launch account bricked silently before: sizer budget <
        one minimum-viable condor, and the daily-loss breaker smaller than a
        single trade's stop. Runs at startup and whenever NLV moves >20%;
        alerts LOUDLY instead of letting the account sit structurally unable
        to trade (or unable to survive one normal stop-out).

        W31 (R26 F4) rebuilt it around the REAL builder. It used to call the
        private `_vol_scaled_width` with a $1 underlying and reason about the
        width that came back — a probe that returns the NARROWEST structure
        the builder can ever make. Correct for "can this account afford
        anything at all"; exactly backwards for "can the breaker absorb one
        stop-out", which needs the LARGEST. So `builder_width` was
        bit-identical at $9,750 and $201,475, the breaker arm compared every
        account against a fixed $130, and three of eleven real condors in the
        live book carried a single max loss exceeding the whole daily breaker
        while this check reported problems=0.

        Nothing downstream of width selection ran either: strike snapping was
        unmodelled and `max_loss_over_budget` had no arm, so on a $5 grid the
        check was silent while the builder emitted ZERO.
        """
        try:
            # W31 (R26 F4 w25-coherence-probe-uses-a-budget-the-live-builder-
            # never-receives): probe with the budget the builder is ACTUALLY
            # handed — `nlv * risk.max_position_risk_pct`, the same expression
            # the live scan uses. W25 used min(gate, sizer), reasoning that a
            # trade must clear both caps. True, but they bind at DIFFERENT
            # layers: gate 6b (manager.py) rejects on max_position_risk_pct
            # before the sizer (position_sizer.py) ever returns its 0-contract
            # verdict on max_position_pct. Pre-shrinking the builder's budget
            # to the min models a NARROWER wing than the builder will build
            # whenever the gate cap is the looser of the two — so the check
            # passed while the real structure blew the sizer's cap and every
            # entry was refused. The sizer cap is not ignored: it is now tested
            # against the RESULT, in A3.
            budget_gate = nlv * self._settings.risk.max_position_risk_pct
            budget_sizer = nlv * self._settings.positions.max_position_pct
            budget = budget_gate
            min_credit = contract_float("AIT_IC_MIN_CREDIT")
            breaker_usd = nlv * self._settings.risk.max_daily_loss_pct

            # The CPU-bound half: build the probe chains and run the real
            # builder on each, once, off the event loop.
            probe = await asyncio.to_thread(self._probe_cap_grid, budget, chains)
            probes = probe["probes"]
            emitted = [p for p in probes if p["max_loss"] is not None]
            # The LARGEST structure the builder can emit here. This is the
            # number the breaker must survive, and the one the $1 probe could
            # never see.
            worst = max((p["max_loss"] for p in emitted), default=0.0)

            problems = []

            # A1 — NO STRUCTURE. Subsumes W25's separate affordability and
            # credit-floor arms: both asked "will the builder emit?" of a
            # model. Now the builder answers, and says why.
            if not emitted:
                _detail = "; ".join(
                    f"${p['grid']:.0f} grid -> "
                    f"{_PROBE_REASON_PHRASE.get(p['reason'], p['reason'])} "
                    f"[{p['reason']}]"
                    for p in probes
                )
                _advised = probe["advised_budget"]
                _fix = (f"Needs a per-trade budget of ~${_advised:.0f} "
                        f"(measured: the smallest budget at which the real "
                        f"builder emits)"
                        if _advised else
                        "No budget up to 85x this one makes the builder emit — "
                        "the credit floors, not the budget, are what refuse")
                problems.append(
                    f"the REAL builder emits ZERO trades at a per-trade budget "
                    f"of ${budget:.0f} (risk.max_position_risk_pct) — {_detail}. "
                    f"{_fix}. Credit floor ${min_credit:.2f} "
                    f"(AIT_IC_MIN_CREDIT).")

            # A2 — BREAKER. W25's version measured "one normal stop-out" from
            # the NARROWEST structure the builder can make, so it compared
            # every account above ~$6k against a fixed $130 and never fired.
            # With the credit stop disabled (AIT_CREDIT_LOSS_LIMIT=0) nothing
            # closes a condor early, so the honest bound on one trade's loss is
            # the defined-risk max loss of the LARGEST structure — measured
            # here from the signal the builder actually produced.
            # W64: the multiple is risk.breaker_to_max_loss_ratio, not a literal
            # 1.5. It was the one number in this arm with no config home, and it
            # decided whether the alert fired (at $5k: $150 breaker vs 1.5 x $124.50
            # = $187 fired against an otherwise coherent 3%/3% setup).
            _ratio = self._settings.risk.breaker_to_max_loss_ratio
            if worst > 0 and breaker_usd < _ratio * worst:
                problems.append(
                    f"daily-loss breaker ${breaker_usd:.0f} "
                    f"({self._settings.risk.max_daily_loss_pct:.1%} of NLV) < {_ratio:g}x "
                    f"(risk.breaker_to_max_loss_ratio) "
                    f"the largest single max loss the builder emits here "
                    f"(${worst:.0f}, defined-risk: the credit stop is DISABLED, "
                    f"so nothing closes it early) — one stopped trade halts the "
                    f"whole day. risk.max_position_risk_pct "
                    f"({self._settings.risk.max_position_risk_pct:.1%}) funds a "
                    f"trade the breaker cannot absorb; they need to be set "
                    f"together.")

            # A3 — THE TWO-KNOB CAP GRID (the W25 ADDENDUM, registered then and
            # still not linked). gate 6b and the sizer enforce the same
            # quantity at different layers with nothing tying them: raising
            # risk.max_position_risk_pct above positions.max_position_pct lets
            # the builder construct a structure gate 6b passes and the sizer
            # then refuses, one 0-contract verdict at a time, logged as routine.
            # DETECTION ONLY — linking the knobs moves live sizing and wants a
            # backtest, so it stays registered rather than fixed here.
            if worst > 0 and worst > budget_sizer:
                problems.append(
                    f"cap-grid SKEW: the builder is funded ${budget:.0f} by "
                    f"risk.max_position_risk_pct and emits a structure risking "
                    f"${worst:.0f}, but positions.max_position_pct caps a "
                    f"position at ${budget_sizer:.0f} — gate 6b passes it and "
                    f"the position sizer then returns 0 contracts, so entries "
                    f"brick silently. Lower risk.max_position_risk_pct or raise "
                    f"positions.max_position_pct.")

            log.info("launch_size_coherence",
                     nlv=round(nlv), budget=round(budget),
                     budget_gate=round(budget_gate), budget_sizer=round(budget_sizer),
                     breaker=round(breaker_usd),
                     worst_max_loss=round(worst),
                     probes=probes,
                     problems=len(problems),
                     # W31: the TEXT, not just the count. _send_notification
                     # returns early when Telegram is unconfigured, so the
                     # count was all an operator reading logs ever saw.
                     problem_text=problems)
            if problems:
                await self._send_notification(
                    "CAP-GRID INCOHERENT at NLV ${:,.0f}:\n- ".format(nlv)
                    + "\n- ".join(problems)
                    + "\nFix knobs (risk.max_position_risk_pct / breaker / "
                      "credit floors) before expecting trades."
                )
            return True
        except Exception as _e:  # noqa: BLE001
            # W31: still caught — a self-test must never kill the trading
            # cycle — but no longer SILENT. This used to be a log.warning that
            # discarded the verdict, so any change that made the check throw
            # (a renamed private method, a chain the builder chokes on) turned
            # it into a permanent no-op reporting nothing, indistinguishable in
            # the logs from a clean account. threshold=1 because this site runs
            # ONCE per process: the default streak of 5 can never be reached,
            # which is the same can't-fire shape as the arms this wave removed.
            # Returning False (rather than raising) keeps the promise above
            # literally true while telling the caller not to record this NLV as
            # checked, so the next cycle retries.
            log.exception("coherence_check_failed", error=str(_e))
            await self._note_loop_error(
                "size_coherence_check", str(_e), threshold=1)
            return False

    async def _note_loop_error(self, kind: str, error: str, *,
                               threshold: int = 5) -> None:
        """R8 (incident 2026-07-10): 516 fast-monitor + 145 trading-cycle
        errors produced ZERO alerts — per-iteration try/excepts logged a
        WARNING and kept looping while stops/TPs were dead all session. Any
        streak of >=`threshold` consecutive failures at one site pages Telegram
        once, re-pages hourly while it persists; the streak resets on the first
        clean iteration (_clear_loop_errors).

        W31: `threshold` exists because the default of 5 assumes a LOOP. A
        site that runs once per process can never reach five consecutive
        failures, so routing it here would have paged never — the same
        can't-fire shape as the dead arms W31 removed. Default unchanged, so
        every existing caller keeps the streak semantics R8 designed."""
        import time as _t
        st = getattr(self, "_err_streaks", None)
        if st is None:
            st = self._err_streaks = {}
        n, last_alert = st.get(kind, (0, 0.0))
        n += 1
        if n >= threshold and (_t.time() - last_alert) > 3600:
            last_alert = _t.time()
            # W6/log-contracts-2: "LOOP IMPAIRED" existed ONLY inside the
            # Telegram f-string, so every log consumer (deploy gate, status,
            # first-RTH liveness check) reported a healthy loop through an
            # R8-class outage. Emit it as a structured event too.
            log.error("loop_impaired", kind=kind, consecutive_failures=n)
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

    async def _check_trading_enabled(self) -> bool | None:
        """Probe whether orders can actually be placed; alert loudly if not.

        Guardrail against the silent read-only lock. Called at startup and
        periodically. On read-only it fires a single Telegram alert (until it
        recovers) so the user knows immediately to restart Gateway — instead
        of finding out at end of day that nothing traded.
        """
        try:
            can_trade = await self._ibkr.verify_can_trade()
        except Exception as _e:  # noqa: BLE001 — the probe must not break the loop
            can_trade = None
            log.warning("trading_probe_raised", error=str(_e))

        if can_trade is None:
            # W18 (w17-08): UNKNOWN is not OK. The probe could not establish
            # that orders work, and this alarm is the ONLY thing standing
            # between a READ-ONLY Gateway session and a day of silently
            # rejected orders — the failure mode Gateway actually has, and
            # which recurs on its daily restart. Page on its own throttle key
            # so it cannot be starved by the read-only page.
            if self._alert_gate("trading_probe_unknown", 3600):
                await self._send_notification(
                    "CANNOT VERIFY TRADING: the read-only probe did not "
                    "complete, so it is UNKNOWN whether IB Gateway accepts "
                    "orders. If Gateway logged in read-only, orders are being "
                    "rejected silently. Check the Gateway window."
                )
                log.critical("trading_verify_unknown_alerted")
            return None

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

    def _post_stop_cooldown_until(self, symbol: str) -> CooldownBlock | None:
        """R12-B4 query, extracted so tests can execute it against a real DB.

        Returns the most recent qualifying stop close inside the cooldown window
        as (symbol, exit_time), else None. W26: carries the BLOCKING symbol
        because it is no longer necessarily the symbol being blocked — an
        operator seeing "QQQ entry blocked" needs to know a SPY stop caused it.

        W26 (logic-exit-risk-03): the window is CLUSTER-SCOPED and GRADUATED.
        This asked `WHERE symbol=?`, so a SPY touch stop left QQQ and IWM free to
        re-enter the same move — and the bot's own SECTOR_GROUPS calls all three
        one ~0.95 cluster, i.e. the same macro bet. Now:
          - the stopped symbol ITSELF keeps the full one-trading-day window;
          - cluster SIBLINGS get risk.post_stop_cluster_cooldown_minutes
            (0 = disabled, restoring the pre-W26 per-symbol behaviour).
        Graduated rather than cluster-wide-for-a-full-day because the universe is
        a single cluster: the strict reading darkens the entire book for the rest
        of the session and all of the next. See the config field's description.

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
        from ait.risk.correlation import cluster_symbols

        # Tolerant read on purpose. Several test files construct a deliberately
        # minimal orchestrator (`__new__`, DB file only) to exercise this query in
        # isolation, and that is a pattern worth keeping rather than forcing every
        # caller to carry a Settings. Absent config means sibling scoping OFF —
        # the pre-W26 per-symbol behaviour — which can only ever refuse FEWER
        # entries, never more. Not a silent-disable hazard in production: the
        # field has a Pydantic default and the models are extra="forbid", so a
        # config missing it is a startup error, never a quiet 0.
        _risk_cfg = getattr(getattr(self, "_settings", None), "risk", None)
        _sibling_minutes = int(
            getattr(_risk_cfg, "post_stop_cluster_cooldown_minutes", 0) or 0
        )
        # 0 = sibling scoping disabled: ask only about the symbol itself, which
        # is byte-for-byte the pre-W26 query.
        _scope = sorted(cluster_symbols(symbol)) if _sibling_minutes > 0 else [symbol]

        # One query PER symbol, keeping the original single-row shape rather than
        # an unbounded `IN (...)` fetch: the scope is <=4 symbols, only each
        # symbol's MOST RECENT qualifying stop can block (an older one cannot
        # block if the newest does not), and the proven query is reused verbatim.
        import sqlite3 as _sq12
        _latest: list[tuple[str, str]] = []
        _con = _sq12.connect("file:data/ait_state.db?mode=ro", uri=True)
        try:
            for _sym in _scope:
                _row = _con.execute(
                    "SELECT exit_time FROM trades WHERE symbol=? AND status='closed' "
                    "AND (COALESCE(exit_reason_detailed,'') LIKE '%stop_loss%' "
                    "OR COALESCE(exit_reason_detailed,'') LIKE '%short_strike_touch%') "
                    "ORDER BY exit_time DESC LIMIT 1", (_sym,)).fetchone()
                if _row and _row[0]:
                    _latest.append((_sym, _row[0]))
        finally:
            _con.close()

        # numeric-pairs-02 (R22): the SELF window is ONE TRADING day, not a flat
        # 30 wall-clock hours — see _post_stop_cooldown_cutoff. W26: the SIBLING
        # window is a plain duration and needs no calendar, being scoped in hours
        # rather than sessions.
        _self_cutoff = _post_stop_cooldown_cutoff()
        _sibling_cutoff = (
            _cooldown_now() - timedelta(minutes=_sibling_minutes)
            if _sibling_minutes > 0 else None
        )

        _blocks: list[CooldownBlock] = []
        for _sym, _raw in _latest:
            # W21: this reader selects exit_time in SQL but parses the row value,
            # so it carries no column name on the parse line and the migration's
            # column-name grep missed it. Found by the suite, not by reading.
            _exit_t = parse_trade_ts(_raw)
            _cutoff = _self_cutoff if _sym == symbol else _sibling_cutoff
            if _cutoff is not None and _exit_t >= _cutoff:
                _blocks.append(CooldownBlock(symbol=_sym, exit_time=_raw))
        if not _blocks:
            return None
        # Report the most recent blocking stop, so the log names the freshest
        # cause rather than whichever symbol happened to sort first.
        return max(_blocks, key=lambda b: parse_trade_ts(b.exit_time))

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
            # W21: entry_time is now timezone-aware, so the cutoff it is
            # compared against must be too. A naive cutoff here raised
            # "can't compare offset-naive and offset-aware datetimes", the
            # bare `except` below caught it, and the guard FAILED CLOSED —
            # every entry refused as "unverified". Caught by the suite.
            cutoff = now_et() - timedelta(minutes=COOLDOWN_MINUTES)
            for rt in self._state.get_recent_trades(n=30):
                if rt.symbol == signal.symbol and rt.strategy == signal.strategy_name:
                    try:
                        entered = parse_trade_ts(rt.entry_time)
                    except (ValueError, TypeError):
                        continue
                    if entered >= cutoff:
                        log.info("cooldown_skip", symbol=signal.symbol,
                                 strategy=signal.strategy_name,
                                 minutes_ago=int((now_et() - entered).total_seconds() / 60))
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

        # R12-B4 (user-approved): post-stop re-entry discipline — extracted
        # to _cooldown_blocks_entry so its FAIL DIRECTION is testable by
        # execution rather than by reading the source.
        if await self._cooldown_blocks_entry(signal):
            return False

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
        # W42 (R26 set-membership-6): third independent copy of the same set.
        _neutral = signal.strategy_name in RANGE_GATED_STRATEGIES
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
            if self._earnings.would_hold_through_earnings(signal.symbol, today_et(), expiry_date):
                # W44 (fail-direction-03): same distinction as the
                # is_near_earnings gate in _scan_symbol -- an outage pages,
                # a normal earnings block just logs.
                info = self._earnings.get_next_earnings(signal.symbol)
                if not info.data_available:
                    log.warning("earnings_data_unavailable_entry_refused", symbol=signal.symbol)
                    if self._alert_gate("earnings_data_unavailable", interval_s=3600):
                        await self._send_notification(
                            f"EARNINGS DATA UNAVAILABLE for {signal.symbol} — no "
                            f"confirmed earnings answer and no usable "
                            f"last-known-good. New entries for this symbol are "
                            f"being refused FAIL-CLOSED until this clears. Exits "
                            f"are unaffected."
                        )
                else:
                    log.info("trade_blocked_earnings", symbol=signal.symbol, expiry=str(signal.expiry))
                return True  # symbol blocked through earnings

            # W51 (model-vs-reality-ex-div-assignment-void): same slot as the
            # earnings gate above, one slot later. Only relevant for a
            # signal that SELLS a call — the leg an ex-div early-assignment
            # actually hits.
            if self._signal_sells_a_call(signal) and self._dividends.would_hold_through_ex_div(
                    signal.symbol, today_et(), expiry_date):
                # Same data_available distinction as the earnings gate.
                info = self._dividends.get_next_ex_div(signal.symbol)
                if not info.data_available:
                    log.warning("dividend_data_unavailable_entry_refused", symbol=signal.symbol)
                    if self._alert_gate("dividend_data_unavailable", interval_s=3600):
                        await self._send_notification(
                            f"DIVIDEND DATA UNAVAILABLE for {signal.symbol} — no "
                            f"confirmed ex-dividend answer and no usable "
                            f"last-known-good. New short-call entries for this "
                            f"symbol are being refused FAIL-CLOSED until this "
                            f"clears. Exits are unaffected."
                        )
                else:
                    log.info("trade_blocked_ex_div", symbol=signal.symbol, expiry=str(signal.expiry))
                return True  # symbol blocked through ex-dividend

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
            elif 0 < int(revalidation.position_size or 0) < adjusted_size:
                # W52a (registered in W50): validate_trade can APPROVE at a
                # smaller size than asked (min(request, sizer) and the
                # correlation haircut). Only `.approved` was read here, so
                # the multiplied size went out even when the sizer had just
                # cut it — the multiplier bypassed the sizer exactly where the
                # sizer said "less". A position_size of 0 on an approval is
                # a stub/legacy shape, not a cap.
                log.info("multiplier_size_capped_by_revalidation",
                         symbol=signal.symbol, strategy=signal.strategy_name,
                         wanted=adjusted_size, using=revalidation.position_size,
                         reason=revalidation.reason)
                adjusted_size = int(revalidation.position_size)

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
                # W10: the range model's own verdict, in its own columns, with
                # the spec it was produced under. `confidence` above is
                # whatever number actually drove risk validation and CHANGES
                # MEANING with gate mode (it was the range probability while
                # the gates were armed, and is the observe-mode neutral
                # baseline since 2026-09-01) — which is precisely why the
                # verdict could not be scored across the mode change.
                **self._entry_range_context(signal.symbol),
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

    def _cas_to_closing(self, trade: TradeRecord) -> bool:
        """W45 (bot-day-05, R12-mirror): the CAS to CLOSING happens on the
        line immediately before the placeOrder call it guards, not after --
        a hard kill in between leaves a CLOSING row (armed for the
        reconciler's existing working-order recovery, reconciler.py
        ~1128-1170) instead of a live, untracked broker order on a row that
        still reads FILLED. Called only once a builder has already decided
        to place something; every defer/no-quote/qualification-failure path
        returns before reaching this, so nothing needs a revert on the
        not-placed side -- a path that never CASes never needs to un-CAS.
        """
        ok = self._state.update_trade_status(
            trade.trade_id, TradeStatus.CLOSING,
            from_statuses=(TradeStatus.FILLED, TradeStatus.PARTIAL))
        if not ok:
            log.critical("closing_transition_refused_before_place",
                         trade_id=trade.trade_id)
        return ok

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
                # W45: the CAS to CLOSING (if it happened at all) already
                # happened inside the builder, immediately before ITS OWN
                # placeOrder call -- every path that reaches here with
                # exit_trade None returned from the builder before that
                # point, so the trade is still FILLED/PARTIAL, unchanged.
                # Nothing to revert.
                log.error("exit_order_not_placed", trade_id=pos.trade_id)
                return

            # W45: CLOSING was already CAS'd inside the builder, on the line
            # immediately before the placeOrder call that produced
            # exit_trade -- see _cas_to_closing. Nothing left to do here but
            # register the fill tracker; the old post-placement CAS (and its
            # cancel-on-race-loss cleanup, which only existed because the
            # order used to be placed BEFORE this point) is gone.

            # Register the exit order with the executor for fill tracking
            if exit_trade.order.orderId:
                # W12: hand the exit's fill-quality context to the ledger so
                # slippage is measurable on BOTH halves of the round trip.
                _q = getattr(self, "_last_exit_quality", (0.0, 0.0, 0.0))
                self._executor.register_exit_order(
                    order_id=exit_trade.order.orderId,
                    trade_id=pos.trade_id,
                    exit_reason=pos.exit_reason,
                    estimated_pnl=pos.unrealized_pnl,
                    signal_price=_q[0], live_mid=_q[1], nbbo_spread=_q[2],
                )
                self._last_exit_quality = (0.0, 0.0, 0.0)  # consume

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

    @staticmethod
    def _partition_roll_positions(
        positions: list[PositionStatus], rolls_enabled: bool,
    ) -> tuple[list[PositionStatus], list[PositionStatus]]:
        """W61: split not-already-exiting positions into (roll_positions,
        remaining_for_thesis_reeval). When rolls_enabled is False,
        roll_positions is always [] and every ROLL-action position falls
        through to remaining -- byte-identical to this wave's own prior
        behavior (a ROLL verdict is reported, never acted on)."""
        candidates = [p for p in positions if not p.should_exit and p.partial_exit_quantity == 0]
        if not rolls_enabled:
            return [], candidates
        rolls = [p for p in candidates if p.action == PositionAction.ROLL]
        remaining = [p for p in candidates if p.action != PositionAction.ROLL]
        return rolls, remaining

    def _gather_ev_leg_inputs(
        self, prepared, spot: float,
    ) -> tuple[list[dict], list[dict], list[dict]] | None:
        """W54b: build ev_should_roll's three per-leg input lists from
        `prepared`'s ALREADY-fetched quotes (no second broker round-trip) --
        solves each leg's own IV via solve_iv_from_price, the SAME
        py_vollib call site the live chain scan already uses. Returns None
        (defer to the heuristic, never guess) if a REQUIRED (closing or
        opening) leg's quote or IV solve fails; a carried leg's own failure
        just drops it from the carried list -- ev_should_roll already
        treats an empty carried list as carried_net=0.0, the SAME fallback
        carried_value has elsewhere in this arc.
        """
        r = self._settings.options.risk_free_rate

        def _leg_input(leg: dict) -> dict | None:
            quote = prepared.quotes.get(leg.get("conId"))
            if not quote or quote[0] is None or quote[1] is None:
                return None
            mid = (quote[0] + quote[1]) / 2.0
            try:
                expiry = datetime.strptime(str(leg["expiry"])[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError, KeyError):
                return None
            t = max((expiry - now_et().date()).days / 365.0, 0.001)
            flag = "c" if str(leg.get("right", "")).upper().startswith("C") else "p"
            iv, outcome = solve_iv_from_price(mid, spot, float(leg["strike"]), t, r, flag)
            if outcome != "solved":
                return None
            return {"strike": float(leg["strike"]), "action": leg["action"],
                    "spot": spot, "t": t, "iv": iv, "flag": flag}

        closing_raw = [_leg_input(leg) for leg in prepared.closing_legs]
        if any(x is None for x in closing_raw):
            return None
        opening_raw = [_leg_input(leg) for leg in prepared.opening_legs]
        if any(x is None for x in opening_raw):
            return None
        closing_inputs: list[dict] = [x for x in closing_raw if x is not None]
        opening_inputs: list[dict] = [x for x in opening_raw if x is not None]
        carried_inputs: list[dict] = [
            x for x in (_leg_input(leg) for leg in prepared.carried_legs) if x is not None
        ]
        return closing_inputs, opening_inputs, carried_inputs

    async def _compare_close_or_roll(self, pos: PositionStatus) -> bool:
        """W54b: for a rule 3/rule 3e roll candidate (should_exit=True AND
        pos.roll_side set -- the dual shape only rule 3/3e ever produce,
        never condor_defense), decide whether to actually roll instead of
        closing. Called from _trading_cycle's exit loop, BEFORE
        _execute_exit, for exactly those positions.

        Returns True if a roll was ATTEMPTED (successfully or not -- the
        caller must NOT also call _execute_exit either way: a failed
        attempt already logged/handled itself via _execute_roll's own
        fail-safe returns, and should_exit=True will still be there next
        cycle to retry closing if the roll genuinely didn't happen).
        Returns False if closing should proceed exactly as it does today
        (execute_rolls off, no target/combo found, or the comparison
        itself says close).
        """
        cfg = self._settings.assignment_response
        if not cfg.execute_rolls:
            return False

        trade = self._find_trade_record(pos.trade_id)
        if not trade or not pos.roll_side:
            return False

        roll_target = await select_roll_target(
            trade, pos.roll_side, self._options_chain, cfg)
        if roll_target is None:
            return False

        prepared = await prepare_roll_combo(
            self._ibkr, trade, pos.roll_side, roll_target, trade.symbol,
            trade.quantity, order_ref=roll_order_ref(trade.trade_id, "PENDING"),
            debit_cap=-cfg.roll_min_credit,
        )
        if prepared is None:
            return False

        if cfg.policy == "roll_only":
            should_roll = True
        else:  # close_or_roll
            if cfg.comparison_model == "heuristic":
                should_roll = heuristic_should_roll(prepared)
            else:  # "ev" -- data-quality fallback first (W54a's own field)
                if roll_target.iv_is_guessed:
                    log.warning("ev_comparison_iv_guessed_fallback", trade_id=trade.trade_id)
                    should_roll = heuristic_should_roll(prepared)
                else:
                    ev_inputs = self._gather_ev_leg_inputs(prepared, pos.current_price)
                    if ev_inputs is None:
                        log.warning("ev_comparison_inputs_unavailable_fallback",
                                   trade_id=trade.trade_id)
                        should_roll = heuristic_should_roll(prepared)
                    else:
                        closing_i, opening_i, carried_i = ev_inputs
                        ev_verdict = ev_should_roll(
                            closing_i, opening_i, carried_i,
                            self._settings.options.risk_free_rate)
                        should_roll = (ev_verdict if ev_verdict is not None
                                      else heuristic_should_roll(prepared))

        if not should_roll:
            return False

        await self._execute_roll(pos, prepared=prepared, roll_target=roll_target)
        return True

    async def _execute_roll(
        self, pos: PositionStatus, prepared=None, roll_target: RollTarget | None = None,
    ) -> None:
        """W61: act on a ROLL verdict — select a replacement vertical,
        prepare/qualify/price the combo, seed the DB rows, and place it.
        Callers: condor_defense (gated in _trading_cycle on execute_rolls,
        a separate opt-in from condor_defense.enabled itself), and — as of
        W54b — rule 3/3e, via _compare_close_or_roll.

        `prepared`/`roll_target` (W54b): when the caller already has them
        (e.g. _compare_close_or_roll, which must prepare a combo to run the
        close-or-roll comparison in the first place), pass them through
        directly to skip a SECOND chain fetch / quote poll for the same
        decision — get_chain's own measured 110-209s median cost makes
        re-fetching for the same roll wasteful, not just slow. Fully
        backward compatible: condor_defense's own call site passes neither.

        Each step, cheapest/most-reversible first — W54a's roll-count cap
        first of all, before any chain fetch — so a failure partway through
        leaves the least mess — same principle roll_booking.book_roll_
        fill's own promote-B-first ordering already established. A
        placement failure at the end is logged and left for the
        reconciler's own stale-pending sweep to clean up (cancel/restore) on
        its normal 30-minute cadence — no new recovery path is invented
        here.
        """
        trade = self._find_trade_record(pos.trade_id)
        if not trade or not pos.roll_side:
            log.warning("roll_skipped_no_trade_or_side", trade_id=pos.trade_id)
            return

        # W54a: the cheapest possible check, first, before any chain fetch —
        # a hard cap on repeat-rolling ONE position, regardless of which
        # system triggered this call.
        depth = self._state.count_roll_ancestors(trade.trade_id)
        if depth >= self._settings.assignment_response.max_rolls_per_position:
            log.warning("roll_cap_reached", trade_id=trade.trade_id, depth=depth,
                       max_rolls=self._settings.assignment_response.max_rolls_per_position)
            return

        try:
            if roll_target is None:
                roll_target = await select_roll_target(
                    trade, pos.roll_side, self._options_chain,
                    self._settings.assignment_response,
                )
                if roll_target is None:
                    log.warning("roll_no_target_found", trade_id=trade.trade_id,
                               side=pos.roll_side)
                    return

            if prepared is None:
                # roll_min_credit is a CREDIT floor (leg_net_price's BUY+/
                # SELL- convention makes a net credit NEGATIVE) —
                # prepare_roll_combo's debit_cap is a DEBIT ceiling, so
                # "require at least $X credit" becomes "refuse mid_net >
                # -X": negate, don't pass through.
                prepared = await prepare_roll_combo(
                    self._ibkr, trade, pos.roll_side, roll_target, trade.symbol,
                    trade.quantity, order_ref=roll_order_ref(trade.trade_id, "PENDING"),
                    debit_cap=-self._settings.assignment_response.roll_min_credit,
                )
                if prepared is None:
                    log.warning("roll_prepare_failed", trade_id=trade.trade_id,
                               side=pos.roll_side)
                    return

            if not cas_roll_closing_trade(self._state, trade.trade_id):
                log.warning("roll_cas_to_closing_refused", trade_id=trade.trade_id)
                return

            opening = seed_roll_opening_trade(
                self._state, trade, prepared.opening_legs, roll_target,
                trade.quantity, entry_price_estimate=prepared.mid_net,
            )
            # PreparedRollCombo.order is deliberately typed `object` (avoids
            # importing roll_executor.py's own ib_insync-typed Order here —
            # see place_roll_combo's docstring on the circular-import reason).
            prepared.order.orderRef = (  # type: ignore[attr-defined]
                roll_order_ref(trade.trade_id, opening.trade_id))

            placed = await self._executor.place_roll_combo(
                prepared, trade.trade_id, opening.trade_id,
                prepared.carried_legs, prepared.carried_value,
            )
            if not placed:
                log.error("roll_placement_failed", trade_id=trade.trade_id,
                         opening_trade_id=opening.trade_id)
                return

            log.info("roll_order_submitted", trade_id=trade.trade_id,
                     opening_trade_id=opening.trade_id, side=pos.roll_side)

        except Exception as e:  # noqa: BLE001 — a roll attempt must never crash the cycle
            log.error("roll_execution_failed", trade_id=pos.trade_id, error=str(e))
            self._watchdog.record_error("trading_loop", f"roll failed: {e}")

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
            if not self._cas_to_closing(trade):
                return None
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
        if not self._cas_to_closing(trade):
            return None
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
                    # W20a (R25 tz-01): ET, not the host's local clock. This
                    # value is PERSISTED into trade_context.entry_signals and
                    # becomes training data — a naive-local hour here writes a
                    # CEST value into a corpus whose historical rows are ET.
                    from ait.utils.time import now_et as _now_et
                    out[name] = float(_now_et().hour)
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

    async def _synthesize_combo_cost(self, qualified_legs: list[dict],
                                     poll_rounds: int = 4,
                                     credit_buyback: bool = False) -> float | None:
        """W7 (R24 logic-exit-risk-01): build the combo's marketable cost from
        the FOUR LEG quotes when the BAG contract never ticks.

        Root cause of the 08-31 incident: every one of 63 exits took the
        no-quote branch (``combo_exit_limit`` appears zero times in any
        retained log) and was priced off a formula instead of the market,
        while the ENTRY path's Bag quotes arrive normally. Legs quote
        reliably, so synthesize: a leg we BUY back costs its ask, a leg we
        SELL returns its bid. Returns the as-defined cost for the reversed
        combo (positive = we pay), or None if the cost cannot be stated
        without understating it.

        W33 (R26 zero-bid-sold-wing / one-sided-leg): every leg is priced from
        its CONSERVATIVE side and never from the other one.
          - a leg we BUY needs an offer. Its bid understates the cost, so a
            missing ask returns None (the caller's fallback is then a
            decision, not an accident).
          - a leg we SELL is worth its bid, and 0.00 IS a bid — the normal
            end-state of a winning condor's wing. W7 mapped it to "no quote"
            (``b > 0``) and either aborted the whole synthesis over one
            worthless wing (7 of 7 exits on 09-11) or, given an ask, sold the
            wing at its ASK.
          - IBKR's -1 is a statement too ("nobody bids"), distinct from NaN or
            a silent ticker (no data arrived). On a CREDIT buyback a bidless
            wing contributes 0 — assuming we receive nothing can only raise
            the stated cost of a structure whose value sits in the legs we
            buy. Anywhere else the sold legs ARE the value (closing a long
            straddle would be "priced" at zero and given away), so the caller
            must opt in with ``credit_buyback``; without it, and for missing
            data in every case, the answer is None.
        """
        import math as _m
        total = 0.0
        shown: dict[str, str] = {}
        self._last_leg_poll = shown          # W33 telemetry, read by the caller
        for leg in qualified_legs:
            contract = leg.get("_contract")
            if contract is None:
                return None
            is_buy = str(leg.get("action", "")).upper() == "BUY"
            key = str(leg.get("conId"))
            bid = ask = None
            shown[key] = "none"
            try:
                self._ibkr.ib.reqMktData(contract, "", False, False)
                for _ in range(poll_rounds):
                    await asyncio.sleep(0.5)
                    t = self._ibkr.ib.ticker(contract)
                    if not t:
                        continue
                    b, a = t.bid, t.ask
                    shown[key] = (f"{b:g}" if b is not None else "none") + "/" + (
                        f"{a:g}" if a is not None else "none")
                    bid = b if (b is not None and not _m.isnan(b) and b >= 0) else None
                    if (bid is None and credit_buyback and b is not None
                            and not _m.isnan(b) and b == -1):
                        bid = 0.0                 # IBKR: "no bid exists"
                    ask = a if (a is not None and not _m.isnan(a) and a > 0) else None
                    if (ask if is_buy else bid) is not None:
                        break
            except Exception as e:  # noqa: BLE001
                log.debug("leg_quote_failed", error=str(e))
                return None
            finally:
                try:
                    self._ibkr.ib.cancelMktData(contract)
                except Exception:  # noqa: BLE001
                    pass
            ratio = float(leg.get("ratio", 1) or 1)
            if is_buy:
                # buying this leg back — we pay the offer, never the bid
                if ask is None:
                    log.warning("leg_quote_unavailable",
                                con_id=leg.get("conId"), action="BUY",
                                bid=bid, shown=shown[key],
                                note="no offer on a leg we must buy; its bid "
                                     "would understate the cost to close")
                    return None
                total += ratio * ask
            else:
                # selling this leg — we receive the bid (0.00 included),
                # never the ask
                if bid is None:
                    log.warning("leg_quote_unavailable",
                                con_id=leg.get("conId"), action="SELL",
                                shown=shown[key],
                                note="no bid data on a leg we sell; its ask "
                                     "would understate the cost to close")
                    return None
                total -= ratio * bid
        return round(total, 2)

    def _booked_commission_estimate(self, trade) -> float:
        """W37: the commission estimate the FILL actually subtracted.

        Falls back to today's formula for closes booked before W37 (no key) or
        when the store cannot be read — the pre-W37 behaviour, never worse.
        """
        try:
            raw = self._state.get_state(f"commission_est_{trade.trade_id}", "")
            # Only a stored STRING is a remembered estimate (a mocked store
            # returns a MagicMock, and float(MagicMock()) is 1.0).
            if isinstance(raw, str) and raw:
                return float(raw)
        except Exception:  # noqa: BLE001
            pass
        from ait.execution.executor import TradeExecutor as _TE
        return float(_TE.commission_estimate(trade))

    def _forget_commission_estimate(self, trade_id: str) -> None:
        try:
            self._state.delete_state(f"commission_est_{trade_id}")
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _archived_partial_pnl(trade) -> float:
        """W50 (money-flow-05 / trade-life-partial-slice-pnl-erased): sum of
        partial-exit slices close_trade already folded into this trade's
        realized_pnl (state.py:close_trade, before the open_positions row is
        deleted). Those slices were ALSO fed to daily_stats/breaker in real
        time as each one closed — executor._book_partial_exit and
        orchestrator._execute_partial_exit both call record_partial_pnl — so
        the MONEY booked at final close must exclude them, or they are
        counted twice. Reads trades.partial_exits (the archive close_trade
        writes) rather than anything threaded through completed_exits or the
        exit_outbox row, so the live callback path and the crash-recovery
        drain derive the identical number from the identical source.
        """
        try:
            if not trade or not getattr(trade, "partial_exits", None):
                return 0.0
            import json as _json
            return round(sum(float(e.get("pnl", 0.0))
                             for e in _json.loads(trade.partial_exits)), 2)
        except Exception:  # noqa: BLE001 — never block booking on this
            return 0.0

    @staticmethod
    def _legs_count(trade) -> int:
        """W48: expected leg count for a trade's completeness math (the
        commission true-up needs `2 * legs_n` execution rows — entry + exit
        per leg). Extracted so both true-up call sites derive it the same
        way; previously inline in `_true_up_commission` only."""
        try:
            import json as _json
            return max(1, len(_json.loads(trade.legs))) if trade and trade.legs else 4
        except Exception:  # noqa: BLE001
            return 4

    @staticmethod
    def _signal_sells_a_call(signal) -> bool:
        """W51 (model-vs-reality-ex-div-assignment-void): does this SIGNAL
        (pre-trade — not yet a TradeRecord) sell a call leg? The only shape
        the ex-div entry gate cares about, since that's the leg an early
        assignment actually hits. Extracted as its own testable unit rather
        than left inline in _try_execute, matching _legs_count/
        _archived_partial_pnl above.

        signal.legs carries {"contract": OptionContract, "action": ...} (the
        pre-serialization in-memory shape) — NOT trade.legs' flat JSON dicts
        used by portfolio.py's rule 3e (the exit-side twin of this gate),
        which is a genuinely different shape, not just a naming accident:
        a Signal is built once by a strategy and never round-tripped through
        JSON until execute_signal writes the trade row.
        """
        if signal.legs:
            return any(
                str(leg.get("action", "")).upper() == "SELL"
                and getattr(leg.get("contract"), "right", None) == "C"
                for leg in signal.legs
            )
        return (str(signal.action).upper() == "SELL"
               and signal.contract is not None and signal.contract.right == "C")

    def _settle_deferred_trueups(self) -> None:
        """R16: settle deferred commission true-ups once the day's
        commissionReports have all landed (the booking-time ledger was partial).
        W37: extracted from _post_market so it can be driven for real, and
        reverses the REMEMBERED estimate rather than re-deriving one.

        W48 (trade-life-commission-trueup-gate-measures-wrong-quantity, Path
        B): this used to have NO completeness check at all -- any nonzero
        `_real` settled, even against a still-partial ledger, permanently
        overstating realized_pnl by whatever exit-side commissions hadn't
        landed yet; and the `trueup_pending_*` marker was deleted
        UNCONDITIONALLY, so a trade caught mid-ledger never got a second
        chance. Now mirrors `_true_up_commission`'s own completeness gate
        exactly (excluding the BAG summary row, which reports its own
        -- genuinely real -- commission before leg rows do; see
        `count_executions_with_commission`'s docstring) and keeps the marker
        set, not deleted, whenever the ledger is still incomplete -- the
        NEXT post-market run (or a later session) retries.
        """
        for _tid in self._state.pending_trueup_trade_ids():
            try:
                _t = self._find_trade_by_id(_tid)
                if _t is None:
                    # No trade row for this marker (should not happen in
                    # practice) -- nothing to ever settle against; clear the
                    # orphan rather than retry it forever.
                    self._state.delete_state(f"trueup_pending_{_tid}")
                    continue
                _legs_n = self._legs_count(_t)
                _exec_n = self._state.count_executions_with_commission(
                    _tid, exclude_con_id=BAG_CON_ID)
                if _exec_n < 2 * _legs_n:
                    log.info("commission_trueup_deferred_partial_ledger_post_market",
                             trade_id=_tid, executions=_exec_n,
                             expected=2 * _legs_n)
                    continue
                _real = self._state.total_commission(_tid)
                if _real > 0 and _t.realized_pnl is not None:
                    _delta = round(self._booked_commission_estimate(_t) - _real, 2)
                    if abs(_delta) >= 0.01:
                        _new_pnl = round(float(_t.realized_pnl) + _delta, 2)
                        self._state.update_trade_realized_pnl(_tid, _new_pnl)
                        _st = self._state.get_daily_stats()
                        _st.total_pnl = round(_st.total_pnl + _delta, 2)
                        self._state.update_daily_stats(_st)
                        log.info("commission_trueup_settled_post_market",
                                 trade_id=_tid, delta=_delta)
                    self._forget_commission_estimate(_tid)
                    # W48: Path A already stamped trades.commission with
                    # broker truth once complete; Path B never did, leaving
                    # the PF verdict's commission line at whatever the
                    # fill-time estimate happened to write.
                    try:
                        self._state.update_trade_commission(_tid, _real)
                    except Exception:  # noqa: BLE001
                        pass
                self._state.delete_state(f"trueup_pending_{_tid}")
            except Exception as _e:  # noqa: BLE001
                log.warning("post_market_trueup_failed", trade_id=_tid,
                            error=str(_e))

    def _true_up_commission(self, trade_id: str, pnl: float) -> float:
        """Replace the fill-time commission ESTIMATE inside a realized P&L
        with the broker's actual ledger figure. Returns the corrected P&L.

        The fill-time booking subtracts commission_estimate(); the executions
        ledger carries what IBKR really charged. Measured on our own 94 leg
        fills, per-leg cost varies $0.62-$1.06 with exchange routing, so the
        estimate is never exactly right and the ledger must win.

        R16: commissionReports for the exit legs arrive ASYNC — truing up
        against a partial ledger (entry legs only) overstates realized_pnl
        permanently, so require a COMPLETE ledger (>= 2x leg count) and
        otherwise defer to the post-market re-true-up.

        W48 (trade-life-commission-trueup-gate-measures-wrong-quantity): the
        completeness gate used to be `count_executions(trade_id) >= 2*legs_n`
        -- raw ROW presence. `_sweep_executions` (executor.py) writes a row
        the instant a fill is seen, commission-known or not (0.0 either way,
        pre-W48), and combo orders add a BAG summary row on top -- so a
        4-leg condor's 10 total rows (8 legs + 2 BAG) satisfied `>= 8` while
        real leg commissions were still outstanding. Now gates on rows whose
        commission is actually KNOWN (`count_executions_with_commission`,
        NULL until a real commissionReport lands), excluding the BAG row
        (which reports its own -- genuinely real, typically 0.0 --
        commission before the legs do, and would otherwise let a couple of
        quick BAG reports substitute for still-pending leg ones).

        R24 regress-live-09 / pnl-money-02: this used to live inline in
        _process_completed_exits only, so any close booked by the outbox
        drain (crash between the close and the callback, or a callback later
        than the drain's grace period) kept the estimate forever and left
        trades.commission at 0.0 — a one-directional overstatement of the
        go-live P&L. One authority, both paths.
        """
        try:
            real = self._state.total_commission(trade_id)
            trade = self._find_trade_by_id(trade_id)
            legs_n = self._legs_count(trade)
            exec_n = self._state.count_executions_with_commission(
                trade_id, exclude_con_id=BAG_CON_ID)
            if real > 0 and trade and exec_n >= 2 * legs_n:
                delta = round(self._booked_commission_estimate(trade) - real, 2)
                self._forget_commission_estimate(trade_id)
                if abs(delta) >= 0.01:
                    pnl = round(pnl + delta, 2)
                    self._state.update_trade_realized_pnl(trade_id, pnl)
                    log.info("realized_pnl_commission_trueup",
                             trade_id=trade_id, delta=delta,
                             real_comm=round(real, 2))
                # Stamp the real figure so the PF verdict reads broker truth.
                try:
                    self._state.update_trade_commission(trade_id, real)
                except Exception:  # noqa: BLE001
                    pass
            elif trade:
                self._state.set_state(f"trueup_pending_{trade_id}",
                                      datetime.now().date().isoformat())
                log.info("commission_trueup_deferred_partial_ledger",
                         trade_id=trade_id, executions=exec_n,
                         expected=2 * legs_n)
        except Exception as e:  # noqa: BLE001 — booking must never die here
            log.warning("commission_trueup_failed",
                        trade_id=trade_id, error=str(e))
        return pnl

    async def _cooldown_blocks_entry(self, signal) -> bool:
        """R12-B4 post-stop re-entry discipline. True = refuse this entry.

        The old cooldown keyed on ENTRY time, so a symbol stopped out on day 3
        of a trend could re-enter on the very next scan into the same move —
        the most autocorrelated loss sequence condors produce.

        W15 (R24 regress-live-07 / time-calendar-02): the caller used to
        swallow every exception with a bare `pass`. W4 had just given this
        gate a NYSE-calendar dependency and a fresh sqlite connection, so a
        calendar fault or a busy/locked DB silently DISABLED the guard for
        every symbol on every scan — no log line, no page — and the bot could
        re-enter the symbol it was just stopped out of (the 2026-08-24 QQQ
        shape). Now fails CLOSED for credit structures, matching how W2
        hardened the sibling calendar-dependent gates (PDT fail-direction-09,
        the executor's market-hours guard fail-direction-12). Debit shapes
        keep the permissive behaviour: they are not the autocorrelated-loss
        case this rule exists for.
        """
        try:
            block = self._post_stop_cooldown_until(signal.symbol)
            if block:
                # W26: stopped_symbol is NOT always symbol — a cluster sibling's
                # stop blocks too now, and the operator needs the cause named.
                log.info("entry_blocked_post_stop_cooldown",
                         symbol=signal.symbol, stopped_symbol=block.symbol,
                         stopped_at=block.exit_time,
                         via_cluster=block.symbol != signal.symbol)
                return True
            return False
        except Exception as e:  # noqa: BLE001
            log.error("post_stop_cooldown_check_failed_fail_closed",
                      symbol=signal.symbol, error=str(e))
            try:
                from ait.strategies.base import CREDIT_STRATEGIES as _CS15
                is_credit = signal.strategy_name in _CS15
            except Exception:  # noqa: BLE001
                is_credit = True
            if not is_credit:
                return False
            try:
                if self._alert_gate(f"cooldown_unavailable:{signal.symbol}", 3600):
                    await self._send_notification(
                        f"COOLDOWN CHECK FAILED ({signal.symbol}): the "
                        f"post-stop re-entry guard could not run "
                        f"({type(e).__name__}). Credit entries are blocked "
                        f"until it recovers.")
            except Exception:  # noqa: BLE001 — paging must not un-block the gate
                pass
            return True

    def _entry_range_context(self, symbol: str) -> dict:
        """W10: range-model fields for save_trade_context, from this scan's
        stashed prediction. All None when the model produced nothing — that
        is the real answer for a symbol whose model failed its edge gate, and
        it must be distinguishable from a genuine low probability."""
        pred = None
        try:
            pred = (getattr(self, "_entry_range_pred", None) or {}).get(symbol)
        except Exception:  # noqa: BLE001
            pred = None
        mode = ("armed" if self._settings.ml.entry_gates_enabled else "observe")
        if pred is None:
            return {"range_probability": None, "range_threshold_pct": None,
                    "range_horizon_days": None, "range_model_version": None,
                    "entry_gates_mode": mode}
        return {
            "range_probability": float(pred.probability_in_range),
            "range_threshold_pct": float(pred.threshold_pct),
            "range_horizon_days": int(pred.horizon_days),
            "range_model_version": getattr(pred, "model_version", "") or "",
            "entry_gates_mode": mode,
        }

    def _exit_band_cap(self, trade: TradeRecord,
                       mark_cost: float | None = None) -> float | None:
        """W7: the acceptable limit IBKR stated when it rejected the previous
        exit attempt (executor.parse_price_band_reject persists it).

        Pre-W7 the reject reason was discarded and the identical unexecutable
        limit was re-placed every backoff — 61 of 63 exits on 08-31, a
        take-profit looping 32 min and 2h55m.

        W33: until W33 no band was ever captured, so W7's one-hour lifetime
        was never exercised. Live, it is a hazard: a band stated against a
        take-profit at 13:31 would cap a touch stop at 13:50 BELOW a market
        that had moved through it, and an order under the market draws no
        new reject to correct it. A band is the broker's view of one moment:
        it expires after ``exit_band_ttl_seconds`` (the reject backoff is
        30-180s, so a live reject loop keeps refreshing it), and it is
        ignored outright once our own fresh mark has risen to or through it.
        If it still applies, the broker says so again — one reject cycle.
        """
        try:
            import json as _json
            raw = self._state.get_state(f"exit_band_{trade.trade_id}", "")
            if not raw:
                return None
            blob = _json.loads(raw)
            at = blob.get("at")
            ttl = float(getattr(self._settings.exit,
                                "exit_band_ttl_seconds", 300.0))
            if at and (datetime.now()
                       - datetime.fromisoformat(at)).total_seconds() > ttl:
                return None
            _b = blob.get("band")
            if (mark_cost is not None and _b is not None
                    and float(mark_cost) >= float(_b)):
                log.warning("exit_band_ignored_mark_has_moved_through_it",
                            trade_id=trade.trade_id, band=_b,
                            mark_cost=round(float(mark_cost), 4))
                return None
            band = blob.get("band")
            if band is None:
                mkt = blob.get("market")
                # No explicit band, but the broker named the market price:
                # a buyback just above it is the marketable, acceptable side.
                return round(float(mkt) * 1.05, 2) if mkt else None
            return float(band)
        except Exception:  # noqa: BLE001 — a bad band must never block an exit
            return None

    def _exit_price_ceiling(self, mark_cost: float, width: float,
                            exit_cross: float) -> float:
        """W7 (R24 logic-exit-risk-01): the most we may pay to buy back a
        credit structure, anchored to what it is CURRENTLY worth.

        The R16 bound was ``max(2*mark, entry_credit + 0.25*wing_width)``.
        That second term was meant as a floor for an unreliable mark, but it
        sits inside a max(), so at the promoted $39-60 wings it DOMINATED:
        a SPY condor marked at 1.40 priced its take-profit at
        4.29 + 0.25*39 = 14.04 — ten times the mark. IBKR's price band
        rejected 61 of 63 exits (the broker's guard was the only thing
        bounding the price) and the two that filled did so at NBBO by
        exchange price improvement, not because anything here was correct.

        The wing width stays the structural cap (a deep-ITM condor really is
        worth ~width). ``mark + exit_cross`` is the floor, so the order
        remains marketable at a near-worthless buyback.

        W33 (R26 mark-ceiling-clamps-below-the-ask-at-low-marks): the ceiling
        is compared against an ASK-side price while the mark is a mid, so it
        must leave room for the half-spread — and a four-leg combo's summed
        half-spread does not shrink with the mark (0.52-0.58 measured on QQQ
        09-11). A purely proportional 1.5x gave a 0.60 mark 0.30 of headroom
        and overwrote a marketable 1.18 synthesized limit with an unmarketable
        0.90. ``exit_min_ceiling_headroom`` is the floor on that headroom; at
        the incident marks (1.40, 2.94, 8.76) the multiple still binds and the
        ceilings are unchanged.
        """
        mult = float(self._settings.exit.exit_mark_multiple)
        headroom = float(getattr(self._settings.exit,
                                 "exit_min_ceiling_headroom", 0.0) or 0.0)
        ceiling = mark_cost + max(exit_cross, (mult - 1.0) * mark_cost, headroom)
        if width:
            ceiling = min(width, ceiling)
        return round(ceiling, 2)

    #: Combo (BAG) limit prices on US equity options are penny-incremented.
    _EXIT_TICK = 0.01

    @classmethod
    def _under_band(cls, band: float) -> float:
        """W33 (R26 band-cap-clamps-at-or-above-the-band): the highest limit
        IBKR will take. Its text is "at or MORE aggressive than 1.745", so
        1.745 itself is refused and ``round(1.745, 2)`` is 1.75 or 1.74 by
        float accident. One whole tick strictly below: 1.745 -> 1.74,
        3.50 -> 3.49."""
        import math as _m
        ticks = _m.ceil(round(float(band) / cls._EXIT_TICK, 6)) - 1
        return round(ticks * cls._EXIT_TICK, 2)

    def _mark_cost_and_age(self, trade: TradeRecord) -> tuple[float | None, float | None]:
        """(per-share cost to close per the monitor's marks, mark age in s).

        unrealized = (entry_credit - current_cost) * 100 * qty, so
        current_cost = entry_credit - unrealized/(100*qty).
        """
        try:
            mark = self._state.get_position_mark(trade.trade_id)
            if not mark or not trade.entry_price or not trade.quantity:
                return None, None
            age = (datetime.now()
                   - datetime.fromisoformat(mark["mark_time"])).total_seconds()
            cost = float(trade.entry_price) - (
                float(mark["unrealized_pnl"]) / (100.0 * trade.quantity))
            return max(0.0, cost), age
        except Exception:  # noqa: BLE001 — a broken mark must never block an exit
            return None, None

    def _marked_cost_to_close(self, trade: TradeRecord) -> float | None:
        """Per-share cost to close a CREDIT structure per the monitor's marks.

        Returns None when the mark is absent or stale (>15 min). A stale mark
        is still the best anchor the no-quote branch has — it reads it through
        _mark_cost_and_age (W33) rather than falling back to the wing width.
        """
        cost, age = self._mark_cost_and_age(trade)
        if cost is None or age is None or age > 900:
            return None
        return cost

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
                # W7: retained so the exit can synthesize a combo quote from
                # the individual legs when the BAG itself never ticks.
                "_contract": qualified,
            })

        combo = ContractBuilder.combo(
            symbol=trade.symbol,
            legs=[{k: v for k, v in l.items() if not k.startswith("_")}
                  for l in qualified_legs],
        )

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
        # W33 (R26 stale-exit-quality-tuple): whatever a previous exit left
        # here belongs to another order. An exit that returned without placing
        # (broker reject, deferral) never consumed its tuple, and the no-quote
        # branch never refreshed it, so the NEXT trade's slippage record — a
        # go-live gate input — carried the wrong trade's numbers.
        self._last_exit_quality = (0.0, 0.0, 0.0)
        self._last_leg_poll = {}
        _bag_shown = "none"
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
                    _bag_shown = (
                        (f"{ticker.bid:g}" if ticker.bid is not None else "none")
                        + "/"
                        + (f"{ticker.ask:g}" if ticker.ask is not None else "none"))
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
                        # W33b: a quoted exit that did NOT fill must not
                        # repeat itself. First live exit on W33, 2026-09-21,
                        # DELAYED data (warning 10167 on this very BAG): BUY
                        # LMT 8.67 off an 8.46/8.57 quote worked 11 minutes
                        # unfilled; the re-price, 9.76 off 9.50/9.66, likewise
                        # — each built from a quote ~15 minutes old in a
                        # rising tape, with no memory that the last one
                        # failed. Every consecutive unfilled placement for
                        # this trade now raises the limit by a fraction of the
                        # marketable price. Credit buybacks only (a positive
                        # price we PAY); the caps below still bind; it is a
                        # LIMIT, so it fills at the offer, not at our number.
                        if (limit_price is not None and limit_price > 0
                                and trade.strategy in CREDIT_STRATEGIES):
                            import time as _time_q
                            _qa: dict = getattr(self, "_exit_quoted_attempts", None)
                            if _qa is None:
                                _qa = self._exit_quoted_attempts = {}
                            _qrec = _qa.get(trade.trade_id)
                            _qnow = _time_q.monotonic()
                            _qn = (_qrec["n"] if _qrec
                                   and (_qnow - _qrec["at"]) < 3600.0 else 0)
                            _qa[trade.trade_id] = {"n": _qn + 1, "at": _qnow}
                            if _qn:
                                _pct = float(getattr(
                                    self._settings.exit,
                                    "exit_unfilled_escalation_pct", 0.05))
                                _base = limit_price
                                limit_price = round(_base * (1.0 + _pct * _qn), 2)
                                log.warning(
                                    "combo_exit_limit_escalated",
                                    symbol=trade.symbol, trade_id=trade.trade_id,
                                    prior_unfilled_attempts=_qn,
                                    marketable=_base, escalated_to=limit_price)
                    else:  # SELL
                        if bid is not None:
                            limit_price = round(bid - EXIT_CROSS, 2)
                        elif ask is not None:
                            limit_price = round(ask - EXIT_CROSS, 2)
                    # W12 (R24 data-integrity-08): stash the EXIT's fill-
                    # quality context. Entries have recorded this since R19;
                    # exits never did, so every measured slippage figure
                    # covered only half the round trip.
                    _mk_q = self._marked_cost_to_close(trade)
                    _two_sided = bid is not None and ask is not None
                    self._last_exit_quality = (
                        round(_mk_q, 4) if _mk_q else 0.0,
                        round((bid + ask) / 2.0, 4) if _two_sided else 0.0,
                        round(abs(ask - bid), 4) if _two_sided else 0.0,
                    )
        except Exception as e:
            log.warning("combo_mid_price_failed", error=str(e))

        # W7 (R24 logic-exit-risk-01): the BAG stayed silent on 63 of 63 exits
        # while the same account quotes legs fine. Synthesize the combo's
        # marketable cost from the legs rather than falling through to a
        # formula-priced order.
        if limit_price is None:
            synth = await self._synthesize_combo_cost(
                qualified_legs,
                credit_buyback=trade.strategy in CREDIT_STRATEGIES)
            if synth is not None:
                limit_price = round(
                    synth + EXIT_CROSS if combo_action == "BUY"
                    else synth - EXIT_CROSS, 2)
                log.info("combo_exit_limit_from_legs", symbol=trade.symbol,
                         trade_id=trade.trade_id, synthesized_cost=synth,
                         action=combo_action, limit_price=limit_price,
                         note="BAG never ticked; priced from the four legs")
                # W12: the synthesized cost IS the market's marketable price
                # here; no two-sided combo NBBO exists to measure a spread.
                _mk = self._marked_cost_to_close(trade)
                self._last_exit_quality = (round(_mk, 4) if _mk else 0.0,
                                           round(abs(synth), 4), 0.0)

        # W33: what the feed actually showed on this attempt. The 09-11
        # diagnosis (which leg was dead, and whether it was 0.00 or NaN) had to
        # be reconstructed from a probe because nothing logged it.
        log.info("combo_exit_quote_poll", symbol=trade.symbol,
                 trade_id=trade.trade_id, bag=_bag_shown,
                 legs=dict(getattr(self, "_last_leg_poll", {}) or {}),
                 priced=limit_price is not None)

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
            anchor = self._exit_price_ceiling(mark_cost, width, EXIT_CROSS)
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

        # W7 (R24 logic-exit-risk-02): the broker told us, on the previous
        # attempt, the most aggressive price it will accept. Honour it — this
        # is what turns an endless re-place loop into a converging exit. The
        # band is a CAP for a BUY-to-close; the exit still has to be at least
        # marketable, so never drop below the marked cost plus the cross.
        # W33: "at or more aggressive than X" refuses X itself. The old clamp
        # resolved to exactly the band on every input (max(min(l, b), min(f, b))
        # with l > b is b), so the re-priced order was rejected again. One tick
        # under it is the most aggressive price the broker will take; a
        # mark-derived floor cannot lift it back over.
        _band = self._exit_band_cap(trade, mark_cost)
        if _band is not None and limit_price is not None:
            _under = self._under_band(_band)
            if limit_price > _under:
                log.warning(
                    "combo_exit_limit_capped_at_broker_band",
                    symbol=trade.symbol, trade_id=trade.trade_id,
                    requested=limit_price, band=_band, capped_to=_under,
                )
                limit_price = _under

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
            #
            # W33 (R26 no-quote-fallback-prices-at-the-ceiling /
            # raw-wing-width-fallback-still-reachable): W7 used the CEILING as
            # the order price — 1.5x the mark, 0.94-1.14 above IBKR's band on
            # 7 of 7 attempts on 09-11 — re-placed the identical price every
            # backoff, and with a mark 901s old priced at the raw width ($39-60
            # wings: the structural max loss). Now every no-quote order is
            #   anchor + cross + n * step, capped — never the cap itself:
            #   fresh mark : anchor = mark,        cap = ceiling(mark)
            #   stale mark : anchor = last mark,   cap = max(ceiling, blind cap)
            #   no mark    : anchor = entry credit, cap = blind cap
            # where the blind cap is min(width, k * entry credit): what we may
            # pay with no knowledge of the present. n counts this trade's
            # consecutive no-quote placements, so an order that rests unfilled
            # walks toward the offer instead of repeating itself. The broker
            # band, once stated, still caps all of it.
            _credit = float(trade.entry_price or 0.0)
            _blind_cap = (round(min(width, _credit * float(getattr(
                self._settings.exit, "exit_blind_credit_multiple", 2.0))), 2)
                if _credit > 0 else None)
            _anchor, _age, _basis = mark_cost, None, "fresh_mark"
            _cap = (self._exit_price_ceiling(mark_cost, width, EXIT_CROSS)
                    if mark_cost is not None else None)
            if _anchor is None:
                _stale, _age = self._mark_cost_and_age(trade)
                if _stale is not None:
                    _anchor, _basis = _stale, "stale_mark"
                    _cap = max(self._exit_price_ceiling(_stale, width, EXIT_CROSS),
                               _blind_cap or 0.0)
                elif _blind_cap is not None:
                    _anchor, _basis, _cap = _credit, "entry_credit", _blind_cap
            if _anchor is None or _cap is None:
                log.critical(
                    "combo_exit_deferred_no_quote_no_anchor",
                    symbol=trade.symbol, trade_id=trade.trade_id,
                    wing_width=width,
                    note="no quote, no mark and no entry credit — any price "
                         "would be a guess; refusing, will retry next cycle",
                )
                if self._alert_gate(f"exit_noquote:{trade.trade_id}"):
                    await self._send_notification(
                        f"EXIT DEFERRED (no quotes, no mark): {trade.symbol} "
                        f"{trade.strategy} — nothing to price the close from. "
                        f"Retrying every cycle. Check the data feed."
                    )
                return None

            import time as _time
            _nq: dict = getattr(self, "_exit_noquote_attempts", None)
            if _nq is None:
                _nq = self._exit_noquote_attempts = {}
            _rec = _nq.get(trade.trade_id)
            _now_m = _time.monotonic()
            _n = (_rec["n"] if _rec and (_now_m - _rec["at"]) < 3600.0 else 0)
            _nq[trade.trade_id] = {"n": _n + 1, "at": _now_m}
            _step = float(getattr(self._settings.exit, "exit_reprice_step", 0.10))
            fallback = round(min(_cap, _anchor + EXIT_CROSS + _n * _step), 2)
            _band_fb = self._exit_band_cap(trade, mark_cost)
            if _band_fb is not None:
                fallback = min(fallback, self._under_band(_band_fb))
            self._last_exit_quality = (round(_anchor, 4), 0.0, 0.0)
            log.critical(
                "combo_exit_no_quote_pricing_from_anchor",
                symbol=trade.symbol, trade_id=trade.trade_id,
                wing_width=width, mark_cost=mark_cost, basis=_basis,
                anchor=round(_anchor, 4),
                mark_age_s=None if _age is None else int(_age),
                attempt=_n + 1, cap=_cap, band=_band_fb, limit_price=fallback,
            )
            if self._alert_gate(f"exit_noquote:{trade.trade_id}"):
                await self._send_notification(
                    f"EXIT WITHOUT QUOTES: {trade.symbol} {trade.strategy} — "
                    f"no combo NBBO; pricing the close from the "
                    f"{_basis.replace('_', ' ')} at ${fallback:.2f} "
                    f"(cap ${_cap:.2f}, stepping each attempt). Check data "
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
        if not self._cas_to_closing(trade):
            return None
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

        # R16: settle deferred commission true-ups (see the method).
        self._settle_deferred_trueups()

        # W37: a remembered commission estimate is only needed until its
        # true-up has run; sweep the ones no open or still-pending trade owns
        # (same orphan class as trade_maxloss_* above).
        try:
            _keep = {t.trade_id for t in self._state.get_open_trades()} | set(
                self._state.pending_trueup_trade_ids())
            for _k in self._state.state_keys_like("commission_est_%"):
                if _k[len("commission_est_"):] not in _keep:
                    self._state.delete_state(_k)
        except Exception as _e:  # noqa: BLE001
            log.warning("commission_est_key_sweep_failed", error=str(_e))

        # 1. Reconcile with IBKR
        recon = await self._reconciler.reconcile()

        # W37 (time-calendar-03): the trading cycle stops at the bell, so
        # nothing drained after this reconcile — the one that discovers manual
        # flattens and assignments. Rows younger than the drain's 120s grace
        # stay for the next session, where they are now credited to TODAY's
        # date instead of being dropped.
        try:
            await self._drain_exit_outbox()
        except Exception as _e:  # noqa: BLE001 — never block the rest of post-market
            log.warning("post_market_outbox_drain_failed", error=str(_e))

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
            f"DAILY SUMMARY ({today_et().isoformat()})\n"
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

    async def _snapshot_marks_on_shutdown(self) -> None:
        """Closes the bot-day-09 gap (W47 registered, not fixed): stamp each
        open position's mark one last time before the process actually
        exits, instead of leaving it up to whatever the last regular
        FAST_MONITOR_CADENCE_SECONDS (~30s) pass happened to persist.

        Deliberately just one more check_positions() call -- the SAME call
        _monitor_positions_fast already makes every cycle, under the SAME
        _get_position_lock -- so this reuses 100% of the existing quote-
        fetch/marks-missing/persistence machinery (portfolio.py's
        update_position_mark) rather than inventing new mechanics. Read-
        only from a TRADING perspective: check_positions() never places an
        order or triggers an exit itself (that dispatch lives in
        _trading_cycle, never called here) -- it only returns a verdict per
        position, which this method discards.

        Safe to call here specifically because IBKR is still connected at
        this point in the shutdown sequence (main.py's ibkr.disconnect()
        runs only after run()/_shutdown() both return), and because nothing
        else can be concurrently evaluating positions by the time
        _shutdown() is ever reached: _protection_task_loop (the only other
        _get_position_lock holder) is unconditionally cancelled and awaited
        in _trading_loop's own `finally` before _trading_loop returns, and
        that return is itself a precondition for run()'s outer loop to
        exit and _shutdown() to run.

        Bounded and fail-safe: a slow quote or a broker hiccup here must
        never block or crash the rest of shutdown (the notification below
        is the more important signal that the bot is actually going away).
        10s leaves room inside the 30s total GRACEFUL_SHUTDOWN_TIMEOUT
        (main.py) alongside the notify-drain's own 15s sub-budget just
        below.
        """
        try:
            async with self._get_position_lock():
                positions = await asyncio.wait_for(
                    self._portfolio.check_positions(),
                    timeout=SHUTDOWN_MARKS_SNAPSHOT_TIMEOUT_SECONDS)
            log.info("shutdown_marks_snapshotted", count=len(positions))
        except asyncio.TimeoutError:
            log.warning("shutdown_marks_snapshot_timed_out")
        except Exception as e:  # noqa: BLE001 — must never block shutdown
            log.warning("shutdown_marks_snapshot_failed", error=str(e))

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("orchestrator_shutting_down")
        await self._snapshot_marks_on_shutdown()
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

            # W30 (R26 F3, finding #3): declared manual positions have no
            # local trade row — without this they are invisible to every
            # cap above (aggregate risk, concentration, correlation, count).
            positions_for_risk.extend(self._reconciler.declared_position_risk_rows())

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
        NO session in between — the gate could never see the weekend.

        W15 (R24 time-calendar-01): this counted WEEKDAYS and its docstring
        claimed "erring toward MORE sessions is the fail-safe direction". That
        is backwards. The consumer blocks when ``sessions <= blackout``, so
        MORE sessions means FEWER blocks — counting a NYSE holiday as a
        session makes the gate fail OPEN. Executed proof (clock pinned to Fri
        2026-09-04, Labor Day 09-07 a holiday): an event on Wed 09-09 gave
        _sessions_until=2 while only ONE real session (Tue) separates entry
        from the event, so a blackout of 1 did not block. Now counts real
        trading days via the same calendar the cooldown uses.
        """
        if calendar_days is None:
            return None
        # W20b (R25 tz-01): anchored on datetime.now().date() — the HOST's
        # local date. On a CEST (= ET+6) box the local date rolls over at
        # 18:00 ET, so any evening run counted sessions from TOMORROW.
        # zoneinfo, not ait.utils.time: the calendar package this function
        # already guards against can take that module down with it.
        from zoneinfo import ZoneInfo as _ZI
        today = datetime.now(_ZI("US/Eastern")).date()
        sessions = 0
        for i in range(1, max(1, calendar_days) + 1):
            day = today + timedelta(days=i)
            try:
                from ait.utils.time import is_trading_day as _itd
                trading = _itd(day)
            except Exception:  # noqa: BLE001
                # W20c: the fallback used `day.weekday() < 5` — the EXACT
                # weekday heuristic W15 removed from the primary path above,
                # and for the same reason: the consumer blocks when
                # `sessions <= blackout`, so counting a NYSE holiday as a
                # session inflates the count and makes the gate fail OPEN.
                # W15 fixed the instance and left the fallback carrying the
                # defect. A holiday is not knowable without the calendar, so
                # the fallback now errs the SAFE way — fewer sessions, more
                # blocking — instead of the direction that admits an entry.
                trading = False
            if trading:
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
