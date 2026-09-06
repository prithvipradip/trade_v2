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

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime

from ait.bot.state import StateManager, TradeRecord, TradeStatus
from ait.broker.ibkr_client import IBKRClient
from ait.strategies.base import CREDIT_STRATEGIES
from ait.config.settings import ExitConfig
from ait.execution.exit_policy import (
    EXPIRY_APPROACHING_DTE,
    macro_flatten_window_days,
    take_profit_targets,
)
from ait.data.market_data import MarketDataService
from ait.data.quality import DataQualityValidator
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.pdt_guard import PDTGuard
from ait.utils.logging import get_logger
from ait.config.runtime_env import contract_flag, contract_float  # R19: ONE authority for env-contract defaults

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

    # Entry cash-flow direction — shared source of truth in strategies/base.py
    CREDIT_STRATEGIES = CREDIT_STRATEGIES

    def __init__(
        self,
        ibkr_client: IBKRClient,
        market_data: MarketDataService,
        state: StateManager,
        circuit_breaker: CircuitBreaker,
        pdt_guard: PDTGuard,
        exit_config: ExitConfig | None = None,
        earnings_calendar=None,
        economic_calendar=None,
    ) -> None:
        self._ibkr = ibkr_client
        self._market_data = market_data
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._pdt_guard = pdt_guard
        self._exit_config = exit_config or ExitConfig()
        self._earnings = earnings_calendar
        self._economic_cal = economic_calendar
        # Optional async notifier (orchestrator wires _send_notification):
        # used for prolonged marks-missing and PDT-blocked-stop alerts, both
        # states where positions are silently unprotected (audit R2).
        self._notify_cb = None
        self._marks_missing_streak: dict[str, int] = {}
        self._pdt_alerted: set[str] = set()
        # R14: exit-input staleness gate. `DataQualityValidator` has existed
        # (and been instantiated in the orchestrator) since the beginning with
        # ZERO call sites — the exit path never checked a single quote.
        self._quality = DataQualityValidator(
            max_staleness_seconds=self._exit_config.max_quote_staleness_seconds,
        )
        self._last_quote_ts: dict[str, datetime] = {}   # symbol -> last tick time
        self._frozen_alerted: set[str] = set()
        self._touch_confirm: dict[str, int] = {}        # trade_id -> agreeing ticks
        # fail-direction-04: once-per-outage latch for a touch stop that is
        # RAISING instead of evaluating (same shape as _frozen_alerted, keyed
        # by trade because the failure is per position, not per feed).
        self._touch_fail_alerted: set[str] = set()

    MARKS_MISSING_ALERT_TICKS = 10  # ~5 min at the 30s fast-monitor cadence

    async def _spot_quote(self, symbol: str, persist: bool = True) -> tuple[float | None, str]:
        """R14: the underlying's price for exit decisions, WITH a health verdict.

        Returns (mid, health) where health is one of:
          "fresh"    — validator passed and the tick time advanced.
          "degraded" — the validator flagged it (stale / crossed / wide spread /
                       outlier jump), or it came from a fallback with no real
                       bid/ask. Usable, but not on its own.
          "frozen"   — the exchange tick time has NOT advanced since the last
                       look. The feed is dead: the number is a fossil, and it
                       will keep being returned, unchanged, forever.
          "missing"  — no quote at all.

        The distinction that matters is frozen-vs-old. A DELAYED feed (the bot
        defaults to market-data type 4) still ticks; its numbers are behind but
        they move, and staleness alone would condemn every quote and disable the
        touch stop outright. A FROZEN feed stops advancing — that is the state
        worth acting on, and it is invisible to an absolute-age check.

        `persist=False` makes this a pure READ (A4 report invariant): the
        last-seen tick time is not advanced and the frozen-feed alert latch is
        not cleared. R16: the write was unconditional, so a report call
        CONSUMED the timestamp advance — within market_data's 15s quote cache
        the very next monitor pass saw an unchanged timestamp, classified the
        feed "frozen", and downgraded a real short-strike touch from
        fire-immediately to a 2-tick confirmation (+30-60s on the one exit
        rule that caps a credit structure's loss).

        R19: the same self-collision one level down. _last_quote_ts is keyed
        by SYMBOL but a check_positions pass can evaluate the symbol more than
        once (two positions on one underlying). The first evaluation advanced
        the timestamp, so the second read the SAME 15s-cached quote, saw
        `timestamp <= prev_ts` and called a healthy feed "frozen" — every
        pass, forever, for that position: touch stops downgraded to 2-tick
        confirmation and the once-per-outage STALE EXIT FEED page spent on
        nothing. Classify each symbol ONCE per pass and reuse the verdict
        (`<` alone cannot replace `<=`: a genuinely frozen feed repeats its
        timestamp, it never rewinds).
        """
        # Per-pass memo — reads only, never a mutation of the tick-time state.
        # persist=False (report path) neither fills nor consumes it.
        pass_seen = getattr(self, "_pass_quote_verdicts", None)
        if persist and pass_seen is not None and symbol in pass_seen:
            return pass_seen[symbol]

        def _verdict(result: "tuple[float | None, str]") -> "tuple[float | None, str]":
            if persist and pass_seen is not None:
                pass_seen[symbol] = result
            return result

        try:
            quote = await self._market_data.get_quote(symbol)
        except Exception as e:  # noqa: BLE001 — never let a quote fetch kill exits
            log.warning("spot_quote_failed", symbol=symbol, error=str(e))
            return _verdict((None, "missing"))
        if quote is None or quote.mid <= 0:
            return _verdict((None, "missing"))

        # getattr-guarded like the rest of this class: some tests build the
        # manager via __new__ and never run __init__.
        seen_ts = getattr(self, "_last_quote_ts", None)
        if seen_ts is None:
            seen_ts = self._last_quote_ts = {}
        prev_ts = seen_ts.get(symbol)
        if persist:
            seen_ts[symbol] = quote.timestamp
        if prev_ts is not None and quote.timestamp <= prev_ts:
            return _verdict((quote.mid, "frozen"))

        quality = self._quality_validator().validate_quote(
            symbol=symbol,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last or quote.mid,
            volume=quote.volume,
            timestamp=quote.timestamp.timestamp(),
        )
        if not quality.is_valid:
            log.warning("exit_quote_degraded", symbol=symbol,
                        issues=quality.issues,
                        staleness_s=round(quality.staleness_seconds, 1))
            return _verdict((quote.mid, "degraded"))

        if persist:
            # R16: clearing the once-per-outage page latch is alert STATE —
            # a report must not re-arm it (nor, below, fire the page itself).
            getattr(self, "_frozen_alerted", set()).discard(symbol)
        return _verdict((quote.mid, "fresh"))

    def _quality_validator(self) -> DataQualityValidator:
        q = getattr(self, "_quality", None)
        if q is None:
            q = self._quality = DataQualityValidator(
                max_staleness_seconds=self._exit_config.max_quote_staleness_seconds,
            )
        return q

    async def _alert_frozen_feed(self, symbol: str, trade_id: str) -> None:
        """Page ONCE per outage: the touch stop is a credit position's primary
        loss cap, and it is now running on a feed that has stopped moving."""
        alerted = getattr(self, "_frozen_alerted", None)
        if alerted is None:
            alerted = self._frozen_alerted = set()
        if symbol in alerted:
            return
        alerted.add(symbol)
        notify = getattr(self, "_notify_cb", None)
        if not notify:
            return
        try:
            await notify(
                f"STALE EXIT FEED: {symbol} — the underlying quote has stopped "
                f"advancing, and the short-strike touch stop reads it directly. "
                f"Exits still fire, but now need {self._exit_config.touch_confirm_ticks} "
                f"agreeing ticks. Check the market-data feed ({trade_id})."
            )
        except Exception:  # noqa: BLE001
            pass

    async def _alert_touch_stop_failed(self, trade, error: Exception) -> None:
        """Page ONCE per outage: the touch stop is not evaluating at all.

        fail-direction-04 (blind-spot hunt 2026-08-25): the touch-stop block's
        except swallowed every exception at log.debug. The flat credit stop
        defaults to DISABLED (AIT_CREDIT_LOSS_LIMIT contract default "0", see
        the is_credit branch below), so on a credit structure the short-strike
        touch is the ONLY loss exit — a swallowed exception silently switches
        loss protection off for the life of the trade, re-failing every 30s
        tick with nothing but a DEBUG line in a rotating file.

        Throttled like _alert_frozen_feed: a 30s monitor must not page every
        tick. The latch clears when the block next evaluates cleanly, so a
        LATER outage pages again. Never raises — it runs inside an except
        handler, and an exception here would abort check_positions' loop and
        take every OTHER position's exit evaluation down with it.
        """
        try:
            alerted = getattr(self, "_touch_fail_alerted", None)
            if alerted is None:
                alerted = self._touch_fail_alerted = set()
            if trade.trade_id in alerted:
                return
            alerted.add(trade.trade_id)
            notify = getattr(self, "_notify_cb", None)
            if not notify:
                return
            await notify(
                f"TOUCH STOP NOT EVALUATING: {trade.symbol} {trade.strategy} "
                f"({trade.trade_id}) — the short-strike touch check raised "
                f"{type(error).__name__}: {error}. This is the only loss exit "
                f"on a credit structure; the position now has DTE/expiry exits "
                f"only. Check the trade's legs data."
            )
        except Exception:  # noqa: BLE001 — an alert must never break exits
            pass

    async def check_positions(self) -> list[PositionStatus]:
        """Check all open positions and determine which need action."""
        open_trades = self._state.get_open_trades()
        if not open_trades:
            return []

        # R19: one spot classification per symbol per pass (see _spot_quote).
        # Fresh dict every pass — a verdict must never outlive the pass that
        # produced it, or a feed that froze between passes would go unseen.
        self._pass_quote_verdicts: dict[str, tuple[float | None, str]] = {}

        option_marks = self._get_option_marks()
        statuses = []
        try:
            for trade in open_trades:
                if trade.status not in (TradeStatus.FILLED, TradeStatus.PARTIAL):
                    continue

                status = await self._evaluate_position(trade, option_marks)
                if status:
                    statuses.append(status)
        finally:
            # The memo is pass-scoped: nothing outside this loop may reuse it.
            self._pass_quote_verdicts = None

        # Log summary
        exits_needed = [s for s in statuses if s.should_exit or s.partial_exit_quantity > 0]
        if exits_needed:
            log.info(
                "positions_needing_action",
                count=len(exits_needed),
                reasons=[f"{s.symbol}: {s.exit_reason}" for s in exits_needed],
            )

        return statuses

    def _get_option_marks(self) -> dict[tuple, float]:
        """Index current IBKR option marks by (symbol, expiry, strike, right)."""
        marks: dict[tuple, float] = {}
        for item in self._ibkr.ib.portfolio():
            c = item.contract
            if c.secType != "OPT":
                continue
            price = item.marketPrice
            if price is None or math.isnan(price):
                continue
            expiry_key = c.lastTradeDateOrContractMonth.replace("-", "")[:8]
            marks[(c.symbol, expiry_key, float(c.strike), c.right)] = float(price)
        return marks

    def _option_position_unrealized(
        self, trade: TradeRecord, marks: dict[tuple, float] | None = None
    ) -> float | None:
        """Unrealized P&L for an option position from IBKR per-leg marks.

        Computes the position's current liquidation value per contract
        (sum of leg marks, signed by whether we are long or short each leg)
        and compares it against the signed entry cash flow:
        debit position: paid entry  -> unrealized = (net_liq - entry) * 100 * qty
        credit position: received entry -> unrealized = (net_liq + entry) * 100 * qty
        (net_liq is negative for credit positions: it's what we'd pay to close)

        Returns None when any leg has no usable IBKR mark — callers should
        skip exit evaluation rather than act on a partial price.
        """
        try:
            legs = json.loads(trade.legs) if trade.legs else []
        except (ValueError, TypeError):
            legs = []

        if not legs:
            # Single-contract trade (long_call/long_put/covered_call/CSP):
            # synthesize a one-leg description from the trade row.
            if trade.contract_type == "call":
                right = "C"
            elif trade.contract_type == "put":
                right = "P"
            else:
                return None
            if not trade.expiry or not trade.strike:
                return None
            # Buy/sell is the POSITION polarity, not the market bias.
            # trade.direction encodes the signal's view (BULLISH/BEARISH),
            # so a cash-secured put (BULLISH) would wrongly read as "BUY".
            # Credit strategies are premium SELLERS; everything else BUYS.
            action = "SELL" if trade.strategy in self.CREDIT_STRATEGIES else "BUY"
            legs = [{"strike": trade.strike, "right": right,
                     "action": action, "expiry": trade.expiry}]

        if marks is None:
            marks = self._get_option_marks()

        net_liq = 0.0  # current value of the position per contract, from our side
        for leg in legs:
            expiry_key = str(leg.get("expiry", "")).replace("-", "")[:8]
            key = (trade.symbol, expiry_key, float(leg["strike"]), leg["right"])
            mark = marks.get(key)
            if mark is None:
                return None
            sign = 1.0 if str(leg.get("action", "BUY")).upper() == "BUY" else -1.0
            net_liq += sign * mark

        if trade.strategy in self.CREDIT_STRATEGIES:
            signed_entry = -abs(trade.entry_price)
        else:
            signed_entry = abs(trade.entry_price)

        return (net_liq - signed_entry) * 100 * trade.quantity

    async def _evaluate_position(
        self, trade: TradeRecord, option_marks: dict[tuple, float] | None = None
    , persist: bool = True) -> PositionStatus | None:
        """Evaluate a single position for exit conditions."""
        current_price, spot_health = await self._spot_quote(trade.symbol, persist=persist)

        if current_price is None:
            # R14: a missing UNDERLYING quote used to abandon the whole
            # evaluation — no PositionStatus, so no stop, no take-profit and,
            # worst of all, no DTE/expiry safety exit. An option's P&L comes
            # from its own leg marks, not from spot, so only the touch and the
            # assignment-ITM refinement actually need this price. Losing it must
            # degrade those two, not switch the position off. (Same principle as
            # marks_missing below; that path already got this right.)
            if trade.contract_type == "stock":
                log.warning("cannot_evaluate_position", symbol=trade.symbol,
                            reason="no price for a STOCK position (P&L needs it)")
                return None
            log.warning(
                "underlying_quote_missing",
                trade_id=trade.trade_id, symbol=trade.symbol,
                note="touch/assignment checks skipped this tick; P&L and DTE "
                     "safety exits still active",
            )
            current_price = 0.0

        # Calculate unrealized P&L.
        # For options this must be priced from the OPTION position's market
        # value (per-leg marks from IBKR), never from the underlying price:
        # (stock_price - option_premium) is meaningless and historically
        # produced 2800%+ phantom "profits" that tripped take-profit exits.
        multiplier = 100  # Options multiplier
        if trade.contract_type == "stock":
            multiplier = 1

        is_short = trade.direction.value == "short"
        # Premium polarity for exit routing: credit strategies (we SOLD
        # premium) target a 50% buyback and can be assigned; debit strategies
        # (we BOUGHT premium) ride the long-side target. Cannot use is_short
        # here — iron_condor/short_strangle are stored direction=long/neutral.
        is_credit = trade.strategy in self.CREDIT_STRATEGIES
        if trade.contract_type == "stock":
            if is_short:
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
            else:
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            unrealized_pnl = self._option_position_unrealized(trade, option_marks)

        # Missing marks must NOT make the position unmanageable: skip only
        # the P&L-driven rules (stop/take-profit/partials) but keep the
        # DTE-based safety exits (assignment risk, expiry close) alive —
        # otherwise a position with no marks rides to expiry unprotected.
        marks_missing = unrealized_pnl is None
        if marks_missing:
            unrealized_pnl = 0.0
            log.warning(
                "cannot_price_position",
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                strategy=trade.strategy,
                reason="no IBKR marks for one or more legs; P&L exit rules "
                       "skipped this tick, DTE safety exits still active",
            )
            # PROLONGED-outage alert (audit R2): one missing tick is routine;
            # N consecutive means stop/TP protection has been OFF for the
            # whole stretch (data slot stolen, farm outage) — a short-vol
            # book riding a selloff unmanaged. Alert ONCE per outage.
            # (getattr-guarded: some tests construct via __new__.)
            streaks = getattr(self, "_marks_missing_streak", None) if persist else None
            notify = getattr(self, "_notify_cb", None)
            if streaks is not None:
                streak = streaks.get(trade.trade_id, 0) + 1
                streaks[trade.trade_id] = streak
                if streak == self.MARKS_MISSING_ALERT_TICKS and notify:
                    try:
                        await notify(
                            f"MARKS MISSING x{streak} ticks: {trade.symbol} "
                            f"{trade.strategy} — stop/take-profit protection is "
                            f"NOT running (data outage?). DTE safety exits only."
                        )
                    except Exception:  # noqa: BLE001
                        pass
        else:
            streaks = getattr(self, "_marks_missing_streak", None)
            if streaks is not None:
                streaks.pop(trade.trade_id, None)

        cost_basis = abs(trade.entry_price) * trade.quantity * multiplier
        pnl_pct = unrealized_pnl / cost_basis if cost_basis > 0 else 0.0

        # Days to expiry
        dte = None
        if trade.expiry:
            try:
                expiry_date = date.fromisoformat(trade.expiry)
                dte = (expiry_date - date.today()).days
            except ValueError:
                pass

        # Update high water mark — always persist so journaling data stays
        # accurate (but never let a marks-missing 0.0 touch it)
        prev_hwm = self._state.get_high_water_mark(trade.trade_id)
        hwm = max(prev_hwm, pnl_pct)
        if not marks_missing and persist:
            self._state.update_high_water_mark(trade.trade_id, hwm)
            # Persist the live mark so status/dashboard show real per-position
            # unrealized P&L (guarded: a marks-missing tick must never
            # overwrite a real mark with 0.0). Audit 2026-07-07 item 1.4.
            self._state.update_position_mark(trade.trade_id, unrealized_pnl, pnl_pct)

        # Get learning overrides for stop/take-profit (if any)
        stop_loss_pct = self._exit_config.initial_stop_loss_pct
        trailing_stop_pct = self._exit_config.trailing_stop_pct
        breakeven_trigger = self._exit_config.breakeven_trigger_pct

        if is_credit:
            # R6 (2026-07-09, user-approved): credit-structure P&L oscillates
            # by design, and the equity-style breakeven(+30%)/trailing(25%)
            # tiers — with the vol multiplier TIGHTENING stops on calm
            # underlyings — scratched statistical winners before the 40% TP
            # (SPY IC peaked +32.3% of credit, trail-stopped out at +$5.80
            # the same day). Practitioner-standard for defined-risk short
            # premium: one flat loss limit as a multiple of credit received;
            # the wings already cap the true tail. No breakeven tier, no
            # trailing, no vol adjustment. Take-profit tiers unchanged.
            import os as _os_x
            # R12-B1 (evidence backtest, 17mo walk-forward, 26 matched
            # trades): flat credit stops fired THROUGH their trigger on gaps
            # (-1.25x trigger realized -2.01x in the worked example) and every
            # flat level underperformed both no-stop and touch-close
            # (touch +$756/PF 1.38 > none -$24 > 1.25x -$134 > 2.0x -$1,067).
            # Default 0 = flat stop DISABLED; short-strike-touch close (rule
            # 1b below) is the early loss exit; the wings cap the tail.
            _loss_mult = contract_float("AIT_CREDIT_LOSS_LIMIT")
            effective_stop = -_loss_mult if _loss_mult > 0 else -999.0
        else:
            # Volatility-adjusted stops (debit structures only): widen stops
            # for high-volatility underlyings
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

        # 1. Dynamic stop loss (trailing/breakeven) — needs real marks: a
        # marks-missing pnl_pct of 0.0 would spuriously trip a breakeven stop
        #
        # R12-B POST-STOP COOLDOWN — NEEDS THE ORCHESTRATOR (flagged for main
        # thread; this file only evaluates exits, it cannot see entries).
        # Evidence: the existing re-entry cooldown (orchestrator GUARD #2,
        # ~line 1609) keys on ENTRY time — a symbol stopped out this scan can
        # re-enter NEXT scan into the same trend that just stopped it.
        # Spec: in orchestrator._should_skip_signal (GUARD #2 block), ALSO
        # skip when a recent stop-out exists keyed on EXIT time, duration
        # 1 TRADING day (exit_time's next trading session must have STARTED
        # before re-entry). Exact query against bot_state.db:
        #   SELECT COUNT(*) FROM trades
        #    WHERE symbol = :symbol
        #      AND status = 'CLOSED'
        #      AND (exit_reason_detailed LIKE '%stop_loss%'
        #           OR exit_reason_detailed LIKE '%short_strike_touch%')
        #      AND exit_time >= :cutoff_iso
        # (2026-08-25: short_strike_touch added — the touch stop below is the
        #  primary loss exit and its reason string does NOT contain
        #  'stop_loss'; the 08-24 QQQ re-entry 13 min after a touch stop was
        #  this mismatch. Authority: TradingOrchestrator._post_stop_cooldown_until.)
        # where :cutoff_iso = start (09:30 ET) of the PREVIOUS trading day if
        # now is during RTH, i.e. block while exit_time falls within the
        # current or immediately preceding trading session; weekends/holidays
        # via the existing market-calendar helper. Non-zero count -> skip
        # with log tag "post_stop_cooldown_skip".
        if not marks_missing and pnl_pct <= effective_stop:
            should_exit = True
            if hwm >= breakeven_trigger and effective_stop >= 0:
                exit_reason = f"trailing_stop (P&L: {pnl_pct:.1%}, peak: {hwm:.1%}, stop: {effective_stop:.1%})"
            elif hwm >= breakeven_trigger:
                exit_reason = f"breakeven_stop (P&L: {pnl_pct:.1%}, peak: {hwm:.1%})"
            else:
                exit_reason = f"stop_loss (P&L: {pnl_pct:.1%})"

        # 1b. R12-B1: short-strike TOUCH close for credit structures — the
        # evidence-backed early loss exit (avg loss at touch 0.475x credit,
        # matching the audit's 0.49x; only 1 of 7 touched trades recovered).
        # Uses the UNDERLYING price, so it protects even when option marks
        # are missing. Env AIT_CREDIT_TOUCH_STOP=0 disables.
        if (not should_exit and is_credit
                and os.environ.get("AIT_CREDIT_TOUCH_STOP", "1") == "1"):
            try:
                import json as _json
                _legs = _json.loads(trade.legs) if trade.legs else []
                _short_put = max((float(l["strike"]) for l in _legs
                                  if str(l.get("action", "")).upper() == "SELL"
                                  and str(l.get("right", l.get("type", ""))).upper().startswith("P")),
                                 default=None)
                _short_call = min((float(l["strike"]) for l in _legs
                                   if str(l.get("action", "")).upper() == "SELL"
                                   and str(l.get("right", l.get("type", ""))).upper().startswith("C")),
                                  default=None)
                # R14: reuse the quote already fetched (and health-checked) for
                # this tick instead of a second unvalidated read. A missing
                # quote leaves _spot at 0.0, which touches nothing.
                _spot = current_price
                if (_short_put or _short_call) and _spot:
                    _touched = ""
                    if _short_put and _spot <= _short_put:
                        _touched = (f"short_strike_touch (spot {_spot:.2f} "
                                    f"<= put {_short_put:.2f})")
                    elif _short_call and _spot >= _short_call:
                        _touched = (f"short_strike_touch (spot {_spot:.2f} "
                                    f">= call {_short_call:.2f})")

                    # The touch stop is the only exit rule that acts DIRECTLY on
                    # the underlying's price, and it used to fire on a single
                    # unvalidated print. On a healthy quote that is still right:
                    # the evidence says a touch rarely recovers (1 of 7), so
                    # speed matters. On a degraded or frozen one, demand
                    # corroboration first — a lone bad tick must not liquidate a
                    # healthy condor, and a fossil price must not do it twice.
                    #
                    # We still FIRE on a confirmed touch off a bad feed rather
                    # than going quiet. The asymmetry is the whole argument: a
                    # missed real breach costs the wing width (or, on a strangle,
                    # is unbounded); a false one costs the spread. Never trade a
                    # bounded loss for an unbounded silence.
                    confirms = getattr(self, "_touch_confirm", None)
                    if confirms is None:
                        confirms = self._touch_confirm = {}

                    if _touched and spot_health == "fresh":
                        should_exit = True
                        exit_reason = _touched
                        # R16: persist-guarded like the increment below — the
                        # streak is protection state, and A4 says a report
                        # must not advance OR clear it.
                        if persist:
                            confirms.pop(trade.trade_id, None)
                    elif _touched:
                        need = max(1, int(getattr(
                            self._exit_config, "touch_confirm_ticks", 2)))
                        seen = confirms.get(trade.trade_id, 0) + 1
                        if persist:
                            confirms[trade.trade_id] = seen
                        if spot_health == "frozen" and persist:
                            # R16: A4 — "must not ... fire alerts". The page
                            # is once-per-outage; spending it on a report both
                            # notifies off a report and silences the monitor.
                            await self._alert_frozen_feed(trade.symbol, trade.trade_id)
                        if seen >= need:
                            should_exit = True
                            exit_reason = f"{_touched} [{spot_health} quote, {seen} agreeing ticks]"
                        else:
                            log.warning(
                                "touch_pending_confirmation",
                                trade_id=trade.trade_id, symbol=trade.symbol,
                                spot_health=spot_health, seen=seen, need=need,
                                note="touch seen on a bad quote — holding for "
                                     "corroboration rather than acting on one print",
                            )
                    elif persist:
                        # Spot pulled back inside the strikes: the streak dies.
                        # R16: this pop ran regardless of `persist`, so the
                        # post-market summary (the sole get_portfolio_summary
                        # caller today) could WIPE a pending degraded/frozen
                        # touch-confirmation streak that the 30s monitor had
                        # been accumulating — the report deciding an exit.
                        confirms.pop(trade.trade_id, None)
                # fail-direction-04: the block evaluated. Clear the page latch
                # so a LATER failure counts as a new outage (alert STATE, so
                # persist-guarded exactly like the streak above).
                if persist:
                    _tf_ok = getattr(self, "_touch_fail_alerted", None)
                    if _tf_ok:
                        _tf_ok.discard(trade.trade_id)
            except Exception as _e:  # noqa: BLE001 — protection, never a crash source
                # fail-direction-04: this was log.debug, and nothing else. The
                # touch stop is the ONLY loss exit on credit structures (the
                # flat credit stop is disabled by contract default), so any
                # exception here disables loss protection for the life of the
                # trade — invisibly, on every 30s tick. Surface it as an ERROR
                # and page once per outage; the except still stands so ONE bad
                # trade row cannot kill check_positions' loop over the others.
                log.error(
                    "touch_stop_evaluation_failed",
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    strategy=trade.strategy,
                    error=str(_e),
                    error_type=type(_e).__name__,
                    note="short-strike touch stop did not evaluate — this is "
                         "the only loss exit on a credit structure; DTE and "
                         "expiry exits are all that remain for this position",
                )
                if persist:
                    # A4 report invariant: a read-only summary pass must not
                    # fire alerts (nor consume the once-per-outage page).
                    await self._alert_touch_stop_failed(trade, _e)

        # 2. Take profit (time-decay adjusted) — needs real marks
        # R13-CRIT-2: this was `elif`, chained to the R12 touch-stop `if`
        # above. For every credit position with the touch stop enabled (the
        # default), the touch `if` was taken, so take-profit / assignment /
        # DTE / delta / earnings below were STRUCTURALLY UNREACHABLE — a
        # condor could only ever exit via stop or touch. Each branch now
        # guards on `not should_exit` explicitly.
        if (not should_exit
                and not marks_missing and not is_credit and pnl_pct >= take_profit_long):
            should_exit = True
            exit_reason = f"take_profit (P&L: {pnl_pct:.1%}, target: {take_profit_long:.1%})"
        elif (not should_exit
                and not marks_missing and is_credit and pnl_pct >= take_profit_short):
            should_exit = True
            exit_reason = f"take_profit_short (P&L: {pnl_pct:.1%}, target: {take_profit_short:.1%})"

        # 3. Assignment risk — ITM shorts on expiry day (DTE 0-1) MUST close
        # Short puts ITM → stock gets put to you (cash drain)
        # Short calls ITM → shares called away (may cause forced cover)
        elif (not should_exit
                and dte is not None and dte <= 1 and is_credit and trade.strike):
            try:
                # R14: reuse the health-checked quote. This branch only refines
                # the exit REASON — both paths exit — so a missing/stale spot
                # costs us the ITM label, never the exit itself.
                current_underlying = current_price
                if current_underlying:
                    itm = False
                    if trade.contract_type == "put" and current_underlying < trade.strike:
                        itm = True
                    elif trade.contract_type == "call" and current_underlying > trade.strike:
                        itm = True
                    if itm:
                        should_exit = True
                        exit_reason = (
                            f"assignment_risk (short {trade.contract_type} ITM, "
                            f"underlying=${current_underlying:.2f}, strike=${trade.strike:.2f})"
                        )
            except Exception:
                pass
            if not should_exit:
                should_exit = True
                exit_reason = f"expiry_approaching (DTE: {dte})"

        # 3a. Time decay — close other positions with 5 or fewer DTE
        # R12-B GUARD (cadence): the entry-DTE floor is CONFIG-side
        # (options.dte_range min raised 7 -> 14; 7-DTE entries were 2-day
        # churn, best case ~60% cost-dominated). This DTE<=5 forced exit plus
        # the 14-DTE entry floor implies every position gets >=9 calendar
        # days of theta runway — do NOT lower dte_range[0] below ~10 without
        # revisiting this exit, or entries become forced-exit churn again.
        elif not should_exit and dte is not None and dte <= EXPIRY_APPROACHING_DTE:
            should_exit = True
            exit_reason = f"expiry_approaching (DTE: {dte})"

        # 3b. Delta breach — close if directional risk ballooned
        # For neutral strategies (iron condor, straddle), delta should stay small
        # If abs(delta) > 0.50, position has taken on large directional exposure
        #
        # R16 WARNING: this rule is INERT in the current deployment — there is
        # no market-data subscription behind ib.ticker() in the exit path, so
        # _get_position_delta always returns None (it now logs
        # delta_breach_rule_inert once per session). Do not count 3b as live
        # protection; the touch stop at rule 1b covers the same blowout. Full
        # explanation and what a real fix costs: _get_position_delta docstring.
        #
        # R17: the delta check used to gate only the BODY of this branch, not
        # entry into it — trade.strategy membership alone claimed the elif
        # slot, so a None/no-breach delta still silently blocked rule 3c
        # (earnings pre-close) from ever running for every position of these
        # strategies once DTE>5. Delta is now part of the condition itself:
        # this branch only claims the slot when there's a real breach to act
        # on, so 3c gets its turn whenever 3b doesn't fire (i.e. always,
        # today, since delta is inert).
        elif (not should_exit
                and trade.strategy in ("iron_condor", "short_strangle", "long_straddle",
                                       "cash_secured_put", "covered_call")
                and (pos_delta := self._get_position_delta(trade)) is not None
                and abs(pos_delta) > 0.50):
            should_exit = True
            exit_reason = f"delta_breach (|Δ|={abs(pos_delta):.2f} > 0.50)"

        # 3c. IV crush pre-close — for SHORT premium strategies, close 2 days
        # before earnings to capture theta without eating the earnings IV crush
        elif not should_exit and self._earnings and trade.strategy in (
            "iron_condor", "short_strangle", "cash_secured_put", "covered_call",
        ):
            try:
                info = self._earnings.get_next_earnings(trade.symbol)
                if info and info.next_earnings_date:
                    from datetime import date as _d
                    days_to_earnings = (info.next_earnings_date - _d.today()).days
                    # R6: the pnl_pct > 0 gate meant a LOSING short-premium
                    # position — maximum gamma/vega into the event — rode
                    # through earnings hoping for breakeven. Close regardless
                    # of P&L; the event risk is the same either way.
                    if 0 <= days_to_earnings <= 2:
                        should_exit = True
                        exit_reason = f"pre_earnings_iv_crush (days={days_to_earnings}, pnl={pnl_pct:.1%})"
            except Exception:
                pass

        # 3d. Macro event flatten — close short-premium positions 1 day before
        # FOMC/CPI/NFP to avoid vol expansion. DISABLED for data-collection
        # window — set AIT_SKIP_MACRO_EVENTS=1 to re-enable.
        # R13-CRIT-1: a local `import os` here made `os` function-local, so
        # line ~363 (touch-stop env read) raised UnboundLocalError on EVERY
        # credit-position tick — killing the whole exit monitor. Module-level
        # import (line 16) is the only one allowed in this function.
        # PLAN 2026-08-04: iron_condor REMOVED from the flatten list (paired
        # with pre_event_blackout_days 4->1). Defined-risk holds THROUGH the
        # event — the wings cap a surprise, and the post-event vol crush is
        # the trade's payoff; flattening at d2e<=1 sold the insurance and
        # refused the premium. Undefined-risk keeps the early exit.
        # R20 #5a follow-up: strategy list + per-strategy window (5 for
        # strangle-class, 1 for CSP/CC) now come from exit_policy.py's
        # MACRO_FLATTEN_WINDOW_DAYS — the single source the research engine
        # already reads, instead of a hand-copied tuple+conditional here.
        _evt_window = macro_flatten_window_days(trade.strategy)
        if (contract_flag("AIT_SKIP_MACRO_EVENTS")
                and not should_exit and self._economic_cal
                and _evt_window is not None):
            try:
                days_to_event = self._economic_cal.days_until_next_event()
                # R6 (user-approved): undefined-risk (strangles) exits EARLY
                # — a Thu/Fri close ahead of a Mon/Tue event avoids carrying
                # naked weekend gap risk for ~2 sessions of residual theta.
                # Defined-risk keeps the tight window (wings cap the gap).
                if days_to_event is not None and days_to_event <= _evt_window:
                    should_exit = True
                    exit_reason = f"macro_event_flatten (days_to_event={days_to_event})"
            except Exception:
                pass

        # 4. Check for partial exit milestones (only if not already exiting
        # fully, and only with real marks)
        if not should_exit and not marks_missing and trade.quantity > 1:
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
                    # A PDT-vetoed exit means a STOP or take-profit wanted to
                    # fire and couldn't — the position rides unprotected
                    # (audit R2/H3). Alert once per trade so the human can
                    # decide to close manually. (getattr-guarded for tests
                    # constructing via __new__.)
                    _pdt_seen = getattr(self, "_pdt_alerted", None) if persist else None
                    _notify = getattr(self, "_notify_cb", None)
                    if _pdt_seen is not None and trade.trade_id not in _pdt_seen and _notify:
                        _pdt_seen.add(trade.trade_id)
                        try:
                            await _notify(
                                f"PDT BLOCKED EXIT: {trade.symbol} {trade.strategy} "
                                f"wanted to exit (P&L {pnl_pct:+.1%}) but closing "
                                f"today would trip PDT. Position is riding "
                                f"unprotected until tomorrow — review manually."
                            )
                        except Exception:  # noqa: BLE001
                            pass

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

        R12-B RESTING GTC TAKE-PROFIT — NEEDS EXECUTOR/ORCHESTRATOR (flagged
        for main thread; deliberately NOT implemented here — this module is
        the synthetic monitor, order placement lives in executor.py).
        Evidence: synthetic TP exits currently cross ask+$0.10, surrendering
        6-17% of each win; a resting broker-side order also survives bot
        death and monitor blindness (marks-missing outages).
        Spec:
          1. On entry fill (executor confirms the BAG entry), place a GTC
             BAG LIMIT order to CLOSE the combo at price
             credit * (1 - tp_tier(dte)), tp_tier from the table below
             (0.50/0.40/0.30/0.20). Record its orderId on the trade row.
          2. At tier transitions (DTE crossing 20, 10, 5), cancel-and-replace
             the GTC at the new tier price (orchestrator daily tick).
          3. Cancel the resting GTC whenever ANY other exit path fires
             (stop, DTE close, assignment risk, delta breach, earnings/macro
             flatten, manual) BEFORE placing that exit order.
        SAFETY ANALYSIS (double-fill): both the resting GTC and a synthetic
        exit can be in flight simultaneously — if the GTC fills while a
        synthetic exit order is working (or vice versa) the second fill
        REVERSES the position (short condor -> long condor), the exact class
        of bug as the CLOSING->FILLED duplicate-close in Tier A1. Therefore
        the synthetic exit path MUST, before placing any exit order:
          (a) check the trade row for a resting TP orderId;
          (b) request its status — if Filled/PendingSubmit-with-fills, do NOT
              place the synthetic exit (the TP already closed the position);
          (c) otherwise cancel the GTC, CONFIRM the cancel ack, then place
              the synthetic exit;
          (d) startup reconcile must rebuild the resting-TP map from broker
              open orders (survives restarts; ties into Tier A1's tracker
              rebuild).
        """
        # R20 #5a follow-up: was an inline copy of the ladder now owned by
        # exit_policy.py (the research engine already reads it from there) —
        # kept in sync only by test_r20_research_validity.py's parity check.
        # Wired directly so there is one implementation, not two verified-equal.
        return take_profit_targets(dte, self._exit_config.time_decay_scaling)

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

        R16 — READ THIS BEFORE RELYING ON EXIT RULE 3b (delta_breach).
        This is INERT in the current deployment and cannot be fixed from this
        module. `ib.ticker(contract)` only returns a live Ticker while a
        market-data subscription for that contract is open, and the exit stack
        opens none: every reqMktData in the codebase is snapshot-style and
        cancelled in a finally block (options_chain.py:483 at entry,
        orchestrator._option_nbbo, market_data). So ticker() is None (or a
        days-stale entry-time fossil), found_any stays False, and rule 3b at
        _evaluate_position can never fire on live greeks.

        The obvious alternative source is NOT one: ib_insync 0.9.86's
        PortfolioItem carries no greeks at all (fields: contract, position,
        marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL,
        account — verified), which is also why the risk manager's
        portfolio-delta gate is documented dead. Making 3b real needs a
        genuine per-leg subscription (reqMktData + ~2s settle + cancel, the
        options_chain pattern) issued from the 30s exit loop — a broker-side
        change this module should not own unilaterally, on a feed where
        run_orchestrator.py:43-49 already documents greeks as "sporadic".

        Until then the rule is left in place but ANNOUNCED: the touch stop
        covers the same directional blowout, and a safety rule that is quietly
        dead is worse than one that says so.
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
            if not found_any:
                self._warn_delta_rule_inert(trade)
                return None
            return total_delta
        except Exception:
            return None

    def _warn_delta_rule_inert(self, trade) -> None:
        """R16: say it out loud, once per session, that exit rule 3b is dead.

        Once per PROCESS (not per position, not per tick): the condition is
        structural, so per-tick logging would be pure noise at the 30s
        cadence, and total silence is what let a rule advertised in the exit
        chain sit inert since inception.
        """
        seen = getattr(self, "_delta_rule_inert_logged", None)
        if seen:
            return
        self._delta_rule_inert_logged = True
        log.warning(
            "delta_breach_rule_inert",
            trade_id=getattr(trade, "trade_id", "?"),
            symbol=getattr(trade, "symbol", "?"),
            note="exit rule 3b (delta_breach) has NO greeks to read: the exit "
                 "path holds no market-data subscription for the leg "
                 "contracts and PortfolioItem carries no greeks. Treat rule 3b "
                 "as OFF — the short-strike touch stop is the live guard "
                 "against the same directional blowout.",
        )

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
            # A4 (deep-audit MP-F5): the SUMMARY is a report — it must not
            # advance HWM, marks, missing-streaks, or fire alerts (the EOD
            # call was mutating protection state on stale after-hours marks).
            status = await self._evaluate_position(trade, persist=False)
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
