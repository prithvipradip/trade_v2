"""Pydantic-validated configuration for AIT.

All configuration is loaded from config.yaml with environment variable overrides.
Validation catches misconfigurations BEFORE the bot starts trading.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from ait.utils.cadence import QUOTE_STALENESS_BUDGET_SECONDS


class _StrictModel(BaseModel):
    """R13 (human-factors lens): every config model silently IGNORED unknown
    keys (pydantic default), so a one-letter typo in config.yaml silently ran
    the code default — executed proof: `max_contracts_per_trad` traded 10x
    size, `paper_trading_mod` re-enabled live-only overlays, all with zero
    warnings. Unknown keys are now a fail-loud config error at load time.
    SentimentConfig keeps `extra="allow"` deliberately (retired-subsystem
    tombstone keys may linger in old configs)."""

    model_config = {"extra": "forbid"}


def _parse_hhmm(value: str, field: str = "time") -> tuple[int, int]:
    """W69: validate an "HH:MM" (24h, ET) string and return (hour, minute).

    One parser for every config clock value so a typo ("10.30", "25:00") is a
    startup error, not a silently-wrong intraday gate.
    """
    parts = str(value).split(":")
    if len(parts) != 2 or not all(p.isdigit() and len(p) == 2 for p in parts):
        raise ValueError(f"{field} must be 'HH:MM' (24h ET), got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{field} must be a valid 24h time, got {value!r}")
    return hour, minute


class TradingConfig(_StrictModel):
    mode: Literal["paper", "live"] = "paper"
    universe: list[str] = ["SPY", "QQQ", "AAPL"]
    scan_interval_seconds: int = Field(default=300, ge=30, le=3600)
    max_daily_trades: int = Field(default=5, ge=1, le=50)
    trading_hours_only: bool = True
    no_new_entries_minutes_before_close: int = Field(default=15, ge=0, le=120,
        description="W35 (R26 entry-blackout-cutoff-900s-is-a-bare-literal): "
                    "no new entries inside this many minutes of the session "
                    "close. Was `remaining < 900` in bot/scheduler.py — a "
                    "value that manages trading with no config home (R19c).")
    first_hour_end_et: str = Field(default="10:30",
        description="W69: end of the 'first hour' (ET). Before this time the "
                    "intraday trade budget is first_hour_trade_budget and "
                    "directional signals must clear first_hour_min_confidence. "
                    "Was `hour_min < 10.5` in orchestrator._get_trade_budget "
                    "and the first_hour_confidence_gate.")
    midday_end_et: str = Field(default="14:00",
        description="W69: end of the mid-day window (ET). From here to the "
                    "close the whole remaining max_daily_trades is released. "
                    "Was `hour_min < 14.0` in orchestrator._get_trade_budget.")
    first_hour_trade_budget: int = Field(default=2, ge=1, le=50,
        description="W69: cumulative trade ceiling while before "
                    "first_hour_end_et (still capped by max_daily_trades). "
                    "Was a literal 2 in orchestrator._get_trade_budget.")
    midday_trade_budget: int = Field(default=4, ge=1, le=50,
        description="W69: cumulative trade ceiling between first_hour_end_et "
                    "and midday_end_et (still capped by max_daily_trades). "
                    "Was a literal 4 in orchestrator._get_trade_budget.")
    first_hour_min_confidence: float = Field(default=0.85, ge=0.50, le=0.99,
        description="W69: confidence a DIRECTIONAL signal needs before "
                    "first_hour_end_et (neutral/range-gated structures such as "
                    "iron condors are exempt). Was a literal 0.85 in the "
                    "orchestrator's first_hour_confidence_gate.")

    @field_validator("first_hour_end_et", "midday_end_et")
    @classmethod
    def _validate_clock(cls, v: str, info) -> str:
        _parse_hhmm(v, info.field_name)
        return v

    @model_validator(mode="after")
    def _intraday_windows_are_ordered(self) -> "TradingConfig":
        if self.first_hour_end_hour >= self.midday_end_hour:
            raise ValueError(
                f"trading.first_hour_end_et ({self.first_hour_end_et}) must be "
                f"earlier than trading.midday_end_et ({self.midday_end_et})")
        return self

    @property
    def first_hour_end_hour(self) -> float:
        """first_hour_end_et as fractional ET hours (10:30 -> 10.5)."""
        h, m = _parse_hhmm(self.first_hour_end_et, "first_hour_end_et")
        return h + m / 60.0

    @property
    def midday_end_hour(self) -> float:
        """midday_end_et as fractional ET hours (14:00 -> 14.0)."""
        h, m = _parse_hhmm(self.midday_end_et, "midday_end_et")
        return h + m / 60.0


class AccountConfig(_StrictModel):
    pdt_protection: bool = True
    pdt_account_under_25k: bool = True
    capital: float = Field(
        default=5_000.0, ge=1_000.0, le=10_000_000.0,
        description="THE one capital number. Research (run_backtest, run_optimizer, "
                    "the weekly health backtest, walk-forward, ablations, dashboard) "
                    "starts every run with exactly this much TOTAL equity, shared "
                    "across all symbols as one pool. Live uses it as the simulated "
                    "NLV target when simulate_live_capital is true. Replaces "
                    "backtest.initial_capital and the six disagreeing copies "
                    "(.env AIT_SIMULATED_CAPITAL, two --capital 50000 defaults, "
                    "WalkForwardConfig 100k, dashboard 50k).")
    simulate_live_capital: bool = Field(
        default=True,
        description="When true, broker/account.py scales every broker balance down "
                    "to `capital` so the WHOLE live risk stack sizes as that small "
                    "account while the real paper balance sits underneath. Set false "
                    "at go-live (RUNBOOK checklist). The env var AIT_SIMULATED_CAPITAL "
                    "still overrides per machine ('0' = off) because config.yaml is "
                    "shared between boxes through git.")
    capital_tier_small_min: float = Field(default=2_000.0, ge=0.0,
        description="W69: NLV at/above which the SMALL capital tier starts "
                    "(below it: MICRO). The tier selects the universe filter "
                    "in risk/capital_tiers.py. Was a literal 2_000.")
    capital_tier_medium_min: float = Field(default=5_000.0, ge=0.0,
        description="W69: NLV at/above which the MEDIUM tier starts. Was a "
                    "literal 5_000 in risk/capital_tiers.py.")
    capital_tier_large_min: float = Field(default=25_000.0, ge=0.0,
        description="W69: NLV at/above which the LARGE tier starts. Was a "
                    "literal 25_000 in risk/capital_tiers.py.")

    @model_validator(mode="after")
    def _capital_tiers_ascend(self) -> "AccountConfig":
        if not (self.capital_tier_small_min < self.capital_tier_medium_min
                < self.capital_tier_large_min):
            raise ValueError(
                "account.capital_tier_small_min < capital_tier_medium_min < "
                "capital_tier_large_min must hold strictly (got "
                f"{self.capital_tier_small_min} / {self.capital_tier_medium_min} / "
                f"{self.capital_tier_large_min})")
        return self


class PositionConfig(_StrictModel):
    max_open_positions: int = Field(default=5, ge=1, le=20)
    max_position_pct: float = Field(default=0.05, ge=0.01, le=0.30,
        description="The position sizer's hard per-position cap, fraction of NLV (a defined-risk "
                    "structure's max loss per contract must fit). Must stay >= risk.max_position_risk_pct "
                    "(Settings validates, W64). W86 (2026-10-21, operator): ceiling 0.25 -> 0.30 so the "
                    "sizer cap can sit above a raised per-trade budget (risk.max_position_risk_pct, ceiling "
                    "0.25); the value in config.yaml is unchanged. The ceilings are in code on purpose: "
                    "a limit in the same file as the value lets one edit loosen both.")
    max_portfolio_delta: float = Field(default=0.30, ge=0.05, le=1.0)
    max_portfolio_risk_pct: float = Field(default=0.20, ge=0.005, le=0.50,
        description="Aggregate capital-at-risk across ALL open positions as a "
                    "fraction of NLV. Was a phantom knob (never read) while the "
                    "real cap was hardcoded 0.20 in manager.py — now wired "
                    "(audit 2026-07-07 item 3.3). Default matches the "
                    "2026-06-30 operating value.")
    entry_ladder_start: float = Field(default=0.25, ge=0.0, le=1.0,
        description="W36: config home for AIT_ENTRY_LADDER_START — the "
                    "fraction of the way from mid to the marketable price at "
                    "which a combo entry ladder starts.")
    entry_order_timeout_seconds: int = Field(default=240, ge=30, le=3600,
        description="W36: config home for AIT_ENTRY_ORDER_TIMEOUT — how long "
                    "an entry order may work before it is cancelled.")
    max_contracts_per_trade: int = Field(default=10, ge=1, le=100,
        description="Hard cap on contracts per trade. Low values (e.g. 1) keep "
                    "cost-per-trade minimal so the account can hold many more "
                    "concurrent positions — maximizes trade COUNT for learning.")
    partial_fill_cancel_seconds: float = Field(default=30.0, ge=5.0, le=300.0,
        description="W70: an entry order that is only PARTIALLY filled has its "
                    "unfilled remainder cancelled once it has been working this long, "
                    "so the position is not left with orphaned legs riding stale "
                    "prices. Was a literal 30 in executor.check_fills. The check runs "
                    "on the 30s fast-monitor tick, so values under 30 act at the next "
                    "tick. Floor 5s: under that a normal leg-by-leg combo fill could be "
                    "cancelled mid-burst (judgement, not measured). Must stay below "
                    "entry_order_timeout_seconds, which would cancel it anyway. IMPACT: "
                    "lower = orphaned remainders cleaned up sooner but a slow fill is "
                    "cut short; higher = remainder rides at stale prices longer.")
    absent_order_grace_seconds: float = Field(default=30.0, ge=10.0, le=120.0,
        description="W70: an ENTRY order that is absent from the broker trade list is "
                    "treated as still pending (not cancelled) while younger than this. "
                    "Guards the deep-audit MP-F2b incident: a mid-cycle disconnect wipes "
                    "ib_insync's caches, so every working entry briefly looks cancelled. "
                    "Was a literal 30 in executor._determine_fill_status. Floor 10s "
                    "(judgement: the one measured acknowledgement took under 1s, the "
                    "monitor ticks every 30s, and the verdict is re-checked against the "
                    "broker anyway). Must stay below entry_order_timeout_seconds. "
                    "IMPACT: too low = a connection blip can flip live entries to "
                    "CANCELLED while still working at IBKR; too high = a genuinely "
                    "cancelled entry occupies its slot longer.")
    entry_ladder_step_seconds: float = Field(default=45.0, ge=30.0, le=300.0,
        description="W70: an unfilled entry combo's limit price steps toward marketable "
                    "every this many seconds (steps at 0.25 / 0.60 / 1.00 of the "
                    "offset). Was the class constant _LADDER_STEP_SECS = 45. Floor 30s "
                    "= the fast-monitor tick: below it the steps are skipped, not "
                    "honored (target_step = age // step). Observed: the first reprice "
                    "of a real order fired at +49.4s because it is only seen on a tick. "
                    "IMPACT: lower = chases the market faster and gives away more of "
                    "the spread; higher = more time at the patient price.")

    @model_validator(mode="after")
    def _entry_durations_fit_the_timeout(self) -> "PositionConfig":
        for name in ("partial_fill_cancel_seconds", "absent_order_grace_seconds",
                     "entry_ladder_step_seconds"):
            if getattr(self, name) >= self.entry_order_timeout_seconds:
                raise ValueError(
                    f"positions.{name} ({getattr(self, name)}) must be below "
                    f"positions.entry_order_timeout_seconds "
                    f"({self.entry_order_timeout_seconds}): the entry would be "
                    "cancelled by the timeout before this could act. Lower it, or "
                    "raise entry_order_timeout_seconds.")
        return self


class PositionSizingConfig(_StrictModel):
    """W69: the position sizer's multiplier curves (risk/position_sizer.py).

    Every default is the literal the sizer carried before this wave, so a
    config.yaml that omits the section sizes exactly as before. The sizer
    multiplies six adjustments (confidence, volatility, strategy, IV rank,
    losing-day drawdown, VIX) into the position budget; this section is those
    curves' breakpoints and factors. position_sizer.py still owns the formulas.
    """

    confidence_size_floor: float = Field(default=0.30, ge=0.05, le=1.0,
        description="Lowest confidence multiplier (the confidence curve runs "
                    "0.5 at risk.min_confidence to 1.0 at confidence 1.0 and "
                    "is clamped to [this, 1.0]).")
    iv_full_size_max: float = Field(default=0.20, ge=0.01, le=2.0,
        description="Implied vol at/below which volatility_adjustment is 1.0.")
    iv_floor_size_min: float = Field(default=0.60, ge=0.02, le=3.0,
        description="Implied vol at/above which volatility_adjustment bottoms "
                    "out at iv_floor_size_factor. Linear in between. Must "
                    "exceed iv_full_size_max.")
    iv_floor_size_factor: float = Field(default=0.50, ge=0.05, le=1.0,
        description="The volatility multiplier at/above iv_floor_size_min.")
    iv_rank_adj_min: float = Field(default=0.80, ge=0.1, le=1.0,
        description="IV-rank multiplier floor: a SELLING strategy at IV rank 0 "
                    "(or a BUYING one at rank 100) is sized at this factor.")
    iv_rank_adj_max: float = Field(default=1.20, ge=1.0, le=3.0,
        description="IV-rank multiplier ceiling: a SELLING strategy at IV rank "
                    "100 (or a BUYING one at rank 0) is sized at this factor.")
    drawdown_tiers: list[list[float]] = Field(
        default=[[3, 0.50], [5, 0.25]],
        description="[consecutive_losing_days, size_factor] pairs, ascending "
                    "by days; the highest threshold met wins (0 losing days "
                    "or below the first row = 1.0). Defensive throttle: halve "
                    "after 3 losing days, quarter after 5.")
    vix_size_tiers: list[list[float]] = Field(
        default=[[25, 0.75], [30, 0.50]],
        description="[vix_at_or_above, size_factor] pairs, ascending by VIX; "
                    "the highest threshold met wins (below the first row = "
                    "1.0). Shrinks positions where gamma risk is amplified: "
                    "VIX>=25 elevated (75%), VIX>=30 crisis (half).")
    strategy_risk: dict[str, float] = Field(
        default={
            "long_call": 1.0,        # Full premium at risk
            "long_put": 1.0,
            "covered_call": 0.5,     # Shares + short call
            "cash_secured_put": 0.5,
            "bull_call_spread": 0.6,  # Defined risk (max loss = net debit)
            "bear_put_spread": 0.6,
            "iron_condor": 0.4,      # Defined risk, both sides
            "long_straddle": 1.2,    # Double premium
            "short_strangle": 0.8,   # Undefined risk, margin required
            "calendar_spread": 0.7,  # Defined risk (debit), 2 expiries
            "event_straddle": 0.7,   # Defined risk (debit), capped at premium paid
        },
        description="Per-strategy fraction of the position budget. A strategy "
                    "missing from this table is sized at 1.0 (the sizer's "
                    "existing fallback).")

    @field_validator("drawdown_tiers", "vix_size_tiers")
    @classmethod
    def _validate_tier_rows(cls, v: list[list[float]], info) -> list[list[float]]:
        """Same discipline as RiskConfig.credit_cap_vix_tiers: the sizer
        unpacks each row into (threshold, factor) and lets the last row whose
        threshold is met win, so wrong arity, unsorted thresholds or a factor
        outside (0, 1] would silently apply the wrong size."""
        name = info.field_name
        prev = float("-inf")
        for row in v:
            if len(row) != 2:
                raise ValueError(f"{name} row {row!r} must be exactly [threshold, factor]")
            threshold, factor = row
            if not (0.0 < factor <= 1.0):
                raise ValueError(f"{name} factor {factor!r} must be in (0, 1]")
            if threshold <= prev:
                raise ValueError(
                    f"{name} thresholds must be strictly increasing "
                    f"(row {row!r} does not exceed {prev!r})")
            prev = threshold
        return v

    @field_validator("strategy_risk")
    @classmethod
    def _validate_strategy_risk(cls, v: dict[str, float]) -> dict[str, float]:
        for name, mult in v.items():
            if not (0.0 < mult <= 5.0):
                raise ValueError(
                    f"position_sizing.strategy_risk[{name!r}]={mult!r} must be in (0, 5]")
        return v

    @model_validator(mode="after")
    def _iv_curve_is_ordered(self) -> "PositionSizingConfig":
        if self.iv_floor_size_min <= self.iv_full_size_max:
            raise ValueError(
                f"position_sizing.iv_floor_size_min ({self.iv_floor_size_min}) must "
                f"exceed iv_full_size_max ({self.iv_full_size_max})")
        if self.iv_rank_adj_min > self.iv_rank_adj_max:
            raise ValueError(
                "position_sizing.iv_rank_adj_min must not exceed iv_rank_adj_max")
        return self


class RiskConfig(_StrictModel):
    max_daily_loss_pct: float = Field(default=0.045, ge=0.005, le=0.20,
        description="Daily loss breaker, fraction of NLV (realised AND mark-to-market "
                    "trips; entries only, exits keep running). W64: 0.02 -> 0.045 at the "
                    "$5k account (breaker $225 vs the $3-wide condor's $201-212 max "
                    "loss on real chains); the startup self-test's rule is breaker >= "
                    "breaker_to_max_loss_ratio x the largest max loss the builder emits. "
                    "W86 (2026-10-21, operator): ceiling 0.10 -> 0.20 (it dated from the initial "
                    "commit, no recorded reason); the value is unchanged. A breaker below one "
                    "full max loss is reported by the startup check (alert-only) and, as always, "
                    "only blocks ENTRIES after the loss.")
    breaker_to_max_loss_ratio: float = Field(default=1.0, ge=0.5, le=3.0,
        description="W64: the startup coherence check (orchestrator A2) alerts when the "
                    "daily-loss breaker is below this multiple of the LARGEST single max "
                    "loss the real condor builder emits. Was a hardcoded 1.5 (one stopped "
                    "trade can't halt the day alone). 1.0 = one full max-loss condor "
                    "exactly reaches the breaker; 1.5 = the original, most conservative "
                    "rule. Alert-only: it never changes live sizing or the breaker.")
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    pause_minutes_after_losses: int = Field(default=30, ge=5, le=120)
    max_api_failures: int = Field(default=5, ge=2, le=20)
    min_confidence: float = Field(default=0.65, ge=0.50, le=0.95)
    pre_event_blackout_days: int = Field(default=1, ge=0, le=7,
        description="Block NEW credit entries within N calendar days of a "
                    "macro event (NFP/CPI/PCE). PLAN 2026-08-03: relaxed 4->1 "
                    "(user-approved) - the <=4 window blacked out ~half of all "
                    "trading days and refused the richest premium; every 14-30 "
                    "DTE hold spans events regardless, wings cap the surprise.")
    max_position_risk_pct: float = Field(default=0.05, ge=0.005, le=0.25,
        description="Per-trade max_loss cap as a fraction of NLV (was "
                    "hardcoded 0.03 in manager.py — audit item 3.3), AND the budget the "
                    "condor builder is handed (wing width = budget/90 snapped to the "
                    "strike grid). History: 0.03 -> 0.0133 on 2026-09-29 to match the "
                    "breaker (max_daily_loss_pct / 1.5); W64 (2026-10-08) -> 0.05 at the "
                    "$5k account, MEASURED on live TWS chains: budgets $66/$150 build "
                    "nothing at the $0.70 credit floor ($2-wide credits only $0.62-0.68); "
                    "$250 builds a $3-wide condor (credit $0.88-0.99, max loss "
                    "$201-212). Must stay <= positions.max_position_pct (the sizer's "
                    "cap) — Settings validates it (W64). W86 (2026-10-21, operator): ceiling "
                    "0.10 -> 0.25; the value is unchanged. A raise is only accepted while it also "
                    "stays <= positions.max_portfolio_risk_pct and <= risk.max_symbol_concentration_pct "
                    "(Settings validates): above either, gate 6b-2 / 6c would refuse EVERY entry, "
                    "silently. Above about 4.5k of budget the vol-scaled wing width binds before the "
                    "budget does (strategies/iron_condor.py).")
    max_correlation: float = Field(default=0.75, ge=0.30, le=0.99,
        description="Correlation above which two symbols count as the same bet.")
    max_correlated_positions: int = Field(default=2, ge=1, le=8,
        description="Max simultaneous positions within one correlated cluster "
                    "(SPY/QQQ/IWM/DIA correlate ~0.95). Was a CorrelationGuard "
                    "code default — audit item 3.3.")
    post_stop_cluster_cooldown_minutes: int = Field(default=120, ge=0, le=1440,
        description="W26 (logic-exit-risk-03): post-stop re-entry cooldown for "
                    "CLUSTER SIBLINGS of a stopped-out symbol. The stopped "
                    "symbol itself keeps the full R12-B4 one-trading-day window; "
                    "siblings get this shorter one. Graduated on purpose: the "
                    "universe (SPY/QQQ/IWM) is ONE cluster, so applying the full "
                    "trading-day window cluster-wide darkens the whole book for "
                    "the rest of the session AND all of the next (a Mon 14:00 "
                    "stop clears only Wed), which fights the volume-for-learning "
                    "config. The hazard R12-B4 targets is IMMEDIATE re-entry into "
                    "the same adverse move — the live 08-24 QQQ instance was 13 "
                    "minutes. 0 DISABLES sibling scoping entirely, restoring the "
                    "pre-W26 per-symbol behaviour without a code change.")
    max_credit_positions: int = Field(default=6, ge=1, le=20,
        description="Max simultaneous short-premium (credit) positions. The "
                    "delta gate is dead and the daily breaker only sees "
                    "realized P&L, so without this the whole book can be "
                    "short vol into a gap (audit R2).")
    credit_cap_vix_tiers: list[list[float]] = Field(
        default=[[20.0, 6], [25.0, 4], [999.0, 2]],
        description="R20: VIX-tiered credit-position caps as [vix_below, cap] "
                    "pairs, first match wins — config home for the hardcoded "
                    "'6 if vix<20 else 4 if vix<25 else 2' in manager.py "
                    "(register CD item). max_credit_positions stays the "
                    "absolute ceiling.")
    max_symbol_concentration_pct: float = Field(default=0.20, ge=0.05, le=0.50,
        description="R20: max fraction of account in ONE symbol (gate 6c) — "
                    "was a hardcoded 0.20 that numerically shadowed "
                    "positions.max_portfolio_risk_pct while meaning something "
                    "different.")
    iv_rank_cap: float = Field(default=85.0, ge=0.0, le=100.0,
        description="W36: config home for AIT_IV_RANK_CAP — no new short "
                    "premium above this IV rank (R12: top-bucket IV carried "
                    "negative VRP).")
    max_positions_per_expiry: int = Field(default=3, ge=1, le=20,
        description="W36: config home for AIT_MAX_PER_EXPIRY — positions "
                    "sharing one expiry stack pin/gamma risk into a single "
                    "settlement date.")
    skip_macro_events: bool = Field(default=True,
        description="R19c: macro-event protection (entry gates + rule-3d "
                    "flatten for undefined-risk shapes). Config home for "
                    "AIT_SKIP_MACRO_EVENTS ('1'=on). Protective default ON; "
                    "2026-07-08 user decision.")
    credit_vix_halt: float = Field(default=28.0, ge=15.0, le=60.0,
        description="No NEW credit entries when VIX is at/above this level — "
                    "cheap vol-regime brake for short-premium strategies.")
    weekend_gap_cutoff_et: str = Field(default="14:30",
        description="W69: from this ET time on a Friday, weekend_gap_strategies "
                    "need weekend_gap_min_confidence to be entered (weekend gap "
                    "risk for undefined-risk shorts). Was `now.hour == 14 and "
                    "now.minute >= 30` in risk/manager.py, on a naive local "
                    "clock (the W20/W21 bug class) — it now reads now_et().")
    weekend_gap_min_confidence: float = Field(default=0.90, ge=0.50, le=0.99,
        description="W69: confidence a weekend_gap_strategies entry needs after "
                    "weekend_gap_cutoff_et on Friday. Was a literal 0.90 in "
                    "risk/manager.py.")
    weekend_gap_strategies: list[str] = Field(default=["short_strangle"],
        description="W69: strategies subject to the Friday-afternoon "
                    "confidence rule. Iron condors are defined-risk and long "
                    "straddles benefit from weekend vol, so only the "
                    "undefined-risk short is listed by default. Was the "
                    "literal tuple ('short_strangle',) in risk/manager.py.")

    @field_validator("weekend_gap_cutoff_et")
    @classmethod
    def _validate_weekend_cutoff(cls, v: str) -> str:
        _parse_hhmm(v, "weekend_gap_cutoff_et")
        return v

    @property
    def weekend_gap_cutoff_minutes(self) -> int:
        """weekend_gap_cutoff_et as minutes since ET midnight (14:30 -> 870)."""
        h, m = _parse_hhmm(self.weekend_gap_cutoff_et, "weekend_gap_cutoff_et")
        return h * 60 + m

    @field_validator("credit_cap_vix_tiers")
    @classmethod
    def validate_credit_cap_vix_tiers(cls, v: list[list[float]]) -> list[list[float]]:
        """R20b review follow-up: manager.py consumes each row via two-value
        unpacking (`for ceiling, cap in ...`) and takes the FIRST match in
        list order — a malformed row (wrong arity), a non-integer/non-positive
        cap, or unsorted ceilings all passed startup validation before this
        fix and could crash trade validation or silently apply the wrong cap.
        """
        if not v:
            raise ValueError("credit_cap_vix_tiers must not be empty")
        prev_ceiling = float("-inf")
        for row in v:
            if len(row) != 2:
                raise ValueError(
                    f"credit_cap_vix_tiers row {row!r} must be exactly "
                    "[vix_ceiling, cap]"
                )
            ceiling, cap = row
            if cap <= 0 or int(cap) != cap:
                raise ValueError(
                    f"credit_cap_vix_tiers cap {cap!r} must be a positive integer"
                )
            if ceiling <= prev_ceiling:
                raise ValueError(
                    "credit_cap_vix_tiers ceilings must be strictly increasing "
                    f"(row {row!r} does not exceed the prior ceiling "
                    f"{prev_ceiling!r})"
                )
            prev_ceiling = ceiling
        return v


# W88b: the ONE schema default of the Black-Scholes risk-free rate; the research pricing helpers (pricing.py,
# replay.py, iv_surface.py, the engine's partial-engine class default) import it instead of repeating 0.05.
DEFAULT_RISK_FREE_RATE = 0.05


class OptionsConfig(_StrictModel):
    delta_range: list[float] = [0.20, 0.50]
    dte_range: list[int] = [14, 45]
    min_open_interest: int = Field(default=100, ge=0)
    min_volume: int = Field(default=50, ge=0)
    max_bid_ask_spread_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    risk_free_rate: float = Field(default=DEFAULT_RISK_FREE_RATE, ge=0.0, le=0.25,
        description="W36: the Black-Scholes risk-free rate used to solve IVs "
                    "and backfill Greeks (was a bare `r = 0.05` in "
                    "options_chain._calculate_greeks). No dividend yield is "
                    "modelled: the R24 register checked carry/dividend "
                    "structure and cleared it.")
    strategies: list[str] = [
        "long_call",
        "long_put",
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
    ]

    @field_validator("delta_range")
    @classmethod
    def validate_delta_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or not (0 < v[0] < v[1] < 1):
            raise ValueError("delta_range must be [low, high] where 0 < low < high < 1")
        return v

    @field_validator("dte_range")
    @classmethod
    def validate_dte_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or not (1 <= v[0] < v[1] <= 365):
            raise ValueError("dte_range must be [min_dte, max_dte] where 1 <= min < max <= 365")
        return v


class DteSearchConfig(_StrictModel):
    """W79a: the opt-in DTE dimensions of an Optuna study (research only).

    Off by default and switched on per study (``run_backtest.py --search-dte``): the existing
    ``IRON_CONDOR_SPACE`` is untouched, so every other study is unchanged. When enabled the
    condor search ADDS entry DTE, the expiry guard and the hold fraction (the fraction REPLACES
    ``max_hold_days`` for credit structures, so "hold between 0 and DTE" is one dimension for
    every entry DTE). The credit floors (``ic_min_credit*``) are deliberately NOT searched: they
    are live economics floors and a search would optimise against an unmeasured cost model.
    """

    enabled: bool = Field(default=False,
        description="W79a: add the DTE dimensions to the iron-condor search. Per-study switch "
                    "(CLI --search-dte); requires optimize_per_window.")
    entry_dte: list[int] = Field(default=[0, 45],
        description="W79a: inclusive range of the searched entry DTE (calendar days). The "
                    "operator's stated research range is 0-45; the live dte_range validator "
                    "still forbids 0 until W80.")
    guard_dte: list[int] = Field(default=[0, 5],
        description="W79a: inclusive range of the searched expiry_guard_dte: entries with at "
                    "most this many DTE are exempt from live rules 3/3a (the close at DTE<=5). "
                    "0-5 spans 'rules apply to everything' to 'every short entry is exempt'.")
    hold_fraction: list[float] = Field(default=[0.05, 1.0],
        description="W79a: range of the searched max_hold_fraction: close after this share of "
                    "the position's own life. 1.0 = hold to expiry (or the flatten).")
    search_flatten: bool = Field(default=False,
        description="W79a: also search flatten_minutes_before_close. OFF because the W77 study "
                    "and the W79a flatness check show it has NO effect while the take-profit "
                    "ladder is on (positions never reach the bell): a flat dimension only "
                    "wastes trials (the W65 inert-wing_k rule).")
    flatten_minutes: list[int] = Field(default=[30, 120],
        description="W79a: range of the searched flatten (minutes before the close) when "
                    "search_flatten is on. A hard backstop, not a profit lever.")
    multivariate: bool = Field(default=False,
        description="W79a: TPESampler(multivariate=True, group=True). Off: the search space is "
                    "conditional (the guard decides which of the other dimensions bind) and "
                    "grouped TPE is an experimental sampler in optuna.")
    strict: bool = Field(default=False,
        description="W79a: a per-window optimisation that FAILS aborts the study instead of "
                    "silently falling back to the base config (the W79 'searched nothing' "
                    "failure). Recommended on for any study whose numbers will be quoted.")

    @staticmethod
    def _pair(v, lo, hi, name, floatish=False):
        if len(v) != 2 or not (lo <= v[0] <= v[1] <= hi):
            raise ValueError(f"{name} must be [low, high] with {lo} <= low <= high <= {hi}")
        return v

    @field_validator("entry_dte")
    @classmethod
    def _v_entry(cls, v: list[int]) -> list[int]:
        return cls._pair(v, 0, 45, "dte_search.entry_dte")

    @field_validator("guard_dte")
    @classmethod
    def _v_guard(cls, v: list[int]) -> list[int]:
        return cls._pair(v, 0, 45, "dte_search.guard_dte")

    @field_validator("hold_fraction")
    @classmethod
    def _v_frac(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or not (0.0 < v[0] <= v[1] <= 1.0):
            raise ValueError("dte_search.hold_fraction must be [low, high] with 0 < low <= high <= 1")
        return v

    @field_validator("flatten_minutes")
    @classmethod
    def _v_flat(cls, v: list[int]) -> list[int]:
        return cls._pair(v, 5, 390, "dte_search.flatten_minutes")


class LearnerConfig(_StrictModel):
    """W88b: thresholds of the research ``BacktestLearner`` (backtesting/learner.py), which adapts strategy
    multipliers, the confidence floor and symbol disabling between walk-forward windows. RESEARCH ONLY: it is a
    separate mechanism from the live learning layer (``learning:``), so there is no parity link. Every default is
    the literal the module used before W88; the research overlay file lists them all."""

    windows_before_adapting: int = Field(default=2, ge=1, le=20,
        description="Windows whose trades must be accumulated before any adaptation (the first window only "
                    "accumulates).")
    min_trades_to_learn: int = Field(default=15, ge=2, le=500,
        description="A strategy / symbol / confidence band needs this many trades before it is judged.")
    min_trades_to_disable: int = Field(default=25, ge=2, le=1000,
        description="Trades needed before a strategy or symbol is disabled outright (more evidence than reducing).")
    protected_strategies: list[str] = Field(default=["iron_condor"],
        description="Strategies the learner may reduce but never disable (condors are regime-dependent).")
    strategy_multiplier_min: float = Field(default=0.30, ge=0.0, le=1.0,
        description="Floor of a strategy's sizing multiplier after reductions.")
    strategy_multiplier_max: float = Field(default=2.0, ge=1.0, le=10.0,
        description="Cap of a strategy's sizing multiplier after boosts.")
    reduce_win_rate: float = Field(default=0.30, ge=0.0, le=1.0,
        description="Win rate below which (with negative average P&L) a strategy is reduced / disabled.")
    reduce_factor: float = Field(default=0.80, gt=0.0, lt=1.0,
        description="Multiplier applied to a losing strategy's size multiplier per adaptation.")
    reduce_min_change: float = Field(default=0.03, ge=0.0, le=0.5,
        description="A reduction smaller than this (absolute multiplier change) is not applied.")
    boost_win_rate: float = Field(default=0.58, ge=0.0, le=1.0,
        description="Win rate above which (with positive average P&L) a strategy is boosted.")
    boost_min_trades: int = Field(default=6, ge=1, le=500,
        description="Trades needed to boost. NOTE: inert at any value <= min_trades_to_learn (the shipped 6 vs 15), "
                    "because that already gates the strategy loop; it bites once set above it.")
    boost_factor: float = Field(default=1.20, gt=1.0, le=3.0,
        description="Multiplier applied to a winning strategy's size multiplier per adaptation.")
    boost_min_change: float = Field(default=0.05, ge=0.0, le=0.5,
        description="A boost smaller than this (absolute multiplier change) is not applied.")
    reenable_win_rate: float = Field(default=0.50, ge=0.0, le=1.0,
        description="Win rate above which a disabled strategy is re-enabled.")
    reenable_min_trades: int = Field(default=10, ge=1, le=500,
        description="Trades needed to re-enable. NOTE: inert at any value <= min_trades_to_learn (the shipped 10 "
                    "vs 15); it bites once set above it.")
    reenable_multiplier: float = Field(default=0.5, gt=0.0, le=1.0,
        description="Size multiplier a re-enabled strategy restarts at.")
    confidence_cap: float = Field(default=0.85, ge=0.5, le=1.0,
        description="Highest minimum-confidence the learner may raise the entry floor to.")
    confidence_low_band_upper: float = Field(default=0.65, ge=0.0, le=1.0,
        description="Entry confidence below this is the 'low' band the learner watches.")
    confidence_mid_band_upper: float = Field(default=0.75, ge=0.0, le=1.0,
        description="Entry confidence below this (and not low) is the 'medium' band.")
    confidence_raise_win_rate: float = Field(default=0.35, ge=0.0, le=1.0,
        description="If the low band wins less often than this (with negative average P&L), the floor is raised.")
    confidence_raise_step: float = Field(default=0.05, gt=0.0, le=0.5,
        description="How much the minimum-confidence floor is raised per adaptation.")
    confidence_raise_min_change: float = Field(default=0.01, ge=0.0, le=0.5,
        description="A raise smaller than this is not applied (the cap was reached).")
    symbol_disable_win_rate: float = Field(default=0.20, ge=0.0, le=1.0,
        description="Win rate below which (with negative average P&L and min_trades_to_disable trades) a symbol "
                    "is disabled for the following windows.")
    default_entry_confidence: float = Field(default=0.65, ge=0.0, le=1.0,
        description="Confidence assumed for a trade record that carries none.")

    @model_validator(mode="after")
    def _ordered(self) -> "LearnerConfig":
        if not self.strategy_multiplier_min <= 1.0 <= self.strategy_multiplier_max:
            raise ValueError("learner: strategy_multiplier_min <= 1 <= strategy_multiplier_max")
        if not self.confidence_low_band_upper < self.confidence_mid_band_upper:
            raise ValueError("learner: confidence_low_band_upper must be below confidence_mid_band_upper")
        if self.min_trades_to_disable < self.min_trades_to_learn:
            raise ValueError("learner: min_trades_to_disable must be >= min_trades_to_learn")
        return self


class LegacyEngineConfig(_StrictModel):
    """W88b: constants of the research engine's LEGACY / dormant paths (the fallback direction model used when no
    predictor is loaded, the naked-margin proxy of the strangle builders, the directional-entry margin and the
    thesis-invalidation exit). None is on the iron-condor path under ``engine_model: realistic``, but each changes
    results for the structures that do use it, so each is visible here. Defaults = the pre-W88 literals."""

    naked_margin_fraction: float = Field(default=0.20, gt=0.0, le=1.0,
        description="Margin proxy for NAKED short structures (short strangle): this fraction of spot x 100 per "
                    "contract. Defined-risk condors size from max loss and never read it.")
    fallback_weight_5d: float = Field(default=0.6, ge=0.0, le=1.0,
        description="Weight of the 5-day return in the fallback direction score.")
    fallback_weight_20d: float = Field(default=0.4, ge=0.0, le=1.0,
        description="Weight of the 20-day return in the fallback direction score.")
    fallback_threshold: float = Field(default=0.005, ge=0.0, le=0.1,
        description="Score magnitude above which the fallback calls BULLISH / BEARISH rather than NEUTRAL.")
    fallback_confidence_base: float = Field(default=0.5, ge=0.0, le=1.0,
        description="Confidence of a directional fallback call at zero score magnitude.")
    fallback_confidence_slope: float = Field(default=10.0, ge=0.0, le=100.0,
        description="Confidence added per unit of score magnitude.")
    fallback_confidence_cap: float = Field(default=0.95, ge=0.5, le=1.0,
        description="Cap of a directional fallback confidence.")
    fallback_neutral_confidence: float = Field(default=0.4, ge=0.0, le=1.0,
        description="Confidence the fallback reports for a NEUTRAL call.")
    bearish_confidence_margin: float = Field(default=0.05, ge=0.0, le=0.5,
        description="A BEARISH directional entry needs this much more confidence than min_confidence.")
    thesis_invalidation_confidence: float = Field(default=0.80, ge=0.5, le=1.0,
        description="Legacy engine only: a prediction at/above this confidence AGAINST an open directional "
                    "position closes it (inert under engine_model realistic, where condors skip the check).")

    @model_validator(mode="after")
    def _weights(self) -> "LegacyEngineConfig":
        if self.fallback_weight_5d + self.fallback_weight_20d <= 0:
            raise ValueError("legacy_engine: the fallback direction weights cannot both be zero")
        return self


class ObjectivesConfig(_StrictModel):
    """W88b: constants of the Optuna objectives (optimization/objectives.py): the ``composite`` objective is
    sharpe_weight x Sharpe + win_rate_weight x win rate - drawdown_penalty x |max drawdown|, and the
    ``profit_factor`` objective is capped so an infinite profit factor stays comparable."""

    sharpe_weight: float = Field(default=0.4, ge=0.0, le=10.0, description="Weight of the Sharpe ratio.")
    win_rate_weight: float = Field(default=0.4, ge=0.0, le=10.0, description="Weight of the win rate.")
    drawdown_penalty: float = Field(default=0.2, ge=0.0, le=10.0,
        description="Weight of the absolute max drawdown, subtracted.")
    profit_factor_cap: float = Field(default=10.0, gt=1.0, le=1000.0,
        description="The profit_factor objective is min(profit factor, this): an infinite factor (no losing "
                    "trade) must stay finite for Optuna's pruner.")


class WalkForwardTuningConfig(_StrictModel):
    """W88b: walk-forward / optimizer constants that used to be literals in walkforward.py, optimizer.py and
    the range-threshold helper. RESEARCH ONLY. Defaults = the pre-W88 literals."""

    warm_start_min_win_rate: float = Field(default=0.75, ge=0.0, le=1.0,
        description="A window's Optuna study is warm-started from the PRIOR window's best parameters only if that "
                    "window's out-of-sample win rate was at least this (else from the global best).")
    warm_start_min_trades: int = Field(default=3, ge=0, le=100,
        description="...and it had at least this many trades.")
    range_threshold_multiplier: float = Field(default=1.25, gt=0.0, le=5.0,
        description="Adaptive range threshold = realized vol x sqrt(horizon/252) x this (about 1.25 standard "
                    "deviations).")
    range_threshold_low_clip: float = Field(default=0.02, gt=0.0, le=0.5,
        description="Lower clip of the adaptive range threshold.")
    range_threshold_high_clip: float = Field(default=0.15, gt=0.0, le=0.9,
        description="Upper clip of the adaptive range threshold.")
    range_threshold_lookback_days: int = Field(default=60, ge=10, le=504,
        description="Trailing days of training data the adaptive threshold's realized vol uses.")
    range_threshold_min_observations: int = Field(default=10, ge=2, le=504,
        description="Fewer return observations than this and the threshold falls back.")
    range_threshold_fallback: float = Field(default=0.05, gt=0.0, le=0.5,
        description="The fixed +/- containment band used when the adaptive threshold cannot be computed "
                    "(= live's range-model band).")
    optimize_val_split_ratio: float = Field(default=0.20, gt=0.0, lt=0.5,
        description="Fraction of an Optuna training window held out as validation when the objective is "
                    "scored on a validation slice (H2).")
    inert_probe_iv_fallback: float = Field(default=0.15, gt=0.0, le=2.0,
        description="IV assumed by the 'is wing_k inert' probe when the data carries no implied vol.")
    inert_probe_iv_quantile: float = Field(default=0.10, ge=0.0, le=1.0,
        description="Quantile of the data's IV the probe uses (the low end: the narrowest wing the study needs).")
    inert_probe_dte_fallback: int = Field(default=21, ge=1, le=60,
        description="Entry DTE the probe assumes when no settings are loaded.")
    optimize_seed: int = Field(default=42, ge=0, le=2**31 - 1,
        description="W90a: seed of the Optuna TPE sampler (the walk-forward's per-window studies and a bare StrategyOptimizer). "
                    "Was a literal 42 in WalkForwardConfig and StrategyOptimizer.")
    # --- W90b: the walk-forward geometry and evaluation constants (were dataclass / helper literals) ---
    train_days: int = Field(default=365, ge=60, le=3650,
        description="W90b: calendar days of each walk-forward TRAINING window (also the StrategyOptimizer's history "
                    "window). Was a literal in WalkForwardConfig, StrategyOptimizer and the run_backtest / run_optimizer "
                    "--train-days defaults; --train-days still overrides it for one run.")
    test_days: int = Field(default=63, ge=5, le=730,
        description="W90b: calendar days of each out-of-sample TEST window (~3 months).")
    step_days: int = Field(default=63, ge=1, le=730,
        description="W90b: calendar days the window slides forward. Equal to test_days = NON-overlapping test windows "
                    "(BT-M5: a 21-day step with a 63-day test simulated every date ~3x and triple-counted trades).")
    gap_days: int = Field(default=5, ge=0, le=90,
        description="W90b: purge gap between the end of training and the start of the test window.")
    optimize_n_trials: int = Field(default=50, ge=1, le=5000,
        description="W90b: Optuna trials per walk-forward window when --optimize-per-window is on. --optimize-n-trials "
                    "overrides it for one run.")
    optimizer_n_trials: int = Field(default=100, ge=1, le=5000,
        description="W90b: Optuna trials of a standalone StrategyOptimizer / run_optimizer.py run (--n-trials overrides).")
    min_train_rows: int = Field(default=50, ge=10, le=2000,
        description="W90b: a symbol with fewer training-window bars than this is skipped for the window (no model can be "
                    "trained on it).")
    min_test_rows: int = Field(default=5, ge=1, le=500,
        description="W90b: ...and one with fewer test-window bars than this is skipped.")
    context_padding_days: int = Field(default=900, ge=0, le=5000,
        description="W90b: extra calendar days of VIX / SPY context loaded before the training window so the slowest "
                    "feature windows are warm (context = train_days + this).")
    context_span_padding_days: int = Field(default=400, ge=0, le=5000,
        description="W90b: ...or, when a research history span is configured, the whole span plus this.")
    fetch_padding_days: int = Field(default=100, ge=0, le=1000,
        description="W90b: calendar days added to train_days + test_days when fetching the price history (a margin for "
                    "weekends, holidays and feature warm-up).")
    history_span_padding_days: int = Field(default=30, ge=0, le=365,
        description="W90b: calendar days added to the configured research-history span when sizing the price fetch.")
    oos_direction_return_threshold: float = Field(default=0.02, gt=0.0, le=0.5,
        description="W90b: the forward return that separates bullish / neutral / bearish when the out-of-sample "
                    "DIRECTION model is scored (a diagnostic; it does not change any trade).")
    oos_calibration_bins: int = Field(default=8, ge=2, le=50,
        description="W90b: confidence bins of the out-of-sample direction-model reliability chart.")
    prob_score_bins: int = Field(default=9, ge=2, le=50,
        description="W90b: probability bins of the out-of-sample range-model reliability chart.")
    oos_min_extra_rows: int = Field(default=20, ge=0, le=500,
        description="W90b: the out-of-sample direction evaluation needs this many bars beyond its label horizon.")
    oos_min_labelled_rows: int = Field(default=10, ge=2, le=500,
        description="W90e: ... and at least this many labelled rows, else it reports nothing for the window.")
    oos_range_min_extra_rows: int = Field(default=5, ge=0, le=500,
        description="W90e: the out-of-sample RANGE evaluation needs this many bars beyond its label horizon.")
    min_windows: int = Field(default=3, ge=1, le=100,
        description="W90e: a walk-forward run with fewer windows than this aborts (and run_backtest.py warns that too few "
                    "windows produced results): too few to judge anything.")
    window_score_full_credit_trades: int = Field(default=5, ge=1, le=500,
        description="W90e: a window's score for the global best parameters is win rate x sqrt(min(1, trades / this)), so a "
                    "window with fewer trades than this counts for proportionally less.")
    oos_feature_context_bars: int = Field(default=252, ge=20, le=2000,
        description="W95: bars of the training window's tail that are put in front of each out-of-sample test window so the "
                    "features have their look-back (capped at the training window's length, so early windows still run). The "
                    "default is one trading year: the IV-rank feature looks back ml.features.vol_regime.iv_rank_lookback (252) "
                    "bars. A shorter context does not fail, it changes the feature VALUES the model sees in the test window "
                    "(iv_rank below 252, distance_sma200 below 200) so they differ from the ones it was trained on; for that "
                    "reason a value below the feature spec's longest look-back (FeatureSpec.longest_lookback(): by default the IV-rank "
                    "range, 252) is refused when the walk-forward starts. Not "
                    "backtest.context_bars, which is the engine's own intraday look-back.")
    meta_label_min_window_rows: int = Field(default=60, ge=1, le=5000,
        description="W95: the meta-labeler is trained for a window only when the training window has MORE than this many bars "
                    "(otherwise the window runs without one). A minimum-length guard, not a context length: the labelling run "
                    "uses the whole training window.")
    min_fetched_rows: int = Field(default=100, ge=0, le=10000,
        description="W95: a symbol is used by a walk-forward run only when MORE than this many daily bars were fetched for it "
                    "(fewer is treated as no data).")
    min_backtest_rows: int = Field(default=60, ge=4, le=5000,
        description="W95: the optimizer's minimum number of daily bars. A symbol fetched with fewer is left out, and a trial on "
                    "fewer raises 'insufficient data'. With a validation split on, the split is made only with at least twice "
                    "this many bars, the training part is never shorter than this, and a validation slice shorter than half of "
                    "it is dropped (the whole window is scored instead).")

    @model_validator(mode="after")
    def _clips(self) -> "WalkForwardTuningConfig":
        if not self.range_threshold_low_clip < self.range_threshold_high_clip:
            raise ValueError("walkforward_tuning: range_threshold_low_clip must be below range_threshold_high_clip")
        return self


class EngineTuningConfig(_StrictModel):
    """W90c: constants of the research engine's LIVE path (iron condors) that were literals in ``Backtester`` and
    ``pricing``: the IV-skew coefficients of the leg-pricing model, the look-back windows of the realized-vol and
    IV-rank-rise entry gates, and the minimum history before the first decision. RESEARCH ONLY. Defaults = the
    pre-W90c literals."""

    put_skew_coefficient: float = Field(default=0.10, ge=0.0, le=2.0,
        description="Legacy (pre-W5) put-skew term of the leg-IV model: skew_factor x (log-moneyness OTM) x this. The "
                    "W5 measured steepening (skew_slope_per_pct_otm) is added on top.")
    call_skew_coefficient: float = Field(default=0.02, ge=0.0, le=2.0,
        description="Call-skew term of the leg-IV model: skew_factor x max(0, ln(K/S)) x this. Mild on purpose: the "
                    "measured call-side slope was sign-unstable across symbols (W5), so there is no measured value.")
    iv_rank_rise_lookback_rows: int = Field(default=11, ge=2, le=252,
        description="Rows of the feature IV rank the rise veto looks at: the veto compares the last value with the "
                    "one this many rows back minus one (11 rows = a 10-day rise).")
    vol_gate_lookback_rows: int = Field(default=11, ge=3, le=252,
        description="Rows of closes the realized-vol entry gate measures (11 closes = 10 daily returns).")
    min_history_bars: int = Field(default=20, ge=5, le=252,
        description="The engine takes no decision before this many bars of history (the larger of this and "
                    "backtest.context_bars is skipped at the start).")
    fallback_iv: float = Field(default=0.25, ge=0.01, le=3.0,
        description="W90e: the annualized volatility the engine assumes when it has nothing better: a position that carries "
                    "no entry_iv (exit repricing, half-spread model) and a realized-vol estimate with too few prices.")
    drawdown_high_lookback_rows: int = Field(default=60, ge=2, le=1000,
        description="W95: window (rows of daily closes) of the rolling high the drawdown entry gate measures the fall from "
                    "(backtest.pct_from_60d_high_threshold); fewer rows are used while the history is shorter. The result "
                    "keeps the name pct_from_60d_high whatever this is set to.")
    replay_lookback_days: int = Field(default=60, ge=0, le=3650,
        description="W95: calendar days of history the single-trade replay (backtesting/replay.py) gives the engine in front "
                    "of the trade's entry date. 60 calendar days is about 42 trading rows, so a replayed engine's gates that "
                    "look back further than that (the drawdown gate's 60 rows) run on a shorter history than in a full run.")


class SurfaceTuningConfig(_StrictModel):
    """W90f: the acceptance rules of the research IV surface (``backtesting/iv_surface.py``), which prices legs from real option
    bars when ``backtest.pricing_source`` is ``surface``. RESEARCH ONLY. Defaults = the pre-W90f module constants. A surface is
    cached on disk; the fit rules are part of its cache key, so changing one rebuilds it."""

    min_volume: int = Field(default=10, ge=0, le=100000,
        description="A bar needs at least this daily volume to enter a smile fit (thin bars have noisy solved IVs).")
    min_price: float = Field(default=0.05, ge=0.0, le=10.0,
        description="... and a close of at least this many dollars (below it the IV is tick noise).")
    max_abs_moneyness: float = Field(default=0.10, gt=0.0, le=1.0,
        description="Only strikes within |ln(K/S)| of this are fitted (10% either side of the money).")
    min_points: int = Field(default=5, ge=3, le=1000,
        description="A (date, expiry) smile is fitted only from at least this many bars (a quadratic needs 3; more makes it stable).")
    min_span: float = Field(default=0.01, ge=0.0, le=1.0,
        description="... and only when the fitted bars cover at least this much log-moneyness (1%).")
    iv_lower_bound: float = Field(default=0.03, gt=0.0, lt=5.0,
        description="Solved implied vols below this are discarded as unsolvable / noise.")
    iv_upper_bound: float = Field(default=2.5, gt=0.0, le=10.0,
        description="... and above this.")
    clamp_margin: float = Field(default=0.01, ge=0.0, le=1.0,
        description="A smile is evaluated at most this far (in log-moneyness) outside the range it was fitted on; beyond it the "
                    "edge value is held flat.")
    max_stale_days: int = Field(default=5, ge=0, le=60,
        description="A query may use a surface dated at most this many calendar days before it (weekends and holidays); older is "
                    "treated as no surface.")

    @model_validator(mode="after")
    def _bounds(self) -> "SurfaceTuningConfig":
        if not self.iv_lower_bound < self.iv_upper_bound:
            raise ValueError("surface_tuning: iv_lower_bound must be below iv_upper_bound")
        return self


class FrictionFitConfig(_StrictModel):
    """W90f: the minimum data a leg-spread (friction) model needs to be fitted (``backtesting/friction.py``). Used by the offline
    calibration (``scripts/calibrate_option_spreads.py``, the short-DTE collector), not by a backtest. RESEARCH ONLY. Defaults =
    the pre-W90f literals."""

    min_samples: int = Field(default=5, ge=3, le=100000,
        description="Fewer usable (mid, half-spread) samples than this and no model is fitted at all.")
    min_sessions_for_cv: int = Field(default=3, ge=2, le=1000,
        description="Leave-one-session-out cross-validation needs at least this many trading days; with fewer the forms are "
                    "compared in-sample.")
    min_train_rows: int = Field(default=3, ge=1, le=100000,
        description="A cross-validation fold whose training part has fewer rows than this is skipped.")
    min_affine_points: int = Field(default=3, ge=2, le=100000,
        description="The affine form (a + b * mid) is fitted only from at least this many points, else it falls back to a constant.")


class SpreadTuningConfig(_StrictModel):
    """W90f: sizing numbers of the research engine's single-structure builders: the put / call credit spreads, the iron butterfly,
    the jade lizard and the debit structures. The iron-condor builder does not read them (it sizes from the risk budget and
    ``wing_k``). RESEARCH ONLY: the live selector trades iron condors only, so these are for studies of the other structures
    before any of them is made live. Defaults = the pre-W90f literals."""

    max_cost_fraction_of_capital: float = Field(default=0.25, gt=0.0, le=1.0,
        description="A single spread's max loss (credit structures) or cost (debit structures) per contract may not exceed this "
                    "fraction of capital; a larger one is not opened.")
    wing_expected_move_fraction: float = Field(default=0.5, gt=0.0, le=5.0,
        description="Wing width of the iron butterfly, put / call credit spreads and jade lizard = wing_k x expected move x this "
                    "(floored at the wing floor). 0.5 = half an expected move.")


class BacktestConfig(_StrictModel):
    @field_validator("default_symbols")
    @classmethod
    def _valid_default_symbols(cls, v: list[str]) -> list[str]:
        out = [str(s).strip().upper() for s in v]
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s) for s in out):
            raise ValueError(f"default_symbols must be ticker symbols, got {v}")
        return list(dict.fromkeys(out))

    @field_validator("macro_event_types")
    @classmethod
    def _valid_macro_event_types(cls, v: list[str]) -> list[str]:
        allowed = {"FOMC", "CPI", "NFP", "GDP", "PCE"}
        bad = sorted(set(v) - allowed)
        if bad or not v:
            raise ValueError(
                f"macro_event_types must be a non-empty subset of {sorted(allowed)}; got {v}")
        return list(dict.fromkeys(v))

    position_size_pct: float | None = Field(default=None, ge=0.01, le=0.25,
        description="Fraction of capital risked per trade (max-loss basis). W64: None "
                    "(the default) resolves to risk.max_position_risk_pct — the SAME "
                    "per-trade budget live funds — so research and live cannot drift. "
                    "Set a number only for a deliberate sizing study. W86: ceiling 0.50 -> 0.25 = "
                    "risk.max_position_risk_pct's ceiling, so a YAML value cannot size what live's "
                    "schema forbids (an explicit constructor argument is not bounded by this).")
    history_start_date: date | None = Field(default=None,
        description="W65: first date of the research history. When set it wins over "
                    "history_days: the research loader trims frames to it and walk-forward "
                    "windows start there. config.yaml pins 2024-01-01 = the first day of the "
                    "operator's economic-calendar CSV, so the macro gate is ON in every window.")
    history_days: int = Field(default=1826, ge=400, le=6000,
        description="W65: calendar days of history a research run requests when no "
                    "history_start_date is set (legacy fallback). The walk-forward used to "
                    "fetch only train+test+100 days (528) = 2 windows.")
    research_table_prefix: str = Field(default="research_", pattern=r"^([a-z][a-z0-9_]*_)?$",
        description="W65: table prefix of the IB-sourced research history in historical.db "
                    "(research_daily_prices, research_daily_iv, research_intraday_prices, "
                    "research_econ_events). Empty = the live tables. Keeps research data in one "
                    "consistent source and the live tables untouched.")
    default_symbols: list[str] = Field(default=["QQQ"], min_length=1, max_length=20,
        description="W87a: the symbols a research entry point (run_backtest.py, run_optimizer.py) runs when "
                    "--symbols is not given. QQQ because it is the one symbol with the historical option "
                    "prices an IV surface can be built from, and the primary instrument of the programme "
                    "(it was a hard-coded ten-name list in run_backtest.py that crashed on the first name "
                    "without dividend rows). Research only: live's universe is trading.universe, which "
                    "research never reads.")
    vlmc_min_coverage: float = Field(default=0.95, ge=0.5, le=1.0,
        description="W65: minimum fraction of trading days that must have 5-min bars before the "
                    "walk-forward trains with VLMC/intraday-fractal features. Below it the run "
                    "falls back to daily-only features with a WARNING instead of mixing windows "
                    "with and without VLMC inside one run.")
    macro_event_types: list[str] = Field(default=["FOMC", "CPI", "NFP", "PCE", "GDP"],
        description="W65: which release types the RESEARCH macro-event gate blocks around "
                    "(default = the five the live gate uses). Subset of FOMC, CPI, NFP, GDP, PCE.")
    condor_any_direction: bool = Field(default=False,
        description="W66: research selects an iron-condor-only strategy set on ANY market "
                    "direction, like live (live's IronCondor ignores market_direction; the "
                    "engine only selected it on NEUTRAL days, ~81% of days removed on the "
                    "2024-2026 span). False keeps the legacy NEUTRAL-only rule (default, so "
                    "existing results do not move); a candidate for the operator to flip.")
    friction_model: str = Field(default="per_leg_usd", pattern="^(legacy|per_leg_usd)$",
        description="W66: research option-friction model for iron condors. per_leg_usd = the "
                    "fitted per-leg DOLLAR half-spread (option_leg_spread_params, from IB BID/ASK "
                    "bars) paid on all four legs at entry AND exit; legacy = half_spread_pct x "
                    "leg mid at entry and the same % on the net value at exit, which overcharged "
                    "a $0.95-credit QQQ condor $0.669 vs ~$0.085 measured (W65). A symbol with "
                    "no fitted model falls back to legacy with a WARNING.")
    friction_entry_capture: float = Field(default=1.0, ge=0.0, le=2.0,
        description="W66: share of the quoted half-spread paid per leg at ENTRY (1.0 = cross "
                    "the full half-spread, conservative; real limit fills at/inside the mid "
                    "capture less). Tune downward only from the fill-vs-model report.")
    friction_exit_capture: float = Field(default=1.0, ge=0.0, le=2.0,
        description="W66: share of the quoted half-spread paid per leg at EXIT (same "
                    "convention as friction_entry_capture).")
    friction_stress_multiplier: float = Field(default=1.0, ge=0.0, le=5.0,
        description="W66: multiplies the fitted model. The sample is calm-regime (IB serves "
                    "no expired-option history); raise it to stress-test, or sweep it with "
                    "scripts/friction_sensitivity.py.")
    budget_aware_wings: bool = Field(default=True,
        description="W64: the walk-forward/optimizer condor builder caps the wing to "
                    "the per-trade budget exactly like live (IronCondor.affordable_"
                    "width, $1 floor when capped) and rejects a structure whose max "
                    "loss exceeds the budget. False restores the old uncapped "
                    "wing_k x expected-move wing for before/after comparisons only.")
    wing_floor_dollars: float = Field(default=2.0, ge=1.0, le=10.0,
        description="W70: THE minimum iron-condor wing, in dollars, for LIVE ENTRIES AND "
                    "RESEARCH alike. This key lives under `backtest:` but it is also the "
                    "env-contract home of AIT_IC_WING_FLOOR, which the live builder "
                    "(strategies/iron_condor.py _vol_scaled_width) reads — so editing it "
                    "changes what the live bot trades, not only what a backtest simulates. "
                    "The wing is wing_k x price x IV x sqrt(DTE/365), but never narrower "
                    "than this. It binds only when the per-trade budget does not: a "
                    "budget-capped wing may shrink below it (at today's $5k account the "
                    "budget carries ~$2.78, so the floor matters only when the vol-scaled "
                    "width comes out under $2), but never below $1 "
                    "(invariants.IC_BUDGET_CAPPED_MIN_WING). Precedence for live: "
                    "env/.env AIT_IC_WING_FLOOR > this key > 2.0. Bounds 1.0-10.0: under "
                    "$1 contradicts the budget-capped minimum; over $10 is not sensible "
                    "for this account. History: default was 5.0 (the old backtest "
                    "default, still read by ~15 study scripts and run_optimizer); "
                    "walk-forward, run_backtest and the engine already used 2.0 "
                    "(R6 parity with the live literal this key replaces).")
    wing_k: float = Field(default=1.6, ge=0.1, le=3.0,
        description="Vol-scaled wing multiplier: wing_width = wing_k × price × IV × sqrt(DTE/365). "
                    "Optuna optimizes this per walk-forward window. "
                    "1.0 = 1-sigma expected move. wing_floor_dollars is the hard minimum. "
                    "R19: default raised 1.0 -> 1.6 to match the LIVE promoted value "
                    "(runtime_env.CONTRACT_DEFAULTS, 2026-08-04 SHADOW R3). This was the "
                    "third fork of wing_k: any module constructing BacktestConfig() bare "
                    "researched 1.0-wing structures while live traded 1.6, so backtests "
                    "did not describe the live book. Unlike the risk knobs below, a "
                    "divergent wing_k is not 'safer' in either direction — it is simply a "
                    "different strategy, so it must track the live value.")
    ic_min_credit_width: float = Field(default=0.10, ge=0.01, le=0.50,
        description="R19c: credit/width ratio floor for condor entries — the "
                    "config.yaml home for AIT_IC_MIN_CREDIT_WIDTH (contract "
                    "default 0.10 since the 2026-08-04 wide-wing promotion). "
                    "Precedence: env > this > CONTRACT_DEFAULTS.")
    ic_iv_rank_floor: float = Field(default=15.0, ge=0.0, le=100.0,
        description="W36: config home for AIT_IRON_CONDOR_IV_FLOOR — the "
                    "condor builder refuses below this IV rank. It did so "
                    "SILENTLY: on 2026-09-18 it refused every condor all "
                    "session (IWM 3.6, QQQ 12.3, SPY 12.7) with no log line.")
    ic_delta_min: float = Field(default=0.15, ge=0.01, le=0.50,
        description="W36: config home for AIT_IC_DELTA_MIN (R12-B delta band).")
    ic_delta_max: float = Field(default=0.30, ge=0.05, le=0.60,
        description="W36: config home for AIT_IC_DELTA_MAX (R12-B delta band).")
    ic_em_min: float = Field(default=0.6, ge=0.0, le=3.0,
        description="W36: config home for AIT_IC_EM_MIN — each short strike "
                    "must sit at least this many expected moves from spot.")
    ic_em_max: float = Field(default=1.3, ge=0.1, le=5.0,
        description="W36: config home for AIT_IC_EM_MAX.")
    ic_min_credit: float = Field(default=0.70, ge=0.10, le=5.0,
        description="R19c: absolute minimum total credit ($/share) for a "
                    "condor — config.yaml home for AIT_IC_MIN_CREDIT. At a 50% "
                    "TP the gross must clear ~3x round-trip costs.")
    ic_credit_ratio_at_20d: float = Field(default=0.45, ge=0.05, le=1.0,
        description="W31: config home for AIT_IC_CREDIT_RATIO_20D — the "
                    "practitioner credit/width ratio at ~0.20 delta, used by "
                    "the sizing self-test to advise a per-trade budget. Was a "
                    "bare `credit_ratio_at_20d = 0.45` at orchestrator.py, so "
                    "the $140 funding figure that drove the simulated-capital "
                    "decision derived from a literal the operator could not "
                    "see or change (R19c violation, R26 F4). The default IS "
                    "the literal it replaces. R26 measured the REALIZED ratio "
                    "at the designed 0.20-delta shorts at 0.375-0.386; moving "
                    "this value changes the advised budget, so a retune is "
                    "registered as its own wave rather than folded in here.")
    credit_loss_limit: float = Field(default=0.0, ge=0.0, le=5.0,
        description="R19c: flat credit-structure stop as a multiple of credit "
                    "received; 0 = DISABLED (R6/R12-B1 evidence: every flat "
                    "level underperformed touch-close). Config home for "
                    "AIT_CREDIT_LOSS_LIMIT.")
    iv_floor: float = Field(default=0.20, ge=0.05, le=1.0,
        description="Minimum synthetic IV used for option pricing. "
                    "Prevents near-zero credits in calm markets.")
    delta_iv_scale: float = Field(default=0.0, ge=0.0, le=1.0,
        description="IV-driven delta scaling for strangles. "
                    "0=static delta, 1=full IV response. "
                    "High IV → lower effective delta → further OTM strikes.")
    max_concurrent_positions: int = Field(default=1, ge=1, le=5,
        description="Maximum number of simultaneously open positions. "
                    "1 = original single-position behavior. "
                    "3 = up to 3 concurrent iron condors / strangles.")
    max_entry_vol_annual: float = Field(default=0.80, ge=0.15, le=1.50,
        description="Maximum 10-day realized vol (annualized) allowed for iron condor / "
                    "short strangle entry. Entries above this are skipped. "
                    "Optuna tunes per window in range [0.25, 0.90].")
    optimize_n_trials: int = Field(default=50, ge=5, le=500,
        description="Optuna trials per walk-forward window.")
    optimize_patience: int = Field(default=20, ge=0, le=500,
        description="Early stopping: halt after this many consecutive non-improving trials. "
                    "0 = disabled.")
    optimize_min_trades: int = Field(default=10, ge=1, le=100,
        description="Min trade count for full objective score. "
                    "Trials below this are penalised quadratically; < 3 trades always scores −100.")
    optimize_hard_min_trades: int = Field(default=3, ge=1, le=50,
        description="W67: below this many trades a trial is scored on the graded floor "
                    "(optimize_below_floor_score + optimize_below_floor_gradient x n/hard_min), not the old flat -100 "
                    "that left TPE nothing to climb (28 of 30 W65 studies never left it).")
    optimize_below_floor_gradient: float = Field(default=20.0, ge=0.0, le=90.0,
        description="W67: points the below-floor score rises from -100 (0 trades) toward "
                    "optimize_below_floor_score + this value (hard_min_trades - 1 trades), so "
                    "trading beats not trading. 0 restores a flat score.")
    optimize_below_floor_score: float = Field(default=-100.0, ge=-1000.0, le=-1.0,
        description="W95: the score of a trial with ZERO trades, the bottom of the graded below-floor ramp (the base that "
                    "optimize_below_floor_gradient rises from). It must stay below what any real trial can score, so every "
                    "starved trial ranks last: -100 is far below any Sharpe / win-rate / profit-factor objective. Change "
                    "it only if you add an objective whose scores can be that low or lower. The base plus the gradient "
                    "must stay negative.")

    @model_validator(mode="after")
    def _below_floor_ramp_stays_negative(self) -> "BacktestConfig":
        if self.optimize_below_floor_score + self.optimize_below_floor_gradient >= 0:
            raise ValueError(
                f"backtest: optimize_below_floor_score ({self.optimize_below_floor_score}) + optimize_below_floor_gradient "
                f"({self.optimize_below_floor_gradient}) must stay below 0, or a trial with too few trades could outrank "
                f"a real one")
        return self
    gates_mode: str = Field(default="observe", pattern="^(observe|enforce)$",
        description="W67: research ML-gate mode. observe = live parity today "
                    "(ml.entry_gates_enabled=false: the range-model floor and fractal veto are "
                    "computed and recorded as would_reject:* but never block); enforce = the "
                    "research engine's historical behaviour (they block). The gate study "
                    "(scripts/gate_study.py) compares the two on the same days.")
    iv_rank_gate: bool = Field(default=False,
        description="W68: research applies live's two IV-RANK gates to credit entries -- iron_condor "
                    "refuses rank < ic_iv_rank_floor (15) and the risk manager refuses rank > "
                    "risk.iv_rank_cap (85); an unmeasurable rank fails closed. Rank = percentile of "
                    "today's IV in the last 252 stored daily IVs (ait.data.iv_rank, shared with "
                    "live). False = the engine's historical absolute-level floor (iv_floor) only. "
                    "When True, lower iv_floor to its 0.05 minimum so it stays a pricing sanity "
                    "floor and not a second entry gate live does not have.")
    fractal_veto_spread: float = Field(default=0.0, ge=0.0, le=1.0,
        description="W67: absolute hurst_scale_spread above which neutral credit structures "
                    "are vetoed (0 = veto off, live parity). Replaces the research-only "
                    "max(hurst_regime_threshold, 0.20) x multiplier that the OOS run, the "
                    "optimizer trials and the meta-label shadow run applied inconsistently. "
                    "Shared with the live orchestrator via ait.ml.regime_gate.")
    penalty_on_range_prob: float = Field(default=0.0, ge=0.0, le=0.5,
        description="W67: P(in range) is lowered by this x min(spread/threshold, 3) when the "
                    "fractal regime is chaotic (0 = untouched, live parity). Condors are "
                    "gated on the range probability, not on direction, so this is the "
                    "penalty that can actually reach their entry statistic.")
    # Intraday execution params (Fix 1 / Gap H)
    scan_interval_minutes: int = Field(default=60, ge=5, le=240,
        description="How often (minutes) to scan for entry signals within a trading session.")
    entry_window_start_et: str = Field(default="10:30",
        description="Earliest allowed entry time (ET). Signals before this are ignored.")
    entry_window_end_et: str = Field(default="15:30",
        description="Latest allowed entry time (ET). No new entries after this time.")
    limit_order_timeout_bars: int = Field(default=3, ge=1, le=20,
        description="Cancel a pending limit order after this many 5-min bars without a fill.")
    # Options spread model params (Fix 5)
    spread_base: float = Field(default=0.03, ge=0.005, le=0.20,
        description="Base per-leg half-spread cost ($). Represents minimum friction in liquid markets.")
    spread_iv_sensitivity: float = Field(default=0.10, ge=0.0, le=0.50,
        description="Additional half-spread per unit of IV above 0.20. Higher IV → wider market.")
    spread_dte_sensitivity: float = Field(default=0.005, ge=0.0, le=0.05,
        description="Additional half-spread per DTE below 21. Near-expiry options are wider.")
    spread_cap: float = Field(default=0.15, ge=0.01, le=0.50,
        description="Maximum per-leg half-spread ($). Prevents unrealistic spread in stress regimes.")
    # W5 research-honesty block (blindspot_composition_hunt_20260825.md).
    # Three CONFIRMED research_validity findings whose combined size is the
    # whole per-condor expectancy — the researched edge may be sign-flipped,
    # so these values are the measured ones, not the convenient ones.
    commission_per_contract: float = Field(default=0.65, ge=0.0, le=10.0,
        description="Finding model-vs-reality-commission-constant: IBKR base "
                    "options commission ($/contract, per leg, charged on entry "
                    "AND exit). This is ONLY the commission line — the "
                    "regulatory/exchange/clearing fees that ride with every "
                    "fill live in regulatory_fees_per_contract below. The "
                    "engine used to charge this 0.65 as the ENTIRE per-leg "
                    "cost, which is what the finding measured as a ~41% "
                    "understatement of real friction.")
    regulatory_fees_per_contract: float = Field(default=0.2668, ge=0.0, le=5.0,
        description="Finding model-vs-reality-commission-constant: the "
                    "regulatory/exchange/clearing fees IBKR adds to every "
                    "option fill, on top of commission_per_contract, per leg, "
                    "entry AND exit. Default is the finding's OWN measurement "
                    "on data/ait_state.db executions (78 leg fills, BAG "
                    "summary rows excluded): all-in per contract-leg mean "
                    "0.9168 (median 1.0284, min 0.6195, max 1.0586); "
                    "0.9168 - 0.65 = 0.2668. It reproduces the finding's "
                    "headline gap exactly: a 4-leg condor round trip costs "
                    "(0.65 + 0.2668) x 4 x 2 = $7.33/contract against the old "
                    "flat-0.65 model's $5.20 = the measured $2.13/contract "
                    "understatement. Set to 0.0 ONLY to reproduce a pre-W5 "
                    "study; live P&L is trued up to the real ledger "
                    "(orchestrator total_commission), so 0.0 means research "
                    "and live disagree by this amount on every leg.")
    skew_slope_per_pct_otm: float = Field(default=0.0736, ge=0.0, le=1.0,
        description="Finding model-vs-reality-skew-10x-flat: RELATIVE IV "
                    "uplift per 1 percentage point of OTM distance for the "
                    "put side, i.e. leg_iv gains base_iv x this x "
                    "(100 x |ln(K/S)|) on top of the legacy hardcoded skew. "
                    "The engine's hardcoded put slope was 0.10 absolute IV "
                    "per unit |log-moneyness|; the finding's regression on "
                    "the repo's own 3,159 IBKR quotes "
                    "(data/historical.db option_spread_samples, 2026-08-11) "
                    "measured 1.141 SPY dte20 (n=140), 0.983 QQQ dte20 "
                    "(n=81), 0.916 IWM dte24 (n=57) — 9-11x steeper — with a "
                    "sample-weighted mean of 1.049. Default 0.0736 makes the "
                    "TOTAL put slope hit that 1.049 at the finding's own "
                    "probe anchor (SPY atm_iv 0.1289, skew_factor 1.0): "
                    "0.10 + 0.1289 x 100 x 0.0736 = 1.049. Consequence of "
                    "the old value: wings were near-free in research (long "
                    "put $0.08 model vs $0.71-0.92 real chain mid), so every "
                    "wing_k/wide-wing study preferred wider wings than "
                    "reality prices. The CALL side deliberately keeps its "
                    "legacy 0.02 term: the same regression measured "
                    "0.064/0.112/-0.089 there — sign-unstable across symbols "
                    "and therefore not a calibrated slope. 0.0 restores "
                    "pre-W5 pricing exactly (backward-compat escape hatch "
                    "for reproducing an old study, NOT an honest surface).")
    # Fractal regime params (Gap Z5) — also used by live orchestrator for parity with backtest
    hurst_regime_threshold: float = Field(default=0.20, ge=0.05, le=0.50,
        description="Hurst scale-spread above which fractal confidence penalty is applied. "
                    "R20b: also the optimizer's trial-baseline (was a 0.20 literal in bt_kwargs).")
    hurst_regime_penalty: float = Field(default=0.10, ge=0.0, le=0.30,
        description="Confidence deducted when fractal regime is chaotic. "
                    "R20b: also the optimizer's trial-baseline (was a 0.10 literal in bt_kwargs).")
    multifractal_max_width: float = Field(default=0.50, ge=0.20, le=0.80,
        description="Multifractal width above which fractal confidence penalty is applied. "
                    "R20b: also the optimizer's trial-baseline (was a 0.50 literal in "
                    "bt_kwargs; the pre-registration named it multifractal_width_threshold "
                    "— this EXISTING field is that knob, no duplicate was added).")
    # R20b (pre-registered PLAN 2026-08-21): config homes for the optimizer's
    # remaining non-searched engine baselines — they were frozen literals in
    # StrategyOptimizer._run_backtest's bt_kwargs, so a config change could
    # never reach a trial backtest. Defaults = the 2026-08-21 operating
    # literals, and config.yaml declares the same values (no divergence).
    stop_loss_pct: float = Field(default=0.35, ge=0.05, le=1.0,
        description="R20b: baseline stop-loss as a fraction of position value for "
                    "trial/engine backtests (options decay fast — cut at 35%). "
                    "Optuna may search per window; this is the non-searched baseline.")
    profit_target_pct: float = Field(default=0.50, ge=0.05, le=3.0,
        description="R20b: baseline take-profit as a fraction of position value for "
                    "trial/engine backtests (take profits at 50%). Optuna may search "
                    "per window; this is the non-searched baseline.")
    max_hold_days: int = Field(default=30, ge=1, le=120,
        description="R20b: baseline maximum holding period (calendar days) before a "
                    "trial/engine backtest force-closes a position.")
    engine_model: Literal["legacy", "realistic"] = Field(default="legacy",
        description="W74: which fidelity the research engine runs at. 'legacy' = the "
                    "engine every prior study used (entry priced at the DAILY close with "
                    "same-day IV, expiry = entry + N calendar days even on a weekend, "
                    "whole-day time, no same-session exit check, intraday take-profit "
                    "decided AFTER friction and blind to touches, condors invalidated on "
                    "direction, long strikes priced unrounded then re-marked rounded). "
                    "'realistic' fixes each (W72 leads L2-L6): fill-time spot, "
                    "previous-session IV, listed session expiries, fractional time to the "
                    "bell, same-session exit checks, mid-based intraday decisions, per-bar "
                    "touch stops, live's condor thesis exemption, rounded strikes priced "
                    "as stored. Flipping it changes every research baseline (assumption "
                    "A10): record a before/after on the reference study first. "
                    "Operator-owned; default 'legacy' so no study moves silently.")
    friction_short_dte: bool = Field(default=False,
        description="W76: charge 0-6 DTE legs from DTE-band friction instead of the single "
                    "all-DTE per-leg model (friction_model='per_leg_usd' only). The fitted model "
                    "has no sample below 11 DTE, so short-DTE legs were charged an extrapolated "
                    "number. When True a band with fewer than friction_min_sessions_per_band "
                    "collected sessions is charged the all-DTE model floored at "
                    "friction_extrapolated_floor_usd per leg and the run reports how many legs "
                    "were extrapolated; a band with enough sessions uses its own fit. Also "
                    "applies to the EXIT of a long-dated position once its remaining DTE is "
                    "below 7. Default False: no study moves silently.")
    friction_min_sessions_per_band: int = Field(default=15, ge=1, le=250,
        description="W76: sessions of real IB bid/ask bars a 0-6 DTE band needs before its own "
                    "fit replaces the conservative extrapolation. 15 follows the W66 fit's "
                    "grouped leave-one-session-out CV, which needs a few sessions per fold.")
    friction_extrapolated_floor_usd: float = Field(default=0.01, ge=0.005, le=0.10,
        description="W76: per-leg HALF-spread floor ($/share) used where a short-DTE band has no "
                    "fitted model. $0.01 is one full option tick paid on each side, a "
                    "conservative stand-in for QQQ/SPY/IWM at 0-6 DTE (quoted $0.01-0.03 wide). "
                    "Operator-owned: raise it for a harsher study, lower it once real bars exist.")
    pricing_source: Literal["bs_legacy", "surface"] = Field(default="bs_legacy",
        description="W75: where the research engine gets each condor leg's implied vol. "
                    "'bs_legacy' = one stored 30-day IV plus a skew with no DTE term (every "
                    "prior study). 'surface' = the smile of the leg's OWN expiry from the "
                    "real option bars (ait.backtesting.iv_surface), fit on the previous "
                    "session so nothing from the decision day leaks. Out-of-sample, bar to "
                    "bar, it cut median option-price error by 19% (1-2 DTE), 46% (3-7), 43% "
                    "(8-14) and 49% (15-45) and removed the +1.4-1.9 vol-point richness "
                    "'bs_legacy' carries at 3+ DTE. QQQ only (the bars exist for QQQ); a "
                    "symbol or date without a smile falls back per leg and is counted. "
                    "Operator-owned: default 'bs_legacy' so no study moves silently.")
    expiry_guard_dte: int | None = Field(default=None, ge=0, le=45,
        description="W74: positions ENTERED with at most this many days to expiry are "
                    "exempt from the DTE-based closes (live rules 3/3a: DTE <= "
                    "EXPIRY_APPROACHING_DTE) and are governed by max_hold_fraction and "
                    "the expiry-day flatten instead. None = today's behaviour: the rule "
                    "closes every position, which makes any 0-5 DTE entry exit on the "
                    "next bar. Searched 0-5 by the DTE study (W79).")
    max_hold_fraction: float | None = Field(default=None, gt=0.0, le=1.0,
        description="W74: close a credit position after this share of its life (entry "
                    "-> expiry close). None = the legacy day cap (max_hold_days). "
                    "Replaces max_hold_days for credit structures when set, so 'hold "
                    "anywhere between 0 and DTE' is one dimension for every entry DTE.")
    flatten_minutes_before_close: int | None = Field(default=None, ge=5, le=390,
        description="W74: on a position's expiry date, flatten this many minutes before "
                    "the session close (13:00 ET on early-close days). None = off. A hard "
                    "backstop so no position reaches settlement, assignment or an "
                    "unhedged gap after a one-sided expiry (the exposure that cost two "
                    "manual trades more than their defined maximum).")
    learner: LearnerConfig = Field(default_factory=LearnerConfig,
        description="W88b: thresholds of the research BacktestLearner (listed in full in the research overlay file).")
    legacy_engine: LegacyEngineConfig = Field(default_factory=LegacyEngineConfig,
        description="W88b: constants of the engine's legacy / dormant paths (listed in full in the research overlay file).")
    objectives: ObjectivesConfig = Field(default_factory=ObjectivesConfig,
        description="W88b: constants of the Optuna objectives (composite weights, profit-factor cap).")
    surface_tuning: SurfaceTuningConfig = Field(default_factory=SurfaceTuningConfig,
        description="W90f: acceptance rules of the research IV surface (listed in full in the research overlay file).")
    friction_fit: FrictionFitConfig = Field(default_factory=FrictionFitConfig,
        description="W90f: minimum data for fitting a leg-spread model (listed in full in the research overlay file).")
    spread_tuning: SpreadTuningConfig = Field(default_factory=SpreadTuningConfig,
        description="W90f: sizing numbers of the non-condor single-structure builders (listed in full in the research overlay file).")
    engine_tuning: EngineTuningConfig = Field(default_factory=EngineTuningConfig,
        description="W90c: IV-skew coefficients, gate look-backs and the minimum history of the research engine "
                    "(listed in full in the research overlay file).")
    walkforward_tuning: WalkForwardTuningConfig = Field(default_factory=WalkForwardTuningConfig,
        description="W88b: walk-forward / optimizer constants (warm start, adaptive range threshold, validation "
                    "split, inert-parameter probe).")
    dte_search: DteSearchConfig = Field(default_factory=DteSearchConfig,
        description="W79a: opt-in DTE dimensions for an Optuna study (see DteSearchConfig). "
                    "Research only; the live bot never reads it.")
    pin_band_usd: float = Field(default=0.10, ge=0.0, le=1.0,
        description="W77: a SHORT leg within this many dollars of its strike at the expiry "
                    "bell is a pin: whether it is assigned cannot be known at the close, so "
                    "the engine scores both outcomes and books the worse (assumption A5 of "
                    "the short-DTE programme). Realistic engine only; a position only "
                    "reaches the bell when it is exempt from the expiry guard and not "
                    "flattened. Operator-owned: 0.10 is a judgement, not a measurement "
                    "(none of the three real expiries on the 2026-08/10 statement was "
                    "within it).")
    assignment_fee_per_share: float = Field(default=0.015, ge=0.0, le=0.10,
        description="W77: $/share charged on stock delivered by ASSIGNMENT at expiry. "
                    "Measured on the operator's 2026-09/10 statement (0.0151-0.0155 on the "
                    "three assignments); exercise-delivered shares carried no fee there, "
                    "so none is charged on them. Realistic engine only.")
    stock_commission_per_share: float = Field(default=0.01, ge=0.0, le=0.10,
        description="W77: $/share to close shares left over after a one-sided expiry at "
                    "the next session's open. Measured 0.0100 on the same statement. "
                    "Realistic engine only.")
    iv_rank_rise_gate: bool = Field(default=True,
        description="W71c: apply the IV-rank RISE veto at all. RESEARCH-ONLY: live has no such "
                    "veto (it gates on the stored-IV percentile LEVEL, condor floor 15 / credit "
                    "cap 85). False skips the veto but the rise is still computed and recorded "
                    "per entry (iv_rank_rise_10d), and the optimizer stops searching "
                    "iv_rank_rise_threshold (a searched value would otherwise re-enable the "
                    "gate per trial). Schema default True so a bare Backtester and every "
                    "regression baseline are unchanged; the research overlay sets False "
                    "(operator decision D9, PLAN.md W71) together with the bar for turning it "
                    "back on.")
    iv_rank_rise_threshold: float = Field(default=0.30, ge=0.0, le=10.0,
        description="R20b: suppress iron-condor entry when IV rank rose more than "
                    "this over the last 10 days (Exp 20 veto). Values > 1 disable "
                    "the veto (IV rank is 0-1). Ignored while iv_rank_rise_gate is false.")
    regime_veto_gate: bool = Field(default=True,
        description="W81: apply the trending-down REGIME veto at all. RESEARCH-ONLY: live has "
                    "no such entry veto (it uses regime for the direction override, the "
                    "meta-label features and roll defense only). The veto blocks a neutral "
                    "credit entry when 5-day realized vol exceeds 20-day (vol_regime_expanding) "
                    "and the close is more than regime_down_threshold below its 20-day SMA. "
                    "False skips the VETO only: the regime class is still computed and "
                    "recorded per entry (regime_class, regime_veto_suppressed). Schema default "
                    "True so a bare Backtester and every regression baseline are unchanged; "
                    "whether the research overlay turns it off is operator decision D10 "
                    "(PLAN.md, 2026-10-13 programme entry), to be taken on the W81 A/B.")
    regime_down_threshold: float = Field(default=-0.05, ge=-0.50, le=0.0,
        description="W81: price_vs_sma_20 below this (with vol_regime_expanding) classes a day "
                    "'trending_down' in the research engine: the regime veto, the "
                    "entry_regime stored on each position and the meta-label regime features "
                    "all read it. Was a bare -0.05 at three engine sites (Exp 22/23: -0.02 "
                    "blocked profitable 2-4% corrections; -0.05 cleared the Yen-carry and "
                    "tariff-shock episodes). Research only; live's RegimeDetector has its own "
                    "thresholds (the regime: block).")
    regime_up_threshold: float = Field(default=0.02, ge=0.0, le=0.50,
        description="W81: price_vs_sma_20 above this (with vol_regime_expanding) classes a day "
                    "'trending_up' in the research engine (position entry_regime and the "
                    "meta-label regime features; it never vetoes). Was a bare 0.02. Research "
                    "only.")
    ex_div_entry_gate: Literal["off", "live", "exact", "last_k_days"] = Field(default="off",
        description="W84: the research TWIN of live's ex-dividend entry block (W51, orchestrator "
                    "trade_blocked_ex_div): refuse a new position that SELLS A CALL when an "
                    "ex-dividend date falls between the entry day and the position's expiry "
                    "(early-assignment risk on the short call). 'live' = live's rule exactly: "
                    "ONE projected ex-date (dividends.project_next_ex_div over the ex-dates "
                    "known before the entry day). 'exact' = any real ex-date in [entry, "
                    "expiry]. 'last_k_days' = only a real ex-date within ex_div_entry_k_days "
                    "of expiry (where little extrinsic is left; the F2 candidate). 'off' = no "
                    "block; a day it would have blocked is still flagged ex_div_in_window when "
                    "the dividend table is loaded. Needs the {research_table_prefix}dividends "
                    "table (scripts/backfill_dividends.py); any mode but 'off' raises without "
                    "it. Schema default 'off' so a bare Backtester and every regression "
                    "baseline are unchanged; the research overlay sets 'live' (parity; PLAN.md "
                    "2026-10-16 ex-dividend finding: live refuses 14-17% of sessions).")
    ex_div_entry_k_days: int = Field(default=5, ge=1, le=30,
        description="W84: K for ex_div_entry_gate 'last_k_days': block only when a real "
                    "ex-date is at most K calendar days before the position's expiry. Not "
                    "searched by the optimizer. Ignored in the other modes.")
    min_edge_over_baseline: float = Field(default=0.05, ge=0.0, le=1.0,
        description="R20b: minimum weighted CV edge over the base rate for the "
                    "range predictor to activate as an entry gate (Exp 28 quality "
                    "floor; 0.0 = always use the model).")
    slippage_pct: float = Field(default=0.01, ge=0.0, le=0.10,
        description="W70: fractional price slippage the research engine adds on top of "
                    "the modeled spread for DEBIT / long-option structures (long call/put, "
                    "straddles, debit spreads). Research only. NOT applied to iron "
                    "condors (credit structures use the friction model) so it has no "
                    "effect on the IC-only strategy set. Was the engine's "
                    "slippage_pct=0.01 constructor default.")
    delta_short: float = Field(default=0.20, ge=0.05, le=0.50,
        description="W70: |delta| the research engine targets for the SHORT strikes of "
                    "its structures. Research only: live's iron condor short delta is the "
                    "constant iron_condor.SHORT_DELTA_TARGET (0.20), NOT this key, so "
                    "changing it moves research away from live (the parity manifest shows both). Was the engine's "
                    "delta_short=0.20.")
    delta_long: float = Field(default=0.30, ge=0.05, le=0.60,
        description="W70: |delta| the research engine targets for LONG single options "
                    "and long legs of debit structures. Research only; unused by the "
                    "iron condor (its wings are set by wing_k, not delta). Was "
                    "delta_long=0.30.")
    hurst_hard_veto_multiplier: float = Field(default=1.5, ge=0.0, le=5.0,
        description="W70: legacy derivation of the fractal HARD veto when "
                    "fractal_veto_spread is unset: veto spread = max(hurst_regime_"
                    "threshold, 0.20) x this. 0 = no derived veto. Research only. NOTE "
                    "WalkForwardConfig deliberately passes 0.0 (disabled; Exp 20 "
                    "post-mortem: any multiplier blocked all QQQ entries), so this "
                    "default 1.5 only governs a bare Backtester(). Was the engine's "
                    "hurst_hard_veto_multiplier=1.5.")
    pct_from_60d_high_threshold: float = Field(default=-1.0, ge=-1.0, le=0.0,
        description="W70: research-only experimental entry gate: refuse a neutral "
                    "credit entry when spot is more than this fraction below its 60-day "
                    "high (e.g. -0.05). -1.0 = DISABLED (the default; values <= -0.99 "
                    "never gate). Not part of live. Was the engine's "
                    "pct_from_60d_high_threshold=-1.0.")
    skew_factor: float = Field(default=1.0, ge=0.0, le=5.0,
        description="W70: multiplier on the engine's legacy flat IV-skew term (put "
                    "0.10 / call 0.02 per unit log-moneyness). 0.0 = flat IV. Additive "
                    "with skew_slope_per_pct_otm, the W5 measured steepening. Research "
                    "only. Was the engine's skew_factor=1.0.")
    context_bars: int = Field(default=0, ge=0, le=1000,
        description="W70: extra intraday bars of lookback the research engine keeps as "
                    "feature context (lookback = max(20, this)). 0 = the engine "
                    "minimum. Research harness knob. Was context_bars=0.")


class TreeHyperparamsConfig(_StrictModel):
    """W88c: the gradient-boosted-tree hyperparameters a model family trains with (XGBoost and LightGBM alike).

    SHARED between live and research: ``ml.hyperparams`` drives the direction ensemble, the range model and the
    vol-magnitude model; ``ml.meta_label_hyperparams`` drives the meta-labeler (a different, shallower profile on
    purpose). The defaults are the values the code used as literals before W88c, so ``config.yaml`` need not
    change anything; research may differ via the overlay or a study, and the parity manifest shows it. Every saved
    model records the hyperparameters it was trained with. Not here on purpose: ``n_jobs`` / ``verbosity``
    (speed and noise, no effect on the model) and the objective / metric of each task (they define the task)."""

    n_estimators: int = Field(default=200, ge=10, le=5000, description="Number of boosted trees.")
    max_depth: int = Field(default=5, ge=1, le=20, description="Maximum tree depth.")
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0, description="Shrinkage per boosting step.")
    subsample: float = Field(default=0.8, gt=0.0, le=1.0, description="Row subsample ratio per tree.")
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0, description="Column subsample ratio per tree.")
    random_state: int | None = Field(default=42, ge=0, le=2**31 - 1,
        description="Model seed. null = unseeded (the meta-labeler's historical behaviour).")


def _ordered(label: str, **named: int) -> None:
    """Raise unless the named windows are strictly increasing in the order given."""
    vals = list(named.values())
    if any(a >= b for a, b in zip(vals, vals[1:])):
        raise ValueError(f"features.{label}: windows must be strictly increasing, got "
                         + ", ".join(f"{k}={v}" for k, v in named.items()))


class MomentumWindows(_StrictModel):
    """W88d: windows of the momentum features (RSI, MACD, rate of change). A window that appears in a column NAME
    (``rsi_14``, ``roc_5``) renames that column when changed (``rsi_10``)."""

    rsi_long: int = Field(default=14, ge=2, le=200, description="Long RSI period (column rsi_<n>).")
    rsi_short: int = Field(default=7, ge=2, le=200, description="Short RSI period (column rsi_<n>).")
    macd_fast: int = Field(default=12, ge=2, le=200, description="MACD fast EMA span (also the raw ema_<n> column).")
    macd_slow: int = Field(default=26, ge=3, le=400, description="MACD slow EMA span (also the raw ema_<n> column).")
    macd_signal: int = Field(default=9, ge=2, le=200, description="MACD signal EMA span.")
    roc_short: int = Field(default=5, ge=1, le=200, description="Shortest rate-of-change period (column roc_<n>).")
    roc_mid: int = Field(default=10, ge=2, le=200, description="Middle rate-of-change period (column roc_<n>).")
    roc_long: int = Field(default=20, ge=3, le=400, description="Longest rate-of-change period (column roc_<n>).")

    @model_validator(mode="after")
    def _check(self) -> "MomentumWindows":
        _ordered("momentum.rsi", rsi_short=self.rsi_short, rsi_long=self.rsi_long)
        _ordered("momentum.macd", macd_fast=self.macd_fast, macd_slow=self.macd_slow)
        _ordered("momentum.roc", roc_short=self.roc_short, roc_mid=self.roc_mid, roc_long=self.roc_long)
        return self


class VolatilityWindows(_StrictModel):
    """W88d: windows of the volatility features (ATR, Bollinger bands, realized volatility)."""

    atr_window: int = Field(default=14, ge=2, le=200, description="ATR window (raw column atr_<n>; atr_pct divides it by price).")
    bb_window: int = Field(default=20, ge=2, le=400, description="Bollinger band window.")
    bb_std: float = Field(default=2.0, gt=0.0, le=5.0, description="Bollinger band width in standard deviations.")
    rv_long: int = Field(default=20, ge=3, le=400, description="Long realized-volatility window (column realized_vol_<n>; also the vol-regime 'medium').")
    rv_short: int = Field(default=10, ge=2, le=400, description="Short realized-volatility window (column realized_vol_<n>).")

    @model_validator(mode="after")
    def _check(self) -> "VolatilityWindows":
        _ordered("volatility.rv", rv_short=self.rv_short, rv_long=self.rv_long)
        return self


class VolumeWindows(_StrictModel):
    """W88d: windows of the volume features."""

    volume_avg_window: int = Field(default=20, ge=2, le=400,
        description="Average-volume window (column volume_sma_<n>_ratio; the slow side of volume_trend; the weekly "
                    "volume-confirmation average).")
    volume_fast_window: int = Field(default=5, ge=1, le=200, description="Fast average-volume window of volume_trend.")
    obv_period: int = Field(default=5, ge=1, le=200, description="Period of the on-balance-volume change (obv_change).")

    @model_validator(mode="after")
    def _check(self) -> "VolumeWindows":
        _ordered("volume", volume_fast_window=self.volume_fast_window, volume_avg_window=self.volume_avg_window)
        return self


class TrendWindows(_StrictModel):
    """W88d: windows of the trend features (simple moving averages and their slopes)."""

    sma_fast: int = Field(default=10, ge=2, le=400, description="Fast SMA (columns sma_<n>, sma_<n>_slope, sma_<n>_<m>_cross).")
    sma_mid: int = Field(default=20, ge=3, le=400, description="Middle SMA (columns sma_<n>, price_vs_sma_<n>; the weekly daily-trend test).")
    sma_slow: int = Field(default=50, ge=4, le=400, description="Slow SMA (columns sma_<n>, price_vs_sma_<n>).")
    slope_period: int = Field(default=5, ge=1, le=100, description="Periods over which an SMA slope is measured.")

    @model_validator(mode="after")
    def _check(self) -> "TrendWindows":
        _ordered("trend.sma", sma_fast=self.sma_fast, sma_mid=self.sma_mid, sma_slow=self.sma_slow)
        return self


class MultiTimeframeWindows(_StrictModel):
    """W88d: windows of the simulated-weekly features."""

    weekly_close_window: int = Field(default=5, ge=2, le=60, description="Daily bars averaged into a simulated weekly close.")
    weekly_trend_window: int = Field(default=4, ge=2, le=60, description="Simulated weeks averaged for the weekly trend.")
    weekly_trend_shift: int = Field(default=5, ge=1, le=60, description="Bars back the weekly trend is compared with.")
    weekly_rsi_period: int = Field(default=14, ge=2, le=200, description="RSI period of weekly_rsi.")
    weekly_momentum_period: int = Field(default=25, ge=2, le=200, description="Rate-of-change period of weekly_momentum.")


class VolRegimeWindows(_StrictModel):
    """W88d: windows of the realized-volatility regime ('IV proxy') features."""

    fast: int = Field(default=5, ge=2, le=100, description="Fast realized-vol window of vol_regime_expanding.")
    slow: int = Field(default=60, ge=5, le=400, description="Slow realized-vol window (vol_ratio, the mean-reversion signal).")
    iv_rank_lookback: int = Field(default=252, ge=20, le=1000, description="Lookback of the iv_rank range (one trading year).")
    iv_rank_min_periods: int = Field(default=20, ge=2, le=500, description="Minimum periods before iv_rank is defined.")
    vol_trend_period: int = Field(default=5, ge=1, le=100, description="Periods of the vol_trend slope.")
    vol_of_vol_window: int = Field(default=20, ge=2, le=400, description="Window of vol_of_vol.")

    @model_validator(mode="after")
    def _check(self) -> "VolRegimeWindows":
        if self.iv_rank_min_periods > self.iv_rank_lookback:
            raise ValueError("features.vol_regime: iv_rank_min_periods cannot exceed iv_rank_lookback")
        return self


def _min_within(label: str, **pairs: tuple[int, int]) -> None:
    """W97: raise unless every ``min_periods`` is at most its window (name -> (min_periods, window))."""
    for name, (mn, win) in pairs.items():
        if mn > win:
            raise ValueError(f"features.{label}: {name} min_periods ({mn}) cannot exceed its window ({win})")


class MarketStructureWindows(_StrictModel):
    """W97: windows of the market-structure features (put/call proxy, vol term structure, skew proxy, momentum divergence,
    the long moving average, range compression). Of these only ``above_sma<n>`` / ``distance_sma<n>`` are model inputs and
    carry the long window in their name; the other columns stay in the frame under fixed names. Defaults = the pre-W97 literals."""

    put_call_window: int = Field(default=10, ge=2, le=200,
        description="W97: days the up-volume and down-volume are summed over for the put/call proxy.")
    put_call_min_periods: int = Field(default=5, ge=1, le=200,
        description="W97: ... and the fewest days it needs before it reports (the rest read neutral).")
    term_fast: int = Field(default=5, ge=2, le=200,
        description="W97: short realized-vol window of the vol term-structure proxy (short vol / long vol).")
    term_slow: int = Field(default=60, ge=3, le=500, description="W97: long realized-vol window of the same proxy.")
    skew_window: int = Field(default=20, ge=2, le=400,
        description="W97: window of the downside / upside volatility ratio (the skew proxy).")
    skew_min_periods: int = Field(default=10, ge=1, le=400, description="W97: fewest days the skew proxy needs.")
    divergence_short: int = Field(default=5, ge=1, le=200,
        description="W97: short return horizon of the momentum divergence (short return minus long return).")
    divergence_long: int = Field(default=60, ge=2, le=500, description="W97: long return horizon of the same measure.")
    long_sma_window: int = Field(default=200, ge=20, le=1000,
        description="W97: the long moving average behind above_sma<n> and distance_sma<n> (the column names carry this "
                    "number). Also the history needed before they read anything but their neutral value.")
    long_sma_min_periods: int = Field(default=100, ge=2, le=1000,
        description="W97: fewest days the long moving average needs before it is defined.")
    range_fast: int = Field(default=5, ge=1, le=200,
        description="W97: short window of the range-compression ratio (short average range / long average range).")
    range_slow: int = Field(default=20, ge=2, le=400, description="W97: long window of the same ratio.")

    @model_validator(mode="after")
    def _check(self) -> "MarketStructureWindows":
        _ordered("market_structure.term", term_fast=self.term_fast, term_slow=self.term_slow)
        _ordered("market_structure.divergence", divergence_short=self.divergence_short, divergence_long=self.divergence_long)
        _ordered("market_structure.range", range_fast=self.range_fast, range_slow=self.range_slow)
        _min_within("market_structure", put_call=(self.put_call_min_periods, self.put_call_window),
                    skew=(self.skew_min_periods, self.skew_window), long_sma=(self.long_sma_min_periods, self.long_sma_window))
        return self


class CrossAssetWindows(_StrictModel):
    """W97: windows of the cross-asset features (VIX level changes and z-score, relative strength against SPY, SPY momentum
    and RSI, correlation with SPY). A window that appears in a column name renames the column when changed
    (``vix_change_5d`` -> ``vix_change_3d``). Defaults = the pre-W97 literals."""

    vix_change_short: int = Field(default=5, ge=1, le=100, description="W97: short VIX change horizon (column vix_change_<n>d).")
    vix_change_long: int = Field(default=20, ge=2, le=400, description="W97: long VIX change horizon (column vix_change_<n>d).")
    vix_z_window: int = Field(default=60, ge=5, le=500, description="W97: window of the VIX z-score (vix_zscore).")
    vix_z_min_periods: int = Field(default=20, ge=2, le=500, description="W97: fewest days the VIX z-score needs.")
    vix_term_fast: int = Field(default=5, ge=2, le=200,
        description="W97: fast VIX average of vix_term_spread (fast average minus slow average, over the slow one).")
    vix_term_slow: int = Field(default=20, ge=3, le=400, description="W97: slow VIX average of vix_term_spread.")
    rel_short: int = Field(default=5, ge=1, le=100, description="W97: shortest relative-strength horizon (column rel_strength_<n>d).")
    rel_mid: int = Field(default=20, ge=2, le=200, description="W97: middle relative-strength horizon (column rel_strength_<n>d).")
    rel_long: int = Field(default=60, ge=3, le=500, description="W97: longest relative-strength horizon (column rel_strength_<n>d).")
    spy_momentum_days: int = Field(default=10, ge=1, le=200, description="W97: SPY return horizon (column spy_momentum_<n>d).")
    spy_rsi_period: int = Field(default=14, ge=2, le=200, description="W97: SPY RSI period (column spy_rsi_<n>).")
    corr_window: int = Field(default=20, ge=3, le=400, description="W97: window of the rolling correlation with SPY (column correlation_spy_<n>d).")
    corr_min_periods: int = Field(default=10, ge=2, le=400, description="W97: fewest days that correlation needs.")
    min_context_rows: int = Field(default=20, ge=1, le=2000,
        description="W97: a VIX or SPY history needs MORE than this many rows to be used (otherwise its columns read neutral).")

    @model_validator(mode="after")
    def _check(self) -> "CrossAssetWindows":
        _ordered("cross_asset.vix_change", vix_change_short=self.vix_change_short, vix_change_long=self.vix_change_long)
        _ordered("cross_asset.vix_term", vix_term_fast=self.vix_term_fast, vix_term_slow=self.vix_term_slow)
        _ordered("cross_asset.rel", rel_short=self.rel_short, rel_mid=self.rel_mid, rel_long=self.rel_long)
        _min_within("cross_asset", vix_z=(self.vix_z_min_periods, self.vix_z_window), corr=(self.corr_min_periods, self.corr_window))
        return self


class MacroWindows(_StrictModel):
    """W97: windows of the macro features (FRED yield curve and the dollar index). Defaults = the pre-W97 literals."""

    change_days: int = Field(default=20, ge=1, le=400,
        description="W97: horizon of the yield-curve and 10-year-yield changes (columns yield_curve_change_<n>d, us_10y_change_<n>d).")
    dxy_change_short: int = Field(default=5, ge=1, le=200, description="W97: short dollar-index change horizon (column dxy_change_<n>d).")
    dxy_change_long: int = Field(default=20, ge=2, le=400, description="W97: long dollar-index change horizon (column dxy_change_<n>d).")
    dxy_center: float = Field(default=100.0, gt=0.0, le=1000.0,
        description="W97: the dollar-index level that reads 0 in dxy_level_norm = (dxy - center) / scale.")
    dxy_scale: float = Field(default=20.0, gt=0.0, le=1000.0,
        description="W97: points of the dollar index that make 1.0 in dxy_level_norm (clipped to +/-2).")

    @model_validator(mode="after")
    def _check(self) -> "MacroWindows":
        _ordered("macro.dxy_change", dxy_change_short=self.dxy_change_short, dxy_change_long=self.dxy_change_long)
        return self


class FractalWindows(_StrictModel):
    """W97: windows and minimum lengths of the daily fractal group (wavelet Hurst, power-spectrum slope, multi-scale Hurst,
    multifractal width). The column names do not embed a window. The wavelet family, the MFDFA scales and the clip bounds
    stay fixed algorithm constants. Defaults = the pre-W97 literals."""

    hurst_short_window: int = Field(default=60, ge=16, le=1000,
        description="W97: bars of the short Hurst estimate (hurst_short); with fewer bars than the long window only this one is reported.")
    hurst_long_window: int = Field(default=180, ge=32, le=2000, description="W97: bars of the long Hurst estimate (hurst_long).")
    hurst_min_len: int = Field(default=50, ge=16, le=1000,
        description="W97: the wavelet Hurst estimator returns its neutral 0.5 for a series shorter than this.")
    psd_min_rows: int = Field(default=60, ge=8, le=2000,
        description="W97: the power-spectrum slope returns its neutral value (beta 2.0) for a series shorter than this.")
    min_rows: int = Field(default=50, ge=20, le=1000,
        description="W97: the fractal group is all zeros below this many rows, and its expanding estimate starts at this row.")
    mfdfa_min_returns: int = Field(default=500, ge=100, le=20000,
        description="W97: the multifractal width and asymmetry need at least this many returns (else 0).")

    @model_validator(mode="after")
    def _check(self) -> "FractalWindows":
        _ordered("fractal.hurst", hurst_short_window=self.hurst_short_window, hurst_long_window=self.hurst_long_window)
        return self


class IntradayWindows(_StrictModel):
    """W97: bar counts of the intraday (VLMC) features computed from 5-minute bars. The 26 column names are a FIXED list and
    do not follow these numbers (``intraday_momentum_1h`` keeps its name and then means ``hour_bars`` bars). Defaults = the
    pre-W97 literals."""

    hour_bars: int = Field(default=12, ge=2, le=78,
        description="W97: bars that make 'an hour' (12 five-minute bars): the hourly momentum, the recent-range window and the power-hour window.")
    rsi_period: int = Field(default=7, ge=2, le=100, description="W97: RSI period of intraday_rsi.")
    atr_period: int = Field(default=14, ge=2, le=100, description="W97: ATR window of intraday_atr_pct.")
    closing_bars: int = Field(default=3, ge=2, le=20, description="W97: last bars of the session behind the closing-imbalance features.")
    min_bars: int = Field(default=20, ge=5, le=2000, description="W97: fewer intraday bars than this and no intraday feature is computed.")
    session_min_rows: int = Field(default=10, ge=1, le=78,
        description="W97: when per-day features are built from stored bars, a day with fewer bars than this is skipped.")
    load_buffer_days: int = Field(default=10, ge=0, le=365,
        description="W97: calendar days added to the feature frame's length when the store can only load 'the last n days' "
                    "(the fallback loader; stores that load a date range ignore it).")
    session_min_bars: int = Field(default=3, ge=1, le=78,
        description="W97: today's session needs at least this many bars for the session-structure features (else they read 0).")
    hurst_min_bars: int = Field(default=78, ge=16, le=5000, description="W97: bars (one session) the intraday wavelet Hurst needs.")
    scale_spread_min_bars: int = Field(default=200, ge=32, le=5000,
        description="W97: bars (about three sessions) the intraday multi-scale Hurst spread needs.")
    wavelet_min_bars: int = Field(default=40, ge=16, le=5000, description="W97: bars the wavelet-energy features need.")
    psd_min_bars: int = Field(default=100, ge=8, le=5000, description="W97: bars the intraday power-spectrum slope needs.")
    mfdfa_min_returns: int = Field(default=500, ge=100, le=20000,
        description="W97: returns (about seven sessions) the intraday multifractal width needs.")


class FeaturesConfig(_StrictModel):
    """W88d: the indicator windows of the core feature groups (``ml/features.py``).

    SHARED by live and research (like ``ml.hyperparams``): the defaults are the windows the code used as literals before
    W88d, so ``config.yaml`` need not change anything; research may differ via the overlay or a study. Changing a window
    changes the feature VALUES, and for the thirteen model columns whose name embeds a window it changes the NAME too
    (``rsi_14`` -> ``rsi_10``). Every saved model records a hash of these windows and a model trained under other windows
    is not served (it is retrained). W97 added the market-structure, cross-asset, macro, fractal and intraday groups:
    their windows are here too, and the spec hash covers them ONLY when one differs from its default, so at the default
    values the hash (and every saved model) is unchanged. Not here on purpose: clip bounds, scales, neutral values, the
    wavelet family and the MFDFA scales (algorithm constants), the seasonality encoding, and the 252-day annualisation
    constant."""

    momentum: MomentumWindows = Field(default_factory=MomentumWindows)
    volatility: VolatilityWindows = Field(default_factory=VolatilityWindows)
    volume: VolumeWindows = Field(default_factory=VolumeWindows)
    trend: TrendWindows = Field(default_factory=TrendWindows)
    multi_timeframe: MultiTimeframeWindows = Field(default_factory=MultiTimeframeWindows)
    vol_regime: VolRegimeWindows = Field(default_factory=VolRegimeWindows)
    # W97: hashed only when a value differs from its default (ml/feature_spec.py), so listing them changes no saved model
    market_structure: MarketStructureWindows = Field(default_factory=MarketStructureWindows)
    cross_asset: CrossAssetWindows = Field(default_factory=CrossAssetWindows)
    macro: MacroWindows = Field(default_factory=MacroWindows)
    fractal: FractalWindows = Field(default_factory=FractalWindows)
    intraday: IntradayWindows = Field(default_factory=IntradayWindows)

    @model_validator(mode="after")
    def _check(self) -> "FeaturesConfig":
        _ordered("vol_regime vs volatility", fast=self.vol_regime.fast, rv_short=self.volatility.rv_short,
                 rv_long=self.volatility.rv_long, slow=self.vol_regime.slow)
        from ait.config.feature_names import derive_names, validate_names     # leaf: no ait.ml import at load time
        validate_names(derive_names(self))
        return self


class MetaLabelNeutralsConfig(_StrictModel):
    """W90c: the neutral value each meta-label input takes when the feature (or the trade's record of it) is missing.
    SHARED by live and research: the engine's entry gate, both training-data builders and the live entry snapshot
    used to carry their own copies (the live snapshot read 0 for a missing column where the others read these).
    Defaults = the pre-W90c literals; a "neutral" is the middle of the feature's usual range."""

    rsi: float = Field(default=50.0, ge=0.0, le=100.0, description="Both RSI windows.")
    bb_position: float = Field(default=0.5, ge=0.0, le=1.0, description="Position inside the Bollinger bands.")
    volume_ratio: float = Field(default=1.0, ge=0.0, le=10.0, description="Volume / its average.")
    realized_vol: float = Field(default=0.20, ge=0.0, le=5.0, description="Annualized realized volatility.")
    atr_pct: float = Field(default=0.01, ge=0.0, le=1.0, description="ATR as a fraction of price.")
    weekly_trend_aligned: float = Field(default=0.5, ge=0.0, le=1.0, description="Weekly-trend agreement flag.")
    volume_confirmation: float = Field(default=0.0, ge=0.0, le=1.0, description="Volume-confirmation flag.")
    macd_hist: float = Field(default=0.0, ge=-100.0, le=100.0, description="MACD histogram.")
    px_vs_mid: float = Field(default=0.0, ge=-1.0, le=1.0, description="Price vs its mid-window SMA.")
    cross: float = Field(default=0.5, ge=0.0, le=1.0, description="Fast/slow SMA cross state.")
    vix_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Scaled VIX level.")
    iv_rank: float = Field(default=0.5, ge=0.0, le=1.0, description="IV rank (0-1).")
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0,
        description="Sentiment (the feed was retired in R12-C, so this is the value every row has).")
    primary_confidence: float = Field(default=0.60, ge=0.0, le=1.0,
        description="Primary-model confidence for a trade record that carries none (backtest builder).")
    hour_of_day: int = Field(default=10, ge=0, le=23,
        description="Entry hour for rows that carry no timestamp (backtest entries are effectively at the open).")


class MLConfig(_StrictModel):
    meta_label_neutrals: MetaLabelNeutralsConfig = Field(default_factory=MetaLabelNeutralsConfig,
        description="W90c: neutral values of missing meta-label inputs (shared by live and research).")
    features: FeaturesConfig = Field(default_factory=FeaturesConfig,
        description="W88d: indicator windows of the core feature groups (defaults = the pre-W88d literals).")
    hyperparams: TreeHyperparamsConfig = Field(default_factory=TreeHyperparamsConfig,
        description="W88c: tree hyperparameters of the direction ensemble, range model and vol-magnitude model "
                    "(defaults = the pre-W88c literals: 200 / 5 / 0.05 / 0.8 / 0.8, seed 42).")
    meta_label_hyperparams: TreeHyperparamsConfig = Field(
        default_factory=lambda: TreeHyperparamsConfig(n_estimators=100, max_depth=3, random_state=None),
        description="W88c: the meta-labeler's own profile (100 trees, depth 3, unseeded: small dataset, shallow "
                    "on purpose).")
    ensemble_weights: dict[str, float] = {"xgboost": 0.5, "lightgbm": 0.5}
    retrain_interval_days: int = Field(default=7, ge=1, le=30)
    lookback_days: int = Field(default=504, ge=60, le=2520)
    retrain_ib_client_id: int = Field(default=94, ge=2, le=999_999,
        description="W89: IB API client id the supervisor's daily retrain connects with, to fetch daily "
                    "bars from IB (no Yahoo fallback). Must differ from IBKR_CLIENT_ID and its fallback "
                    "ladder (base+100..base+104) and from data_collection.short_dte_client_id (91); the "
                    "retrain refuses to run otherwise. 90 = backfill_historical_data.py by convention.")
    min_training_samples: int = Field(default=100, ge=30)
    # ABLATION VERDICT REVERSED 2026-08-08 (rule B1 pre-registered in PLAN).
    # The 08-03 "gates veto nothing" run was VACUOUS: per-window training
    # failed 96/96 times (train_days=126 left ~27 samples vs the 100 floor),
    # so all three arms were the same ungated engine and the live artifact
    # leaked in as a look-ahead predictor. Re-run with train_days=365 + the
    # R16 fence, 11y / 68 windows / 242 models genuinely trained:
    #   gates OFF n=274 PF 1.05 DD 34.57%   gates ON n=92 PF 1.27 DD 9.97%
    # The gates are a DRAWDOWN control above all — 34.6% -> 10.0% across
    # 2018/2020/2022 vol events — and they clear B1 (PF +0.22 > 0.10, DD
    # better, n>=30, 40 trading windows). Default back ON.
    # COST, measured and accepted: ~66% of candidate entries rejected.
    entry_gates_enabled: bool = True
    range_min_confidence: float = Field(default=0.65, ge=0.50, le=0.90,
        description="Floor for model-overridden signal confidence (range/"
                    "vol-mag). 0.65 beat 0.55 across every backtest metric. "
                    "Was hardcoded in orchestrator — audit item 3.3.")
    # --- W90d: the live-shared label / validation / retrain constants (were literals in ml/*.py; values unchanged) ---
    range_threshold_pct: float = Field(default=0.05, gt=0.0, le=0.5,
        description="W90d: the LIVE range model's containment band (+/- this fraction), the wing width the live condor sells. "
                    "ml.range_spec.live_range_spec reads it. Research does NOT use it: it adapts the band per window "
                    "(backtest.walkforward_tuning.range_threshold_*). A saved range model records the band it was trained at "
                    "and a mismatch is flagged (needs retrain), so changing this means a retrain.")
    range_min_edge_over_baseline: float = Field(default=0.05, ge=0.0, le=1.0,
        description="W90d: the live range model is only used when its cross-validated edge over the baseline hit rate is at "
                    "least this (RangePredictor's default; research passes backtest.min_edge_over_baseline).")
    vol_magnitude_threshold_pct: float = Field(default=0.07, gt=0.0, le=0.5,
        description="W90d: the vol-magnitude model's 'big move' size (a +/- fraction); bigger than the range band on purpose.")
    vol_magnitude_horizon_days: int = Field(default=30, ge=1, le=120,
        description="W90d: ... and the number of days ahead it looks for that move.")
    vol_magnitude_min_edge_over_baseline: float = Field(default=0.10, ge=0.0, le=1.0,
        description="W90d: the vol-magnitude model is only used when its CV edge over the baseline is at least this.")
    direction_label_threshold_pct: float = Field(default=0.01, gt=0.0, le=0.2,
        description="W90d: the direction model's 3-class label: a forward return above +this is bullish, below -this "
                    "bearish, otherwise neutral.")
    direction_label_horizon_days: int = Field(default=5, ge=1, le=60,
        description="W90d: ... measured over this many days forward (also the rows dropped from the end of the training set).")
    direction_cv_splits: int = Field(default=5, ge=2, le=20,
        description="W90d: walk-forward validation folds of the direction model (and of the retrain's hyperparameter search).")
    direction_cv_gap_days: int = Field(default=5, ge=0, le=60,
        description="W90d: purge gap between the direction model's training and validation rows.")
    cv_min_val_rows: int = Field(default=20, ge=5, le=500,
        description="W90d: smallest validation fold of the direction model's walk-forward split.")
    cv_min_train_rows: int = Field(default=50, ge=10, le=2000,
        description="W90d: a direction-model fold whose training part is shorter than this is skipped (with fewer than two "
                    "usable folds the split falls back to sklearn's TimeSeriesSplit).")
    binary_cv_splits: int = Field(default=4, ge=2, le=20,
        description="W90d: walk-forward folds of the two binary models (range, vol-magnitude).")
    binary_cv_min_gap_days: int = Field(default=5, ge=0, le=60,
        description="W90d: purge gap of the binary models is the larger of this and the label horizon (the horizon must be "
                    "covered or the overlapping label windows leak into validation: audit 2026-07-07 item 2.4).")
    label_balance_min: float = Field(default=0.05, gt=0.0, lt=0.5,
        description="W90d: a binary label whose positive rate is below this is warned about (too few positives to learn from).")
    label_balance_max: float = Field(default=0.98, gt=0.5, lt=1.0,
        description="W90d: ... or above this (almost always positive). Warning only.")
    min_class_samples: int = Field(default=10, ge=2, le=1000,
        description="W90d: a binary model is not trained unless each class has at least this many rows.")
    retrain_optuna_trials: int = Field(default=50, ge=1, le=2000,
        description="W90d: Optuna trials of the live retrain's optional hyperparameter search.")
    retrain_optuna_seed: int = Field(default=42, ge=0, le=2**31 - 1,
        description="W90d: seed of that search's TPE sampler.")
    rollback_margin: float = Field(default=0.05, gt=0.0, le=0.5,
        description="W90d: a retrained direction model whose average CV accuracy is more than this below the previous one's "
                    "(5 percentage points) is rolled back.")
    observe_mode_neutral_confidence: float = Field(default=0.60, ge=0.50, le=0.90,
        description="OBSERVE MODE ONLY (entry_gates_enabled=false): the "
                    "confidence a DIRECTION-NEUTRAL structure (iron_condor, "
                    "short_strangle) carries into risk validation. "
                    "trade-life-gatesoff-reintroduces-neutral-autoreject "
                    "(2026-09-01): with gates off nothing writes "
                    "model_overridden, so eff_conf fell back to the "
                    "DIRECTIONAL confidence and manager.py rejected anything "
                    "below risk.min_confidence — in exactly the neutral "
                    "regime a condor wants. Only trending-aligned days "
                    "survived, i.e. condors entered ONLY in their worst "
                    "regime (adverse selection). The risk manager's "
                    "min_confidence is a DIRECTIONAL gate; a market-neutral "
                    "structure is not paid for direction, so it is validated "
                    "on this neutral baseline instead. Must stay >= "
                    "risk.min_confidence or observe mode blocks itself; the "
                    "0.60 default sits above the shipped 0.50 with headroom. "
                    "Ignored entirely when gates are ON.")
    regime_override_confidence: float = Field(default=0.80, ge=0.50, le=0.99,
        description="W69: a HIGH_VOLATILITY regime classification at or above "
                    "this confidence overrides the ML direction to NEUTRAL "
                    "at entry (W93: 'at or above'; it was a strict 'above'). Was a literal 0.8 in orchestrator (the R5 "
                    "classifier floors high-vol confidence at 0.80 so any "
                    "VIX/vol trigger clears it — keep this at or below regime.high_vol_confidence_floor, 0.80). "
                    "Iron condors ignore the direction, so this only matters for direction-sensitive strategies.")
    # --- Drift detection (ml/drift.py). W92: switchable, default OFF; every number explained. Guide: RESEARCH_GUIDE "Drift detection". ---
    drift_detection_enabled: bool = Field(default=False,
        description="W92: master switch of the live drift detector. OFF (default): it records nothing and always reports 'no "
                    "drift', so it can never ask for an early retrain. ON: it scores the direction model's live calls against "
                    "what happened and asks for an early retrain when its accuracy or calibration degrades (the numbers below). "
                    "Caveat: outcomes are only recorded for directional strategies (not iron_condor / short_strangle / "
                    "long_straddle), so in an iron-condor-only book switching it on changes nothing until directional trades "
                    "close.")
    drift_window_size: int = Field(default=50, ge=20, le=500,
        description="W69: N = how many of the most recent RESOLVED predictions (a prediction whose trade has closed) the detector "
                    "looks at. Larger = steadier but slower to notice a change; smaller = quicker but noisier.")
    drift_accuracy_threshold: float = Field(default=0.40, ge=0.20, le=0.80,
        description="W69: if the share of those predictions that were right falls below this, the model is flagged as drifting "
                    "and an early retrain is requested. 0.40 = right less than 40% of the time; a 3-way direction call guessing "
                    "at random is right about 33%, so this is only just above chance.")
    drift_high_confidence: float = Field(default=0.80, ge=0.60, le=0.99,
        description="W69: predictions made with at least this confidence form the 'high-confidence group', which should be right "
                    "about this often (80% confident should mean about 80% right).")
    drift_calibration_threshold: float = Field(default=0.25, ge=0.05, le=0.60,
        description="W69: if the high-confidence group is right LESS often than drift_high_confidence by more than this (0.25 = 25 "
                    "percentage points, i.e. it is right under 55% of the time at the defaults), the model is overconfident and "
                    "is flagged as drifting.")
    drift_min_samples: int = Field(default=20, ge=5, le=500,
        description="W92: nothing is reported until at least this many predictions in the window have an outcome (it was a literal "
                    "20 in check_drift). Must not exceed drift_window_size.")
    drift_min_high_confidence_samples: int = Field(default=5, ge=1, le=500,
        description="W92: the calibration test needs at least this many resolved high-confidence predictions (it was a literal 5); "
                    "with fewer it is skipped and only accuracy counts.")
    drift_check_interval: int = Field(default=10, ge=1, le=500,
        description="W92: the full check runs at most once per this many newly recorded predictions; in between, the last report is "
                    "returned. The detector always documented this behaviour but never implemented it (the value was stored and "
                    "unused). 1 = check on every call.")
    model_versions_kept: int = Field(default=5, ge=1, le=100,
        description="W92: how many saved direction-model versions are kept on disk; older ones are deleted each time a model is "
                    "saved (it was a literal keep=5). This is also how far back a rollback can go.")

    @model_validator(mode="after")
    def _drift_numbers_are_consistent(self) -> "MLConfig":
        if self.drift_min_samples > self.drift_window_size:
            raise ValueError(f"ml: drift_min_samples ({self.drift_min_samples}) cannot exceed drift_window_size "
                             f"({self.drift_window_size}): the detector would never report")
        return self

    @field_validator("ensemble_weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return v

    @model_validator(mode="after")
    def _label_balance_is_ordered(self) -> "MLConfig":
        if not self.label_balance_min < self.label_balance_max:
            raise ValueError("ml: label_balance_min must be below label_balance_max")
        return self


def regime_trend_history_need(cfg: "RegimeConfig") -> int:
    """W95: the fewest daily bars the regime classifier's own measures can be computed from. An SMA slope over
    ``trend_slope_lookback`` days needs the SMA's period plus that many bars, a deviation needs its SMA's period, the
    momentum return needs ``trend_momentum_days`` + 1, and the 20-day realized volatility needs 21 (20 returns). The
    single definition: ``RegimeConfig`` refuses a ``min_history_rows`` below it."""
    return max(max(cfg.trend_sma_periods) + cfg.trend_slope_lookback, max(cfg.trend_deviation_periods),
               cfg.trend_momentum_days + 1, 21)


class RegimeConfig(_StrictModel):
    """W69: market-regime classification thresholds (ml/regime.py `_classify`).

    Thresholds only. The confidence each classification returns is part of the
    classifier's design (and the R5 audit fix that makes a high-vol trigger
    clear the 0.70/0.80 consumer gates) and stays in code. Defaults are the
    pre-W69 literals, so omitting the section changes nothing.
    """

    high_vol_vix: float = Field(default=30.0, ge=15.0, le=80.0,
        description="VIX above this classifies HIGH_VOLATILITY.")
    high_vol_realized: float = Field(default=0.40, ge=0.15, le=2.0,
        description="Annualised realised vol above this classifies "
                    "HIGH_VOLATILITY (either trigger is enough).")
    iv_crush_iv_rank_max: float = Field(default=0.25, ge=0.0, le=1.0,
        description="IV_CRUSH needs IV rank (0-1) below this ...")
    iv_crush_vol_max: float = Field(default=0.20, ge=0.01, le=1.0,
        description="... realised vol below this ...")
    iv_crush_trend_max: float = Field(default=0.20, ge=0.0, le=1.0,
        description="... and |trend strength| below this.")
    mean_revert_hurst_max: float = Field(default=0.40, ge=0.0, le=0.5,
        description="MEAN_REVERTING needs the Hurst exponent below this ...")
    mean_revert_trend_max: float = Field(default=0.25, ge=0.0, le=1.0,
        description="... and |trend strength| below this.")
    low_vol_vix: float = Field(default=15.0, ge=5.0, le=30.0,
        description="VIX below this (or realised vol below low_vol_realized) "
                    "is the LOW_VOLATILITY branch.")
    low_vol_realized: float = Field(default=0.12, ge=0.02, le=0.60,
        description="Realised vol below this is the LOW_VOLATILITY branch.")
    low_vol_trend_breakout: float = Field(default=0.30, ge=0.05, le=1.0,
        description="Inside the low-vol branch, |trend| above this is "
                    "TRENDING_UP/DOWN instead of LOW_VOLATILITY.")
    strong_trend: float = Field(default=0.35, ge=0.05, le=1.0,
        description="|trend strength| above this (outside the branches above) "
                    "classifies TRENDING_UP/DOWN.")
    # --- W92: how the trend strength (-1 to +1) is measured, and the confidence each class reports. Guide: RESEARCH_GUIDE "The regime
    # classifier". Live-only: research does not use the RegimeDetector. ---
    trend_sma_periods: list[int] = Field(default=[10, 20, 50], min_length=1, max_length=6,
        description="W92: the trend score averages the slope of the simple moving average of each of these lengths (days).")
    trend_slope_lookback: int = Field(default=5, ge=1, le=60,
        description="W92: a moving average's slope is its percentage change over this many days ...")
    trend_slope_scale: float = Field(default=100.0, gt=0.0, le=10000.0,
        description="W92: ... multiplied by this and clamped to [-1, 1] (100 = a 1% move in 5 days scores 1.0).")
    trend_deviation_periods: list[int] = Field(default=[20, 50], min_length=1, max_length=6,
        description="W92: the trend score also averages how far the price sits above or below the moving average of each of these "
                    "lengths ...")
    trend_deviation_scale: float = Field(default=10.0, gt=0.0, le=10000.0,
        description="W92: ... as a fraction of that average, multiplied by this and clamped to [-1, 1] (10 = 10% away scores 1.0).")
    trend_momentum_days: int = Field(default=20, ge=1, le=252,
        description="W92: ... and the price return over this many days ...")
    trend_momentum_scale: float = Field(default=5.0, gt=0.0, le=10000.0,
        description="W92: ... multiplied by this and clamped to [-1, 1] (5 = a 20% return scores 1.0).")
    high_vol_confidence_floor: float = Field(default=0.80, ge=0.5, le=1.0,
        description="W92: a HIGH_VOLATILITY classification reports a confidence of at least this. The live entry rule 'a high-vol "
                    "regime at or above ml.regime_override_confidence forces direction NEUTRAL' compares against it, so floor and "
                    "ml.regime_override_confidence are a must-agree pair: keep ml.regime_override_confidence at or below this floor, "
                    "or a day triggered by VIX alone (which scores exactly the floor) never reaches the override. At the shipped "
                    "0.80 / 0.80 it does (W93: the rule was a strict 'above' and did not).")
    high_vol_confidence_cap: float = Field(default=0.90, ge=0.5, le=1.0,
        description="W92: ... and at most this.")
    high_vol_confidence_vol_scale: float = Field(default=0.50, gt=0.0, le=5.0,
        description="W92: between the floor and the cap the confidence is realized volatility divided by this (0.40 vol -> 0.80).")
    iv_crush_confidence: float = Field(default=0.70, ge=0.0, le=1.0,
        description="W92: the confidence an IV_CRUSH classification reports (reported only: no live rule compares against it).")
    mean_revert_confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0,
        description="W92: a MEAN_REVERTING classification reports max(this, 1 - Hurst exponent) (reported only).")
    low_vol_confidence: float = Field(default=0.70, ge=0.0, le=1.0,
        description="W92: the confidence a LOW_VOLATILITY classification reports (reported only).")
    trend_confidence_cap: float = Field(default=0.95, ge=0.5, le=1.0,
        description="W92: a TRENDING_UP / TRENDING_DOWN classification reports min(this, |trend strength|) (reported only).")
    range_confidence_floor: float = Field(default=0.50, ge=0.0, le=1.0,
        description="W92: a RANGE_BOUND classification reports max(this, 1 - 2 x |trend strength|) (reported only).")
    # --- W95: how much history the classifier needs. Guide: RESEARCH_GUIDE "The regime classifier" ---
    min_history_rows: int = Field(default=60, ge=21, le=1000,
        description="W95: the classifier needs at least this many daily bars; with fewer it returns RANGE_BOUND at confidence 0 "
                    "(it makes no call). Must cover the longest look-back of the trend measure (the longest trend_sma_periods "
                    "entry plus trend_slope_lookback, the longest trend_deviation_periods entry, trend_momentum_days + 1): "
                    "55 at the defaults.")
    iv_rank_min_rows: int = Field(default=60, ge=2, le=1000,
        description="W95: the IV-rank proxy (today's 20-day realized vol as a percentile of its own recent range) is computed "
                    "only from at least this many 20-day-vol values, else it reports a neutral 0.5. Each value needs 20 bars "
                    "of its own, so at the default the proxy stays 0.5 until the history reaches about 80 bars, although "
                    "min_history_rows alone would let the classifier run at 60.")
    iv_rank_window_rows: int = Field(default=252, ge=2, le=2000,
        description="W95: the IV-rank proxy compares today's 20-day vol with the highest and lowest 20-day vol over the last "
                    "this many values (252 = one trading year; fewer are used while the history is shorter). Not the stored-IV "
                    "rank of ait.data.iv_rank and not the ml.features.vol_regime.iv_rank_lookback feature: a separate proxy "
                    "built from prices alone.")

    @model_validator(mode="after")
    def _vol_bands_are_ordered(self) -> "RegimeConfig":
        if self.low_vol_realized >= self.high_vol_realized:
            raise ValueError(
                f"regime.low_vol_realized ({self.low_vol_realized}) must be below "
                f"high_vol_realized ({self.high_vol_realized})")
        if self.low_vol_vix >= self.high_vol_vix:
            raise ValueError(
                f"regime.low_vol_vix ({self.low_vol_vix}) must be below "
                f"high_vol_vix ({self.high_vol_vix})")
        if self.high_vol_confidence_floor > self.high_vol_confidence_cap:       # W92
            raise ValueError(
                f"regime.high_vol_confidence_floor ({self.high_vol_confidence_floor}) cannot exceed "
                f"high_vol_confidence_cap ({self.high_vol_confidence_cap})")
        need = regime_trend_history_need(self)                                 # W95
        if self.min_history_rows < need:
            raise ValueError(
                f"regime.min_history_rows ({self.min_history_rows}) is below the {need} bars the trend measure needs "
                f"(longest trend_sma_periods + trend_slope_lookback, longest trend_deviation_periods, "
                f"trend_momentum_days + 1, and the 21 bars of the volatility measure): raise it or shorten those")
        return self


class MetaLabelConfig(_StrictModel):
    # R12-C: default flipped to False (was True). The meta-labeler was trained
    # on corrupted data and rejects everything; config.yaml has carried
    # `enabled: false` since that finding, so this only changes what happens
    # if the key is ever omitted — default-off is the safe direction.
    enabled: bool = False
    min_probability: float = Field(default=0.50, ge=0.30, le=0.80)
    retrain_with_primary: bool = True  # Retrain when primary model retrains
    min_trades_for_training: int = Field(default=30, ge=10, le=1000,
        description="W92: the meta-labeler is not trained on fewer closed trades than this (it was the module constant "
                    "MIN_TRADES_FOR_TRAINING = 30). min_closes_to_enable below is the separate question of when its output is "
                    "TRUSTED for live entries.")
    min_class_trades: int = Field(default=5, ge=2, le=500,
        description="W92: ... and it needs at least this many profitable AND this many unprofitable trades (a literal 5).")
    cv_max_splits: int = Field(default=5, ge=2, le=20,
        description="W92: its training cross-validation uses at most this many folds ...")
    cv_min_rows_per_split: int = Field(default=10, ge=2, le=500,
        description="W92: ... with one fold per this many rows, never fewer than 2 (it was min(5, max(2, rows // 10))).")
    min_feature_coverage: int = Field(default=15, ge=1, le=20,
        description="W90c: of the 20 meta-label inputs, at least this many must carry real (non-constant) data "
                    "before a freshly trained model is armed (R16 arm guard: the 07-18 artifact was trained on 9 "
                    "of 20 and rejected everything). Was MIN_FEATURE_COVERAGE = 15.")
    max_artifact_age_days: int = Field(default=45, ge=1, le=3650,
        description="W90c: a saved meta-label model older than this is stale and refused at load. Was "
                    "MAX_ARTIFACT_AGE_DAYS = 45.")
    min_closes_to_enable: int = Field(default=100, ge=30, le=500,
        description="Real, verdict-eligible closed trades (the same population "
                    "reporting.go_live.real_close_count/the go-live scorecard grades) "
                    "required before the meta-label gate is trusted for LIVE entries -- "
                    "regardless of `enabled` above or whether a model has already "
                    "trained. W49 (policy-vs-impl-7): config.yaml's own comment said "
                    "50, PLAN.md's tracking table said 100+, and the code enforced "
                    "neither at arm time -- this is now the one number, adjustable "
                    "here without a code deploy. Floor of 30 matches "
                    "meta_label.min_trades_for_training, the separate (unchanged) "
                    "statistical minimum for fitting the model at all; this field is "
                    "the different question of when its output should be TRUSTED for "
                    "live risk.")


class EarningsConfig(_StrictModel):
    """W44 (R26 fail-direction-03): EarningsCalendar used to build an
    identically-shaped "unknown" answer whether a symbol genuinely has no
    upcoming earnings or the Yahoo fetch broke, and both gates read unknown
    as "allow" -- a fetch outage silently removed earnings protection. Fixed
    with a last-known-good: a fetch failure now reuses the most recent
    successful answer (which, for the ETF-only live universe, already says
    "no earnings" -- so an outage costs nothing there) and only refuses when
    there is no good answer to fall back on."""

    refuse_entries_when_data_unavailable: bool = Field(default=True,
        description="Refuse new entries for a symbol when its earnings data "
                    "is unknown (never fetched, or the fetch failed with no "
                    "usable last-known-good). Operator escape hatch if this "
                    "ever misfires live -- flip to False to restore the old "
                    "always-allow behavior without a code deploy.")
    lkg_max_age_hours: int = Field(default=168, ge=1, le=720,
        description="How long a successful earnings fetch may be reused "
                    "after a later fetch fails. Earnings dates move rarely, "
                    "so a week-old confirmed answer beats refusing entries "
                    "outright; past this, treat the symbol as unknown again.")
    buffer_days_before: int = Field(default=2, ge=0, le=10,
        description="Block new entries/flag exits starting this many "
                    "calendar days before a known earnings date.")
    buffer_days_after: int = Field(default=1, ge=0, le=10,
        description="Keep the earnings danger zone open this many calendar "
                    "days after the earnings date (post-print IV settling).")


class DividendConfig(_StrictModel):
    """W51 (model-vs-reality-ex-div-assignment-void): DividendCalendar's
    twin of EarningsConfig above — same last-known-good policy, applied to
    ex-dividend dates instead of earnings dates. Unlike earnings (dead for
    the ETF-only live universe — no symbol has a per-company earnings date),
    every name in the universe pays a real, regular dividend, so this gate
    is live every quarter (monthly for TLT) rather than effectively inert."""

    refuse_entries_when_data_unavailable: bool = Field(default=True,
        description="Refuse new entries for a symbol when its dividend data "
                    "is unknown (never fetched, or the fetch failed with no "
                    "usable last-known-good). Operator escape hatch if this "
                    "ever misfires live -- flip to False to restore the old "
                    "always-allow behavior without a code deploy.")
    lkg_max_age_hours: int = Field(default=168, ge=1, le=720,
        description="How long a successful dividend fetch may be reused "
                    "after a later fetch fails. Ex-dividend dates move "
                    "rarely once a schedule is established, so a week-old "
                    "confirmed answer beats refusing entries outright; past "
                    "this, treat the symbol as unknown again.")


class AssignmentResponseConfig(_StrictModel):
    """W53 (roll-mechanics groundwork): whether an at-risk short leg (rule 3/
    3e in portfolio.py) closes outright or rolls to a later expiry instead.
    `policy` and `execute_rolls` both default off/False so shipping this
    config surface changes NO live behavior until the operator explicitly
    flips both (rule 3/3e are wired to consult `policy` as of W54b, but
    `close_only` is the default and `execute_rolls=False` is a separate,
    additionally-required gate before anything is ever compared or
    placed). A roll always replaces the full short+long vertical, never a
    single leg alone, so the position stays defined-risk through the
    swap -- see this wave's PLAN.md entry for why."""

    policy: Literal["close_only", "roll_only", "close_or_roll"] = Field(
        default="close_only",
        description="close_only: today's behavior, always close an at-risk "
                    "leg. roll_only/close_or_roll: rule 3/3e report a roll "
                    "candidate (PositionStatus.roll_side) instead of closing "
                    "outright, but execute_rolls below is ADDITIONALLY "
                    "required before anything is actually compared or placed "
                    "-- same two-layer gate condor_defense.enabled/"
                    "execute_rolls already uses.")
    execute_rolls: bool = Field(default=False,
        description="W54b: a SEPARATE opt-in from policy above, mirroring "
                    "CondorDefenseConfig.execute_rolls exactly. policy != "
                    "'close_only' alone still only makes rule 3/3e REPORT a "
                    "roll candidate -- execute_rolls=True is additionally "
                    "required before the orchestrator ever runs the close-"
                    "or-roll comparison or places a real order. Defaults "
                    "False so shipping this wave changes NO live behavior "
                    "until the operator explicitly flips both this and "
                    "policy.")
    comparison_model: Literal["heuristic", "ev"] = Field(
        default="heuristic",
        description="Which model decides close-vs-roll once policy == "
                    "'close_or_roll' AND execute_rolls is True. 'heuristic': "
                    "the cheap "
                    "credit/max-loss proxy (execution/roll_comparison.py, "
                    "heuristic_should_roll). 'ev': prices both sides at each "
                    "leg's own model-implied fair value instead of raw market "
                    "quotes (ev_should_roll) -- more precise in principle, but "
                    "has NOT been backtested: the historical per-strike vol-"
                    "surface backfill this needs (PLAN.md's 2026-09-27 W54 "
                    "scoping note, finding [6]) is still running as of W54a. "
                    "RECOMMENDED: stay on 'heuristic' until that backtest "
                    "exists and validates 'ev' against real history -- do not "
                    "switch this to 'ev' in a live/paper account before then.")
    max_rolls_per_position: int = Field(default=2, ge=0, le=10,
        description="W54a: hard cap on how many times ONE position may be "
                    "rolled (walks rolled_from_trade_id via state.py's "
                    "count_roll_ancestors), enforced once in orchestrator."
                    "_execute_roll regardless of which system triggered the "
                    "roll. Catches unbounded repeat-rolling a single credit/"
                    "max-loss check per attempt wouldn't -- PLAN.md's W54 "
                    "scoping note [3]c calls this non-negotiable.")
    roll_target_dte_min: int = Field(default=14, ge=1, le=90,
        description="Floor of the later-expiry search window for a roll "
                    "target. Matches options.dte_range's live floor (14) so "
                    "a rolled position lands in the same DTE sweet spot a "
                    "fresh entry would, well clear of EXPIRY_APPROACHING_DTE "
                    "(5) and rule 3's dte<=1 danger zone.")
    roll_target_dte_max: int = Field(default=30, ge=2, le=120,
        description="Ceiling of the later-expiry search window for a roll "
                    "target. Matches options.dte_range's live ceiling (30).")
    roll_min_credit: float = Field(default=0.20, ge=0.0, le=5.0,
        description="Refuse a roll whose net credit (old vertical bought "
                    "back, new vertical sold) is below this. Provisional: "
                    "no roll has ever executed live to calibrate against; "
                    "GAP_AUDIT_R7 estimated ~$2.60+crossing commission for a "
                    "2-leg roll, roughly double for this wave's 4-leg "
                    "vertical roll, so this floor needs real evidence before "
                    "it should be trusted at the cent.")
    roll_short_delta_target: float = Field(default=0.20, ge=0.05, le=0.50,
        description="W69: target |delta| of the NEW short strike when a roll "
                    "re-establishes the at-risk vertical. Mirrors the iron "
                    "condor entry short delta (~0.20): a rolled vertical looks "
                    "like a fresh entry at the new expiry, not a replica of "
                    "the decayed old strike. Was _ROLL_SHORT_DELTA_TARGET in "
                    "execution/roll_selector.py.")
    roll_short_delta_tolerance: float = Field(default=0.07, ge=0.01, le=0.25,
        description="W69: how far the nearest liquid strike's |delta| may sit "
                    "from roll_short_delta_target before the roll is refused "
                    "(select_roll_target returns None). 0.07 matches "
                    "Strategy._find_strike_by_delta's default tolerance. Was "
                    "_ROLL_SHORT_DELTA_TOLERANCE in roll_selector.py.")


class CondorDefenseConfig(_StrictModel):
    """W56 (tastytrade-style tested/untested-side condor defense, decision
    layer): per open iron condor, classify the TESTED short's synthetic
    delta (W55, execution/synthetic_delta.py) plus DTE against this table
    to decide hold / roll-untested / roll-tested / close. `enabled` defaults
    to False so shipping this config surface and execution/condor_defense.py
    behind it changes NO live behavior — nothing calls into it yet (the
    same convention AssignmentResponseConfig above and W53/W55's additive
    modules use). See PLAN.md's 2026-09-22 ROADMAP EXTENSION [4]/[5] for the
    playbook table this mirrors and the GAP_AUDIT_R7 POST-VERDICT reasoning
    for the off-by-default choice — thin edge at 1 contract, and changing
    exit mechanics mid-sample corrupts the 50-close measurement."""

    enabled: bool = Field(default=False,
        description="Off: check_positions never consults this config (once "
                    "a future wave wires it in) — always hold. On: the "
                    "decision table below becomes live. Stays off pending "
                    "the 50-close paper verdict, per GAP_AUDIT_R7.")
    side: Literal["tested_only", "untested_only", "both"] = Field(
        default="untested_only",
        description="Which rolls are allowed. tested_only: only roll the "
                    "side under pressure (a debit, buys distance). "
                    "untested_only: only roll the side NOT under pressure "
                    "(a credit, widens the aggregate breakeven — the "
                    "playbook's primary recommendation). both: either, per "
                    "the table. A roll the current side setting disallows "
                    "holds instead (does not escalate to close) — the "
                    "0.40-delta close row is never side-filtered.")
    monitor_delta_max: float = Field(default=0.25, ge=0.0, le=1.0,
        description="Below this, the tested short is not under enough "
                    "pressure to act on — hold. Document's table, row 1.")
    roll_untested_delta_min: float = Field(default=0.25, ge=0.0, le=1.0,
        description="Floor of the tested-short delta band (inclusive) that "
                    "triggers rolling the UNTESTED side for a credit. "
                    "Document's table, row 2.")
    roll_untested_delta_max: float = Field(default=0.30, ge=0.0, le=1.0,
        description="Ceiling of the roll-untested delta band (exclusive — "
                    "at this delta the tested-side row below takes over). "
                    "Document's table, row 2.")
    roll_untested_dte_min: int = Field(default=14, ge=1, le=365,
        description="Floor (inclusive) of the DTE window the roll-untested "
                    "row applies in. Document's table, row 2.")
    roll_untested_dte_max: int = Field(default=21, ge=1, le=365,
        description="Ceiling (inclusive) of the DTE window the "
                    "roll-untested row applies in — beyond this the "
                    "tested-side row takes over. Document's table, row 2.")
    roll_tested_delta_min: float = Field(default=0.30, ge=0.0, le=1.0,
        description="Floor (inclusive) of the tested-short delta band that "
                    "triggers rolling the TESTED side itself for a debit, "
                    "when DTE is also >= roll_tested_dte_min. Document's "
                    "table, row 3 (low-gamma / >21 DTE).")
    roll_tested_dte_min: int = Field(default=22, ge=1, le=365,
        description="Floor (inclusive) of the DTE window the roll-tested "
                    "row applies in — explicit (not just \"not in the "
                    "roll-untested window\") so the two bands can't "
                    "silently overlap if either is retuned independently.")
    close_delta_threshold: float = Field(default=0.40, ge=0.0, le=1.0,
        description="At or above this delta, close outright regardless of "
                    "DTE or side setting — high-gamma / near-expiry, the "
                    "document's table row 4. Not side-filtered: closing is "
                    "not a roll.")
    execute_rolls: bool = Field(default=False,
        description="W61: a SEPARATE opt-in from `enabled` above. enabled=True "
                    "alone still only makes CLOSE actionable and reports a ROLL "
                    "verdict (action=ROLL) without acting on it — execute_rolls="
                    "True is additionally required before orchestrator._execute_"
                    "roll ever selects a target, prepares a combo, or places a "
                    "real order. Defaults False so shipping the wiring changes "
                    "NO live behavior until the operator explicitly flips both.")

    @model_validator(mode="after")
    def _validate_bands(self) -> "CondorDefenseConfig":
        if not (self.monitor_delta_max <= self.roll_untested_delta_min
                < self.roll_untested_delta_max <= self.roll_tested_delta_min
                < self.close_delta_threshold):
            raise ValueError(
                "condor_defense delta bands must satisfy monitor_delta_max <= "
                "roll_untested_delta_min < roll_untested_delta_max <= "
                "roll_tested_delta_min < close_delta_threshold "
                f"(got {self.monitor_delta_max}, {self.roll_untested_delta_min}, "
                f"{self.roll_untested_delta_max}, {self.roll_tested_delta_min}, "
                f"{self.close_delta_threshold})")
        if not (self.roll_untested_dte_min <= self.roll_untested_dte_max
                < self.roll_tested_dte_min):
            raise ValueError(
                "condor_defense DTE windows must satisfy roll_untested_dte_min "
                "<= roll_untested_dte_max < roll_tested_dte_min "
                f"(got {self.roll_untested_dte_min}, {self.roll_untested_dte_max}, "
                f"{self.roll_tested_dte_min})")
        return self


class SentimentConfig(_StrictModel):
    """R12-C tombstone: the sentiment stack (ait.sentiment, ib_news,
    fundamentals_db) is retired to deprecated/src/ — verified zero influence
    on iron-condor decisions. Nothing reads this config anymore.

    Kept as a permissive stub (extra="allow") so an existing config.yaml
    `sentiment:` block — including its nested `sources:` mapping — still
    validates instead of crashing load_settings() at bot startup. That is the
    least-breaking path: no config edit required, no consumer left to care
    what the values are.
    """

    enabled: bool = False
    model_config = {"extra": "allow"}


class ExitConfig(_StrictModel):
    exit_cross_amount: float = Field(default=0.10, ge=0.01, le=0.50,
        description="How far a combo EXIT limit crosses the spread so the "
                    "close actually fills (was hardcoded EXIT_CROSS in "
                    "orchestrator — audit item 3.3).")
    exit_mark_multiple: float = Field(default=1.5, ge=1.05, le=3.0,
        description="W7 (R24 logic-exit-risk-01): ceiling on a credit "
                    "buyback as a multiple of the CURRENT marked cost to "
                    "close. The R16 bound was "
                    "max(2*mark, entry_credit + 0.25*wing_width) — at the "
                    "promoted $39-60 wings the width term dominated and "
                    "priced routine take-profits at 6-10x the mark (SPY sent "
                    "BUY LMT 14.04 against a 1.40 mark). IBKR's price band "
                    "rejected 61 of 63 exits; the broker's guard was the ONLY "
                    "thing bounding the price. The bound is now anchored to "
                    "the mark alone; the wing width remains the structural "
                    "cap above it, and mark+exit_cross_amount is the floor so "
                    "the order stays marketable.")
    exit_min_ceiling_headroom: float = Field(default=0.60, ge=0.10, le=2.0,
        description="W33 (R26 mark-ceiling-clamps-below-the-ask-at-low-marks): "
                    "the least absolute headroom the mark-anchored ceiling "
                    "leaves above the mark. The ceiling is compared with an "
                    "ASK-side price and the mark is a mid; a four-leg combo's "
                    "summed half-spread does not shrink with the mark (0.52-"
                    "0.58 measured on QQQ 09-11), so a purely proportional "
                    "1.5x overwrote marketable limits at marks under ~1.1.")
    exit_band_ttl_seconds: float = Field(default=300.0, ge=30.0, le=3600.0,
        description="W33: how long a broker-stated price band caps later exit "
                    "attempts. The band is the broker's view of one moment; "
                    "W7 kept it an hour, which (once bands were actually "
                    "captured) could pin a stop below a market that had "
                    "moved. A live reject loop re-states it every 30-180s.")
    exit_unfilled_escalation_pct: float = Field(default=0.05, ge=0.0, le=0.25,
        description="W33b: each consecutive QUOTED exit placement that did not "
                    "fill raises the next limit by this fraction of the "
                    "marketable price (n-th retry: x(1 + pct*n)), under the "
                    "same caps (wing width, mark-anchored ceiling, broker "
                    "band). On delayed data every quote is ~15 minutes old, "
                    "so `ask + cross` chases a trending tape without ever "
                    "catching it (2026-09-21: 8.67 then 9.76, both unfilled). "
                    "0 disables.")
    exit_reprice_step: float = Field(default=0.10, ge=0.01, le=1.0,
        description="W33: how far each consecutive NO-QUOTE exit placement "
                    "steps above the last (anchor + cross + n*step, capped). "
                    "W7 re-placed the identical price every backoff.")
    exit_pending_rest_seconds: float = Field(default=300.0, ge=30.0, le=1800.0,
        description="W52a: how long a working, unfilled close order rests "
                    "before check_fills cancels it so the monitor can "
                    "re-trigger (and W33b escalate) the exit. The old "
                    "hard-coded 300s, now the value for every non-stop exit.")
    exit_stop_pending_rest_seconds: float = Field(default=120.0, ge=15.0, le=1800.0,
        description="W52a: the same rest for STOP exits (stop_loss, "
                    "short_strike_touch, trailing_stop, breakeven_stop). "
                    "2026-09-21 QQQ: the touch stop took 25 minutes to fill "
                    "because each unfilled placement rested the take-profit "
                    "300s on 15-minute-old quotes. A marketable limit that "
                    "has not filled in two minutes is not going to.")
    exit_blind_credit_multiple: float = Field(default=2.0, ge=1.0, le=5.0,
        description="W33 (R26 raw-wing-width-fallback-still-reachable): with "
                    "no quote and no fresh mark, the most a credit buyback "
                    "may pay is this multiple of the entry credit (and never "
                    "more than the wing width). W7 priced that state at the "
                    "raw width — the structural max loss.")
    credit_touch_stop: bool = Field(default=True,
        description="W36: config home for AIT_CREDIT_TOUCH_STOP — close a "
                    "credit structure when the underlying touches a short "
                    "strike (R12-B1: the best-ranked credit stop).")
    trailing_stop_pct: float = Field(default=0.25, ge=0.10, le=0.50)
    breakeven_trigger_pct: float = Field(default=0.30, ge=0.10, le=0.80)
    partial_exit_levels: list[dict] = [
        {"pnl_pct": 0.50, "close_pct": 0.33},
        {"pnl_pct": 1.00, "close_pct": 0.33},
    ]
    time_decay_scaling: bool = True
    volatility_adjusted_stops: bool = True
    initial_stop_loss_pct: float = Field(default=0.50, ge=0.15, le=0.75)
    auto_hedge: bool = False
    # R14: staleness gate on exit inputs. The touch stop is the only exit rule
    # that acts DIRECTLY on the underlying's price, and it read that price with
    # no quality check at all — a frozen feed could fire it on a breach that had
    # long since passed, or hide a real one. Budget is generous (6x the fast
    # monitor's cadence) precisely because the monitor's own cadence and quote
    # cache TTL are what it is being generous about — see ait.utils.cadence
    # (W47 numeric-pairs-07) for the shared authority these derive from.
    max_quote_staleness_seconds: float = Field(default=QUOTE_STALENESS_BUDGET_SECONDS, ge=30.0, le=900.0,
        description="Underlying quote older than this (by exchange tick time) "
                    "marks the touch stop's input DEGRADED — it then needs two "
                    "agreeing ticks to fire instead of one.")
    touch_confirm_ticks: int = Field(default=2, ge=1, le=5,
        description="Consecutive agreeing evaluations required before a touch "
                    "stop fires on a degraded/frozen quote. 1 = fire on a "
                    "single stale print (the pre-R14 behaviour).")
    max_quote_spread_pct: float = Field(default=0.15, ge=0.02, le=0.50,
        description="Underlying quote bid-ask spread above this marks the "
                    "touch stop's input DEGRADED — same two-agreeing-ticks "
                    "confirmation as a stale or outlier print. Was a "
                    "data.quality.py code default, never config-backed "
                    "(qual-04, R25).")
    max_quote_jump_pct: float = Field(default=0.10, ge=0.02, le=0.50,
        description="Underlying quote change vs. the last confirmed price "
                    "above this marks the touch stop's input DEGRADED "
                    "(qual-01/04, R25). Was a data.quality.py code default, "
                    "never config-backed.")
    thesis_direction_flip_confidence: float = Field(default=0.70, ge=0.50, le=0.99,
        description="W69: the open-position thesis check exits when the ML "
                    "direction has flipped against the entry direction with "
                    "confidence above this. Was a literal 0.70 in "
                    "orchestrator._check_thesis_valid "
                    "(iron_condor/iron_butterfly are exempt from the check "
                    "entirely, so this only steers directional strategies).")
    thesis_regime_shift_confidence: float = Field(default=0.70, ge=0.50, le=0.99,
        description="W69: the thesis check treats a regime shift (trending "
                    "-> high volatility) as invalidating only when the "
                    "regime classification confidence is above this. Was a "
                    "literal 0.70 in orchestrator._check_thesis_valid "
                    "(iron_condor/iron_butterfly are exempt from the check "
                    "entirely, so this only steers directional strategies).")
    thesis_vix_spike_pct: float = Field(default=0.30, ge=0.05, le=2.0,
        description="W69: the thesis check exits when VIX has risen more than "
                    "this fraction since entry (0.30 = +30%). Was a literal "
                    "0.30 in orchestrator._check_thesis_valid "
                    "(iron_condor/iron_butterfly are exempt from the check "
                    "entirely, so this only steers directional strategies).")
    order_visibility_grace_seconds: float = Field(default=10.0, ge=5.0, le=60.0,
        description="W70: a freshly placed EXIT order is ignored by the fill checker "
                    "while younger than this, so it has time to appear in IBKR's trade "
                    "list (a brand-new order would otherwise read as cancelled). Was a "
                    "literal 10 in executor.check_fills. Evaluated on the 30s tick. "
                    "Floor 5s is an engineering margin, not a measurement: the one "
                    "acknowledgement we could time took under 1s (2026-09-28, an entry, "
                    "delayed-data paper); no exit order has been timed yet (see the "
                    "order_ack_latency log event). Must stay below "
                    "exit_stop_pending_rest_seconds. IMPACT: too low = a not-yet-visible "
                    "order is misread as cancelled and the bot re-arms a duplicate "
                    "close; too high = a genuinely dead order is noticed later.")
    foreign_exit_escalation_seconds: float = Field(default=300.0, ge=60.0, le=3600.0,
        description="W70: an exit order that belongs to ANOTHER clientId cannot be "
                    "cancelled from this session, so once it has been working this long "
                    "the executor pages the operator instead of looping on a cancel "
                    "that cannot land. Was a literal 300 in executor.check_fills. "
                    "Must stay >= exit_pending_rest_seconds: paging before a normal "
                    "order would even have been cancelled just reports routine slowness. "
                    "IMPACT: lower = earlier, noisier pages; higher = a stuck "
                    "cross-session close sits unnoticed longer.")
    zombie_exit_revert_seconds: float = Field(default=900.0, ge=120.0, le=7200.0,
        description="W70: the zombie cap. An exit order that is still unresolved this "
                    "long after placement is checked against the broker and, only if "
                    "the broker answers and does not list it, the position is reverted "
                    "CLOSING->FILLED so a fresh exit can fire (otherwise it is held "
                    "CLOSING and the operator is paged). Was a literal 900 in "
                    "executor.check_fills. It is evaluated only after the cancel was "
                    "requested at exit_pending_rest_seconds, so the window between "
                    "them is how long a cancel is given to resolve; startup requires "
                    "that window to be at least as long as the longer rest "
                    "(cap >= 2 x max(exit_pending_rest_seconds, "
                    "exit_stop_pending_rest_seconds)). IMPACT: too low = re-arms before "
                    "the real outcome can show (a second close alongside the first: "
                    "both fill and the condor flips into an inverse structure); too "
                    "high = a dead order wedges the position in CLOSING longer.")

    @model_validator(mode="after")
    def _exit_durations_are_ordered(self) -> "ExitConfig":
        if self.exit_stop_pending_rest_seconds > self.exit_pending_rest_seconds:
            raise ValueError(
                f"exit.exit_stop_pending_rest_seconds ({self.exit_stop_pending_rest_seconds}) "
                f"must not exceed exit.exit_pending_rest_seconds "
                f"({self.exit_pending_rest_seconds}): a stop exit must be re-priced no "
                "later than an ordinary one.")
        if self.order_visibility_grace_seconds >= self.exit_stop_pending_rest_seconds:
            raise ValueError(
                f"exit.order_visibility_grace_seconds ({self.order_visibility_grace_seconds}) "
                f"must be below exit.exit_stop_pending_rest_seconds "
                f"({self.exit_stop_pending_rest_seconds}): the grace would swallow the rest.")
        if self.foreign_exit_escalation_seconds < self.exit_pending_rest_seconds:
            raise ValueError(
                f"exit.foreign_exit_escalation_seconds ({self.foreign_exit_escalation_seconds}) "
                f"must be >= exit.exit_pending_rest_seconds ({self.exit_pending_rest_seconds}): "
                "paging before a normal order would have been cancelled is just noise.")
        floor = 2 * max(self.exit_pending_rest_seconds, self.exit_stop_pending_rest_seconds)
        if self.zombie_exit_revert_seconds < floor:
            raise ValueError(
                f"exit.zombie_exit_revert_seconds ({self.zombie_exit_revert_seconds}) must be "
                f">= 2 x the longer rest time = {floor}: the cancel needs at least as long "
                "to resolve as the order was given to fill, otherwise a fill that is "
                "merely not yet visible is re-armed as a duplicate close. Raise it, or "
                "lower exit_pending_rest_seconds / exit_stop_pending_rest_seconds.")
        return self


class LearningAnalyzerConfig(_StrictModel):
    """W69: the closed-trade analyzer's insight cutoffs (learning/analyzer.py).

    These decide when a pattern in recent closes becomes an insight the
    adaptor may act on. Minimum-sample counts (3 / 5 trades) stay in code.
    Defaults are the pre-W69 literals.
    """

    strategy_poor_win_rate: float = Field(default=0.35, ge=0.05, le=0.60,
        description="A strategy (>=5 closes) below this win rate gets a "
                    "disable_<strategy> insight.")
    strategy_strong_win_rate: float = Field(default=0.65, ge=0.50, le=0.95,
        description="A strategy (>=5 closes, positive P&L) above this win "
                    "rate gets a boost_<strategy> insight.")
    confidence_band_poor_win_rate: float = Field(default=0.40, ge=0.05, le=0.60,
        description="A confidence band (>=3 closes, negative avg P&L) below "
                    "this win rate suggests raising min_confidence.")
    symbol_poor_win_rate: float = Field(default=0.30, ge=0.05, le=0.60,
        description="A symbol (>=3 closes, negative P&L) below this win rate "
                    "gets a remove_symbol_<sym> insight.")
    regime_poor_win_rate: float = Field(default=0.35, ge=0.05, le=0.60,
        description="A market regime (>=3 closes, negative P&L) below this "
                    "win rate gets a reduce_trading_in_<regime> insight.")
    hour_poor_win_rate: float = Field(default=0.30, ge=0.05, le=0.60,
        description="An entry hour (>=5 closes, negative P&L) below this win "
                    "rate gets a block_hour_<h> insight.")
    exit_giveback_threshold: float = Field(default=0.30, ge=0.05, le=1.0,
        description="Average peak-minus-exit above this (with exit_min_avg_peak) "
                    "suggests a tighter trailing stop.")
    exit_min_avg_peak: float = Field(default=0.20, ge=0.0, le=1.0,
        description="Minimum average peak for the giveback insight to apply.")
    direction_peak_threshold: float = Field(default=0.05, ge=0.0, le=1.0,
        description="High-water mark at/above this counts a trade's direction "
                    "as at least partially correct.")
    direction_good_accuracy: float = Field(default=0.60, ge=0.50, le=0.95,
        description="Direction accuracy above this (with "
                    "direction_exit_problem_rate) blames exit management.")
    direction_exit_problem_rate: float = Field(default=0.40, ge=0.05, le=0.95,
        description="Share of correct-direction trades that still lost, above "
                    "which exit management is flagged.")
    direction_poor_accuracy: float = Field(default=0.45, ge=0.05, le=0.60,
        description="Direction accuracy below this flags the model for retrain.")


class LearningConfig(_StrictModel):
    enabled: bool = True
    lookback_days: int = Field(default=30, ge=7, le=180)
    max_adaptations_per_cycle: int = Field(default=3, ge=1, le=10,
        description="W69: now actually read — LearningEngine builds its "
                    "AdaptationLimits from these four keys (they were "
                    "silently ignored; the dataclass defaults ran instead).")
    min_insight_confidence: float = Field(default=0.60, ge=0.40, le=0.95)
    min_confidence_floor: float = Field(default=0.50, ge=0.40, le=0.80)
    min_confidence_ceiling: float = Field(default=0.90, ge=0.70, le=0.99)
    analyzer: LearningAnalyzerConfig = LearningAnalyzerConfig()
    paper_trading_mode: bool = Field(
        default=False,
        description=(
            "When True, disables all live-only overlays (adaptor confidence/stop/trailing/"
            "take-profit overrides, Thompson sampling reranking, meta-labeler gate, options flow gate) "
            "to produce a clean backtest-equivalent paper-trading run for P&L comparison."
        ),
    )


class TelegramConfig(_StrictModel):
    enabled: bool = Field(default=True,
        description="W39: the ONLY switch here that does anything. It gates "
                    "the notifier at its single construction site in main.py; "
                    "the effective state is `enabled and bool(token and "
                    "chat_id)`. Turning it off also stops the manual-position "
                    "POLLER, so a HALT_UNTRACKED prompt can no longer be "
                    "answered and entries stay frozen — the warning at startup "
                    "says so.\n"
                    "The four `send_trades` / `send_errors` / "
                    "`send_daily_summary` / `send_circuit_breaker` flags that "
                    "used to sit beside this were DELETED rather than wired: "
                    "nothing in src/ ever read them, so an operator setting "
                    "one to false changed nothing. Wiring them would have "
                    "added four new fail-open decisions on alert paths, which "
                    "is the wrong direction for this programme — an alert that "
                    "silently does not fire is the R8 class. If per-category "
                    "muting is ever wanted it needs its own wave, fail-closed, "
                    "with tests per path.")


class NotificationsConfig(_StrictModel):
    telegram: TelegramConfig = TelegramConfig()


class DashboardConfig(_StrictModel):
    enabled: bool = Field(default=True,
        description="W69: now read by the supervisor — false skips launching "
                    "the Streamlit analytics dashboard (and its health-check "
                    "respawn). It did nothing before. The log viewer and "
                    "status page are separate and unaffected.")
    port: int = Field(default=8501, ge=1024, le=65535,
        description="W69: the Streamlit analytics dashboard port, read by "
                    "orchestration/master.py. It was hardcoded 8501 there and "
                    "this key was silently ignored.")
    weblog_port: int = Field(default=8502, ge=1024, le=65535,
        description="W69: the web_logs.py log-viewer port (was a literal in "
                    "web_logs.py and master.py).")
    status_port: int = Field(default=8503, ge=1024, le=65535,
        description="W69: the status_server.py live-status/scorecard port "
                    "(was a literal in status_server.py and master.py).")

    @model_validator(mode="after")
    def _ports_are_distinct(self) -> "DashboardConfig":
        ports = {"port": self.port, "weblog_port": self.weblog_port,
                 "status_port": self.status_port}
        if len(set(ports.values())) != len(ports):
            raise ValueError(
                f"dashboard ports must be distinct (three local servers share one "
                f"host): {ports}")
        return self


class LoggingConfig(_StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "logs/ait.log"
    max_bytes: int = Field(default=10_485_760, ge=1_048_576)
    backup_count: int = Field(default=5, ge=1, le=20)


class SizingProbeConfig(_StrictModel):
    """W31: the synthetic chain the sizing self-test builds to run the REAL
    condor builder against.

    These are SELF-TEST FIXTURES, not trading values, and they live in their
    own section for that reason: nothing in the entry, exit or backtest path
    reads them, and putting them under `backtest:` would invite a future
    `_resolve_setting` call to pull a fixture into the real engine. They are
    in config.yaml rather than buried as literals because the self-test's
    verdict depends on them — an operator who cannot see the probe cannot
    judge the answer (the R19c argument, applied to a diagnostic).
    """

    underlying_price: float = Field(default=660.0, ge=1.0, le=10_000.0,
        description="Probe spot, near the live universe's largest name so the "
                    "structure the builder returns resembles what it really "
                    "builds. The $1 price this replaced drove the vol term "
                    "under the builder's own $2 floor, so EVERY account was "
                    "sized from the narrowest structure the builder can make "
                    "(R26 F4: `builder_width` was bit-identical at $9.75k and "
                    "$201k, and the breaker arm compared against a fixed $130).")
    implied_vol: float = Field(default=0.20, ge=0.01, le=3.0,
        description="Probe IV. Wing width scales with it, so this sets how "
                    "large a structure the self-test believes the builder can "
                    "emit; the breaker arm is sized from the result.")
    dte: int = Field(default=21, ge=1, le=365,
        description="Probe days to expiry — the builder's own fallback DTE.")
    strike_grids: list[float] = Field(default=[1.0, 5.0],
        description="Strike spacings to probe, in dollars. BOTH matter: a $5 "
                    "grid is what reproduces R26 F4's silent case, where "
                    "snapping widens the structure past the budget and the "
                    "builder emits zero while the old check reported "
                    "problems=0. A $1 grid alone cannot see it.")
    iv_rank: float = Field(default=55.0, ge=0.0, le=100.0,
        description="IV rank handed to the builder. Must clear "
                    "backtest.ic_iv_rank_floor or the builder refuses before "
                    "constructing anything and the self-test measures the "
                    "floor instead of the cap grid.")


class ParityConfig(_StrictModel):
    """W72: tolerances for the research-vs-reality parity replay.

    ``scripts/parity_replay.py`` runs known real trades (the operator's manual
    trades plus the bot's own fills) through the research engine's pricing code
    and compares modelled entry credit / exit value with what actually filled.
    These numbers decide whether a DTE band's research is trustworthy enough to
    run an Optuna study on, so they live in config (research overlay), not in
    the script. A band that fails is reported FAIL; a band with fewer trades
    than ``min_trades_per_band`` is reported INSUFFICIENT, never PASS.
    """

    credit_error_pct_max: float = Field(default=0.15, gt=0.0, le=1.0,
        description="Median |modelled - actual| entry credit, as a fraction of "
                    "the actual credit, above which a DTE band FAILS parity. "
                    "Default 0.15: a condor credit that is 15% off is already "
                    "the size of the edge at 1-2 DTE (credits there are 8-25% "
                    "of wing width). Operator-owned; tighten it as the model "
                    "improves.")
    credit_bias_pct_max: float = Field(default=0.10, gt=0.0, le=1.0,
        description="|median signed error| / actual credit above which a band "
                    "FAILS. Separate from the spread check on purpose: a model "
                    "that is consistently 8% rich passes the spread test and "
                    "still manufactures a fake edge in every study.")
    min_trades_per_band: int = Field(default=3, ge=1, le=1000,
        description="A DTE band with fewer closed trades than this is reported "
                    "INSUFFICIENT instead of PASS/FAIL. The replay set is small "
                    "(tens of trades), so small bands are common and a verdict "
                    "on two trades would be noise presented as a result.")
    dte_band_upper_edges: list[int] = Field(default=[2, 7, 14, 45],
        description="Inclusive upper edges (calendar DTE at entry) of the "
                    "parity bands: [2, 7, 14, 45] gives 0-2, 3-7, 8-14, 15-45 "
                    "and a trailing 46+ catch-all. Matches the DTE regimes "
                    "whose pricing and friction assumptions differ most.")

    @field_validator("dte_band_upper_edges")
    @classmethod
    def _edges_strictly_ascending(cls, v: list[int]) -> list[int]:
        if not v or any(e < 0 for e in v) or any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("dte_band_upper_edges must be non-empty, >= 0 and strictly ascending")
        return v


class DataCollectionConfig(_StrictModel):
    """W76: scheduled data collection run by the SUPERVISOR (research data only).

    The short-DTE friction collector (``scripts/calibrate_option_spreads.py --source ib-bars
    --short-dte``) must run once per RTH session, after the close, and IB keeps nothing for an
    expired contract, so a day nobody runs it is a day of data lost for good. With
    ``short_dte_enabled`` the supervisor runs it itself on weekdays. It only requests
    historical bid/ask bars: no orders, a separate client id, no interaction with the bot.
    Default OFF: enable it after one successful manual ``--dry-run`` (docs/RUNBOOK.md).
    """

    short_dte_enabled: bool = Field(default=False,
        description="Run the 0-6 DTE friction collector from the supervisor on weekdays. The "
                    "supervisor and TWS/Gateway must be running at that time; a day the box is "
                    "off is skipped (nothing is lost but the day). Takes effect at the next "
                    "supervisor restart (a process runs on the settings it started with).")
    short_dte_time_et: str = Field(default="15:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="ET wall-clock time of the daily run, HH:MM. MEASURED 2026-10-13: once the "
                    "session ends IB no longer serves that day's expiring contracts, so a run "
                    "after the close can never capture DTE 0 (a 16:10 run listed expiries from "
                    "the NEXT day). 15:30 is late enough that the entry window's last bar "
                    "(14:30-15:30) has ended, so one run holds the whole expiry day, and early "
                    "enough that the DTE-0 contracts (requested first, ~6 minutes) are done "
                    "~20 minutes before the close even if the start is delayed. Running later "
                    "adds no bar (the 15:30 bar is outside the entry window).")
    short_dte_symbols: list[str] = Field(default=["QQQ"], min_length=1, max_length=5,
        description="Underlyings to collect. Each costs ~30-55 minutes at IB's historical-data "
                    "pacing (60 requests per 10 minutes), run sequentially.")
    short_dte_client_id: int = Field(default=91, ge=1, le=999_999,
        description="IB API client id for the collector. Must differ from IBKR_CLIENT_ID and its "
                    "fallback ladder (base+100..base+104); the job refuses to run otherwise.")
    short_dte_max_dte: int = Field(default=6, ge=1, le=14,
        description="Furthest expiry (days) to sample. 6 matches the friction bands (0-2, 3-6).")
    short_dte_duration: str = Field(default="7 D", pattern=r"^\d{1,3} [DW]$",
        description="History requested per contract. A contract's last week is all a 0-6 DTE "
                    "sample can come from, so 7 D is enough; re-sampled bars are free "
                    "(INSERT OR IGNORE).")
    lifecycle: bool = Field(default=False,
        description="W82: capture each contract's LIFE instead of 'every contract <= "
                    "short_dte_max_dte, every day'. Stage A = expiring today..+"
                    "lifecycle_expiring_days (the perishable part, DTE 0 first); stage B = "
                    "far-listed expiries at 15-21 DTE whose DTE 31-45 tail is not stored, capped "
                    "per run. Each request asks lifecycle_duration of hourly bars, so a run also "
                    "recovers the preceding month. About 20-35 minutes a day versus 72-108 for "
                    "0-14 / 0-45 DTE every day, and every contract is downloaded about twice in "
                    "its life instead of 5-11 times (PLAN.md 2026-10-16 design). Default off: "
                    "the original behaviour. short_dte_max_dte is ignored when on.")
    lifecycle_duration: str = Field(default="30 D", pattern=r"^\d{1,3} [DW]$",
        description="History requested per contract in lifecycle mode. 30 D is what IB serves "
                    "for hourly BID/ASK bars of a listed option (the all-DTE sampler has used it "
                    "since W66).")
    lifecycle_expiring_days: int = Field(default=1, ge=0, le=5,
        description="Stage A covers expiries today..today+N. 1 (default) means every contract "
                    "is requested on its last day AND the day before, so one missed day costs "
                    "at most that day's bars.")
    lifecycle_thin_every: int = Field(default=5, ge=1, le=10,
        description="Plan strikes from every Nth session of the underlying's path. MEASURED "
                    "2026-10-15 on the QQQ chain: 5 gives about 55 contracts per expiry, 3 about "
                    "80, 1 about 140 (each costs ~22 s at IB's pacing).")
    lifecycle_incremental_sessions: int = Field(default=3, ge=1, le=10,
        description="An expiry captured before (typically the day before) is planned over its "
                    "newest N sessions only, unthinned: its last day adds little else, so the "
                    "second capture costs about a third of the first.")
    lifecycle_max_far_contracts: int = Field(default=120, ge=0, le=400,
        description="Stage-B contract cap per run (120 is about 43 minutes at IB's pacing); "
                    "the first owed expiry is always taken so a backlog drains, the rest is "
                    "deferred to the next run.")
    short_dte_timeout_minutes: int = Field(default=150, ge=10, le=600,
        description="Hard wall-clock cap for one run; the job kills the process after this.")
    short_dte_idle_minutes: int = Field(default=20, ge=2, le=120,
        description="Kill the run if it prints nothing for this long (progress is printed every "
                    "10 contracts, about 3.6 minutes apart, so silence means a hang).")

    @field_validator("short_dte_symbols")
    @classmethod
    def _symbols_are_tickers(cls, v: list[str]) -> list[str]:
        out = [x.strip().upper() for x in v]
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.]{0,5}", x) for x in out):
            raise ValueError(f"short_dte_symbols must be tickers, got {v}")
        if len(set(out)) != len(out):
            raise ValueError("short_dte_symbols must not repeat a ticker")
        return out


class IBKREnvConfig(BaseSettings):
    """IBKR connection settings from environment variables."""

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_account: str = ""

    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False, "extra": "ignore"}


class APIKeysConfig(BaseSettings):
    """API keys from environment variables — never stored in config.yaml."""

    polygon_api_key: str = ""
    finnhub_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = {"env_file": ".env", "env_prefix": "", "case_sensitive": False, "extra": "ignore"}


class Settings(_StrictModel):
    """Root configuration — validated on startup."""

    trading: TradingConfig = TradingConfig()
    account: AccountConfig = AccountConfig()
    positions: PositionConfig = PositionConfig()
    position_sizing: PositionSizingConfig = PositionSizingConfig()
    risk: RiskConfig = RiskConfig()
    options: OptionsConfig = OptionsConfig()
    backtest: BacktestConfig = BacktestConfig()
    ml: MLConfig = MLConfig()
    regime: RegimeConfig = RegimeConfig()
    meta_label: MetaLabelConfig = MetaLabelConfig()
    earnings: EarningsConfig = EarningsConfig()
    dividends: DividendConfig = DividendConfig()
    assignment_response: AssignmentResponseConfig = AssignmentResponseConfig()
    condor_defense: CondorDefenseConfig = CondorDefenseConfig()
    exit: ExitConfig = ExitConfig()
    learning: LearningConfig = LearningConfig()
    sentiment: SentimentConfig = SentimentConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()
    sizing_probe: SizingProbeConfig = SizingProbeConfig()
    parity: ParityConfig = ParityConfig()  # W72: research-vs-reality replay tolerances
    data_collection: DataCollectionConfig = DataCollectionConfig()  # W76: supervisor-run collection

    # Loaded from environment
    ibkr: IBKREnvConfig = IBKREnvConfig()
    api_keys: APIKeysConfig = APIKeysConfig()

    @model_validator(mode="after")
    def _per_trade_knobs_are_linked(self) -> "Settings":
        """W64 + W86: link the live per-trade knobs (registered W25 ADDENDUM / W31 A3).

        risk.max_position_risk_pct funds the condor builder and gate 6b;
        positions.max_position_pct is the position sizer's hard cap. For a
        defined-risk condor both bound the SAME number (one structure's max
        loss), at different layers. If the gate cap exceeds the sizer cap, gate
        6b passes a structure the sizer then refuses with 0 contracts, one
        routine-looking verdict at a time, and entries brick silently. A startup
        error is the same philosophy as extra="forbid"; the coherence check's A3
        arm stays as the runtime guard.
        """
        if self.positions.max_position_pct < self.risk.max_position_risk_pct:
            raise ValueError(
                f"positions.max_position_pct ({self.positions.max_position_pct}) must be "
                f">= risk.max_position_risk_pct ({self.risk.max_position_risk_pct}): the "
                "sizer's cap would refuse structures the risk gate passes (W64 link)")
        # W86: the other two caps one structure's max loss must also fit under. Gate 6b-2
        # (positions.max_portfolio_risk_pct, open + new risk) and gate 6c
        # (risk.max_symbol_concentration_pct) were never tied to the per-trade budget, so a
        # raised budget could exceed either and every entry would be refused, one routine
        # rejection at a time. Same philosophy as the sizer link: a startup error.
        if self.positions.max_portfolio_risk_pct < self.risk.max_position_risk_pct:
            raise ValueError(
                f"positions.max_portfolio_risk_pct ({self.positions.max_portfolio_risk_pct}) must be "
                f">= risk.max_position_risk_pct ({self.risk.max_position_risk_pct}): gate 6b-2 would "
                "refuse every entry (W86 link)")
        if self.risk.max_symbol_concentration_pct < self.risk.max_position_risk_pct:
            raise ValueError(
                f"risk.max_symbol_concentration_pct ({self.risk.max_symbol_concentration_pct}) must be "
                f">= risk.max_position_risk_pct ({self.risk.max_position_risk_pct}): gate 6c would "
                "refuse every entry (W86 link)")
        return self


def resolve_position_size_pct(explicit, settings) -> float:
    """W64: the per-trade budget fraction research uses.

    explicit arg > backtest.position_size_pct (a deliberate sizing study) >
    risk.max_position_risk_pct (what live funds the builder with) >
    RiskConfig's own default. `settings` may be None or a partial test stub.
    """
    if explicit is not None:
        return float(explicit)
    bt = getattr(settings, "backtest", None)
    if bt is not None and getattr(bt, "position_size_pct", None) is not None:
        return float(bt.position_size_pct)
    rk = getattr(settings, "risk", None)
    if rk is not None and getattr(rk, "max_position_risk_pct", None) is not None:
        return float(rk.max_position_risk_pct)
    return float(RiskConfig().max_position_risk_pct)


def resolve_account_capital(explicit, settings) -> float:
    """W90a: the total starting equity a research object uses.

    explicit arg > the loaded ``account.capital`` (the frozen research view, so an overlay or a study reaches it) >
    ``AccountConfig``'s own default. Replaces the stale $10,000 / $50,000 / $100,000 constructor defaults that duplicated
    ``account.capital`` and drifted from it. ``settings`` may be None or a partial test stub."""
    if explicit is not None:
        return float(explicit)
    acct = getattr(settings, "account", None)
    if acct is not None and getattr(acct, "capital", None) is not None:
        return float(acct.capital)
    return float(AccountConfig().capital)


def loaded_account_capital() -> float:
    """``resolve_account_capital(None, <the loaded settings>)`` for dataclass ``default_factory``s, which have no settings handle."""
    try:
        return resolve_account_capital(None, load_settings())
    except Exception:  # noqa: BLE001 - no config.yaml -> AccountConfig's default
        return float(AccountConfig().capital)


def loaded_cash_yield() -> float:
    """W90a: the annual yield credited to idle cash in the DISPLAY-only cash-drag-adjusted return: ``options.risk_free_rate``
    (a T-bill proxy). Was a literal 0.05 in two result classes."""
    try:
        return float(load_settings().options.risk_free_rate)
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema default
        return float(DEFAULT_RISK_FREE_RATE)


def loaded_ml_config() -> "MLConfig":
    """W90d: ``ml`` of the loaded settings (the frozen research view inside a research process), else the schema defaults.
    The one resolver the range / vol-magnitude predictors and the hyperparameter helpers use for their live-shared
    constants (label thresholds, CV geometry, class-balance limits)."""
    try:
        return load_settings().ml
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return MLConfig()


def loaded_surface_tuning() -> "SurfaceTuningConfig":
    """W90f: ``backtest.surface_tuning`` of the loaded settings (the frozen research view in a research process), else defaults."""
    try:
        return load_settings().backtest.surface_tuning
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return SurfaceTuningConfig()


def loaded_friction_fit() -> "FrictionFitConfig":
    """W90f: ``backtest.friction_fit`` of the loaded settings, else the schema defaults."""
    try:
        return load_settings().backtest.friction_fit
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return FrictionFitConfig()


def loaded_engine_tuning() -> "EngineTuningConfig":
    """W90c: ``backtest.engine_tuning`` of the loaded settings (the frozen research view inside a research process),
    else the schema defaults."""
    try:
        return load_settings().backtest.engine_tuning
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return EngineTuningConfig()


def loaded_meta_label_neutrals() -> "MetaLabelNeutralsConfig":
    """W90c: ``ml.meta_label_neutrals`` of the loaded settings, else the schema defaults."""
    try:
        return load_settings().ml.meta_label_neutrals
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return MetaLabelNeutralsConfig()


def resolve_max_contracts_per_trade(explicit, settings) -> int:
    """W64: live's hard per-trade contract cap (positions.max_contracts_per_trade)."""
    if explicit is not None:
        return int(explicit)
    pos = getattr(settings, "positions", None)
    if pos is not None and getattr(pos, "max_contracts_per_trade", None) is not None:
        return int(pos.max_contracts_per_trade)
    return int(PositionConfig().max_contracts_per_trade)


def resolve_dashboard_port(field: str, cli_port: int | None = None) -> int:
    """W69: the port a dashboard child process (web_logs.py, status_server.py) binds.

    explicit `--port` from the supervisor > config.yaml dashboard.<field> >
    DashboardConfig's own default. The supervisor passes --port from ITS frozen
    settings, so what it logs and health-checks is always what the child binds
    even if config.yaml was edited after the supervisor started. Run standalone
    (no --port) the script reads the file itself.
    """
    settings = None
    if cli_port is None:
        try:
            settings = load_settings()
        except Exception:  # noqa: BLE001 — a broken file must not stop a log viewer
            settings = None
    return int(resolve_config_value(cli_port, "dashboard", field, DashboardConfig, settings))


def resolve_config_value(explicit, section: str, field: str, fallback_cls, settings):
    """explicit > loaded settings.<section>.<field> > fallback_cls().<field>.

    R20b review follow-up: the "explicit arg > config > fallback-class-default"
    precedence was hand-duplicated at 4+ call sites (engine.py, walkforward.py,
    optimizer.py, the ML predictors, run_backtest.py's parity manifest) —
    moved here as the ONE shared implementation so a future change to the
    resolution semantics only needs to happen once.

    `settings` is an already-loaded Settings object, or None (load_settings()
    failed or was never attempted). `settings is None` is logged at WARNING —
    every config-backed knob silently reverting to its (sometimes stricter,
    sometimes looser) pydantic default because config.yaml went missing
    mid-run is exactly the class of silent divergence this whole PR exists to
    prevent, and load_settings()'s own divergence report never runs on this
    path (it's the last line inside a SUCCESSFUL load). A partial settings
    stub (missing this one section/field, e.g. a test fixture) stays silent —
    that's an intentional, narrower degradation tests rely on.
    """
    if explicit is not None:
        return explicit
    if settings is None:
        from ait.utils.logging import get_logger
        get_logger("config.settings").warning(
            "config_unavailable_using_fallback_default",
            section=section, field=field,
        )
        return getattr(fallback_cls(), field)
    try:
        return getattr(getattr(settings, section), field)
    except Exception:  # noqa: BLE001 — partial stub -> model default, silent
        return getattr(fallback_cls(), field)


# W35: the deploy-order hazard registered in W33. Long-lived processes re-read
# config.yaml at RUNTIME (executor.commission_estimate, options_chain's
# liquidity resolver, the range/vol predictors, the supervisor's first page)
# and validate it against the settings MODEL they imported at start. Every
# model is extra=forbid, so adding ONE key to config.yaml — which every wave
# with a new config home does — made load_settings() raise inside every
# running process until it restarted. It happened live on 2026-09-18 (11:48-
# 11:57 ET, position open). A process now freezes the settings it started
# with: later reads get a COPY of that snapshot and never touch the file.
# That is also the documented contract ("editing .env or config.yaml requires
# a process restart", options_chain.py) — until now only half true, because a
# process ran on its startup settings AND on whatever the file said later.
_FROZEN: "Settings | None" = None


def freeze_settings(settings: "Settings") -> None:
    """Make `settings` this PROCESS's configuration for the rest of its life.

    Called once by the two long-lived entry points (ait.main, the supervisor's
    main()). Tests, scripts and research never freeze, so they keep reading
    the file.
    """
    global _FROZEN
    _FROZEN = settings.model_copy(deep=True)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Overlay wins key by key; nested mappings merge, everything else (lists, scalars, an
    explicit null) replaces. Returns a new dict."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(config_path: str | Path = "config.yaml",
                  overlay_path: str | Path | None = None,
                  *, ignore_frozen: bool = False,
                  extra_overlays: "Sequence[dict] | None" = None) -> Settings:
    """Load settings from YAML file, with env var overrides for secrets.

    W68: `overlay_path` (research only, see ait/config/research.py) is deep-merged OVER the base file
    before validation, so an overlay key the schema does not know is a startup error exactly like a
    typo in config.yaml (extra="forbid"), and a later live change in the base flows into research.
    `ignore_frozen` reads the file(s) even when a process-wide snapshot exists: the research parity
    manifest uses it to show what the LIVE bot's config.yaml says next to the research overlay.
    W87b: `extra_overlays` are already-parsed dicts merged over the result in order (a study's
    ``settings:`` block); like the file overlay they may not set the env-only sections, and an unknown
    key is a validation error.
    """
    if _FROZEN is not None and not ignore_frozen:
        return _FROZEN.model_copy(deep=True)
    config_path = Path(config_path)

    # R5 audit: silently falling back to all-defaults gave a materially
    # different bot (default universe, meta-label ON, FinBERT ON -> crash
    # loop, 10 contracts/trade) whenever launched from the wrong cwd.
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()} — refusing to "
            "run on pydantic defaults. Launch from the repo root or pass an "
            "explicit --config path."
        )
    with open(config_path) as f:
        yaml_data = yaml.safe_load(f) or {}
    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        if not overlay_path.exists():
            raise FileNotFoundError(
                f"Research overlay not found: {overlay_path.resolve()} -- refusing to run "
                "research on the live config by accident.")
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f) or {}
        for forbidden in ("ibkr", "api_keys"):
            if forbidden in overlay:
                raise ValueError(f"the research overlay may not set {forbidden!r} (secrets "
                                 "come from the environment only)")
        yaml_data = _deep_merge(yaml_data, overlay)
    for extra in extra_overlays or ():
        for forbidden in ("ibkr", "api_keys"):
            if forbidden in extra:
                raise ValueError(f"a study may not set {forbidden!r} (secrets come from the environment only)")
        yaml_data = _deep_merge(yaml_data, extra)

    # IBKR and API keys come from environment only
    yaml_data["ibkr"] = IBKREnvConfig().model_dump()
    yaml_data["api_keys"] = APIKeysConfig().model_dump()

    settings = Settings.model_validate(yaml_data)
    _report_default_divergence(settings)
    _report_fixed_limits()
    _report_order_timeline(settings)
    return settings


def _report_order_timeline(settings: "Settings") -> None:
    """W70: log the EFFECTIVE order-lifecycle durations, tunable and fixed, in one event.

    The timeline is spread across two config sections and a few fixed values; this
    puts the whole picture in one log line so nobody has to reconstruct it. Never
    blocks startup.
    """
    try:
        from ait.config.invariants import BROKER_SNAPSHOT_TTL_SECONDS
        from ait.utils.cadence import FAST_MONITOR_CADENCE_SECONDS
        from ait.utils.logging import get_logger  # lazy: keep settings import-light
        ex, po = settings.exit, settings.positions
        get_logger("config.settings").info(
            "order_timeline_in_effect",
            fast_monitor_tick_seconds=FAST_MONITOR_CADENCE_SECONDS,
            fixed={"broker_snapshot_ttl_seconds": BROKER_SNAPSHOT_TTL_SECONDS},
            exit_order={
                "order_visibility_grace_seconds": ex.order_visibility_grace_seconds,
                "exit_stop_pending_rest_seconds": ex.exit_stop_pending_rest_seconds,
                "exit_pending_rest_seconds": ex.exit_pending_rest_seconds,
                "exit_band_ttl_seconds": ex.exit_band_ttl_seconds,
                "foreign_exit_escalation_seconds": ex.foreign_exit_escalation_seconds,
                "zombie_exit_revert_seconds": ex.zombie_exit_revert_seconds,
            },
            entry_order={
                "entry_ladder_step_seconds": po.entry_ladder_step_seconds,
                "partial_fill_cancel_seconds": po.partial_fill_cancel_seconds,
                "absent_order_grace_seconds": po.absent_order_grace_seconds,
                "entry_order_timeout_seconds": po.entry_order_timeout_seconds,
            },
            note="each age is evaluated on the fast-monitor tick, so a duration under "
                 "the tick takes effect at the next tick. Meaning of each: docs/RUNBOOK.md "
                 "'Order lifecycle timeline'.",
        )
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        pass


def _report_fixed_limits() -> None:
    """W70: log the limits that are deliberately NOT configurable, once per load.

    Sits beside `config_default_divergence` so a log reader sees both halves of
    "what the system will do": the tunables that differ from code defaults, and
    the fixed limits no config edit can change. Source of truth and the reason
    each one is fixed: ait/config/invariants.py (and docs/RUNBOOK.md, which a drift
    test keeps in step). Like its sibling it can never block startup.
    """
    try:
        from ait.config.invariants import invariant_values
        from ait.utils.logging import get_logger  # lazy: keep settings import-light
        limits = invariant_values()
        get_logger("config.settings").info(
            "fixed_limits_in_effect",
            count=len(limits),
            limits=limits,
            note="fixed in code on purpose (ait/config/invariants.py); not set by "
                 "config.yaml or .env. Why each is fixed: docs/RUNBOOK.md 'Fixed limits'.",
        )
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        pass


# R19 (user audit): the SECOND shadowing layer, after the env contract.
# Every module that builds a config model BARE — Backtester(), WalkForwardConfig(),
# the optimizer, the trainer — gets the pydantic Field default, NOT the
# config.yaml operating value. Those disagreed on 5 fields, so research
# validated a different bot than the one that trades.
#
# wing_k was ALIGNED (a divergent wing size is not "safer", it is a different
# strategy). The rest are deliberately left STRICTER in code than in yaml —
# a bare construction should fail safe, not inherit a loosened operating
# value — but the divergence must never again be SILENT. This reports it
# once at load, so `grep config_default_divergence` answers "is research
# running the same knobs as live?" in one line.
# R19 SECURITY: sections whose VALUES must never enter a log line. api_keys /
# ibkr carry the Finnhub + Polygon keys, the Telegram bot token and the account
# number — the exact class that put a live Finnhub key into 11 log files (R13
# #12, still pending rotation as U10). The divergence report exists to compare
# NUMERIC knobs; secrets are reported as names only, never values.
_SECRET_SECTIONS: frozenset[str] = frozenset({"api_keys", "ibkr"})
_SECRET_FIELD_HINTS: tuple[str, ...] = (
    "key", "token", "secret", "password", "account", "chat_id",
)
_DIVERGENCE_EXEMPT: frozenset[str] = frozenset()


# W86: names that contain a secret-looking token but are plain settings, so the divergence report
# must not hide them ("account" matched account.pdt_account_under_25k, a boolean that changes
# behaviour and was invisible in the report).
_NOT_SECRET: frozenset[str] = frozenset({"account.pdt_account_under_25k"})


def _is_secret(section_name: str, field_name: str) -> bool:
    if f"{section_name}.{field_name}" in _NOT_SECRET:
        return False
    if section_name in _SECRET_SECTIONS:
        return True
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SECRET_FIELD_HINTS)


def default_divergences(settings: "Settings") -> list[tuple[str, object, object]]:
    """Fields whose config.yaml value differs from the pydantic default.

    Returns (dotted_field, code_default, active_value). Used by the startup
    report and by the W86 class guard (tests/test_w86_divergence_and_concurrency.py)
    so no NEW divergence appears unclassified.

    W86: also covers list / tuple fields (universe, dte_range, strategies) and fields
    whose default is None but whose active value is set (backtest.position_size_pct).
    Still skipped: nested models, and fields with no default at all.
    """
    from pydantic import BaseModel as _BM
    from pydantic_core import PydanticUndefined as _Undef

    out: list[tuple[str, object, object]] = []
    for section_name, section in settings:
        model_fields = getattr(type(section), "model_fields", None)
        if not model_fields:
            continue
        for field_name, field in model_fields.items():
            dotted = f"{section_name}.{field_name}"
            if dotted in _DIVERGENCE_EXEMPT:
                continue
            if _is_secret(section_name, field_name):
                continue  # R19: names only for secrets — never values
            default = field.get_default(call_default_factory=True)
            if default is _Undef or isinstance(default, _BM):
                continue
            active = getattr(section, field_name, None)
            if isinstance(active, _BM):
                continue
            scalar = isinstance(default, (int, float, str, bool))
            listy = isinstance(default, (list, tuple))
            if default is None:
                if active is not None and isinstance(active, (int, float, str, bool)):
                    out.append((dotted, default, active))
            elif (scalar or listy) and active != default:
                out.append((dotted, default, active))
    return out


# W86: WHY each divergence exists and whether research is affected. The startup note used to say that
# every module building a config model bare "runs the code_default instead", which is no longer true:
# the research stack resolves its knobs from the LOADED settings (resolve_config_value; config.yaml,
# or the research overlay once a research entry point froze it). This registry says, per divergent
# field, which of four things it is, and tests/test_w86_divergence_and_concurrency.py fails when a
# divergence appears that is not classified here (same pattern as backtesting/gate_parity.py).
DIVERGENCE_LIVE_ONLY = "live_only"                  # only live code reads it; research is unaffected
DIVERGENCE_RESEARCH_LOADED = "research_reads_loaded"  # research reads the loaded value (no skew)
DIVERGENCE_DELIBERATE = "deliberate_research_difference"  # research differs on purpose; shown in the manifest
DIVERGENCE_DEAD = "dead"                            # nothing reads it
DIVERGENCE_CLASSES: dict[str, tuple[str, str]] = {
    "trading.universe": (DIVERGENCE_LIVE_ONLY, "research takes its symbols from the CLI / study arguments, else backtest.default_symbols"),
    "trading.max_daily_trades": (DIVERGENCE_LIVE_ONLY, "live orchestrator cap; the engine has no daily-trade cap"),
    "account.pdt_account_under_25k": (DIVERGENCE_LIVE_ONLY, "read only by risk/pdt_guard.py"),
    "positions.max_open_positions": (DIVERGENCE_RESEARCH_LOADED,
                                     "walk-forward pool cap reads the loaded value (allocate_pool)"),
    "positions.max_contracts_per_trade": (DIVERGENCE_RESEARCH_LOADED,
                                          "resolve_max_contracts_per_trade, shown in the parity manifest"),
    "risk.min_confidence": (DIVERGENCE_RESEARCH_LOADED,
                            "engine / walk-forward / optimizer resolve it from the loaded settings"),
    "options.dte_range": (DIVERGENCE_RESEARCH_LOADED, "the engine's entry DTE is options.dte_range[0] (loaded)"),
    "options.min_open_interest": (DIVERGENCE_LIVE_ONLY, "chain liquidity filter; research prices from the model"),
    "options.min_volume": (DIVERGENCE_LIVE_ONLY, "chain liquidity filter; research prices from the model"),
    "options.max_bid_ask_spread_pct": (DIVERGENCE_LIVE_ONLY,
                                       "chain / executor spread filter; research charges the friction model"),
    "options.strategies": (DIVERGENCE_RESEARCH_LOADED,
                           "W87a: run_backtest.py / run_optimizer.py default --strategies to this list "
                           "(config.resolve_universe); an explicit --strategies still wins"),
    "backtest.position_size_pct": (DIVERGENCE_RESEARCH_LOADED,
                                   "resolve_position_size_pct: explicit > this > risk.max_position_risk_pct"),
    "backtest.max_concurrent_positions": (DIVERGENCE_RESEARCH_LOADED,
                                          "W86: WalkForwardConfig / StrategyOptimizer resolve it from the "
                                          "loaded settings (was a hard-coded 1)"),
    "ml.entry_gates_enabled": (DIVERGENCE_DELIBERATE,
                               "research uses backtest.gates_mode (overlay: enforce); the manifest shows both"),
    "learning.paper_trading_mode": (DIVERGENCE_LIVE_ONLY, "read only by the live orchestrator"),
    "sentiment.enabled": (DIVERGENCE_DEAD, "retired subsystem; SentimentConfig is a tombstone"),
    "logging.max_bytes": (DIVERGENCE_LIVE_ONLY, "setup_logging in the live process"),
    "logging.backup_count": (DIVERGENCE_LIVE_ONLY, "setup_logging in the live process"),
}
DIVERGENCE_KINDS = frozenset({DIVERGENCE_LIVE_ONLY, DIVERGENCE_RESEARCH_LOADED, DIVERGENCE_DELIBERATE,
                              DIVERGENCE_DEAD})


def classify_divergences(diverged: list[tuple[str, object, object]]) -> dict[str, list[str]]:
    """{class: [dotted fields]} for a default_divergences() result; unregistered ones under
    "unclassified" (the guard test and the startup report both surface them)."""
    out: dict[str, list[str]] = {k: [] for k in sorted(DIVERGENCE_KINDS)}
    out["unclassified"] = []
    for dotted, _code, _active in diverged:
        kind = DIVERGENCE_CLASSES.get(dotted, ("unclassified", ""))[0]
        out.setdefault(kind, []).append(dotted)
    return out


def _report_default_divergence(settings: "Settings") -> None:
    try:
        diverged = default_divergences(settings)
        if diverged:
            from ait.utils.logging import get_logger  # lazy: keep settings import-light
            log = get_logger("config.settings")
            log.info(
                "config_default_divergence",
                count=len(diverged),
                fields={d: {"code_default": c, "active": a} for d, c, a in diverged},
                by_class=classify_divergences(diverged),
                note="config.yaml overrides these code defaults. W86: the research stack resolves "
                     "its knobs from the LOADED settings, so a divergence only makes research differ "
                     "from live when it is classed deliberate_research_difference (the parity manifest "
                     "shows it) or is 'unclassified' (nobody has looked yet: add it to "
                     "DIVERGENCE_CLASSES in config/settings.py). A module that builds a config model "
                     "BARE would still run the code default.",
            )
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        pass

