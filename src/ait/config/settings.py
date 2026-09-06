"""Pydantic-validated configuration for AIT.

All configuration is loaded from config.yaml with environment variable overrides.
Validation catches misconfigurations BEFORE the bot starts trading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class _StrictModel(BaseModel):
    """R13 (human-factors lens): every config model silently IGNORED unknown
    keys (pydantic default), so a one-letter typo in config.yaml silently ran
    the code default — executed proof: `max_contracts_per_trad` traded 10x
    size, `paper_trading_mod` re-enabled live-only overlays, all with zero
    warnings. Unknown keys are now a fail-loud config error at load time.
    SentimentConfig keeps `extra="allow"` deliberately (retired-subsystem
    tombstone keys may linger in old configs)."""

    model_config = {"extra": "forbid"}


class TradingConfig(_StrictModel):
    mode: Literal["paper", "live"] = "paper"
    universe: list[str] = ["SPY", "QQQ", "AAPL"]
    scan_interval_seconds: int = Field(default=300, ge=30, le=3600)
    max_daily_trades: int = Field(default=5, ge=1, le=50)
    trading_hours_only: bool = True


class AccountConfig(_StrictModel):
    pdt_protection: bool = True
    pdt_account_under_25k: bool = True


class PositionConfig(_StrictModel):
    max_open_positions: int = Field(default=5, ge=1, le=20)
    max_position_pct: float = Field(default=0.05, ge=0.01, le=0.25)
    max_portfolio_delta: float = Field(default=0.30, ge=0.05, le=1.0)
    max_portfolio_risk_pct: float = Field(default=0.20, ge=0.005, le=0.50,
        description="Aggregate capital-at-risk across ALL open positions as a "
                    "fraction of NLV. Was a phantom knob (never read) while the "
                    "real cap was hardcoded 0.20 in manager.py — now wired "
                    "(audit 2026-07-07 item 3.3). Default matches the "
                    "2026-06-30 operating value.")
    max_contracts_per_trade: int = Field(default=10, ge=1, le=100,
        description="Hard cap on contracts per trade. Low values (e.g. 1) keep "
                    "cost-per-trade minimal so the account can hold many more "
                    "concurrent positions — maximizes trade COUNT for learning.")


class RiskConfig(_StrictModel):
    max_daily_loss_pct: float = Field(default=0.02, ge=0.005, le=0.10)
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
    max_position_risk_pct: float = Field(default=0.03, ge=0.005, le=0.10,
        description="Per-trade max_loss cap as a fraction of NLV (was "
                    "hardcoded 0.03 in manager.py — audit item 3.3).")
    max_correlation: float = Field(default=0.75, ge=0.30, le=0.99,
        description="Correlation above which two symbols count as the same bet.")
    max_correlated_positions: int = Field(default=2, ge=1, le=8,
        description="Max simultaneous positions within one correlated cluster "
                    "(SPY/QQQ/IWM/DIA correlate ~0.95). Was a CorrelationGuard "
                    "code default — audit item 3.3.")
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
    skip_macro_events: bool = Field(default=True,
        description="R19c: macro-event protection (entry gates + rule-3d "
                    "flatten for undefined-risk shapes). Config home for "
                    "AIT_SKIP_MACRO_EVENTS ('1'=on). Protective default ON; "
                    "2026-07-08 user decision.")
    credit_vix_halt: float = Field(default=28.0, ge=15.0, le=60.0,
        description="No NEW credit entries when VIX is at/above this level — "
                    "cheap vol-regime brake for short-premium strategies.")

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


class OptionsConfig(_StrictModel):
    delta_range: list[float] = [0.20, 0.50]
    dte_range: list[int] = [14, 45]
    min_open_interest: int = Field(default=100, ge=0)
    min_volume: int = Field(default=50, ge=0)
    max_bid_ask_spread_pct: float = Field(default=0.10, ge=0.01, le=0.50)
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


class BacktestConfig(_StrictModel):
    initial_capital: float = Field(default=100_000.0, ge=1_000.0, le=10_000_000.0)
    position_size_pct: float = Field(default=0.05, ge=0.01, le=0.50,
        description="Fraction of capital risked per trade (max-loss basis).")
    wing_floor_dollars: float = Field(default=5.0, ge=0.50, le=50.0,
        description="Hard minimum spread width in dollars (safety floor only). "
                    "Wing sizing is now primarily driven by wing_k × expected_move.")
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
    ic_min_credit: float = Field(default=0.70, ge=0.10, le=5.0,
        description="R19c: absolute minimum total credit ($/share) for a "
                    "condor — config.yaml home for AIT_IC_MIN_CREDIT. At a 50% "
                    "TP the gross must clear ~3x round-trip costs.")
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
    iv_rank_rise_threshold: float = Field(default=0.30, ge=0.0, le=10.0,
        description="R20b: suppress iron-condor entry when IV rank rose more than "
                    "this over the last 10 days (Exp 20 veto). Values > 1 disable "
                    "the veto (IV rank is 0-1).")
    min_edge_over_baseline: float = Field(default=0.05, ge=0.0, le=1.0,
        description="R20b: minimum weighted CV edge over the base rate for the "
                    "range predictor to activate as an entry gate (Exp 28 quality "
                    "floor; 0.0 = always use the model).")


class MLConfig(_StrictModel):
    ensemble_weights: dict[str, float] = {"xgboost": 0.5, "lightgbm": 0.5}
    retrain_interval_days: int = Field(default=7, ge=1, le=30)
    lookback_days: int = Field(default=504, ge=60, le=2520)
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

    @field_validator("ensemble_weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return v


class MetaLabelConfig(_StrictModel):
    # R12-C: default flipped to False (was True). The meta-labeler was trained
    # on corrupted data and rejects everything; config.yaml has carried
    # `enabled: false` since that finding, so this only changes what happens
    # if the key is ever omitted — default-off is the safe direction.
    enabled: bool = False
    min_probability: float = Field(default=0.50, ge=0.30, le=0.80)
    retrain_with_primary: bool = True  # Retrain when primary model retrains


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
    # long since passed, or hide a real one. Budget is generous because the
    # fast monitor runs on a 30s cadence and quotes are cached for 15s.
    max_quote_staleness_seconds: float = Field(default=180.0, ge=30.0, le=900.0,
        description="Underlying quote older than this (by exchange tick time) "
                    "marks the touch stop's input DEGRADED — it then needs two "
                    "agreeing ticks to fire instead of one.")
    touch_confirm_ticks: int = Field(default=2, ge=1, le=5,
        description="Consecutive agreeing evaluations required before a touch "
                    "stop fires on a degraded/frozen quote. 1 = fire on a "
                    "single stale print (the pre-R14 behaviour).")


class LearningConfig(_StrictModel):
    enabled: bool = True
    lookback_days: int = Field(default=30, ge=7, le=180)
    max_adaptations_per_cycle: int = Field(default=3, ge=1, le=10)
    min_insight_confidence: float = Field(default=0.60, ge=0.40, le=0.95)
    min_confidence_floor: float = Field(default=0.50, ge=0.40, le=0.80)
    min_confidence_ceiling: float = Field(default=0.90, ge=0.70, le=0.99)
    paper_trading_mode: bool = Field(
        default=False,
        description=(
            "When True, disables all live-only overlays (adaptor confidence/stop/trailing/"
            "take-profit overrides, Thompson sampling reranking, meta-labeler gate, options flow gate) "
            "to produce a clean backtest-equivalent paper-trading run for P&L comparison."
        ),
    )


class TelegramConfig(_StrictModel):
    enabled: bool = True
    send_trades: bool = True
    send_errors: bool = True
    send_daily_summary: bool = True
    send_circuit_breaker: bool = True


class NotificationsConfig(_StrictModel):
    telegram: TelegramConfig = TelegramConfig()


class DashboardConfig(_StrictModel):
    enabled: bool = True
    port: int = Field(default=8501, ge=1024, le=65535)


class LoggingConfig(_StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "logs/ait.log"
    max_bytes: int = Field(default=10_485_760, ge=1_048_576)
    backup_count: int = Field(default=5, ge=1, le=20)


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
    risk: RiskConfig = RiskConfig()
    options: OptionsConfig = OptionsConfig()
    backtest: BacktestConfig = BacktestConfig()
    ml: MLConfig = MLConfig()
    meta_label: MetaLabelConfig = MetaLabelConfig()
    exit: ExitConfig = ExitConfig()
    learning: LearningConfig = LearningConfig()
    sentiment: SentimentConfig = SentimentConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()

    # Loaded from environment
    ibkr: IBKREnvConfig = IBKREnvConfig()
    api_keys: APIKeysConfig = APIKeysConfig()


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


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load settings from YAML file, with env var overrides for secrets."""
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

    # IBKR and API keys come from environment only
    yaml_data["ibkr"] = IBKREnvConfig().model_dump()
    yaml_data["api_keys"] = APIKeysConfig().model_dump()

    settings = Settings.model_validate(yaml_data)
    _report_default_divergence(settings)
    return settings


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


def _is_secret(section_name: str, field_name: str) -> bool:
    if section_name in _SECRET_SECTIONS:
        return True
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SECRET_FIELD_HINTS)


def default_divergences(settings: "Settings") -> list[tuple[str, object, object]]:
    """Fields whose config.yaml value differs from the pydantic default.

    Returns (dotted_field, code_default, active_value). Used by the startup
    report and by tests/test_r19_config_authority.py to pin that no NEW
    divergence appears unnoticed.
    """
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
            default = field.default
            if default is None or repr(default).startswith("PydanticUndefined"):
                continue
            active = getattr(section, field_name, None)
            if isinstance(default, (int, float, str, bool)) and active != default:
                out.append((dotted, default, active))
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
                note="config.yaml overrides these; any module constructing a "
                     "config model BARE (engine/walkforward/optimizer/trainer) "
                     "runs the code_default instead — research vs live skew.",
            )
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        pass

