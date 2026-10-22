"""Backtesting engine for simulating trading strategies on historical data.

Runs the trading loop day-by-day over an OHLCV DataFrame, applying
ML predictions (if available), strategy selection, risk rules,
and trade simulation with Black-Scholes options pricing.

Supports:
- Debit strategies: long_call, long_put, bull_call_spread, bear_put_spread
- Credit strategies: iron_condor, short_strangle (profit from theta decay)
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ait.backtesting.pricing import (
    OptionType,
    black_scholes_price,
    expected_move as _expected_move,  # W74: fractional-time expected move
    find_strike_by_delta,
    ic_leg_marks,  # W72: single owner of condor leg pricing (engine + parity replay)
    ic_leg_marks_from_ivs,  # W75: the same legs at explicit per-leg (surface) IVs
    realized_vol,
    skewed_leg_iv,
)
from ait.backtesting import iv_model  # W87a: the IV proxy constants, shared with the parity replay
from ait.backtesting import settlement  # W77: expiry settlement / assignment rules
from ait.config.invariants_research import (  # W88a: fixed research limits, one visible home
    IV_SCALE_MAX,
    IV_SCALE_MIN,
    RESEARCH_DELTA_MAX,
    RESEARCH_DELTA_MIN,
    SETTLEMENT_LOSS_TOLERANCE,
)
from ait.backtesting.result import BacktestResult
from ait.backtesting.ex_div import ex_div_blocks, sells_a_call  # W84: live's ex-div entry block
# R20 #5a: credit TP ladder / DTE close / macro-flatten windows come from the
# shared exit-policy module (pure, import-light) instead of a hand copy of
# live portfolio.py — a live tuning can no longer silently de-sync research.
from ait.execution import exit_policy
from ait.ml.regime_gate import apply_range_prob_delta, fractal_gate
from ait.strategies.base import CREDIT_STRATEGIES, SignalDirection
from ait.utils.logging import get_logger
from ait.utils.time import ET, get_market_close, time_to_expiry_years  # W74: real session clock
from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING  # W70: fixed, visible
from ait.config.runtime_env import contract_flag, contract_float  # R19: ONE authority for env-contract defaults

log = get_logger("backtesting.engine")


@lru_cache(maxsize=4096)
def _session_close(d: date) -> time:
    """W74: the NYSE session close for ``d`` (13:00 ET on early-close days)."""
    return get_market_close(d)

# Strategies that collect premium (short theta).
# R16: this module used to declare its OWN set under the same name with
# different membership from strategies/base.py — see the comment there. The
# membership now lives in base.py (live + backtest-only arms) and is imported,
# so the two can never drift apart again. The imported name stays module-level
# here, so existing `from ait.backtesting.engine import CREDIT_STRATEGIES`
# callers keep working unchanged.
def research_regime(vol_expanding: bool, px_vs_sma: float, down: float, up: float) -> str:
    """W81: the research engine's four-way day class from two features.

    ``vol_expanding`` is `vol_regime_expanding > 0.5` (5-day realized vol above 20-day) and
    ``px_vs_sma`` is `price_vs_sma_20`. ONE definition for the trending-down veto, the
    `entry_regime` stored on each position and the meta-label context (three sites that each
    carried the same bare -0.05 / 0.02). Research only: live's regime comes from
    `ait.ml.regime.RegimeDetector` with the `regime:` config block.
    """
    if not vol_expanding:
        return "range_bound"
    if px_vs_sma < down:
        return "trending_down"
    if px_vs_sma > up:
        return "trending_up"
    return "high_volatility"


# Strategies that pay premium (long theta)
DEBIT_STRATEGIES = {"long_call", "long_put", "bull_call_spread", "bear_put_spread", "long_straddle"}

# R16 #9: direction-neutral credit structures sharing the direction-gate
# bypass and the five neutral entry risk gates (hurst hard veto,
# trending-down regime veto, range-model gate, 10d realized-vol gate,
# IV-rank-rise veto). jade_lizard and call_credit_spread are backtest-only
# shadow arms selectable ONLY on NEUTRAL signals (_select_strategy), so they
# belong to the same entry population as the condor/strangle arms — they were
# previously missing from every tuple, which made those arms either
# zero-trade (blocked by the directional-confidence gate that NEUTRAL@0.4
# can never pass) or, when ML emitted NEUTRAL>=0.55, completely ungated.
NEUTRAL_CREDIT_GATED = (
    "iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor",
    "short_strangle", "jade_lizard", "call_credit_spread",
)


# R20b review follow-up: this used to be a private copy of the "explicit >
# config > fallback-class-default" precedence, duplicated independently in
# walkforward.py/optimizer.py/the ML predictors/run_backtest.py. Moved to
# ait.config.settings as the ONE shared implementation (also reused by those
# other call sites); re-imported under the original name so every existing
# `_resolve_setting(...)` call in this file needs no change.
from ait.config.settings import resolve_config_value as _resolve_setting


class Backtester:
    """Simulates trading strategies against historical OHLCV data."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategies: list[str],
        # R20b: initial_capital (and max_concurrent_positions below) stay
        # EXPLICIT constructor defaults per the pre-registration — they are
        # test-harness sizing knobs, not strategy economics: every research
        # entry point (walkforward, optimizer, run_backtest) passes its own
        # capital, and a $10k bare-engine default cannot bias a comparison
        # the way a divergent gate threshold can.
        initial_capital: float = 10_000.0,
        # W5 finding model-vs-reality-commission-constant: was a bare 0.65
        # literal that stood for the WHOLE per-leg cost. IBKR charges ~$0.65
        # commission PLUS regulatory/exchange/clearing fees on every option
        # leg, entry AND exit; the repo's own executions ledger measures the
        # all-in at 0.9168/contract-leg (mean; median 1.0284), so the flat
        # 0.65 understated a condor round trip by $2.13/contract (~41%).
        # None -> resolve from load_settings().backtest.* (see the resolution
        # block below); an explicit value still wins, so walkforward.py's
        # pinned 0.65 keeps its commission line and only gains the fees.
        commission_per_contract: float | None = None,
        regulatory_fees_per_contract: float | None = None,
        slippage_pct: float | None = None,   # W70: None -> backtest.slippage_pct
        position_size_pct: float = 0.05,
        # W64 live-parity switches, OFF for a bare Backtester (R20b: the bare
        # engine's sizing knobs stay explicit harness defaults; hundreds of engine
        # tests pin them). The research ENTRY POINTS (WalkForwardConfig,
        # StrategyOptimizer) resolve both from config and pass them explicitly.
        # budget_aware_wings: cap the condor wing to the per-trade budget like
        # live (IronCondor.affordable_width) and reject a structure whose max
        # loss exceeds it. max_contracts_per_trade: live's hard cap
        # (positions.max_contracts_per_trade, config 1); None = uncapped.
        budget_aware_wings: bool = False,
        max_contracts_per_trade: int | None = None,
        # W65: research macro calendar (EconomicCalendar built from the operator's CSV).
        # None = the hardcoded live calendar (legacy behaviour).
        economic_calendar: Any = None,
        # W66: live's IronCondor ignores market_direction entirely ("Iron condors work
        # in ANY direction"), but this engine only selects it when the day's direction
        # is NEUTRAL, which removed ~81% of days on the 2024-2026 research span. True =
        # an IC-only strategy set is selected on ANY direction (live behavior). False
        # (default) = the legacy NEUTRAL-only rule, so existing results do not move.
        condor_any_direction: bool = False,
        # W66 friction model v2. "legacy" = `half_spread_pct x leg_mid` per leg at entry and
        # the same % on the NET value at exit (the engine's historical behaviour, kept as
        # the bare default so engine-level tests/results do not move). "per_leg_usd" =
        # the fitted per-leg DOLLAR half-spread (option_leg_spread_params, from IB BID/ASK
        # bars) charged symmetrically on all four legs at entry AND exit; applies to the
        # iron condor only, and falls back to legacy (with a WARNING) when the symbol has
        # no fitted model. The research entry points resolve it from backtest.friction_*.
        friction_model: str = "legacy",
        friction_entry_capture: float | None = None,   # share of the quoted half-spread paid at entry (None -> backtest.*)
        friction_exit_capture: float | None = None,    # ... and at exit (1.0 = full, conservative; None -> backtest.*)
        friction_stress_multiplier: float | None = None,  # scales the model (calm-sample hedge/sweeps; None -> backtest.*)
        # W67 gate parity. gates_mode "enforce" (bare default, the engine's historical
        # behaviour) blocks entries on the range-model floor and the fractal hard veto;
        # "observe" mirrors live today (ml.entry_gates_enabled=false): both are still
        # COMPUTED and recorded (entry decision + `would_reject:*` funnel counters) but
        # never block. The research entry points resolve it from backtest.gates_mode.
        gates_mode: str = "enforce",
        # W68 IV-rank parity. Live gates credit entries on IV RANK (0-100 percentile of today's
        # IV in the last 252 stored daily IVs; ait.data.iv_rank): iron_condor refuses rank <
        # AIT_IRON_CONDOR_IV_FLOOR (15) and the risk manager refuses rank > AIT_IV_RANK_CAP (85);
        # an unmeasurable rank fails closed. False (bare default) = the engine's historical
        # absolute-level floor only; the research entry points resolve backtest.iv_rank_gate.
        iv_rank_gate: bool = False,
        # Precomputed rank by date (ait.data.iv_rank.iv_rank_series over the symbol's FULL IV
        # history), so a short test window still sees live's 252-observation lookback. None =
        # derived from this engine's own `implied_vol` column.
        iv_rank_series: pd.Series | None = None,
        # Absolute fractal hard-veto cutoff (0 = off). None = derive from the legacy
        # hurst_hard_veto_multiplier (max(threshold, 0.20) * multiplier) so explicit
        # callers keep their exact behaviour. See ait.ml.regime_gate.
        fractal_veto_spread: float | None = None,
        # Share of the fractal penalty that ALSO lowers the range model's P(in range)
        # before the range floor (0 = live parity: the range probability is untouched).
        penalty_on_range_prob: float | None = None,   # W70: None -> backtest.penalty_on_range_prob
        # W70 D (registered, NOT resolved): stop/target stay explicit harness defaults
        # (0.50 / 1.00) although config.yaml's research baselines are 0.35 / 0.50 — a
        # deliberate divergence the operator will revisit; see PLAN.md W70.
        stop_loss_pct: float = 0.50,
        profit_target_pct: float = 1.00,
        max_hold_days: int | None = None,      # W70: None -> backtest.max_hold_days
        # R20b (pre-registered PLAN 2026-08-21): was 0.55, silently shadowing
        # the config with a DIFFERENT value. None -> resolve from
        # load_settings().risk.min_confidence (yaml 0.50). See the resolution
        # block below for why risk.min_confidence is the documented home.
        min_confidence: float | None = None,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float | None = None,       # W70: None -> exit.trailing_stop_pct
        breakeven_trigger_pct: float | None = None,   # W70: None -> exit.breakeven_trigger_pct
        predictor: Any = None,
        range_predictor: Any = None,
        # R20b: was 0.55 vs live's ml.range_min_confidence = 0.65 (config.yaml;
        # 0.65 beat 0.55 across every backtest metric) — the parity gap the
        # floor sweep exposed. None -> resolve from load_settings().ml.
        range_min_confidence: float | None = None,
        context_bars: int | None = None,      # W70: None -> backtest.context_bars
        delta_short: float | None = None,     # W70: None -> backtest.delta_short
        delta_long: float | None = None,      # W70: None -> backtest.delta_long
        # R20b: was 0.12 vs config.yaml backtest.iv_floor = 0.20 — a bare
        # engine priced systematically thinner credits than every configured
        # path. None -> resolve from load_settings().backtest.iv_floor.
        iv_floor: float | None = None,
        # R6 parity: live iron_condor._vol_scaled_width enforces min $2, not $5.
        wing_floor_dollars: float | None = None,   # W70: None -> backtest.wing_floor_dollars (2.0)
        # R16 #8: None -> resolve env AIT_IC_WING_K (default 1.0), mirroring
        # live iron_condor.py:107 — same pattern as ic_min_credit_width below.
        wing_k: float | None = None,
        delta_iv_scale: float | None = None,   # W70: None -> backtest.delta_iv_scale
        skew_factor: float | None = None,      # W70: None -> backtest.skew_factor
        # W5 finding model-vs-reality-skew-10x-flat: measured put-side skew is
        # 9-11x steeper than the hardcoded 0.10 slope in _get_leg_iv, so model
        # wings were near-free ($0.08 vs $0.71-0.92 real). None -> resolve
        # from load_settings().backtest.skew_slope_per_pct_otm; 0.0 restores
        # pre-W5 pricing exactly.
        skew_slope_per_pct_otm: float | None = None,
        # W70: every knob in this block is None -> backtest.<same name> (the model
        # defaults equal the former literals 0.20 / 0.10 / 1.5 / 0.50 / 0.30 / -1.0 / 0.05).
        hurst_regime_threshold: float | None = None,
        hurst_regime_penalty: float | None = None,
        hurst_hard_veto_multiplier: float | None = None,
        multifractal_max_width: float | None = None,
        iv_rank_rise_threshold: float | None = None,
        iv_rank_rise_gate: bool | None = None,          # W71c: None -> backtest.iv_rank_rise_gate
        regime_veto_gate: bool | None = None,           # W81: None -> backtest.regime_veto_gate
        regime_down_threshold: float | None = None,     # W81: None -> backtest.regime_down_threshold
        regime_up_threshold: float | None = None,       # W81: None -> backtest.regime_up_threshold
        ex_div_entry_gate: str | None = None,           # W84: None -> backtest.ex_div_entry_gate
        ex_div_entry_k_days: int | None = None,         # W84: None -> backtest.ex_div_entry_k_days
        risk_free_rate: float | None = None,            # W81: None -> options.risk_free_rate
        pct_from_60d_high_threshold: float | None = None,
        min_edge_over_baseline: float | None = None,
        features_cache: pd.DataFrame | None = None,
        # R20b: stays EXPLICIT (with initial_capital above) per the
        # pre-registration — harness sizing knob; 1 = the original
        # single-position semantics unit tests depend on.
        max_concurrent_positions: int = 1,
        max_entry_vol_annual: float | None = None,   # W70: None -> backtest.max_entry_vol_annual
        # Options bid-ask spread model (per-leg, IV/DTE-aware). W70: None -> backtest.spread_*
        # (these four config keys were read by NOTHING before; the engine and the
        # walk-forward each carried their own literals).
        spread_base: float | None = None,
        spread_iv_sensitivity: float | None = None,
        spread_dte_sensitivity: float | None = None,
        spread_cap: float | None = None,
        # Cross-asset market context forwarded from walk-forward engine (Gap Z3)
        market_context: dict | None = None,
        # Earnings-aware skip (Gap Z4): if symbol is given, skip entries near earnings dates
        symbol: str | None = None,
        # W70 (parity): the earnings danger window is live's, from earnings.buffer_days_before /
        # buffer_days_after (None -> config). It replaces the private `earnings_skip_days=2`,
        # which blocked [earnings-1, earnings+2]; live blocks [earnings-2, earnings+1].
        earnings_buffer_days_before: int | None = None,
        earnings_buffer_days_after: int | None = None,
        # Intraday engine (Fix 1): 5-min execution loop.
        # R20 #4/#5b: None -> resolve from load_settings().backtest (config.yaml
        # authority; BacktestConfig defaults when no config is reachable). The
        # engine used to hardcode '09:30' while BacktestConfig declares '10:30'
        # — the documented Fix 1/Gap H parity value that
        # scripts/export_production_params.py reports as production parity —
        # so every intraday-gated study traded a window the config contract
        # says is forbidden, and the exported params lied about it. '10:30'
        # (skip open volatility) is the historically correct value; '09:30'
        # was constructor drift.
        intraday_store: Any = None,
        # W52a: the store this run reads (calibrated spreads today; the
        # walk-forward threads its own db_path here). None keeps the
        # cwd-relative default — which is how the 2026-09-22 study silently
        # ran on an empty auto-created historical.db from a dev worktree.
        db_path: Any = None,
        scan_interval_minutes: int | None = None,
        entry_window_start_et: str | None = None,
        entry_window_end_et: str | None = None,
        limit_order_timeout_bars: int | None = None,
        # Per-window MetaLabeler for OOS signal filtering (Gap Z1)
        meta_labeler: Any = None,
        # H2 val-split: skip new entries before this date (full df still used for feature warmup)
        eval_start_date: "date | None" = None,
        # --- R6 exit/construction parity with the live bot ---
        # Flat loss limit for CREDIT structures as a multiple of credit
        # received (live portfolio.py reads env AIT_CREDIT_LOSS_LIMIT with
        # default "0" = flat stop DISABLED — R12-B1 evidence: touch-close beat
        # every flat level). None -> resolve from env with the SAME default as
        # live (R16 #6 parity: this used to default to 1.25, so every backtest
        # simulated an active stop the live book does not run).
        credit_loss_limit_mult: float | None = None,
        # Iron condor construction gates (live iron_condor.py R6):
        # minimum mid-price total credit (env AIT_IC_MIN_CREDIT, $0.70) and
        # minimum credit/max-width ratio (env AIT_IC_MIN_CREDIT_WIDTH, 0.20).
        ic_min_credit: float | None = None,
        ic_min_credit_width: float | None = None,
        # W52a: the live expected-move band on condor short strikes
        # (iron_condor.py R12-B gate: each short must sit within
        # [em_min, em_max] x expected move of spot). None -> the same contract
        # keys live reads (AIT_IC_EM_MIN/MAX, home backtest.ic_em_min/max).
        # ON by default (AIT_BT_EM_GATE=0 or em_gate=False turns it off).
        # The engine's whole-dollar 0.20-delta placement sits at ~0.90-1.00
        # EM on the call side and ~0.68-0.73 on the put side; on the live
        # store every one of the 57 baseline condors is inside the band
        # (W52a: n 57 / PF 0.628 gate on and off, to the cent), so the
        # recorded study baselines do not move. On a $100 synthetic series
        # with EM under ~5 the $1 strike grid can push a put outside — the
        # gate then rejects, as live would. The ratios are recorded on
        # every position either way.
        ic_em_min: float | None = None,
        ic_em_max: float | None = None,
        em_gate: bool | None = None,
        # Macro-event gate: live orchestrator blocks CREDIT entries <=4 days
        # before FOMC/CPI/NFP/GDP/PCE; live portfolio flattens defined-risk
        # credit <=1 day before (strangles <=5) when AIT_SKIP_MACRO_EVENTS=1.
        # Uses the hardcoded-2026 calendar in ait.data.economic_calendar —
        # for pre-2026 backtest windows the gate is effectively inactive
        # (days-to-event is always > 4), which is a documented limitation.
        macro_event_gate: bool = True,
        # R16 #7: live reads pre_event_blackout_days from LOADED settings
        # (orchestrator.py: self._settings.risk.pre_event_blackout_days), so a
        # yaml override must reach the engine too. None -> resolve from
        # load_settings(), falling back to the RiskConfig default when no
        # config.yaml is reachable from the CWD.
        pre_event_blackout_days: int | None = None,
        # R16 #1 (look-ahead fence): when no predictor is supplied, standalone
        # runs may fall back to loading the LIVE models/ensemble.pkl. Walk-
        # forward OOS windows must NEVER do that (the live artifact is trained
        # on the very future periods being scored) — walkforward passes False,
        # and the window then trades UNGATED (predictor=None), which is the
        # engine's designed fallback and what the ablation studies claimed to
        # measure.
        allow_live_model_fallback: bool = True,
        # DTE/hold-cap decoupling fix (2026-08-24): entry_dte is the REAL
        # time-to-expiration used for BS pricing and decay, fixed near what
        # live actually gets (options_chain.py picks the nearest expiry >=
        # dte_range[0]) — it is NOT Optuna-searched. max_hold_days remains a
        # separate, deliberately-synthetic Optuna-searched hold-period cap
        # (live has no "close after N days held" rule; only DTE/touch/TP
        # exits) for exploring shorter/longer forced-hold horizons. Before
        # this fix both concepts were the SAME variable (self._max_hold_days
        # doubled as the option's simulated expiration), so every prior
        # walk-forward study's "optimal max_hold_days" was silently also
        # picking the simulated DTE. None -> resolve options.dte_range[0]
        # from load_settings() (config.yaml authority; falls back to
        # OptionsConfig's pydantic default when no config.yaml is reachable).
        entry_dte: int | None = None,
        # W74 short-DTE parity programme. All four are None -> backtest.<same name>
        # (schema defaults: 'legacy' / off / off / off), so a bare engine behaves as
        # before. engine_model='realistic' turns on the W72 lead fixes L2-L6; the
        # three exit knobs are independent of it. See BacktestConfig for the meaning.
        engine_model: str | None = None,
        pricing_source: str | None = None,     # W75: None -> backtest.pricing_source
        friction_short_dte: bool | None = None,                  # W76: None -> backtest.*
        friction_min_sessions_per_band: int | None = None,       # W76: None -> backtest.*
        friction_extrapolated_floor_usd: float | None = None,    # W76: None -> backtest.*
        expiry_guard_dte: int | None = None,
        max_hold_fraction: float | None = None,
        flatten_minutes_before_close: int | None = None,
        pin_band_usd: float | None = None,                       # W77: None -> backtest.*
        assignment_fee_per_share: float | None = None,           # W77: None -> backtest.*
        stock_commission_per_share: float | None = None,         # W77: None -> backtest.*
        # Research-only (scripts/post_event_study.py): block CREDIT entries for
        # this many calendar days AFTER an event date (day 0 = the event day
        # itself, when the decision lands intraday). None = off, which is the
        # engine's behaviour before this parameter existed. The live pre-event
        # gate above is untouched. `post_event_dates` is the event list this
        # gate uses — the study passes FOMC decision days 2022-2026, because
        # ait.data.economic_calendar only knows 2026-2027 and a buffer tested
        # on the 2026 windows alone would be a test of one year.
        post_event_blackout_days: int | None = None,
        post_event_dates: tuple | None = None,
        # Research-only (scripts/wing_budget_study.py, 2026-10-08): the live
        # builder's budget-aware wing cap (iron_condor._vol_scaled_width +
        # the max_loss > risk_budget reject), as a dollar budget per
        # contract. None = off (the engine's behaviour before this existed).
        max_loss_per_contract_dollars: float | None = None,
        # Research-only: cap contracts per entry (live trades 1 lot,
        # positions.max_contracts_per_trade). None = off.
        contracts_cap: int | None = None,
        # R20b follow-up: optional pre-loaded Settings, so a caller building
        # many Backtester instances (StrategyOptimizer: one per Optuna trial)
        # can load config.yaml ONCE and thread it through instead of every
        # construction re-reading and re-validating the file from disk.
        # None (default) preserves the original behavior — load it here.
        settings: Any = None,
        # Research-only credit take-profit ladder (scripts/take_profit_study.py
        # via WalkForwardConfig.credit_tp_ladder). None = the shared live
        # ladder in exit_policy.CREDIT_TP_LADDER; live never sets this.
        credit_tp_ladder: Any = None,
    ) -> None:
        # W70: load settings FIRST. Constructor knobs that have a config twin are None by
        # default and resolve here with the project-wide precedence explicit arg > loaded
        # settings.<section>.<field> > the config model's own default (resolve_config_value).
        # The model defaults equal the literals these parameters used to carry, so a bare
        # Backtester() is unchanged at today's config.yaml. Knobs deliberately kept explicit
        # (initial_capital, position_size_pct, max_concurrent_positions, stop_loss_pct,
        # profit_target_pct) are NOT here: see PLAN.md W70 "registered, not changed".
        if settings is not None:
            _settings = settings
        else:
            try:
                from ait.config.settings import load_settings as _ls
                _settings = _ls()
            except Exception as _e:  # noqa: BLE001 — no config.yaml -> per-field defaults
                _settings = None
                # W52a: this used to be silent. A run from the wrong cwd is
                # a different harness (every settings-backed knob at its
                # model default) and must say so.
                log.warning("engine_settings_unavailable_using_defaults",
                            cwd=os.getcwd(), error=str(_e))
        from ait.config.settings import (
            BacktestConfig, EarningsConfig, ExitConfig, MLConfig, OptionsConfig, RiskConfig,
        )
        _bt = lambda v, f: _resolve_setting(v, "backtest", f, BacktestConfig, _settings)  # noqa: E731
        slippage_pct = float(_bt(slippage_pct, "slippage_pct"))
        friction_entry_capture = float(_bt(friction_entry_capture, "friction_entry_capture"))
        friction_exit_capture = float(_bt(friction_exit_capture, "friction_exit_capture"))
        friction_stress_multiplier = float(_bt(friction_stress_multiplier, "friction_stress_multiplier"))
        penalty_on_range_prob = float(_bt(penalty_on_range_prob, "penalty_on_range_prob"))
        max_hold_days = int(_bt(max_hold_days, "max_hold_days"))
        trailing_stop_pct = float(_resolve_setting(
            trailing_stop_pct, "exit", "trailing_stop_pct", ExitConfig, _settings))
        breakeven_trigger_pct = float(_resolve_setting(
            breakeven_trigger_pct, "exit", "breakeven_trigger_pct", ExitConfig, _settings))
        context_bars = int(_bt(context_bars, "context_bars"))
        delta_short = float(_bt(delta_short, "delta_short"))
        delta_long = float(_bt(delta_long, "delta_long"))
        # W70: the SAME contract call live uses (AIT_IC_WING_FLOOR, home
        # backtest.wing_floor_dollars) — one key, one value, live and research.
        wing_floor_dollars = (float(wing_floor_dollars) if wing_floor_dollars is not None
                              else contract_float("AIT_IC_WING_FLOOR"))
        delta_iv_scale = float(_bt(delta_iv_scale, "delta_iv_scale"))
        skew_factor = float(_bt(skew_factor, "skew_factor"))
        hurst_regime_threshold = float(_bt(hurst_regime_threshold, "hurst_regime_threshold"))
        hurst_regime_penalty = float(_bt(hurst_regime_penalty, "hurst_regime_penalty"))
        hurst_hard_veto_multiplier = float(_bt(hurst_hard_veto_multiplier, "hurst_hard_veto_multiplier"))
        multifractal_max_width = float(_bt(multifractal_max_width, "multifractal_max_width"))
        iv_rank_rise_threshold = float(_bt(iv_rank_rise_threshold, "iv_rank_rise_threshold"))
        iv_rank_rise_gate = bool(_bt(iv_rank_rise_gate, "iv_rank_rise_gate"))
        regime_veto_gate = bool(_bt(regime_veto_gate, "regime_veto_gate"))
        regime_down_threshold = float(_bt(regime_down_threshold, "regime_down_threshold"))
        regime_up_threshold = float(_bt(regime_up_threshold, "regime_up_threshold"))
        ex_div_entry_gate = str(_bt(ex_div_entry_gate, "ex_div_entry_gate"))
        ex_div_entry_k_days = int(_bt(ex_div_entry_k_days, "ex_div_entry_k_days"))
        _research_prefix = str(_bt(None, "research_table_prefix"))
        # W81: the SAME key live's Greeks read (options_chain._risk_free_rate); was a bare
        # `r = 0.05` at three pricing sites below.
        risk_free_rate = float(_resolve_setting(
            risk_free_rate, "options", "risk_free_rate", OptionsConfig, _settings))
        pct_from_60d_high_threshold = float(_bt(pct_from_60d_high_threshold, "pct_from_60d_high_threshold"))
        min_edge_over_baseline = float(_bt(min_edge_over_baseline, "min_edge_over_baseline"))
        max_entry_vol_annual = float(_bt(max_entry_vol_annual, "max_entry_vol_annual"))
        spread_base = float(_bt(spread_base, "spread_base"))
        spread_iv_sensitivity = float(_bt(spread_iv_sensitivity, "spread_iv_sensitivity"))
        spread_dte_sensitivity = float(_bt(spread_dte_sensitivity, "spread_dte_sensitivity"))
        spread_cap = float(_bt(spread_cap, "spread_cap"))
        earnings_buffer_days_before = int(_resolve_setting(
            earnings_buffer_days_before, "earnings", "buffer_days_before", EarningsConfig, _settings))
        earnings_buffer_days_after = int(_resolve_setting(
            earnings_buffer_days_after, "earnings", "buffer_days_after", EarningsConfig, _settings))
        self._data = self._prepare_data(data)
        self._strategies = strategies
        self._initial_capital = initial_capital
        # W5: _commission / _regulatory_fees / _cost_per_contract_leg are
        # resolved in the loaded-settings block below (they need _settings),
        # same as _iv_floor.
        self._slippage_pct = slippage_pct
        self._spread_base = spread_base
        self._spread_iv_sensitivity = spread_iv_sensitivity
        self._spread_dte_sensitivity = spread_dte_sensitivity
        self._spread_cap = spread_cap
        # R16: prefer EMPIRICALLY CALIBRATED per-symbol spreads over the
        # config formula. scripts/calibrate_option_spreads.py fits the model
        # to real observed bid/ask quotes and stores it in
        # option_spread_params — a table that existed, was never populated,
        # and (once populated) was still read by nothing. Measured 2026-08-11
        # on 3,159 quotes, config's base=0.04 overstates the real base by
        # 2-13x (SPY 0.012 / QQQ 0.003 / IWM 0.020), so every simulated
        # credit has been charged too much friction. AIT_BT_CALIBRATED_SPREADS=0
        # restores the config formula for A/B runs.
        self._calibrated_spreads = None
        self._position_size_pct = position_size_pct
        self._budget_aware_wings = bool(budget_aware_wings)
        self._condor_any_direction = bool(condor_any_direction)
        if friction_model not in ("legacy", "per_leg_usd"):
            raise ValueError(f"friction_model must be 'legacy' or 'per_leg_usd', got {friction_model!r}")
        self._friction_model = friction_model
        self._friction_entry_capture = float(friction_entry_capture)
        self._friction_exit_capture = float(friction_exit_capture)
        self._friction_stress = float(friction_stress_multiplier)
        # W76: friction by DTE band (0-6 DTE). Off = the single all-DTE model, as before.
        self._friction_short_dte = bool(_bt(friction_short_dte, "friction_short_dte"))
        self._friction_min_sessions = int(_bt(friction_min_sessions_per_band,
                                              "friction_min_sessions_per_band"))
        self._friction_floor_usd = float(_bt(friction_extrapolated_floor_usd,
                                             "friction_extrapolated_floor_usd"))
        self._friction_stats = {"fitted": 0, "default": 0, "extrapolated": 0}
        self._leg_spread_model = None  # set by _load_leg_spread_model() when per_leg_usd
        # W66: entry-funnel counters (see _reject / BacktestResult.entry_funnel)
        self._funnel: Counter = Counter()
        self._max_contracts_per_trade = (
            int(max_contracts_per_trade) if max_contracts_per_trade else None)
        self._stop_loss_pct = stop_loss_pct
        self._profit_target_pct = profit_target_pct
        self._max_hold_days = max_hold_days
        # R20b: _min_confidence / _range_min_confidence / _iv_floor are
        # resolved in the loaded-settings block below (they need _settings).
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct
        self._breakeven_trigger_pct = breakeven_trigger_pct
        self._context_bars = context_bars
        self._range_predictor = range_predictor
        self._delta_short = delta_short
        self._delta_long = delta_long
        self._wing_floor_dollars = wing_floor_dollars
        # R16 #8: mirror the live resolution when wing_k is not explicitly
        # configured (R19 contract: env AIT_IC_WING_K > config.yaml
        # backtest.wing_k > CONTRACT_DEFAULTS 1.6 — the comment here used to
        # say "default 1.0", stale since the 2026-08-04 promotion).
        self._wing_k = (
            float(wing_k) if wing_k is not None
            else contract_float("AIT_IC_WING_K")
        )
        self._delta_iv_scale = delta_iv_scale
        self._skew_factor = skew_factor
        # W5: _skew_slope_per_pct_otm resolved in the settings block below.
        self._hurst_regime_threshold = hurst_regime_threshold
        self._hurst_regime_penalty = hurst_regime_penalty
        self._hurst_hard_veto_multiplier = hurst_hard_veto_multiplier
        self._multifractal_max_width = multifractal_max_width
        if gates_mode not in ("observe", "enforce"):
            raise ValueError(f"gates_mode must be 'observe' or 'enforce', got {gates_mode!r}")
        self._gates_mode = gates_mode
        self._iv_rank_gate = bool(iv_rank_gate)
        self._iv_rank_floor = contract_float("AIT_IRON_CONDOR_IV_FLOOR")
        self._iv_rank_cap = contract_float("AIT_IV_RANK_CAP")
        self._iv_rank_series_arg = iv_rank_series
        self._iv_rank_arr = self._build_iv_rank_array()
        # W67: the shared fractal gate (ait.ml.regime_gate) replaces the inline copy.
        from ait.ml.regime_gate import SHAPE_PROPORTIONAL, FractalGateConfig, resolve_veto_spread
        self._fractal_cfg = FractalGateConfig(
            threshold=float(hurst_regime_threshold), penalty=float(hurst_regime_penalty),
            multifractal_max_width=float(multifractal_max_width),
            veto_spread=resolve_veto_spread(
                hurst_regime_threshold, hurst_hard_veto_multiplier, fractal_veto_spread),
            penalty_on_range_prob=float(penalty_on_range_prob),
            shape=SHAPE_PROPORTIONAL)  # the engine's historical shape; live uses live_flat
        self._iv_rank_rise_threshold = iv_rank_rise_threshold
        self._iv_rank_rise_gate = iv_rank_rise_gate     # W71c: False = no veto; the rise is still logged
        self._regime_veto_gate = regime_veto_gate       # W81: False = no veto; the class is still logged
        self._regime_down = regime_down_threshold       # W81: was a bare -0.05 at three sites
        self._regime_up = regime_up_threshold           # W81: was a bare 0.02 at three sites
        from ait.backtesting.ex_div import MODES as _EX_DIV_MODES
        if ex_div_entry_gate not in _EX_DIV_MODES:
            raise ValueError(f"ex_div_entry_gate {ex_div_entry_gate!r} not in {_EX_DIV_MODES}")
        self._ex_div_mode = ex_div_entry_gate           # W84: live's ex-div entry block, research twin
        self._ex_div_k = ex_div_entry_k_days
        self._research_table_prefix = _research_prefix
        self._risk_free_rate = risk_free_rate           # W81: was a bare 0.05 at three sites
        self._pct_from_60d_high_threshold = pct_from_60d_high_threshold
        self._min_edge_over_baseline = min_edge_over_baseline
        self._features_cache = features_cache
        self._max_concurrent_positions = max_concurrent_positions
        self._max_entry_vol_annual = max_entry_vol_annual
        self._market_context = market_context
        self._symbol = symbol or ""
        # W76: coerce to Path. HistoricalDataStore does `db_path.parent`, so a plain str made
        # the friction/spread loaders fail with a WARNING and the run silently fell back to
        # the LEGACY % friction model (the W74/W75 reference studies were mislabelled for
        # exactly this reason).
        self._db_path = Path(db_path) if db_path is not None else None
        # R16: must run AFTER _symbol is set — the calibration is per-symbol.
        if os.environ.get("AIT_BT_CALIBRATED_SPREADS", "1") != "0":
            self._load_calibrated_spreads()
        if self._friction_model == "per_leg_usd":
            self._load_leg_spread_model()
        self._earnings_dates: set[date] = self._load_earnings_dates(
            symbol, earnings_buffer_days_before, earnings_buffer_days_after)
        self._ex_dates: list[date] | None = self._load_ex_dates()   # W84: after _symbol/_db_path
        self._intraday_store = intraday_store
        # R20: resolve LOADED settings once for every config-backed knob below
        # (extends the R16 #7 pre_event_blackout_days pattern).
        # R20b follow-up: reuse a caller-supplied `settings` (e.g.
        # StrategyOptimizer loads it once and threads it through per Optuna
        # trial) instead of unconditionally re-reading + re-validating
        # config.yaml on every Backtester construction.
        # W70: _settings is loaded at the top of __init__ (the constructor knobs resolve from it).
        # R20 #4/#5b + R20b (pre-registered PLAN 2026-08-21): every knob below
        # follows the same explicit > settings.<section>.<field> > config-model
        # default precedence (_resolve_setting); explicit constructor args
        # always win, a partial stub or missing config.yaml degrades to that
        # field's pydantic default.
        #   iv_floor             0.12 -> settings.backtest.iv_floor      (yaml 0.20)
        #   range_min_confidence 0.55 -> settings.ml.range_min_confidence (yaml 0.65)
        #   min_confidence       0.55 -> settings.risk.min_confidence     (yaml 0.50)
        # min_confidence's home is risk.min_confidence because the engine
        # consumes it as the DIRECTIONAL-confidence entry gate (run():
        # `confidence < effective_min_conf` skips the entry, with the
        # neutral-credit bypass) — exactly the gate live reads from
        # settings.risk.min_confidence (orchestrator.py `neutral_only =
        # prediction.confidence < min_confidence`), and the production-params
        # exporter already maps "min_confidence" -> ("risk", "min_confidence")
        # (optimization/results.py). It is NOT ml.range_min_confidence (the
        # range-gate floor, a separate knob resolved below).
        # entry_dte: real time-to-expiration for BS pricing/decay, fixed near
        # live's actual chain selection (dte_range[0]) — see constructor
        # comment. Not itself Optuna-searched; max_hold_days (below) stays
        # the searched hold-cap knob.
        self._post_event_blackout_days = (
            int(post_event_blackout_days) if post_event_blackout_days is not None else None)
        self._post_event_dates = tuple(sorted(
            d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
            for d in (post_event_dates or ())))
        self._max_loss_cap = (float(max_loss_per_contract_dollars)
                              if max_loss_per_contract_dollars is not None else None)
        self._contracts_cap = int(contracts_cap) if contracts_cap is not None else None
        self._entry_dte = int(entry_dte) if entry_dte is not None else int(
            _resolve_setting(None, "options", "dte_range", OptionsConfig, _settings)[0])
        # W74: fidelity switch + the three short-DTE exit knobs (None = off = legacy).
        self._engine_model = str(_bt(engine_model, "engine_model"))
        if self._engine_model not in ("legacy", "realistic"):
            raise ValueError(f"engine_model must be 'legacy' or 'realistic', got {self._engine_model!r}")
        self._realistic = self._engine_model == "realistic"
        # W75: leg IVs from the real-bar surface of each leg's own expiry (default
        # off). A symbol/date without a smile falls back per leg and is counted.
        self._pricing_source = str(_bt(pricing_source, "pricing_source"))
        if self._pricing_source not in ("bs_legacy", "surface"):
            raise ValueError(
                f"pricing_source must be 'bs_legacy' or 'surface', got {self._pricing_source!r}")
        self._surface_stats = {"hit": 0, "miss": 0}
        self._surface = self._load_surface() if self._pricing_source == "surface" else None
        _g = _bt(expiry_guard_dte, "expiry_guard_dte")
        self._expiry_guard_dte = None if _g is None else int(_g)
        _hf = _bt(max_hold_fraction, "max_hold_fraction")
        self._max_hold_fraction = None if _hf is None else float(_hf)
        if self._max_hold_fraction is not None and not 0.0 < self._max_hold_fraction <= 1.0:
            raise ValueError(f"max_hold_fraction must be in (0, 1], got {self._max_hold_fraction}")
        _fm = _bt(flatten_minutes_before_close, "flatten_minutes_before_close")
        self._flatten_minutes = None if _fm is None else int(_fm)
        # W77: expiry settlement knobs (realistic engine only; see ait.backtesting.settlement).
        self._pin_band = float(_bt(pin_band_usd, "pin_band_usd"))
        self._assignment_fee = float(_bt(assignment_fee_per_share, "assignment_fee_per_share"))
        self._stock_commission = float(_bt(stock_commission_per_share, "stock_commission_per_share"))
        self._settlement_stats = {"settled": 0, "pending": 0, "assigned_gap": 0,
                                  "fallback": 0, "over_max_loss": 0}
        self._scan_interval_minutes = int(_resolve_setting(
            scan_interval_minutes, "backtest", "scan_interval_minutes", BacktestConfig, _settings))
        self._entry_window_start_et = str(_resolve_setting(
            entry_window_start_et, "backtest", "entry_window_start_et", BacktestConfig, _settings))
        self._entry_window_end_et = str(_resolve_setting(
            entry_window_end_et, "backtest", "entry_window_end_et", BacktestConfig, _settings))
        self._limit_order_timeout_bars = int(_resolve_setting(
            limit_order_timeout_bars, "backtest", "limit_order_timeout_bars", BacktestConfig, _settings))
        self._iv_floor = float(_resolve_setting(
            iv_floor, "backtest", "iv_floor", BacktestConfig, _settings))
        # W5 model-vs-reality-commission-constant: per-leg, per-side broker
        # cost = commission + regulatory/exchange/clearing fees. Both charged
        # on entry AND exit (see run() and _calc_pnl). _commission stays the
        # commission LINE (walkforward/optimizer pass it explicitly); the
        # all-in figure the engine actually debits is
        # _cost_per_contract_leg = 0.65 + 0.2668 = 0.9168 by default, the
        # measured ledger mean.
        self._commission = float(_resolve_setting(
            commission_per_contract, "backtest", "commission_per_contract",
            BacktestConfig, _settings))
        self._regulatory_fees = float(_resolve_setting(
            regulatory_fees_per_contract, "backtest",
            "regulatory_fees_per_contract", BacktestConfig, _settings))
        self._cost_per_contract_leg = self._commission + self._regulatory_fees
        # W5 model-vs-reality-skew-10x-flat: relative put-side IV uplift per
        # percentage point OTM, consumed in _get_leg_iv (the single per-leg IV
        # input point every builder and reprice path goes through).
        self._skew_slope_per_pct_otm = float(_resolve_setting(
            skew_slope_per_pct_otm, "backtest", "skew_slope_per_pct_otm",
            BacktestConfig, _settings))
        self._range_min_confidence = float(_resolve_setting(
            range_min_confidence, "ml", "range_min_confidence", MLConfig, _settings))
        self._min_confidence = float(_resolve_setting(
            min_confidence, "risk", "min_confidence", RiskConfig, _settings))
        # R20 #5a: live's TP ladder is gated by exit.time_decay_scaling
        # (portfolio.py _get_take_profit_targets); the engine ran the ladder
        # unconditionally, so flipping the flag moved live to flat 0.50
        # targets while research kept the ladder. No constructor override —
        # always follows the loaded/default config, same as before.
        self._exit_time_decay_scaling = bool(_resolve_setting(
            None, "exit", "time_decay_scaling", ExitConfig, _settings))
        self._credit_tp_ladder = (
            exit_policy.validate_ladder(credit_tp_ladder)
            if credit_tp_ladder is not None else None
        )
        self._meta_labeler = meta_labeler
        self._eval_start_date = eval_start_date

        # R6 parity knobs — None resolves to the SAME env var + default the
        # live bot reads, so an un-configured backtest matches live behavior.
        # R16 #6: default "0" = flat credit stop DISABLED, matching live
        # portfolio.py:413 exactly (R12-B1: flat stops fired through their
        # trigger on gaps and every level underperformed touch-close; live
        # runs touch-close-only, which the engine does NOT yet mirror — a
        # documented structural divergence, not a parity value mismatch).
        self._credit_loss_limit_mult = (
            float(credit_loss_limit_mult) if credit_loss_limit_mult is not None
            else contract_float("AIT_CREDIT_LOSS_LIMIT")
        )
        # R16: short-strike touch stop — live's PRIMARY loss exit, mirrored
        # here at last via daily High/Low (they bracket the true intraday
        # path). ON by default = live parity. AIT_BT_TOUCH_STOP=0 restores
        # the old close-only behaviour for before/after comparisons ONLY.
        self._touch_stop_enabled = (
            os.environ.get("AIT_BT_TOUCH_STOP", "1") != "0")
        self._ic_min_credit = (
            float(ic_min_credit) if ic_min_credit is not None
            else contract_float("AIT_IC_MIN_CREDIT")
        )
        self._ic_min_credit_width = (
            float(ic_min_credit_width) if ic_min_credit_width is not None
            else contract_float("AIT_IC_MIN_CREDIT_WIDTH")
        )
        self._ic_em_min = (
            float(ic_em_min) if ic_em_min is not None
            else contract_float("AIT_IC_EM_MIN")
        )
        self._ic_em_max = (
            float(ic_em_max) if ic_em_max is not None
            else contract_float("AIT_IC_EM_MAX")
        )
        self._em_gate = (
            bool(em_gate) if em_gate is not None
            else os.environ.get("AIT_BT_EM_GATE", "1") != "0"
        )
        self._macro_event_gate = macro_event_gate
        self._economic_cal = None
        if macro_event_gate:
            try:
                from ait.data.economic_calendar import EconomicCalendar
                self._economic_cal = (economic_calendar if economic_calendar is not None
                                      else EconomicCalendar())
            except Exception:  # noqa: BLE001 — gate is best-effort, never fatal
                self._economic_cal = None

        # R16 #7: resolve the macro blackout window ONCE, from the same source
        # live reads (loaded settings), instead of pinning to RiskConfig()
        # defaults on every entry day.
        self._pre_event_blackout_days = int(_resolve_setting(
            pre_event_blackout_days, "risk", "pre_event_blackout_days", RiskConfig, _settings))

        # R16 #1: explicit, loudly-logged predictor mode. Never silently score
        # research windows with the live (future-trained) artifact.
        self._allow_live_model_fallback = bool(allow_live_model_fallback)
        if predictor is not None:
            self._predictor = predictor
            log.info("ml_predictor_mode", mode="explicit_window_model")
        elif self._allow_live_model_fallback:
            self._predictor = self._load_predictor()
            if self._predictor is not None:
                log.warning(
                    "ml_predictor_mode",
                    mode="LIVE_ARTIFACT_FALLBACK",
                    version=getattr(self._predictor, "model_version", "?"),
                    note="scoring with live models/ensemble.pkl — INVALID for "
                         "walk-forward OOS windows (look-ahead leak); pass "
                         "allow_live_model_fallback=False for research runs",
                )
            else:
                log.info("ml_predictor_mode", mode="ungated_no_model_found")
        else:
            self._predictor = None
            log.info(
                "ml_predictor_mode",
                mode="ungated_fallback_disabled",
                note="no window model and live-artifact fallback disabled — "
                     "direction comes from _simple_direction only",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _reject(self, reason: str) -> None:
        """W66: record that an evaluated day ended in this entry rejection.

        Most of the engine's entry gates used to be silent (the condor builder
        returned None without a log), so an operator could not see WHY a run
        produced 0-9 trades. Every `continue`/`return None` on the entry path now
        counts here; `BacktestResult.entry_funnel` carries the totals."""
        self._funnel[reason] += 1

    def _build_iv_rank_array(self) -> np.ndarray | None:
        """W68: IV rank aligned to `self._data` rows (NaN = unmeasurable), or None when the gate
        is off. Uses only observations at or before each row (no look-ahead)."""
        if not getattr(self, "_iv_rank_gate", False):
            return None
        from ait.data.iv_rank import iv_rank_series
        ser = self._iv_rank_series_arg
        if ser is None:
            if "implied_vol" in self._data.columns:
                ser = iv_rank_series(self._data["implied_vol"])
            else:
                log.warning("iv_rank_gate_no_iv_series",
                            note="iv_rank_gate on but no implied_vol column: every credit "
                                 "entry fails closed, as live does when the rank is unknown")
                return np.full(len(self._data), np.nan)
        return ser.reindex(self._data.index).to_numpy(dtype=float)

    def _would_reject(self, reason: str) -> None:
        """W67: observe mode -- the gate WOULD have rejected this day but did not.

        Counted apart from `_reject` (prefix `would_reject:`) so the conservation
        invariant `days_evaluated == entries_taken + sum(top-level rejections)` still
        holds, and so the gate study can see exactly which days an enforced gate drops."""
        self._funnel[f"would_reject:{reason}"] += 1

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        self._funnel = Counter()
        capital = self._initial_capital
        trades: list[dict] = []
        open_positions: list[dict] = []

        dates = self._data.index
        if len(dates) < 2:
            log.warning("insufficient_data", rows=len(dates))
            return BacktestResult(
                initial_capital=self._initial_capital,
                final_capital=self._initial_capital,
            )

        log.info(
            "backtest_start",
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            bars=len(dates),
            strategies=self._strategies,
        )

        lookback = max(20, self._context_bars)

        for i in range(lookback, len(dates)):
            today = dates[i]
            today_date = today.date() if hasattr(today, "date") else today
            hist = self._data.iloc[: i + 1]
            row = self._data.iloc[i]
            self._funnel["days_evaluated"] += 1

            # --- 1. Check exits on open positions ---
            still_open = []
            # Load today's intraday bars once for the whole exit loop (Fix 1c)
            _session_for_exit = None
            if self._intraday_store is not None:
                _sym = self._symbol
                if _sym:
                    _session_for_exit = self._intraday_store.load_intraday_range(
                        symbol=_sym, start_date=today_date, end_date=today_date
                    )

            closed_today: list[dict] = []   # W78: positions this session's exit step closed
            for pos in open_positions:
                # W77: shares carried over the expiry bell are closed at THIS session's
                # open, before any other check (the next row of the frame is the next
                # session, so weekends and holidays are handled).
                if pos.get("settlement") is not None:
                    _o = row["Open"] if "Open" in row else row["Close"]
                    pos.update(self._close_pending_settlement(pos, float(_o), today_date))
                    capital += pos["pnl"]
                    trades.append(pos)
                    log.debug("trade_closed", strategy=pos["strategy"],
                              pnl=f"{pos['pnl']:.2f}", reason=pos["exit_reason"])
                    continue
                # Thesis re-evaluation: exit early if ML direction strongly contradicts entry (Gap Z6).
                thesis_exit = self._check_thesis_invalidation(pos, hist)
                if thesis_exit is not None:
                    underlying = row["Close"]
                    days_held = (today_date - date.fromisoformat(pos["entry_date"])).days
                    current_val = self._reprice_position(pos, underlying, days_held, hist)
                    thesis_exit["pnl"] = round(self._calc_pnl(pos, current_val), 2)
                    thesis_exit["exit_price"] = round(current_val, 4)
                    pos.update(thesis_exit)
                    capital += pos["pnl"]
                    trades.append(pos)
                    log.debug("thesis_invalidated", strategy=pos["strategy"], pnl=f"{pos['pnl']:.2f}")
                    continue

                # Intraday exit check (Fix 1c): scan 5-min bars for stops/targets.
                intraday_exit = None
                if _session_for_exit is not None and not _session_for_exit.empty:
                    intraday_exit = self._check_intraday_exit(pos, _session_for_exit, today_date)

                exit_info = intraday_exit or self._check_exit(pos, row, today_date, hist)
                if exit_info is not None:
                    pos.update(exit_info)
                    capital += pos["pnl"]
                    trades.append(pos)
                    closed_today.append(pos)
                    log.debug(
                        "trade_closed",
                        strategy=pos["strategy"],
                        pnl=f"{pos['pnl']:.2f}",
                        reason=pos["exit_reason"],
                    )
                else:
                    still_open.append(pos)
            open_positions = still_open

            # Track current capital for strategy selection
            self._current_capital = capital

            # --- 2. Generate new signal (skip if at position limit) ---
            if len(open_positions) >= self._max_concurrent_positions:
                self._reject("max_concurrent")
                continue

            # H2 val-split: skip new entries before eval_start_date (exits still processed above)
            if self._eval_start_date is not None and today_date < self._eval_start_date:
                self._reject("before_eval_start")
                continue

            # W78: the DECISION FRAME. A realistic engine with intraday entries decides at the
            # scan time (10:30 ET), so the gates and models may use history through the PREVIOUS
            # session only (assumption A3). `hist` still carries day D's full bar for pricing
            # and exits; every read below that is a decision input uses `hist_dec`
            # (and `_dec_lag` for the IV-rank array). Legacy / daily-close entries are unchanged.
            _dec_lag = 1 if (self._realistic and self._intraday_store is not None) else 0
            hist_dec = hist.iloc[: len(hist) - _dec_lag]
            direction, confidence, features_df = self._get_direction(hist_dec, market_context=self._market_context)

            # Decision-chain dict — populated as each gate is evaluated;
            # attached to the trade record so the dashboard can render the
            # full entry reasoning without re-running the backtest.
            _entry_decision: dict = {
                "direction_class": direction.value if hasattr(direction, "value") else str(direction),
                "direction_conf": round(float(confidence), 4),
                "range_gate": {"prob": None, "threshold": self._range_min_confidence, "pass": None},
                "vol_gate": {"vol_10d": None, "max": self._max_entry_vol_annual, "pass": True},
                "meta_label": {"take": True, "prob": None, "threshold": 0.5},
                "fractal_gate": {"hurst_spread": 0.0, "threshold": self._hurst_regime_threshold, "pass": True},
                "regime": "range_bound",
                "earnings_skip": False,
                "macro_gate": {"days_to_event": None, "blocked": False},
            }

            # Apply fractal regime gate to confidence for credit strategies.
            # Features are already computed inside _get_direction — reuse them.
            # W67: ONE implementation shared with the live orchestrator
            # (ait.ml.regime_gate.fractal_gate); this site only applies the verdict.
            _fractal_verdict = None
            if not features_df.empty:
                last_f = features_df.iloc[-1]
                spread = float(last_f.get("hurst_scale_spread", 0.0))
                mf_w   = float(last_f.get("multifractal_width",  0.0))
                _fractal_verdict = fractal_gate(spread, mf_w, self._fractal_cfg)
                _entry_decision["fractal_gate"] = {
                    "hurst_spread": round(spread, 4),
                    "multifractal_width": round(mf_w, 4),
                    "threshold": self._hurst_regime_threshold,
                    "pass": not _fractal_verdict.over_spread,
                }
                # Hard veto is optional (veto_spread 0 = off) and only binds the
                # neutral credit structures. Exp 20 post-mortem showed QQQ
                # hurst_spread rarely drops below ~0.43 in normal conditions;
                # overly tight thresholds can block all entries.
                _neutral_strat = bool(set(self._strategies) & set(NEUTRAL_CREDIT_GATED))
                if _fractal_verdict.veto and _neutral_strat:
                    _entry_decision["fractal_gate"]["hard_veto"] = True
                    log.debug(
                        "hard_veto_fired",
                        component="backtesting.engine",
                        strategies=self._strategies,
                        hurst_spread=round(spread, 4),
                        hard_veto_threshold=round(self._fractal_cfg.veto_spread, 4),
                        hurst_regime_threshold=round(self._hurst_regime_threshold, 4),
                        gates_mode=self._gates_mode,
                    )
                    if self._gates_mode == "enforce":
                        self._reject("hurst_hard_veto")
                        continue
                    self._would_reject("hurst_hard_veto")
                confidence = max(0.0, confidence - _fractal_verdict.directional_penalty)

            effective_min_conf = self._min_confidence

            # Iron condors and short strangles are market-neutral: high directional
            # confidence signals a trending regime — exactly when they fail. Skip the
            # direction gate for these strategies; the range model below is the sole
            # entry filter. Directional strategies still require confidence ≥ min_conf.
            _neutral_only = bool(set(self._strategies) & set(NEUTRAL_CREDIT_GATED))
            if not _neutral_only:
                if confidence < effective_min_conf:
                    self._reject("direction_confidence")
                    continue
                if direction == SignalDirection.BEARISH and confidence < effective_min_conf + 0.05:
                    self._reject("direction_confidence_bearish")
                    continue

            # MetaLabeler gate (Gap Z1): applied during OOS evaluation when a trained
            # per-window model is provided, exactly mirroring the live orchestrator.
            if self._meta_labeler is not None and self._meta_labeler.is_trained:
                meta_ctx: dict = {"primary_confidence": float(confidence)}
                if not features_df.empty:
                    last_f = features_df.iloc[-1]
                    vol_exp = float(last_f.get("vol_regime_expanding", 0.0)) > 0.5
                    px_sma  = float(last_f.get("price_vs_sma_20", 0.0))
                    meta_ctx.update({
                        # W81: the class thresholds are backtest.regime_{down,up}_threshold
                        # (were bare -0.05 / 0.02 here, in the veto and in entry_regime).
                        "regime_trending_up":   1.0 if (vol_exp and px_sma > self._regime_up) else 0.0,
                        "regime_trending_down":  1.0 if (vol_exp and px_sma < self._regime_down) else 0.0,
                        "regime_high_vol":       1.0 if (vol_exp and self._regime_down <= px_sma <= self._regime_up) else 0.0,
                        "regime_range_bound":    0.0 if vol_exp else 1.0,
                        "vix":                   float(last_f.get("vix_level", 0.5)),
                        "iv_rank":               float(last_f.get("iv_rank", 0.5)),
                        "rsi_14":                float(last_f.get("rsi_14", 50.0)),
                        "rsi_7":                 float(last_f.get("rsi_7",  50.0)),
                        "bb_position":           float(last_f.get("bb_position", 0.5)),
                        "volume_sma_20_ratio":   float(last_f.get("volume_sma_20_ratio", 1.0)),
                        "realized_vol_20":       float(last_f.get("realized_vol_20", 0.20)),
                        "atr_pct":               float(last_f.get("atr_pct", 0.01)),
                        "weekly_trend_aligned":  float(last_f.get("weekly_trend_aligned", 0.5)),
                        "volume_confirmation":   float(last_f.get("volume_confirmation", 0.0)),
                        "macd_hist":             float(last_f.get("macd_hist", 0.0)),
                        "price_vs_sma_20":       px_sma,
                        "sma_10_20_cross":       float(last_f.get("sma_10_20_cross", 0.5)),
                    })
                meta_ctx.setdefault("sentiment_score", 0.0)
                meta_ctx.setdefault("hour_of_day", 10)
                try:
                    meta_signal = self._meta_labeler.predict(meta_ctx)
                    if meta_signal is not None:
                        _entry_decision["meta_label"] = {
                            "take": bool(meta_signal.take_trade),
                            "prob": round(float(getattr(meta_signal, "probability", 0.5)), 4),
                            "threshold": 0.5,
                        }
                        if not meta_signal.take_trade:
                            self._reject("meta_label")
                            continue
                except Exception:
                    pass  # meta-labeler errors are non-fatal

            strategy = self._select_strategy(direction, hist_dec, confidence, features_df)
            if strategy is None:
                self._reject("no_strategy")
                continue

            # W68: live's two IV-rank gates (floor in iron_condor.py, cap in the risk manager).
            if self._iv_rank_gate and strategy in CREDIT_STRATEGIES:
                _ivr = float(self._iv_rank_arr[i - _dec_lag]) if self._iv_rank_arr is not None else float("nan")
                if np.isnan(_ivr):
                    self._reject("iv_rank_unknown")   # live fails closed on an unmeasurable rank
                    continue
                _entry_decision["iv_rank"] = round(_ivr, 1)
                if strategy == "iron_condor" and _ivr < self._iv_rank_floor:
                    self._reject("iv_rank_floor")
                    continue
                if _ivr > self._iv_rank_cap:
                    self._reject("iv_rank_cap")
                    continue

            # Earnings proximity skip (Gap Z4): matches live orchestrator behaviour.
            if today_date in self._earnings_dates:
                log.debug("earnings_skip", date=str(today_date), strategy=strategy)
                self._reject("earnings")
                continue

            # Macro-event entry gate (live orchestrator credit-block parity).
            # R16 #7: the blackout window is resolved once in __init__ from
            # the LOADED settings (constructor override > load_settings() >
            # RiskConfig default) — the same source live reads — so a yaml
            # override no longer changes live only.
            # LIMITATION: the calendar is hardcoded 2026 + 2027-H1, so for
            # pre-2026 windows days-to-event is always > window and the gate
            # never fires (documented in the parity manifest).
            if self._economic_cal is not None and strategy in CREDIT_STRATEGIES:
                _blackout = self._pre_event_blackout_days
                try:
                    _d2e = self._economic_cal.days_until_next_event(today_date)
                except Exception:  # noqa: BLE001
                    _d2e = None
                _entry_decision["macro_gate"] = {
                    "days_to_event": _d2e,
                    "blocked": _d2e is not None and _d2e <= _blackout,
                }
                if _d2e is not None and _d2e <= _blackout:
                    log.debug(
                        "macro_event_entry_skip",
                        date=str(today_date),
                        strategy=strategy,
                        days_to_event=_d2e,
                    )
                    self._reject("macro_event")
                    continue

            # Research-only post-event blackout (see __init__). Inert unless
            # scripts/post_event_study.py sets it.
            if strategy in CREDIT_STRATEGIES and self._post_event_blocked(today_date):
                _entry_decision["post_event_gate"] = {
                    "days_since_event": self._days_since_event(today_date),
                    "blocked": True,
                }
                log.debug("post_event_entry_skip", date=str(today_date),
                          strategy=strategy,
                          days_since_event=self._days_since_event(today_date))
                self._reject("post_event")
                continue

            # Regime gate: iron condors / short strangles fail in trending-down regimes.
            # Exp 21 post-mortem: 4 trending_down trades, 25% win rate, avg PnL -$195.
            # Both macro dislocations in the dataset (Yen carry unwind Aug-2024,
            # tariff shock Mar-2026) occurred in this regime. Trending_up (100% win)
            # and high_volatility (62% win) maintain positive EV and are not blocked.
            # Exp 22: threshold -0.02 was too loose — blocked profitable 2-4% corrections
            # in W02, causing Optuna to adapt badly. Both structural failures cleared -0.05
            # (Yen carry -6-8%, tariff shock -8-10%); raised to -0.05 for Exp 23.
            if strategy in NEUTRAL_CREDIT_GATED and not features_df.empty:
                _last_f        = features_df.iloc[-1]
                _vol_exp_flag  = float(_last_f.get("vol_regime_expanding", 0.0)) > 0.5
                _px_sma_val    = float(_last_f.get("price_vs_sma_20", 0.0))
                _regime_class  = research_regime(
                    _vol_exp_flag, _px_sma_val, self._regime_down, self._regime_up)
                _entry_decision["regime_class"] = _regime_class
                # W81: backtest.regime_veto_gate=False skips the VETO only (research-only gate;
                # live has none): the class is still recorded, and a day the veto WOULD have
                # blocked is flagged so a study can evaluate it later (D10).
                if _regime_class == "trending_down" and not self._regime_veto_gate:
                    _entry_decision["regime_veto_suppressed"] = True
                if _regime_class == "trending_down" and self._regime_veto_gate:
                    _entry_decision["regime_veto"] = True
                    log.debug(
                        "regime_veto_fired",
                        component="backtesting.engine",
                        strategy=strategy,
                        regime=_regime_class,
                        vol_regime_expanding=round(float(_last_f.get("vol_regime_expanding", 0.0)), 4),
                        price_vs_sma_20=round(_px_sma_val, 4),
                    )
                    self._reject("regime_veto")
                    continue

            # Range model gate: for neutral credit structures, replace
            # confidence with P(stays in range). Skip if below range threshold.
            if strategy in NEUTRAL_CREDIT_GATED:
                if self._range_predictor is None:
                    # R16 #3: honor the walkforward "range model absent ->
                    # block entries" contract. walkforward raises
                    # range_min_confidence to 1.0 when window range training
                    # FAILED; with no predictor no probability can ever reach
                    # it, so the entry must be blocked HERE. Previously the
                    # whole gate was skipped when the predictor was None and
                    # entries flowed ungated — every recent study (ablation,
                    # wingk, shadow rounds) ran with no range gate at all.
                    if self._range_min_confidence >= 1.0:
                        _entry_decision["range_gate"] = {
                            "prob": None,
                            "threshold": self._range_min_confidence,
                            "pass": False,
                        }
                        log.debug(
                            "range_gate_blocked_no_predictor",
                            date=str(today_date),
                            strategy=strategy,
                            threshold=self._range_min_confidence,
                            gates_mode=self._gates_mode,
                        )
                        if self._gates_mode == "enforce":
                            self._reject("range_gate_no_predictor")
                            continue
                        self._would_reject("range_gate_no_predictor")
                else:
                    try:
                        rp = self._range_predictor.predict(
                            hist_dec,
                            symbol=self._symbol,
                            market_context=self._market_context,
                            min_edge_override=self._min_edge_over_baseline,
                        )
                        # W67: the fractal penalty may also lower P(in range) (default 0
                        # = off, live parity); the raw model output is recorded beside it.
                        _p_range = (None if rp is None else float(rp.probability_in_range))
                        if _p_range is not None and _fractal_verdict is not None:
                            _p_range = apply_range_prob_delta(_p_range, _fractal_verdict)
                        if _p_range is None or _p_range < self._range_min_confidence:
                            if self._gates_mode == "enforce":
                                self._reject("range_gate_below_threshold")
                                continue  # bad range setup → skip
                            self._would_reject("range_gate_below_threshold")
                            _entry_decision["range_gate"] = {
                                "prob": None if _p_range is None else round(_p_range, 4),
                                "threshold": self._range_min_confidence, "pass": False}
                        else:
                            _entry_decision["range_gate"] = {
                                "prob": round(_p_range, 4),
                                "threshold": self._range_min_confidence,
                                "pass": True,
                            }
                        if rp is not None:
                            _entry_decision["range_gate"]["raw_prob"] = round(
                                float(rp.probability_in_range), 4)
                            confidence = _p_range
                    except Exception:
                        pass

            # Realized-vol entry gate: neutral credit structures cannot profit
            # during high-volatility regimes (e.g. tariff shocks, VIX > 40).
            if strategy in NEUTRAL_CREDIT_GATED:
                recent_close = hist_dec["Close"].iloc[-11:]
                if len(recent_close) >= 11:
                    vol_10d = recent_close.pct_change().std() * (252 ** 0.5)
                    _entry_decision["vol_gate"] = {
                        "vol_10d": round(float(vol_10d), 4),
                        "max": self._max_entry_vol_annual,
                        "pass": vol_10d <= self._max_entry_vol_annual,
                    }
                    if vol_10d > self._max_entry_vol_annual:
                        log.debug(
                            "vol_gate_skip",
                            strategy=strategy,
                            vol_10d=f"{vol_10d:.2%}",
                            max_vol=f"{self._max_entry_vol_annual:.2%}",
                        )
                        self._reject("vol_gate")
                        continue

            # R12-C: AEKF (OU-Kou-GARCH) direction veto removed — the GARCH
            # family moved to deprecated/research/ and RangePredictor no longer
            # produces ou_jump_state, so the veto could never fire again.

            # Rising IV rank filter: if IV rank has risen by more than iv_rank_rise_threshold
            # over the last 10 days, market is in directional stress — skip iron condor entry.
            # Rise value is always logged (not just on veto) for threshold tuning.
            # W71c: backtest.iv_rank_rise_gate=False skips the VETO only (research-only gate; live
            # has none): the rise is still computed and recorded so a study can evaluate it later.
            if strategy in NEUTRAL_CREDIT_GATED and not features_df.empty:
                if "iv_rank" in features_df.columns and len(features_df) >= 11:
                    iv_rank_series = features_df["iv_rank"].iloc[-11:]
                    iv_rank_rise = float(iv_rank_series.iloc[-1]) - float(iv_rank_series.iloc[0])
                    _entry_decision["iv_rank_rise_10d"] = round(iv_rank_rise, 4)
                    if self._iv_rank_rise_gate and iv_rank_rise > self._iv_rank_rise_threshold:
                        _entry_decision["iv_rank_veto"] = {
                            "rise": round(iv_rank_rise, 4),
                            "threshold": self._iv_rank_rise_threshold,
                        }
                        log.debug(
                            "iv_rank_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            iv_rank_rise=round(iv_rank_rise, 4),
                            threshold=self._iv_rank_rise_threshold,
                        )
                        self._reject("iv_rank_rise")
                        continue

            # Drawdown-from-60d-high gate: block iron condor entry when the underlying is
            # in a sustained decline relative to its 60-day rolling high.  Catches slow-grind
            # bear phases that iv_rank_rise misses (W12 tariff shock: rise=0.021 but -16% from high).
            # Default threshold -1.0 disables the gate; Optuna tunes per window in [-0.15, -0.05].
            # R16: this was the ONE neutral-credit gate the string-patched
            # builder rollout missed — every sibling gate above already reads
            # NEUTRAL_CREDIT_GATED, so an activated threshold gave the
            # iron_condor baseline a drawdown veto the butterfly/wide/broken/
            # jade/ccs arms silently skipped, biasing any promotion study.
            if strategy in NEUTRAL_CREDIT_GATED and self._pct_from_60d_high_threshold > -0.99:
                _close_series = hist_dec["Close"]
                _rolling_high = _close_series.rolling(min(60, len(_close_series))).max().iloc[-1]
                if _rolling_high and _rolling_high > 0:
                    _pct_from_high = (float(_close_series.iloc[-1]) - float(_rolling_high)) / float(_rolling_high)
                    _entry_decision["pct_from_60d_high"] = round(_pct_from_high, 4)
                    if _pct_from_high < self._pct_from_60d_high_threshold:
                        _entry_decision["drawdown_veto"] = {
                            "pct_from_high": round(_pct_from_high, 4),
                            "threshold": self._pct_from_60d_high_threshold,
                        }
                        log.debug(
                            "drawdown_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            pct_from_high=round(_pct_from_high, 4),
                            threshold=self._pct_from_60d_high_threshold,
                        )
                        self._reject("pct_from_high")
                        continue

            # --- 3. Build the trade ---
            # Intraday entry window gate (Fix 1g / Gap D): when 5-min data is available,
            # only enter during the configured ET window and simulate a limit-order fill.
            entry_time_str: str | None = None
            limit_price: float | None = None
            fill_time_str: str | None = None
            _session_bars = None
            if self._intraday_store is not None:
                from ait.data.historical import HistoricalDataStore
                session_bars = self._intraday_store.load_intraday_range(
                    symbol=self._symbol,
                    start_date=today_date,
                    end_date=today_date,
                )
                if not session_bars.empty:
                    _session_bars = session_bars
                    # Find first bar in the entry window
                    window_bars = session_bars[
                        session_bars.index.to_series().apply(
                            lambda ts: self._is_in_entry_window(ts)
                        )
                    ]
                    if window_bars.empty:
                        self._reject("no_bars_in_entry_window")
                        continue  # No bars in window today — skip entry

                    # Use VWAP of first scan bar as limit price (mid-market proxy)
                    first_scan_bar = window_bars.iloc[0]
                    scan_time = window_bars.index[0]
                    entry_time_str = scan_time.isoformat()

                    # Build partial hist up to scan time for feature computation
                    hist_partial = HistoricalDataStore.slice_intraday_up_to(
                        session_bars, scan_time.time()
                    )
                    if hist_partial.empty:
                        limit_price = float(first_scan_bar["Close"])
                    else:
                        limit_price = float(hist_partial["Close"].iloc[-1])

                    # Try to fill limit order on subsequent bars
                    bars_after_scan = window_bars.iloc[1:]
                    filled, bars_waited, fill_time_str = self._try_limit_fill(
                        limit_price, bars_after_scan, self._limit_order_timeout_bars
                    )
                    if not filled:
                        self._reject("limit_not_filled")
                        continue  # Limit order expired without fill — skip entry

            # W78 (realistic): the exit step above evaluated every held position over the WHOLE
            # session, so a position that closes at 15:00 has already left `open_positions` by
            # the time a 10:30 entry is considered. At the fill instant it was still alive:
            # count it, or max_concurrent_positions is violated intraday (measured in W77: 20 of
            # 179 entries at 3 DTE, 7 of 136 at 7 DTE).
            if self._realistic and fill_time_str is not None and closed_today:
                _fill_ts = datetime.fromisoformat(fill_time_str).astimezone(ET)
                _alive = len(open_positions) + sum(
                    1 for _p in closed_today if self._exit_instant(_p) >= _fill_ts)
                if _alive >= self._max_concurrent_positions:
                    self._reject("max_concurrent_intraday")
                    continue

            # R20 #1: run() has had self._market_context all along (walkforward
            # passes the window-aligned VIX/SPY frames) but never forwarded it
            # here, so _get_iv's priority-2 VIX branch and the R16 per-symbol
            # vol calibration (VXN/VIX 1.228 QQQ, 1.33 IWM) were DEAD at
            # entry-pricing time — every study priced entries off the
            # priority-3 synthetic fallback realized_vol*1.15.
            # W74 (realistic): what was knowable at the FILL — the limit price the
            # order rested at, the fill time, and history through the previous
            # session only (W72 lead L2).
            fill_ctx = None
            if self._realistic and fill_time_str is not None and limit_price is not None:
                fill_ctx = {"spot": float(limit_price),
                            "as_of": datetime.fromisoformat(fill_time_str).astimezone(ET),
                            "iv_hist": hist.iloc[:-1]}
            pos = self._build_position(
                strategy, direction, row, hist, today_date, capital,
                market_context=self._market_context, fill_ctx=fill_ctx,
            )
            if pos is None:
                self._reject("builder_none")
                continue

            # W84: live's ex-dividend ENTRY block (W51, orchestrator trade_blocked_ex_div),
            # checked where live checks it: on the BUILT position's real legs and expiry.
            if self._ex_div_mode != "off" and self._ex_dates and sells_a_call(pos):
                _exp_d = date.fromisoformat(str(pos["expiry_date"])[:10])
                _blocked, _ex_d = ex_div_blocks(self._ex_div_mode, self._ex_dates, today_date,
                                                _exp_d, self._ex_div_k)
                if _blocked:
                    _entry_decision["ex_div"] = {"mode": self._ex_div_mode, "ex_date": str(_ex_d),
                                                 "expiry": str(_exp_d)}
                    log.debug("ex_div_entry_block", date=str(today_date), ex_date=str(_ex_d),
                              expiry=str(_exp_d), mode=self._ex_div_mode)
                    self._reject("ex_div_entry")
                    continue

            if entry_time_str:
                pos["entry_time"] = entry_time_str
            if limit_price is not None:
                pos["limit_price"] = round(float(limit_price), 4)
            if fill_time_str is not None:
                pos["fill_time"] = fill_time_str

            # Store signal context for MetaLabeler training (Gap Z9)
            pos["entry_confidence"] = round(float(confidence), 4)
            pos["entry_direction"] = direction.value if hasattr(direction, "value") else str(direction)
            if not features_df.empty:
                last_f = features_df.iloc[-1]
                pos["entry_iv_rank"] = round(float(last_f.get("iv_rank", 0.0)), 4)
                pos["entry_vix_level"] = round(float(last_f.get("vix_level", 0.0)), 4)
                vol_expanding = float(last_f.get("vol_regime_expanding", 0.0)) > 0.5
                px_vs_sma = float(last_f.get("price_vs_sma_20", 0.0))
                pos["entry_regime"] = research_regime(
                    vol_expanding, px_vs_sma, self._regime_down, self._regime_up)
                # Features snapshot for dashboard decision drawer
                pos["features_at_entry"] = {
                    "rsi_14":              round(float(last_f.get("rsi_14", 0.0)), 2),
                    "macd_hist":           round(float(last_f.get("macd_hist", 0.0)), 5),
                    "bb_position":         round(float(last_f.get("bb_position", 0.0)), 3),
                    "atr_pct":             round(float(last_f.get("atr_pct", 0.0)), 4),
                    "realized_vol_20":     round(float(last_f.get("realized_vol_20", 0.0)), 4),
                    "iv_rank":             round(float(last_f.get("iv_rank", 0.0)), 3),
                    "vix_level":           round(float(last_f.get("vix_level", 0.0)), 2),
                    "hurst_wavelet":       round(float(last_f.get("hurst_wavelet", 0.0)), 3),
                    # R12-C: sentiment_composite / put_call_ratio dropped with
                    # the sentiment/flow feature retirement (constant columns).
                }
            else:
                pos["entry_regime"] = "range_bound"  # default when history too short
                pos["features_at_entry"] = {}

            # Finalize decision chain with resolved regime
            _entry_decision["direction_conf"] = round(float(confidence), 4)
            _entry_decision["regime"] = pos.get("entry_regime", "range_bound")
            pos["decision"] = _entry_decision

            # Iron condor leg structure for the dashboard drawer
            if pos.get("strategy") in ("iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor"):
                ep = pos.get("entry_price", 0.0)  # net credit per share
                # Each spread contributes ep/2 to the net credit.
                # Short leg = ep/2 + wing_cost; long leg = -wing_cost (debit).
                # Using wing_cost = ep*0.05 so net = 2*(0.55-0.05)*ep = ep.
                pos["legs"] = [
                    {"type": "short_put",  "strike": pos.get("short_put_strike"),  "premium": round(ep * 0.55, 4)},
                    {"type": "long_put",   "strike": pos.get("long_put_strike"),   "premium": round(-ep * 0.05, 4)},
                    {"type": "short_call", "strike": pos.get("short_call_strike"), "premium": round(ep * 0.55, 4)},
                    {"type": "long_call",  "strike": pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            elif pos.get("strategy") in ("put_credit_spread", "call_credit_spread"):
                ep = pos.get("entry_price", 0.0)
                pos["legs"] = [
                    {"type": "short", "strike": pos.get("short_put_strike") or pos.get("short_call_strike"), "premium": round(ep * 1.05, 4)},
                    {"type": "long",  "strike": pos.get("long_put_strike")  or pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            elif pos.get("strategy") == "jade_lizard":
                # R16: jade_lizard fell to the else below, so it got legs=[]
                # and NO credit/max_loss keys — result.py:194 then fell back to
                # `abs(pnl)*2` as the risk proxy, computing the arm's
                # capital_utilization / cash_drag_adjusted_return off a P&L
                # multiple instead of margin, and rendering an empty leg drawer.
                # 3 legs: short put + short call + long call wing. Premium split
                # follows the sibling branches' credit-share convention.
                ep = pos.get("entry_price", 0.0)
                pos["legs"] = [
                    {"type": "short_put",  "strike": pos.get("short_put_strike"),  "premium": round(ep * 0.55, 4)},
                    {"type": "short_call", "strike": pos.get("short_call_strike"), "premium": round(ep * 0.50, 4)},
                    {"type": "long_call",  "strike": pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                # Undefined-risk put side: max_loss_per_share is the builder's
                # (short_put_strike - credit) stress figure, NOT a structural cap.
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            else:
                pos["legs"] = []

            # Deep-audit BT-M6: entry commission is included in _calc_pnl
            # (subtracted from the trade P&L added back at exit) — debiting
            # capital here as well double-counted it in final_capital.
            # W5 model-vs-reality-commission-constant: the debit is now the
            # ALL-IN per-leg cost (commission + regulatory/exchange/clearing
            # fees), charged per leg on entry and again on exit in _calc_pnl.
            n_legs = pos.get("n_legs", 1)
            entry_commission = self._cost_per_contract_leg * pos["contracts"] * n_legs
            pos["entry_commission"] = entry_commission

            self._funnel["entries_taken"] += 1
            # W74 (realistic): a position opened at T is exit-checked on the bars
            # AFTER T the same session. The legacy loop exits before it enters, so
            # a same-day round trip (0-DTE, or a touch an hour after the fill)
            # was impossible.
            _same_day_exit = None
            if self._realistic and fill_time_str is not None and _session_bars is not None:
                _after = _session_bars[_session_bars.index > datetime.fromisoformat(fill_time_str)]
                _same_day_exit = self._check_intraday_exit(pos, _after, today_date)
            if _same_day_exit is not None:
                pos.update(_same_day_exit)
                capital += pos["pnl"]
                trades.append(pos)
                log.debug("trade_closed_same_session", strategy=pos["strategy"],
                          pnl=f"{pos['pnl']:.2f}", reason=pos["exit_reason"])
            else:
                open_positions.append(pos)

        # --- Force-close remaining positions ---
        last_row = self._data.iloc[-1]
        last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
        for pos in open_positions:
            exit_info = self._force_close(pos, last_row, last_date, self._data)
            pos.update(exit_info)
            capital += pos["pnl"]
            trades.append(pos)

        result = BacktestResult(
            trades=trades,
            initial_capital=self._initial_capital,
            final_capital=round(capital, 2),
            start_date=dates[0].date() if hasattr(dates[0], "date") else dates[0],
            end_date=last_date,
            entry_funnel=dict(self._funnel),
        )

        log.info(
            "backtest_complete",
            total_trades=result.total_trades,
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
            max_dd=f"{result.max_drawdown:.2%}",
            win_rate=f"{result.win_rate:.2%}",
        )
        if self._friction_bands is not None:
            _fs = self._friction_stats
            _tot = max(sum(_fs.values()), 1)
            log.info("friction_summary", fitted=_fs["fitted"], default=_fs["default"],
                     extrapolated=_fs["extrapolated"],
                     extrapolated_share=round(_fs["extrapolated"] / _tot, 4))
        if self._settlement_stats and (self._settlement_stats["settled"]
                                       or self._settlement_stats["pending"]):
            log.info("settlement_summary", **self._settlement_stats)
        if self._surface is not None:
            _h, _m = self._surface_stats["hit"], self._surface_stats["miss"]
            log.info("surface_pricing_summary", legs_priced=_h + _m, from_surface=_h,
                     fell_back=_m, coverage=round(_h / max(_h + _m, 1), 4))
        return result

    @classmethod
    def compare_exit_modes(
        cls, data: pd.DataFrame, strategies: list[str], **kwargs: Any
    ) -> dict:
        """Run backtest with both fixed and trailing stops, return comparison.

        R6 parity note: credit structures always use the flat-loss-limit +
        DTE-laddered TP path, so fixed-vs-trailing deltas reflect DEBIT
        trades only.
        """
        shared = {k: v for k, v in kwargs.items() if k != "trailing_stop_enabled"}

        fixed_bt = cls(data, strategies, trailing_stop_enabled=False, **shared)
        trailing_bt = cls(data, strategies, trailing_stop_enabled=True, **shared)

        fixed_result = fixed_bt.run()
        fixed_result.exit_mode = "fixed"
        trailing_result = trailing_bt.run()
        trailing_result.exit_mode = "trailing"

        return {
            "fixed": fixed_result,
            "trailing": trailing_result,
            "delta": {
                "total_return": trailing_result.total_return - fixed_result.total_return,
                "win_rate": trailing_result.win_rate - fixed_result.win_rate,
                "sharpe_ratio": trailing_result.sharpe_ratio - fixed_result.sharpe_ratio,
                "max_drawdown": trailing_result.max_drawdown - fixed_result.max_drawdown,
                "profit_factor": trailing_result.profit_factor - fixed_result.profit_factor,
            },
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has a DatetimeIndex and required columns."""
        df = data.copy()
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower.capitalize()
        if col_map:
            df = df.rename(columns=col_map)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Direction prediction
    # ------------------------------------------------------------------

    def _load_predictor(self) -> Any:
        """Try to load trained DirectionPredictor."""
        try:
            from ait.config.settings import MLConfig
            from ait.ml.ensemble import DirectionPredictor
            predictor = DirectionPredictor(MLConfig())
            if predictor.load_models():
                log.info("ml_predictor_loaded", version=predictor.model_version)
                return predictor
        except Exception as e:
            log.debug("ml_predictor_unavailable", reason=str(e))
        return None

    def _load_ex_dates(self) -> "list[date] | None":
        """W84: the symbol's stored ex-dividend dates, ascending, for the ex-dividend gate.

        Loaded only when the gate is on, so ``off`` opens no store and changes nothing.
        LOUD (W79a): a gate that is on without data would silently allow every entry and
        read as "live parity", so a missing or empty table raises with the fix.
        """
        if self._ex_div_mode == "off":
            return None
        sym = (self._symbol or "").upper()
        table = f"{self._research_table_prefix}dividends"
        fix = (f"run: python scripts/backfill_dividends.py --symbols {sym or '<SYMBOL>'} "
               f"--table-prefix {self._research_table_prefix}")
        if not sym:
            raise ValueError(f"ex_div_entry_gate={self._ex_div_mode!r} needs a symbol (the "
                             f"ex-dividend calendar is per symbol)")
        from ait.data.historical import HistoricalDataStore
        store = (HistoricalDataStore(db_path=self._db_path, table_prefix=self._research_table_prefix)
                 if self._db_path is not None
                 else HistoricalDataStore(table_prefix=self._research_table_prefix))
        dates = sorted(d for d, _amount in store.load_dividends(sym))
        if not dates:
            raise ValueError(f"ex_div_entry_gate={self._ex_div_mode!r} but {table} has no rows "
                             f"for {sym}; {fix}")
        log.info("ex_div_dates_loaded", symbol=sym, table=table, rows=len(dates),
                 first=str(dates[0]), last=str(dates[-1]), mode=self._ex_div_mode)
        return dates

    @staticmethod
    def _load_earnings_dates(symbol: str | None, before_days: int,
                             after_days: int) -> set[date]:
        """Pre-fetch historical earnings dates from yfinance (Gap Z4).

        Returns every calendar date inside the earnings danger window of any
        announcement, so the main loop can do an O(1) membership test: the window
        is live's — [earnings - before_days, earnings + after_days] — i.e. the
        days on which live's EarningsCalendar refuses entries
        (`-buffer_after <= days_until <= buffer_before`). W70: this used to build
        `range(-1, skip_days + 1)` around each date, i.e. 1 day before to
        `skip_days` AFTER — the mirror image of live's 2-before / 1-after.
        Returns an empty set if symbol is None, both buffers are 0, or yfinance
        is unavailable.
        """
        if not symbol or (before_days <= 0 and after_days <= 0):
            return set()
        try:
            import yfinance as yf
            from datetime import timedelta
            ticker = yf.Ticker(symbol)
            eds = getattr(ticker, "earnings_dates", None)
            if eds is None or (hasattr(eds, "empty") and eds.empty):
                return set()
            danger_dates: set[date] = set()
            for idx_val in eds.index:
                ed = idx_val.date() if hasattr(idx_val, "date") else idx_val
                for delta in range(-int(before_days), int(after_days) + 1):
                    danger_dates.add(ed + timedelta(days=delta))
            log.debug("earnings_dates_loaded", symbol=symbol, count=len(danger_dates))
            return danger_dates
        except Exception as e:
            log.debug("earnings_dates_load_failed", symbol=symbol, error=str(e))
            return set()

    def _parse_et_time(self, time_str: str):
        """Parse an HH:MM Eastern Time string into a datetime.time object."""
        from datetime import time as dt_time
        h, m = time_str.split(":")
        return dt_time(int(h), int(m))

    def _is_in_entry_window(self, bar_time) -> bool:
        """Return True if bar_time is within the ET entry window.

        Accepts a datetime.time (assumed already ET) or a timestamp. R5 audit
        C4: intraday bars are stored tz-aware UTC, but this compared the RAW
        clock time against ET strings — only 13:30-15:30 *UTC* bars passed,
        so every backtest entered solely in the first ~2h of the session
        (DST-shifting), regardless of the configured window. Convert
        tz-aware timestamps to America/New_York first.
        """
        from datetime import time as dt_time
        start = self._parse_et_time(self._entry_window_start_et)
        end = self._parse_et_time(self._entry_window_end_et)
        if isinstance(bar_time, dt_time):
            t = bar_time
        else:
            ts = bar_time
            if getattr(ts, "tzinfo", None) is not None:
                from zoneinfo import ZoneInfo
                ts = ts.tz_convert("America/New_York") if hasattr(ts, "tz_convert") \
                    else ts.astimezone(ZoneInfo("America/New_York"))
            t = ts.time()
        return start <= t < end

    def _check_intraday_exit(
        self, pos: dict, session_bars: "pd.DataFrame", current_date: date
    ) -> dict | None:
        """Check intraday 5-min bars for stop-loss or profit-target triggers.

        Returns exit dict if triggered on any bar, else None.
        """
        if self._realistic:
            return self._check_intraday_exit_realistic(pos, session_bars, current_date)
        if session_bars is None or session_bars.empty:
            return None

        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", pos.get("position_type", "debit"))
        expiry_str = pos.get("expiry_date") or pos.get("exit_date")
        if not expiry_str:
            return None
        expiry = date.fromisoformat(str(expiry_str)[:10])

        days_held = (current_date - date.fromisoformat(pos["entry_date"])).days
        remaining_dte = max(0, (expiry - current_date).days)
        exit_half_spread = self._options_half_spread(float(pos.get("entry_iv", 0.25)), remaining_dte)

        for row in session_bars.itertuples(index=True):
            underlying = row.Close
            current_val = self._reprice_position(pos, underlying, days_held, None)
            current_val = self._apply_exit_friction(
                pos, current_val, underlying, days_held, None, exit_half_spread, trade_type)
            if trade_type == "credit":
                pnl_pct = (entry_price - current_val) / entry_price if entry_price > 0 else 0.0
            else:
                pnl_pct = (current_val - entry_price) / entry_price if entry_price > 0 else 0.0

            # Check stop / profit — routed per trade type (R6 exit parity)
            result = self._dispatch_exit_check(pos, trade_type, pnl_pct, current_date)

            if result is not None:
                bar_ts = row.Index
                bar_dt = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)
                pnl = self._calc_pnl(pos, current_val)
                result["pnl"] = round(pnl, 2)
                result["exit_price"] = round(current_val, 4)
                result["exit_underlying"] = round(float(underlying), 4)
                result["exit_time"] = bar_dt
                return result

        return None

    def _check_intraday_exit_realistic(
        self, pos: dict, session_bars: "pd.DataFrame", current_date: date
    ) -> dict | None:
        """Intraday exit scan at live fidelity (W72 leads L3, L4).

        Differences from the legacy scan, each a measured defect:
          * L3 — the take-profit DECISION uses the mid mark vs the credit
            received (live's pnl_pct is the broker mark over cost basis); the
            spread is paid on the booked exit price. The legacy scan decided on
            the after-friction value, so the same price state exited on the
            daily path and not intraday.
          * L4 — each bar's OHLC is handed to the credit rules, so a short-strike
            TOUCH is evaluated per bar (live polls spot every 30 s). The legacy
            scan passed no row, so a take-profit bar later in the day beat an
            earlier pierce.
          * The decision instant is the bar END, which the expiry-day flatten and
            the hold fraction need.
        """
        if session_bars is None or session_bars.empty:
            return None
        if pos.get("settlement") is not None:
            return None   # W77: waiting for the next open
        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", pos.get("position_type", "debit"))
        expiry_str = pos.get("expiry_date") or pos.get("exit_date")
        if not expiry_str:
            return None
        expiry = date.fromisoformat(str(expiry_str)[:10])
        days_held = (current_date - date.fromisoformat(pos["entry_date"])).days
        remaining_dte = max(0, (expiry - current_date).days)
        exit_half_spread = self._options_half_spread(float(pos.get("entry_iv", 0.25)), remaining_dte)
        bar_len = pd.Timedelta(minutes=5)

        for row in session_bars.itertuples(index=True):
            underlying = row.Close
            bar_ts = row.Index
            bar_end = (bar_ts + bar_len)
            as_of = bar_end.to_pydatetime().astimezone(ET) if hasattr(bar_end, "to_pydatetime") else bar_end
            mid_value = self._reprice_position(pos, underlying, days_held, None, as_of)
            if trade_type == "credit":
                pnl_pct = (entry_price - mid_value) / entry_price if entry_price > 0 else 0.0
            else:
                pnl_pct = (mid_value - entry_price) / entry_price if entry_price > 0 else 0.0
            bar = pd.Series({"Open": float(row.Open), "High": float(row.High),
                             "Low": float(row.Low), "Close": float(row.Close)})
            result = self._dispatch_exit_check(pos, trade_type, pnl_pct, current_date, bar, as_of)
            if result is None:
                continue
            if result.pop("_settle", False):
                # W77: this bar ends at the bell. Nothing trades after it.
                return self._settle_at_bell(
                    pos, float(underlying), current_date,
                    exit_time=bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts))
            value, und = mid_value, underlying
            if result.get("exit_reason") == "touch_stop":
                value, und = self._touch_exit_mark(pos, result, days_held, None, trade_type, as_of)
            value = self._apply_exit_friction(
                pos, value, und, days_held, None, exit_half_spread, trade_type, as_of)
            pnl = self._calc_pnl(pos, value)
            result["pnl"] = round(pnl, 2)
            result["exit_price"] = round(value, 4)
            result["exit_underlying"] = round(float(und), 4)
            result["exit_time"] = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)
            return result
        return None

    def _try_limit_fill(
        self, limit_price: float, session_bars: "pd.DataFrame", timeout_bars: int
    ) -> "tuple[bool, int, str | None]":
        """Simulate limit order fill on subsequent 5-min bars.

        Returns (filled, bars_waited, fill_time_iso). fill_time_iso is the ISO
        timestamp of the bar where Low ≤ limit_price ≤ High, or None if unfilled.
        """
        bars = session_bars.iloc[:timeout_bars]
        mask = (bars["Low"] <= limit_price) & (limit_price <= bars["High"])
        if mask.any():
            first = int(mask.values.argmax())
            ts = bars.index[first]
            fill_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            return True, first + 1, fill_ts
        return False, min(len(session_bars), timeout_bars), None

    # R12-C: _load_directional_model removed — no callers; DirectionalModel
    # retired to deprecated/src/directional.py.

    def _get_direction(
        self,
        hist: pd.DataFrame,
        market_context: dict | None = None,
    ) -> tuple[SignalDirection, float, pd.DataFrame]:
        """Get market direction prediction and pre-computed feature matrix.

        Returns (direction, confidence, features_df). The features_df is shared
        with the fractal gate and _select_strategy to avoid double computation.
        market_context (VIX, SPY, macro) is forwarded to the predictor and
        range predictor so they use cross-asset features, matching live inference (Gap Z3).
        """
        from ait.ml.features import FeatureEngine
        _cache_hit = False
        if self._features_cache is not None and not self._features_cache.empty:
            today = pd.Timestamp(hist.index[-1]).normalize()
            mask = self._features_cache.index <= today
            features_df = self._features_cache[mask]
            if features_df.empty:
                features_df = FeatureEngine().compute(hist)
            else:
                _cache_hit = True
        else:
            features_df = FeatureEngine().compute(hist)

        if self._predictor is not None:
            try:
                if _cache_hit and not features_df.empty and hasattr(self._predictor, "predict_from_features"):
                    # Cache available — predict_from_features bypasses the FeatureEngine
                    # re-run inside predict(), reducing OOS cost from O(N) to O(1) per bar.
                    pred = self._predictor.predict_from_features(
                        features_df.iloc[-1], symbol=self._symbol or ""
                    )
                else:
                    pred = self._predictor.predict(hist, market_context=market_context)
                if pred is not None:
                    return pred.direction, pred.confidence, features_df
            except Exception:
                pass
        direction, confidence = self._simple_direction(hist)
        return direction, confidence, features_df

    @staticmethod
    def _quick_rsi(close: np.ndarray, period: int = 14) -> float:
        """Fast RSI calculation from close prices array."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _simple_direction(hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Estimate direction from recent returns and trend."""
        close = hist["Close"]
        ret_5 = close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0.0
        ret_20 = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0.0
        score = 0.6 * ret_5 + 0.4 * ret_20

        threshold = 0.005
        if score > threshold:
            return SignalDirection.BULLISH, min(0.5 + abs(score) * 10, 0.95)
        elif score < -threshold:
            return SignalDirection.BEARISH, min(0.5 + abs(score) * 10, 0.95)
        return SignalDirection.NEUTRAL, 0.4

    # ------------------------------------------------------------------
    # Strategy selection — now IV-aware
    # ------------------------------------------------------------------

    def _days_since_event(self, today: date) -> int | None:
        """Calendar days since the most recent post_event_date on/before today."""
        past = [d for d in self._post_event_dates if d <= today]
        return (today - past[-1]).days if past else None

    def _post_event_blocked(self, today: date) -> bool:
        """True when a research post-event blackout is set and today sits inside
        it: 0 <= days_since_event <= post_event_blackout_days."""
        n = self._post_event_blackout_days
        if n is None or not self._post_event_dates:
            return False
        since = self._days_since_event(today)
        return since is not None and 0 <= since <= n

    def _select_strategy(
        self,
        direction: SignalDirection,
        hist: pd.DataFrame,
        confidence: float = 0.65,
        features_df: pd.DataFrame | None = None,
    ) -> str | None:
        """Pick a strategy based on direction, IV regime, and available strategies.

        High IV → prefer credit strategies (sell expensive premium)
        Low IV → prefer debit strategies (buy cheap premium)
        """
        available = set(self._strategies)
        # R20 #1: a bare, context-free IV-estimation call sat here with its
        # result UNUSED — dead code that invited a future reader to assume
        # selection was VIX-aware. Removed rather than context-threaded: the
        # regime proxy below is realized-vol based.

        # IV rank proxy: compare current IV to its range
        close_arr = hist["Close"].values
        rv_short = realized_vol(close_arr, window=10)
        rv_long = realized_vol(close_arr, window=60) if len(close_arr) > 61 else rv_short
        iv_regime_high = rv_short > rv_long * 1.1  # Short-term vol elevated

        # BT-M8: iron_condor used to be FORCED whenever present, silently
        # never trading the other requested strategies (and misleading every
        # multi-strategy breakdown). Prefer it on NEUTRAL, but respect the
        # requested strategy set for directional signals.
        if direction == SignalDirection.NEUTRAL:
            candidates = available & CREDIT_STRATEGIES
        elif (self._condor_any_direction and available
              and available <= set(NEUTRAL_CREDIT_GATED)):
            # W66 live parity: a neutral-credit-only set (live's IC-only selector)
            # trades on any direction.
            candidates = available & CREDIT_STRATEGIES
        elif direction == SignalDirection.BULLISH:
            candidates = available & {"bull_call_spread", "long_call"}
        else:
            candidates = available & {"bear_put_spread", "long_put"}

        if not candidates:
            return None
        if "iron_condor" in candidates:
            return "iron_condor"  # preferred among the eligible set
        return sorted(candidates)[0]

    # ------------------------------------------------------------------
    # Position building — full BS pricing for each strategy type
    # ------------------------------------------------------------------

    def _get_leg_iv(self, base_iv: float, strike: float, underlying: float,
                    option_type: OptionType) -> float:
        """Apply vol skew: OTM puts carry extra IV, OTM calls carry mild extra IV.

        Legacy (pre-W5) linear model in log-moneyness space:
        - Put: +1% IV per 10% below ATM (e.g. 20% OTM put → +2% IV)
        - Call: +0.2% IV per 10% above ATM
        skew_factor=0.0 restores flat IV; default 1.0 approximates market skew.

        W5 finding model-vs-reality-skew-10x-flat: that put slope (0.10
        absolute IV per unit |log-moneyness|) is 9-11x FLATTER than the
        surface measured on the repo's own 3,159 IBKR quotes (1.141 SPY /
        0.983 QQQ / 0.916 IWM), which is why model wings were near-free —
        long put $0.08 against a $0.71-0.92 real chain mid — and every
        wing_k / wide-wing study preferred wider wings than the market
        actually prices. The correction is the config-backed MULTIPLICATIVE
        term below: leg IV picks up base_iv x skew_slope_per_pct_otm x
        (percentage points OTM), so the surface steepens with the vol level
        the way the measured one does.

        skew_slope_per_pct_otm = 0.0 reproduces the pre-W5 output EXACTLY
        (backward-compat escape hatch for reproducing an old study). The call
        side keeps its legacy 0.02 term on purpose: the same regression put
        the call-side slope at 0.064 / 0.112 / -0.089 — sign-unstable across
        symbols, so there is no measured slope to ship there.
        """
        # W72: the arithmetic lives in pricing.skewed_leg_iv (the parity replay
        # calls the same function); this method only supplies the engine's knobs.
        return skewed_leg_iv(base_iv, strike, underlying, option_type, self._skew_factor,
                             self._skew_slope_per_pct_otm, self._iv_floor)

    def _options_half_spread(self, iv: float, dte: int) -> float:
        """Compute the per-leg half bid-ask spread for an OTM option.

        Models the empirical relationship between IV, DTE, and spread width:
        - Higher IV → wider spreads (market makers demand more edge)
        - Lower DTE → wider spreads (gamma risk increases near expiry)

        Returns the half-spread as a fraction of the option mid price.
        """
        iv_term = self._spread_iv_sensitivity * max(0.0, iv - iv_model.HALF_SPREAD_REF_IV)
        dte_term = self._spread_dte_sensitivity * max(0.0, iv_model.HALF_SPREAD_REF_DTE - dte)
        return min(self._spread_cap, self._spread_base + iv_term + dte_term)

    def _get_iv(self, hist: pd.DataFrame, market_context: dict | None = None) -> float:
        """Return IV for the current bar, in priority order:

        1. Stored IBKR daily implied_vol from the backfill (most realistic).
        2. VIX proxy from market_context — accepts scalar or full DataFrame.
           QQQ IV ≈ VIX × 1.10 (QQQ carries a small premium over SPX vol).
        3. Synthetic fallback: realized_vol × 1.15.

        Returns raw estimated IV without flooring. Credit-strategy entry gating
        (iv < iv_floor → no trade) is enforced separately in _build_position so
        that _get_iv remains a pure estimation function.
        """
        # Priority 1: stored IBKR IV
        if "implied_vol" in hist.columns:
            last_iv = hist["implied_vol"].dropna()
            if not last_iv.empty:
                stored = float(last_iv.iloc[-1])
                if stored > 0:
                    return stored

        # Priority 2: VIX proxy from market context (scalar or DataFrame),
        # scaled PER SYMBOL. R16: a flat 1.10 multiplier was applied to every
        # underlying, but VIX measures SPX vol — QQQ and IWM are structurally
        # more volatile. Measured against the real implied indices and 10y
        # realized vol:
        #   VXN/VIX  (NDX vs SPX, implied)  median 1.228   <- QQQ's true anchor
        #   QQQ/SPY  (realized, 10y)        median 1.354
        #   IWM/SPY  (realized, 10y)        median 1.465
        # So 1.10 understated QQQ implied vol by ~12% at the median (and by
        # ~34% on days like 2026-08-11: VIX 15.6 vs VXN 22.9) — biasing every
        # simulated QQQ credit low and every expected-move wing narrow.
        # SPY takes 1.00 (SPY IV ~ VIX by construction). QQQ takes the
        # measured VXN/VIX median. IWM has no free implied index (RVX is not
        # served by yfinance), so it takes its realized ratio shrunk by the
        # same implied-vs-realized factor VXN/QQQ exhibits (1.228/1.354):
        # 1.465 * 0.907 ~ 1.33. Override per symbol via AIT_BT_VOL_MULT_<SYM>.
        if market_context:
            vix = market_context.get("vix_close") or market_context.get("vix")
            if vix is not None:
                mult = self._symbol_vol_multiplier()
                if hasattr(vix, "reindex"):
                    # Full VIX DataFrame — align to current hist and take last value
                    vix_aligned = vix["Close"].reindex(hist.index, method="ffill")
                    if not vix_aligned.empty:
                        vix_val = float(vix_aligned.iloc[-1])
                        if vix_val > 0:
                            return vix_val / 100.0 * mult
                else:
                    vix_val = float(vix)
                    if vix_val > 0:
                        return vix_val / 100.0 * mult

        # Priority 3: synthetic fallback (realized vol x the shared premium, W87a)
        return iv_model.synthetic_iv(hist["Close"].values)

    # W74: class-level defaults for the fidelity switch and the three short-DTE exit
    # knobs. Tests (and any harness) build partial engines with ``Backtester.__new__``
    # and set only what they need; those instances must stay on LEGACY semantics
    # (the same idiom as the W70 executor durations). __init__ overwrites all four.
    _engine_model: str = "legacy"
    _friction_short_dte: bool = False       # W76
    _friction_bands = None                  # W76: BandedFriction when friction_short_dte
    _pricing_source: str = "bs_legacy"      # W75
    _surface = None                         # W75: IVSurface when pricing_source == "surface"
    _realistic: bool = False
    _expiry_guard_dte: int | None = None
    _max_hold_fraction: float | None = None
    _flatten_minutes: int | None = None
    # W77: settlement knobs; the stats dict is created on first use for partial engines.
    _pin_band: float = 0.10
    _assignment_fee: float = 0.015
    _stock_commission: float = 0.01
    # W81: the former bare literals, so a partial `Backtester.__new__` engine keeps the legacy
    # behaviour; __init__ overwrites all four from config.
    _regime_veto_gate: bool = True
    _regime_down: float = -0.05
    _regime_up: float = 0.02
    # W84: a partial engine has no ex-dividend block (the schema default).
    _ex_div_mode: str = "off"
    _ex_div_k: int = 5
    _ex_dates: list | None = None
    _research_table_prefix: str = "research_"
    _risk_free_rate: float = 0.05
    _settlement_stats: dict | None = None

    # R16: VIX -> per-symbol implied-vol multipliers (see _get_iv). Unknown
    # symbols keep the historical 1.10 so nothing silently changes for
    # underlyings this calibration was never measured on.
    _VOL_MULT_VS_VIX = iv_model.VOL_MULT_VS_VIX      # W87a: aliases of the shared definitions
    _VOL_MULT_DEFAULT = iv_model.VOL_MULT_DEFAULT

    def _load_calibrated_spreads(self) -> None:
        """R16: adopt fitted per-symbol spread params when they exist.

        Silent no-op when the symbol has no calibration (keeps the config
        formula), so an uncalibrated underlying behaves exactly as before.
        """
        sym = (self._symbol or "").upper()
        if not sym:
            return
        try:
            from ait.data.historical import HistoricalDataStore
            # W52a: the store this run was pointed at, not the cwd default.
            _store = (HistoricalDataStore(db_path=self._db_path)
                      if self._db_path is not None else HistoricalDataStore())
            p = _store.load_spread_params(sym)
            if not p:
                log.info("calibrated_spreads_absent", symbol=sym,
                         db_path=str(getattr(_store, "_db_path", "")))
                return
            self._spread_base = float(p.get("spread_base", self._spread_base))
            self._spread_iv_sensitivity = float(
                p.get("spread_iv_sensitivity", self._spread_iv_sensitivity))
            self._spread_dte_sensitivity = float(
                p.get("spread_dte_sensitivity", self._spread_dte_sensitivity))
            self._calibrated_spreads = p
            log.info("calibrated_spreads_loaded", symbol=sym,
                     base=round(self._spread_base, 5),
                     iv_sens=round(self._spread_iv_sensitivity, 5),
                     dte_sens=round(self._spread_dte_sensitivity, 6))
        except Exception as e:  # noqa: BLE001 — never block a run on this
            log.debug("calibrated_spreads_unavailable", symbol=sym, error=str(e))

    def _load_leg_spread_model(self) -> None:
        """W66: adopt the fitted per-leg dollar half-spread model for this symbol.

        Absent model -> stay on the legacy % model and say so loudly: a silent fallback
        would put a 7-8x overcharge back into a run the operator believes is v2."""
        from ait.backtesting.friction import LegSpreadModel
        sym = (self._symbol or "").upper()
        row = None
        if sym:
            try:
                from ait.data.historical import HistoricalDataStore
                _store = (HistoricalDataStore(db_path=self._db_path)
                          if self._db_path is not None else HistoricalDataStore())
                row = _store.load_leg_spread_model(sym)
            except Exception as e:  # noqa: BLE001 — reported below, never blocks a run
                log.warning("friction_params_unreadable", symbol=sym, error=str(e))
        if not row:
            log.warning("friction_params_absent", symbol=sym,
                        note="friction_model=per_leg_usd requested but no fitted model; "
                             "using the LEGACY % model for this symbol")
            return
        self._leg_spread_model = LegSpreadModel.from_row(row)
        log.info("friction_model_loaded", symbol=sym, form=row["form"],
                 a=round(float(row["a"]), 5), b=round(float(row["b"]), 5), n=row.get("n"),
                 calibrated_on=row.get("calibrated_on"))
        if self._friction_short_dte:
            self._load_friction_bands(sym, _store)

    def _load_friction_bands(self, sym: str, store) -> None:
        """W76: per-DTE-band models. Only a band with >= friction_min_sessions_per_band
        collected sessions is used; every other short-DTE band is answered by the all-DTE
        model floored at friction_extrapolated_floor_usd and COUNTED as extrapolated."""
        from ait.backtesting.friction import BandedFriction, LegSpreadModel
        try:
            rows = store.load_leg_spread_band_models(sym)
        except Exception as e:  # noqa: BLE001 - reported, never blocks a run
            log.warning("friction_bands_unreadable", symbol=sym, error=str(e))
            rows = {}
        usable, short = {}, []
        for band, r in rows.items():
            if int(r.get("n_sessions") or 0) >= self._friction_min_sessions:
                usable[band] = LegSpreadModel.from_row(r)
            else:
                short.append((band, int(r.get("n_sessions") or 0)))
        self._friction_bands = BandedFriction(self._leg_spread_model, usable,
                                              self._friction_floor_usd)
        log.info("friction_bands_loaded", symbol=sym, fitted=sorted(usable),
                 below_min_sessions=short, min_sessions=self._friction_min_sessions,
                 floor_usd=self._friction_floor_usd,
                 note="0-6 DTE bands not listed under fitted are charged the all-DTE model "
                      "floored at the tick and reported as extrapolated")

    def _leg_friction_usd(self, capture: float, leg_mids, dte: int | None = None) -> float:
        """W66: dollars/share to cross the spread on every leg once (v2 model).

        W76: with friction_short_dte on and the position's DTE known, the cost comes from
        the DTE-band model (or the floored extrapolation) and the outcome is tallied."""
        from ait.backtesting.friction import banded_condor_leg_cost_usd, condor_leg_cost_usd
        if self._friction_bands is not None and dte is not None:
            cost, ex = banded_condor_leg_cost_usd(self._friction_bands, leg_mids, int(dte),
                                                  capture, self._friction_stress)
            self._friction_stats["extrapolated" if ex else "fitted"] += 1
            return cost
        if self._friction_bands is not None:
            self._friction_stats["default"] += 1
        return condor_leg_cost_usd(self._leg_spread_model, leg_mids, capture,
                                   self._friction_stress)

    def _remaining_dte(self, pos: dict, days_held: int, as_of: datetime | None) -> int | None:
        """Calendar days from the mark instant to this position's expiry (None if unknown)."""
        try:
            expiry = date.fromisoformat(str(pos["expiry_date"])[:10])
            on = (as_of.date() if as_of is not None else
                  date.fromisoformat(str(pos["entry_date"])[:10]) + timedelta(days=days_held))
        except (KeyError, ValueError, TypeError):
            return None
        return max((expiry - on).days, 0)

    def _apply_exit_friction(self, pos: dict, value: float, underlying: float,
                             days_held: int, hist: pd.DataFrame | None,
                             half_spread_pct: float, trade_type: str,
                             as_of: datetime | None = None) -> float:
        """Price the exit transaction cost onto a repriced value.

        v2 iron condor: + capture x sum(leg half-spreads) at the legs' CURRENT mids (buying
        back the shorts at the ask / selling the wings at the bid). Everything else (and
        the legacy model): the original `% x net value` shortcut."""
        if self._leg_spread_model is not None and pos.get("strategy") == "iron_condor":
            sc, sp, lc, lp = self._reprice_ic_legs(pos, underlying, days_held, hist, as_of)
            return value + self._leg_friction_usd(
                self._friction_exit_capture, (sc, sp, lc, lp),
                self._remaining_dte(pos, days_held, as_of) if self._friction_bands is not None
                else None)
        return value * (1 + half_spread_pct) if trade_type == "credit" \
            else value * (1 - half_spread_pct)

    def _symbol_vol_multiplier(self) -> float:
        """Per-symbol VIX scaling, env-overridable for sensitivity runs."""
        sym = (self._symbol or "").upper()
        env = os.environ.get(f"AIT_BT_VOL_MULT_{sym}")
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        return self._VOL_MULT_VS_VIX.get(sym, self._VOL_MULT_DEFAULT)

    def _build_position(
        self,
        strategy: str,
        direction: SignalDirection,
        row: pd.Series,
        hist: pd.DataFrame,
        today_date: date,
        capital: float,
        market_context: dict | None = None,
        fill_ctx: dict | None = None,
    ) -> dict | None:
        """Build a position dict with proper BS pricing for the strategy type.

        ``fill_ctx`` (realistic engine model, intraday fill only) carries what was
        knowable at the fill: ``spot`` (the limit price the order rested at),
        ``as_of`` (tz-aware fill time) and ``iv_hist`` (history through the
        PREVIOUS session). W72 lead L2: the legacy path priced a 10:30 fill at the
        daily CLOSE with the same-day stored IV.
        """
        underlying = row["Close"]
        dte = self._entry_dte
        expiry_override: date | None = None
        t_override: float | None = None
        iv_hist = hist
        if self._realistic:
            from ait.backtesting.expiry_calendar import listed_expiry
            expiry_override = listed_expiry(today_date, self._entry_dte)
            dte = (expiry_override - today_date).days
            if fill_ctx is not None:
                underlying = float(fill_ctx["spot"])
                t_override = time_to_expiry_years(fill_ctx["as_of"], expiry_override,
                                                  close=_session_close(expiry_override))
                iv_hist = fill_ctx["iv_hist"]
        iv = self._get_iv(iv_hist, market_context=market_context)
        if self._surface is not None:
            # W75: strike selection and the expected move use the TARGET EXPIRY's
            # at-the-money IV (live reads the chain's ATM IV for that expiry), not
            # the 30-day constant-maturity figure.
            _exp_atm = (expiry_override if expiry_override is not None
                        else today_date + timedelta(days=dte))
            _atm = self._surface.iv(today_date - timedelta(days=1), _exp_atm,
                                    float(underlying), float(underlying))
            if _atm is not None:
                iv = float(_atm)
        t = dte / 365.0 if t_override is None else t_override
        r = self._risk_free_rate     # W81: options.risk_free_rate (was a bare 0.05)

        # Entry gate for credit strategies: insufficient IV → premiums too small to
        # justify spread costs and risk. Honest no-trade rather than fake-pricing at floor.
        if strategy in CREDIT_STRATEGIES and iv < self._iv_floor:
            self._reject("builder:iv_floor")  # detail of the loop's builder_none
            return None

        if strategy in CREDIT_STRATEGIES:
            pos = self._build_credit_position(strategy, underlying, iv, t, r, dte, today_date, capital,
                                              expiry=expiry_override)
        else:
            pos = self._build_debit_position(strategy, direction, underlying, iv, t, r, dte, today_date, capital)
        # W64: live trades at most positions.max_contracts_per_trade contracts
        # (sizer hard cap, config 1). The engine sized off capital x pct alone,
        # so a larger account backtested 2-4 contracts/trade where live trades 1.
        # All dollar fields (credit, max_loss, commission, pnl) are derived from
        # `contracts` AFTER this point, so capping here is exact.
        if (pos is not None and self._max_contracts_per_trade
                and pos.get("contracts", 0) > self._max_contracts_per_trade):
            pos["contracts"] = self._max_contracts_per_trade
        return pos

    def _build_debit_position(
        self, strategy: str, direction: SignalDirection,
        S: float, iv: float, t: float, r: float, dte: int,
        today_date: date, capital: float,
    ) -> dict | None:
        """Build a debit (long option/spread) position."""

        if strategy == "long_call":
            strike = find_strike_by_delta(S, t, iv, self._delta_long, OptionType.CALL, r)
            price = black_scholes_price(S, strike, t, r,
                self._get_leg_iv(iv, strike, S, OptionType.CALL), OptionType.CALL)
            opt_type = "call"
        elif strategy == "long_put":
            strike = find_strike_by_delta(S, t, iv, -self._delta_long, OptionType.PUT, r)
            price = black_scholes_price(S, strike, t, r,
                self._get_leg_iv(iv, strike, S, OptionType.PUT), OptionType.PUT)
            opt_type = "put"
        elif strategy == "bull_call_spread":
            long_strike = find_strike_by_delta(S, t, iv, self._delta_long, OptionType.CALL, r)
            short_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            long_price = black_scholes_price(S, long_strike, t, r,
                self._get_leg_iv(iv, long_strike, S, OptionType.CALL), OptionType.CALL)
            short_price = black_scholes_price(S, short_strike, t, r,
                self._get_leg_iv(iv, short_strike, S, OptionType.CALL), OptionType.CALL)
            price = long_price - short_price  # Net debit
            strike = long_strike
            opt_type = "call"
            if price <= 0:
                return None
            return self._finalize_spread_position(
                strategy, "debit", price, S, iv, dte, today_date, capital,
                long_strike=long_strike, short_strike=short_strike, opt_type="call",
            )
        elif strategy == "bear_put_spread":
            long_delta = -self._delta_long
            short_delta = -self._delta_short
            long_strike = find_strike_by_delta(S, t, iv, long_delta, OptionType.PUT, r)
            short_strike = find_strike_by_delta(S, t, iv, short_delta, OptionType.PUT, r)
            long_price = black_scholes_price(S, long_strike, t, r,
                self._get_leg_iv(iv, long_strike, S, OptionType.PUT), OptionType.PUT)
            short_price = black_scholes_price(S, short_strike, t, r,
                self._get_leg_iv(iv, short_strike, S, OptionType.PUT), OptionType.PUT)
            price = long_price - short_price  # Net debit
            strike = long_strike
            opt_type = "put"
            if price <= 0:
                return None
            return self._finalize_spread_position(
                strategy, "debit", price, S, iv, dte, today_date, capital,
                long_strike=long_strike, short_strike=short_strike, opt_type="put",
            )
        elif strategy == "long_strangle":
            # Buy OTM call + buy OTM put — profit from large moves in either direction.
            # Delta is IV-scaled: high IV → go further OTM for cheaper entry on vol expansion.
            iv_scale = max(IV_SCALE_MIN, min(IV_SCALE_MAX, self._iv_floor / iv))
            effective_delta = self._delta_long * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(RESEARCH_DELTA_MIN, min(RESEARCH_DELTA_MAX, effective_delta))
            call_strike = find_strike_by_delta(S, t, iv, effective_delta, OptionType.CALL, r)
            put_strike  = find_strike_by_delta(S, t, iv, -effective_delta, OptionType.PUT, r)

            call_price = black_scholes_price(S, call_strike, t, r,
                self._get_leg_iv(iv, call_strike, S, OptionType.CALL), OptionType.CALL)
            put_price  = black_scholes_price(S, put_strike, t, r,
                self._get_leg_iv(iv, put_strike, S, OptionType.PUT), OptionType.PUT)

            total_cost = (call_price + put_price) * (1 + self._slippage_pct)
            cost_per_contract = total_cost * 100
            if cost_per_contract <= 0 or capital < cost_per_contract:
                return None

            contracts = int(capital * self._position_size_pct / cost_per_contract)
            if contracts < 1:
                if cost_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None

            return {
                "symbol": "SIM", "strategy": "long_strangle",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "debit",
                "entry_date": str(today_date), "entry_price": round(total_cost, 4),
                "contracts": contracts, "n_legs": 2,
                "long_call_strike": round(call_strike, 0),
                "long_put_strike": round(put_strike, 0),
                "strike": round(S, 0), "option_type": "strangle",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }
        else:
            return None

        # Single-leg sizing
        price *= (1 + self._slippage_pct)  # Buy at ask
        cost_per_contract = price * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None
        contracts = int(capital * self._position_size_pct / cost_per_contract)
        if contracts < 1:
            return None

        return {
            "symbol": "SIM",
            "strategy": strategy,
            "direction": direction.value,
            "trade_type": "debit",
            "entry_date": str(today_date),
            "entry_price": round(price, 4),
            "contracts": contracts,
            "n_legs": 1,
            "strike": round(strike, 0),
            "option_type": opt_type,
            "entry_iv": round(iv, 4),
            "underlying_at_entry": round(S, 2),
            "expiry_date": str(today_date + timedelta(days=dte)),
            "high_water_mark": 0.0,
        }

    def _log_unaffordable(self, strategy: str, risk_per_contract: float,
                          capital: float) -> None:
        """R16: attribute a zero-trade credit arm to SIZING, not to 'no edge'.

        The `contracts < 1` budget gate is intentional policy (refusing a
        structure whose single-contract risk exceeds position_size_pct of
        capital), and it is shared by every condor-family builder. But it was
        completely silent, so a shadow-tournament arm whose per-contract risk
        never fits the budget at index-ETF prices (jade_lizard's S*0.20*100
        margin is the extreme case) reported n=0 with no error and no reason.
        """
        log.debug(
            "credit_position_unaffordable",
            component="backtesting.engine",
            strategy=strategy,
            risk_per_contract=round(float(risk_per_contract), 2),
            budget=round(float(capital) * self._position_size_pct, 2),
            capital=round(float(capital), 2),
        )

    def _build_credit_position(
        self, strategy: str,
        S: float, iv: float, t: float, r: float, dte: int,
        today_date: date, capital: float, expiry: date | None = None,
    ) -> dict | None:
        """Build a credit (short premium) position.

        Iron condor: sell OTM call+put, buy further OTM wings.
        P&L is inverted: collect premium upfront, buy back cheaper later.
        """
        if strategy == "iron_condor":
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)

            # Wings: vol-scaled by wing_k; wing_floor_dollars is the hard safety minimum.
            # wing_k=1.0 → 1-sigma expected move; wing_k=2.0 → 2-sigma (wider, safer).
            if self._realistic:
                # W74: fractional years, so a 0-DTE entry with hours left has a
                # small positive move instead of em_unavailable.
                expected_move = _expected_move(S, iv, t)
            else:
                expected_move = S * iv * (dte / 365.0) ** 0.5
            # W52a: the live EM sanity gate (iron_condor.py R12-B), same
            # rule and same contract keys: each short strike's distance from
            # spot must sit within [em_min, em_max] x expected move; reject
            # when the EM is unavailable. On unless em_gate is off (see
            # __init__); the ratios are recorded on the position either way.
            if expected_move <= 0:
                log.debug("condor_rejected_em_unavailable", date=str(today_date))
                self._reject("builder:em_unavailable")  # detail of the loop's builder_none
                return None
            em_ratio_call = (short_call_strike - S) / expected_move
            em_ratio_put = (S - short_put_strike) / expected_move
            if self._em_gate and not (
                    self._ic_em_min <= em_ratio_put <= self._ic_em_max
                    and self._ic_em_min <= em_ratio_call <= self._ic_em_max):
                log.debug("condor_rejected_em_band", date=str(today_date),
                          em_ratio_call=round(em_ratio_call, 3),
                          em_ratio_put=round(em_ratio_put, 3),
                          em_min=self._ic_em_min, em_max=self._ic_em_max)
                self._reject("builder:em_band")  # detail of the loop's builder_none
                return None
            wing_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            wing_capped = False
            if self._budget_aware_wings:
                # W64: live parity (iron_condor.py _vol_scaled_width). The per-trade
                # budget is capital x position_size_pct; the widest wing it can
                # carry is IronCondor.affordable_width(budget) (the ONE authority,
                # W31), floor relaxed to $1 when budget-capped. Without this the
                # engine built the full wing_k x EM wing (~$50 on SPY) at any
                # account size and merely refused it, so a small account backtested
                # ZERO trades where live builds a $2-wide condor.
                from ait.strategies.iron_condor import IronCondor as _IC
                _budget = float(capital) * self._position_size_pct
                _afford = _IC.affordable_width(_budget)
                if _afford < wing_width:
                    wing_width = float(max(IC_BUDGET_CAPPED_MIN_WING, round(_afford)))
            # Research-only budget cap, mirroring live _vol_scaled_width:
            # width <= budget / (100 * max(0.5, 1 - credit/width floor)).
            if self._max_loss_cap is not None:
                _affordable = self._max_loss_cap / (
                    100.0 * max(0.5, 1.0 - self._ic_min_credit_width))
                if _affordable < wing_width:
                    wing_width = max(1.0, _affordable)
                    wing_capped = True
            long_call_strike = short_call_strike + wing_width
            long_put_strike = short_put_strike - wing_width
            if self._realistic:
                # W74 / W72 lead L6: price the strikes that are STORED and re-marked.
                # The legacy build priced the unrounded long strikes, then stored
                # them rounded, so a fresh condor re-marked away from its entry
                # credit on day 0. The wing is now the WIDER of the two rounded
                # wings (what live uses for credit/width and max loss).
                long_call_strike = float(round(long_call_strike))
                long_put_strike = float(round(long_put_strike))
                _call_w = long_call_strike - short_call_strike
                _put_w = short_put_strike - long_put_strike
                if min(_call_w, _put_w) <= 0:
                    self._reject("builder:wing_degenerate")  # detail of the loop's builder_none
                    return None
                wing_width = max(_call_w, _put_w)

            # Price each leg with its own skewed IV (W72: pricing.ic_leg_marks,
            # the same call the daily re-mark and the parity replay use)
            if self._surface is not None:
                _exp = expiry if expiry is not None else today_date + timedelta(days=dte)
                _ivs = self._surface_leg_ivs(
                    _exp, (short_call_strike, short_put_strike, long_call_strike, long_put_strike),
                    S, today_date, iv)
                short_call_price, short_put_price, long_call_price, long_put_price = \
                    ic_leg_marks_from_ivs(S, short_call_strike, short_put_strike,
                                          long_call_strike, long_put_strike, t, r, *_ivs)
            else:
                short_call_price, short_put_price, long_call_price, long_put_price = ic_leg_marks(
                    S, short_call_strike, short_put_strike, long_call_strike, long_put_strike,
                    t, r, iv, self._skew_factor, self._skew_slope_per_pct_otm, self._iv_floor)

            # Net credit received at mid prices
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                self._reject("builder:net_credit_nonpositive")  # detail of the loop's builder_none
                return None

            # R6 live-parity construction gates (src/ait/strategies/iron_condor.py
            # applies both to the MID-price total credit, before friction):
            # 1. Cost floor — live rejects total_credit < AIT_IC_MIN_CREDIT
            #    ($0.70 default): at a 50% TP the gross must clear ~3x the
            #    round-trip commission/crossing cost.
            if net_credit < self._ic_min_credit:
                self._reject("builder:credit_floor")  # detail of the loop's builder_none
                return None
            # 2. Credit-to-width gate — live rejects total_credit/max_width <
            #    AIT_IC_MIN_CREDIT_WIDTH (0.20 default). Engine wings are
            #    symmetric, so max_width == wing_width.
            if wing_width > 0 and (net_credit / wing_width) < self._ic_min_credit_width:
                self._reject("builder:credit_width_ratio")  # detail of the loop's builder_none
                return None

            # W64: the MID credit, before friction. Live's budget gate
            # (iron_condor.py max_loss > risk_budget) is evaluated on mid-price
            # strikes' credit; the engine deducted half-spread friction first and
            # then checked the budget against the inflated max loss, refusing
            # trades live takes at small size. Kept for the budget-aware path.
            _mid_credit = net_credit
            # Per-leg spread cost: 4 legs × half-spread (sell at bid, buy at ask)
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            spread_cost = 4 * half_spread * avg_leg_mid
            if self._leg_spread_model is not None:
                # W66 v2: the fitted DOLLAR half-spread on each of the four legs
                spread_cost = self._leg_friction_usd(
                    self._friction_entry_capture,
                    (short_call_price, short_put_price, long_call_price, long_put_price),
                    dte if self._friction_bands is not None else None)
            net_credit = max(0.0, net_credit - spread_cost)
            if net_credit <= 0:
                self._reject("builder:credit_after_friction_nonpositive")  # detail of the loop's builder_none
                return None

            # Max loss = wing width - net credit (per share)
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            # Position sizing based on max loss (margin requirement)
            import math
            # W64: budget-aware runs gate affordability + size on the MID-credit
            # max loss (live parity, see _mid_credit above). The recorded
            # position (max_loss_per_share below, P&L) still carries the friction.
            _gate_per_contract = (
                (wing_width - _mid_credit) * 100
                if self._budget_aware_wings else max_loss_per_contract)
            if (_gate_per_contract <= 0 or math.isnan(_gate_per_contract)
                    or capital < _gate_per_contract):
                self._reject("builder:capital_below_max_loss")  # detail of the loop's builder_none
                return None
            # Research-only: live rejects a built condor whose max loss still
            # exceeds the budget (iron_condor.py, max_loss > risk_budget).
            if self._max_loss_cap is not None and max_loss_per_contract > self._max_loss_cap:
                self._reject("builder:research_max_loss_cap")  # detail of the loop's builder_none
                return None
            contracts = int(capital * self._position_size_pct / _gate_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, _gate_per_contract, capital)
                self._reject("builder:unaffordable_contracts")  # detail of the loop's builder_none
                return None
            if self._contracts_cap is not None:
                contracts = min(contracts, self._contracts_cap)

            return {
                "symbol": "SIM",
                "strategy": "iron_condor",
                "direction": SignalDirection.NEUTRAL.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),  # Credit received per share
                "mid_credit": round(_mid_credit, 4),  # W66: before friction
                "entry_friction_usd": round(spread_cost, 4),  # W66: per share, whole condor
                "contracts": contracts,
                "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0),  # Reference: underlying at entry
                "option_type": "iron_condor",
                "entry_iv": round(iv, 4),
                "em_ratio_call": round(em_ratio_call, 4),  # W52a
                "em_ratio_put": round(em_ratio_put, 4),
                "wing_width": round(wing_width, 2),
                "wing_capped": wing_capped,
                "max_loss_per_contract": round(max_loss_per_contract, 2),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                # W65: the per-contract risk the BUDGET gate used (mid-credit when
                # budget-aware, else the friction-inclusive max loss). The walk-forward
                # pool allocator reuses it so its affordability/aggregate-risk gates
                # agree with the engine's, instead of re-rejecting what the engine took.
                "budget_gate_risk_per_contract": round(_gate_per_contract, 2),
                "expiry_date": str(expiry if expiry is not None
                                   else today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
                **({"engine_model": "realistic", "entry_dte_actual": int(dte)}
                   if self._realistic else {}),
            }

        elif strategy in ("wide_wing_condor", "broken_wing_condor"):
            # SHADOW ROUND 3 (PLAN 2026-08-04, pre-registered): wings-cost
            # interpolants between the condor (PF 0.97) and the wings-free
            # strangle benchmark (PF 1.30). wide_wing: BOTH wings at 2x the
            # standard k*EM distance; broken_wing: call wing standard, put
            # wing 2x (skew makes the put wing the expensive insurance).
            # REGISTERED DEVIATION from the live ratio gate: the credit floor
            # is held ABSOLUTE at the standard-width equivalent
            # (0.20 * std_width) so the entry population matches the live
            # condor; the ratio floor scales by std/max width.
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            std_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            call_width = std_width * (2.0 if strategy == "wide_wing_condor" else 1.0)
            put_width = std_width * 2.0
            long_call_strike = short_call_strike + call_width
            long_put_strike = short_put_strike - put_width
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0 or net_credit < self._ic_min_credit:
                return None
            max_width = max(call_width, put_width)
            scaled_floor = self._ic_min_credit_width * (std_width / max_width)
            if max_width > 0 and (net_credit / max_width) < scaled_floor:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            net_credit = max(0.0, net_credit - 4 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None
            max_loss_per_share = max_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, max_loss_per_contract, capital)
                return None
            return {
                "symbol": "SIM",
                "strategy": strategy,
                "direction": SignalDirection.NEUTRAL.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),
                "contracts": contracts,
                "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0),
                "option_type": strategy,
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "iron_butterfly":
            # SHADOW TOURNAMENT candidate (PLAN 2026-08-03): condor variant
            # with BOTH shorts at-the-money — much larger credit, so the
            # credit/width ratio clears in low IV where condors starve.
            # Same wing logic (wing_k x expected move, floor applies).
            atm = round(S)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            short_call_strike = short_put_strike = float(atm)
            long_call_strike = atm + wing_width
            long_put_strike = atm - wing_width
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                return None
            if net_credit < self._ic_min_credit:
                return None
            if wing_width > 0 and (net_credit / wing_width) < self._ic_min_credit_width:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            net_credit = max(0.0, net_credit - 4 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, max_loss_per_contract, capital)
                return None
            return {
                "symbol": "SIM", "strategy": "iron_butterfly",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0), "option_type": "iron_butterfly",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "put_credit_spread":
            # Sell higher strike put, buy lower strike put = bullish credit spread
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)

            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)

            long_put_strike = short_put_strike - wing

            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)

            net_credit = short_put_price - long_put_price
            if net_credit <= 0:
                return None

            # Per-leg spread cost: 2 legs
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_put_price + long_put_price) / 2
            net_credit = max(0.0, net_credit - 2 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None

            max_loss_per_share = wing - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None

            # Position sizing
            size_pct = self._position_size_pct
            contracts = int(capital * size_pct / max_loss_per_contract)
            if contracts < 1:
                if max_loss_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None

            return {
                "symbol": "SIM",
                "strategy": "put_credit_spread",
                "direction": SignalDirection.BULLISH.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),
                "contracts": contracts,
                "n_legs": 2,
                "short_put_strike": round(short_put_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(short_put_strike, 0),
                "option_type": "put",
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "call_credit_spread":
            # SHADOW R2: bearish twin of put_credit_spread.
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            net_credit = short_call_price - long_call_price
            if net_credit <= 0:
                return None
            half_spread = self._options_half_spread(iv, dte)
            net_credit = max(0.0, net_credit - 2 * half_spread *
                             (short_call_price + long_call_price) / 2)
            if net_credit <= 0:
                return None
            max_loss_per_share = wing - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                if max_loss_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None
            return {
                "symbol": "SIM", "strategy": "call_credit_spread",
                "direction": SignalDirection.BEARISH.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 2,
                "short_call_strike": round(short_call_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "strike": round(short_call_strike, 0), "option_type": "call",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "jade_lizard":
            # SHADOW R2 BENCHMARK ONLY (PLAN 2026-08-03): short put + short
            # call spread. Naked put side => NEVER live-promotable at current
            # capital. Sized by the strangle margin convention (S*0.20*100).
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            net_credit = short_put_price + short_call_price - long_call_price
            if net_credit <= 0:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_mid = (short_put_price + short_call_price + long_call_price) / 3
            net_credit = max(0.0, net_credit - 3 * half_spread * avg_mid)
            if net_credit <= 0:
                return None
            margin_per_contract = S * 0.20 * 100
            if capital < margin_per_contract:
                return None
            contracts = int(capital * self._position_size_pct / margin_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, margin_per_contract, capital)
                return None
            return {
                "symbol": "SIM", "strategy": "jade_lizard",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 3,
                "short_put_strike": round(short_put_strike, 0),
                "short_call_strike": round(short_call_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "strike": round(S, 0), "option_type": "jade_lizard",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max(short_put_strike - net_credit,
                                                wing - net_credit), 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "short_strangle":
            # Sell OTM call + sell OTM put — no wings (naked short premium).
            # Delta is IV-scaled: high IV → go further OTM for more breathing room.
            iv_scale = max(IV_SCALE_MIN, min(IV_SCALE_MAX, self._iv_floor / iv))
            effective_delta = self._delta_short * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(RESEARCH_DELTA_MIN, min(RESEARCH_DELTA_MAX, effective_delta))
            short_call = find_strike_by_delta(S, t, iv, effective_delta, OptionType.CALL, r)
            short_put  = find_strike_by_delta(S, t, iv, -effective_delta, OptionType.PUT, r)

            call_price = black_scholes_price(S, short_call, t, r,
                self._get_leg_iv(iv, short_call, S, OptionType.CALL), OptionType.CALL)
            put_price  = black_scholes_price(S, short_put, t, r,
                self._get_leg_iv(iv, short_put, S, OptionType.PUT), OptionType.PUT)

            net_credit = call_price + put_price
            if net_credit <= 0:
                return None

            # Per-leg spread cost: 2 legs (both short)
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (call_price + put_price) / 2
            net_credit = max(0.0, net_credit - 2 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None

            # Margin approx: 20% of underlying per strangle (standard naked-option requirement)
            margin_per_contract = S * 0.20 * 100
            contracts = int(capital * self._position_size_pct / margin_per_contract)
            if contracts < 1:
                return None

            return {
                "symbol": "SIM", "strategy": "short_strangle",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 2,
                "short_call_strike": round(short_call, 0),
                "short_put_strike": round(short_put, 0),
                "strike": round(S, 0), "option_type": "strangle",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        return None

    def _finalize_spread_position(
        self, strategy: str, trade_type: str, net_cost: float,
        S: float, iv: float, dte: int, today_date: date, capital: float,
        long_strike: float, short_strike: float, opt_type: str,
    ) -> dict | None:
        """Finalize a vertical spread position."""
        net_cost *= (1 + self._slippage_pct)  # Slippage on debit
        cost_per_contract = net_cost * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None
        size_pct = self._position_size_pct
        contracts = int(capital * size_pct / cost_per_contract)
        if contracts < 1:
            # Allow 1 contract only if affordable within 25% of capital
            if cost_per_contract <= capital * 0.25:
                contracts = 1
            else:
                return None

        direction = SignalDirection.BULLISH if "bull" in strategy else SignalDirection.BEARISH

        return {
            "symbol": "SIM",
            "strategy": strategy,
            "direction": direction.value,
            "trade_type": "debit",
            "entry_date": str(today_date),
            "entry_price": round(net_cost, 4),
            "contracts": contracts,
            "n_legs": 2,
            "long_strike": round(long_strike, 0),
            "short_strike": round(short_strike, 0),
            "strike": round(long_strike, 0),
            "option_type": opt_type,
            "entry_iv": round(iv, 4),
            "underlying_at_entry": round(S, 2),
            "max_profit_per_share": round(abs(short_strike - long_strike) - net_cost, 4),
            "expiry_date": str(today_date + timedelta(days=dte)),
            "high_water_mark": 0.0,
        }

    # ------------------------------------------------------------------
    # Exit logic — handles both debit and credit positions
    # ------------------------------------------------------------------

    def _load_surface(self):
        """Build/load the real-bar IV surface for this symbol, or None (with a WARNING)."""
        sym = (self._symbol or "").upper()
        if not sym:
            log.warning("surface_pricing_unavailable", reason="engine has no symbol",
                        fallback="bs_legacy")
            return None
        try:
            from ait.backtesting.iv_surface import build_or_load
            from ait.data.historical import DB_PATH
            db = Path(self._db_path) if self._db_path is not None else DB_PATH
            surf = build_or_load(db, sym, db.parent / "research_cache")
        except Exception as exc:  # noqa: BLE001 - research must degrade loudly, not crash
            log.warning("surface_pricing_unavailable", symbol=sym, error=str(exc),
                        fallback="bs_legacy")
            return None
        if len(surf) == 0:
            log.warning("surface_pricing_unavailable", symbol=sym, reason="no smiles in the bars",
                        fallback="bs_legacy")
            return None
        log.info("surface_pricing_active", symbol=sym, smiles=len(surf),
                 first=str(surf.dates()[0]), last=str(surf.dates()[-1]))
        return surf

    def _surface_leg_ivs(self, expiry: date, strikes: tuple, S: float, on: date,
                         base_iv: float) -> tuple[float, float, float, float]:
        """Per-leg IVs (short call, short put, long call, long put) from the surface as it
        stood at the PREVIOUS session (assumption A3: nothing from ``on`` is used).

        A leg the surface cannot price falls back to the legacy base IV + skew and is
        counted in ``self._surface_stats`` (logged at the end of the run)."""
        opts = (OptionType.CALL, OptionType.PUT, OptionType.CALL, OptionType.PUT)
        surface_date = on - timedelta(days=1)
        out = []
        for K, opt in zip(strikes, opts):
            iv = self._surface.iv(surface_date, expiry, float(K), float(S))
            if iv is None:
                self._surface_stats["miss"] += 1
                iv = self._get_leg_iv(base_iv, float(K), float(S), opt)
            else:
                self._surface_stats["hit"] += 1
                iv = max(float(iv), self._iv_floor)
            out.append(iv)
        return out[0], out[1], out[2], out[3]

    def _position_dte(self, pos: dict) -> int:
        """W74: calendar days from THIS position's entry to ITS expiry.

        The re-mark used the engine-wide ``self._entry_dte``, which is only right
        while every position is entered with the same DTE and a synthetic
        ``entry + N`` expiry. Reading the position's own dates is identical in
        that legacy case (proved by the W72 reference run) and correct as soon
        as expiries differ (listed-session expiries, per-candidate DTE).
        """
        try:
            return (date.fromisoformat(str(pos["expiry_date"])[:10])
                    - date.fromisoformat(str(pos["entry_date"])[:10])).days
        except (KeyError, ValueError, TypeError):
            return self._entry_dte

    def _get_current_iv(self, pos: dict, hist: pd.DataFrame | None) -> float:
        """Mean-reverting IV: blends entry IV with current realized vol.

        Weights 70% toward entry IV (forward-looking anchor) and 30% toward
        the current realized vol × 1.15 premium, so vega P&L is non-zero
        without wild swings driven purely by realized vol noise.
        """
        entry_iv = pos.get("entry_iv") or pos.get("iv", self._iv_floor)
        if hist is None:
            return entry_iv
        return iv_model.remark_iv(entry_iv, hist["Close"].values, self._iv_floor)

    def _eod(self, d: date) -> datetime:
        """W74: the daily model's decision instant: the session close of ``d``."""
        return datetime.combine(d, _session_close(d), tzinfo=ET)

    def _remark_t(self, pos: dict, days_held: int, as_of: datetime | None) -> float:
        """Time to expiry, in years, for a re-mark.

        Legacy: whole calendar days, floored at 0.0001 y (~53 min) so a position
        never reprices at t=0. Realistic with a decision instant: the real
        fractional time to the expiry session's bell, so a 0-DTE position marked
        an hour after its fill has 4.5 hours of life, not 53 minutes. (The
        legacy form made a freshly-entered 0-DTE condor look instantly
        profitable: priced with 5.5 h at the fill, marked with 0.9 h one bar
        later.)
        """
        if self._realistic and as_of is not None:
            try:
                expiry = date.fromisoformat(str(pos["expiry_date"])[:10])
            except (KeyError, ValueError, TypeError):
                expiry = None
            if expiry is not None:
                return time_to_expiry_years(as_of, expiry, close=_session_close(expiry))
        return max(max(self._position_dte(pos) - days_held, 0) / 365.0, 0.0001)

    def _reprice_ic_legs(self, pos: dict, underlying: float, days_held: int,
                         hist: pd.DataFrame | None = None,
                         as_of: datetime | None = None) -> tuple[float, float, float, float]:
        """The four condor leg mids (short call, short put, long call, long put) at this
        underlying/age. Single source for `_reprice_position` (net value) and the W66
        per-leg exit friction, so the two can never disagree on the leg prices."""
        iv = self._get_current_iv(pos, hist)
        t = self._remark_t(pos, days_held, as_of)
        r = self._risk_free_rate     # W81: options.risk_free_rate (was a bare 0.05)
        if self._surface is not None:
            try:
                _exp = date.fromisoformat(str(pos["expiry_date"])[:10])
                _on = (as_of.date() if as_of is not None else
                       date.fromisoformat(str(pos["entry_date"])[:10]) + timedelta(days=days_held))
            except (KeyError, ValueError, TypeError):
                _exp = None
            if _exp is not None:
                _ivs = self._surface_leg_ivs(
                    _exp, (pos["short_call_strike"], pos["short_put_strike"],
                           pos["long_call_strike"], pos["long_put_strike"]),
                    underlying, _on, iv)
                return ic_leg_marks_from_ivs(
                    underlying, pos["short_call_strike"], pos["short_put_strike"],
                    pos["long_call_strike"], pos["long_put_strike"], t, r, *_ivs)
        return ic_leg_marks(
            underlying, pos["short_call_strike"], pos["short_put_strike"],
            pos["long_call_strike"], pos["long_put_strike"], t, r, iv,
            self._skew_factor, self._skew_slope_per_pct_otm, self._iv_floor)

    def _reprice_position(self, pos: dict, underlying: float, days_held: int,
                          hist: pd.DataFrame | None = None,
                          as_of: datetime | None = None) -> float:
        """Reprice a position at the current underlying using Black-Scholes.

        Returns the current value per share:
        - For debit positions: current option/spread value (want it to go UP)
        - For credit positions: current cost to buy back (want it to go DOWN)
        """
        iv = self._get_current_iv(pos, hist)
        # Decay to the REAL expiration (entry_dte), not the hold-cap —
        # max_hold_days now closes the position early via its own exit
        # check (_check_exit_*) rather than by fabricating a shorter option.
        t = self._remark_t(pos, days_held, as_of)
        r = self._risk_free_rate     # W81: options.risk_free_rate (was a bare 0.05)

        strategy = pos["strategy"]

        if strategy in ("iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor"):
            sc, sp, lc, lp = self._reprice_ic_legs(pos, underlying, days_held, hist, as_of)
            return (sc + sp) - (lc + lp)

        elif strategy == "put_credit_spread":
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return sp - lp

        elif strategy == "call_credit_spread":
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            return sc - lc

        elif strategy == "jade_lizard":
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            return sp + sc - lc

        elif strategy in ("bull_call_spread", "bear_put_spread"):
            long_strike = pos["long_strike"]
            short_strike = pos["short_strike"]
            if pos["option_type"] == "call":
                long_val = black_scholes_price(underlying, long_strike, t, r,
                    self._get_leg_iv(iv, long_strike, underlying, OptionType.CALL), OptionType.CALL)
                short_val = black_scholes_price(underlying, short_strike, t, r,
                    self._get_leg_iv(iv, short_strike, underlying, OptionType.CALL), OptionType.CALL)
            else:
                long_val = black_scholes_price(underlying, long_strike, t, r,
                    self._get_leg_iv(iv, long_strike, underlying, OptionType.PUT), OptionType.PUT)
                short_val = black_scholes_price(underlying, short_strike, t, r,
                    self._get_leg_iv(iv, short_strike, underlying, OptionType.PUT), OptionType.PUT)
            return long_val - short_val

        elif strategy == "short_strangle":
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return sc + sp  # Cost to buy back (want DOWN for profit)

        elif strategy == "long_strangle":
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return lc + lp  # Current value (want UP for profit)

        else:
            # Single-leg option
            opt_type = OptionType.CALL if pos["option_type"] == "call" else OptionType.PUT
            return black_scholes_price(underlying, pos["strike"], t, r,
                self._get_leg_iv(iv, pos["strike"], underlying, opt_type), opt_type)

    def _check_thesis_invalidation(
        self, pos: dict, hist: pd.DataFrame
    ) -> dict | None:
        """Return exit dict if the ML direction strongly contradicts the entry thesis.

        A NEUTRAL (iron condor / short strangle) entry is invalidated when the
        current ML prediction is STRONGLY directional (confidence ≥ 0.80).
        A directional entry (BULLISH/BEARISH) is invalidated when the prediction
        flips to the opposite direction with confidence ≥ 0.80.

        Returns None if no invalidation, or {"exit_date", "exit_reason"} dict.
        """
        if self._predictor is None:
            return None
        # W74 / W72 lead L5: live's _check_thesis_valid EXEMPTS defined-risk NEUTRAL
        # structures entirely (R16, 2026-08-07: a condor profits in either
        # direction, so a direction "flip" is meaningless). The legacy engine
        # exited condors on a >= 0.80 directional call.
        if self._realistic and pos.get("strategy") in ("iron_condor", "iron_butterfly"):
            return None
        # Only re-evaluate after at least 2 days held (avoid same-bar noise)
        entry_date = date.fromisoformat(pos["entry_date"])
        if not hasattr(hist.index[-1], "date"):
            return None
        current_date = hist.index[-1].date() if hasattr(hist.index[-1], "date") else hist.index[-1]
        if (current_date - entry_date).days < 2:
            return None

        try:
            pred = self._predictor.predict(hist)
        except Exception:
            return None
        if pred is None or pred.confidence < 0.80:
            return None

        entry_dir = pos.get("direction", SignalDirection.NEUTRAL.value)
        pred_dir = pred.direction.value if hasattr(pred.direction, "value") else str(pred.direction)

        invalidated = False
        if entry_dir == SignalDirection.NEUTRAL.value:
            # Iron condor / short strangle: strong directional signal invalidates the range thesis
            if pred_dir in (SignalDirection.BULLISH.value, SignalDirection.BEARISH.value):
                invalidated = True
        elif entry_dir == SignalDirection.BULLISH.value and pred_dir == SignalDirection.BEARISH.value:
            invalidated = True
        elif entry_dir == SignalDirection.BEARISH.value and pred_dir == SignalDirection.BULLISH.value:
            invalidated = True

        if invalidated:
            return {
                "exit_date": str(current_date),
                "exit_time": str(current_date),
                "exit_reason": f"thesis_invalidated:{pred_dir}@{pred.confidence:.2f}",
            }
        return None

    def _check_exit(self, pos: dict, row: pd.Series, current_date: date,
                    hist: pd.DataFrame | None = None) -> dict | None:
        """Check if a position should be exited."""
        if pos.get("settlement") is not None:
            return None   # W77: waiting for the next open; run() closes it, never twice
        underlying = row["Close"]
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (current_date - entry_date).days

        _eod_asof = self._eod(current_date) if self._realistic else None
        current_value = self._reprice_position(pos, underlying, days_held, hist, _eod_asof)

        trade_type = pos.get("trade_type", "debit")
        entry_price = pos["entry_price"]

        if trade_type == "credit":
            # Credit position: we received entry_price, now it costs current_value to close
            # Profit when current_value < entry_price (cheaper to buy back)
            # pnl_pct: positive = profitable (value decayed)
            if entry_price > 0:
                pnl_pct = (entry_price - current_value) / entry_price
            else:
                pnl_pct = 0.0
        else:
            # Debit position: we paid entry_price, now it's worth current_value
            if entry_price > 0:
                pnl_pct = (current_value - entry_price) / entry_price
            else:
                pnl_pct = 0.0

        # Apply per-leg bid-ask spread at exit (scales with current IV and remaining DTE)
        exit_iv = float(pos.get("entry_iv", 0.25))
        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = max(0, (expiry - current_date).days)
        exit_half_spread = self._options_half_spread(exit_iv, remaining_dte)
        # Buy back at the ask / sell at the bid (W66: per-leg dollars for v2 condors)
        current_value = self._apply_exit_friction(
            pos, current_value, underlying, days_held, hist, exit_half_spread, trade_type,
            _eod_asof)

        result = self._dispatch_exit_check(pos, trade_type, pnl_pct, current_date, row)

        if result is not None and result.pop("_settle", False):
            # W77: the expiry bell. Settled (closed now) or left holding shares (None).
            return self._settle_at_bell(pos, float(underlying), current_date)

        if result is not None:
            # R16 touch-stop: live transacts AT the touched strike intraday, not
            # at the close. Reprice the structure at that level so a pierce that
            # recovered by the bell is booked at what live would actually have
            # paid — the whole point of modelling the touch.
            if result.get("exit_reason") == "touch_stop":
                current_value, underlying = self._touch_exit_mark(
                    pos, result, days_held, hist, trade_type, _eod_asof)
                current_value = self._apply_exit_friction(
                    pos, current_value, underlying, days_held, hist, exit_half_spread,
                    trade_type, _eod_asof)

            # Calculate actual P&L
            pnl = self._calc_pnl(pos, current_value)
            result["pnl"] = round(pnl, 2)
            result["exit_price"] = round(current_value, 4)
            result["exit_underlying"] = round(float(underlying), 4)
            if "exit_time" not in result:
                result["exit_time"] = result.get("exit_date")  # EOD exit — no intraday timestamp
            return result

        return None

    def _touch_exit_mark(self, pos: dict, result: dict, days_held: int,
                         hist: pd.DataFrame | None, trade_type: str,
                         as_of: datetime | None = None) -> tuple[float, float]:
        """Mid value and underlying a short-strike TOUCH exit is booked at.

        R16: live transacts AT the touched strike, so the structure is repriced
        there. W5 model-vs-reality-touch-gap-pricing: a session that OPENED
        beyond the short strike could never be closed AT the strike, so the
        gapped open is repriced too and the WORSE of the two is booked —
        conservative by construction. Pops the two bookkeeping keys off
        ``result``. W74: extracted from ``_check_exit`` so the intraday scan
        prices a touch with the very same code.
        """
        _tu = float(result.pop("touch_underlying"))
        current_value = self._reprice_position(pos, _tu, days_held, hist, as_of)
        underlying = _tu
        _gap = result.pop("touch_gap_underlying", None)
        if _gap is not None:
            _gv = self._reprice_position(pos, float(_gap), days_held, hist, as_of)
            _worse = (_gv > current_value if trade_type == "credit" else _gv < current_value)
            if _worse:
                current_value = _gv
                underlying = float(_gap)
        return current_value, underlying

    def _dispatch_exit_check(
        self, pos: dict, trade_type: str, pnl_pct: float, current_date: date,
        row: pd.Series | None = None, as_of: datetime | None = None
    ) -> dict | None:
        """Route exit logic by trade type (R6 exit parity with live).

        CREDIT structures mirror live portfolio.py check_position(): ONE flat
        loss limit at credit_loss_limit_mult x credit received plus the
        DTE-laddered take-profit — NO trailing stop, NO breakeven tier, NO
        vol-adjusted stops. DEBIT structures keep the existing behaviour
        (trailing/breakeven when enabled, else fixed stop + profit target).
        """
        if trade_type == "credit":
            return self._check_exit_credit(pos, pnl_pct, current_date, row, as_of)
        if self._trailing_stop_enabled:
            return self._check_exit_trailing(pos, pnl_pct, current_date)
        return self._check_exit_fixed(pos, pnl_pct, current_date)

    @staticmethod
    def _credit_take_profit_pct(
        dte: int | None,
        time_decay_scaling: bool = True,
        ladder: "exit_policy.Ladder | None" = None,
    ) -> float:
        """DTE-laddered take-profit target as a fraction of credit received.

        R20 #5a: the ladder (0.50/0.40/0.30/0.20 + breakpoints) used to be a
        hand copy of portfolio.py _get_take_profit_targets — now imported from
        the shared ait.execution.exit_policy authority. Stays a staticmethod
        with defaulted args because run_backtest.py calls it UNBOUND for its
        drift-check table; the exit path passes self._exit_time_decay_scaling
        and self._credit_tp_ladder (None unless a research study set one).
        """
        return exit_policy.credit_take_profit_pct(
            dte, time_decay_scaling=time_decay_scaling, ladder=ladder
        )

    def _entry_ts(self, pos: dict) -> datetime:
        """W74: when this position was opened, tz-aware ET.

        The intraday fill time when the entry was simulated on 5-minute bars, else
        the daily-model convention: priced at the day's close.
        """
        for key in ("fill_time", "entry_time"):
            v = pos.get(key)
            if v:
                try:
                    ts = datetime.fromisoformat(str(v))
                except ValueError:
                    continue
                return ts.astimezone(ET) if ts.tzinfo else ts.replace(tzinfo=ET)
        d = date.fromisoformat(str(pos["entry_date"])[:10])
        return datetime.combine(d, _session_close(d), tzinfo=ET)

    def _exit_instant(self, pos: dict) -> datetime:
        """W78: when a closed position's exit took effect, tz-aware ET.

        An intraday exit carries its bar timestamp in ``exit_time``; a daily-path exit is
        decided at the session close (the daily-model convention), so a date-only value
        maps to that session's close."""
        v = pos.get("exit_time") or pos.get("exit_date")
        d = date.fromisoformat(str(pos.get("exit_date") or v)[:10])
        if v and len(str(v)) > 10:
            try:
                ts = datetime.fromisoformat(str(v))
                return ts.astimezone(ET) if ts.tzinfo else ts.replace(tzinfo=ET)
            except ValueError:
                pass
        return datetime.combine(d, _session_close(d), tzinfo=ET)

    def _check_exit_credit(self, pos: dict, pnl_pct: float, current_date: date,
                           row: pd.Series | None = None,
                           as_of: datetime | None = None) -> dict | None:
        """Credit-structure exits — R6 parity with live portfolio.py.

        pnl_pct is fraction of credit received (positive = value decayed).
        Order mirrors live check_position(): (0) SHORT-STRIKE TOUCH,
        (1) flat loss limit, (2) DTE-laddered take-profit, (3)
        expiry-approaching close at DTE<=5, (4) macro-event flatten.
        """
        # 0. R16: SHORT-STRIKE TOUCH — live's PRIMARY loss exit, and until now
        # the engine's single largest structural divergence. Live watches spot
        # every 30s and closes the moment price reaches a short strike; the
        # engine only ever saw the daily CLOSE, so an intraday pierce that
        # recovered by the bell was scored as an untouched winner. Every study
        # to date (wing_k, shadow R1-R3, ablations) therefore measured a
        # DIFFERENT exit policy than the one running live — biased toward
        # optimism precisely on the days that hurt.
        # Daily High/Low bracket the true intraday path, so they detect the
        # touch without intraday bars. The exit is priced AT the touched
        # strike (the level live would have transacted at), not at the close.
        # W5 model-vs-reality-touch-gap-pricing: the strike is the right exit
        # level only for an INTRADAY pierce. When the session OPENS already
        # beyond a short strike there was never a chance to transact at the
        # strike — options do not trade overnight and a resting stop would
        # trigger at the open too — so booking the strike prices the standard
        # condor loss path as a near-scratch. Measured: 1%+ overnight gaps on
        # 9.6% of SPY / 18.1% QQQ / 17.8% IWM days in the very 604-day sample
        # every study runs on, understating the loss by $324-$1,405/contract
        # at 1%-3.5% gaps. The gapped open is reported alongside the strike
        # and _check_exit prices the exit at the WORSE of the two.
        if row is not None and self._touch_stop_enabled:
            _sp = pos.get("short_put_strike")
            _sc = pos.get("short_call_strike")
            try:
                _low = float(row["Low"]) if "Low" in row else None
                _high = float(row["High"]) if "High" in row else None
            except (TypeError, ValueError):
                _low = _high = None
            try:
                _open = float(row["Open"]) if "Open" in row else None
            except (TypeError, ValueError):
                _open = None
            _touched = None
            _gapped = None
            if _sp and _low is not None and _low <= float(_sp):
                _touched = float(_sp)
                if _open is not None and _open <= float(_sp):
                    _gapped = _open
            elif _sc and _high is not None and _high >= float(_sc):
                _touched = float(_sc)
                if _open is not None and _open >= float(_sc):
                    _gapped = _open
            if _touched is not None:
                _out = {"exit_date": str(current_date),
                        "exit_reason": "touch_stop",
                        "touch_underlying": _touched}
                if _gapped is not None:
                    _out["touch_gap_underlying"] = _gapped
                return _out
        # HWM still tracked for journaling/analysis parity (live persists it
        # even though credit exits no longer trail off it).
        pos["high_water_mark"] = max(pos.get("high_water_mark", 0.0), pnl_pct)

        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = (expiry - current_date).days

        # 1. Flat loss limit: one stop at -mult x credit (env
        # AIT_CREDIT_LOSS_LIMIT). R16 #6 parity with live portfolio.py:413-414:
        # default 0 = flat stop DISABLED (mult <= 0 mirrors live's -999
        # sentinel); the wings cap the true tail, and R12-B1 showed touch-close
        # beats every flat level. Env override still re-arms the stop.
        if self._credit_loss_limit_mult > 0 and pnl_pct <= -self._credit_loss_limit_mult:
            return {"exit_date": str(current_date), "exit_reason": "credit_loss_limit"}

        # 2. DTE-laddered take-profit (shared exit_policy ladder — R20 #5a
        # also honors exit.time_decay_scaling exactly as live does; the flag
        # off collapses to live's flat 0.50 short target).
        if pnl_pct >= self._credit_take_profit_pct(
                remaining_dte, self._exit_time_decay_scaling, self._credit_tp_ladder):
            return {"exit_date": str(current_date), "exit_reason": "take_profit_short"}

        # 2b. Max-hold-days cap — RESEARCH-ONLY knob (Optuna-searched): live
        # has no "close after N days held" rule, only the DTE/touch/TP exits
        # in this function. 2026-08-24 fix: before this, max_hold_days WAS
        # entry_dte (the same variable), so this trigger never existed — the
        # position simply couldn't outlive its own fabricated expiration.
        # Now entry_dte is fixed near live's real chain selection and
        # max_hold_days independently forces an earlier close, letting
        # Optuna explore shorter/longer synthetic hold horizons.
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        # W74: decision instant. Intraday callers pass the bar's end; the daily path
        # decides at the close (the daily-model convention).
        _asof = as_of if as_of is not None else datetime.combine(
            current_date, _session_close(current_date), tzinfo=ET)
        _bell = datetime.combine(expiry, _session_close(expiry), tzinfo=ET)

        # 2a. W74 expiry-day flatten: a hard backstop so no position reaches
        # settlement/assignment mechanics. Off (None) = legacy.
        if self._flatten_minutes is not None and exit_policy.expiry_day_flatten_due(
                _asof, expiry, self._flatten_minutes, close_et=_session_close(expiry)):
            return {"exit_date": str(current_date), "exit_reason": "expiry_day_flatten"}

        # 2b. Hold cap. W74: a fraction of the position's own life REPLACES the
        # day count when set, so "hold anywhere between 0 and DTE" is one
        # dimension for every entry DTE.
        if self._max_hold_fraction is not None:
            if exit_policy.hold_limit_reached(self._entry_ts(pos), _asof, _bell,
                                              self._max_hold_fraction):
                return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}
        elif held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        # 3. Expiry-approaching close — live closes any position at
        # DTE <= exit_policy.EXPIRY_APPROACHING_DTE (portfolio.py rule 3a;
        # rule 3 assignment risk at DTE<=1 is subsumed). W74: only for positions
        # ENTERED above expiry_guard_dte (None = every position, as before).
        if remaining_dte <= exit_policy.EXPIRY_APPROACHING_DTE and \
                exit_policy.expiry_guard_applies(self._position_dte(pos), self._expiry_guard_dte):
            reason = "expiry" if current_date >= expiry else "expiry_approaching"
            return {"exit_date": str(current_date), "exit_reason": reason}

        # 3b. W74: nothing outlives its expiry SESSION. Time-aware on purpose: on
        # the expiry date itself the position is alive until the bell, so the
        # intraday scan must reach the flatten cutoff first; the daily path
        # decides at the close, where this fires. Unreachable in legacy (rule 3
        # fires first for every position); it closes guard-exempt entries.
        if _asof >= _bell:
            out = {"exit_date": str(current_date), "exit_reason": "expiry"}
            # W77: a realistic engine settles at the bell (exercise / assignment /
            # expiry, shares carried over a gap) instead of re-marking and paying exit
            # friction on legs that do not trade. Only structures whose legs are known
            # (_settle_legs_of); anything else keeps the re-mark.
            if self._realistic and self._settle_legs_of(pos) is not None:
                out["_settle"] = True
            return out

        # 4. Macro-event flatten (portfolio.py rule 3d parity). PLAN
        # 2026-08-04: defined-risk EXEMPT — condors hold through events
        # (wings cap the surprise; the vol crush is the payoff). Only
        # undefined/assignment-risk strategies keep the early exit.
        # R20 #5a: the strategy list AND per-strategy windows (strangle-class
        # 5 days, CSP/CC 1) come from the shared exit_policy table — the R16
        # jade_lizard omission was exactly the hand-copy drift this kills.
        # 2026+2027-H1 calendar: inactive for pre-2026 windows.
        _flat_window = exit_policy.macro_flatten_window_days(pos.get("strategy"))
        if (self._economic_cal is not None
                and contract_flag("AIT_SKIP_MACRO_EVENTS")
                and _flat_window is not None):
            try:
                _d2e = self._economic_cal.days_until_next_event(current_date)
            except Exception:  # noqa: BLE001
                _d2e = None
            if _d2e is not None and _d2e <= _flat_window:
                return {"exit_date": str(current_date), "exit_reason": "macro_event_flatten"}

        return None

    def _check_exit_fixed(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Fixed stop-loss / take-profit (DEBIT trades only since R6 parity)."""
        stop = self._stop_loss_pct

        # Stop loss
        if pnl_pct <= -stop:
            return {"exit_date": str(current_date), "exit_reason": "stop_loss"}

        # Profit target
        target = self._profit_target_pct
        if pnl_pct >= target:
            return {"exit_date": str(current_date), "exit_reason": "profit_target"}

        # Max-hold-days cap (research-only knob; see _check_exit_credit)
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        if held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        # Expiry
        expiry = date.fromisoformat(pos["expiry_date"])
        if current_date >= expiry:
            return {"exit_date": str(current_date), "exit_reason": "expiry"}

        return None

    def _check_exit_trailing(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Dynamic trailing stop."""
        pos["high_water_mark"] = max(pos.get("high_water_mark", 0.0), pnl_pct)
        hwm = pos["high_water_mark"]

        if hwm < self._breakeven_trigger_pct:
            effective_stop = -self._stop_loss_pct
            stop_label = "stop_loss"
        else:
            effective_stop = max(0.0, hwm - self._trailing_stop_pct)
            stop_label = "breakeven_stop" if effective_stop == 0.0 else "trailing_stop"

        if pnl_pct <= effective_stop:
            return {"exit_date": str(current_date), "exit_reason": stop_label}

        # Max-hold-days cap (research-only knob; see _check_exit_credit)
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        if held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        expiry = date.fromisoformat(pos["expiry_date"])
        if current_date >= expiry:
            return {"exit_date": str(current_date), "exit_reason": "expiry"}

        return None

    def _calc_pnl(self, pos: dict, current_value: float) -> float:
        """Calculate P&L including commissions.

        For debit positions: PnL = (current_value - entry_price) * 100 * contracts
        For credit positions: PnL = (entry_price - current_value) * 100 * contracts
        """
        contracts = pos["contracts"]
        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", "debit")
        n_legs = pos.get("n_legs", 1)

        if trade_type == "credit":
            raw_pnl = (entry_price - current_value) * 100 * contracts
        else:
            raw_pnl = (current_value - entry_price) * 100 * contracts

        # W5 model-vs-reality-commission-constant: all-in per-leg cost
        # (commission + regulatory/exchange/clearing fees), same figure the
        # entry side debits — a 4-leg condor round trip therefore costs
        # _cost_per_contract_leg x 4 x 2 per contract.
        exit_commission = self._cost_per_contract_leg * contracts * n_legs
        total_commission = pos.get("entry_commission", 0) + exit_commission

        return raw_pnl - total_commission

    # W77: option legs by strategy, as (position key, right, side). A strategy absent here
    # (or a position missing a key) keeps the re-mark close, so settlement can never
    # silently drop a leg.
    _SETTLE_LEG_KEYS: dict[str, tuple[tuple[str, str, str], ...]] = {
        **{s: (("short_call_strike", "C", "short"), ("long_call_strike", "C", "long"),
               ("short_put_strike", "P", "short"), ("long_put_strike", "P", "long"))
           for s in ("iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor")},
        "put_credit_spread": (("short_put_strike", "P", "short"), ("long_put_strike", "P", "long")),
        "call_credit_spread": (("short_call_strike", "C", "short"), ("long_call_strike", "C", "long")),
        "short_strangle": (("short_call_strike", "C", "short"), ("short_put_strike", "P", "short")),
        "jade_lizard": (("short_put_strike", "P", "short"), ("short_call_strike", "C", "short"),
                        ("long_call_strike", "C", "long")),
    }

    def _settle_legs_of(self, pos: dict) -> "list[settlement.SettleLeg] | None":
        """W77: the position's option legs for settlement, or None when unknown."""
        keys = self._SETTLE_LEG_KEYS.get(pos.get("strategy"))
        if keys is None:
            return None
        out = []
        for key, right, side in keys:
            k = pos.get(key)
            if k is None:
                return None
            out.append(settlement.SettleLeg(float(k), right, side))
        return out

    def _bump_settle(self, key: str) -> None:
        if self._settlement_stats is None:
            self._settlement_stats = {"settled": 0, "pending": 0, "assigned_gap": 0,
                                      "fallback": 0, "over_max_loss": 0}
        self._settlement_stats[key] = self._settlement_stats.get(key, 0) + 1

    def _settle_at_bell(self, pos: dict, S: float, current_date: date,
                        exit_time: str | None = None) -> dict | None:
        """W77: settle a credit structure at the expiry bell at underlying ``S``.

        Nets to zero shares in every scenario -> closed now as ``expiry_settled`` at
        intrinsic (no option exit friction, no option exit commission: expired, exercised
        and assigned legs do not trade). Otherwise the position stays open holding the
        delivered shares (``pos["settlement"]``) and ``run()`` closes it at the next
        session's open; returns None in that case.
        """
        legs = self._settle_legs_of(pos)
        if legs is None:   # defensive: the caller only sets _settle when legs are known
            return None
        if settlement.needs_overnight(legs, S, self._pin_band):
            pos["settlement"] = {
                "underlying": round(float(S), 4), "expiry": str(current_date),
                "legs": [[l.strike, l.right, l.side] for l in legs]}
            self._bump_settle("pending")
            return None
        res = settlement.worst_case(legs, S, self._pin_band, S,
                                    self._assignment_fee, self._stock_commission)
        self._bump_settle("settled")
        return self._finish_settlement(pos, res, S, str(current_date),
                                       exit_time or str(current_date), "expiry_settled")

    def _close_pending_settlement(self, pos: dict, price: float, on_date: date,
                                  reason: str = "assigned_gap",
                                  exit_time: str | None = None) -> dict:
        """W77: close a position that carried assigned/exercised shares past the bell,
        marking them at ``price`` (the next session's open; the last close at the end of
        a backtest). Pins are re-resolved at the original settlement price."""
        sett = pos.pop("settlement")
        legs = [settlement.SettleLeg(float(k), r, s) for k, r, s in sett["legs"]]
        res = settlement.worst_case(legs, float(sett["underlying"]), self._pin_band,
                                    float(price), self._assignment_fee, self._stock_commission)
        if reason == "assigned_gap":
            self._bump_settle("assigned_gap")
        out = self._finish_settlement(
            pos, res, float(price), str(on_date),
            exit_time or datetime.combine(on_date, time(9, 30), tzinfo=ET).isoformat(), reason)
        out["settlement_underlying"] = float(sett["underlying"])
        return out

    def _finish_settlement(self, pos: dict, res: "settlement.Resolution", underlying: float,
                           exit_date: str, exit_time: str, reason: str) -> dict:
        """Price a settled position: credit - cost, less the stock fees and the ENTRY
        commission. No option exit commission: nothing was traded."""
        contracts = pos["contracts"]
        gross = (pos["entry_price"] - res.cost) * 100 * contracts
        fees = res.fees * 100 * contracts
        pnl = gross - fees - pos.get("entry_commission", 0)
        max_loss = pos.get("max_loss")
        if max_loss is not None and gross < -float(max_loss) - SETTLEMENT_LOSS_TOLERANCE:
            self._bump_settle("over_max_loss")   # the loss exceeded the DEFINED maximum
        return {
            "exit_date": exit_date, "exit_time": exit_time, "exit_reason": reason,
            "exit_price": round(res.cost, 4), "exit_underlying": round(float(underlying), 4),
            "pnl": round(pnl, 2), "settlement_legs": list(res.statuses),
            "net_shares": int(res.net_shares), "settlement_fees_usd": round(fees, 2),
        }

    def _force_close(self, pos: dict, last_row: pd.Series, last_date: date,
                     hist: pd.DataFrame | None = None) -> dict:
        """Force-close a position at end of backtest."""
        if pos.get("settlement") is not None:   # W77: shares still held; mark at the close
            return self._close_pending_settlement(
                pos, float(last_row["Close"]), last_date, reason="backtest_end",
                exit_time=str(last_date))
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (last_date - entry_date).days
        _eod_asof = self._eod(last_date) if self._realistic else None
        current_value = self._reprice_position(pos, last_row["Close"], days_held, hist, _eod_asof)

        trade_type = pos.get("trade_type", "debit")
        exit_iv = float(pos.get("entry_iv", 0.25))
        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = max(0, (expiry - last_date).days)
        exit_half_spread = self._options_half_spread(exit_iv, remaining_dte)
        current_value = self._apply_exit_friction(
            pos, current_value, last_row["Close"], days_held, hist, exit_half_spread,
            trade_type, _eod_asof)

        pnl = self._calc_pnl(pos, current_value)
        return {
            "exit_date": str(last_date),
            "exit_time": str(last_date),
            "exit_price": round(current_value, 4),
            "pnl": round(pnl, 2),
            "exit_reason": "backtest_end",
        }
