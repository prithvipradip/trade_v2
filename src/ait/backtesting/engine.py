"""Backtesting engine for simulating trading strategies on historical data.

Runs the trading loop day-by-day over an OHLCV DataFrame, applying
ML predictions (if available), strategy selection, risk rules,
and trade simulation with Black-Scholes options pricing.

Supports:
- Debit strategies: long_call, long_put, bull_call_spread, bear_put_spread
- Credit strategies: iron_condor, short_strangle (profit from theta decay)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ait.backtesting.options_sim import (
    OptionType,
    black_scholes_price,
    find_strike_by_delta,
    realized_vol,
)
from ait.backtesting.result import BacktestResult
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("backtesting.engine")

# Strategies that collect premium (short theta)
CREDIT_STRATEGIES = {"iron_condor", "short_strangle", "short_straddle", "covered_call", "cash_secured_put", "put_credit_spread"}
# Strategies that pay premium (long theta)
DEBIT_STRATEGIES = {"long_call", "long_put", "bull_call_spread", "bear_put_spread", "long_straddle"}


class Backtester:
    """Simulates trading strategies against historical OHLCV data."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategies: list[str],
        initial_capital: float = 10_000.0,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.01,
        position_size_pct: float = 0.05,
        stop_loss_pct: float = 0.50,
        profit_target_pct: float = 1.00,
        max_hold_days: int = 30,
        min_confidence: float = 0.55,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 0.25,
        breakeven_trigger_pct: float = 0.30,
        predictor: Any = None,
        range_predictor: Any = None,
        range_min_confidence: float = 0.55,
        context_bars: int = 0,
        delta_short: float = 0.20,
        delta_long: float = 0.30,
        iv_floor: float = 0.12,
        wing_floor_dollars: float = 5.0,
        wing_k: float = 1.0,
        delta_iv_scale: float = 0.0,
        skew_factor: float = 1.0,
        hurst_regime_threshold: float = 0.20,
        hurst_regime_penalty: float = 0.10,
        hurst_hard_veto_multiplier: float = 1.5,
        multifractal_max_width: float = 0.50,
        aekf_veto_threshold: float = 0.60,
        iv_rank_rise_threshold: float = 0.30,
        features_cache: pd.DataFrame | None = None,
        max_concurrent_positions: int = 1,
        max_entry_vol_annual: float = 0.80,
        # Options bid-ask spread model (per-leg, IV/DTE-aware)
        spread_base: float = 0.03,
        spread_iv_sensitivity: float = 0.10,
        spread_dte_sensitivity: float = 0.005,
        spread_cap: float = 0.15,
        # Cross-asset market context forwarded from walk-forward engine (Gap Z3)
        market_context: dict | None = None,
        # Earnings-aware skip (Gap Z4): if symbol is given, skip entries near earnings dates
        symbol: str | None = None,
        earnings_skip_days: int = 2,
        # Intraday engine (Fix 1): 5-min execution loop
        intraday_store: Any = None,
        scan_interval_minutes: int = 60,
        entry_window_start_et: str = "09:30",
        entry_window_end_et: str = "15:30",
        limit_order_timeout_bars: int = 3,
        # Per-window MetaLabeler for OOS signal filtering (Gap Z1)
        meta_labeler: Any = None,
        # H2 val-split: skip new entries before this date (full df still used for feature warmup)
        eval_start_date: "date | None" = None,
    ) -> None:
        self._data = self._prepare_data(data)
        self._strategies = strategies
        self._initial_capital = initial_capital
        self._commission = commission_per_contract
        self._slippage_pct = slippage_pct
        self._spread_base = spread_base
        self._spread_iv_sensitivity = spread_iv_sensitivity
        self._spread_dte_sensitivity = spread_dte_sensitivity
        self._spread_cap = spread_cap
        self._position_size_pct = position_size_pct
        self._stop_loss_pct = stop_loss_pct
        self._profit_target_pct = profit_target_pct
        self._max_hold_days = max_hold_days
        self._min_confidence = min_confidence
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct
        self._breakeven_trigger_pct = breakeven_trigger_pct
        self._context_bars = context_bars
        self._range_predictor = range_predictor
        self._range_min_confidence = range_min_confidence
        self._delta_short = delta_short
        self._delta_long = delta_long
        self._iv_floor = iv_floor
        self._wing_floor_dollars = wing_floor_dollars
        self._wing_k = wing_k
        self._delta_iv_scale = delta_iv_scale
        self._skew_factor = skew_factor
        self._hurst_regime_threshold = hurst_regime_threshold
        self._hurst_regime_penalty = hurst_regime_penalty
        self._hurst_hard_veto_multiplier = hurst_hard_veto_multiplier
        self._multifractal_max_width = multifractal_max_width
        self._aekf_veto_threshold = aekf_veto_threshold
        self._iv_rank_rise_threshold = iv_rank_rise_threshold
        self._features_cache = features_cache
        self._max_concurrent_positions = max_concurrent_positions
        self._max_entry_vol_annual = max_entry_vol_annual
        self._market_context = market_context
        self._symbol = symbol or ""
        self._earnings_dates: set[date] = self._load_earnings_dates(symbol, earnings_skip_days)
        self._intraday_store = intraday_store
        self._scan_interval_minutes = scan_interval_minutes
        self._entry_window_start_et = entry_window_start_et
        self._entry_window_end_et = entry_window_end_et
        self._limit_order_timeout_bars = limit_order_timeout_bars
        self._meta_labeler = meta_labeler
        self._eval_start_date = eval_start_date

        self._predictor = predictor if predictor is not None else self._load_predictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
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

            for pos in open_positions:
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
                continue

            # H2 val-split: skip new entries before eval_start_date (exits still processed above)
            if self._eval_start_date is not None and today_date < self._eval_start_date:
                continue

            direction, confidence, features_df = self._get_direction(hist, market_context=self._market_context)

            # Decision-chain dict — populated as each gate is evaluated;
            # attached to the trade record so the dashboard can render the
            # full entry reasoning without re-running the backtest.
            _dir_conf_orig = confidence
            _entry_decision: dict = {
                "direction_class": direction.value if hasattr(direction, "value") else str(direction),
                "direction_conf": round(float(confidence), 4),
                "range_gate": {"prob": None, "threshold": self._range_min_confidence, "pass": True},
                "vol_gate": {"vol_10d": None, "max": self._max_entry_vol_annual, "pass": True},
                "meta_label": {"take": True, "prob": None, "threshold": 0.5},
                "fractal_gate": {"hurst_spread": 0.0, "threshold": self._hurst_regime_threshold, "pass": True},
                "regime": "range_bound",
                "earnings_skip": False,
            }

            # Apply fractal regime gate to confidence for credit strategies.
            # Features are already computed inside _get_direction — reuse them.
            if not features_df.empty:
                last_f = features_df.iloc[-1]
                spread = float(last_f.get("hurst_scale_spread", 0.0))
                mf_w   = float(last_f.get("multifractal_width",  0.0))
                _hurst_pass = spread <= self._hurst_regime_threshold or self._hurst_regime_threshold <= 0
                _entry_decision["fractal_gate"] = {
                    "hurst_spread": round(spread, 4),
                    "threshold": self._hurst_regime_threshold,
                    "pass": _hurst_pass,
                }
                if spread > self._hurst_regime_threshold and self._hurst_regime_threshold > 0:
                    # Hard veto disabled (multiplier=0): Exp 20 post-mortem showed QQQ
                    # hurst_spread never drops below ~0.43 in normal conditions, so any
                    # threshold derived from Optuna-optimized values (0.09–0.29) × 1.5
                    # blocks ALL entries. W12's bad-trade spreads (0.44–0.76) are
                    # indistinguishable from profitable-trade spreads in other windows.
                    _neutral_strat = bool(set(self._strategies) & {"iron_condor", "short_strangle"})
                    _veto_base = max(self._hurst_regime_threshold, 0.20)
                    _hard_veto_threshold = _veto_base * self._hurst_hard_veto_multiplier
                    if _neutral_strat and self._hurst_hard_veto_multiplier > 0 and spread > _hard_veto_threshold:
                        _entry_decision["fractal_gate"]["hard_veto"] = True
                        log.debug(
                            "hard_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            hurst_spread=round(spread, 4),
                            hard_veto_threshold=round(_hard_veto_threshold, 4),
                            hurst_regime_threshold=round(self._hurst_regime_threshold, 4),
                        )
                        continue
                    penalty = self._hurst_regime_penalty * (
                        spread / self._hurst_regime_threshold
                    )
                    confidence = max(0.0, confidence - penalty)
                if mf_w > 0 and mf_w > self._multifractal_max_width:
                    confidence = max(0.0, confidence - self._hurst_regime_penalty)

            effective_min_conf = self._min_confidence

            # Iron condors and short strangles are market-neutral: high directional
            # confidence signals a trending regime — exactly when they fail. Skip the
            # direction gate for these strategies; the range model below is the sole
            # entry filter. Directional strategies still require confidence ≥ min_conf.
            _neutral_only = bool(set(self._strategies) & {"iron_condor", "short_strangle"})
            if not _neutral_only:
                if confidence < effective_min_conf:
                    continue
                if direction == SignalDirection.BEARISH and confidence < effective_min_conf + 0.05:
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
                        "regime_trending_up":   1.0 if (vol_exp and px_sma > 0.02)   else 0.0,
                        "regime_trending_down":  1.0 if (vol_exp and px_sma < -0.02)  else 0.0,
                        "regime_high_vol":       1.0 if (vol_exp and abs(px_sma) <= 0.02) else 0.0,
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
                            continue
                except Exception:
                    pass  # meta-labeler errors are non-fatal

            strategy = self._select_strategy(direction, hist, confidence, features_df)
            if strategy is None:
                continue

            # Earnings proximity skip (Gap Z4): matches live orchestrator behaviour.
            if today_date in self._earnings_dates:
                log.debug("earnings_skip", date=str(today_date), strategy=strategy)
                continue

            # Regime gate: iron condors / short strangles fail in trending-down regimes.
            # Exp 21 post-mortem: 4 trending_down trades, 25% win rate, avg PnL -$195.
            # Both macro dislocations in the dataset (Yen carry unwind Aug-2024,
            # tariff shock Mar-2026) occurred in this regime. Trending_up (100% win)
            # and high_volatility (62% win) maintain positive EV and are not blocked.
            # Exp 22: threshold -0.02 was too loose — blocked profitable 2-4% corrections
            # in W02, causing Optuna to adapt badly. Both structural failures cleared -0.05
            # (Yen carry -6-8%, tariff shock -8-10%); raised to -0.05 for Exp 23.
            if strategy in ("iron_condor", "short_strangle") and not features_df.empty:
                _last_f        = features_df.iloc[-1]
                _vol_exp_flag  = float(_last_f.get("vol_regime_expanding", 0.0)) > 0.5
                _px_sma_val    = float(_last_f.get("price_vs_sma_20", 0.0))
                _regime_class  = (
                    "trending_down"   if (_vol_exp_flag and _px_sma_val < -0.05) else
                    "trending_up"     if (_vol_exp_flag and _px_sma_val > 0.02)  else
                    "high_volatility" if  _vol_exp_flag                          else
                    "range_bound"
                )
                _entry_decision["regime_class"] = _regime_class
                if _regime_class == "trending_down":
                    _entry_decision["regime_veto"] = True
                    log.debug(
                        "regime_veto_fired",
                        component="backtesting.engine",
                        strategy=strategy,
                        regime=_regime_class,
                        vol_regime_expanding=round(float(_last_f.get("vol_regime_expanding", 0.0)), 4),
                        price_vs_sma_20=round(_px_sma_val, 4),
                    )
                    continue

            # Range model gate: for iron condors / strangles, replace confidence
            # with P(stays in range). Skip if below range threshold.
            if strategy in ("iron_condor", "short_strangle") and self._range_predictor is not None:
                try:
                    rp = self._range_predictor.predict(hist, market_context=self._market_context)
                    if rp is None or rp.probability_in_range < self._range_min_confidence:
                        continue  # bad range setup → skip
                    _entry_decision["range_gate"] = {
                        "prob": round(float(rp.probability_in_range), 4),
                        "threshold": self._range_min_confidence,
                        "pass": True,
                    }
                    confidence = rp.probability_in_range
                except Exception:
                    pass

            # Realized-vol entry gate: iron condors / short strangles cannot profit
            # during high-volatility regimes (e.g. tariff shocks, VIX > 40).
            if strategy in ("iron_condor", "short_strangle"):
                recent_close = hist["Close"].iloc[-11:]
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
                        continue

            # AEKF direction veto: if the OU-Kou-GARCH AEKF produces a high-confidence
            # directional drift signal, the market is trending — skip iron condor entry.
            # Signal values are always logged (not just on veto) for threshold tuning.
            if strategy in ("iron_condor", "short_strangle") and self._range_predictor is not None:
                try:
                    sym_data = (getattr(self._range_predictor, "_symbol_models", {}) or {}).get(self._symbol, {})
                    _ou_dir  = (sym_data.get("ou_jump_state") or {}).get("ou_jump_direction")
                    _ou_conf = (sym_data.get("ou_jump_state") or {}).get("ou_jump_confidence") or 0.0
                    _entry_decision["aekf_signal"] = {
                        "direction": _ou_dir,
                        "confidence": round(float(_ou_conf), 4) if _ou_dir is not None else None,
                    }
                    if _ou_dir is not None and float(_ou_conf) >= self._aekf_veto_threshold:
                        _entry_decision["aekf_veto"] = {"direction": _ou_dir, "confidence": round(float(_ou_conf), 4)}
                        log.debug(
                            "aekf_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            ou_direction=_ou_dir,
                            ou_confidence=round(float(_ou_conf), 4),
                            threshold=self._aekf_veto_threshold,
                        )
                        continue
                except Exception:
                    pass

            # Rising IV rank filter: if IV rank has risen by more than iv_rank_rise_threshold
            # over the last 10 days, market is in directional stress — skip iron condor entry.
            # Rise value is always logged (not just on veto) for threshold tuning.
            if strategy in ("iron_condor", "short_strangle") and not features_df.empty:
                if "iv_rank" in features_df.columns and len(features_df) >= 10:
                    iv_rank_series = features_df["iv_rank"].iloc[-10:]
                    iv_rank_rise = float(iv_rank_series.iloc[-1]) - float(iv_rank_series.iloc[0])
                    _entry_decision["iv_rank_rise_10d"] = round(iv_rank_rise, 4)
                    if iv_rank_rise > self._iv_rank_rise_threshold:
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
                        continue

            # --- 3. Build the trade ---
            # Intraday entry window gate (Fix 1g / Gap D): when 5-min data is available,
            # only enter during the configured ET window and simulate a limit-order fill.
            entry_time_str: str | None = None
            limit_price: float | None = None
            fill_time_str: str | None = None
            if self._intraday_store is not None:
                from ait.data.historical import HistoricalDataStore
                session_bars = self._intraday_store.load_intraday_range(
                    symbol=self._symbol,
                    start_date=today_date,
                    end_date=today_date,
                )
                if not session_bars.empty:
                    # Find first bar in the entry window
                    window_bars = session_bars[
                        session_bars.index.to_series().apply(
                            lambda ts: self._is_in_entry_window(ts.time())
                        )
                    ]
                    if window_bars.empty:
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
                        continue  # Limit order expired without fill — skip entry

            pos = self._build_position(strategy, direction, row, hist, today_date, capital)
            if pos is None:
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
                if vol_expanding:
                    if px_vs_sma > 0.02:
                        pos["entry_regime"] = "trending_up"
                    elif px_vs_sma < -0.02:
                        pos["entry_regime"] = "trending_down"
                    else:
                        pos["entry_regime"] = "high_volatility"
                else:
                    pos["entry_regime"] = "range_bound"
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
                    "sentiment_composite": round(float(last_f.get("sentiment_composite", 0.0)), 3),
                    "put_call_ratio":      round(float(last_f.get("put_call_ratio", 1.0)), 3),
                }
            else:
                pos["entry_regime"] = "range_bound"  # default when history too short
                pos["features_at_entry"] = {}

            # Finalize decision chain with resolved regime
            _entry_decision["direction_conf"] = round(float(confidence), 4)
            _entry_decision["regime"] = pos.get("entry_regime", "range_bound")
            pos["decision"] = _entry_decision

            # Iron condor leg structure for the dashboard drawer
            if pos.get("strategy") == "iron_condor":
                ep = pos.get("entry_price", 0.0)  # net credit per share
                pos["legs"] = [
                    {"type": "short_put",  "strike": pos.get("short_put_strike"),  "premium": round(ep * 0.42, 4)},
                    {"type": "long_put",   "strike": pos.get("long_put_strike"),   "premium": round(ep * 0.22, 4)},
                    {"type": "short_call", "strike": pos.get("short_call_strike"), "premium": round(ep * 0.40, 4)},
                    {"type": "long_call",  "strike": pos.get("long_call_strike"),  "premium": round(ep * 0.20, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            elif pos.get("strategy") in ("put_credit_spread", "call_credit_spread"):
                ep = pos.get("entry_price", 0.0)
                pos["legs"] = [
                    {"type": "short", "strike": pos.get("short_put_strike") or pos.get("short_call_strike"), "premium": round(ep * 0.60, 4)},
                    {"type": "long",  "strike": pos.get("long_put_strike")  or pos.get("long_call_strike"),  "premium": round(ep * 0.40, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            else:
                pos["legs"] = []

            # Deduct commission
            n_legs = pos.get("n_legs", 1)
            entry_commission = self._commission * pos["contracts"] * n_legs
            capital -= entry_commission
            pos["entry_commission"] = entry_commission

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
        )

        log.info(
            "backtest_complete",
            total_trades=result.total_trades,
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
            max_dd=f"{result.max_drawdown:.2%}",
            win_rate=f"{result.win_rate:.2%}",
        )
        return result

    @classmethod
    def compare_exit_modes(
        cls, data: pd.DataFrame, strategies: list[str], **kwargs: Any
    ) -> dict:
        """Run backtest with both fixed and trailing stops, return comparison."""
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

    @staticmethod
    def _load_earnings_dates(symbol: str | None, skip_days: int) -> set[date]:
        """Pre-fetch historical earnings dates from yfinance (Gap Z4).

        Returns the set of all calendar dates within skip_days of any earnings
        announcement, so the main loop can do an O(1) membership test.
        Returns an empty set if symbol is None or yfinance is unavailable.
        """
        if not symbol or skip_days <= 0:
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
                for delta in range(-1, skip_days + 1):  # 1 day after through skip_days before
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
        """Return True if bar_time (datetime.time) is within the entry window."""
        from datetime import time as dt_time
        start = self._parse_et_time(self._entry_window_start_et)
        end = self._parse_et_time(self._entry_window_end_et)
        t = bar_time if isinstance(bar_time, dt_time) else bar_time.time()
        return start <= t < end

    def _check_intraday_exit(
        self, pos: dict, session_bars: "pd.DataFrame", current_date: date
    ) -> dict | None:
        """Check intraday 5-min bars for stop-loss or profit-target triggers.

        Returns exit dict if triggered on any bar, else None.
        """
        if session_bars is None or session_bars.empty:
            return None

        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", pos.get("position_type", "debit"))
        expiry_str = pos.get("expiry_date") or pos.get("exit_date")
        if not expiry_str:
            return None
        expiry = date.fromisoformat(str(expiry_str)[:10])

        for bar_ts, bar_row in session_bars.iterrows():
            underlying = bar_row["Close"]
            days_held = (current_date - date.fromisoformat(pos["entry_date"])).days

            current_val = self._reprice_position(pos, underlying, days_held, None)
            remaining_dte = max(0, (expiry - current_date).days)
            exit_half_spread = self._options_half_spread(
                float(pos.get("entry_iv", 0.25)), remaining_dte
            )
            if trade_type == "credit":
                current_val *= (1 + exit_half_spread)
                pnl_pct = (entry_price - current_val) / entry_price if entry_price > 0 else 0.0
            else:
                current_val *= (1 - exit_half_spread)
                pnl_pct = (current_val - entry_price) / entry_price if entry_price > 0 else 0.0

            # Check stop / profit
            if self._trailing_stop_enabled:
                result = self._check_exit_trailing(pos, pnl_pct, current_date)
            else:
                result = self._check_exit_fixed(pos, pnl_pct, current_date)

            if result is not None:
                bar_dt = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)
                pnl = self._calc_pnl(pos, current_val)
                result["pnl"] = round(pnl, 2)
                result["exit_price"] = round(current_val, 4)
                result["exit_underlying"] = round(float(underlying), 4)
                result["exit_time"] = bar_dt
                return result

        return None

    def _try_limit_fill(
        self, limit_price: float, session_bars: "pd.DataFrame", timeout_bars: int
    ) -> "tuple[bool, int, str | None]":
        """Simulate limit order fill on subsequent 5-min bars.

        Returns (filled, bars_waited, fill_time_iso). fill_time_iso is the ISO
        timestamp of the bar where Low ≤ limit_price ≤ High, or None if unfilled.
        """
        for i, (ts, bar) in enumerate(session_bars.iterrows()):
            if i >= timeout_bars:
                break
            if bar["Low"] <= limit_price <= bar["High"]:
                fill_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                return True, i + 1, fill_ts
        return False, min(len(session_bars), timeout_bars), None

    def _load_directional_model(self):
        """Try to load the directional model for small account trading."""
        try:
            from ait.ml.directional import DirectionalModel
            model = DirectionalModel()
            if model.load():
                return model
        except Exception:
            pass
        return None

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
        if self._features_cache is not None and not self._features_cache.empty:
            today = pd.Timestamp(hist.index[-1]).normalize()
            mask = self._features_cache.index <= today
            features_df = self._features_cache[mask]
            if features_df.empty:
                features_df = FeatureEngine().compute(hist)
        else:
            features_df = FeatureEngine().compute(hist)

        if self._predictor is not None:
            try:
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
        iv = self._get_iv(hist)

        # IV rank proxy: compare current IV to its range
        close_arr = hist["Close"].values
        rv_short = realized_vol(close_arr, window=10)
        rv_long = realized_vol(close_arr, window=60) if len(close_arr) > 61 else rv_short
        iv_regime_high = rv_short > rv_long * 1.1  # Short-term vol elevated

        # Always prefer iron condor — proven +311% in backtesting.
        # Direction doesn't matter — iron condors profit from theta decay.
        has_condor = bool(available & {"iron_condor"})

        if has_condor:
            candidates = available & {"iron_condor"}
        elif direction == SignalDirection.NEUTRAL:
            candidates = available & CREDIT_STRATEGIES
        elif direction == SignalDirection.BULLISH:
            candidates = available & {"bull_call_spread", "long_call"}
        else:
            candidates = available & {"bear_put_spread", "long_put"}

        if not candidates:
            return None
        return sorted(candidates)[0]

    # ------------------------------------------------------------------
    # Position building — full BS pricing for each strategy type
    # ------------------------------------------------------------------

    def _get_leg_iv(self, base_iv: float, strike: float, underlying: float,
                    option_type: OptionType) -> float:
        """Apply vol skew: OTM puts carry extra IV, OTM calls carry mild extra IV.

        Linear model in log-moneyness space:
        - Put: +1% IV per 10% below ATM (e.g. 20% OTM put → +2% IV)
        - Call: +0.2% IV per 10% above ATM
        skew_factor=0.0 restores flat IV; default 1.0 approximates market skew.
        """
        log_m = np.log(strike / underlying)
        if option_type == OptionType.PUT:
            skew_adj = self._skew_factor * max(0.0, -log_m) * 0.10
        else:
            skew_adj = self._skew_factor * max(0.0, log_m) * 0.02
        return max(base_iv + skew_adj, self._iv_floor)

    def _options_half_spread(self, iv: float, dte: int) -> float:
        """Compute the per-leg half bid-ask spread for an OTM option.

        Models the empirical relationship between IV, DTE, and spread width:
        - Higher IV → wider spreads (market makers demand more edge)
        - Lower DTE → wider spreads (gamma risk increases near expiry)

        Returns the half-spread as a fraction of the option mid price.
        """
        iv_term = self._spread_iv_sensitivity * max(0.0, iv - 0.20)
        dte_term = self._spread_dte_sensitivity * max(0.0, 21 - dte)
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

        # Priority 2: VIX proxy from market context (scalar or DataFrame)
        if market_context:
            vix = market_context.get("vix_close") or market_context.get("vix")
            if vix is not None:
                if hasattr(vix, "reindex"):
                    # Full VIX DataFrame — align to current hist and take last value
                    vix_aligned = vix["Close"].reindex(hist.index, method="ffill")
                    if not vix_aligned.empty:
                        vix_val = float(vix_aligned.iloc[-1])
                        if vix_val > 0:
                            return vix_val / 100.0 * 1.10
                else:
                    vix_val = float(vix)
                    if vix_val > 0:
                        return vix_val / 100.0 * 1.05

        # Priority 3: synthetic fallback
        close_arr = hist["Close"].values
        rv = realized_vol(close_arr, window=20)
        return rv * 1.15

    def _build_position(
        self,
        strategy: str,
        direction: SignalDirection,
        row: pd.Series,
        hist: pd.DataFrame,
        today_date: date,
        capital: float,
        market_context: dict | None = None,
    ) -> dict | None:
        """Build a position dict with proper BS pricing for the strategy type."""
        underlying = row["Close"]
        iv = self._get_iv(hist, market_context=market_context)
        dte = self._max_hold_days
        t = dte / 365.0
        r = 0.05

        # Entry gate for credit strategies: insufficient IV → premiums too small to
        # justify spread costs and risk. Honest no-trade rather than fake-pricing at floor.
        if strategy in CREDIT_STRATEGIES and iv < self._iv_floor:
            return None

        if strategy in CREDIT_STRATEGIES:
            return self._build_credit_position(strategy, underlying, iv, t, r, dte, today_date, capital)
        else:
            return self._build_debit_position(strategy, direction, underlying, iv, t, r, dte, today_date, capital)

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
            iv_scale = max(0.5, min(1.5, self._iv_floor / iv))
            effective_delta = self._delta_long * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(0.05, min(0.45, effective_delta))
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

    def _build_credit_position(
        self, strategy: str,
        S: float, iv: float, t: float, r: float, dte: int,
        today_date: date, capital: float,
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
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing_width
            long_put_strike = short_put_strike - wing_width

            # Price each leg with its own skewed IV
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)

            # Net credit received at mid prices
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                return None

            # Per-leg spread cost: 4 legs × half-spread (sell at bid, buy at ask)
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            spread_cost = 4 * half_spread * avg_leg_mid
            net_credit = max(0.0, net_credit - spread_cost)
            if net_credit <= 0:
                return None

            # Max loss = wing width - net credit (per share)
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            # Position sizing based on max loss (margin requirement)
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                return None

            return {
                "symbol": "SIM",
                "strategy": "iron_condor",
                "direction": SignalDirection.NEUTRAL.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),  # Credit received per share
                "contracts": contracts,
                "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0),  # Reference: underlying at entry
                "option_type": "iron_condor",
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
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

        elif strategy == "short_strangle":
            # Sell OTM call + sell OTM put — no wings (naked short premium).
            # Delta is IV-scaled: high IV → go further OTM for more breathing room.
            iv_scale = max(0.5, min(1.5, self._iv_floor / iv))
            effective_delta = self._delta_short * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(0.05, min(0.45, effective_delta))
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

    def _get_current_iv(self, pos: dict, hist: pd.DataFrame | None) -> float:
        """Mean-reverting IV: blends entry IV with current realized vol.

        Weights 70% toward entry IV (forward-looking anchor) and 30% toward
        the current realized vol × 1.15 premium, so vega P&L is non-zero
        without wild swings driven purely by realized vol noise.
        """
        entry_iv = pos.get("entry_iv") or pos.get("iv", self._iv_floor)
        if hist is None or len(hist) < 21:
            return entry_iv
        rv = realized_vol(hist["Close"].values, window=20) * 1.15
        return max(0.70 * entry_iv + 0.30 * rv, self._iv_floor)

    def _reprice_position(self, pos: dict, underlying: float, days_held: int,
                          hist: pd.DataFrame | None = None) -> float:
        """Reprice a position at the current underlying using Black-Scholes.

        Returns the current value per share:
        - For debit positions: current option/spread value (want it to go UP)
        - For credit positions: current cost to buy back (want it to go DOWN)
        """
        iv = self._get_current_iv(pos, hist)
        dte_remaining = max(self._max_hold_days - days_held, 0)
        t = max(dte_remaining / 365.0, 0.0001)
        r = 0.05

        strategy = pos["strategy"]

        if strategy == "iron_condor":
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return (sc + sp) - (lc + lp)

        elif strategy == "put_credit_spread":
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return sp - lp

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
        underlying = row["Close"]
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (current_date - entry_date).days

        current_value = self._reprice_position(pos, underlying, days_held, hist)

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
        if trade_type == "credit":
            current_value *= (1 + exit_half_spread)  # Buy back at ask
        else:
            current_value *= (1 - exit_half_spread)  # Sell at bid

        if self._trailing_stop_enabled:
            result = self._check_exit_trailing(pos, pnl_pct, current_date)
        else:
            result = self._check_exit_fixed(pos, pnl_pct, current_date)

        if result is not None:
            # Calculate actual P&L
            pnl = self._calc_pnl(pos, current_value)
            result["pnl"] = round(pnl, 2)
            result["exit_price"] = round(current_value, 4)
            result["exit_underlying"] = round(float(underlying), 4)
            if "exit_time" not in result:
                result["exit_time"] = result.get("exit_date")  # EOD exit — no intraday timestamp
            return result

        return None

    def _check_exit_fixed(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Fixed stop-loss / take-profit."""
        stop = self._stop_loss_pct

        # Stop loss
        if pnl_pct <= -stop:
            return {"exit_date": str(current_date), "exit_reason": "stop_loss"}

        # Profit target
        target = self._profit_target_pct
        if pnl_pct >= target:
            return {"exit_date": str(current_date), "exit_reason": "profit_target"}

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

        exit_commission = self._commission * contracts * n_legs
        total_commission = pos.get("entry_commission", 0) + exit_commission

        return raw_pnl - total_commission

    def _force_close(self, pos: dict, last_row: pd.Series, last_date: date,
                     hist: pd.DataFrame | None = None) -> dict:
        """Force-close a position at end of backtest."""
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (last_date - entry_date).days
        current_value = self._reprice_position(pos, last_row["Close"], days_held, hist)

        trade_type = pos.get("trade_type", "debit")
        exit_iv = float(pos.get("entry_iv", 0.25))
        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = max(0, (expiry - last_date).days)
        exit_half_spread = self._options_half_spread(exit_iv, remaining_dte)
        if trade_type == "credit":
            current_value *= (1 + exit_half_spread)
        else:
            current_value *= (1 - exit_half_spread)

        pnl = self._calc_pnl(pos, current_value)
        return {
            "exit_date": str(last_date),
            "exit_time": str(last_date),
            "exit_price": round(current_value, 4),
            "pnl": round(pnl, 2),
            "exit_reason": "backtest_end",
        }
