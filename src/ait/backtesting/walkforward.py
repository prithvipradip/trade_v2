"""Walk-forward backtester with multi-symbol support.

Trains the ML model on historical data, then tests on unseen future data,
sliding the window forward. This is the gold standard for validating
trading strategies — it prevents overfitting by never testing on training data.

Usage:
    from ait.backtesting.walkforward import WalkForwardBacktester
    bt = WalkForwardBacktester(
        symbols=["SPY", "QQQ", "AAPL"],
        strategies=["long_call", "bull_call_spread", "iron_condor"],
    )
    result = await bt.run()
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ait.backtesting.engine import Backtester
from ait.backtesting.learner import BacktestLearner
from ait.backtesting.result import BacktestResult
from ait.config.settings import MLConfig
from ait.ml.ensemble import DirectionPredictor
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("backtesting.walkforward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""

    train_days: int = 365        # ~1 year training window (calendar days)
    test_days: int = 63          # ~3 months test window
    step_days: int = 21          # ~1 month step between windows
    gap_days: int = 5            # Purge gap between train and test
    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.03  # 3% realistic for multi-leg options
    position_size_pct: float = 0.05
    wing_floor_dollars: float = 5.0
    wing_k: float = 1.0
    iv_floor: float = 0.12
    delta_iv_scale: float = 0.0
    stop_loss_pct: float = 0.35            # Cut losses at 35% (options decay fast)
    profit_target_pct: float = 0.50         # Take profits at 50% (don't be greedy)
    max_hold_days: int = 21                 # 3 weeks max (avoid deep theta decay)
    min_confidence: float = 0.55
    range_min_confidence: float = 0.55     # threshold for range model on iron condors
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.25
    breakeven_trigger_pct: float = 0.30
    max_concurrent_positions: int = 3
    max_entry_vol_annual: float = 0.80
    optimize_per_window: bool = False
    optimize_n_trials: int = 50
    optimize_patience: int = 0       # 0 = disabled; N = stop after N non-improving trials
    optimize_min_trades: int = 10    # Penalise trials with fewer trades than this floor
    optimize_seed: int = 42          # TPESampler seed — fix for reproducibility
    range_threshold_pct: float = 0.05  # Target move % for range model; links to strategy profitability
    hurst_regime_threshold: float = 0.20
    hurst_regime_penalty: float = 0.10
    multifractal_max_width: float = 0.50
    # Intraday execution params (Fix 1 / Gap H)
    scan_interval_minutes: int = 60        # how often to scan for signals during a session
    entry_window_start_et: str = "10:30"   # earliest allowed entry time (ET)
    entry_window_end_et: str = "15:30"     # latest allowed entry time (ET)
    limit_order_timeout_bars: int = 3      # cancel limit order after N 5-min bars without fill
    # Options spread model params (Fix 5 / Gap E)
    spread_base: float = 0.03             # base half-spread per leg ($)
    spread_iv_sensitivity: float = 0.10   # additional spread per unit IV above 0.20
    spread_dte_sensitivity: float = 0.005 # additional spread per DTE below 21
    spread_cap: float = 0.15              # maximum half-spread per leg ($)


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    backtest_result: BacktestResult
    model_accuracy: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward backtest results."""

    windows: list[WindowResult] = field(default_factory=list)
    symbol_results: dict[str, BacktestResult] = field(default_factory=dict)
    strategy_results: dict[str, dict] = field(default_factory=dict)
    initial_capital: float = 10_000.0
    config: WalkForwardConfig | None = None

    @property
    def total_trades(self) -> int:
        return sum(w.backtest_result.total_trades for w in self.windows)

    @property
    def total_return(self) -> float:
        if not self.windows:
            return 0.0
        # Chain returns across windows
        equity = self.initial_capital
        for w in self.windows:
            equity *= (1 + w.backtest_result.total_return)
        return (equity - self.initial_capital) / self.initial_capital

    @property
    def win_rate(self) -> float:
        all_trades = []
        for w in self.windows:
            all_trades.extend(w.backtest_result.trades)
        if not all_trades:
            return 0.0
        return sum(1 for t in all_trades if t.get("pnl", 0) > 0) / len(all_trades)

    @property
    def sharpe_ratio(self) -> float:
        all_pnls = []
        for w in self.windows:
            all_pnls.extend(t.get("pnl", 0) for t in w.backtest_result.trades)
        if len(all_pnls) < 2:
            return 0.0
        mean = np.mean(all_pnls)
        std = np.std(all_pnls, ddof=1)
        if std == 0:
            return 0.0
        return float((mean / std) * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        for w in self.windows:
            for t in w.backtest_result.trades:
                equity += t.get("pnl", 0)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        all_trades = []
        for w in self.windows:
            all_trades.extend(w.backtest_result.trades)
        gross_wins = sum(t["pnl"] for t in all_trades if t.get("pnl", 0) > 0)
        gross_losses = abs(sum(t["pnl"] for t in all_trades if t.get("pnl", 0) <= 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def avg_window_return(self) -> float:
        if not self.windows:
            return 0.0
        returns = [w.backtest_result.total_return for w in self.windows]
        return float(np.mean(returns))

    @property
    def consistency(self) -> float:
        """Fraction of windows that were profitable."""
        if not self.windows:
            return 0.0
        profitable = sum(1 for w in self.windows if w.backtest_result.total_return > 0)
        return profitable / len(self.windows)

    def _all_trades(self) -> list[dict]:
        out = []
        for w in self.windows:
            out.extend(w.backtest_result.trades)
        return out

    @property
    def sortino_ratio(self) -> float:
        """Downside-only volatility ratio — better for option-selling skew."""
        all_pnls = np.array([t.get("pnl", 0) for t in self._all_trades()])
        if len(all_pnls) < 2:
            return 0.0
        mean_pnl = float(all_pnls.mean())
        downside = all_pnls[all_pnls < 0]
        if len(downside) < 2:
            return float("inf") if mean_pnl > 0 else 0.0
        ds_std = float(downside.std(ddof=1))
        if ds_std == 0:
            return 0.0
        return (mean_pnl / ds_std) * float(np.sqrt(252))

    @property
    def avg_win(self) -> float:
        wins = [t["pnl"] for t in self._all_trades() if t.get("pnl", 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(t["pnl"]) for t in self._all_trades() if t.get("pnl", 0) <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def win_loss_ratio(self) -> float:
        if self.avg_loss == 0:
            return float("inf") if self.avg_win > 0 else 0.0
        return self.avg_win / self.avg_loss

    @property
    def expectancy(self) -> float:
        wr = self.win_rate
        return wr * self.avg_win - (1 - wr) * self.avg_loss

    @property
    def best_trade(self) -> float:
        trades = self._all_trades()
        return max((t.get("pnl", 0) for t in trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        trades = self._all_trades()
        return min((t.get("pnl", 0) for t in trades), default=0.0)

    @property
    def capital_utilization(self) -> float:
        """Avg % of initial capital deployed across the backtest period."""
        trades = self._all_trades()
        if not trades or not self.windows:
            return 0.0
        first_date = self.windows[0].test_start
        last_date = self.windows[-1].test_end
        total_days = max(1, (last_date - first_date).days)

        from datetime import datetime, date as _date
        def to_d(d):
            if d is None: return None
            if isinstance(d, _date) and not isinstance(d, datetime): return d
            if hasattr(d, "date"): return d.date()
            if isinstance(d, str):
                try: return datetime.fromisoformat(d.split("T")[0]).date()
                except Exception: return None
            return None

        capital_days = 0.0
        for t in trades:
            entry = to_d(t.get("entry_date"))
            exit_ = to_d(t.get("exit_date"))
            risk = t.get("max_loss") or t.get("cost") or abs(t.get("pnl", 0)) * 2
            if entry and exit_ and risk:
                hold = max(1, (exit_ - entry).days)
                capital_days += risk * hold
        return capital_days / (self.initial_capital * total_days) if self.initial_capital > 0 else 0.0

    @property
    def cash_drag_adjusted_return(self) -> float:
        """Total return + idle-cash T-bill yield adjustment."""
        if not self.windows:
            return self.total_return
        first_date = self.windows[0].test_start
        last_date = self.windows[-1].test_end
        years = max(0.01, (last_date - first_date).days / 365.25)
        idle_pct = max(0, 1 - self.capital_utilization)
        return self.total_return + (0.05 * idle_pct * years)

    @property
    def raroc(self) -> float:
        """Return on capital actually deployed."""
        util = self.capital_utilization
        return self.total_return / util if util > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  WALK-FORWARD BACKTEST RESULTS",
            "=" * 60,
            f"  Windows:           {len(self.windows)}",
            f"  Total Trades:      {self.total_trades}",
            f"  Total Return:      {self.total_return:.2%}",
            f"  Cash-Drag Adj Ret: {self.cash_drag_adjusted_return:.2%}  (idle cash @ 5%)",
            "-" * 60,
            "  RISK-ADJUSTED",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:.2f}  (downside-only vol)",
            f"  Max Drawdown:      {self.max_drawdown:.2%}",
            "-" * 60,
            "  TRADE QUALITY",
            f"  Win Rate:          {self.win_rate:.2%}",
            f"  Avg Win:           ${self.avg_win:,.2f}",
            f"  Avg Loss:          ${self.avg_loss:,.2f}",
            f"  Win/Loss Ratio:    {self.win_loss_ratio:.2f}  (>1 = winners bigger)",
            f"  Expectancy/Trade:  ${self.expectancy:,.2f}",
            f"  Best Trade:        ${self.best_trade:,.2f}",
            f"  Worst Trade:       ${self.worst_trade:,.2f}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            "-" * 60,
            "  CAPITAL EFFICIENCY",
            f"  Utilization:       {self.capital_utilization:.1%}  (avg % deployed)",
            f"  RAROC:             {self.raroc:.1%}  (return on deployed)",
            "-" * 60,
            "  CONSISTENCY",
            f"  Profitable Windows:{self.consistency:.0%}",
            f"  Avg Window Return: {self.avg_window_return:.2%}",
            "-" * 60,
        ]

        if self.strategy_results:
            lines.append("  STRATEGY BREAKDOWN:")
            for strat, data in sorted(
                self.strategy_results.items(),
                key=lambda x: x[1].get("total_pnl", 0),
                reverse=True,
            ):
                lines.append(
                    f"    {strat:25s} | trades={data['trades']:3d} | "
                    f"win={data['win_rate']:.0%} | pnl=${data['total_pnl']:,.0f}"
                )

        if self.symbol_results:
            lines.append("  SYMBOL BREAKDOWN:")
            for sym, result in sorted(
                self.symbol_results.items(),
                key=lambda x: x[1].total_return,
                reverse=True,
            ):
                lines.append(
                    f"    {sym:8s} | return={result.total_return:.2%} | "
                    f"trades={result.total_trades:3d} | sharpe={result.sharpe_ratio:.2f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def equity_curve(self) -> pd.DataFrame:
        """Generate equity curve DataFrame from all trades across windows."""
        rows = []
        equity = self.initial_capital

        for w in self.windows:
            for t in w.backtest_result.trades:
                equity += t.get("pnl", 0)
                rows.append({
                    "date": t.get("exit_date", ""),
                    "equity": equity,
                    "pnl": t.get("pnl", 0),
                    "strategy": t.get("strategy", ""),
                    "symbol": t.get("symbol", ""),
                    "window": w.window_id,
                })

        if not rows:
            return pd.DataFrame(columns=["date", "equity", "pnl", "strategy", "symbol", "window"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)


class WalkForwardBacktester:
    """Walk-forward backtester with multi-symbol, multi-strategy support.

    Slides a train/test window across history:
    1. Train ML model on [train_start, train_end]
    2. Skip gap_days (purge)
    3. Backtest on [test_start, test_end]
    4. Slide forward by step_days and repeat

    Supports:
    - Multiple symbols (aggregated or per-symbol results)
    - Multiple strategies (with per-strategy breakdown)
    - Buy-and-hold benchmark comparison
    - Equity curve generation
    """

    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        config: WalkForwardConfig | None = None,
        db_path: "Path | None" = None,
        progress_dir: "Path | None" = None,
    ) -> None:
        self._symbols = symbols
        self._strategies = strategies
        self._config = config or WalkForwardConfig()
        self._db_path = db_path
        self._progress_dir = Path(progress_dir) if progress_dir else None
        self._global_best_params: dict | None = None
        self._global_best_score: float = -1.0

    async def run(self, data: dict[str, pd.DataFrame] | None = None) -> WalkForwardResult:
        """Run walk-forward backtest.

        Args:
            data: Pre-loaded data as {symbol: OHLCV DataFrame}.
                  If None, fetches from Yahoo Finance.
        """
        if data is None:
            data = await self._fetch_data()

        if not data:
            log.error("no_data_for_backtest")
            return WalkForwardResult(initial_capital=self._config.initial_capital)

        # Load VIX data for the full backtest period (yfinance fallback when not in DB).
        # Used as Priority-2 IV proxy and as a live feature for the MetaLabeler.
        from ait.data.market_data import load_daily_ohlcv as _load_ohlcv
        try:
            _vix_full: pd.DataFrame = _load_ohlcv(
                "^VIX", days=self._config.train_days + 900, db_path=self._db_path
            )
            # yfinance may return tz-aware index; normalize to tz-naive to match OHLCV data
            if not _vix_full.empty and _vix_full.index.tz is not None:
                _vix_full.index = _vix_full.index.tz_localize(None)
        except Exception:
            _vix_full = pd.DataFrame()

        # Generate walk-forward windows
        windows = self._generate_windows(data)
        log.info("walk_forward_windows", count=len(windows))

        # Self-learning adapter — adapts between windows
        learner = BacktestLearner(base_confidence=self._config.min_confidence)

        # Run each window
        window_results = []
        prev_best_params: dict | None = None   # best Optuna params from prior window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            log.info(
                "running_window",
                window=i + 1,
                train=f"{train_start} to {train_end}",
                test=f"{test_start} to {test_end}",
            )

            # Get adapted config from learner (starts at defaults, improves each window)
            learned_config = learner.get_config()
            effective_min_conf = learned_config["min_confidence"]
            active_strategies = [
                s for s in self._strategies
                if learner.is_strategy_enabled(s)
            ]
            if not active_strategies:
                active_strategies = self._strategies  # Safety: never disable everything

            # Slice data for this window
            all_symbol_trades = []
            model_accuracy = 0.0
            active_symbols = 0
            curr_best_params: dict | None = None
            prev_oos = window_results[-1].backtest_result if window_results else None
            # Use full capital per symbol for position sizing — splitting by symbol count
            # makes iron condors impossible on stocks priced >$50 (max_loss_per_contract too large)
            per_symbol_capital = self._config.initial_capital

            for symbol, df in data.items():
                # Skip symbols the learner has disabled
                if not learner.is_symbol_allowed(symbol):
                    log.info("learner_skipping_symbol", symbol=symbol, window=i + 1)
                    continue

                # Use string slicing to handle both tz-aware and tz-naive indexes
                train_df = df[str(train_start):str(train_end)]
                test_df = df[str(test_start):str(test_end)]

                if len(train_df) < 50 or len(test_df) < 5:
                    continue

                active_symbols += 1

                # Per-window parameter optimization (optional)
                window_cfg = self._config
                if self._config.optimize_per_window:
                    new_cfg, best_params = self._optimize_window_params(
                        train_df, symbol, i + 1, active_strategies,
                        prior_oos_result=prev_oos,
                        prior_best_params=prev_best_params,
                    )
                    window_cfg = new_cfg or self._config
                    if best_params is not None:
                        curr_best_params = best_params

                # Train ML model on training window for this symbol
                # intraday_store is opened inside _train_window_model using self._db_path
                predictor = self._train_window_model(train_df, symbol, i + 1)
                range_predictor = self._train_window_range_model(
                    train_df, symbol, i + 1,
                    max_hold_days=window_cfg.max_hold_days,
                    threshold_pct=self._config.range_threshold_pct,
                )
                if predictor and predictor.is_trained:
                    model_accuracy = max(
                        model_accuracy,
                        max(predictor.cv_scores.values()) if predictor.cv_scores else 0.0,
                    )

                # Train MetaLabeler on training-window trade outcomes (Gap Z1 long-term)
                _meta_artifact = (
                    self._progress_dir / f"window_{i + 1:03d}" / symbol
                    if self._progress_dir else None
                )
                # VIX slice for the training window (shadow MetaLabeler Backtester)
                _vix_train_ctx: dict | None = None
                if not _vix_full.empty:
                    _vix_t = _vix_full.reindex(train_df.index, method="ffill").dropna(how="all")
                    if not _vix_t.empty:
                        _vix_train_ctx = {"vix": _vix_t}

                meta_labeler = self._train_window_meta_labeler(
                    train_df=train_df,
                    symbol=symbol,
                    window_id=i + 1,
                    predictor=predictor,
                    window_cfg=window_cfg,
                    artifact_dir=_meta_artifact,
                    vix_ctx=_vix_train_ctx,
                )

                # Prepend training data context so ML features can be computed.
                # vol_60 is the longest non-min_periods rolling window (60 bars);
                # with 60 context rows + 1 OOS row = 61-row hist, exactly 1 valid
                # feature row survives dropna → predictions work from the first OOS bar.
                context_bars = 60
                test_with_context = pd.concat([train_df.tail(context_bars), test_df])

                # Get position size scaling from learner (per-strategy multipliers)
                # Use the average multiplier across active strategies as an overall scaling
                strategy_mults = [
                    learner.get_strategy_multiplier(s) for s in active_strategies
                ]
                avg_mult = sum(strategy_mults) / len(strategy_mults) if strategy_mults else 1.0
                effective_position_size = self._config.position_size_pct * min(avg_mult, 2.0)

                # When per-window optimization is active the optimized window_cfg
                # is the source of truth for min_confidence; otherwise defer to
                # the self-learning subsystem.
                confidence_threshold = (
                    window_cfg.min_confidence
                    if self._config.optimize_per_window
                    else effective_min_conf
                )

                # Align VIX to this window's date range for IV estimation
                _vix_ctx: dict | None = None
                if not _vix_full.empty:
                    _vix_w = _vix_full.reindex(test_with_context.index, method="ffill").dropna(how="all")
                    if not _vix_w.empty:
                        _vix_ctx = {"vix": _vix_w}

                # Each symbol gets its proportional share of capital
                bt = Backtester(
                    data=test_with_context,
                    context_bars=context_bars,
                    strategies=active_strategies,
                    initial_capital=per_symbol_capital,
                    commission_per_contract=window_cfg.commission_per_contract,
                    slippage_pct=window_cfg.slippage_pct,
                    position_size_pct=effective_position_size,
                    stop_loss_pct=window_cfg.stop_loss_pct,
                    profit_target_pct=window_cfg.profit_target_pct,
                    max_hold_days=window_cfg.max_hold_days,
                    min_confidence=confidence_threshold,
                    trailing_stop_enabled=window_cfg.trailing_stop_enabled,
                    trailing_stop_pct=window_cfg.trailing_stop_pct,
                    breakeven_trigger_pct=window_cfg.breakeven_trigger_pct,
                    predictor=predictor,
                    range_predictor=range_predictor,
                    range_min_confidence=getattr(
                        window_cfg, "range_min_confidence", 0.55
                    ),
                    hurst_regime_threshold=getattr(
                        window_cfg, "hurst_regime_threshold", 0.20
                    ),
                    hurst_regime_penalty=getattr(
                        window_cfg, "hurst_regime_penalty", 0.10
                    ),
                    multifractal_max_width=getattr(
                        window_cfg, "multifractal_max_width", 0.50
                    ),
                    iv_floor=window_cfg.iv_floor,
                    wing_floor_dollars=window_cfg.wing_floor_dollars,
                    wing_k=window_cfg.wing_k,
                    delta_iv_scale=window_cfg.delta_iv_scale,
                    max_concurrent_positions=window_cfg.max_concurrent_positions,
                    max_entry_vol_annual=window_cfg.max_entry_vol_annual,
                    spread_base=window_cfg.spread_base,
                    spread_iv_sensitivity=window_cfg.spread_iv_sensitivity,
                    spread_dte_sensitivity=window_cfg.spread_dte_sensitivity,
                    spread_cap=window_cfg.spread_cap,
                    meta_labeler=meta_labeler,
                    market_context=_vix_ctx,
                )
                result = bt.run()

                # Tag trades with symbol
                for t in result.trades:
                    t["symbol"] = symbol
                all_symbol_trades.extend(result.trades)

            # Aggregate window result
            if all_symbol_trades:
                total_pnl = sum(t.get("pnl", 0) for t in all_symbol_trades)
                window_capital = self._config.initial_capital
                window_result = WindowResult(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    backtest_result=BacktestResult(
                        trades=all_symbol_trades,
                        initial_capital=window_capital,
                        final_capital=window_capital + total_pnl,
                        start_date=test_start,
                        end_date=test_end,
                    ),
                    model_accuracy=model_accuracy,
                )
                window_results.append(window_result)
                self._write_window_progress(i + 1, window_result, curr_best_params)

                # Track globally best Optuna params by OOS quality score
                if curr_best_params is not None:
                    br = window_result.backtest_result
                    score = br.win_rate * (min(1.0, br.total_trades / 5) ** 0.5)
                    if score > self._global_best_score:
                        self._global_best_score = score
                        self._global_best_params = curr_best_params

                # Feed this window's trades into the learner for next-window adaptation
                learning_summary = learner.process_window(all_symbol_trades, i + 1)
                if learning_summary.get("adaptations"):
                    log.info(
                        "self_learning_adapted",
                        window=i + 1,
                        adaptations=[a["change"] for a in learning_summary["adaptations"]],
                    )
            else:
                self._write_window_progress(i + 1, None, curr_best_params,
                                            train_start=train_start, train_end=train_end,
                                            test_start=test_start, test_end=test_end)

            # Carry forward best params for next window's warm-start decision.
            # Only update when OOS produced trades — keeps prev_best_params aligned
            # with prev_oos (which only contains windows that had trades). Without
            # this guard, a 0-trade window's Optuna params would overwrite the last
            # profitable params, causing warm-start to seed from degenerate values.
            if curr_best_params is not None and all_symbol_trades:
                prev_best_params = curr_best_params

        # Build aggregated results
        result = WalkForwardResult(
            windows=window_results,
            initial_capital=self._config.initial_capital,
            config=self._config,
        )

        # Compute per-symbol results
        result.symbol_results = self._compute_symbol_results(window_results, data)

        # Compute per-strategy breakdown
        result.strategy_results = self._compute_strategy_results(window_results)

        log.info(
            "walk_forward_complete",
            windows=len(window_results),
            total_trades=result.total_trades,
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
        )

        # Log final learner state
        log.info("self_learning_final_state", summary=learner.summary())

        return result

    def _write_window_progress(
        self,
        window_id: int,
        window_result: "WindowResult | None",
        best_params: "dict | None" = None,
        **date_kwargs,
    ) -> None:
        if self._progress_dir is None:
            return
        self._progress_dir.mkdir(parents=True, exist_ok=True)

        if window_result is not None:
            br = window_result.backtest_result
            strategy_counts: dict[str, int] = {}
            for t in br.trades:
                s = t.get("strategy", "unknown")
                strategy_counts[s] = strategy_counts.get(s, 0) + 1

            # Per-trade detail: intraday-aware fields (Gap I)
            trades_detail = [
                {
                    "symbol":       t.get("symbol", ""),
                    "strategy":     t.get("strategy", ""),
                    "entry_date":   str(t.get("entry_date", "")),
                    "entry_time":   str(t.get("entry_time", t.get("entry_date", ""))),
                    "exit_date":    str(t.get("exit_date", "")),
                    "exit_time":    str(t.get("exit_time",  t.get("exit_date", ""))),
                    "exit_reason":  t.get("exit_reason", ""),
                    "pnl":          round(t.get("pnl", 0), 2),
                    "entry_confidence": t.get("entry_confidence"),
                    "entry_regime": t.get("entry_regime", ""),
                }
                for t in br.trades
            ]

            payload = {
                "window": window_id,
                "test_start": str(window_result.test_start),
                "test_end": str(window_result.test_end),
                "pnl": round(br.final_capital - br.initial_capital, 2),
                "return_pct": round(br.total_return * 100, 4),
                "trades": br.total_trades,
                "win_rate": round(br.win_rate, 4),
                "sharpe": round(br.sharpe_ratio, 4),
                "max_drawdown": round(br.max_drawdown * 100, 4),
                "strategies": strategy_counts,
                "best_params": best_params or {},
                "trades_detail": trades_detail,
            }
        else:
            payload = {
                "window": window_id,
                "test_start": str(date_kwargs.get("test_start", "")),
                "test_end": str(date_kwargs.get("test_end", "")),
                "pnl": 0.0,
                "return_pct": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "strategies": {},
                "best_params": best_params or {},
                "note": "no_trades",
            }

        path = self._progress_dir / f"window_{window_id:03d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("window_progress_written", window=window_id, path=str(path),
                 pnl=payload["pnl"], trades=payload["trades"])

    def _optimize_window_params(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        strategies: list[str],
        prior_oos_result: "BacktestResult | None" = None,
        prior_best_params: "dict | None" = None,
    ) -> "tuple[WalkForwardConfig | None, dict | None]":
        """Run Optuna on the training slice and return (updated_config, best_params).

        Runs one Optuna study per strategy (per-strategy optimization) so each
        strategy's ~12D space gets full coverage at the trial budget rather than
        a shared 48D joint space. Warm-starts each strategy's study from its own
        subset of prior params.
        Returns (None, None) on any failure (caller falls back to original config).
        """
        try:
            from ait.ml.features import FeatureEngine
            from ait.optimization.optimizer import StrategyOptimizer

            try:
                # Pass intraday_store+symbol so VLMC features appear in features_cache,
                # matching the training path used by _train_window_model (Gap B fix).
                _intraday_store = None
                if self._db_path is not None:
                    from ait.data.historical import HistoricalDataStore
                    _intraday_store = HistoricalDataStore(db_path=self._db_path)
                features_cache = FeatureEngine().compute(
                    train_df, intraday_store=_intraday_store, symbol=symbol
                )
            except Exception:
                features_cache = None

            # Per-strategy optimization: one Optuna study per strategy.
            # Each study seeds from its own subset of prior/global-best params.
            all_best_params: dict = {}
            best_values: dict[str, float] = {}

            for strategy in strategies:
                warm: dict | None = None
                source: str = "cold_start"

                if prior_best_params and prior_oos_result is not None:
                    if (prior_oos_result.win_rate >= 0.75
                            and prior_oos_result.total_trades >= 3):
                        subset = {k: v for k, v in prior_best_params.items()
                                  if k.startswith(f"{strategy}__")}
                        warm = subset or None
                        source = "prior_window"
                    elif self._global_best_params is not None:
                        subset = {k: v for k, v in self._global_best_params.items()
                                  if k.startswith(f"{strategy}__")}
                        warm = subset or None
                        source = "global_best"
                elif self._global_best_params is not None:
                    subset = {k: v for k, v in self._global_best_params.items()
                              if k.startswith(f"{strategy}__")}
                    warm = subset or None
                    source = "global_best_no_prior"

                log.info(
                    "per_strategy_optimization_starting",
                    window=window_id, strategy=strategy, warm_start=source,
                )
                opt = StrategyOptimizer(
                    symbols=[symbol],
                    strategies=[strategy],
                    n_trials=self._config.optimize_n_trials,
                    n_jobs=1,
                    objective="composite",
                    study_name=f"wf_w{window_id}_{symbol}_{strategy}",
                    initial_capital=self._config.initial_capital,
                    features_cache=features_cache,
                    position_size_pct=self._config.position_size_pct,
                    wing_floor_dollars=self._config.wing_floor_dollars,
                    wing_k=self._config.wing_k,
                    iv_floor=self._config.iv_floor,
                    delta_iv_scale=self._config.delta_iv_scale,
                    patience=self._config.optimize_patience,
                    min_trades=self._config.optimize_min_trades,
                    max_concurrent_positions=self._config.max_concurrent_positions,
                    max_entry_vol_annual=self._config.max_entry_vol_annual,
                    seed=self._config.optimize_seed,
                    intraday_store=_intraday_store,
                    symbol=symbol,
                )
                res = opt.run(data={symbol: train_df}, prior_params=warm)
                all_best_params.update(res.best_params)
                best_values[strategy] = res.best_value

            # Build an updated config from merged best params, falling back to originals
            import dataclasses
            cfg_dict = dataclasses.asdict(self._config)
            for key, val in all_best_params.items():
                _, _, param_name = key.partition("__")
                if param_name in cfg_dict:
                    cfg_dict[param_name] = val
            new_cfg = WalkForwardConfig(**cfg_dict)
            log.info(
                "window_params_optimized",
                window=window_id,
                symbol=symbol,
                best_values=best_values,
            )
            return new_cfg, all_best_params
        except Exception as e:
            log.warning("window_optimization_failed", window=window_id, symbol=symbol, error=str(e))
            return None, None

    def _train_window_range_model(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        max_hold_days: int = 30,
        threshold_pct: float = 0.05,
    ):
        """Train range predictor on this window's training data."""
        try:
            from ait.ml.range_predictor import RangePredictor
            intraday_store = None
            if self._db_path is not None:
                from ait.data.historical import HistoricalDataStore
                intraday_store = HistoricalDataStore(db_path=self._db_path)
            rp = RangePredictor(threshold_pct=threshold_pct, horizon_days=max_hold_days)
            accs = rp.train(train_df, symbol=symbol, intraday_store=intraday_store)
            if accs and rp.is_trained:
                avg = sum(accs.values()) / len(accs)
                log.info("window_range_model_trained",
                         window=window_id, symbol=symbol,
                         accuracy=f"{avg:.3f}")
                return rp
        except Exception as e:
            log.debug("range_model_train_failed", window=window_id,
                      symbol=symbol, error=str(e))
        return None

    def _train_window_model(
        self, train_df: pd.DataFrame, symbol: str, window_id: int
    ) -> "DirectionPredictor | None":
        """Train ML model on a training window's data.

        Returns a trained DirectionPredictor, or None if training fails.
        Each window gets a fresh model to prevent data leakage.
        """
        try:
            intraday_store = None
            if self._db_path is not None:
                from ait.data.historical import HistoricalDataStore
                intraday_store = HistoricalDataStore(db_path=self._db_path)

            ml_config = MLConfig()
            predictor = DirectionPredictor(ml_config)
            accuracies = predictor.train(
                train_df, symbol=symbol, intraday_store=intraday_store
            )

            if accuracies:
                avg_acc = sum(accuracies.values()) / len(accuracies)
                log.info(
                    "window_model_trained",
                    window=window_id,
                    symbol=symbol,
                    accuracy=f"{avg_acc:.3f}",
                    models=list(accuracies.keys()),
                )
                return predictor

        except Exception as e:
            log.debug("window_model_training_failed", symbol=symbol, window=window_id, error=str(e))

        return None

    def _train_window_meta_labeler(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        predictor: "DirectionPredictor | None",
        window_cfg: "WalkForwardConfig",
        artifact_dir: "Path | None" = None,
        vix_ctx: "dict | None" = None,
    ) -> "MetaLabeler | None":
        """Train MetaLabeler on trade outcomes from the training window (Gap Z1 long-term).

        Runs a shadow Backtester on the TRAINING data using the just-trained direction
        predictor, collects closed trades with their entry context, looks up the
        corresponding FeatureEngine rows, and trains the MetaLabeler on those outcomes.
        Saves 'meta_labeler.pkl' alongside model.pkl in the window artifact directory.

        Returns a trained MetaLabeler, or None if insufficient data / training failed.
        """
        from ait.ml.features import FeatureEngine
        from ait.ml.meta_label import MetaLabeler

        try:
            # Build the feature matrix for the training window (same call as training path)
            intraday_store = None
            if self._db_path is not None:
                from ait.data.historical import HistoricalDataStore
                intraday_store = HistoricalDataStore(db_path=self._db_path)

            features_df = FeatureEngine().compute(
                train_df, intraday_store=intraday_store, symbol=symbol
            )

            # Shadow Backtester: run on training data to get labelled trade outcomes.
            # Use a reduced context prefix (same 60-bar convention as the OOS run).
            context_bars = 60
            if len(train_df) <= context_bars:
                return None

            train_head = train_df.head(len(train_df) - context_bars)  # actual training slice
            train_with_ctx = train_df  # full training window already has its own context

            shadow_bt = Backtester(
                data=train_with_ctx,
                context_bars=0,
                strategies=["iron_condor"],  # meta-labeler targets the primary strategy
                initial_capital=window_cfg.initial_capital,
                commission_per_contract=window_cfg.commission_per_contract,
                slippage_pct=window_cfg.slippage_pct,
                position_size_pct=window_cfg.position_size_pct,
                stop_loss_pct=window_cfg.stop_loss_pct,
                profit_target_pct=window_cfg.profit_target_pct,
                max_hold_days=window_cfg.max_hold_days,
                min_confidence=window_cfg.min_confidence,
                trailing_stop_enabled=window_cfg.trailing_stop_enabled,
                trailing_stop_pct=window_cfg.trailing_stop_pct,
                breakeven_trigger_pct=window_cfg.breakeven_trigger_pct,
                predictor=predictor,
                iv_floor=window_cfg.iv_floor,
                wing_floor_dollars=window_cfg.wing_floor_dollars,
                wing_k=window_cfg.wing_k,
                max_concurrent_positions=window_cfg.max_concurrent_positions,
                max_entry_vol_annual=window_cfg.max_entry_vol_annual,
                spread_base=window_cfg.spread_base,
                spread_iv_sensitivity=window_cfg.spread_iv_sensitivity,
                spread_dte_sensitivity=window_cfg.spread_dte_sensitivity,
                spread_cap=window_cfg.spread_cap,
                market_context=vix_ctx,
            )
            shadow_result = shadow_bt.run()

            if not shadow_result.trades:
                log.debug(
                    "meta_labeler_no_shadow_trades",
                    window=window_id,
                    symbol=symbol,
                )
                return None

            meta_labeler = MetaLabeler()
            training_df = meta_labeler.build_training_data_from_backtest(
                trades=shadow_result.trades,
                features_df=features_df,
            )

            if training_df.empty:
                return None

            stats = meta_labeler.train(training_df)
            if not stats:
                return None

            log.info(
                "meta_labeler_window_trained",
                window=window_id,
                symbol=symbol,
                trades=len(shadow_result.trades),
                accuracy=f"{stats.get('accuracy', 0):.3f}",
                precision=f"{stats.get('precision', 0):.3f}",
            )

            if artifact_dir is not None:
                from pathlib import Path as _Path
                meta_labeler.save_to_path(_Path(artifact_dir) / "meta_labeler.pkl")

            return meta_labeler

        except Exception as e:
            log.debug(
                "meta_labeler_training_failed",
                window=window_id,
                symbol=symbol,
                error=str(e),
            )
            return None

    def benchmark_buy_hold(self, data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Compute buy-and-hold return for each symbol as a benchmark."""
        results = {}
        for symbol, df in data.items():
            if len(df) < 2:
                continue
            start_price = df["Close"].iloc[0]
            end_price = df["Close"].iloc[-1]
            results[symbol] = (end_price - start_price) / start_price
        results["portfolio"] = float(np.mean(list(results.values()))) if results else 0.0
        return results

    def _generate_windows(self, data: dict[str, pd.DataFrame]) -> list[tuple[date, date, date, date]]:
        """Generate walk-forward train/test date windows."""
        # Find common date range across all symbols
        min_date = max(df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
                       for df in data.values())
        max_date = min(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
                       for df in data.values())

        cfg = self._config
        windows = []
        current = min_date

        while True:
            train_start = current
            train_end = train_start + timedelta(days=cfg.train_days)
            test_start = train_end + timedelta(days=cfg.gap_days)
            test_end = test_start + timedelta(days=cfg.test_days)

            if test_end > max_date:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=cfg.step_days)

        return windows

    async def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Load daily OHLCV from IB store (fallback: Yahoo Finance)."""
        from ait.data.market_data import load_daily_ohlcv

        data = {}
        fetch_days = self._config.train_days + self._config.test_days + 100

        for symbol in self._symbols:
            try:
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda s=symbol: load_daily_ohlcv(
                        s, days=fetch_days, db_path=self._db_path
                    ),
                )
                if df is not None and len(df) > 100:
                    data[symbol] = df
                    log.info("data_fetched", symbol=symbol, rows=len(df))
            except Exception as e:
                log.warning("data_fetch_failed", symbol=symbol, error=str(e))

        return data

    def _compute_symbol_results(
        self, windows: list[WindowResult], data: dict[str, pd.DataFrame]
    ) -> dict[str, BacktestResult]:
        """Aggregate results per symbol across all windows."""
        symbol_trades: dict[str, list] = {}

        for w in windows:
            for t in w.backtest_result.trades:
                sym = t.get("symbol", "unknown")
                symbol_trades.setdefault(sym, []).append(t)

        results = {}
        per_symbol_capital = self._config.initial_capital / max(len(data), 1)
        for sym, trades in symbol_trades.items():
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            results[sym] = BacktestResult(
                trades=trades,
                initial_capital=per_symbol_capital,
                final_capital=per_symbol_capital + total_pnl,
            )

        return results

    @staticmethod
    def _compute_strategy_results(windows: list[WindowResult]) -> dict[str, dict]:
        """Aggregate results per strategy across all windows."""
        strat_trades: dict[str, list] = {}

        for w in windows:
            for t in w.backtest_result.trades:
                strat = t.get("strategy", "unknown")
                strat_trades.setdefault(strat, []).append(t)

        results = {}
        for strat, trades in strat_trades.items():
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            results[strat] = {
                "trades": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "win_rate": wins / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
            }

        return results
