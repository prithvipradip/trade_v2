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
        context_bars: int = 0,
    ) -> None:
        self._data = self._prepare_data(data)
        self._strategies = strategies
        self._initial_capital = initial_capital
        self._commission = commission_per_contract
        self._slippage_pct = slippage_pct
        self._position_size_pct = position_size_pct
        self._stop_loss_pct = stop_loss_pct
        self._profit_target_pct = profit_target_pct
        self._max_hold_days = max_hold_days
        self._min_confidence = min_confidence
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct
        self._breakeven_trigger_pct = breakeven_trigger_pct
        self._context_bars = context_bars

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
            for pos in open_positions:
                exit_info = self._check_exit(pos, row, today_date)
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

            # --- 2. Generate new signal (one trade per day max) ---
            if open_positions:
                continue

            direction, confidence = self._get_direction(hist)

            # Tier-aware confidence: directional spreads need extreme conviction
            if capital < 2000:
                effective_min_conf = max(self._min_confidence, 0.85)
            elif capital < 5000:
                effective_min_conf = max(self._min_confidence, 0.65)
            else:
                effective_min_conf = self._min_confidence

            if confidence < effective_min_conf:
                continue

            # Bearish bets need slightly higher confidence (market has natural upward drift)
            if direction == SignalDirection.BEARISH and confidence < effective_min_conf + 0.05:
                continue

            strategy = self._select_strategy(direction, hist, confidence)
            if strategy is None:
                continue

            # --- 3. Build the trade ---
            pos = self._build_position(strategy, direction, row, hist, today_date, capital)
            if pos is None:
                continue

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
            exit_info = self._force_close(pos, last_row, last_date)
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

    def _get_direction(self, hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Get market direction prediction.

        Small accounts (<$2k): mechanical put-selling (bullish bias + IV filter).
        Larger accounts: ML ensemble for iron condor timing.
        """
        capital = getattr(self, '_current_capital', self._initial_capital)

        # Small accounts: mechanical put credit spread strategy
        # No ML — just sell puts when IV is high and trend is up
        if capital < 2000:
            return self._mechanical_put_signal(hist)

        # Standard: use ensemble predictor
        if self._predictor is not None:
            try:
                pred = self._predictor.predict(hist)
                if pred is not None:
                    return pred.direction, pred.confidence
            except Exception:
                pass
        return self._simple_direction(hist)

    def _mechanical_put_signal(self, hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Mechanical put credit spread signal for small accounts.

        Rules (from Option Alpha 5-year SPY study):
        1. Sell put spreads only (bullish bias — market up 70% of time)
        2. Only when IV rank > 50 (fat premium)
        3. Only when price > SMA50 (uptrend confirmed)
        4. Skip if recent drop > 3% in 5 days (catching falling knife)
        """
        close = hist["Close"].values
        if len(close) < 60:
            return SignalDirection.NEUTRAL, 0.0

        # IV rank proxy
        from ait.backtesting.options_sim import realized_vol
        rv_20 = realized_vol(close, window=20)
        rv_long = realized_vol(close, window=60) if len(close) > 61 else rv_20
        iv_rank = 0.5  # default
        if rv_long > 0:
            iv_rank = min(1.0, rv_20 / rv_long)

        # Trend: price above SMA50
        sma_50 = float(np.mean(close[-50:]))
        price = float(close[-1])
        above_sma50 = price > sma_50

        # Recent momentum: not falling off a cliff
        ret_5 = price / float(close[-5]) - 1 if len(close) >= 5 else 0.0
        not_crashing = ret_5 > -0.03

        # All conditions met = sell put spread
        if iv_rank > 0.5 and above_sma50 and not_crashing:
            # Confidence scales with IV rank — fatter premium = better trade
            confidence = 0.70 + (iv_rank - 0.5) * 0.4  # 0.70 to 0.90
            return SignalDirection.BULLISH, min(confidence, 0.95)

        # Conditions not met = sit on hands
        return SignalDirection.NEUTRAL, 0.0

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

    def _select_strategy(self, direction: SignalDirection, hist: pd.DataFrame, confidence: float = 0.65) -> str | None:
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

        # Strategy selection: prefer iron condor when capital allows.
        # Small accounts use credit spreads until they can afford condors.
        has_condor = bool(available & {"iron_condor"})

        # Check if capital supports iron condor (needs ~$500+ for $5 wide wings)
        capital = getattr(self, '_current_capital', self._initial_capital)
        condor_affordable = capital >= 2000 and has_condor

        if capital < 2000:
            # Micro tier: put credit spreads (sell puts = collect premium, bullish bias)
            candidates = available & {"put_credit_spread"}
            if not candidates:
                candidates = available & {"bull_call_spread"}
        elif condor_affordable:
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

    def _get_iv(self, hist: pd.DataFrame) -> float:
        """Estimate implied volatility from realized vol with a premium."""
        close_arr = hist["Close"].values
        rv = realized_vol(close_arr, window=20)
        iv = rv * 1.15  # IV premium over realized vol
        return max(iv, 0.10)

    def _build_position(
        self,
        strategy: str,
        direction: SignalDirection,
        row: pd.Series,
        hist: pd.DataFrame,
        today_date: date,
        capital: float,
    ) -> dict | None:
        """Build a position dict with proper BS pricing for the strategy type."""
        underlying = row["Close"]
        iv = self._get_iv(hist)
        dte = self._max_hold_days
        t = dte / 365.0
        r = 0.05

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
            strike = find_strike_by_delta(S, t, iv, 0.30, OptionType.CALL, r)
            price = black_scholes_price(S, strike, t, r, iv, OptionType.CALL)
            opt_type = "call"
        elif strategy == "long_put":
            strike = find_strike_by_delta(S, t, iv, -0.30, OptionType.PUT, r)
            price = black_scholes_price(S, strike, t, r, iv, OptionType.PUT)
            opt_type = "put"
        elif strategy == "bull_call_spread":
            long_strike = find_strike_by_delta(S, t, iv, 0.40, OptionType.CALL, r)
            short_strike = find_strike_by_delta(S, t, iv, 0.20, OptionType.CALL, r)
            long_price = black_scholes_price(S, long_strike, t, r, iv, OptionType.CALL)
            short_price = black_scholes_price(S, short_strike, t, r, iv, OptionType.CALL)
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
            long_strike = find_strike_by_delta(S, t, iv, -0.40, OptionType.PUT, r)
            short_strike = find_strike_by_delta(S, t, iv, -0.20, OptionType.PUT, r)
            long_price = black_scholes_price(S, long_strike, t, r, iv, OptionType.PUT)
            short_price = black_scholes_price(S, short_strike, t, r, iv, OptionType.PUT)
            price = long_price - short_price  # Net debit
            strike = long_strike
            opt_type = "put"
            if price <= 0:
                return None
            return self._finalize_spread_position(
                strategy, "debit", price, S, iv, dte, today_date, capital,
                long_strike=long_strike, short_strike=short_strike, opt_type="put",
            )
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
            # Short legs at 0.20 delta
            short_call_strike = find_strike_by_delta(S, t, iv, 0.20, OptionType.CALL, r)
            short_put_strike = find_strike_by_delta(S, t, iv, -0.20, OptionType.PUT, r)

            # Wings: use 1-sigma expected move (68% probability price stays inside)
            # Full expected_move width means ~68% chance of expiring at max profit
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing_width = max(expected_move, S * 0.05, 5.0)
            long_call_strike = short_call_strike + wing_width
            long_put_strike = short_put_strike - wing_width

            # Price all legs
            short_call_price = black_scholes_price(S, short_call_strike, t, r, iv, OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r, iv, OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r, iv, OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r, iv, OptionType.PUT)

            # Net credit received
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                return None

            # Max loss = wing width - net credit (per share)
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            # Position sizing based on max loss (margin requirement)
            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                return None

            # Apply slippage: we receive less credit than mid
            net_credit *= (1 - self._slippage_pct)

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
            # Short put at 0.20 delta (OTM), long put further OTM for protection
            short_put_strike = find_strike_by_delta(S, t, iv, -0.20, OptionType.PUT, r)

            # Wing width: $1-2 for micro accounts, scales with capital
            wing = max(1.0, min(S * 0.01, 5.0))  # ~1% of stock price, max $5
            if capital < 2000:
                wing = min(wing, 2.0)  # Cap at $2 for micro

            long_put_strike = short_put_strike - wing

            short_put_price = black_scholes_price(S, short_put_strike, t, r, iv, OptionType.PUT)
            long_put_price = black_scholes_price(S, long_put_strike, t, r, iv, OptionType.PUT)

            net_credit = short_put_price - long_put_price
            if net_credit <= 0:
                return None

            max_loss_per_share = wing - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None

            # Position sizing
            size_pct = 0.10 if capital < 2000 else self._position_size_pct
            contracts = int(capital * size_pct / max_loss_per_contract)
            if contracts < 1:
                if max_loss_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None

            net_credit *= (1 - self._slippage_pct)

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
        # Tier-aware position sizing: micro accounts risk more per trade (fewer trades)
        size_pct = 0.08 if capital < 2000 else self._position_size_pct
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

    def _reprice_position(self, pos: dict, underlying: float, days_held: int) -> float:
        """Reprice a position at the current underlying using Black-Scholes.

        Returns the current value per share:
        - For debit positions: current option/spread value (want it to go UP)
        - For credit positions: current cost to buy back (want it to go DOWN)
        """
        iv = pos["entry_iv"]
        dte_remaining = max(self._max_hold_days - days_held, 0)
        t = max(dte_remaining / 365.0, 0.0001)
        r = 0.05

        strategy = pos["strategy"]

        if strategy == "iron_condor":
            # Reprice all 4 legs
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r, iv, OptionType.CALL)
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r, iv, OptionType.PUT)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r, iv, OptionType.CALL)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r, iv, OptionType.PUT)
            # Cost to buy back the iron condor (close the position)
            return (sc + sp) - (lc + lp)

        elif strategy == "put_credit_spread":
            # Credit spread: cost to buy back = short put value - long put value
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r, iv, OptionType.PUT)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r, iv, OptionType.PUT)
            return sp - lp  # Cost to close (want this to decrease)

        elif strategy in ("bull_call_spread", "bear_put_spread"):
            # Reprice the spread
            long_strike = pos["long_strike"]
            short_strike = pos["short_strike"]
            if pos["option_type"] == "call":
                long_val = black_scholes_price(underlying, long_strike, t, r, iv, OptionType.CALL)
                short_val = black_scholes_price(underlying, short_strike, t, r, iv, OptionType.CALL)
            else:
                long_val = black_scholes_price(underlying, long_strike, t, r, iv, OptionType.PUT)
                short_val = black_scholes_price(underlying, short_strike, t, r, iv, OptionType.PUT)
            return long_val - short_val  # Spread value

        else:
            # Single-leg option
            opt_type = OptionType.CALL if pos["option_type"] == "call" else OptionType.PUT
            return black_scholes_price(underlying, pos["strike"], t, r, iv, opt_type)

    def _check_exit(self, pos: dict, row: pd.Series, current_date: date) -> dict | None:
        """Check if a position should be exited."""
        underlying = row["Close"]
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (current_date - entry_date).days

        current_value = self._reprice_position(pos, underlying, days_held)

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

        # Apply slippage to current value for exit
        if trade_type == "credit":
            current_value *= (1 + self._slippage_pct)  # Buy back at ask
        else:
            current_value *= (1 - self._slippage_pct)  # Sell at bid

        if self._trailing_stop_enabled:
            result = self._check_exit_trailing(pos, pnl_pct, current_date)
        else:
            result = self._check_exit_fixed(pos, pnl_pct, current_date)

        if result is not None:
            # Calculate actual P&L
            pnl = self._calc_pnl(pos, current_value)
            result["pnl"] = round(pnl, 2)
            result["exit_price"] = round(current_value, 4)
            return result

        return None

    def _check_exit_fixed(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Fixed stop-loss / take-profit."""
        # Tier-aware stop loss: tighter for small accounts doing directional bets
        capital = getattr(self, '_current_capital', self._initial_capital)
        stop = 0.25 if capital < 2000 else self._stop_loss_pct

        # Stop loss
        if pnl_pct <= -stop:
            return {"exit_date": str(current_date), "exit_reason": "stop_loss"}

        # Profit target
        target = self._profit_target_pct
        if pos.get("trade_type") == "credit":
            # For credit trades: take profit at 50% of max credit collected
            target = min(self._profit_target_pct, 0.50)
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

    def _force_close(self, pos: dict, last_row: pd.Series, last_date: date) -> dict:
        """Force-close a position at end of backtest."""
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (last_date - entry_date).days
        current_value = self._reprice_position(pos, last_row["Close"], days_held)

        trade_type = pos.get("trade_type", "debit")
        if trade_type == "credit":
            current_value *= (1 + self._slippage_pct)
        else:
            current_value *= (1 - self._slippage_pct)

        pnl = self._calc_pnl(pos, current_value)
        return {
            "exit_date": str(last_date),
            "exit_price": round(current_value, 4),
            "pnl": round(pnl, 2),
            "exit_reason": "backtest_end",
        }
