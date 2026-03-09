"""Backtesting engine for simulating trading strategies on historical data.

Runs the trading loop day-by-day over an OHLCV DataFrame, applying
ML predictions (if available), strategy selection, risk rules,
and trade simulation with commission/slippage modeling.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ait.backtesting.result import BacktestResult
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("backtesting.engine")


class Backtester:
    """Simulates trading strategies against historical OHLCV data.

    Args:
        data: DataFrame with columns [Open, High, Low, Close, Volume] and
              a DatetimeIndex or a 'Date' column.
        strategies: List of strategy names to simulate (e.g., ["long_call", "bull_call_spread"]).
        initial_capital: Starting account balance in dollars.
        commission_per_contract: Round-trip commission per contract.
        slippage_pct: Slippage as fraction of option price (0.01 = 1%).
        position_size_pct: Fraction of account to risk per trade.
        stop_loss_pct: Stop-loss as fraction of entry price.
        profit_target_pct: Profit target as fraction of entry price.
        max_hold_days: Maximum days to hold before forced exit (expiry sim).
        min_confidence: Minimum ML confidence to take a trade.
    """

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

        # Try to load the ML predictor (optional)
        self._predictor = self._load_predictor()

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

        # Minimum lookback for direction estimation
        lookback = 20

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

            # --- 2. Generate new signal (one trade per day max) ---
            if open_positions:
                # Simplified: only one position at a time
                continue

            direction, confidence = self._get_direction(hist)

            if confidence < self._min_confidence:
                continue

            strategy = self._select_strategy(direction)
            if strategy is None:
                continue

            # --- 3. Simulate entry ---
            entry_price = self._simulate_entry_price(row, direction)
            if entry_price <= 0:
                continue

            contracts = self._calculate_contracts(capital, entry_price)
            if contracts < 1:
                continue

            entry_cost = self._commission * contracts
            capital -= entry_cost  # Commission on entry

            pos = {
                "symbol": "SIM",
                "strategy": strategy,
                "direction": direction.value,
                "entry_date": str(today_date),
                "entry_price": round(entry_price, 2),
                "contracts": contracts,
                "stop_loss": round(entry_price * (1 - self._stop_loss_pct), 2),
                "profit_target": round(entry_price * (1 + self._profit_target_pct), 2),
                "expiry_date": str(today_date + timedelta(days=self._max_hold_days)),
                "entry_commission": entry_cost,
            }
            open_positions.append(pos)

        # --- Force-close any remaining open positions at last close ---
        last_row = self._data.iloc[-1]
        last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
        for pos in open_positions:
            exit_price = self._apply_slippage(last_row["Close"], is_entry=False)
            pnl = self._calc_pnl(pos, exit_price)
            pos.update({
                "exit_date": str(last_date),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "exit_reason": "backtest_end",
            })
            capital += pnl
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

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has a DatetimeIndex and required columns."""
        df = data.copy()

        # Normalize column names
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

        # Ensure DatetimeIndex
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
        """Try to load trained DirectionPredictor. Returns None if unavailable."""
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

    def _get_direction(self, hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Get market direction prediction.

        Uses ML model if available, otherwise falls back to a simple
        returns-based heuristic.
        """
        # Try ML predictor first
        if self._predictor is not None:
            try:
                pred = self._predictor.predict(hist)
                if pred is not None:
                    return pred.direction, pred.confidence
            except Exception:
                pass

        # Fallback: simple momentum-based direction
        return self._simple_direction(hist)

    @staticmethod
    def _simple_direction(hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Estimate direction from recent returns and trend."""
        close = hist["Close"]

        # Short-term momentum (5-day return)
        ret_5 = close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0.0
        # Medium-term momentum (20-day return)
        ret_20 = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0.0

        # Combined score
        score = 0.6 * ret_5 + 0.4 * ret_20

        # Convert to direction + confidence
        threshold = 0.005  # 0.5%
        if score > threshold:
            direction = SignalDirection.BULLISH
            confidence = min(0.5 + abs(score) * 10, 0.95)
        elif score < -threshold:
            direction = SignalDirection.BEARISH
            confidence = min(0.5 + abs(score) * 10, 0.95)
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.4  # Low confidence for neutral

        return direction, confidence

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, direction: SignalDirection) -> str | None:
        """Pick a strategy based on direction and available strategies."""
        bullish = {"long_call", "bull_call_spread"}
        bearish = {"long_put", "bear_put_spread"}
        neutral = {"iron_condor", "short_strangle", "short_straddle"}

        available = set(self._strategies)

        if direction == SignalDirection.BULLISH:
            candidates = available & bullish
        elif direction == SignalDirection.BEARISH:
            candidates = available & bearish
        else:
            candidates = available & neutral

        if not candidates:
            return None

        # Pick the first available (deterministic for reproducibility)
        return sorted(candidates)[0]

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def _simulate_entry_price(
        self, row: pd.Series, direction: SignalDirection
    ) -> float:
        """Simulate an option entry price from the underlying bar.

        Uses a fraction of the underlying price as a rough proxy for
        an at-the-money option premium.
        """
        underlying = row["Close"]

        # ATM option price proxy: ~3-5% of underlying for ~30 DTE
        option_price_ratio = 0.04
        base_price = underlying * option_price_ratio

        # Add slippage (paying more on entry)
        entry_price = self._apply_slippage(base_price, is_entry=True)
        return entry_price

    def _apply_slippage(self, price: float, is_entry: bool) -> float:
        """Apply slippage model. Entry pays more, exit receives less."""
        if is_entry:
            return price * (1 + self._slippage_pct)
        return price * (1 - self._slippage_pct)

    def _calculate_contracts(self, capital: float, entry_price: float) -> int:
        """Determine number of contracts based on position sizing.

        Each option contract = 100 shares, so cost = entry_price * 100 * contracts.
        """
        risk_amount = capital * self._position_size_pct
        cost_per_contract = entry_price * 100  # Options multiplier
        if cost_per_contract <= 0:
            return 0
        return max(1, int(risk_amount / cost_per_contract))

    def _check_exit(
        self,
        pos: dict,
        row: pd.Series,
        current_date: date,
    ) -> dict | None:
        """Check if a position should be exited.

        Returns exit info dict if exiting, None if holding.
        """
        current_option_price = row["Close"] * 0.04  # Same proxy as entry

        # Stop loss
        if current_option_price <= pos["stop_loss"]:
            exit_price = self._apply_slippage(pos["stop_loss"], is_entry=False)
            pnl = self._calc_pnl(pos, exit_price)
            return {
                "exit_date": str(current_date),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "exit_reason": "stop_loss",
            }

        # Profit target
        if current_option_price >= pos["profit_target"]:
            exit_price = self._apply_slippage(pos["profit_target"], is_entry=False)
            pnl = self._calc_pnl(pos, exit_price)
            return {
                "exit_date": str(current_date),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "exit_reason": "profit_target",
            }

        # Expiry
        expiry = date.fromisoformat(pos["expiry_date"])
        if current_date >= expiry:
            exit_price = self._apply_slippage(current_option_price, is_entry=False)
            pnl = self._calc_pnl(pos, exit_price)
            return {
                "exit_date": str(current_date),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl, 2),
                "exit_reason": "expiry",
            }

        return None

    def _calc_pnl(self, pos: dict, exit_price: float) -> float:
        """Calculate P&L for a position including commissions.

        For long options: pnl = (exit - entry) * 100 * contracts - commissions
        """
        contracts = pos["contracts"]
        raw_pnl = (exit_price - pos["entry_price"]) * 100 * contracts

        # If bearish direction, invert the P&L logic (put gains when price drops)
        if pos.get("direction") == SignalDirection.BEARISH.value:
            raw_pnl = -raw_pnl

        exit_commission = self._commission * contracts
        total_commission = pos.get("entry_commission", 0) + exit_commission

        return raw_pnl - total_commission
