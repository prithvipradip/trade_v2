"""Backtest result container with performance metrics.

Stores simulated trades and computes standard trading performance
metrics for strategy evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Complete backtest output with trades and performance metrics."""

    trades: list[dict] = field(default_factory=list)
    initial_capital: float = 10_000.0
    final_capital: float = 10_000.0
    start_date: date | None = None
    end_date: date | None = None
    exit_mode: str = "fixed"

    # --- Core metrics as properties ---

    @property
    def total_return(self) -> float:
        """Total return as a decimal (e.g., 0.15 = 15%)."""
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of trades with positive P&L."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.get("pnl", 0) for t in self.trades) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross wins / gross losses. >1 is profitable."""
        gross_wins = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) > 0)
        gross_losses = abs(sum(t["pnl"] for t in self.trades if t.get("pnl", 0) <= 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from per-trade P&L."""
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.get("pnl", 0) for t in self.trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        if std_pnl == 0:
            return 0.0
        return float((mean_pnl / std_pnl) * math.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a decimal (e.g., 0.10 = 10%)."""
        if not self.trades:
            return 0.0

        equity = self.initial_capital
        peak = equity
        max_dd = 0.0

        for t in self.trades:
            equity += t.get("pnl", 0)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio — like Sharpe but only penalizes downside volatility.

        Better for option-selling strategies (skewed returns, capped upside).
        """
        if len(self.trades) < 2:
            return 0.0
        pnls = np.array([t.get("pnl", 0) for t in self.trades])
        mean_pnl = float(pnls.mean())
        downside = pnls[pnls < 0]
        if len(downside) < 2:
            # Not enough losses to estimate downside vol — fall back to Sharpe-like
            return float("inf") if mean_pnl > 0 else 0.0
        downside_std = float(downside.std(ddof=1))
        if downside_std == 0:
            return 0.0
        return (mean_pnl / downside_std) * math.sqrt(252)

    @staticmethod
    def _to_date(d):
        """Coerce strings/datetimes to date objects."""
        from datetime import datetime, date as _date
        if d is None:
            return None
        if isinstance(d, _date) and not isinstance(d, datetime):
            return d
        if hasattr(d, "date"):
            return d.date()
        if isinstance(d, str):
            try:
                return datetime.fromisoformat(d.split("T")[0]).date()
            except Exception:
                return None
        return None

    @property
    def avg_win(self) -> float:
        """Average $ size of winning trades."""
        wins = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        """Average $ size of losing trades (absolute value)."""
        losses = [abs(t["pnl"]) for t in self.trades if t.get("pnl", 0) <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def win_loss_ratio(self) -> float:
        """avg_win / avg_loss. >1 means avg winner bigger than avg loser."""
        if self.avg_loss == 0:
            return float("inf") if self.avg_win > 0 else 0.0
        return self.avg_win / self.avg_loss

    @property
    def expectancy(self) -> float:
        """Expected $ per trade: (win_rate * avg_win) - (loss_rate * avg_loss)."""
        wr = self.win_rate
        return wr * self.avg_win - (1 - wr) * self.avg_loss

    @property
    def worst_trade(self) -> float:
        """$ of the single biggest losing trade."""
        if not self.trades:
            return 0.0
        return min(t.get("pnl", 0) for t in self.trades)

    @property
    def best_trade(self) -> float:
        """$ of the single biggest winning trade."""
        if not self.trades:
            return 0.0
        return max(t.get("pnl", 0) for t in self.trades)

    @property
    def avg_hold_days(self) -> float:
        """Average holding period (calendar days) per closed trade."""
        days = []
        for t in self.trades:
            entry = self._to_date(t.get("entry_date"))
            exit_ = self._to_date(t.get("exit_date"))
            if entry and exit_:
                days.append((exit_ - entry).days)
        return float(np.mean(days)) if days else 0.0

    @property
    def capital_utilization(self) -> float:
        """Avg % of capital deployed across the backtest period.

        Sums max-loss exposure of each trade × hold-time, divides by
        capital × total backtest days. Tells you how busy the bot is.
        """
        if not self.trades or not self.start_date or not self.end_date:
            return 0.0
        total_days = max(1, (self.end_date - self.start_date).days)
        capital_days = 0.0
        for t in self.trades:
            entry = self._to_date(t.get("entry_date"))
            exit_ = self._to_date(t.get("exit_date"))
            risk = t.get("max_loss") or t.get("cost") or abs(t.get("pnl", 0)) * 2
            if entry and exit_ and risk:
                hold_days = max(1, (exit_ - entry).days)
                capital_days += risk * hold_days
        if self.initial_capital <= 0:
            return 0.0
        return capital_days / (self.initial_capital * total_days)

    @property
    def cash_drag_adjusted_return(self) -> float:
        """Total return adjusted for idle cash earning T-bill yield (~5%/yr).

        Adds (1 - utilization) × 5% × duration_years to the raw return.
        """
        if not self.start_date or not self.end_date:
            return self.total_return
        years = max(0.01, (self.end_date - self.start_date).days / 365.25)
        idle_pct = max(0, 1 - self.capital_utilization)
        cash_yield = 0.05 * idle_pct * years
        return self.total_return + cash_yield

    @property
    def raroc(self) -> float:
        """Return on capital actually deployed (not idle).

        If utilization is 10% and total return is 50%, RAROC = 500%.
        Honest measure of strategy edge per dollar at risk.
        """
        util = self.capital_utilization
        if util <= 0:
            return 0.0
        return self.total_return / util

    @property
    def drawdown_duration_days(self) -> int:
        """Longest drawdown duration in calendar days."""
        if not self.trades:
            return 0
        equity = self.initial_capital
        peak = equity
        peak_date = self.start_date
        max_duration = 0
        for t in self.trades:
            equity += t.get("pnl", 0)
            ex_date = self._to_date(t.get("exit_date") or t.get("entry_date"))
            if equity >= peak:
                peak = equity
                peak_date = ex_date
            elif peak_date and ex_date:
                duration = (ex_date - peak_date).days
                max_duration = max(max_duration, duration)
        return max_duration

    # --- Output methods ---

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        for col in ("entry_date", "exit_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def summary(self) -> str:
        """Return a formatted text summary of backtest results."""
        duration = ""
        if self.start_date and self.end_date:
            days = (self.end_date - self.start_date).days
            duration = f"  Duration:        {days} calendar days\n"

        lines = [
            "=" * 60,
            f"  BACKTEST RESULTS  (exit_mode={self.exit_mode})",
            "=" * 60,
            f"  Initial Capital:    ${self.initial_capital:,.2f}",
            f"  Final Capital:      ${self.final_capital:,.2f}",
            f"  Total Return:       {self.total_return:.2%}",
            f"  Cash-Drag Adj Ret:  {self.cash_drag_adjusted_return:.2%}  (idle cash @ 5% T-bill)",
            duration.rstrip("\n") if duration else None,
            "-" * 60,
            "  RISK-ADJUSTED",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:      {self.sortino_ratio:.2f}  (downside-only vol)",
            f"  Max Drawdown:       {self.max_drawdown:.2%}",
            f"  Drawdown Duration:  {self.drawdown_duration_days} days",
            "-" * 60,
            "  TRADE QUALITY",
            f"  Total Trades:       {self.total_trades}",
            f"  Win Rate:           {self.win_rate:.2%}",
            f"  Avg Trade P&L:      ${self.avg_trade_pnl:,.2f}",
            f"  Avg Win:            ${self.avg_win:,.2f}",
            f"  Avg Loss:           ${self.avg_loss:,.2f}",
            f"  Win/Loss Ratio:     {self.win_loss_ratio:.2f}  (>1 = winners bigger)",
            f"  Expectancy/Trade:   ${self.expectancy:,.2f}",
            f"  Best Trade:         ${self.best_trade:,.2f}",
            f"  Worst Trade:        ${self.worst_trade:,.2f}",
            f"  Profit Factor:      {self.profit_factor:.2f}",
            f"  Avg Hold Days:      {self.avg_hold_days:.1f}",
            "-" * 60,
            "  CAPITAL EFFICIENCY",
            f"  Utilization:        {self.capital_utilization:.1%}  (avg % deployed)",
            f"  RAROC:              {self.raroc:.1%}  (return on deployed)",
            "=" * 60,
        ]
        return "\n".join(line for line in lines if line is not None)
