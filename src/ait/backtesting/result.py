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
            "=" * 50,
            "  BACKTEST RESULTS",
            "=" * 50,
            f"  Initial Capital: ${self.initial_capital:,.2f}",
            f"  Final Capital:   ${self.final_capital:,.2f}",
            f"  Total Return:    {self.total_return:.2%}",
            duration.rstrip("\n") if duration else None,
            "-" * 50,
            f"  Total Trades:    {self.total_trades}",
            f"  Win Rate:        {self.win_rate:.2%}",
            f"  Avg Trade P&L:   ${self.avg_trade_pnl:,.2f}",
            f"  Profit Factor:   {self.profit_factor:.2f}",
            f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}",
            f"  Max Drawdown:    {self.max_drawdown:.2%}",
            "=" * 50,
        ]
        return "\n".join(line for line in lines if line is not None)
