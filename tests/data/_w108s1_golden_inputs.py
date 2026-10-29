"""W108 S1: the inputs of the default walk-forward golden record (``tests/data/w108s1_walkforward_golden.json``).

A small, seeded, fully offline walk-forward: two synthetic symbols, iron condors only, no window models
(``train_window_models=False``, the research-study harness), the macro calendar off, a $5,000 pool, five 63-day
test windows. The VIX / SPY context loader is replaced by an empty frame so nothing reads a store or the network.

The record was captured at commit 11be785 (before W108 S1) by running this file as a script::

    python tests/data/_w108s1_golden_inputs.py

and is compared by ``tests/test_w108s1_carried_equity.py``: with ``carry_equity`` off (the default) every trade,
every window and every metric except ``total_return`` (and the two figures derived from it) must equal it.
If a later wave moves a research default this record moves with it: regenerate it and say so in PLAN.md.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN_PATH = Path(__file__).with_name("w108s1_walkforward_golden.json")
SYMBOLS = ["SPY", "QQQ"]
CAPITAL = 5000.0


def make_data() -> dict[str, pd.DataFrame]:
    """Seeded synthetic daily bars, one frame per symbol (different seeds, different paths)."""
    out = {}
    for k, (sym, start) in enumerate((("SPY", 450.0), ("QQQ", 380.0))):
        rng = np.random.default_rng(108 + k)
        idx = pd.bdate_range("2022-01-03", periods=620)
        close = start * np.cumprod(1 + rng.normal(0.0002, 0.012, len(idx)))
        out[sym] = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, len(idx))),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, len(idx)))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, len(idx)))),
            "Close": close,
            "Volume": 2_000_000.0,
        }, index=idx)
    return out


def make_config(**overrides):
    from ait.backtesting.walkforward import WalkForwardConfig
    kw = dict(train_days=200, test_days=63, step_days=63, gap_days=5, initial_capital=CAPITAL,
              train_window_models=False, macro_event_gate=False)
    kw.update(overrides)
    return WalkForwardConfig(**kw)


def _empty_context(*_a, **_k) -> pd.DataFrame:
    return pd.DataFrame()


def _no_earnings(*_a, **_k) -> set:
    return set()


def run(config=None, data=None):
    """Run the REAL ``WalkForwardBacktester.run`` on the fixture, offline: the VIX / SPY context loader returns an empty
    frame and the engine's earnings-date lookup (a yfinance request; ETFs have none) an empty set. Both restored after."""
    import ait.data.market_data as md
    from ait.backtesting.engine import Backtester
    from ait.backtesting.walkforward import WalkForwardBacktester
    saved_loader, saved_earnings = md.load_daily_ohlcv, Backtester.__dict__["_load_earnings_dates"]
    md.load_daily_ohlcv = _empty_context
    Backtester._load_earnings_dates = staticmethod(_no_earnings)
    try:
        bt = WalkForwardBacktester(list(SYMBOLS), ["iron_condor"], config=config or make_config())
        return asyncio.run(bt.run(data=data if data is not None else make_data()))
    finally:
        md.load_daily_ohlcv = saved_loader
        Backtester._load_earnings_dates = saved_earnings


_TRADE_KEYS = ("symbol", "strategy", "entry_date", "exit_date", "exit_reason", "contracts", "pnl", "max_loss",
               "credit", "entry_price", "exit_price")
_METRICS = ("total_trades", "win_rate", "sharpe_ratio", "sortino_ratio", "max_drawdown", "profit_factor",
            "avg_window_return", "consistency", "avg_win", "avg_loss", "expectancy", "best_trade", "worst_trade",
            "capital_utilization", "total_return", "cash_drag_adjusted_return", "raroc")


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def snapshot(result) -> dict:
    """Everything a default run reports, as plain JSON values."""
    windows = []
    for w in result.windows:
        br = w.backtest_result
        windows.append({
            "window_id": w.window_id, "test_start": str(w.test_start), "test_end": str(w.test_end),
            "initial_capital": br.initial_capital, "final_capital": br.final_capital,
            "total_return": br.total_return, "max_drawdown": br.max_drawdown,
            "pool_rejections": {k: int(v) for k, v in sorted(w.pool_rejections.items())},
            "trades": [{k: _plain(t.get(k)) for k in _TRADE_KEYS} for t in br.trades],
        })
    ec = result.equity_curve()
    return {
        "windows": windows,
        "metrics": {m: _plain(getattr(result, m)) for m in _METRICS},
        "symbol_results": {s: {"initial_capital": r.initial_capital, "final_capital": r.final_capital,
                               "total_trades": r.total_trades}
                           for s, r in sorted(result.symbol_results.items())},
        "strategy_results": {s: {k: _plain(v) for k, v in sorted(d.items())}
                             for s, d in sorted(result.strategy_results.items())},
        "equity_curve": [[str(d), float(e)] for d, e in zip(ec["date"], ec["equity"])],
        "entry_funnel": result.entry_funnel,
    }


if __name__ == "__main__":
    snap = snapshot(run())
    GOLDEN_PATH.write_text(json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}: {len(snap['windows'])} windows, {snap['metrics']['total_trades']} trades")
