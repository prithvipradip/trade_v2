"""Scenario generator shared by the W88b golden-behaviour test and the script that recorded the golden file."""
from __future__ import annotations

import random


def trades(rng, n, strategy, symbol, win_rate, conf_lo, conf_hi, win=60.0, loss=-90.0):
    out = []
    for _ in range(n):
        won = rng.random() < win_rate
        out.append({"strategy": strategy, "symbol": symbol, "pnl": win if won else loss,
                    "entry_confidence": round(rng.uniform(conf_lo, conf_hi), 3)})
    return out


def scenarios():
    """name -> list of windows, each a list of trade dicts. Seeded: identical on every run."""
    rng = random.Random(8801)
    s = {}
    s["losing_protected_condor"] = [trades(rng, 14, "iron_condor", "QQQ", 0.22, 0.60, 0.80) for _ in range(5)]
    s["losing_then_recovering_unprotected"] = (
        [trades(rng, 14, "long_call", "SPY", 0.10, 0.60, 0.80) for _ in range(6)]
        + [trades(rng, 14, "long_call", "SPY", 0.95, 0.60, 0.80) for _ in range(8)])
    s["winning_spread"] = [trades(rng, 14, "bull_call_spread", "QQQ", 0.75, 0.70, 0.90) for _ in range(5)]
    s["low_confidence_band_and_bad_symbol"] = [
        trades(rng, 8, "iron_condor", "IWM", 0.05, 0.55, 0.64) + trades(rng, 8, "iron_condor", "QQQ", 0.7, 0.66, 0.9)
        for _ in range(6)]
    s["disable_then_reenable"] = (
        [trades(rng, 14, "long_put", "QQQ", 0.25, 0.70, 0.90) for _ in range(10)]
        + [trades(rng, 14, "long_put", "QQQ", 0.97, 0.70, 0.90) for _ in range(18)])
    return s
