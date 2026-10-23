"""W92: seeded inputs for the regime-detector golden record (tests/data/w92_regime_golden.json).

The golden file was recorded from the code BEFORE the 15 regime numbers became settings; the test replays these inputs and must
reproduce it exactly (every class, confidence and trend strength).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

VIXES = (None, 12.0, 18.0, 25.0, 35.0)


def panels():
    """60 seeded 300-bar price panels spanning drift and volatility, each with a VIX level."""
    out = []
    for i in range(60):
        rng = np.random.default_rng(1000 + i)
        drift = (-0.003, -0.001, 0.0, 0.001, 0.003)[i % 5]
        vol = (0.004, 0.008, 0.015, 0.025, 0.04)[(i // 5) % 5]
        close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, 300)))
        idx = pd.bdate_range("2023-01-02", periods=300)
        df = pd.DataFrame({"Open": close, "High": close * 1.003, "Low": close * 0.997, "Close": close, "Volume": 1e6}, index=idx)
        out.append((df, VIXES[i % len(VIXES)]))
    return out


def classify_grid():
    """Every branch of RegimeDetector._classify: trend x vol x vix x iv_rank x hurst."""
    return list(itertools.product((-0.6, -0.4, -0.2, 0.0, 0.2, 0.5), (0.1, 0.2, 0.3, 0.45, 0.7),
                                  (None, 12.0, 20.0, 33.0), (0.1, 0.5, 0.9), (0.3, 0.5, 0.7)))


def replay(detector):
    """The record the golden file holds, computed with ``detector``."""
    out = {"analyze": [], "classify": []}
    for df, vix in panels():
        a = detector.analyze(df, vix)
        out["analyze"].append([a.regime.value, round(float(a.confidence), 12), round(float(a.trend_strength), 12),
                               round(float(a.volatility_level), 12), round(float(a.iv_rank), 12), round(float(a.hurst_exponent), 12)])
    for trend, vol, vix, ivr, hurst in classify_grid():
        r, c = detector._classify(trend, vol, vix, ivr, hurst)
        out["classify"].append([r.value, round(float(c), 12)])
    return out
