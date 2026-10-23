"""Inputs and capture functions for the W97 golden record (macro columns at real values, intraday features, short contexts).

The W88d golden record covers the VIX / SPY cross-asset group, the market-structure group and the fractal group at the
default windows, but not the macro columns (its market context has no ``macros``) nor ``compute_intraday_features``.
``w97_golden.json`` was recorded with the code AS IT WAS before W97 (windows still literals), so a test that recomputes
everything here and compares proves the W97 settings reproduce the old behaviour at their defaults. Seeded: identical
on every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _w88d_golden_inputs as gi88  # tests/data is on sys.path (the test inserts it, as the W88d test does)

TAIL = 60
INTRADAY_SEED = 8     # chosen so that EVERY session slice ends with both up and down bars (RSI defined at periods 5 and 7)


def macros(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    rng = np.random.default_rng(21)
    n = len(index)
    y2 = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.02, n)), index=index)
    y10 = pd.Series(4.3 + np.cumsum(rng.normal(0, 0.02, n)), index=index)
    curve = y10 - y2 - 0.1                                     # crosses zero: exercises the inversion flag
    dxy = pd.Series(103.0 + np.cumsum(rng.normal(0, 0.25, n)), index=index)
    return {"us_2y_yield": y2, "us_10y_yield": y10, "yield_curve": curve, "dxy": dxy}


def full_context() -> tuple[pd.DataFrame, dict]:
    ctx = gi88.market_context()
    base = ctx["_base"]
    return base, {"vix": ctx["vix"], "spy": ctx["spy"], "macros": macros(base.index)}


def short_context() -> tuple[pd.DataFrame, dict]:
    """A 15-row VIX / SPY (below the 20-row minimum) and one macro series missing: the neutral branches."""
    base, ctx = full_context()
    m = macros(base.index)
    m["dxy"] = None
    return base, {"vix": ctx["vix"].iloc[:15], "spy": ctx["spy"].iloc[:15], "macros": m}


FRACTAL_COLUMNS = ["hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2", "hurst_psd_divergence", "hurst_short",
                   "hurst_long", "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry"]


def stub_fractal(self, df):
    """The fractal group is ~95% of a compute() and is covered at the default windows by the W88d golden record (real
    computation); the W97 golden record replaces it with constants so the other groups are compared in seconds."""
    for c in FRACTAL_COLUMNS:
        df[c] = 0.5
    return df


def compute_frames(engine=None) -> dict[str, pd.DataFrame]:
    """The three golden frames, with the fractal group stubbed (see ``stub_fractal``); the patch is undone on exit."""
    from unittest import mock

    from ait.ml.features import FeatureEngine
    fe = engine if engine is not None else FeatureEngine()
    base, ctx = full_context()
    b2, c2 = short_context()
    with mock.patch.object(FeatureEngine, "_add_fractal_features", stub_fractal):
        return {"macro_full": fe.compute(base, market_context=ctx),
                "macro_short_context": fe.compute(b2, market_context=c2),
                "short_260": fe.compute(base.iloc[:260], market_context={k: (v.iloc[:260] if hasattr(v, "iloc") else v)
                                                                          for k, v in ctx.items() if k != "macros"})}


def intraday_frame(sessions: int = 9) -> pd.DataFrame:
    """Seeded 5-minute bars, 78 per regular session (14:30-21:00 UTC), ``sessions`` consecutive weekdays."""
    rng = np.random.default_rng(INTRADAY_SEED)
    days = pd.bdate_range("2024-06-03", periods=sessions)
    rows = []
    price = 100.0
    for d in days:
        start = pd.Timestamp(d.date().isoformat() + "T13:30:00+00:00")
        for k in range(78):
            ts = start + pd.Timedelta(minutes=5 * k)
            o = price
            price = max(1.0, price * (1 + rng.normal(0, 0.0015)))
            hi = max(o, price) * (1 + abs(rng.normal(0, 0.0006)))
            lo = min(o, price) * (1 - abs(rng.normal(0, 0.0006)))
            rows.append((ts, o, hi, lo, price, float(rng.integers(50_000, 400_000))))
    df = pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("Datetime")
    return df


def compute_intraday(engine=None) -> dict[str, dict[str, float]]:
    """compute_intraday_features on slices that hit every minimum-bar branch (full 702 bars, 250, 150, 90, 45, 25, 10)."""
    from ait.ml.features import FeatureEngine
    fe = engine if engine is not None else FeatureEngine()
    df = intraday_frame()
    out = {}
    for n in (len(df), 600, 250, 150, 90, 45, 25, 10):
        out[str(n)] = {k: float(v) for k, v in fe.compute_intraday_features(df.iloc[-n:]).items()}
    return out
