"""W66: pure helpers for sampling option bid/ask spreads from IB historical BID/ASK bars.

IB serves historical BID and ASK bars for currently LISTED options even with the market
closed (verified 2026-10-09 on IWM 20261030 P270/C285/P255), but NOTHING for expired
contracts. So the spread model is calibrated on recent sessions only and assumed
stationary over the 2024-2026 backtest period (see the W66 PLAN entry for the caveat).

No IO here: `scripts/calibrate_option_spreads.py --source ib-bars` does the IB calls and
feeds bars through these functions, so each step is unit-testable on fake bars.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ait.backtesting.pricing import OptionType, black_scholes_price, bs_delta

ET = ZoneInfo("America/New_York")

# |delta| buckets of the SHORT legs the condor builder can pick (delta_short ~0.16),
# bracketed on both sides so the fit sees what a different delta would cost.
SHORT_DELTA_BUCKETS = (0.08, 0.12, 0.16, 0.20, 0.30, 0.40)
WING_WIDTH_DOLLARS = (2.0, 3.0, 5.0)  # long-leg offsets from the 0.16-delta short strike

# W76: 0-6 DTE collection. At hours-to-expiry a Black-Scholes delta bucket is a poor way to pick
# strikes (the IV is unknown and t is tiny), so the plan is anchored on MONEYNESS: strikes at
# these |K/S - 1| fractions on both sides, plus a few delta buckets and wings. Kept compact:
# IB paces historical requests at 60 per 10 minutes, so every contract costs ~22 s.
SHORT_DTE_MAX = 6
SHORT_DTE_MONEYNESS = (0.0025, 0.005, 0.01, 0.02, 0.035)
SHORT_DTE_DELTA_BUCKETS = (0.08, 0.16, 0.30)
SHORT_DTE_WINGS = (2.0, 5.0)


def parse_window(text: str) -> tuple[time, time]:
    a, b = text.split("-")
    ha, ma = a.split(":")
    hb, mb = b.split(":")
    return time(int(ha), int(ma)), time(int(hb), int(mb))


def bar_start_et(raw) -> datetime:
    """IB bar timestamp (epoch/UTC-aware, or naive-assumed-ET) -> ET-aware datetime."""
    ts = pd.Timestamp(raw)
    ts = ts.tz_localize(ET) if ts.tzinfo is None else ts.tz_convert(ET)
    return ts.to_pydatetime()


def in_entry_window(start: datetime, window: tuple[time, time]) -> bool:
    """Keep only bars that overlap the engine's entry window (start inclusive)."""
    t = start.astimezone(ET).time()
    return window[0] <= t < window[1]


def pair_bid_ask(bid_bars, ask_bars, window: tuple[time, time]) -> pd.DataFrame:
    """Join BID and ASK bars on bar start; keep valid quotes inside the entry window.

    `*_bars` are sequences of objects with `.date` and `.close` (ib_insync BarData).
    Invalid rows (no bid/ask, crossed) are dropped, never repaired.
    """
    bids = {bar_start_et(b.date): float(b.close) for b in bid_bars}
    rows = []
    for a in ask_bars:
        start = bar_start_et(a.date)
        if start not in bids or not in_entry_window(start, window):
            continue
        bid, ask = bids[start], float(a.close)
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask <= 0 or ask < bid:
            continue
        rows.append({"bar_time": start.isoformat(), "bid": bid, "ask": ask})
    return pd.DataFrame(rows, columns=["bar_time", "bid", "ask"])


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _bs_fast(S: float, K: float, t: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes via ``math.erfc``: ~100x faster than SciPy's ``norm.cdf`` per call.

    W76: only the IV INVERSION below uses it (62 pricings per sample row, thousands of rows
    per collection run). The shared ``pricing.black_scholes_price`` is deliberately left
    alone: changing its arithmetic would move every research baseline in the last bits."""
    if t <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * t) / v
    d2 = d1 - v
    if is_call:
        return max(S * _ncdf(d1) - K * math.exp(-r * t) * _ncdf(d2), 0.0)
    return max(K * math.exp(-r * t) * _ncdf(-d2) - S * _ncdf(-d1), 0.0)


def implied_vol_from_mid(mid: float, S: float, K: float, t: float, right: str,
                         r: float = 0.05) -> float | None:
    """Bisection IV; None when mid is below intrinsic or above S (no solution)."""
    is_call = right == "C"
    if mid <= 0 or S <= 0 or K <= 0 or t <= 0:
        return None
    lo, hi = 0.01, 4.0
    if _bs_fast(S, K, t, r, lo, is_call) > mid or _bs_fast(S, K, t, r, hi, is_call) < mid:
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if _bs_fast(S, K, t, r, m, is_call) < mid:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def delta_from_mid(mid: float, S: float, K: float, dte: int, right: str,
                   r: float = 0.05,
                   t_years: float | None = None) -> tuple[float | None, float | None]:
    """(delta, iv) implied by the quote mid. (None, None) when it cannot be inverted.

    ``t_years`` (W76) is the real fractional time to the bell: a 0-DTE bar has hours
    left, not ``dte/365 = 0`` (which cannot be inverted at all, so every 0-DTE row
    lost its delta and was dropped). Without it the whole-day form is unchanged."""
    t = float(t_years) if t_years is not None else max(dte, 0) / 365.0
    iv = implied_vol_from_mid(mid, S, K, t, right, r)
    if iv is None:
        return None, None
    ot = OptionType.CALL if right == "C" else OptionType.PUT
    return float(bs_delta(S, K, t, r, iv, ot)), float(iv)


def nearest_listed(values, target: float):
    return min(values, key=lambda v: abs(v - target))


def plan_strikes(spot: float, iv: float, dte: int, listed_strikes, right: str,
                 buckets=SHORT_DELTA_BUCKETS, wings=WING_WIDTH_DOLLARS,
                 ) -> list[tuple[float, str]]:
    """Strikes to request for one (expiry, right): [(strike, role)], de-duplicated.

    Short strikes are the listed strikes nearest the BS strike of each |delta| bucket;
    wings are the listed strikes `wings` dollars beyond the 0.16-delta short strike.
    """
    t = max(dte, 1) / 365.0
    ot = OptionType.CALL if right == "C" else OptionType.PUT
    listed = sorted(float(k) for k in listed_strikes)
    if not listed:
        return []
    from ait.backtesting.pricing import find_strike_by_delta
    out: dict[float, str] = {}
    base_short = None
    for d in buckets:
        k = find_strike_by_delta(spot, t, iv, d if right == "C" else -d, ot, 0.05)
        k = nearest_listed(listed, k)
        out.setdefault(k, "short")
        if abs(d - 0.16) < 1e-9:
            base_short = k
    if base_short is not None:
        sign = 1.0 if right == "C" else -1.0
        for w in wings:
            k = nearest_listed(listed, base_short + sign * w)
            if k not in out:
                out[k] = "wing"
    return sorted(out.items())


def plan_short_dte_strikes(spot: float, iv: float, dte: int, listed_strikes, right: str,
                           ) -> list[tuple[float, str]]:
    """Strikes to request for one short-DTE (expiry, right): [(strike, role)], de-duplicated.

    Moneyness-anchored (``SHORT_DTE_MONEYNESS``, both sides of spot, role 'short' so the
    delta test applies) plus the compact delta buckets and dollar wings of the standard
    plan. Only LISTED strikes are returned.
    """
    listed = sorted(float(k) for k in listed_strikes)
    if not listed or spot <= 0:
        return []
    out: dict[float, str] = {}
    sign = 1.0 if right == "C" else -1.0
    for frac in SHORT_DTE_MONEYNESS:
        k = nearest_listed(listed, spot * (1.0 + sign * frac))
        out.setdefault(k, "short")
    for k, role in plan_strikes(spot, iv, dte, listed, right,
                                buckets=SHORT_DTE_DELTA_BUCKETS, wings=SHORT_DTE_WINGS):
        out.setdefault(k, role)
    return sorted(out.items())


def relevant_mask(df: pd.DataFrame, dte_range=(7, 45), min_mid=0.03,
                  max_abs_delta=0.45) -> pd.Series:
    """Samples the condor can actually trade: DTE 7-45, not penny-dust, not deep ITM.

    Wing rows (role == 'wing') bypass the delta test: far-OTM wings have |delta| ~0.02-0.06
    and ARE traded. Rows whose delta could not be inverted are kept only if they are wings.
    """
    d = df["delta"].abs()
    delta_ok = (d <= max_abs_delta) | (df["role"] == "wing")
    delta_known = d.notna() | (df["role"] == "wing")
    return ((df["dte"] >= dte_range[0]) & (df["dte"] <= dte_range[1])
            & (df["mid"] >= min_mid) & (df["half_spread_usd"] > 0)
            & delta_ok & delta_known)


def session_of(bar_time: str) -> str:
    return bar_time[:10]
