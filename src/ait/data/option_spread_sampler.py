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


def implied_vol_from_mid(mid: float, S: float, K: float, t: float, right: str,
                         r: float = 0.05) -> float | None:
    """Bisection IV; None when mid is below intrinsic or above S (no solution)."""
    ot = OptionType.CALL if right == "C" else OptionType.PUT
    if mid <= 0 or S <= 0 or K <= 0 or t <= 0:
        return None
    lo, hi = 0.01, 4.0
    if black_scholes_price(S, K, t, r, lo, ot) > mid or black_scholes_price(S, K, t, r, hi, ot) < mid:
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if black_scholes_price(S, K, t, r, m, ot) < mid:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def delta_from_mid(mid: float, S: float, K: float, dte: int, right: str,
                   r: float = 0.05) -> tuple[float | None, float | None]:
    """(delta, iv) implied by the quote mid. (None, None) when it cannot be inverted."""
    t = max(dte, 0) / 365.0
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
