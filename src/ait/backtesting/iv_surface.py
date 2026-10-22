"""W75: an implied-vol surface fitted from REAL option bars (research only).

Why. The research engine priced every leg off ONE stored constant-maturity IV
(IB's 30-day figure) plus a skew that has no DTE term. The W72 replay measured
the market's leg IVs 2-4 vol points BELOW that at 3-45 DTE and 1-5 above it at
0-2 DTE, and the sign flip is what a single-maturity IV produces. The repo
already holds ~900k real QQQ daily option bars; this module turns them into a
per-(date, expiry) smile so research can price a leg off the expiry it actually
has, at the strike it actually has.

Construction (all pure, read-only):
  * rows: solved IV, volume >= ``min_volume``, price >= ``min_price`` (tick noise),
    OUT-of-the-money side only (ITM IVs are intrinsic-dominated and American-
    exercise noisy), |ln(K/S)| <= ``max_abs_m``, and **DTE >= 1**. DTE-0 bars are
    excluded on purpose: their IV was solved from the 16:00 close with t floored
    at 0.001 y when ~0 time remains, which is meaningless. A 0-DTE contract is
    priced from the PREVIOUS session's 1-DTE smile of the same expiry, which is
    exactly what was knowable that morning.
  * fit per (date, expiry): volume-weighted quadratic ``iv(m) = a + b*m + c*m^2``
    in log-moneyness ``m = ln(K/S)`` (sticky-moneyness).
  * lookup at an as-of date: the expiry's own smile (listed daily expiries make
    this the common case); otherwise interpolate across the nearest fitted
    expiries linearly in TOTAL VARIANCE, with flat IV extrapolation.

No look-ahead: ``iv(as_of=D-1, ...)`` uses only bars up to D-1 (assumption A3).
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ait.config.settings import DEFAULT_RISK_FREE_RATE  # W88b: one schema default for r

MIN_VOLUME = 10
MIN_PRICE = 0.05
MAX_ABS_M = 0.10
MIN_POINTS = 5
MIN_SPAN = 0.01          # the fit must cover at least 1% of log-moneyness
IV_BOUNDS = (0.03, 2.5)
CLAMP_MARGIN = 0.01      # evaluate at most this far outside the fitted range
MAX_STALE_DAYS = 5       # calendar days a surface date may lag the query date


@dataclass(frozen=True)
class SmileFit:
    as_of: date
    expiry: date
    a: float
    b: float
    c: float
    n: int
    m_lo: float
    m_hi: float
    rmse: float
    r: float                    # the rate the bar IVs were solved with

    def iv_at(self, m: float) -> float:
        """Smile IV at log-moneyness ``m``, clamped to just outside the fitted range."""
        mm = min(max(m, self.m_lo - CLAMP_MARGIN), self.m_hi + CLAMP_MARGIN)
        return self.a + self.b * mm + self.c * mm * mm

    @property
    def atm_iv(self) -> float:
        return self.iv_at(0.0)


def fit_smile(m: np.ndarray, iv: np.ndarray, w: np.ndarray, *, min_points: int = MIN_POINTS,
              min_span: float = MIN_SPAN) -> tuple[float, float, float, float] | None:
    """Weighted quadratic fit; None when the points cannot support one."""
    if len(m) < min_points or float(m.max() - m.min()) < min_span:
        return None
    sw = np.sqrt(np.asarray(w, dtype=float))
    X = np.column_stack([np.ones_like(m), m, m * m])
    try:
        coef, *_ = np.linalg.lstsq(X * sw[:, None], iv * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = X @ coef
    rmse = float(np.sqrt(np.average((pred - iv) ** 2, weights=w)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rmse


def usable_rows(bars: pd.DataFrame, *, min_volume: int = MIN_VOLUME, min_price: float = MIN_PRICE,
                max_abs_m: float = MAX_ABS_M) -> pd.DataFrame:
    """Filter raw option_daily_bars rows down to what a smile should be fit on."""
    b = bars
    ok = (b["iv_outcome"] == "solved") & (b["volume"] >= min_volume) & (b["close"] >= min_price) \
        & (b["underlying_close"] > 0) & (b["dte"] >= 1) \
        & b["iv"].between(IV_BOUNDS[0], IV_BOUNDS[1])
    b = b[ok].copy()
    b["m"] = np.log(b["strike"].astype(float) / b["underlying_close"].astype(float))
    otm = ((b["right"] == "P") & (b["m"] <= 0.0)) | ((b["right"] == "C") & (b["m"] >= 0.0))
    b = b[otm & (b["m"].abs() <= max_abs_m)]
    return b


class IVSurface:
    """Per-(date, expiry) smiles with sticky-moneyness lookup."""

    def __init__(self, fits: Iterable[SmileFit]) -> None:
        self._fits: dict[tuple[date, date], SmileFit] = {}
        self._by_date: dict[date, list[date]] = {}
        for f in fits:
            self._fits[(f.as_of, f.expiry)] = f
            self._by_date.setdefault(f.as_of, []).append(f.expiry)
        for d in self._by_date:
            self._by_date[d].sort()
        self._dates = sorted(self._by_date)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_bars(cls, bars: pd.DataFrame, **kw) -> IVSurface:
        rows = usable_rows(bars, **{k: v for k, v in kw.items()
                                    if k in ("min_volume", "min_price", "max_abs_m")})
        fits: list[SmileFit] = []
        if rows.empty:
            return cls(fits)
        rate_col = "risk_free_rate" if "risk_free_rate" in rows.columns else None
        for (d, e), g in rows.groupby(["date", "expiry"], sort=False):
            m = g["m"].to_numpy(dtype=float)
            iv = g["iv"].to_numpy(dtype=float)
            w = np.sqrt(g["volume"].to_numpy(dtype=float))
            got = fit_smile(m, iv, w, min_points=kw.get("min_points", MIN_POINTS),
                            min_span=kw.get("min_span", MIN_SPAN))
            if got is None:
                continue
            a, b, c, rmse = got
            r = float(g[rate_col].median()) if rate_col else DEFAULT_RISK_FREE_RATE
            fits.append(SmileFit(_as_date(d), _as_date(e), a, b, c, int(len(g)),
                                 float(m.min()), float(m.max()), rmse, r))
        return cls(fits)

    # ---- inspection ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._fits)

    def dates(self) -> list[date]:
        return list(self._dates)

    def expiries(self, as_of: date) -> list[date]:
        return list(self._by_date.get(as_of, ()))

    def smile(self, as_of: date, expiry: date) -> SmileFit | None:
        return self._fits.get((as_of, expiry))

    def latest_on_or_before(self, d: date, max_stale_days: int = MAX_STALE_DAYS) -> date | None:
        """The newest surface date <= ``d`` within ``max_stale_days`` calendar days."""
        lo, hi = 0, len(self._dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._dates[mid] <= d:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        cand = self._dates[lo - 1]
        return cand if (d - cand).days <= max_stale_days else None

    # ---- lookup -------------------------------------------------------
    def iv(self, as_of: date, expiry: date, strike: float, spot: float) -> float | None:
        """Surface IV for ``strike`` of ``expiry`` using the smile known at ``as_of``.

        ``as_of`` is the surface DATE (the bar date whose close fixed it); callers
        pass the PREVIOUS session for a decision on day D. None when no smile can
        be formed (the caller falls back and counts it).
        """
        if spot <= 0 or strike <= 0:
            return None
        d = self.latest_on_or_before(as_of)
        if d is None:
            return None
        m = math.log(strike / spot)
        own = self._fits.get((d, expiry))
        if own is not None:
            return own.iv_at(m)
        exps = self._by_date.get(d, [])
        if not exps:
            return None
        before = [e for e in exps if e < expiry]
        after = [e for e in exps if e > expiry]
        t_q = max((expiry - d).days, 1) / 365.0
        lo = self._fits[(d, before[-1])] if before else None
        hi = self._fits[(d, after[0])] if after else None
        if lo is not None and hi is not None:
            t_lo = max((lo.expiry - d).days, 1) / 365.0
            t_hi = max((hi.expiry - d).days, 1) / 365.0
            w_lo = lo.iv_at(m) ** 2 * t_lo
            w_hi = hi.iv_at(m) ** 2 * t_hi
            frac = (t_q - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
            var = (w_lo + frac * (w_hi - w_lo)) / t_q
            return math.sqrt(max(var, 1e-8))
        edge = lo or hi
        return None if edge is None else edge.iv_at(m)

    # ---- persistence --------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "as_of": f.as_of.isoformat(), "expiry": f.expiry.isoformat(), "a": f.a, "b": f.b,
            "c": f.c, "n": f.n, "m_lo": f.m_lo, "m_hi": f.m_hi, "rmse": f.rmse, "r": f.r,
        } for f in self._fits.values()])

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> IVSurface:
        return cls(SmileFit(date.fromisoformat(r.as_of), date.fromisoformat(r.expiry),
                            float(r.a), float(r.b), float(r.c), int(r.n), float(r.m_lo),
                            float(r.m_hi), float(r.rmse), float(r.r))
                   for r in df.itertuples(index=False))


def _as_date(x) -> date:
    return x if isinstance(x, date) and not isinstance(x, pd.Timestamp) else pd.Timestamp(x).date()


# ---------------------------------------------------------------------------
# Read-only loading + cache
# ---------------------------------------------------------------------------

BAR_COLUMNS = ("date", "expiry", "right", "strike", "dte", "close", "volume",
               "underlying_close", "iv", "iv_outcome", "risk_free_rate")


def load_bars(db_path: str | Path, underlying: str, *, date_from: str | None = None,
              date_to: str | None = None, table: str = "option_daily_bars") -> pd.DataFrame:
    """Rows of the bars table, read through a read-only connection.

    Never opens the store through ``HistoricalDataStore`` (which opens for write
    and may migrate); the backfill may be appending to this table concurrently,
    which a read-only snapshot tolerates.
    """
    sql = f"SELECT {', '.join(BAR_COLUMNS)} FROM {table} WHERE underlying = ?"
    args: list = [underlying]
    if date_from:
        sql += " AND date >= ?"
        args.append(date_from)
    if date_to:
        sql += " AND date <= ?"
        args.append(date_to)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        df = pd.read_sql_query(sql, c, params=args)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


def build_or_load(db_path: str | Path, underlying: str, cache_dir: str | Path,
                  **kw) -> IVSurface:
    """Surface from the cache when it is current, else rebuilt and cached.

    The cache is stamped with the bars' row count and newest date, so it rebuilds
    exactly when the backfill has added rows. Cache lives under ``data/`` (git-ignored).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iv_surface_{underlying}"
    meta_p, data_p = cache_dir / f"{stem}.json", cache_dir / f"{stem}.parquet"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        n, last = c.execute(
            "SELECT COUNT(*), MAX(date) FROM option_daily_bars WHERE underlying = ?",
            (underlying,)).fetchone()
    stamp = {"rows": int(n), "max_date": str(last), "params": {k: kw[k] for k in sorted(kw)}}
    if meta_p.exists() and data_p.exists():
        try:
            if json.loads(meta_p.read_text()) == stamp:
                return IVSurface.from_frame(pd.read_parquet(data_p))
        except Exception:  # noqa: BLE001 - a corrupt cache is just rebuilt
            pass
    surf = IVSurface.from_bars(load_bars(db_path, underlying), **kw)
    surf.to_frame().to_parquet(data_p)
    meta_p.write_text(json.dumps(stamp))
    return surf
