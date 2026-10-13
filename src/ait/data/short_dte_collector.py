"""W76: collect and fit 0-6 DTE option bid/ask friction from IB hourly BID/ASK bars.

Why. The fitted per-leg friction model has no sample below 11 DTE, so every short-DTE leg
the research engine charges is an extrapolation. IB serves historical BID and ASK bars for
currently LISTED contracts only (nothing for expired ones, W66), so there is no way to
backfill: the data has to be collected forward, session by session, while the contracts
that are 0-6 DTE are still listed. This module is that collector plus the per-band fit.

    python scripts/calibrate_option_spreads.py --source ib-bars --short-dte \\
        --symbols QQQ --port 4002 --client-id 91          # once per RTH session

Design:
  * The broker handle is INJECTED (``ib``), so the whole plan -> request -> pair -> store
    pipeline runs against a fake in tests; nothing here connects to IB itself.
  * Each run requests the last ``duration`` (default 7 D) of hourly bars for every listed
    contract expiring within ``max_dte`` days. A contract is therefore re-sampled on each
    of its last ~7 days; ``INSERT OR IGNORE`` makes the overlap free.
  * Samples are keyed by DTE and session, so the run reports how many sessions each band
    has accumulated and how many it still needs.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ait.backtesting.friction import (
    SHORT_BAND_EDGES,
    band_dte_range,
    fit_leg_spread_models,
    friction_band,
)
from ait.data.option_spread_sampler import (
    SHORT_DTE_MAX,
    bar_start_et,
    delta_from_mid,
    pair_bid_ask,
    plan_short_dte_strikes,
    relevant_mask,
    session_of,
)
from ait.utils.time import ET, get_market_close, time_to_expiry_years, today_et

@lru_cache(maxsize=1024)
def _session_close(d: date):
    """The NYSE close for ``d`` (13:00 on early-close days), cached: one lookup per date,
    not per sample row (the calendar call costs ~10 ms)."""
    return get_market_close(d)


DEFAULT_DURATION = "7 D"
SHORT_BANDS = ("0-2", "3-6")      # the bands this module fits (7+ is the all-DTE model's)
MIN_MID_SHORT = 0.01              # a penny option is the tick; tighter than the 7-45 mask's 0.03


@dataclass
class CollectionResult:
    symbol: str
    spot: float | None = None
    planned: int = 0
    requested: int = 0
    unqualified: int = 0
    errors: int = 0
    stored: int = 0
    skipped_partial: int = 0       # bars whose hour had not ended when the run asked (never stored)
    by_dte: dict[int, int] = field(default_factory=dict)   # samples stored per DTE (<= max_dte)
    dry_run: bool = False


def plan_contracts(spot: float, iv: float, today: date, expirations: list[str], strikes: list[float],
                   max_dte: int = SHORT_DTE_MAX) -> list[tuple[date, str, float, str]]:
    """[(expiry, right, strike, role)] for every listed expiry 0..max_dte days out.

    Includes TODAY's expiry (the 0-DTE contract) and only strikes within +-12% of spot.
    """
    exps = []
    for e in expirations:
        try:
            d = datetime.strptime(e, "%Y%m%d").date()
        except ValueError:
            continue
        if 0 <= (d - today).days <= max_dte:
            exps.append(d)
    near = [k for k in strikes if spot > 0 and abs(k / spot - 1.0) <= 0.12]
    plan: list[tuple[date, str, float, str]] = []
    for exp in sorted(set(exps)):
        dte = (exp - today).days
        for right in ("C", "P"):
            for k, role in plan_short_dte_strikes(spot, iv, dte, near, right):
                plan.append((exp, right, k, role))
    return list(dict.fromkeys(plan))


async def collect_short_dte(
    ib: Any, symbol: str, store: Any, window, *, max_dte: int = SHORT_DTE_MAX,
    duration: str = DEFAULT_DURATION, pause_secs: float = 10.8, dry_run: bool = False,
    today: date | None = None, sleep: Callable[[float], Any] = asyncio.sleep,
    out: Callable[[str], None] = print, now: datetime | None = None,
) -> CollectionResult:
    """Request, pair and store hourly BID/ASK samples for every contract <= ``max_dte`` out."""
    from ib_insync import Option, Stock

    from ait.data.historical import SOURCE_IB_BID_ASK_BARS

    res = CollectionResult(symbol=symbol, dry_run=dry_run)
    today = today or today_et()
    now = now or datetime.now(ET)
    und = Stock(symbol, "SMART", "USD")
    q = await ib.qualifyContractsAsync(und)
    if not q:
        out(f"  [{symbol}] ERROR: could not qualify underlying")
        return res
    und = q[0]
    ubars = await ib.reqHistoricalDataAsync(
        und, endDateTime="", durationStr=duration, barSizeSetting="1 hour",
        whatToShow="TRADES", useRTH=True, formatDate=2)
    uprice = {bar_start_et(b.date).isoformat(): float(b.close) for b in ubars}
    if not uprice:
        out(f"  [{symbol}] ERROR: no underlying hourly bars")
        return res
    res.spot = float(ubars[-1].close)
    iv0 = 0.20
    try:
        ivs = store.load_daily_iv(symbol, days=30)
        if ivs is not None and not ivs.empty and 0.01 < float(ivs.iloc[-1]) < 5.0:
            iv0 = float(ivs.iloc[-1])
    except Exception:  # noqa: BLE001 - the default is only a strike-planning hint
        pass

    params = await ib.reqSecDefOptParamsAsync(und.symbol, "", und.secType, und.conId)
    chosen = [p for p in params or [] if p.exchange == "SMART"] or list(params or [])
    expirations = sorted({e for p in chosen for e in p.expirations})
    strikes = sorted({float(k) for p in chosen for k in p.strikes})
    plan = plan_contracts(res.spot, iv0, today, expirations, strikes, max_dte)
    res.planned = len(plan)
    mins = len(plan) * 2 * pause_secs / 60.0
    by_exp = sorted({e for e, *_ in plan})
    out(f"  [{symbol}] spot={res.spot:.2f} expiries={[e.isoformat() for e in by_exp]} "
        f"contracts={len(plan)} (~{mins:.0f} min at {pause_secs:g}s pacing)")
    if dry_run or not plan:
        return res

    for n, (exp, right, k, role) in enumerate(plan, 1):
        c = Option(symbol, exp.strftime("%Y%m%d"), k, right, "SMART")
        try:
            qc = await ib.qualifyContractsAsync(c)
            if not qc:
                res.unqualified += 1
                continue
            bars = {}
            for what in ("BID", "ASK"):
                bars[what] = await ib.reqHistoricalDataAsync(
                    qc[0], endDateTime="", durationStr=duration, barSizeSetting="1 hour",
                    whatToShow=what, useRTH=True, formatDate=2)
                res.requested += 1
                await sleep(pause_secs)
        except Exception as e:  # noqa: BLE001 - pacing violation etc.: back off, continue
            res.errors += 1
            out(f"    [{symbol}] {exp} {right}{k:g}: {type(e).__name__}: {e}")
            await sleep(60.0)
            continue
        paired = pair_bid_ask(bars["BID"], bars["ASK"], window)
        if paired.empty:
            continue
        rows = []
        for r in paired.to_dict("records"):
            u = uprice.get(r["bar_time"])
            start = datetime.fromisoformat(r["bar_time"])
            # W76b: a bar whose hour is still running holds a PARTIAL quote, and the idempotent
            # insert would keep that version forever. Store only bars that have ended (the last
            # bar of a session ends at the bell, not an hour after it starts).
            _end = min(start + timedelta(hours=1),
                       datetime.combine(start.date(), _session_close(start.date()), tzinfo=ET))
            if _end > now:
                res.skipped_partial += 1
                continue
            dte = (exp - start.date()).days
            delta = iv = None
            if u:
                # the quote is the bar's CLOSE: measure the time left from the bar's end
                t = time_to_expiry_years(start + timedelta(hours=1), exp, close=_session_close(exp))
                delta, iv = delta_from_mid((r["bid"] + r["ask"]) / 2.0, u, k, dte, right,
                                           t_years=t if t > 0 else None)
            rows.append({**r, "right": right, "strike": k, "expiry": exp.isoformat(),
                         "dte": dte, "underlying_price": u, "delta": delta, "iv": iv,
                         "role": role})
            if 0 <= dte <= max_dte:
                res.by_dte[dte] = res.by_dte.get(dte, 0) + 1
        res.stored += store.save_spread_bar_samples(symbol, pd.DataFrame(rows),
                                                    source=SOURCE_IB_BID_ASK_BARS)
        if n % 10 == 0:
            out(f"    [{symbol}] {n}/{len(plan)} contracts, {res.stored} samples so far")
    return res


# ---------------------------------------------------------------------------
# Coverage + per-band fit (pure over the stored samples)
# ---------------------------------------------------------------------------

def coverage(df: pd.DataFrame, max_dte: int = SHORT_DTE_MAX) -> dict[int, dict[str, int]]:
    """{dte: {"samples": n, "sessions": n}} for 0..max_dte from stored bar samples."""
    out: dict[int, dict[str, int]] = {}
    if df is None or df.empty:
        return {d: {"samples": 0, "sessions": 0} for d in range(max_dte + 1)}
    sess = df["bar_time"].map(session_of)
    for d in range(max_dte + 1):
        m = df["dte"] == d
        out[d] = {"samples": int(m.sum()), "sessions": int(sess[m].nunique())}
    return out


def band_sessions(df: pd.DataFrame, band: str, edges: tuple[int, ...] = SHORT_BAND_EDGES) -> int:
    """Distinct sessions holding at least one RELEVANT sample inside the band's DTE range."""
    if df is None or df.empty:
        return 0
    lo, hi = band_dte_range(band, edges)
    rel = df[relevant_mask(df, dte_range=(lo, hi), min_mid=MIN_MID_SHORT)]
    return int(rel["bar_time"].map(session_of).nunique()) if len(rel) else 0


def fit_short_dte_bands(symbol: str, store: Any, *, min_sessions: int,
                        bands: tuple[str, ...] = SHORT_BANDS,
                        edges: tuple[int, ...] = SHORT_BAND_EDGES,
                        out: Callable[[str], None] = print) -> dict[str, dict]:
    """Fit and store one friction model per short-DTE band; returns {band: row}.

    A band is stored whenever it can be fit, but the ENGINE only uses it once
    ``n_sessions >= min_sessions``; below that this reports how many are still needed.
    """
    from ait.data.historical import SOURCE_IB_BID_ASK_BARS

    df = store.load_spread_bar_samples(symbol)
    saved: dict[str, dict] = {}
    if df is None or df.empty:
        out(f"  [{symbol}] no bar samples stored")
        return saved
    for band in bands:
        lo, hi = band_dte_range(band, edges)
        rel = df[relevant_mask(df, dte_range=(lo, hi), min_mid=MIN_MID_SHORT)]
        n_sess = int(rel["bar_time"].map(session_of).nunique()) if len(rel) else 0
        if len(rel) < 5:
            out(f"  [{symbol}] band {band} (DTE {lo}-{hi}): {len(rel)} relevant samples, "
                f"{n_sess} sessions - need data ({min_sessions} sessions to activate)")
            continue
        try:
            fit = fit_leg_spread_models(rel["mid"], rel["half_spread_usd"],
                                        rel["bar_time"].map(session_of))
        except ValueError as e:
            out(f"  [{symbol}] band {band}: {e}")
            continue
        best, cv = fit["best"], fit["cv_mae"]
        row = {**best.to_row(), "n": fit["n"], "n_sessions": fit["n_sessions"],
               "cv_mae": cv[best.form], "cv_mae_by_form": cv, "source": SOURCE_IB_BID_ASK_BARS,
               "calibrated_on": today_et().isoformat()}
        store.save_leg_spread_band_model(symbol, band, row)
        saved[band] = row
        status = ("ACTIVE" if fit["n_sessions"] >= min_sessions else
                  f"collecting: {fit['n_sessions']}/{min_sessions} sessions "
                  f"({min_sessions - fit['n_sessions']} more needed)")
        out(f"  [{symbol}] band {band} (DTE {lo}-{hi}): {best.form} a={best.a:.4f} b={best.b:.4f} "
            f"n={fit['n']} sessions={fit['n_sessions']} cvMAE=${cv[best.form]:.4f}  -> {status}")
    return saved


def band_of(dte: int, edges: tuple[int, ...] = SHORT_BAND_EDGES) -> str:
    """Re-exported for the script's reporting (single definition lives in friction)."""
    return friction_band(dte, edges)


# ---------------------------------------------------------------------------
# Status file: the one place progress can be read at a glance (manual or scheduled runs)
# ---------------------------------------------------------------------------

STATUS_SCHEMA = "ait.short_dte_collection/1"


def symbol_status(res: CollectionResult | None, cov: dict[int, dict[str, int]],
                  band_rows: dict[str, dict], sessions_by_band: dict[str, int],
                  min_sessions: int) -> dict:
    """One symbol's block: this run's counters, stored coverage, and band progress.

    ``band_rows`` are the models fit in this run; ``sessions_by_band`` is the stored
    session count per band (also for bands that could not be fit yet)."""
    bands = {}
    for band, n in sessions_by_band.items():
        row = band_rows.get(band) or {}
        bands[band] = {"sessions": int(n), "needed": int(min_sessions),
                       "active": bool(n >= min_sessions and row), "fitted": bool(row),
                       "form": row.get("form"), "n": row.get("n")}
    return {
        "planned": getattr(res, "planned", 0), "requested": getattr(res, "requested", 0),
        "stored": getattr(res, "stored", 0), "unqualified": getattr(res, "unqualified", 0),
        "errors": getattr(res, "errors", 0), "spot": getattr(res, "spot", None),
        "dry_run": bool(getattr(res, "dry_run", False)),
        "coverage": {str(d): v for d, v in cov.items()}, "bands": bands,
    }


def write_status(path, *, ok: bool, error: str | None, symbols: dict[str, dict],
                 min_sessions: int, duration_s: float, short_dte_max: int) -> dict:
    """Atomically write the status file and return what was written."""
    import json
    import os
    from pathlib import Path

    status = {"schema": STATUS_SCHEMA, "written_at": datetime.now(ET).isoformat(timespec="seconds"),
              "ok": bool(ok), "error": error, "duration_s": round(float(duration_s), 1),
              "short_dte_max": int(short_dte_max), "min_sessions": int(min_sessions),
              "symbols": symbols}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, p)
    return status
