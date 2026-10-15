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
    SHORT_DTE_DELTA_BUCKETS,
    SHORT_DTE_WINGS,
    delta_from_mid,
    pair_bid_ask,
    plan_short_dte_strikes,
    plan_strikes,
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
# W82: every closed band of SHORT_BAND_EDGES (0-2, 3-6, 7-13, 14-29, 30-45). A 7+ band is only
# USED by the engine once it has the minimum sessions; before that the all-DTE model answers.
SHORT_BANDS = tuple(friction_band(hi, SHORT_BAND_EDGES) for hi in SHORT_BAND_EDGES)
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
    stage_a: int = 0               # W82: lifecycle expiries captured as stage A / B
    stage_b: int = 0
    deferred_b: list[str] = field(default_factory=list)    # W82: owed far expiries over the cap


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


async def _prepare(ib: Any, symbol: str, store: Any, duration: str, res: CollectionResult,
                   out: Callable[[str], None]):
    """Qualify the underlying, fetch its hourly path and the option chain parameters.

    Returns ``(uprice, iv0, expirations, strikes)`` or None (the reason is on ``out``)."""
    from ib_insync import Stock

    und = Stock(symbol, "SMART", "USD")
    q = await ib.qualifyContractsAsync(und)
    if not q:
        out(f"  [{symbol}] ERROR: could not qualify underlying")
        return None
    und = q[0]
    ubars = await ib.reqHistoricalDataAsync(
        und, endDateTime="", durationStr=duration, barSizeSetting="1 hour",
        whatToShow="TRADES", useRTH=True, formatDate=2)
    uprice = {bar_start_et(b.date).isoformat(): float(b.close) for b in ubars}
    if not uprice:
        out(f"  [{symbol}] ERROR: no underlying hourly bars")
        return None
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
    return uprice, iv0, expirations, strikes


async def _fetch_and_store(ib: Any, symbol: str, store: Any, window, plan, *, uprice: dict,
                           duration: str, pause_secs: float, sleep: Callable[[float], Any],
                           out: Callable[[str], None], now: datetime, report_max_dte: int,
                           res: CollectionResult) -> dict[date, int]:
    """Request BID and ASK bars for every planned contract, pair them and store the samples.

    ``plan`` is [(expiry, right, strike, role), ...]. Returns {expiry: contracts requested}
    so a caller can log what was captured. Shared by the 0-N DTE collector and the lifecycle
    collector: ONE request/pair/store path."""
    from ib_insync import Option

    from ait.data.historical import SOURCE_IB_BID_ASK_BARS

    requested_by_exp: dict[date, int] = {}
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
        requested_by_exp[exp] = requested_by_exp.get(exp, 0) + 1
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
            if 0 <= dte <= report_max_dte:
                res.by_dte[dte] = res.by_dte.get(dte, 0) + 1
        res.stored += store.save_spread_bar_samples(symbol, pd.DataFrame(rows),
                                                    source=SOURCE_IB_BID_ASK_BARS)
        if n % 10 == 0:
            out(f"    [{symbol}] {n}/{len(plan)} contracts, {res.stored} samples so far")
    return requested_by_exp


async def collect_short_dte(
    ib: Any, symbol: str, store: Any, window, *, max_dte: int = SHORT_DTE_MAX,
    duration: str = DEFAULT_DURATION, pause_secs: float = 10.8, dry_run: bool = False,
    today: date | None = None, sleep: Callable[[float], Any] = asyncio.sleep,
    out: Callable[[str], None] = print, now: datetime | None = None,
) -> CollectionResult:
    """Request, pair and store hourly BID/ASK samples for every contract <= ``max_dte`` out."""
    res = CollectionResult(symbol=symbol, dry_run=dry_run)
    today = today or today_et()
    now = now or datetime.now(ET)
    prepared = await _prepare(ib, symbol, store, duration, res, out)
    if prepared is None:
        return res
    uprice, iv0, expirations, strikes = prepared
    plan = plan_contracts(res.spot, iv0, today, expirations, strikes, max_dte)
    res.planned = len(plan)
    mins = len(plan) * 2 * pause_secs / 60.0
    by_exp = sorted({e for e, *_ in plan})
    out(f"  [{symbol}] spot={res.spot:.2f} expiries={[e.isoformat() for e in by_exp]} "
        f"contracts={len(plan)} (~{mins:.0f} min at {pause_secs:g}s pacing)")
    if dry_run or not plan:
        return res
    await _fetch_and_store(ib, symbol, store, window, plan, uprice=uprice, duration=duration,
                           pause_secs=pause_secs, sleep=sleep, out=out, now=now,
                           report_max_dte=max_dte, res=res)
    return res


# ---------------------------------------------------------------------------
# W82: lifecycle capture. A contract is perishable at its EXPIRY, not each day: IB serves 30 days
# of hourly BID/ASK bars for any listed contract in one request, so one capture on (or just
# before) its last day recovers its whole DTE 0-30 profile, and one capture while it is still
# 15-21 DTE reaches DTE 31-45. Two requests per contract's life instead of one per day.
# ---------------------------------------------------------------------------

STAGE_A = "A"            # perishable: expiring today / tomorrow
STAGE_B = "B"            # coverage debt: the DTE 31-45 tail of far-listed expiries
LOOKBACK_DAYS = 30       # IB serves 30 D of hourly BID/ASK bars for a listed option
FAR_WINDOW = (15, 21)    # stage B captures an expiry once, while it is this many DTE
FAR_TAIL_MIN_DTE = 31    # samples at or beyond this DTE are the "far tail"
FAR_TAIL_MIN_SESSIONS = 5
STRIKE_WINDOW = 0.12     # only strikes within +-12% of that day's spot are ever requested


def duration_days(duration: str) -> int:
    """'30 D' -> 30, '2 W' -> 14, '1 M' -> 30 (calendar days of IB's duration string)."""
    parts = str(duration).split()
    n = int(parts[0])
    unit = parts[1].upper() if len(parts) > 1 else "D"
    return n * {"D": 1, "W": 7, "M": 30, "Y": 365}.get(unit, 1)


def session_path(uprice: dict[str, float], ivs: pd.Series | None = None,
                 iv_default: float = 0.20) -> list[tuple[date, float, float]]:
    """[(session, spot, iv)]: the LAST hourly close of each session and that session's stored
    IV (the latest value on or before it; ``iv_default`` when none is plausible)."""
    by_day: dict[date, float] = {}
    for ts in sorted(uprice):
        by_day[date.fromisoformat(ts[:10])] = float(uprice[ts])
    ivmap: dict[date, float] = {}
    if ivs is not None and len(ivs):
        for idx, v in ivs.items():
            ivmap[idx.date() if hasattr(idx, "date") else idx] = float(v)
    ivdays = sorted(ivmap)
    out: list[tuple[date, float, float]] = []
    for d in sorted(by_day):
        prior = [x for x in ivdays if x <= d]
        v = ivmap[prior[-1]] if prior else iv_default
        out.append((d, by_day[d], v if 0.01 < v < 5.0 else iv_default))
    return out


def lifecycle_strikes(expiry: date, right: str, path: list[tuple[date, float, float]],
                      listed_strikes, *, today: date, thin_every: int = 3,
                      ) -> list[tuple[float, str]]:
    """[(strike, role)] to request for one (expiry, right): the UNION over the underlying's
    path of the strikes a condor would have sold that day.

    Every ``thin_every``-th session (the most recent first) applies the existing per-day rule
    with THAT day's spot, stored IV and DTE: moneyness anchors + deltas + wings at <= 6 DTE
    (``plan_short_dte_strikes``), the compact delta buckets + wings above it. A strike that is
    'short' on any day is 'short' (the delta test then applies to its rows; a wing label would
    bypass it)."""
    listed = sorted(float(k) for k in listed_strikes)
    days = [(d, S, iv) for d, S, iv in path if d <= min(today, expiry) and S > 0]
    chosen = days[::-1][::max(1, int(thin_every))]
    out: dict[float, str] = {}
    for d, S, iv in chosen:
        near = [k for k in listed if abs(k / S - 1.0) <= STRIKE_WINDOW]
        if not near:
            continue
        dte = (expiry - d).days
        if dte <= SHORT_DTE_MAX:
            plan = plan_short_dte_strikes(S, iv, dte, near, right)
        else:
            plan = plan_strikes(S, iv, dte, near, right, buckets=SHORT_DTE_DELTA_BUCKETS,
                                wings=SHORT_DTE_WINGS)
        for k, role in plan:
            # 'short' is sticky: a later wing label never downgrades it
            out[k] = "short" if role == "short" else out.get(k, role)
    return sorted(out.items())


@dataclass
class LifecyclePlan:
    contracts: list[tuple[date, str, float, str, str]] = field(default_factory=list)
    stage_a: list[date] = field(default_factory=list)       # expiries captured as stage A
    stage_b: list[date] = field(default_factory=list)       # expiries captured as stage B
    deferred_b: list[date] = field(default_factory=list)    # owed but over this run's cap


def far_tail_done(expiry: date, samples: pd.DataFrame | None, log: pd.DataFrame | None,
                  min_sessions: int = FAR_TAIL_MIN_SESSIONS) -> bool:
    """True when the expiry's DTE 31-45 tail is already stored (a logged stage-B capture, or
    enough stored sessions at DTE >= 31, e.g. from a manual 30 D catch-up run)."""
    iso = expiry.isoformat()
    if log is not None and len(log):
        m = (log["expiry"] == iso) & (log["stage"] == STAGE_B)
        if bool(m.any()):
            return True
    if samples is not None and len(samples):
        e = samples[(samples["expiry"] == iso) & (samples["dte"] >= FAR_TAIL_MIN_DTE)]
        if len(e) and int(e["bar_time"].map(session_of).nunique()) >= min_sessions:
            return True
    return False


def plan_lifecycle(path: list[tuple[date, float, float]], today: date, expirations: list[str],
                   strikes: list[float], *, samples: pd.DataFrame | None = None,
                   log: pd.DataFrame | None = None, expiring_days: int = 1, thin_every: int = 3,
                   max_far_contracts: int = 120,
                   far_window: tuple[int, int] = FAR_WINDOW) -> LifecyclePlan:
    """Stage A (expiring today..today+``expiring_days``) first, DTE 0 first, then stage B
    (far-listed expiries inside ``far_window`` whose tail is not stored), nearest first, up to
    ``max_far_contracts`` (the first expiry is always taken so a backlog drains)."""
    plan = LifecyclePlan()
    exps: list[date] = []
    for e in expirations:
        try:
            exps.append(datetime.strptime(e, "%Y%m%d").date())
        except ValueError:
            continue
    exps = sorted(set(exps))

    def contracts_for(exp: date, stage: str) -> list[tuple[date, str, float, str, str]]:
        out = []
        for right in ("C", "P"):
            for k, role in lifecycle_strikes(exp, right, path, strikes, today=today,
                                             thin_every=thin_every):
                out.append((exp, right, k, role, stage))
        return out

    for exp in exps:
        if 0 <= (exp - today).days <= int(expiring_days):
            plan.stage_a.append(exp)
            plan.contracts += contracts_for(exp, STAGE_A)
    lo, hi = far_window
    owed = [e for e in exps if lo <= (e - today).days <= hi
            and not far_tail_done(e, samples, log)]
    taken = 0
    for exp in owed:                       # nearest first (exps is sorted)
        cs = contracts_for(exp, STAGE_B)
        if taken and taken + len(cs) > int(max_far_contracts):
            plan.deferred_b.append(exp)
            continue
        plan.stage_b.append(exp)
        plan.contracts += cs
        taken += len(cs)
    # dedupe keeping the first (stage A wins over B for a shared contract)
    seen: set[tuple[date, str, float]] = set()
    uniq = []
    for c in plan.contracts:
        key = (c[0], c[1], c[2])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    plan.contracts = uniq
    return plan


async def collect_lifecycle(
    ib: Any, symbol: str, store: Any, window, *, duration: str = f"{LOOKBACK_DAYS} D",
    pause_secs: float = 10.8, dry_run: bool = False, today: date | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep, out: Callable[[str], None] = print,
    now: datetime | None = None, expiring_days: int = 1, thin_every: int = 3,
    max_far_contracts: int = 120,
) -> CollectionResult:
    """One lifecycle run: stage A then stage B (see the section comment)."""
    res = CollectionResult(symbol=symbol, dry_run=dry_run)
    today = today or today_et()
    now = now or datetime.now(ET)
    prepared = await _prepare(ib, symbol, store, duration, res, out)
    if prepared is None:
        return res
    uprice, iv0, expirations, strikes = prepared
    ivs = None
    try:
        ivs = store.load_daily_iv(symbol, days=duration_days(duration) + 10)
    except Exception:  # noqa: BLE001 - per-day IV is a planning hint; iv0 is the fallback
        pass
    path = session_path(uprice, ivs, iv0)
    samples = log = None
    try:
        samples = store.load_spread_bar_samples(symbol)
        log = store.load_capture_log(symbol)
    except Exception:  # noqa: BLE001 - no history means everything is owed, never a crash
        pass
    lp = plan_lifecycle(path, today, expirations, strikes, samples=samples, log=log,
                        expiring_days=expiring_days, thin_every=thin_every,
                        max_far_contracts=max_far_contracts)
    plan = [(e, r, k, role) for e, r, k, role, _stage in lp.contracts]
    stage_of = {e: STAGE_A for e in lp.stage_a} | {e: STAGE_B for e in lp.stage_b}
    res.planned = len(plan)
    res.stage_a, res.stage_b = len(lp.stage_a), len(lp.stage_b)
    res.deferred_b = [e.isoformat() for e in lp.deferred_b]
    mins = len(plan) * 2 * pause_secs / 60.0
    out(f"  [{symbol}] spot={res.spot:.2f} lookback={duration} stage A (expiring "
        f"<= {expiring_days}d)={[e.isoformat() for e in lp.stage_a]} stage B (far tail)="
        f"{[e.isoformat() for e in lp.stage_b]} deferred={res.deferred_b} "
        f"contracts={len(plan)} (~{mins:.0f} min at {pause_secs:g}s pacing)")
    if dry_run or not plan:
        return res
    done = await _fetch_and_store(ib, symbol, store, window, plan, uprice=uprice,
                                  duration=duration, pause_secs=pause_secs, sleep=sleep,
                                  out=out, now=now, report_max_dte=FAR_WINDOW[1] + LOOKBACK_DAYS,
                                  res=res)
    try:
        store.record_captures(symbol, [
            {"expiry": e.isoformat(), "stage": stage_of.get(e, STAGE_A),
             "captured_on": today.isoformat(), "dte_at_capture": (e - today).days,
             "lookback_days": duration_days(duration), "contracts": n}
            for e, n in sorted(done.items())])
    except Exception as e:  # noqa: BLE001 - the log is bookkeeping: never fail the run on it
        out(f"  [{symbol}] WARNING: capture log not written: {type(e).__name__}: {e}")
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
        # W82 lifecycle runs: expiries captured per stage and far expiries owed but deferred
        "stage_a": int(getattr(res, "stage_a", 0) or 0),
        "stage_b": int(getattr(res, "stage_b", 0) or 0),
        "deferred_b": list(getattr(res, "deferred_b", []) or []),
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
