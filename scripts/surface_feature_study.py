"""W118: do IV-surface features predict the out-of-sample result of a QQQ iron condor, by DTE bucket? (walk-forward)

PRE-REGISTERED in PLAN.md "W118 AS PLANNED" (2026-11-01), before this script existed or any run. The constants
below are a copy; the PLAN entry is the authority and is not edited after any result.

One synthetic condor per (session D, target DTE), entered at D's close and held to expiry, scored from REAL option-bar
closes and the expiry-day underlying close (no pricing model, no surface in the label). Features come from the D-1
smile and data dated up to D-1. Short strikes at |delta| 0.20 on the D-1 smile; wing width max($1, 0.35 x expected
move). Friction from the stored per-leg band models; commissions from the loaded settings.

Primary features (the only ones the decision rule tests): ``vrp_h`` (ATM IV minus a HAR forecast of volatility over
the same horizon), ``term_slope`` (ATM IV 7 days minus 30 days), ``rr25_norm`` (normalised 25-delta risk reversal,
NaN outside the smile's fitted strikes). Controls: IV rank and the entry expiry's ATM IV.

Anchored walk-forward: warm-up 2024-10-01..2025-03-31 (fit only), six quarterly test folds to 2026-10-20, a training
row counts only if its EXPIRY is on or before the session 5 sessions before the fold's first test date.

Usage: python scripts/surface_feature_study.py --dry-run [--out <dir>]
       (coverage, misses, out-of-range share, friction fitted vs extrapolated; computes NO outcomes)
Statistics only: no per-row prices or P&L are written.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sqlite3  # noqa: E402
from bisect import bisect_right  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_BASE = ROOT / "reports" / "surface_feature_study"
STUDY_FILE = ROOT / "studies" / "surface_feature_study.yaml"
SYMBOL = "QQQ"

# ---- pre-registered constants (PLAN.md "W118 AS PLANNED") ------------------------------------------------------
WINDOW_START, WINDOW_END = date(2024, 10, 1), date(2026, 10, 20)
TEST_START = date(2025, 4, 1)
FOLDS = ((date(2025, 4, 1), date(2025, 6, 30)), (date(2025, 7, 1), date(2025, 9, 30)),
         (date(2025, 10, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 3, 31)),
         (date(2026, 4, 1), date(2026, 6, 30)), (date(2026, 7, 1), WINDOW_END))
GAP_SESSIONS = 5
TARGET_DTES = (1, 2, 5, 7, 10, 14, 18, 25, 30)            # ADDENDUM 1: the 45-day target is dropped
BUCKETS = {"A": (1, 2), "B": (3, 10), "C": (14, 30)}      # by TARGET DTE, calendar days (ADDENDUM 1: C was 14-45)
MIN_N_EFF = 30
# Not-formed reasons that are a fact about the listed-expiry calendar, not a data defect: reported, but outside the
# 25% VOID share (ADDENDUM 1, amendment 3).
STRUCTURAL_REASONS = ("no_expiry", "collapsed_duplicate_expiry")
EXPIRY_TOLERANCE_DAYS = 2
SHORT_DELTA = 0.20
WING_EM_FRACTION = 0.35
MIN_WING_USD = 1.0
MAX_SMILE_RMSE = 0.03
RR_DELTA = 0.25
MAX_MISSING_SHARE = 0.25
MAX_OUT_OF_RANGE_SHARE = 0.25
MIN_HAR_ROWS = 250
TERM_SHORT_DAYS, TERM_LONG_DAYS = 7, 30


def bucket_of(target: int) -> str:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= target <= hi:
            return name
    raise ValueError(f"target DTE {target} is in no bucket")


def fold_of(d: date) -> int | None:
    """Index of the test fold holding session ``d``; None for the warm-up."""
    for i, (lo, hi) in enumerate(FOLDS):
        if lo <= d <= hi:
            return i
    return None


# --------------------------------------------------------------------------- inputs

@dataclass
class Inputs:
    surface: object                       # IVSurface
    sessions: list[date]                  # NYSE sessions with bars, WINDOW_START-1 .. WINDOW_END
    spot: dict[date, float]               # underlying_close per bar date
    legs: dict[tuple[date, date, str], tuple[np.ndarray, np.ndarray]]   # (date, expiry, right) -> (strikes, closes), usable only
    friction: object                      # BandedFriction
    friction_report: dict
    bt: object                            # settings.backtest
    tuning: object                        # SurfaceTuningConfig
    frame: pd.DataFrame                   # daily OHLC + implied_vol
    iv_rank: pd.Series
    nyse: list[date]                      # NYSE sessions covering the window plus 60 days


def load_friction(db_path: Path, bt) -> tuple[object, dict]:
    """The engine's BandedFriction for QQQ, read through a read-only connection (same rule as Backtester)."""
    from ait.backtesting.friction import BandedFriction, LegSpreadModel
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT form, a, b FROM option_leg_spread_params WHERE symbol = ?", (SYMBOL,)).fetchone()
        bands = {r["dte_band"]: dict(r) for r in
                 c.execute("SELECT * FROM option_leg_spread_band_params WHERE symbol = ?", (SYMBOL,))}
    if row is None:
        raise SystemExit(f"per-leg friction model for {SYMBOL} is missing: python scripts/calibrate_option_spreads.py ...")
    usable = {b: LegSpreadModel.from_row(r) for b, r in bands.items()
              if int(r["n_sessions"]) >= bt.friction_min_sessions_per_band}
    report = {"default_form": row["form"], "min_sessions_per_band": int(bt.friction_min_sessions_per_band),
              "bands": {b: {"form": r["form"], "n_sessions": int(r["n_sessions"]), "used": b in usable}
                        for b, r in sorted(bands.items())}}
    return BandedFriction(LegSpreadModel.from_row(dict(row)), usable, bt.friction_extrapolated_floor_usd), report


def load_inputs(db_path: Path) -> Inputs:
    from ait.backtesting.iv_surface import build_or_load, load_bars
    from ait.backtesting.replay import ResearchData
    from ait.config.settings import load_settings, loaded_surface_tuning
    from ait.data.iv_rank import iv_rank_series
    from ait.data.research_inputs import sessions_between

    settings = load_settings()
    bt, tuning = settings.backtest, loaded_surface_tuning()
    surface = build_or_load(db_path, SYMBOL, db_path.parent / "research_cache")
    bars = load_bars(db_path, SYMBOL, date_to=WINDOW_END.isoformat())
    sessions = sorted(d for d in bars["date"].unique())
    spot = {d: float(v) for d, v in bars.groupby("date")["underlying_close"].first().items()}

    entry = bars[(bars["date"] >= WINDOW_START) & (bars["expiry"] > bars["date"]) & (bars["expiry"] <= WINDOW_END)
                 & (bars["close"] >= tuning.min_price) & (bars["volume"] >= tuning.min_volume)]
    legs = {}
    for (d, e, r), g in entry.groupby(["date", "expiry", "right"], sort=False):
        g = g.sort_values("strike")
        legs[(d, e, r)] = (g["strike"].to_numpy(dtype=float), g["close"].to_numpy(dtype=float))

    friction, friction_report = load_friction(db_path, bt)
    frame = ResearchData(str(db_path), SYMBOL).daily_frame()
    ranks = iv_rank_series(frame["implied_vol"])
    nyse = sessions_between(WINDOW_START - timedelta(days=10), WINDOW_END + timedelta(days=90))
    return Inputs(surface, sessions, spot, legs, friction, friction_report, bt, tuning, frame, ranks, list(nyse))


# --------------------------------------------------------------------------- HAR volatility forecast (as of D-1)

class HarForecaster:
    """Expanding-window HAR forecast of volatility over the next ``h`` sessions, refitted per session on rows whose
    whole target window ended on or before D-1 (``ait.ml.excursion``: log mean GK variance at 1/5/22 sessions)."""

    def __init__(self, frame: pd.DataFrame) -> None:
        from ait.ml.excursion import garman_klass, har_lags
        self._index = {ts.date(): i for i, ts in enumerate(frame.index)}
        self._gk = garman_klass(frame)
        self._X = har_lags(self._gk).to_numpy(dtype=float)
        self._y: dict[int, np.ndarray] = {}
        self._cache: dict[tuple[date, int], float] = {}

    def annual_vol(self, d: date, h: int) -> float:
        """Forecast annualised volatility over sessions D..D+h-1 using information up to D-1; NaN when it cannot be fit."""
        key = (d, h)
        if key in self._cache:
            return self._cache[key]
        out = float("nan")
        p = self._index.get(d)
        if p is not None and h >= 1:
            from ait.ml.excursion import har_fit, har_predict_sigma, har_target
            if h not in self._y:
                self._y[h] = har_target(self._gk, h).to_numpy(dtype=float)
            hi = p - h + 1                                   # rows t <= p-h have their target window ended by D-1
            X, y = self._X[:max(hi, 0)], self._y[h][:max(hi, 0)]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if ok.sum() >= MIN_HAR_ROWS and np.isfinite(self._X[p]).all():
                beta, smear = har_fit(y[ok], X[ok])
                sigma_h = float(har_predict_sigma(beta, smear, self._X[p:p + 1])[0])
                out = sigma_h / math.sqrt(h / 252.0)
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------- panel

def _nearest(arr: np.ndarray, target: float, mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return int(idx[np.argmin(np.abs(arr[idx] - target))])


def _surface_credit(surf, Dm: date, E: date, S: float, t: float, r: float, strikes) -> float:
    """What the D-1 smile says the four chosen legs are worth (short put - long put + short call - long call), for the
    wiring check 'does a feature predict the surface's pricing error'. NaN if any leg cannot be priced."""
    from ait.backtesting.pricing import OptionType, black_scholes_price
    kinds = (OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL)
    vals = []
    for k, kind in zip(strikes, kinds):
        iv = surf.iv(Dm, E, float(k), S)
        if iv is None:
            return float("nan")
        vals.append(black_scholes_price(S, float(k), t, r, iv, kind))
    return float(vals[0] - vals[1] + vals[2] - vals[3])


FEATURE_COLUMNS = ("vrp_h", "term_slope", "rr25_norm", "bf25", "fwd_var", "iv_rank")


def compute_features(inp: Inputs, har: HarForecaster, term_cache: dict, D: date, Dm: date, E: date, smile,
                     atm: float, hold: int) -> dict:
    """Every feature of one (D, expiry) entry. Reads the smile and data dated up to Dm = D-1 only (the HAR fit ends at
    D-1 by construction); the same function is used by the as-of mutation check."""
    from ait.backtesting.surface_features import bf25, forward_variance, rr25_norm, term_slope, years
    surf, S_m = inp.surface, inp.spot[Dm]
    if Dm not in term_cache:
        term_cache[Dm] = term_slope(surf, Dm, S_m, TERM_SHORT_DAYS, TERM_LONG_DAYS)
    later = [e for e in surf.expiries(Dm) if e > E]
    far = surf.smile(Dm, min(later)) if later else None
    rv = har.annual_vol(D, max(hold, 1))
    ivr = inp.iv_rank.get(pd.Timestamp(Dm), float("nan"))
    t_m = years(Dm, E)
    return dict(
        vrp_h=(atm - rv) if not math.isnan(rv) else float("nan"),
        term_slope=term_cache[Dm], rr25_norm=rr25_norm(smile, S_m, t_m, RR_DELTA),
        bf25=bf25(smile, S_m, t_m, RR_DELTA), fwd_var=forward_variance(smile, far) if far else float("nan"),
        iv_rank=float(ivr) if ivr == ivr else float("nan"))


def build_entries(inp: Inputs, *, start: date = WINDOW_START,
                  end: date = WINDOW_END) -> tuple[pd.DataFrame, Counter, Counter]:
    """One row per formed (D, expiry) entry, a Counter of (bucket, reason) for every one that was not, and a Counter
    of (target, "attempts" | "no_expiry") so the calendar gaps can be reported per target.

    Computes entry structure, entry cost and features ONLY. No expiry-day value is read here, so the dry run cannot
    leak an outcome into any decision about the design."""
    from ait.backtesting.friction import banded_condor_leg_cost_usd
    from ait.backtesting.surface_features import smile_atm_iv, smile_is_usable, strike_for_delta, years

    surf, tuning, bt = inp.surface, inp.tuning, inp.bt
    sess_idx = {d: i for i, d in enumerate(inp.sessions)}
    nyse = inp.nyse
    har = HarForecaster(inp.frame)
    commission = 4 * bt.commission_per_contract / 100.0           # $/share for four legs, one way
    rows: list[dict] = []
    miss: Counter = Counter()
    per_target: Counter = Counter()
    term_cache: dict[date, float] = {}

    for D in inp.sessions:
        if not start <= D <= end or D not in sess_idx or sess_idx[D] == 0:
            continue
        Dm = inp.sessions[sess_idx[D] - 1]                         # the as-of session for every feature
        S_D, S_m = inp.spot.get(D), inp.spot.get(Dm)
        if S_D is None or S_m is None:
            continue
        expiries = [e for e in surf.expiries(Dm) if D < e <= end]
        taken: set[date] = set()
        for target in TARGET_DTES:
            bucket = bucket_of(target)
            per_target[(target, "attempts")] += 1
            if not expiries:
                miss[(bucket, "no_expiry")] += 1
                per_target[(target, "no_expiry")] += 1
                continue
            E = min(expiries, key=lambda e: (abs((e - D).days - target), e))
            if abs((E - D).days - target) > EXPIRY_TOLERANCE_DAYS:
                miss[(bucket, "no_expiry")] += 1
                per_target[(target, "no_expiry")] += 1
                continue
            if E in taken:                                         # two targets share a listed expiry: one row
                miss[(bucket, "collapsed_duplicate_expiry")] += 1
                continue
            taken.add(E)
            smile = surf.smile(Dm, E)
            if not smile_is_usable(smile, min_points=tuning.min_points, max_rmse=MAX_SMILE_RMSE):
                miss[(bucket, "smile_unusable")] += 1
                continue
            atm = smile_atm_iv(smile)
            if math.isnan(atm):
                miss[(bucket, "atm_outside_fitted_range")] += 1
                continue
            t_D, dte = years(D, E), (E - D).days
            em = S_D * atm * math.sqrt(t_D)
            width_target = max(MIN_WING_USD, round(WING_EM_FRACTION * em))
            sp, sc = (strike_for_delta(smile, S_D, t_D, SHORT_DELTA, r) for r in ("P", "C"))
            if sp is None or sc is None:
                miss[(bucket, "short_delta_outside_fitted_range")] += 1
                continue
            put_k, put_c = inp.legs.get((D, E, "P"), (np.empty(0), np.empty(0)))
            call_k, call_c = inp.legs.get((D, E, "C"), (np.empty(0), np.empty(0)))
            i_sp = _nearest(put_k, sp[0], put_k < S_D)
            i_sc = _nearest(call_k, sc[0], call_k > S_D)
            if i_sp is None or i_sc is None:
                miss[(bucket, "short_leg_has_no_usable_bar")] += 1
                continue
            i_lp = _nearest(put_k, put_k[i_sp] - width_target, put_k < put_k[i_sp])
            i_lc = _nearest(call_k, call_k[i_sc] + width_target, call_k > call_k[i_sc])
            if i_lp is None or i_lc is None:
                miss[(bucket, "long_leg_has_no_usable_bar")] += 1
                continue
            put_credit, call_credit = put_c[i_sp] - put_c[i_lp], call_c[i_sc] - call_c[i_lc]
            if put_credit <= 0 or call_credit <= 0:
                miss[(bucket, "nonpositive_side_credit")] += 1
                continue
            credit = float(put_credit + call_credit)
            width = float(max(put_k[i_sp] - put_k[i_lp], call_k[i_lc] - call_k[i_sc]))
            if credit >= width:
                miss[(bucket, "credit_not_below_width")] += 1
                continue
            mids = (float(put_c[i_sp]), float(put_c[i_lp]), float(call_c[i_sc]), float(call_c[i_lc]))
            friction, extrapolated = banded_condor_leg_cost_usd(inp.friction, mids, dte, bt.friction_entry_capture)
            hold = bisect_right(nyse, E) - bisect_right(nyse, D)  # NYSE sessions in (D, E]
            feats = compute_features(inp, har, term_cache, D, Dm, E, smile, atm, hold)
            model_credit = _surface_credit(surf, Dm, E, S_D, t_D, smile.r,
                                           (put_k[i_sp], put_k[i_lp], call_k[i_sc], call_k[i_lc]))
            rows.append(dict(
                date=D, expiry=E, target=target, bucket=bucket, dte=dte, hold_sessions=hold, fold=fold_of(D),
                atm_iv=atm, em=em, width=width, credit=credit, risk=width - credit,
                entry_spread=float(friction), entry_comm=float(commission), entry_cost=float(friction + commission),
                extrapolated=bool(extrapolated),
                pricing_resid=(model_credit - credit) if model_credit == model_credit else float("nan"),
                snap_put=float(abs(put_k[i_sp] - sp[0])), snap_call=float(abs(call_k[i_sc] - sc[0])),
                **feats, smile_rmse=smile.rmse, smile_n=smile.n,
                p_short=float(put_k[i_sp]), p_long=float(put_k[i_lp]), c_short=float(call_k[i_sc]),
                c_long=float(call_k[i_lc])))
    return pd.DataFrame(rows), miss, per_target


# --------------------------------------------------------------------------- dry run (coverage only)

def n_eff(rows: pd.DataFrame) -> float:
    """ADDENDUM 1: distinct sessions with a row in the bucket / mean holding sessions. Rows of one session (one per
    target DTE) share their price path and are not independent, so rows are NOT counted."""
    return float(rows["date"].nunique() / rows["hold_sessions"].mean()) if len(rows) else 0.0


def dry_run_report(panel: pd.DataFrame, miss: Counter, per_target: Counter, friction_report: dict) -> dict:
    out: dict = {"window": [str(WINDOW_START), str(WINDOW_END)], "friction": friction_report, "buckets": {},
                 "calendar_gaps_by_target": {
                     str(t): {"attempts": int(per_target[(t, "attempts")]), "no_listed_expiry": int(per_target[(t, "no_expiry")]),
                              "share": round(per_target[(t, "no_expiry")] / per_target[(t, "attempts")], 3)
                              if per_target[(t, "attempts")] else None}
                     for t in TARGET_DTES}}
    test = panel[panel["date"] >= TEST_START]
    for b in BUCKETS:
        formed = panel[panel["bucket"] == b]
        t = test[test["bucket"] == b]
        reasons = {r: n for (bk, r), n in sorted(miss.items()) if bk == b}
        defects = sum(n for r, n in reasons.items() if r not in STRUCTURAL_REASONS)   # ADDENDUM 1, amendment 3
        denom = len(formed) + defects
        entry = {
            "formed": int(len(formed)), "not_formed": reasons,
            "defect_share": (defects / denom) if denom else None,
            "calendar_gaps_no_listed_expiry": int(reasons.get("no_expiry", 0)),
            "collapsed_duplicate_expiry": int(reasons.get("collapsed_duplicate_expiry", 0)),
            "test_rows": int(len(t)), "test_sessions": int(t["date"].nunique()), "test_n_eff": round(n_eff(t), 1),
            "insufficient": bool(n_eff(t) < MIN_N_EFF),
            "mean_hold_sessions": round(float(formed["hold_sessions"].mean()), 2) if len(formed) else None,
        }
        if len(formed):
            entry.update({
                "friction_extrapolated_share": round(float(formed["extrapolated"].mean()), 3),
                "rr25_nan_share": round(float(formed["rr25_norm"].isna().mean()), 3),
                "vrp_nan_share": round(float(formed["vrp_h"].isna().mean()), 3),
                "iv_rank_nan_share": round(float(formed["iv_rank"].isna().mean()), 3),
                "median_width_usd": float(formed["width"].median()),
                "median_credit_over_width": round(float((formed["credit"] / formed["width"]).median()), 3),
                "median_snap_put_usd": float(formed["snap_put"].median()),
                "median_snap_call_usd": float(formed["snap_call"].median()),
                "median_entry_cost_over_credit": round(float((formed["entry_cost"] / formed["credit"]).median()), 3),
            })
        entry["rows_per_fold"] = {str(i): int((t["fold"] == i).sum()) for i in range(len(FOLDS))}
        entry["void_by_missing_share"] = bool(entry["defect_share"] is not None and entry["defect_share"] > MAX_MISSING_SHARE)
        entry["void_rr25_by_out_of_range"] = bool(entry.get("rr25_nan_share", 0) > MAX_OUT_OF_RANGE_SHARE)
        out["buckets"][b] = entry
    return out


def print_dry_run(rep: dict) -> None:
    print(f"\nW118 dry run: window {rep['window'][0]}..{rep['window'][1]} (coverage only, no outcomes computed)")
    fr = rep["friction"]
    print(f"friction: default form {fr['default_form']}, bands need {fr['min_sessions_per_band']} sessions")
    for b, v in fr["bands"].items():
        print(f"  band {b:>5}: {v['n_sessions']:>3} sessions, {v['form']:<9} {'USED' if v['used'] else 'extrapolated/all-DTE'}")
    print("calendar gaps (no listed expiry within the tolerance), per target:")
    for t, g in rep["calendar_gaps_by_target"].items():
        print(f"  target {t:>2}: {g['no_listed_expiry']:>4} of {g['attempts']:>4} sessions ({g['share']:.1%})")
    for b, e in rep["buckets"].items():
        share = f"{e['defect_share']:.1%}" if e["defect_share"] is not None else "n/a"
        print(f"\nbucket {b} (target DTE {BUCKETS[b][0]}-{BUCKETS[b][1]}): formed {e['formed']}, data-defect share {share}, "
              f"{'INSUFFICIENT (n_eff < ' + str(MIN_N_EFF) + ')' if e['insufficient'] else 'n_eff ok'}")
        for r, n in e["not_formed"].items():
            print(f"    not formed: {r:<36} {n}")
        for k in ("test_rows", "test_sessions", "test_n_eff", "mean_hold_sessions", "friction_extrapolated_share", "rr25_nan_share",
                  "vrp_nan_share", "iv_rank_nan_share", "median_width_usd", "median_credit_over_width",
                  "median_snap_put_usd", "median_snap_call_usd", "median_entry_cost_over_credit"):
            if k in e:
                print(f"    {k:<32} {e[k]}")
        print(f"    rows per test fold               {list(e['rows_per_fold'].values())}")
        if e["void_by_missing_share"] or e["void_rr25_by_out_of_range"]:
            print(f"    !! VOID condition would trigger: missing>{MAX_MISSING_SHARE:.0%}={e['void_by_missing_share']}, "
                  f"rr25 out-of-range>{MAX_OUT_OF_RANGE_SHARE:.0%}={e['void_rr25_by_out_of_range']}")


# --------------------------------------------------------------------------- outcomes (the label)

PRIMARY = ("vrp_h", "term_slope", "rr25_norm")             # listed order matters for the 0.8 correlation rule
EXPECTED_SIGN = {"vrp_h": +1}                              # pre-stated direction; the others are two-sided
CONTROLS = ("iv_rank", "atm_iv")
STRESS = 2.0
ALPHA = 0.10
BOOT_N, BOOT_SEED = 2000, 20261101
CANARY_SHUFFLES, CANARY_BOOT_N, CANARY_SHUFFLE_MAX_HITS = 20, 500, 6
CANARY_LEAK_P = 0.01
MIN_FOLD_ROWS, MIN_TRAIN_ROWS = 20, 30
MIN_FOLDS_SAME_SIGN = 4
MAX_FEATURE_CORR = 0.8


def attach_outcomes(panel: pd.DataFrame, inp: Inputs) -> tuple[pd.DataFrame, Counter]:
    """Expiry-day underlying close, settlement value, exit costs, ``pnl_1`` (the label) and ``pnl_2`` (every friction
    term doubled), plus the leaked-feature canary ``leak_rv``. Rows without an expiry-day underlying close are dropped
    and counted per bucket. Intrinsic settlement: no assignment or pin modelling (stated in PLAN.md AS BUILT)."""
    bt = inp.bt
    lost: Counter = Counter()
    has_close = panel["expiry"].map(lambda e: e in inp.spot)
    for b, n in panel.loc[~has_close, "bucket"].value_counts().items():
        lost[b] += int(n)
    df = panel[has_close].copy()
    S_T = df["expiry"].map(inp.spot).astype(float).to_numpy()
    pk, pl, ck, cl = (df[c].to_numpy(dtype=float) for c in ("p_short", "p_long", "c_short", "c_long"))
    intrinsic = [np.maximum(pk - S_T, 0.0), np.maximum(pl - S_T, 0.0), np.maximum(S_T - ck, 0.0), np.maximum(S_T - cl, 0.0)]
    settle = (intrinsic[0] - intrinsic[1]) + (intrinsic[2] - intrinsic[3])
    exit_spread, exit_comm = np.zeros(len(df)), np.zeros(len(df))
    for leg in intrinsic:
        for i in np.flatnonzero(leg > 0):                  # only legs that finish in the money pay an exit cost
            exit_spread[i] += inp.friction.half_spread_usd(float(leg[i]), 0)[0]
            exit_comm[i] += bt.commission_per_contract / 100.0
    exit_spread *= bt.friction_exit_capture
    costs = df["entry_spread"].to_numpy() + df["entry_comm"].to_numpy() + exit_spread + exit_comm
    credit, risk = df["credit"].to_numpy(), df["risk"].to_numpy()
    df["s_T"], df["settle"] = S_T, settle
    df["pnl_1"] = (credit - costs - settle) / risk
    df["pnl_2"] = (credit - STRESS * costs - settle) / risk
    df["max_loss_hit"] = settle >= df["width"].to_numpy() - 1e-9
    df["inside_shorts"] = (S_T >= pk) & (S_T <= ck)
    close = inp.frame["Close"].to_numpy(dtype=float)
    pos = {ts.date(): i for i, ts in enumerate(inp.frame.index)}
    leak = []
    for d, e in zip(df["date"], df["expiry"]):
        i0, i1 = pos.get(d), pos.get(e)
        if i0 is None or i1 is None or i1 <= i0:
            leak.append(float("nan"))
            continue
        r = np.diff(np.log(close[i0:i1 + 1]))
        leak.append(float(np.sqrt(np.mean(r * r) * 252.0)))
    df["leak_rv"] = leak                                    # realised volatility D->expiry: unknowable at D (canary)
    return df, lost


# --------------------------------------------------------------------------- pure statistics

def spearman(x, y) -> float:
    from scipy.stats import rankdata
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def boot_p(draws: np.ndarray, expected_positive: bool) -> float:
    """Percentile-bootstrap p-value: how often the resampled estimate lands on the wrong side of zero."""
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan")
    below, above = 1 + int(np.sum(draws <= 0)), 1 + int(np.sum(draws >= 0))
    if expected_positive:
        return below / (draws.size + 1)
    return min(1.0, 2 * min(below, above) / (draws.size + 1))


def block_bootstrap(stat_fn, session_ids: np.ndarray, block_len: int, n_boot: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap over SESSIONS: consecutive sessions are drawn in blocks of ``block_len`` and every row
    of a drawn session comes with it, so same-day rows and overlapping holding periods stay together.
    ``stat_fn(row_indices) -> 1-D array``. Returns (n_boot, k)."""
    uniq = np.unique(session_ids)
    n = len(uniq)
    rows_of = [np.flatnonzero(session_ids == u) for u in uniq]
    L = max(1, min(int(block_len), n))
    nb = math.ceil(n / L)
    rng = np.random.default_rng(seed)
    first = np.atleast_1d(stat_fn(np.arange(len(session_ids))))
    out = np.empty((n_boot, first.size))
    for b in range(n_boot):
        starts = rng.integers(0, n - L + 1, size=nb)
        sess = (starts[:, None] + np.arange(L)).ravel()[:n]
        out[b] = stat_fn(np.concatenate([rows_of[s] for s in sess]))
    return out


def ecdf_transform(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile of ``x`` in the TRAINING distribution (so a test value is ranked against the past only)."""
    s = np.sort(np.asarray(train, dtype=float))
    return np.searchsorted(s, np.asarray(x, dtype=float), side="right") / max(len(s), 1)


def fold_train_cutoff(fold: int, nyse: list[date]) -> date:
    """Latest EXPIRY a training row may have for ``fold``: the session ``GAP_SESSIONS`` before the fold's first date."""
    from bisect import bisect_left
    return nyse[max(bisect_left(nyse, FOLDS[fold][0]) - GAP_SESSIONS, 0)]


def fold_frames(df: pd.DataFrame, feature: str, nyse: list[date]):
    """Per test fold, with everything fitted on that fold's TRAINING rows only: the percentile-transformed feature and
    outcome residualised on the controls (for the partial correlation) and the tercile cut points of the feature.

    Yields (fold, test_rows_frame). The frame carries ``rf``, ``ry`` (residuals) and ``tercile`` (-1 bottom, 0, +1 top)."""
    need = [feature, "pnl_1", *CONTROLS]
    ok = df[need].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    for fold in range(len(FOLDS)):
        cutoff = fold_train_cutoff(fold, nyse)
        train = df[ok & (df["expiry"] <= cutoff)]
        test = df[ok & (df["fold"] == fold)]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        ctr_tr = np.column_stack([np.ones(len(train))] + [ecdf_transform(train[c].to_numpy(), train[c].to_numpy()) for c in CONTROLS])
        ctr_te = np.column_stack([np.ones(len(test))] + [ecdf_transform(train[c].to_numpy(), test[c].to_numpy()) for c in CONTROLS])
        res = {}
        for name, col in (("rf", feature), ("ry", "pnl_1")):
            u_tr = ecdf_transform(train[col].to_numpy(), train[col].to_numpy())
            u_te = ecdf_transform(train[col].to_numpy(), test[col].to_numpy())
            beta, *_ = np.linalg.lstsq(ctr_tr, u_tr, rcond=None)
            res[name] = u_te - ctr_te @ beta
        lo, hi = np.quantile(train[feature].to_numpy(), [1 / 3, 2 / 3])
        t = test.copy()
        t["rf"], t["ry"] = res["rf"], res["ry"]
        t["tercile"] = np.where(t[feature] <= lo, -1, np.where(t[feature] >= hi, 1, 0))
        yield fold, t


def analyse_feature(df: pd.DataFrame, feature: str, nyse: list[date], *, block_len: int, seed: int,
                    n_boot: int | None = None) -> dict:
    """Every statistic the decision rule needs for ONE feature in ONE bucket, on the pooled out-of-sample rows."""
    n_boot = BOOT_N if n_boot is None else n_boot
    frames = [t for _, t in fold_frames(df, feature, nyse)]
    if not frames:
        return {"feature": feature, "n_rows": 0, "tested": False}
    t = pd.concat(frames)
    t = t.sort_values("date")
    sess = pd.factorize(t["date"])[0]
    x, y, rf, ry = (t[c].to_numpy(dtype=float) for c in (feature, "pnl_1", "rf", "ry"))
    expected_pos = feature in EXPECTED_SIGN

    def stats(idx: np.ndarray) -> np.ndarray:
        return np.array([spearman(x[idx], y[idx]), pearson(rf[idx], ry[idx])])

    rho, partial = stats(np.arange(len(t)))
    draws = block_bootstrap(stats, sess, block_len, n_boot, seed)
    direction = EXPECTED_SIGN.get(feature, 1 if rho >= 0 else -1)
    spread = {}
    for col in ("pnl_1", "pnl_2"):
        top, bot = t.loc[t["tercile"] == 1, col], t.loc[t["tercile"] == -1, col]
        spread[col] = float(direction * (top.mean() - bot.mean())) if len(top) and len(bot) else float("nan")
    fold_signs = {}
    for f, g in t.groupby("fold"):
        r = spearman(g[feature], g["pnl_1"]) if len(g) >= MIN_FOLD_ROWS else float("nan")
        fold_signs[int(f)] = {"rho": r, "rows": int(len(g)), "same_sign": bool(r == r and np.sign(r) == direction)}
    return {
        "feature": feature, "tested": True, "n_rows": int(len(t)), "expected_positive": expected_pos, "direction": direction,
        "rho": rho, "rho_ci90": [float(np.nanpercentile(draws[:, 0], 5)), float(np.nanpercentile(draws[:, 0], 95))],
        "p_raw": boot_p(draws[:, 0], expected_pos),
        "partial_rho": partial,
        "partial_ci90": [float(np.nanpercentile(draws[:, 1], 5)), float(np.nanpercentile(draws[:, 1], 95))],
        "p_partial": boot_p(draws[:, 1], expected_pos),
        "partial_same_sign": bool(partial == partial and rho == rho and np.sign(partial) == np.sign(rho)),
        "tercile_spread_1x": spread["pnl_1"], "tercile_spread_2x": spread["pnl_2"],
        "folds_same_sign": int(sum(v["same_sign"] for v in fold_signs.values())), "folds": fold_signs,
        "pricing_resid_corr": spearman(t[feature], t["pricing_resid"]) if "pricing_resid" in t else float("nan"),
    }


def passes(res: dict, p_holm: float, blocked_by: str | None) -> tuple[bool, list[str]]:
    """The four registered criteria for one feature (n_eff has been checked by the caller)."""
    if not res.get("tested"):
        return False, ["not tested"]
    fails = []
    if blocked_by:
        fails.append(f"correlated above {MAX_FEATURE_CORR} with earlier feature {blocked_by}")
    if not p_holm < ALPHA:
        fails.append("criterion 1: Holm-adjusted p >= 0.10")
    if not (res["partial_same_sign"] and res["p_partial"] < ALPHA):
        fails.append("criterion 2: partial correlation (sign, p < 0.10)")
    if not (res["tercile_spread_1x"] > 0 and res["tercile_spread_2x"] > 0):
        fails.append("criterion 3: tercile spread not positive at 1x and 2x friction")
    if res["folds_same_sign"] < MIN_FOLDS_SAME_SIGN:
        fails.append(f"criterion 4: sign holds in {res['folds_same_sign']} of {len(FOLDS)} folds")
    return not fails, fails


# --------------------------------------------------------------------------- guards: as-of mutation, canaries

def _same(a: float, b: float) -> bool:
    return (a != a and b != b) or abs(a - b) <= 1e-12 * max(1.0, abs(a))


def asof_mutation_check(inp: Inputs, panel: pd.DataFrame, n_samples: int = 8, seed: int = BOOT_SEED) -> dict:
    """The features of an entry at D must not change when EVERYTHING dated on or after D is altered: smiles, daily
    bars, stored IV, spots. Run on the real panel (sampled), through the very function that built it."""
    import dataclasses
    from ait.backtesting.iv_surface import IVSurface, lookup_params
    from ait.backtesting.surface_features import smile_atm_iv
    from ait.data.iv_rank import iv_rank_series

    rng = np.random.default_rng(seed)
    sess_idx = {d: i for i, d in enumerate(inp.sessions)}
    picks = panel.iloc[np.linspace(0, len(panel) - 1, num=min(n_samples, len(panel))).astype(int)]
    bad, checked = [], 0
    for _, row in picks.iterrows():
        D, E = row["date"], row["expiry"]
        Dm = inp.sessions[sess_idx[D] - 1]
        sdf = inp.surface.to_frame()
        later = pd.to_datetime(sdf["as_of"]).dt.date >= D
        sdf.loc[later, "a"] = sdf.loc[later, "a"] * 1.9
        sdf.loc[later, "b"] = sdf.loc[later, "b"] + 0.5
        frame = inp.frame.copy()
        mask = frame.index >= pd.Timestamp(D)
        for col, f in (("Open", 1.1), ("High", 1.3), ("Low", 0.8), ("Close", 1.2)):
            frame.loc[mask, col] = frame.loc[mask, col] * f * rng.uniform(0.9, 1.1, size=int(mask.sum()))
        frame.loc[mask, "implied_vol"] = frame.loc[mask, "implied_vol"] * 3.0
        spot = {d: (v * 1.3 if d >= D else v) for d, v in inp.spot.items()}
        mut = dataclasses.replace(inp, surface=IVSurface.from_frame(sdf, **lookup_params()), frame=frame, spot=spot,
                                  iv_rank=iv_rank_series(frame["implied_vol"]))
        base_smile, mut_smile = inp.surface.smile(Dm, E), mut.surface.smile(Dm, E)
        a = compute_features(inp, HarForecaster(inp.frame), {}, D, Dm, E, base_smile, smile_atm_iv(base_smile), int(row["hold_sessions"]))
        b = compute_features(mut, HarForecaster(frame), {}, D, Dm, E, mut_smile, smile_atm_iv(mut_smile), int(row["hold_sessions"]))
        checked += 1
        diff = [k for k in a if not _same(a[k], b[k])]
        if diff:
            bad.append({"date": str(D), "expiry": str(E), "features_changed": diff})
    return {"checked": checked, "changed": bad, "ok": not bad and checked > 0}


def canaries(df: pd.DataFrame, nyse: list[date], bucket: str, *, block_len: int, sufficient: bool) -> dict:
    """Leaked canary (realised vol D->expiry must be strongly significant in A and B) and shuffled canary (a row-shuffled
    ``vrp_h`` must not be significant more often than chance)."""
    t = df[(df["fold"].notna()) & df[["leak_rv", "pnl_1", "vrp_h"]].notna().all(axis=1)].sort_values("date")
    if len(t) < 3 * MIN_FOLD_ROWS:
        return {"ran": False}
    sess = pd.factorize(t["date"])[0]
    y = t["pnl_1"].to_numpy(dtype=float)
    leak = t["leak_rv"].to_numpy(dtype=float)
    draws = block_bootstrap(lambda i: np.array([spearman(leak[i], y[i])]), sess, block_len, BOOT_N, BOOT_SEED + 1)
    out = {"ran": True, "leak_rho": spearman(leak, y), "leak_p": boot_p(draws[:, 0], False)}
    out["leak_ok"] = bool(out["leak_p"] < CANARY_LEAK_P) if bucket in ("A", "B") else None
    if sufficient:
        v = t["vrp_h"].to_numpy(dtype=float)
        hits = 0
        for k in range(CANARY_SHUFFLES):
            shuf = np.random.default_rng(BOOT_SEED + 100 + k).permutation(v)
            d = block_bootstrap(lambda i: np.array([spearman(shuf[i], y[i])]), sess, block_len, CANARY_BOOT_N, BOOT_SEED + 200 + k)
            hits += int(boot_p(d[:, 0], False) < ALPHA)
        out.update({"shuffle_hits": hits, "shuffle_ok": bool(hits < CANARY_SHUFFLE_MAX_HITS)})
    return out


# --------------------------------------------------------------------------- the study

def analyse(panel: pd.DataFrame, inp: Inputs, miss: Counter, lost_no_close: Counter) -> dict:
    nyse = inp.nyse
    out: dict = {"window": [str(WINDOW_START), str(WINDOW_END)], "friction": inp.friction_report, "buckets": {}}
    test = panel[panel["fold"].notna()]
    pending: dict = {}
    for b in BUCKETS:
        rows = panel[panel["bucket"] == b]
        t = test[test["bucket"] == b]
        reasons = {r: n for (bk, r), n in sorted(miss.items()) if bk == b}
        defects = sum(n for r, n in reasons.items() if r not in STRUCTURAL_REASONS) + lost_no_close.get(b, 0)
        defect_share = defects / (len(rows) + defects + 0.0) if (len(rows) + defects) else 1.0
        eff = n_eff(t)
        block_len = int(t["hold_sessions"].max()) if len(t) else 1
        rr_nan = float(t["rr25_norm"].isna().mean()) if len(t) else 1.0
        entry = {"rows": int(len(rows)), "test_rows": int(len(t)), "test_sessions": int(t["date"].nunique()),
                 "n_eff": round(eff, 1), "block_len_sessions": block_len, "not_formed": reasons,
                 "no_expiry_close": int(lost_no_close.get(b, 0)), "defect_share": defect_share,
                 "rr25_nan_share": rr_nan, "insufficient": bool(eff < MIN_N_EFF), "features": {},
                 "secondary": {}, "void": []}
        if defect_share > MAX_MISSING_SHARE:
            entry["void"].append(f"data-defect share {defect_share:.1%} > {MAX_MISSING_SHARE:.0%}")
        if len(t):
            entry["secondary"] = {"max_loss_hit_rate": round(float(t["max_loss_hit"].mean()), 3),
                                  "inside_shorts_rate": round(float(t["inside_shorts"].mean()), 3),
                                  "mean_pnl_per_risk_1x": round(float(t["pnl_1"].mean()), 4),
                                  "mean_pnl_per_risk_2x": round(float(t["pnl_2"].mean()), 4),
                                  "friction_extrapolated_share": round(float(t["extrapolated"].mean()), 3)}
        for f in PRIMARY:
            if f == "rr25_norm" and rr_nan > MAX_OUT_OF_RANGE_SHARE:
                entry["features"][f] = {"feature": f, "tested": False, "note": "25-delta strikes outside the fitted range too often"}
                continue
            entry["features"][f] = analyse_feature(panel[panel["bucket"] == b], f, nyse, block_len=block_len, seed=BOOT_SEED)
        cs = canaries(panel[panel["bucket"] == b], nyse, b, block_len=block_len, sufficient=not entry["insufficient"])
        entry["canaries"] = cs
        if cs.get("ran") and cs.get("leak_ok") is False:
            entry["void"].append("leaked canary not significant")
        if cs.get("ran") and cs.get("shuffle_ok") is False:
            entry["void"].append("shuffled canary significant too often")
        if not entry["insufficient"]:
            for f, r in entry["features"].items():
                if r.get("tested"):
                    pending[(b, f)] = r["p_raw"]
        out["buckets"][b] = entry
    holm = holm_adjust(pending)
    for b, entry in out["buckets"].items():
        t = test[test["bucket"] == b]
        corr = {}
        for i, f in enumerate(PRIMARY):
            for g in PRIMARY[:i]:
                pair = t[[f, g]].dropna()
                corr[f"{g}~{f}"] = spearman(pair[g], pair[f]) if len(pair) > 3 else float("nan")
        entry["feature_correlations"] = corr
        passed = []
        for i, f in enumerate(PRIMARY):
            r = entry["features"][f]
            blocked = next((g for g in PRIMARY[:i] if abs(corr.get(f"{g}~{f}", 0) or 0) > MAX_FEATURE_CORR), None)
            if entry["insufficient"]:
                r["descriptive_only"] = True
                continue
            ok, fails = passes(r, holm.get((b, f), 1.0), blocked)
            r.update({"p_holm": holm.get((b, f)), "pass": ok, "fails": fails})
            if ok:
                passed.append(f)
        entry["passed"] = passed
        entry["verdict"] = ("VOID" if entry["void"] else "INSUFFICIENT" if entry["insufficient"]
                            else "USEFUL" if passed else "NOT SHOWN")
    return out


def render_markdown(res: dict, mutation: dict, smoke: bool) -> str:
    L = [f"# W118 surface-feature study {'(SMOKE RUN: labels scrambled, verdicts meaningless)' if smoke else ''}".rstrip(), "",
         f"Window {res['window'][0]}..{res['window'][1]}; statistics only (no per-row prices or P&L).", "",
         f"As-of mutation check: {'OK' if mutation['ok'] else 'FAILED'} ({mutation['checked']} entries sampled).", ""]
    for b, e in res["buckets"].items():
        L += [f"## Bucket {b} (target DTE {BUCKETS[b][0]}-{BUCKETS[b][1]}): {e['verdict']}", "",
              f"rows {e['rows']}, test rows {e['test_rows']}, test sessions {e['test_sessions']}, n_eff {e['n_eff']}, "
              f"block {e['block_len_sessions']} sessions, data-defect share {e['defect_share']:.1%}, "
              f"rr25 outside fitted range {e['rr25_nan_share']:.1%}", ""]
        if e["void"]:
            L += ["VOID because: " + "; ".join(e["void"]), ""]
        if e["insufficient"]:
            L += ["INSUFFICIENT: the figures below are descriptive, not a verdict.", ""]
        L += ["| feature | rho | 90% CI | p raw | p Holm | partial rho | partial p | spread 1x | spread 2x | folds same sign | resid corr | pass |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, r in e["features"].items():
            if not r.get("tested"):
                L.append(f"| {f} | not tested | | | | | | | | | | |")
                continue
            ph = r.get("p_holm")
            L.append(f"| {f} | {r['rho']:+.3f} | [{r['rho_ci90'][0]:+.3f}, {r['rho_ci90'][1]:+.3f}] | {r['p_raw']:.3f} | "
                     f"{'' if ph is None else f'{ph:.3f}'} | {r['partial_rho']:+.3f} | {r['p_partial']:.3f} | "
                     f"{r['tercile_spread_1x']:+.3f} | {r['tercile_spread_2x']:+.3f} | {r['folds_same_sign']}/{len(FOLDS)} | "
                     f"{r['pricing_resid_corr']:+.2f} | {'descriptive' if r.get('descriptive_only') else ('YES' if r.get('pass') else 'no')} |")
        for f, r in e["features"].items():
            if r.get("fails"):
                L.append(f"- {f} fails: " + "; ".join(r["fails"]))
        L += ["", "Secondary (reported only): " + json.dumps(e["secondary"]),
              "Feature correlations: " + json.dumps({k: round(v, 3) for k, v in e["feature_correlations"].items()}),
              "Canaries: " + json.dumps(e["canaries"], default=str), ""]
    L += ["Limits: about 18 months of test sessions, one underlying, one regime; adjacent rows overlap (hence the block bootstrap); "
          "bucket A friction is extrapolated; hold-to-expiry intrinsic settlement; NOT SHOWN is a likely valid outcome.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="coverage, misses and friction only; computes no outcomes")
    mode.add_argument("--smoke", action="store_true",
                      help="the whole pipeline with each bucket's outcomes SHUFFLED across rows: exercises every code path "
                           "without revealing any real feature-outcome relationship; writes to <out>_SMOKE, verdicts meaningless")
    mode.add_argument("--run", action="store_true",
                      help="THE pre-registered run (commit the script and studies/surface_feature_study.yaml first)")
    ap.add_argument("--db", default=str(ROOT / "data" / "historical.db"))
    ap.add_argument("--out", default=str(OUT_BASE))
    args = ap.parse_args(argv)

    from study_cli import load_study_config
    study = load_study_config(STUDY_FILE)               # validates the study and its PLAN.md registration
    from ait.config.invariants_research import (
        RESEARCH_MAX_DIVIDEND_GAP_DAYS, RESEARCH_MAX_EVENT_GAP_DAYS, RESEARCH_MAX_NULL_IV_RUN_SESSIONS)
    from ait.data.research_inputs import ResearchRequest, Window, preflight_or_raise
    db = Path(args.db)
    preflight_or_raise(ResearchRequest(
        window=Window(WINDOW_START, WINDOW_END), symbols=(SYMBOL,), db_path=db,
        max_null_iv_run=RESEARCH_MAX_NULL_IV_RUN_SESSIONS, max_dividend_gap_days=RESEARCH_MAX_DIVIDEND_GAP_DAYS,
        max_event_gap_days=RESEARCH_MAX_EVENT_GAP_DAYS, friction_per_leg=True, friction_bands=True, surface=True),
        context="surface_feature_study")

    inp = load_inputs(db)
    assert inp.friction.default is not None and inp.friction.bands, "per-leg friction model / bands did NOT load"
    panel, miss, per_target = build_entries(inp)
    out = Path(args.out)
    if args.dry_run:
        rep = dry_run_report(panel, miss, per_target, inp.friction_report)
        print_dry_run(rep)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dry_run.json").write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {out / 'dry_run.json'}")
        return 0

    panel, lost = attach_outcomes(panel, inp)
    if args.smoke:
        out = out.with_name(out.name + "_SMOKE")
        rng = np.random.default_rng(BOOT_SEED)
        for b in BUCKETS:
            m = (panel["bucket"] == b).to_numpy()
            perm = rng.permutation(int(m.sum()))
            for col in ("pnl_1", "pnl_2"):
                panel.loc[m, col] = panel.loc[m, col].to_numpy()[perm]
    mutation = asof_mutation_check(inp, panel)
    res = analyse(panel, inp, miss, lost)
    res["asof_mutation_check"] = mutation
    res["smoke"] = bool(args.smoke)
    if not mutation["ok"]:
        for e in res["buckets"].values():
            e["void"].append("as-of mutation check failed")
            e["verdict"] = "VOID"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    md = render_markdown(res, mutation, args.smoke)
    (out / "results.md").write_text(md)
    print(md)
    print(f"wrote {out / 'results.json'} and {out / 'results.md'}")
    if args.run:
        from study_cli import write_study_record
        write_study_record("surface_feature_study", args, study, [res], symbols=[SYMBOL], db_path=db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
