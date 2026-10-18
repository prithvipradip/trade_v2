"""W79a: what a DTE-searching walk-forward actually found (pure functions).

A study that lets Optuna choose the entry DTE produces, per window, a list of trials
``{"params": {...entry_dte...}, "value": objective, "n_trades": n, "state": "COMPLETE"}``.
This module turns those into the three things the promotion rule needs:

* the best DTE per window (and how far its objective is from the runner-ups),
* where the TOP-DECILE trials sit (a single lucky trial is not a region),
* the objective and trade count by DTE band, so a band that never trades is visible.

Promotion rule (operator-owned, the programme's A-assumptions): a DTE band is a candidate only
if it is represented in the top decile in at least two thirds of windows AND its parity band
passed (W72/W75) AND its friction is calibrated (W76). This module computes the first
condition; the other two are checked against ``parity_replay`` / the friction status by the
operator and are printed here as reminders, never assumed.

No pandas, numpy or engine imports: tests drive it on synthetic trial lists.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_BAND_EDGES: tuple[int, ...] = (2, 7, 14, 45)   # = parity.dte_band_upper_edges
_DTE_NAMES = ("entry_dte",)


def band_label(dte: int, edges: Iterable[int] = DEFAULT_BAND_EDGES) -> str:
    lo = 0
    for hi in edges:
        if dte <= hi:
            return f"{lo}-{hi}"
        lo = hi + 1
    return f"{lo}+"


def band_order(edges: Iterable[int] = DEFAULT_BAND_EDGES) -> list[str]:
    out, lo = [], 0
    for hi in edges:
        out.append(f"{lo}-{hi}")
        lo = hi + 1
    out.append(f"{lo}+")
    return out


def _param(trial: Mapping[str, Any], name: str) -> Any:
    """A trial's param by bare name; params may carry a ``strategy__`` prefix."""
    params = trial.get("params") or {}
    if name in params:
        return params[name]
    for k, v in params.items():
        if k.rpartition("__")[2] == name:
            return v
    return None


def _scored(trials: Iterable[Mapping[str, Any]]) -> list[dict]:
    out = []
    for t in trials:
        v, dte = t.get("value"), _param(t, "entry_dte")
        if str(t.get("state", "COMPLETE")) != "COMPLETE" or v is None or dte is None:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        out.append({"dte": int(dte), "value": float(v), "n_trades": int(t.get("n_trades") or 0),
                    "guard": _param(t, "expiry_guard_dte"), "fraction": _param(t, "max_hold_fraction")})
    return out


def summarise_window(trials: Iterable[Mapping[str, Any]], *, top_frac: float = 0.10,
                     edges: Iterable[int] = DEFAULT_BAND_EDGES) -> dict | None:
    """One window's findings, or None when no trial carries a DTE and a finite objective."""
    edges = tuple(edges)
    rows = _scored(trials)
    if not rows:
        return None
    rows.sort(key=lambda r: r["value"], reverse=True)
    k = max(1, math.ceil(top_frac * len(rows)))
    top = rows[:k]
    best = rows[0]
    bands: dict[str, dict] = {}
    for r in rows:
        b = bands.setdefault(band_label(r["dte"], edges), {"n": 0, "values": [], "trades": []})
        b["n"] += 1
        b["values"].append(r["value"])
        b["trades"].append(r["n_trades"])
    by_band = {
        lab: {"n": b["n"], "best": max(b["values"]), "median": statistics.median(b["values"]),
              "median_trades": statistics.median(b["trades"])}
        for lab, b in bands.items()}
    return {
        "n_trials": len(rows),
        "best_dte": best["dte"], "best_value": best["value"], "best_n_trades": best["n_trades"],
        "best_guard": best["guard"], "best_fraction": best["fraction"],
        "top_n": k, "top_dtes": sorted(r["dte"] for r in top),
        "top_bands": sorted({band_label(r["dte"], edges) for r in top}),
        "by_band": by_band,
    }


def summarise_study(trials_by_window: Mapping[Any, Iterable[Mapping[str, Any]]], *,
                    top_frac: float = 0.10, edges: Iterable[int] = DEFAULT_BAND_EDGES,
                    min_window_share: float = 2.0 / 3.0) -> dict:
    """Aggregate across windows (``{window_id: [trials]}``)."""
    edges = tuple(edges)
    windows = {}
    for wid, trials in trials_by_window.items():
        s = summarise_window(trials, top_frac=top_frac, edges=edges)
        if s is not None:
            windows[wid] = s
    n = len(windows)
    share = {lab: 0.0 for lab in band_order(edges)}
    for s in windows.values():
        for lab in s["top_bands"]:
            share[lab] = share.get(lab, 0.0) + 1.0
    if n:
        share = {lab: c / n for lab, c in share.items()}
    best_bands = [band_label(s["best_dte"], edges) for s in windows.values()]
    return {
        "n_windows": n,
        "windows": windows,
        "top_decile_band_share": share,
        "candidate_bands": sorted(lab for lab, sh in share.items()
                                  if n and sh >= min_window_share),
        "best_dte_by_window": {w: s["best_dte"] for w, s in windows.items()},
        "best_band_counts": {lab: best_bands.count(lab) for lab in band_order(edges)},
        "min_window_share": min_window_share,
    }


def format_report(study: Mapping[str, Any]) -> list[str]:
    """Plain-text lines for run_backtest.py / the PLAN entry (statistics only)."""
    L: list[str] = []
    n = study.get("n_windows", 0)
    L.append(f"DTE SEARCH: {n} window(s) with scored trials")
    if not n:
        L.append("  no window produced a scored trial with an entry DTE (was the search enabled?)")
        return L
    L.append("  best DTE per window: " + ", ".join(
        f"w{w}={d}" for w, d in sorted(study["best_dte_by_window"].items())))
    L.append("  share of windows with a top-decile trial in each band:")
    for lab, sh in study["top_decile_band_share"].items():
        L.append(f"    {lab:>6s}  {sh:5.0%}")
    cands = study["candidate_bands"]
    L.append(f"  candidate bands (>= {study['min_window_share']:.0%} of windows): "
             f"{', '.join(cands) if cands else 'NONE'}")
    L.append("  a candidate is only promotable if its parity band PASSED and its friction band is "
             "ACTIVE; neither is checked here.")
    return L
