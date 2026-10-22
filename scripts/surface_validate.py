#!/usr/bin/env python
"""W75: out-of-sample validation of the real-bar IV surface against the engine's
current pricing, bar to bar.

For every session D with a surface from D-1:
  * surface   : IV from the (D-1, expiry) smile at D's log-moneyness
  * bs_legacy : the engine's model: the stored 30-day IV known at D-1 plus its
                DTE-free skew (``skewed_leg_iv``)
  * actual    : the IV (and close) the D bar itself solved to
Nothing from D is used to predict D. Reported by DTE band and moneyness, in vol
points and in dollars of option price (price is what a backtest P&L is made of).

    python scripts/surface_validate.py [--underlying QQQ] [--min-volume 50]

Read-only: opens data/historical.db ``mode=ro``; the cache goes under data/
research_cache (git-ignored). Statistics only.
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

import argparse  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((1, 2, "1-2"), (3, 7, "3-7"), (8, 14, "8-14"), (15, 45, "15-45"))
MONEY = ((0.0, 0.01, "<1%"), (0.01, 0.02, "1-2%"), (0.02, 0.04, "2-4%"), (0.04, 0.081, "4-8%"))


def _band(dte: int) -> str | None:
    for lo, hi, lab in BANDS:
        if lo <= dte <= hi:
            return lab
    return None


def _money(absm: float) -> str | None:
    for lo, hi, lab in MONEY:
        if lo <= absm < hi:
            return lab
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--db", default=str(ROOT / "data/historical.db"))
    ap.add_argument("--min-volume", type=int, default=50, help="test rows: min volume")
    ap.add_argument("--min-price", type=float, default=0.10, help="test rows: min close")
    ap.add_argument("--cache", default=str(ROOT / "data/research_cache"))
    args = ap.parse_args(argv)

    from ait.backtesting.iv_surface import build_or_load, load_bars, usable_rows
    from ait.backtesting.pricing import OptionType, black_scholes_price, skewed_leg_iv
    from ait.backtesting.replay import ResearchData
    from ait.config.research import use_research_config
    from ait.config.settings import load_settings

    use_research_config()
    bt = load_settings().backtest
    skew_f, skew_s, floor = float(bt.skew_factor), float(bt.skew_slope_per_pct_otm), float(bt.iv_floor)
    put_c, call_c = float(bt.engine_tuning.put_skew_coefficient), float(bt.engine_tuning.call_skew_coefficient)   # W90c

    print("loading bars / building surface (cached) ...", flush=True)
    surf = build_or_load(args.db, args.underlying, args.cache)
    print(f"surface: {len(surf)} smiles over {len(surf.dates())} dates "
          f"({surf.dates()[0]} .. {surf.dates()[-1]})", flush=True)

    bars = load_bars(args.db, args.underlying)
    test = usable_rows(bars, min_volume=args.min_volume, min_price=args.min_price, max_abs_m=0.08)
    rd = ResearchData(args.db, args.underlying)
    daily_iv = rd.daily_frame()["implied_vol"].dropna()
    daily_iv = daily_iv[daily_iv > 0]

    acc: dict[tuple[str, str, str], list] = defaultdict(list)   # (cut, bucket, model) -> rows
    cover = defaultdict(lambda: [0, 0])                          # band -> [predicted, total]
    dates = surf.dates()
    d_index = {d: i for i, d in enumerate(dates)}
    # previous SURFACE date for each test date (sessions are consecutive in the bars)
    for D, g in test.groupby("date"):
        prev = surf.latest_on_or_before(pd.Timestamp(D).date() - pd.Timedelta(days=1).to_pytimedelta(), 4)
        if prev is None:
            continue
        stored = daily_iv[daily_iv.index < pd.Timestamp(D)]
        if stored.empty:
            continue
        base_iv = float(stored.iloc[-1])
        for r in g.itertuples(index=False):
            band = _band(int(r.dte))
            if band is None:
                continue
            S = float(r.underlying_close)
            K = float(r.strike)
            opt = OptionType.CALL if r.right == "C" else OptionType.PUT
            t = max(int(r.dte), 1) / 365.0
            rate = float(r.risk_free_rate) if r.risk_free_rate == r.risk_free_rate else 0.05
            cover[band][1] += 1
            s_iv = surf.iv(prev, r.expiry, K, S)
            if s_iv is None:
                continue
            cover[band][0] += 1
            l_iv = skewed_leg_iv(base_iv, K, S, opt, skew_f, skew_s, floor, put_c, call_c)
            actual_iv = float(r.iv)
            act_px = float(r.close)
            absm = abs(math.log(K / S))
            mb = _money(absm)
            for model, iv in (("surface", s_iv), ("bs_legacy", l_iv)):
                px = black_scholes_price(S, K, t, rate, iv, opt)
                rec = (iv - actual_iv, px - act_px, act_px)
                acc[("band", band, model)].append(rec)
                if mb:
                    acc[("band+money", f"{band}|{mb}", model)].append(rec)

    def line(rows):
        ive = np.array([x[0] for x in rows]); pxe = np.array([x[1] for x in rows])
        act = np.array([x[2] for x in rows])
        return (len(rows), 100 * np.median(ive), 100 * np.median(np.abs(ive)),
                np.median(pxe), np.median(np.abs(pxe)), 100 * np.median(np.abs(pxe) / act))

    print("\nOUT-OF-SAMPLE, bar to bar (D predicted from D-1). IV error = model - actual, vol points;"
          " price error in $ per share.\n")
    print(f"{'DTE band':9s} {'model':10s} {'n':>7s} {'IV bias':>8s} {'IV |err|':>9s} "
          f"{'px bias':>8s} {'px |err|':>9s} {'px |err|%':>10s} {'coverage':>9s}")
    verdicts = {}
    for _, _, lab in BANDS:
        for model in ("surface", "bs_legacy"):
            rows = acc.get(("band", lab, model))
            if not rows:
                continue
            n, ib, ia, pb, pa, pr = line(rows)
            cv = cover[lab]
            cov = f"{100 * cv[0] / max(cv[1], 1):5.1f}%" if model == "surface" else ""
            print(f"{lab:9s} {model:10s} {n:7d} {ib:+8.2f} {ia:9.2f} {pb:+8.3f} {pa:9.3f} {pr:9.1f}% {cov:>9s}")
            verdicts[(lab, model)] = (pa, pr)
    print("\nBy moneyness (|ln K/S|):")
    print(f"{'band|money':14s} {'model':10s} {'n':>7s} {'IV |err|':>9s} {'px |err|':>9s} {'px |err|%':>10s}")
    for _, _, lab in BANDS:
        for _, _, mlab in MONEY:
            for model in ("surface", "bs_legacy"):
                rows = acc.get(("band+money", f"{lab}|{mlab}", model))
                if rows and len(rows) >= 30:
                    n, ib, ia, pb, pa, pr = line(rows)
                    print(f"{lab + '|' + mlab:14s} {model:10s} {n:7d} {ia:9.2f} {pa:9.3f} {pr:9.1f}%")
    print("\nVerdict (price |err| median, surface vs bs_legacy):")
    for _, _, lab in BANDS:
        s, l = verdicts.get((lab, "surface")), verdicts.get((lab, "bs_legacy"))
        if s and l:
            better = s[0] < l[0]
            print(f"  {lab:6s} surface {s[0]:.3f} vs legacy {l[0]:.3f}  -> "
                  f"{'SURFACE BETTER' if better else 'legacy better'} "
                  f"({100 * (1 - s[0] / l[0]):+.0f}% error reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
