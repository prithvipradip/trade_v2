"""wing_k walk-forward study (decision rule PRE-REGISTERED in PLAN.md
2026-07-28, before results). Sweep k over {0.6, 0.8, 1.0, 1.2}: iron_condor
only, SPY/QQQ/IWM, every other parameter at live/backtest-calibrated values
(ratio floor 0.20, credit floor via env defaults, delta band unchanged).
Writes reports/wingk_study/k_<k>.json + a final table."""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402

OUT = ROOT / "reports" / "wingk_study"
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["SPY", "QQQ", "IWM"]
KS = [0.6, 0.8, 1.0, 1.2]


def metrics(trades):
    pnls = [t.get("pnl", 0.0) or 0.0 for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    wins = sum(1 for p in pnls if p > 0)
    cum = peak = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {"n": len(pnls), "wins": wins, "total": round(sum(pnls), 2),
            "pf": round(gp / gl, 3) if gl else (float("inf") if gp else 0.0),
            "max_dd": round(dd, 2)}


async def run_k(k, bc, data):
    cfg = WalkForwardConfig(
        wing_k=k,
        train_days=126, test_days=63, step_days=42, gap_days=5,
        initial_capital=bc.initial_capital,
        position_size_pct=bc.position_size_pct,
        wing_floor_dollars=bc.wing_floor_dollars,
        iv_floor=bc.iv_floor,
        delta_iv_scale=bc.delta_iv_scale,
        spread_base=bc.spread_base,
        spread_iv_sensitivity=bc.spread_iv_sensitivity,
        spread_dte_sensitivity=bc.spread_dte_sensitivity,
        spread_cap=bc.spread_cap,
        ic_min_credit_width=0.20,
    )
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg,
                               db_path=ROOT / "data" / "historical.db",
                               progress_dir=OUT / f"k{k}")
    result = await bt.run(data=data)
    txt = result.summary()
    def grab(pat, cast=float, default=0.0):
        mm = re.search(pat, txt)
        return cast(mm.group(1).replace(",", "")) if mm else default
    m = {
        "k": k,
        "n": grab(r"Total Trades:\s+(\d+)", int, 0),
        "win_rate": grab(r"Win Rate:\s+([\d.]+)%"),
        "pf": grab(r"Profit Factor:\s+([\d.]+|inf)",
                   lambda v: float("inf") if v == "inf" else float(v)),
        "max_dd_pct": grab(r"Max Drawdown:\s+([\d.]+)%"),
        "total_ret_pct": grab(r"Total Return:\s+(-?[\d.]+)%"),
        "windows": grab(r"Windows:\s+(\d+)", int, 0),
    }
    (OUT / f"k_{k}.json").write_text(json.dumps(m, indent=1))
    try:
        (OUT / f"k_{k}_summary.txt").write_text(result.summary(), encoding="utf-8")
    except Exception:
        pass
    print(f"[wingk] k={k}: {m}", flush=True)
    return m


def load_data():
    # Pre-load and TZ-NORMALIZE: the engine slices windows with naive
    # timestamps; load_daily_ohlcv returns ET-aware indexes, and the mix
    # raised "Cannot compare dtypes" on every window. Naive-ET throughout.
    from ait.data.market_data import load_daily_ohlcv
    data = {}
    for sym in SYMBOLS:
        df = load_daily_ohlcv(sym, days=1500)
        if df is not None and len(df):
            if getattr(df.index, "tz", None) is not None:
                df = df.copy(); df.index = df.index.tz_localize(None)
            data[sym] = df
            print(f"[wingk] data {sym}: {len(df)} rows", flush=True)
    return data


async def main():
    bc = load_settings(str(ROOT / "config.yaml")).backtest
    data = load_data()
    rows = []
    for k in KS:
        try:
            rows.append(await run_k(k, bc, data))
        except Exception as e:  # noqa: BLE001
            print(f"[wingk] k={k} FAILED: {e}", flush=True)
            rows.append({"k": k, "error": str(e)[:200]})
    print("\n[wingk] ===== RESULTS =====")
    print(f"{'k':>4} {'win':>7} {'trades':>7} {'winrate':>8} {'PF':>7} {'DD%':>6} {'ret%':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['k']:>4} ERROR {r['error'][:60]}")
        else:
            print(f"{r['k']:>4} {r['windows']:>7} {r['n']:>7} {r['win_rate']:>8} "
                  f"{r['pf']:>7} {r['max_dd_pct']:>6} {r['total_ret_pct']:>7}")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    # Pre-registered rule: trades >= 30 AND PF > 1.0; highest PF wins;
    # ties within 0.1 -> more trades; none qualify -> keep 1.0.
    q = [r for r in rows if "error" not in r and r["n"] >= 30 and r["pf"] > 1.0]
    if not q:
        print("[wingk] RULE: no qualifier -> KEEP k=1.0 (wait for vol)")
    else:
        best = max(q, key=lambda r: (round(r["pf"], 1), r["n"]))
        print(f"[wingk] RULE -> k={best['k']} (PF {best['pf']}, n {best['n']})")


if __name__ == "__main__":
    asyncio.run(main())
