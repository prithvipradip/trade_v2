"""SHADOW TOURNAMENT (rule PRE-REGISTERED in PLAN 2026-08-03): candidates vs the
IC baseline on identical windows/data, k=0.8. Promotion: PF > 1.2 AND max DD% <=
IC arm, n >= 30 OOS. short_strangle arm = wings-cost BENCHMARK ONLY (never live).
Writes reports/shadow_tournament/<arm>.json + table."""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402

OUT = ROOT / "reports" / "shadow_tournament"
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["SPY", "QQQ", "IWM"]
ARMS = ['call_credit_spread', 'jade_lizard']


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


async def run_k(strat, bc, data):
    cfg = WalkForwardConfig(
        wing_k=0.8,
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
    bt = WalkForwardBacktester(SYMBOLS, [strat], config=cfg,
                               db_path=ROOT / "data" / "historical.db",
                               progress_dir=OUT / strat)
    result = await bt.run(data=data)
    txt = result.summary()
    def grab(pat, cast=float, default=0.0):
        mm = re.search(pat, txt)
        return cast(mm.group(1).replace(",", "")) if mm else default
    m = {
        "arm": strat,
        "n": grab(r"Total Trades:\s+(\d+)", int, 0),
        "win_rate": grab(r"Win Rate:\s+([\d.]+)%"),
        "pf": grab(r"Profit Factor:\s+([\d.]+|inf)",
                   lambda v: float("inf") if v == "inf" else float(v)),
        "max_dd_pct": grab(r"Max Drawdown:\s+([\d.]+)%"),
        "total_ret_pct": grab(r"Total Return:\s+(-?[\d.]+)%"),
        "windows": grab(r"Windows:\s+(\d+)", int, 0),
    }
    (OUT / f"{strat}.json").write_text(json.dumps(m, indent=1))
    try:
        (OUT / f"{strat}_summary.txt").write_text(result.summary(), encoding="utf-8")
    except Exception:
        pass
    print(f"[shadow1b] {strat}: {m}", flush=True)
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
            print(f"[shadow1b] data {sym}: {len(df)} rows", flush=True)
    return data


async def main():
    bc = load_settings(str(ROOT / "config.yaml")).backtest
    data = load_data()
    rows = []
    for strat in ARMS:
        try:
            rows.append(await run_k(strat, bc, data))
        except Exception as e:  # noqa: BLE001
            print(f"[shadow1b] {strat} FAILED: {e}", flush=True)
            rows.append({"arm": strat, "error": str(e)[:200]})
    print("\n[wingk] ===== RESULTS =====")
    print(f"{'arm':>18} {'win':>7} {'trades':>7} {'winrate':>8} {'PF':>7} {'DD%':>6} {'ret%':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['arm']:>18} ERROR {r['error'][:60]}")
        else:
            print(f"{r['arm']:>18} {r['windows']:>7} {r['n']:>7} {r['win_rate']:>8} "
                  f"{r['pf']:>7} {r['max_dd_pct']:>6} {r['total_ret_pct']:>7}")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    print("[shadow1b] PLAN rule: promote only if PF > 1.2 AND max_dd_pct <= IC arm, n >= 30; strangle = benchmark only")


if __name__ == "__main__":
    asyncio.run(main())
