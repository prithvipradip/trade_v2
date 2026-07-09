#!/usr/bin/env python
"""AIT bot status — honest snapshot of what the bot is doing.

get_status() returns a dict consumed by both the CLI (python status.py) and
the live web dashboard (status_server.py). Reads the state DB + logs directly
(no IBKR connection, so it never fights the bot for clientId 1).
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "ait_state.db"
LOGS = ROOT / "logs"


def _tail_lines(path: Path, n: int = 4000) -> list[str]:
    if not path.exists():
        return []
    try:
        data = path.read_bytes()[-2_000_000:]
        return data.decode("utf-8", "replace").splitlines()[-n:]
    except Exception:
        return []


def _proc_running() -> tuple[bool, int]:
    try:
        # A11 (deep-audit): wmic is deprecated/removed on newer Windows.
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
             "| Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=15).stdout
        orch = len(re.findall(r"run_orchestrator\.py", out))
        bot = len(re.findall(r"-m ait\.main", out))
        return (orch >= 1 and bot >= 1), orch + bot
    except Exception:
        return False, 0


def _last_heartbeat() -> str:
    for ln in reversed(_tail_lines(LOGS / "orchestrator.log", 200)):
        if "bot_healthy" in ln:
            m = re.match(r"\[([\d\- :]+)\]", ln)
            return m.group(1) if m else "?"
    return "none"


def _last_connect() -> str:
    for ln in reversed(_tail_lines(LOGS / "ait.log", 3000)):
        if "ibkr_connected" in ln:
            m = re.search(r'"timestamp": "([^"]+)"', ln)
            return m.group(1)[:19] if m else "?"
    return "none"


def _crash_count() -> int:
    return sum(1 for ln in _tail_lines(LOGS / "bot_stdout.log", 50000)
               if "Windows fatal exception" in ln)


def _keeper_relaunches() -> int:
    return sum(1 for ln in _tail_lines(LOGS / "keeper.log", 5000)
               if "DOWN - relaunching" in ln)


def _readonly_recent() -> bool:
    for ln in _tail_lines(LOGS / "bot_stdout.log", 400):
        if "Error 321" in ln and "Read-Only" in ln:
            return True
    return False


def _count_event_today(event: str) -> int:
    today = date.today().isoformat()
    pat = f'"event": "{event}"'
    return sum(1 for ln in _tail_lines(LOGS / "ait.log", 60000)
               if pat in ln and today in ln)


def _last_activity() -> str:
    skip = ("bot_healthy", "<<<", ">>>", "Connection pool", "FutureWarning",
            "Unknown", "Error 200", "Error 10", "waiting_for_phase")
    for ln in reversed(_tail_lines(LOGS / "ait.log", 500)):
        if any(s in ln for s in skip):
            continue
        m = re.search(r'"event": "([^"]+)".*"timestamp": "([^"]+)"', ln)
        if not m:
            m2 = re.search(r'"timestamp": "([^"]+)".*"event": "([^"]+)"', ln)
            if m2:
                return f"{m2.group(1)[:19]}  {m2.group(2)}"
            continue
        return f"{m.group(2)[:19]}  {m.group(1)}"
    return "—"


def get_status() -> dict:
    today = date.today().isoformat()
    alive, nproc = _proc_running()
    out: dict = {
        "asof": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "running": alive,
        "procs": nproc,
        "last_heartbeat": _last_heartbeat(),
        "last_connect": _last_connect(),
        "crashes": _crash_count(),
        "keeper_relaunches": _keeper_relaunches(),
        "readonly": _readonly_recent(),
        "last_activity": _last_activity(),
        "today": {ev: _count_event_today(ev) for ev in
                  ("ml_prediction", "signals_generated", "trade_executed", "trade_rejected")},
        "open_positions": [],
        "unrealized_total": 0.0,
        "pnl_today": 0.0, "pnl_today_n": 0,
        "pnl_life": 0.0, "pnl_life_n": 0,
    }
    if DB.exists():
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT t.symbol, t.strategy, t.status, t.entry_price, t.entry_time, "
                "       op.unrealized_pnl, op.pnl_pct, op.mark_time "
                "FROM trades t LEFT JOIN open_positions op ON op.trade_id = t.trade_id "
                "WHERE t.status NOT IN ('closed') ORDER BY t.entry_time DESC").fetchall()
        except sqlite3.OperationalError:
            # Pre-migration DB (pnl_pct/mark_time added by the bot's
            # StateManager on first start of new code) — degrade gracefully.
            rows = con.execute(
                "SELECT t.symbol, t.strategy, t.status, t.entry_price, t.entry_time, "
                "       op.unrealized_pnl, 0 AS pnl_pct, '' AS mark_time "
                "FROM trades t LEFT JOIN open_positions op ON op.trade_id = t.trade_id "
                "WHERE t.status NOT IN ('closed') ORDER BY t.entry_time DESC").fetchall()
        out["open_positions"] = [
            {"symbol": r["symbol"], "strategy": r["strategy"], "status": r["status"],
             "entry": r["entry_price"], "since": (r["entry_time"] or "")[:16],
             "unrealized": round(r["unrealized_pnl"] or 0.0, 2),
             "pnl_pct": round((r["pnl_pct"] or 0.0) * 100, 1),
             "marked": (r["mark_time"] or "")[:16]}
            for r in rows
        ]
        out["unrealized_total"] = round(
            sum(p["unrealized"] for p in out["open_positions"]), 2)
        real = ("AND exit_reason_detailed NOT LIKE '%migrated%' "
                "AND exit_reason_detailed NOT LIKE '%pending%' "
                "AND exit_reason_detailed NOT LIKE '%never_filled%'")
        t = con.execute(f"SELECT COALESCE(SUM(realized_pnl),0),COUNT(*) FROM trades "
                        f"WHERE status='closed' AND exit_time>=? {real}", (today,)).fetchone()
        l = con.execute(f"SELECT COALESCE(SUM(realized_pnl),0),COUNT(*) FROM trades "
                        f"WHERE status='closed' {real}").fetchone()
        out.update(pnl_today=t[0], pnl_today_n=t[1], pnl_life=l[0], pnl_life_n=l[1])
        con.close()
    return out


def main() -> None:
    s = get_status()
    print("=" * 56)
    print(f"  AIT BOT STATUS   {s['asof']}")
    print("=" * 56)
    print("\nHEALTH")
    print(f"  running          : {'YES' if s['running'] else 'NO'}  ({s['procs']} procs)")
    print(f"  last heartbeat   : {s['last_heartbeat']}")
    print(f"  last IBKR connect: {s['last_connect']}")
    print(f"  native crashes   : {s['crashes']}")
    print(f"  keeper relaunches: {s['keeper_relaunches']}")
    print(f"  read-only recent : {'YES — orders may be rejected!' if s['readonly'] else 'no'}")
    print(f"  last activity    : {s['last_activity']}")
    print("\nTODAY")
    for ev, n in s["today"].items():
        print(f"  {ev:18}: {n}")
    print(f"\nOPEN POSITIONS ({len(s['open_positions'])})   unrealized total: ${s['unrealized_total']:+,.2f}")
    for p in s["open_positions"]:
        print(f"  {p['symbol']:5} {p['strategy']:16} {p['status']:8} entry={p['entry']}  "
              f"unreal=${p['unrealized']:+8.2f} ({p['pnl_pct']:+.1f}%)  since {p['since']}")
    print("\nREALIZED P&L  (real exits only)")
    print(f"  today   : ${s['pnl_today']:>10,.2f}  ({s['pnl_today_n']} closed)")
    # R5: "lifetime" hid the 2026-07-06 reset (95 broken-P&L trades archived
    # to trades_legacy). Label honestly so nobody reads this as full history.
    print(f"  since 07-06 reset: ${s['pnl_life']:>10,.2f}  ({s['pnl_life_n']} closed; pre-reset archive in trades_legacy)")
    print("\n" + "=" * 56)


if __name__ == "__main__":
    main()
