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
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
# W3: every "is this a real close?" / "is this position open?" / go-live
# verdict question is answered by ONE module, shared with the scheduled
# Telegram scorecard in ait.orchestration.master.  Before W3 the two
# surfaces computed different numbers from the same database.
from ait.reporting.go_live import (  # noqa: E402
    compute_go_live_verdict,
    format_pace_line,
    format_verdict_lines,
    not_real_close_sql,
    open_trade_status_sql,
)

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
    """W6/log-contracts-3: crashes land in logs/fatal.log (the faulthandler
    sink), NOT bot_stdout.log — this reported 0 while tracebacks piled up.
    ops_health also distinguishes "no crashes" from "nothing records crashes"."""
    from ait.monitoring.ops_health import count_native_crashes
    return count_native_crashes(LOGS / "fatal.log", (LOGS / "bot_stdout.log",))


def _keeper_relaunches() -> int:
    return sum(1 for ln in _tail_lines(LOGS / "keeper.log", 5000)
               if "DOWN - relaunching" in ln)


def _readonly_recent() -> bool:
    for ln in _tail_lines(LOGS / "bot_stdout.log", 400):
        if "Error 321" in ln and "Read-Only" in ln:
            return True
    return False


def _count_event_today(event: str) -> int:
    """W6/log-contracts-1 + time-authority-1: a fixed tail window missed
    events rotated away (reporting 0 activity for a bot that predicted 112
    times), and substring-matching a LOCAL date against UTC stamps bucketed
    the wrong day. ops_health walks today's rotated backups and buckets by
    local date."""
    from ait.monitoring.ops_health import count_events_today
    return count_events_today(LOGS / "ait.log", (event,)).counts.get(event, 0)


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
            # W3 db-contracts-4: "NOT IN ('closed')" also admitted the
            # TERMINAL statuses cancelled/rejected — three IWM trades
            # cancelled in July showed here (and on the web dashboard) as
            # open at $0 for over a month, reporting 5 open positions on a
            # 2-position book.  Derive from the authority instead:
            # ait.bot.state.TradeStatus's genuinely-open values.
            open_sql = open_trade_status_sql("t.status")
            rows = con.execute(
                "SELECT t.symbol, t.strategy, t.status, t.entry_price, t.entry_time, "
                "       op.unrealized_pnl, op.pnl_pct, op.mark_time "
                "FROM trades t LEFT JOIN open_positions op ON op.trade_id = t.trade_id "
                f"WHERE {open_sql} ORDER BY t.entry_time DESC").fetchall()
        except sqlite3.OperationalError:
            # Pre-migration DB (pnl_pct/mark_time added by the bot's
            # StateManager on first start of new code) — degrade gracefully.
            rows = con.execute(
                "SELECT t.symbol, t.strategy, t.status, t.entry_price, t.entry_time, "
                "       op.unrealized_pnl, 0 AS pnl_pct, '' AS mark_time "
                "FROM trades t LEFT JOIN open_positions op ON op.trade_id = t.trade_id "
                f"WHERE {open_trade_status_sql('t.status')} "
                "ORDER BY t.entry_time DESC").fetchall()
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
        # R17: was missing COALESCE, so any row with a NULL exit_reason_detailed
        # (SQL NULL NOT LIKE x -> NULL -> dropped by WHERE) silently fell out
        # of pnl_today/pnl_life -- unlike the go-live scorecard query below,
        # which already had it. Both the CLI and status_server.py's web
        # dashboard read this via get_status().
        # W3 string-contracts-1/-4: the hand-rolled trio is gone; the shared
        # authority also excludes the reconciler's $0 "needs manual review"
        # sentinels, which passed the old filter and booked as real losses.
        real = not_real_close_sql()
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
    # R7/W3 go-live scorecard: all FIVE of gate 1's criteria, every one
    # either computed AS PINNED or printed UNAVAILABLE with a reason.
    # R19d (user decision 2026-08-20): the VERDICT METRIC is IRON CONDOR
    # closes only. The mission (PLAN line 3) is the IC edge question; the
    # retired experiments (straddle -575, long calls -132, strangles +378)
    # answer nothing about it and were drowning the signal - the mixed
    # record read -$451 at its worst while the condor itself was positive.
    # The all-strategy line is kept underneath for book-level honesty.
    # W3: the numbers come from ait.reporting.go_live, the SAME function the
    # scheduled Friday Telegram scorecard calls - the two surfaces can no
    # longer disagree about the go/no-go number.
    print()
    try:
        v = compute_go_live_verdict(DB)
        for line in format_verdict_lines(v, indent="  "):
            print(line)
        print(format_pace_line(v, indent="  "))
    except Exception as e:  # noqa: BLE001
        # W3 policy-vs-impl-5: this block used to be `except Exception: pass`,
        # so a schema change (or a stale import) silently DELETED the entire
        # gate readout and the operator saw a status page with no gates on
        # it at all. Fail LOUD instead.
        print(f"  !! GATE READOUT FAILED: {type(e).__name__}: {e}")
        print("  !! the go-live verdict is MISSING, not passing - do not "
              "read its absence as green.")
    print("\n" + "=" * 56)


if __name__ == "__main__":
    main()
