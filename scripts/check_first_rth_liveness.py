"""Deploy-checklist step 4, automated: first-RTH liveness after a deploy.

Run ~09:40 ET on the first trading morning after a deploy (RUNBOOK step 4).
Checks, all read-only:
  1. data/bot_heartbeat fresh (<10 min)
  2. scan_symbol_timing events logged after today's open (the R11 timing
     event fires at the end of each symbol scan — zero after open means the
     scan loop is not completing; note: entry gates like economic_event_skip
     legitimately suppress it on event days)
  3. no LOOP IMPAIRED / trading_cycle_error today
  4. open_positions marks fresh (<10 min), when positions exist
  5. entries_frozen state (informational — expected while HALT* files exist)

Sends the verdict to Telegram via the same creds the supervisor uses
(api_keys in .env), prints it, exits 1 on any FAIL. --no-telegram to skip
the send (e.g. dry runs).
"""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAIL_BYTES = 4 << 20  # last 4MB of ait.log
FRESH_SECS = 600


def _log_tail() -> str:
    p = ROOT / "logs" / "ait.log"
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - TAIL_BYTES))
        return f.read().decode("utf-8", errors="ignore")


def _mark_age_seconds(newest_iso: str, now: datetime | None = None) -> float:
    """Age of the newest position mark, in the codebase's ET convention.

    R17: was a hardcoded EDT (UTC-4) offset -- wrong for ~4-5 months a year
    (EST is UTC-5), guaranteeing a false LIVENESS FAIL every winter.
    `now` is injectable for tests; defaults to the real current time.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from ait.utils.time import ET

    mark_dt = datetime.fromisoformat(newest_iso).replace(tzinfo=ET)
    return ((now or datetime.now(ET)) - mark_dt).total_seconds()


def main() -> int:
    results: list[tuple[str, bool | None, str]] = []  # (name, ok/None=info, msg)
    now = time.time()
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    open_utc = f"{today_utc}T13:3"  # 13:30Z..13:39Z prefix window start

    hb = ROOT / "data" / "bot_heartbeat"
    age = (now - hb.stat().st_mtime) if hb.exists() else 1e9
    results.append(("heartbeat", age < FRESH_SECS,
                    f"{int(age)}s old" if hb.exists() else "MISSING"))

    tail = _log_tail()
    scans = [m for m in re.finditer(
        r'"event": "scan_symbol_timing".*?"timestamp": "([^"]+)"', tail)
        if m.group(1) >= f"{today_utc}T13:30"]
    results.append(("scan_symbol_timing", len(scans) > 0,
                    f"{len(scans)} since open" if scans else
                    "ZERO since open (gate day? check economic_event_skip)"))

    impaired = len(re.findall(r"LOOP IMPAIRED", tail))
    cycle_err = len([m for m in re.finditer(
        r'"event": "trading_cycle_error".*?"timestamp": "([^"]+)"', tail)
        if m.group(1).startswith(today_utc)])
    results.append(("no_loop_impairment", impaired == 0 and cycle_err == 0,
                    f"LOOP IMPAIRED x{impaired}, trading_cycle_error x{cycle_err} today"))

    try:
        con = sqlite3.connect(f"file:{(ROOT / 'data' / 'ait_state.db').as_posix()}?mode=ro",
                              uri=True)
        rows = con.execute(
            "SELECT symbol, mark_time FROM open_positions").fetchall()
        con.close()
        if rows:
            newest = max(r[1] or "" for r in rows)
            mage = _mark_age_seconds(newest)
            results.append(("marks_fresh", mage < FRESH_SECS,
                            f"{len(rows)} position(s), newest mark {int(mage)}s old"))
        else:
            results.append(("marks_fresh", None, "no open positions"))
    except Exception as e:  # noqa: BLE001
        results.append(("marks_fresh", False, f"DB check failed: {e}"))

    halts = [p.name for p in (ROOT / "data").glob("HALT*") if p.is_file()]
    frozen = len(re.findall(r'"event": "entries_frozen"', tail))
    results.append(("entry_freeze_state", None,
                    f"halt files: {halts or 'none'}; entries_frozen logged x{frozen}"))

    fails = [r for r in results if r[1] is False]
    lines = [f"[{'ok' if ok else 'INFO' if ok is None else 'FAIL'}] {name}: {msg}"
             for name, ok, msg in results]
    verdict = ("LIVENESS PASS" if not fails else
               f"LIVENESS FAIL ({len(fails)}: {', '.join(f[0] for f in fails)})")
    report = f"POST-DEPLOY {verdict}\n" + "\n".join(lines)
    print(report)

    if "--no-telegram" not in sys.argv:
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from ait.config.settings import load_settings
            s = load_settings(str(ROOT / "config.yaml"))
            token = s.api_keys.telegram_bot_token
            chat = s.api_keys.telegram_chat_id
            if token and chat:
                import urllib.parse
                import urllib.request
                data = urllib.parse.urlencode(
                    {"chat_id": chat, "text": report,
                     "disable_web_page_preview": "true"}).encode()
                urllib.request.urlopen(urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data=data), timeout=10)
                print("(sent to Telegram)")
        except Exception as e:  # noqa: BLE001
            print("telegram send failed:",
                  re.sub(r"/bot[^/\s]+", "/bot***", str(e))[:200])

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
