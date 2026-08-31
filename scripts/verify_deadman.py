#!/usr/bin/env python
"""Verify the dead-man ping end-to-end BEFORE the real URL is armed.

R16 CRITICAL finding: data/deadman_url.txt has never existed, so the only
alert channel that survives whole-machine death has been inert since it was
built (2026-07-09) — through five market-hours outages with live positions.

This proves the plumbing works, so arming it is a pure copy-paste:
  1. Create a free check at https://healthchecks.io  (Period 5 min, Grace 15 min)
  2. Copy its ping URL (looks like https://hc-ping.com/<uuid>)
  3. Save it as the ONLY line of data/deadman_url.txt
  4. Run this script again — it should report ARMED + a successful ping.

W6 bot-day-02 — THE PING MUST ATTEST THE BOT, NOT THE SUPERVISOR.
keeper_ait.bat pings whenever a process-table regex matches 'run_orchestrator'
or 'ait.main', regardless of whether the bot is trading.  All three sustained
bot-down-with-master-alive states are real (master.py:575-584 gateway-down
deferral, :595-602 give-up, :432-443 relaunch loop), so a Telegram-dead +
gateway-down outage produces a green external monitor for hours with live
positions and no exit engine running.

This script therefore:
  * evaluates ait.monitoring.ops_health.bot_liveness() and REFUSES to send a
    success ping when the bot is not demonstrably alive — it sends the
    healthchecks.io /fail endpoint instead, so the outage surfaces immediately;
  * checks whether keeper_ait.bat gates its own ping the same way, and prints
    the exact lines to add when it does not.

Usage: python scripts/verify_deadman.py [--no-ping]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.monitoring.ops_health import bot_liveness  # noqa: E402

URL_FILE = ROOT / "data" / "deadman_url.txt"
KEEPER = ROOT / "keeper_ait.bat"

#: The marker that proves keeper_ait.bat consults the liveness gate.
LIVENESS_MODULE = "ait.monitoring.ops_health"

KEEPER_PATCH = """\
    REM W6 bot-day-02: gate the dead-man on EVIDENCE the bot is trading, not on
    REM a process-table match. Exit 0 = alive (ping), 1 = not alive (ping /fail).
    %PY% -m ait.monitoring.ops_health liveness >nul 2>&1
    if errorlevel 1 (
        echo [keeper] %date% %time% bot NOT LIVE - failing the dead-man >> logs\\keeper.log
        if exist data\\deadman_url.txt (
            for /f "usebackq delims=" %%u in ("data\\deadman_url.txt") do curl.exe -fsS -m 10 "%%u/fail" >nul 2>&1
        )
    ) else (
        if exist data\\deadman_url.txt (
            for /f "usebackq delims=" %%u in ("data\\deadman_url.txt") do curl.exe -fsS -m 10 "%%u" >nul 2>&1
        )
    )"""


def read_deadman_url(path: Path = URL_FILE) -> str | None:
    """First non-empty line of the dead-man URL file, or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def keeper_gating(keeper_text: str) -> dict:
    """Does keeper_ait.bat ping, and is that ping gated on bot liveness?

    Returns ``{"pings", "gated", "detail"}``.  ``gated`` False is the
    bot-day-02 defect: a green monitor that only attests the supervisor.
    """
    pings = "deadman_url.txt" in keeper_text and "curl.exe" in keeper_text
    gated = LIVENESS_MODULE in keeper_text
    if not pings:
        detail = "keeper does not ping at all — the dead-man is not plumbed."
    elif gated:
        detail = ("keeper gates the ping on "
                  f"`python -m {LIVENESS_MODULE} liveness` — the ping attests "
                  "the BOT.")
    else:
        detail = (
            "keeper pings on a PROCESS-TABLE MATCH only (bot-day-02): a "
            "master-alive/bot-dead outage — gateway-down deferral, restart "
            "give-up, or a relaunch loop — pings GREEN for hours with live "
            "positions and no exit engine. Patch keeper_ait.bat's alive branch."
        )
    return {"pings": pings, "gated": gated, "detail": detail}


def ping_target(url: str, ok: bool) -> str:
    """healthchecks.io URL to hit: the check itself, or its /fail endpoint."""
    return url.rstrip("/") if ok else url.rstrip("/") + "/fail"


def _curl(curl: str, url: str) -> int:
    return subprocess.run([curl, "-fsS", "-m", "10", url],
                          capture_output=True, text=True).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    send = "--no-ping" not in argv

    curl = shutil.which("curl") or shutil.which("curl.exe")
    print(f"curl available      : {curl or 'NO — keeper ping cannot work'}")
    if not curl:
        return 1

    keeper_text = ""
    try:
        keeper_text = KEEPER.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    gating = keeper_gating(keeper_text)
    print(f"keeper wired to ping: {gating['pings']}")
    print(f"ping gated on bot   : {gating['gated']}")
    print(f"  -> {gating['detail']}")
    if gating["pings"] and not gating["gated"]:
        print("\nReplace keeper_ait.bat's alive-branch ping block with:\n")
        print(KEEPER_PATCH)
        print()

    verdict = bot_liveness()
    print(f"bot liveness        : {'ALIVE' if verdict.ok else 'NOT ALIVE'} "
          f"[{verdict.state}]")
    print(f"  -> {verdict.detail}")

    # Pass URL_FILE explicitly: a default argument is bound at def time,
    # so reading it here is what makes the path overridable.
    url = read_deadman_url(URL_FILE)
    if url is None:
        print(f"deadman file        : MISSING ({URL_FILE})")
        print("\nSTATUS: NOT ARMED — the bot can die silently, as it did on")
        print("2026-08-05 and 08-06 with two live condors unmanaged.")
        print("Follow steps 1-4 in this file's docstring to arm it.")
        if send:
            # prove the transport works against a public echo endpoint
            rc = _curl(curl, "https://hc-ping.com/ping-test")
            print(f"\ntransport self-test : exit={rc} "
                  f"({'reachable' if rc in (0, 22) else 'NETWORK PROBLEM'})")
        return 2

    if not url.startswith("http"):
        print(f"deadman file        : PRESENT but content is not a URL: {url[:40]!r}")
        return 1
    print(f"deadman file        : PRESENT ({url[:28]}...)")

    if not send:
        print("\nSTATUS: --no-ping, nothing sent.")
        return 0 if verdict.ok else 3

    target = ping_target(url, verdict.ok)
    rc = _curl(curl, target)
    ok = rc == 0
    kind = "success ping" if verdict.ok else "FAIL ping (bot not alive)"
    print(f"live ping           : {kind} -> {'OK' if ok else f'FAILED rc={rc}'}")

    if not ok:
        print("\nSTATUS: file present but the ping FAILED — check the URL.")
        return 1
    if not verdict.ok:
        # Sending a success ping here is exactly the bug: it would paper over
        # a live outage on the one monitor built to survive dead alert channels.
        print("\nSTATUS: ARMED, and DELIBERATELY FAILED the check — the bot is "
              "not demonstrably alive, so a green ping would be a lie.")
        return 3
    print("\nSTATUS: ARMED — healthchecks.io will alert you when pings stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
