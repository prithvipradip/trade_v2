"""Verify the dead-man ping end-to-end BEFORE the real URL is armed.

R16 CRITICAL finding: data/deadman_url.txt has never existed, so the only
alert channel that survives whole-machine death has been inert since it was
built (2026-07-09) — through five market-hours outages with live positions.

This proves the plumbing works, so arming it is a pure copy-paste:
  1. Create a free check at https://healthchecks.io  (Period 5 min, Grace 15 min)
  2. Copy its ping URL (looks like https://hc-ping.com/<uuid>)
  3. Save it as the ONLY line of data/deadman_url.txt
  4. Run this script again — it should report ARMED + a successful ping.

Usage: python scripts/verify_deadman.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL_FILE = ROOT / "data" / "deadman_url.txt"


def main() -> int:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    print(f"curl available      : {curl or 'NO — keeper ping cannot work'}")
    if not curl:
        return 1

    keeper = (ROOT / "keeper_ait.bat").read_text(encoding="utf-8", errors="ignore")
    wired = "deadman_url.txt" in keeper and "curl.exe" in keeper
    print(f"keeper wired to ping: {wired}")

    if not URL_FILE.exists():
        print(f"deadman file        : MISSING ({URL_FILE})")
        print("\nSTATUS: NOT ARMED — the bot can die silently, as it did on")
        print("2026-08-05 and 08-06 with two live condors unmanaged.")
        print("Follow steps 1-4 in this file's docstring to arm it.")
        # prove the transport works against a public echo endpoint
        r = subprocess.run([curl, "-fsS", "-m", "10", "https://hc-ping.com/ping-test"],
                           capture_output=True, text=True)
        print(f"\ntransport self-test : exit={r.returncode} "
              f"({'reachable' if r.returncode in (0, 22) else 'NETWORK PROBLEM'})")
        return 2

    url = URL_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not url.startswith("http"):
        print(f"deadman file        : PRESENT but content is not a URL: {url[:40]!r}")
        return 1
    print(f"deadman file        : PRESENT ({url[:28]}...)")
    r = subprocess.run([curl, "-fsS", "-m", "10", url],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"live ping           : {'OK' if ok else f'FAILED rc={r.returncode}'}")
    print("\nSTATUS: ARMED — healthchecks.io will alert you when pings stop."
          if ok else "\nSTATUS: file present but the ping FAILED — check the URL.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
