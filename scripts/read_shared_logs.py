"""Read a box's published bot events from the `shared-logs` branch.

    python scripts/read_shared_logs.py --list                       # who published which days
    python scripts/read_shared_logs.py --owner ahmed --date 2026-09-28
    python scripts/read_shared_logs.py --owner ahmed --date 2026-09-28 --event condor_rejected_iv_rank_floor
    python scripts/read_shared_logs.py --owner ahmed --date 2026-09-28 --symbol QQQ --limit 50

Reads straight from `origin/shared-logs` after a fetch — no checkout needed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.orchestration import shared_logs as sl  # noqa: E402

REF = f"origin/{sl.SHARED_BRANCH}"


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode("utf-8", "replace").strip() or f"git {args} failed")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--date")
    ap.add_argument("--event", help="print only events with this name")
    ap.add_argument("--symbol")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    if not args.no_fetch:
        _git("fetch", "origin", sl.SHARED_BRANCH)
    if args.list or not (args.owner and args.date):
        for path in sorted(_git("ls-tree", "-r", "--name-only", REF).split()):
            if path.endswith(".jsonl.gz"):
                print(path)
        return 0
    data = _git("show", f"{REF}:{args.owner}_logs/{args.date}.jsonl.gz", binary=True)
    meta, events = sl.read_day(data)
    if args.event or args.symbol:
        shown = [e for e in events
                 if (not args.event or e.get("event") == args.event)
                 and (not args.symbol or e.get("symbol") == args.symbol)]
        for e in shown[: args.limit]:
            print(json.dumps(e))
        print(f"-- {len(shown)} matching of {len(events)} events", file=sys.stderr)
        return 0
    print(json.dumps({"meta": meta, "summary": sl.summarize(events)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
