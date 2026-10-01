"""Publish this box's bot events to the `shared-logs` branch.

    python scripts/publish_shared_logs.py                 # today + last 4 weekdays (catch-up)
    python scripts/publish_shared_logs.py --date 2026-09-28
    python scripts/publish_shared_logs.py --dry-run        # export locally, no git
    python scripts/publish_shared_logs.py --logs-dir ../trade_v2/logs   # read another tree's logs

The owner comes from AIT_LOG_OWNER (set once per machine in .env). Run by the
supervisor at 16:45 ET on weekdays; safe to run by hand. See
src/ait/orchestration/shared_logs.py for the design.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.config.runtime_env import _load_dotenv  # noqa: E402
from ait.orchestration import shared_logs as sl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="one ET date (YYYY-MM-DD); default: today + catch-up")
    ap.add_argument("--dry-run", action="store_true", help="export to reports/, no git")
    ap.add_argument("--logs-dir", default=str(ROOT / "logs"),
                    help="raw logs to read (read-only); default: this checkout's logs/. "
                         "Lets a dev checkout publish a live tree's logs without touching it.")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    owner = sl.resolve_owner()
    if not owner:
        print(json.dumps({"ok": False, "error": f"{sl.OWNER_ENV} not set (or invalid) in .env — publishing is off"}))
        return 0
    days = ([date.fromisoformat(args.date)] if args.date
            else sl.recent_trading_days(datetime.now(sl.ET).date()))
    if args.dry_run:
        out = []
        for d in days:
            dest = ROOT / "reports" / "shared_logs_dryrun" / f"{owner}_logs" / f"{d}.jsonl.gz"
            out.append(sl.export_day(Path(args.logs_dir), d, dest, owner))
        print(json.dumps({"ok": True, "dry_run": out}, indent=1))
        return 0
    res = sl.publish(ROOT, owner, days, logs_dir=Path(args.logs_dir))
    print(json.dumps(res.__dict__))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
