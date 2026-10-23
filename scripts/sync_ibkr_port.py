#!/usr/bin/env python3
"""W94: make .env's IBKR_PORT agree with the IB endpoint that is actually running.

    python scripts/sync_ibkr_port.py            # dry run: show configured vs what is listening
    python scripts/sync_ibkr_port.py --apply    # rewrite ONLY the IBKR_PORT value in .env

Why: the bot and the supervisor now fall back to the other endpoint of the SAME account type
(paper 7497 <-> 4002, live 7496 <-> 4001) when the configured one is down, but they never edit
.env. This is the operator-run one-liner that fixes the file once you see the warning
(`ib_port_fallback`, the status page's "IB endpoint" row, or the digest line).

Safety: only the number after `IBKR_PORT=` changes (the trailing comment, every other line and
the line endings stay byte-identical); the write is atomic (temp file + rename, same mode); a
target outside the configured port's twin pair is refused, so paper can never become live; the
output never prints any other line of .env (it holds credentials). Restart the supervisor
afterwards so it re-reads the file.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ait.data.ib_ports import effective_port, endpoint_label, port_candidates, tcp_listening  # noqa: E402

_LINE = re.compile(r"^(IBKR_PORT=)(\d+)(.*)$", re.DOTALL)


def find_port_line(lines: list[str]) -> tuple[int, int] | None:
    """(line index, current port) of the first `IBKR_PORT=<digits>` line, else None."""
    for i, ln in enumerate(lines):
        m = _LINE.match(ln)
        if m:
            return i, int(m.group(2))
    return None


def rewrite_port(text: str, new_port: int) -> str:
    """`text` with only the IBKR_PORT value replaced (everything else byte-identical)."""
    lines = text.splitlines(keepends=True)
    found = find_port_line(lines)
    if found is None:
        raise ValueError("no IBKR_PORT=<number> line in the file")
    i, _ = found
    m = _LINE.match(lines[i])
    lines[i] = f"{m.group(1)}{int(new_port)}{m.group(3)}"
    return "".join(lines)


def main(argv: list[str] | None = None, *, listening=tcp_listening) -> int:
    ap = argparse.ArgumentParser(description="Sync .env IBKR_PORT with the running IB endpoint.")
    ap.add_argument("--env-file", default=str(ROOT / ".env"), metavar="PATH")
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    ap.add_argument("--port", type=int, default=None,
                    help="target port (default: the endpoint that is listening); must be the "
                         "configured port's same-account-type twin")
    args = ap.parse_args(argv)

    path = Path(args.env_file)
    try:
        with open(path, encoding="utf-8", newline="") as f:   # newline="": keep CRLF as it is
            text = f.read()
    except OSError as e:
        print(f"ERROR: cannot read {path}: {e}")
        return 2
    found = find_port_line(text.splitlines(keepends=True))
    if found is None:
        print("ERROR: no IBKR_PORT=<number> line found; nothing to sync")
        return 2
    configured = found[1]

    target = args.port if args.port is not None else effective_port(configured, listening)
    if target not in port_candidates(configured):
        print(f"REFUSED: {target} is not an endpoint of the same account type as "
              f"{configured} ({endpoint_label(configured)}); allowed: {port_candidates(configured)}")
        return 3
    if target == configured:
        up = listening(configured)
        print(f"IBKR_PORT={configured} ({endpoint_label(configured)}): "
              + ("in sync, nothing to change" if up else "nothing is listening on it or its twin; "
                 "start TWS / IB Gateway (no change made)"))
        return 0

    print(f"IBKR_PORT={configured} ({endpoint_label(configured)}) -> {target} "
          f"({endpoint_label(target)})")
    if not args.apply:
        print("dry run: nothing written (pass --apply to change .env)")
        return 0

    new_text = rewrite_port(text, target)
    tmp = path.with_name(path.name + ".sync_tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    print(f"updated {path.name}: IBKR_PORT is now {target}. Restart the supervisor "
          f"(python run_orchestrator.py) so it re-reads the file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
