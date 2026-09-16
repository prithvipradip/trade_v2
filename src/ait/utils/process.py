"""Cross-platform process-table queries.

R25 (W16 A3): this scan existed in FOUR places — `master.py._scan_foreign_bot_pids`
(psutil, made cross-platform 2026-09-14), `status.py._proc_running` (PowerShell,
still Windows-only), and both `.bat` keepers. The 2026-09-14 fix was applied to
one call site rather than to the idiom, so `status.py` kept reporting the bot
DOWN while it was UP on any non-Windows host.

Two deliberate design choices, both reactions to R25 findings:

1. **A failed scan returns ``None``, never ``[]``.** "I could not look" and "I
   looked and found nothing" are different facts, and collapsing them is exactly
   how `_scan_foreign_bot_pids` became a silent no-op on macOS. Callers must
   choose their own fail direction explicitly at the call site.

2. **Matching is SHAPE-aware, not substring.**
   `"ait.main" in " ".join(cmdline)` also matches `grep -r ait.main src/`, an
   editor with the file open, or an audit session — and the consumer of that
   answer REFUSES to start the trading bot, so a false positive is fail-closed
   and costs a trading session.

   A whole-argv token is not enough either: in ``grep -r ait.main src/`` the
   token *is* its own element. (A W16 test caught exactly that, against an
   earlier version of this module that claimed otherwise.) So the matchers
   model how the real processes are actually spawned:

   * ``module="ait.main"`` — requires ``-m`` IMMEDIATELY followed by the module,
     i.e. ``[sys.executable, "-m", "ait.main", ...]``.
   * ``script="run_orchestrator.py"`` — requires the script as its own argv
     element AND argv[0] to look like a Python interpreter.

   ``contains`` and ``token`` remain for loose callers, clearly labelled as such.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


def _is_python(argv0: str) -> bool:
    return "python" in os.path.basename(argv0 or "").lower()


def scan_pids(
    *,
    module: str | None = None,
    script: str | None = None,
    token: str | None = None,
    contains: str | None = None,
    name_contains: str | None = None,
    exclude_pids: Iterable[int] | None = None,
) -> list[int] | None:
    """PIDs whose command line or executable name matches.

    ``module``       — ``python -m <module>`` (precise; ``-m`` adjacency).
    ``script``       — ``python <script>`` (precise; argv[0] must be python).
    ``token``        — one whole argv element (loose: a grep argument matches).
    ``contains``     — substring of the joined command line (loosest).
    ``name_contains``— substring of the process NAME, case-insensitive
                       (e.g. "java" for a TWS/Gateway JVM).
    Any given criterion must match; criteria are OR-ed.

    Returns the matching PIDs, or ``None`` if the process table could not be
    read at all. ``None`` is not an error to swallow — it means the answer is
    unknown, and the caller must decide what that implies.
    """
    excluded = set(exclude_pids or ())
    excluded.add(os.getpid())
    pids: list[int] = []
    try:
        import psutil

        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = p.info["pid"]
                if pid in excluded:
                    continue
                cmdline = p.info.get("cmdline") or []
                name = p.info.get("name") or ""
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if module is not None and any(
                    cmdline[i] == "-m" and cmdline[i + 1] == module
                    for i in range(len(cmdline) - 1)):
                pids.append(pid)
            elif script is not None and script in cmdline[1:] and _is_python(
                    cmdline[0] if cmdline else ""):
                pids.append(pid)
            elif token is not None and token in cmdline:
                pids.append(pid)
            elif contains is not None and contains in " ".join(cmdline):
                pids.append(pid)
            elif name_contains is not None and name_contains.lower() in name.lower():
                pids.append(pid)
    except Exception:
        return None
    return pids
