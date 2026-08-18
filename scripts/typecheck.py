#!/usr/bin/env python3
"""Blocking type gate for the four packages where crash-class bugs live.

WHY THIS EXISTS
---------------
`self._settings.trading.strategies` shipped live: `strategies` is a field of
OptionsConfig, not TradingConfig. It raised AttributeError inside the trading
cycle 595 times over 3 days. The config models are fully-typed pydantic, so
mypy reports that exact access as [attr-defined] -- at commit time, for free.

WHY NOT JUST `mypy src`
-----------------------
When this was written `mypy src` reported 222 errors across 40 modules
(mostly [import-untyped] for pandas/yaml/psutil stubs, [no-any-return] and
[arg-type]). A `mypy src` gate would be permanently red, i.e. permanently
ignored -- the same reason the ruff step is still commented out in ci.yml.

WHAT THIS DOES INSTEAD
----------------------
Runs mypy over src/ait/{config,bot,execution,risk} and fails ONLY on the
"this name / attribute / argument does not exist" family:

    attr-defined  -- obj.typo, wrong config section  <-- the shipped bug
    name-defined  -- undefined name (dead import, moved symbol)
    call-arg      -- wrong/missing keyword argument name

Those are typo and refactor-rot errors with a near-zero false-positive rate on
this codebase (0 occurrences today, after two justified `# type: ignore`s).
Every other error code is printed as a tally and does NOT fail the build, so
new stub noise or an Optional-narrowing nit can never block a hotfix. CI also
runs a separate NON-BLOCKING `mypy src` step for full visibility.

Adding a code to FATAL_CODES is the intended way to ratchet this up as the
debt comes down.

Usage
-----
    python scripts/typecheck.py                 # gate the default paths
    python scripts/typecheck.py path/to/file.py # gate specific paths
    python scripts/typecheck.py --strict-all    # every code is fatal (local)

Set MYPY_CACHE_DIR to relocate mypy's cache (mypy reads it natively).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Highest-value paths: the trading cycle, order placement, risk gates, and the
# pydantic settings models they all read.
GATED_PATHS: tuple[str, ...] = (
    "src/ait/config",
    "src/ait/bot",
    "src/ait/execution",
    "src/ait/risk",
)

FATAL_CODES = frozenset({"attr-defined", "name-defined", "call-arg"})

# mypy line format: path:line:col: error: message  [code]
_ERROR_RE = re.compile(
    r"^(?P<loc>.+?):(?P<line>\d+):(?:\d+:)? error: (?P<msg>.*?)"
    r"(?:\s+\[(?P<code>[a-z][a-z0-9-]*)\])?$"
)


def run_mypy(paths: list[str]) -> tuple[int, list[str]]:
    """Return (mypy exit code, stdout+stderr lines)."""
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        *paths,
        # Errors in modules we do not gate (ml/, data/, backtesting/, ...) are
        # still type-checked for their signatures but not reported.
        "--follow-imports=silent",
        "--no-error-summary",
        "--no-color-output",
        "--show-error-codes",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="paths to check (default: gated set)")
    parser.add_argument(
        "--strict-all",
        action="store_true",
        help="treat every mypy error code as fatal (for local cleanup work)",
    )
    args = parser.parse_args(argv)

    paths = args.paths or list(GATED_PATHS)
    code, lines = run_mypy(paths)

    fatal: list[str] = []
    tolerated: Counter[str] = Counter()
    for line in lines:
        m = _ERROR_RE.match(line.rstrip())
        if not m:
            continue
        err_code = m.group("code") or "no-code"
        if args.strict_all or err_code in FATAL_CODES:
            fatal.append(line.rstrip())
        else:
            tolerated[err_code] += 1

    # mypy exits 2 on a crash / bad invocation. Never report that as clean.
    if code >= 2 and not fatal:
        print("mypy failed to run (exit %d):" % code, file=sys.stderr)
        print("\n".join(lines[-20:]), file=sys.stderr)
        return 2

    if tolerated:
        tally = ", ".join(f"{c}={n}" for c, n in tolerated.most_common())
        print(f"non-blocking mypy findings in gated paths: {tally}")

    if fatal:
        sys.stdout.flush()  # keep the tally above the blocking list in CI logs
        label = "error" if args.strict_all else "/".join(sorted(FATAL_CODES))
        print(f"\nBLOCKING [{label}] in {', '.join(paths)}:\n", file=sys.stderr)
        for line in fatal:
            print("  " + line, file=sys.stderr)
        print(
            f"\n{len(fatal)} blocking error(s). These are 'the name/attribute/"
            "argument does not exist' -- they crash at runtime, not in review.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no [{'/'.join(sorted(FATAL_CODES))}] errors in {', '.join(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
