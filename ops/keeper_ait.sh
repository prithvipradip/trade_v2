#!/bin/bash
# AIT supervisor keeper — POSIX sibling of keeper_ait.bat (W16 B1, R25 plat-05).
#
# WHY THIS EXISTS
#   keeper_ait.bat is BOTH the ~90s supervisor relauncher AND the gate on the
#   healthchecks.io dead-man ping. It is cmd.exe-only, and `find . -name "*.sh"
#   -o -name "*.plist"` returned ZERO before this file: on macOS nothing
#   relaunched a dead supervisor and the dead-man could never fire. On
#   2026-09-15 the supervisor was found running out of a VS Code integrated
#   terminal (parent chain: zsh <- Code Helper <- Visual Studio Code), i.e.
#   quitting the editor would have killed the trading bot.
#
# THIS FILE IS ADDITIVE. keeper_ait.bat must stay byte-stable —
# tests/test_r16_tail_fixes.py::TestKeeperBat and
# tests/test_w6_ops_surfaces.py assert its exact text. This mirrors its
# BEHAVIOUR, not its text.
#
# DELIBERATELY NOT REIMPLEMENTED (all already exist and are cross-platform):
#   * double-launch safety — master.py::_acquire_singleton_lock (fcntl/msvcrt);
#     the check below is advisory only, the flock is the guard.
#   * keeper.log rotation — master.py::_ROTATE_TARGETS already caps this file
#     at 20MB with 2 backups on a daily sweep.
#
# The 90s sleep here is a plain `sleep`: the .bat's three-fallback ping/timeout
# /Start-Sleep dance exists only because `timeout /t` needs an interactive
# console handle under cmd.exe, which once spun that loop at 1-2 iterations/sec.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PY="${AIT_PYTHON:-python3}"
LOG="$ROOT/logs/keeper.log"
DEADMAN_FILE="$ROOT/data/deadman_url.txt"
INTERVAL="${AIT_KEEPER_INTERVAL:-90}"

mkdir -p "$ROOT/logs"

log() { echo "[keeper] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# 0 = a supervisor or bot process is alive, 1 = none, 2 = table unreadable.
alive_check() {
  "$PY" - <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, "src")
try:
    from ait.utils.process import scan_pids
except Exception:
    sys.exit(2)
orch = scan_pids(script="run_orchestrator.py")
bot = scan_pids(module="ait.main")
if orch is None or bot is None:
    sys.exit(2)
sys.exit(0 if (orch or bot) else 1)
PYEOF
}

log "keeper started (root=$ROOT py=$PY interval=${INTERVAL}s)"

while true; do
  alive_check
  case $? in
    0) state="alive" ;;
    1) state="down" ;;
    *) state="unknown" ;;
  esac

  if [ "$state" = "down" ]; then
    log "orchestrator DOWN - relaunching"
    nohup "$PY" run_orchestrator.py >/dev/null 2>&1 &
    # No ping: the relaunch is not evidence of health, and silence is exactly
    # what trips healthchecks.io. Same contract as the .bat.

  elif [ "$state" = "unknown" ]; then
    # "I could not look" is not "it is down". Relaunching blind would thrash
    # (and page, now that the singleton lock fails closed), so hold — and
    # send NO ping, so a persistent unknown still surfaces as silence.
    log "process table unreadable - not relaunching, withholding ping"

  else
    # ALIVE: ping only when liveness can be positively demonstrated.
    # Gate = exit code of `ops_health liveness` (0 alive / >=1 not), the same
    # 900s heartbeat threshold master.py uses for bot_hung_heartbeat_stale.
    if [ -f "$DEADMAN_FILE" ]; then
      URL="$(head -n1 "$DEADMAN_FILE" | tr -d '[:space:]')"
      if [ -n "$URL" ]; then
        if "$PY" -m ait.monitoring.ops_health liveness >/dev/null 2>&1; then
          curl -fsS -m 10 "$URL" >/dev/null 2>&1
        else
          curl -fsS -m 10 "$URL/fail" >/dev/null 2>&1
          log "deadman: bot NOT demonstrably alive - sent /fail"
        fi
      fi
    fi
  fi

  sleep "$INTERVAL"
done
