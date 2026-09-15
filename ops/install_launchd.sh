#!/bin/bash
# Install the AIT keeper as a launchd LaunchAgent (W16 B1).
#   ops/install_launchd.sh            install + start
#   ops/install_launchd.sh --uninstall
#
# A LaunchAgent (not a Daemon) because the bot needs the user's GUI session:
# TWS runs there, and IB's login/2FA prompt is a window a human must see.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.ait.keeper"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "--uninstall" ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$TARGET" 2>/dev/null || true
    rm -f "$TARGET"
    echo "uninstalled $LABEL"
    exit 0
fi

PY="${AIT_PYTHON:-$(command -v python3)}"
[ -x "$PY" ] || { echo "no python3 found; set AIT_PYTHON" >&2; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__ROOT__|$ROOT|g" -e "s|__PYTHON__|$PY|g" \
    "$ROOT/ops/com.ait.keeper.plist.template" > "$TARGET"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET" 2>/dev/null || launchctl load "$TARGET"

echo "installed $TARGET"
echo "  root   : $ROOT"
echo "  python : $PY"
echo
echo "verify:  launchctl print gui/$(id -u)/$LABEL | head -20"
echo "logs  :  tail -f $ROOT/logs/keeper.log"
[ -f "$ROOT/data/deadman_url.txt" ] || cat <<'WARN'

NOTE: data/deadman_url.txt does not exist, so the keeper will relaunch the
supervisor but send NO dead-man ping. That file is ledger item U1, unarmed
since 07-09 — the keeper is only half a safety net without it.
WARN
