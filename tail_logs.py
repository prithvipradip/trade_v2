"""Live log viewer — readable, color-coded, filtered.

Usage:
    python tail_logs.py              # Follow live logs
    python tail_logs.py --last 50    # Show last 50 events then follow
    python tail_logs.py --trades     # Only show trade events
    python tail_logs.py --no-follow  # Show recent logs and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "logs" / "ait.log"

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

LEVEL_COLORS = {
    "debug": DIM,
    "info": GREEN,
    "warning": YELLOW,
    "error": RED,
    "critical": f"{RED}{BOLD}",
}

# Events worth highlighting
TRADE_EVENTS = {
    "trade_opened", "trade_closed", "order_placed", "order_filled",
    "signal_generated", "position_opened", "position_closed",
    "trade_executed", "trade_entry", "trade_exit",
}
ML_EVENTS = {
    "prediction", "ensemble_trained", "models_loaded", "models_saved",
    "training_complete", "training_symbol",
}
SYSTEM_EVENTS = {
    "ait_starting", "orchestrator_starting", "trading_loop_starting",
    "ibkr_connected", "ibkr_disconnected",
}
LEARNING_EVENTS = {
    "self_learning_adapted", "learner_adapted", "self_learning_final_state",
}
SENTIMENT_EVENTS = {
    "sentiment_result", "fear_greed_reading",
}
SCAN_EVENTS = {
    "scan_symbol", "signals_generated", "strategy_signal",
}

# Skip noisy events
SKIP_PATTERNS = [
    "updatePortfolio:",
    "position: Position(",
    "UserWarning",
    "warnings.warn",
    "Connection pool is full",
    "HTTP Request:",
    "Loading weights:",
    "BertForSequenceClassification",
    "UNEXPECTED",
    "Notes:",
    "feature names",
]


def format_event(line: str, trades_only: bool = False) -> str | None:
    """Parse and format a log line for display."""
    stripped = line.strip()
    if not stripped:
        return None

    # Skip noisy lines
    for pattern in SKIP_PATTERNS:
        if pattern in stripped:
            return None

    # Handle IBKR warnings/errors (non-JSON)
    if stripped.startswith("Error ") or stripped.startswith("Warning "):
        if "margin" in stripped.lower() or "liquidat" in stripped.lower():
            return f"  {RED}{BOLD}!! MARGIN: {stripped}{RESET}"
        if "market data" in stripped.lower():
            return None  # Skip market data subscription warnings
        return f"  {YELLOW}W! {stripped[:120]}{RESET}"

    # Try parsing as JSON structlog
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # Non-JSON line — show if interesting
        if any(kw in stripped.lower() for kw in ["error", "fail", "crash", "margin"]):
            return f"  {RED}{stripped[:150]}{RESET}"
        return None

    event = data.get("event", "unknown")
    level = data.get("level", "info")
    component = data.get("component", "")
    timestamp = data.get("timestamp", "")

    # Filter for trades-only mode
    if trades_only and event not in TRADE_EVENTS:
        return None

    # Format timestamp
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            ts = dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            ts = timestamp[:8]
    else:
        ts = "        "

    # Pick color based on event type
    level_color = LEVEL_COLORS.get(level, "")

    if event in TRADE_EVENTS:
        icon = "$$"
        color = f"{GREEN}{BOLD}"
    elif event in ML_EVENTS:
        icon = "ML"
        color = MAGENTA
    elif event in SYSTEM_EVENTS:
        icon = ">>"
        color = f"{CYAN}{BOLD}"
    elif event in LEARNING_EVENTS:
        icon = "LN"
        color = BLUE
    elif event in SENTIMENT_EVENTS:
        icon = "ST"
        color = CYAN
    elif event in SCAN_EVENTS:
        icon = ".."
        color = DIM
    elif level in ("error", "critical"):
        icon = "!!"
        color = RED
    elif level == "warning":
        icon = "W!"
        color = YELLOW
    else:
        icon = "  "
        color = DIM

    # Build detail string from extra fields
    skip_keys = {"event", "level", "logger", "timestamp", "component"}
    details = {k: v for k, v in data.items() if k not in skip_keys}

    detail_str = ""
    if details:
        parts = []
        for k, v in details.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            elif isinstance(v, str) and len(v) > 80:
                parts.append(f"{k}={v[:80]}...")
            else:
                parts.append(f"{k}={v}")
        detail_str = " | " + ", ".join(parts)

    comp_str = f"{DIM}[{component}]{RESET}" if component else ""

    return f"  {DIM}{ts}{RESET} {icon} {color}{event}{RESET} {comp_str}{detail_str}"


def tail_file(path: Path, last_n: int = 20, follow: bool = True, trades_only: bool = False):
    """Tail a log file with live following."""
    if not path.exists():
        print(f"{RED}Log file not found: {path}{RESET}")
        print(f"Start the bot first: python -m ait.main --paper")
        return

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {CYAN}{BOLD}AIT v2 — LIVE LOG VIEWER{RESET}")
    print(f"  {DIM}File: {path}{RESET}")
    print(f"  {DIM}Mode: {'trades only' if trades_only else 'all events'} | {'following' if follow else 'snapshot'}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    # Read last N lines
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Show last N formatted lines
    shown = 0
    for line in lines[-(last_n * 3):]:  # Read more raw lines since many get filtered
        formatted = format_event(line, trades_only)
        if formatted:
            print(formatted)
            shown += 1
            if shown >= last_n:
                break

    if not follow:
        return

    print(f"\n  {DIM}--- following live (Ctrl+C to stop) ---{RESET}\n")

    # Follow new lines
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # Go to end
        try:
            while True:
                line = f.readline()
                if line:
                    formatted = format_event(line, trades_only)
                    if formatted:
                        print(formatted)
                        sys.stdout.flush()
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n  {DIM}Stopped.{RESET}")


def main():
    parser = argparse.ArgumentParser(description="AIT Live Log Viewer")
    parser.add_argument("--last", type=int, default=30, help="Show last N events")
    parser.add_argument("--trades", action="store_true", help="Only show trade events")
    parser.add_argument("--no-follow", action="store_true", help="Don't follow, just show recent")
    args = parser.parse_args()

    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("")  # Enables ANSI escape codes in Windows terminal

    tail_file(LOG_FILE, last_n=args.last, follow=not args.no_follow, trades_only=args.trades)


if __name__ == "__main__":
    main()
