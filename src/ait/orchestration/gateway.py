"""IB Gateway auto-start and auto-login manager.

Uses IBC (IB Controller) to automatically launch and authenticate
IB Gateway without manual intervention. If IBC is not available,
falls back to launching Gateway directly (requires manual login).

IBC download: https://github.com/IbcAlpha/IBC/releases
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("orchestration.gateway")

# Default paths — override via environment variables
GATEWAY_DIR = Path(os.environ.get(
    "IBKR_GATEWAY_DIR", r"C:\Jts\ibgateway\1044"
))
IBC_DIR = Path(os.environ.get(
    "IBC_DIR", r"C:\IBC"
))

GATEWAY_EXE = GATEWAY_DIR / "ibgateway.exe"
IBC_GATEWAY_SCRIPT = IBC_DIR / "StartGateway.bat"
IBC_INI = IBC_DIR / "config.ini"


def is_gateway_running(host: str = "127.0.0.1", port: int = 4002, timeout: float = 2.0) -> bool:
    """Check if IB Gateway is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def wait_for_gateway(host: str = "127.0.0.1", port: int = 4002,
                     timeout: int = 120, poll_interval: int = 5) -> bool:
    """Wait until Gateway is accepting connections."""
    elapsed = 0
    while elapsed < timeout:
        if is_gateway_running(host, port):
            log.info("gateway_ready", host=host, port=port, wait_seconds=elapsed)
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed % 15 == 0:
            log.info("gateway_waiting", elapsed=elapsed, timeout=timeout)
    log.error("gateway_timeout", timeout=timeout)
    return False


def start_gateway_ibc(trading_mode: str = "paper") -> bool:
    """Start IB Gateway via IBC (automated login).

    Requires IBC installed at IBC_DIR with credentials in config.ini.
    """
    if not IBC_GATEWAY_SCRIPT.exists():
        log.warning("ibc_not_found", path=str(IBC_GATEWAY_SCRIPT))
        return False

    log.info("gateway_starting_ibc", mode=trading_mode)
    try:
        subprocess.Popen(
            [str(IBC_GATEWAY_SCRIPT), "/INLINE"],
            cwd=str(IBC_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return True
    except Exception as e:
        log.error("ibc_start_failed", error=str(e))
        return False


def start_gateway_direct() -> bool:
    """Start IB Gateway directly (requires manual login)."""
    if not GATEWAY_EXE.exists():
        log.error("gateway_exe_not_found", path=str(GATEWAY_EXE))
        return False

    log.info("gateway_starting_direct", exe=str(GATEWAY_EXE))
    try:
        subprocess.Popen(
            [str(GATEWAY_EXE)],
            cwd=str(GATEWAY_DIR),
            creationflags=subprocess.DETACHED_PROCESS,
        )
        return True
    except Exception as e:
        log.error("gateway_start_failed", error=str(e))
        return False


def ensure_gateway(host: str = "127.0.0.1", port: int = 4002,
                   trading_mode: str = "paper") -> bool:
    """Ensure IB Gateway is running. Start it if not.

    Priority:
    1. Already running → return True
    2. IBC available → auto-login
    3. Gateway exe → launch (manual login needed)
    """
    if is_gateway_running(host, port):
        log.info("gateway_already_running", host=host, port=port)
        return True

    log.warning("gateway_not_running", host=host, port=port)

    # Try IBC first (automated login via IB Gateway, not TWS)
    if IBC_GATEWAY_SCRIPT.exists():
        if start_gateway_ibc(trading_mode):
            return wait_for_gateway(host, port, timeout=120)

    # Fallback: launch Gateway directly
    if GATEWAY_EXE.exists():
        if start_gateway_direct():
            log.warning("gateway_manual_login_required",
                        hint="IBC not installed — you must login manually. "
                             "Install IBC from https://github.com/IbcAlpha/IBC/releases "
                             "for fully automated startup.")
            return wait_for_gateway(host, port, timeout=180)

    log.error("gateway_cannot_start",
              hint="Neither IBC nor IB Gateway found. Install IB Gateway and/or IBC.")
    return False


def setup_ibc(username: str, password: str, trading_mode: str = "paper",
              gateway_dir: str | None = None) -> Path:
    """Create IBC config.ini and StartGateway.bat for automated login.

    Call this once to configure IBC, then ensure_gateway() will use it.
    """
    ibc_dir = IBC_DIR
    ibc_dir.mkdir(parents=True, exist_ok=True)
    gw_dir = gateway_dir or str(GATEWAY_DIR)

    # IBC config.ini
    config = f"""# IBC Configuration — auto-generated
# See https://github.com/IbcAlpha/IBC/blob/master/userguide.md

LogToConsole=yes

# Login credentials
IbLoginId={username}
IbPassword={password}
TradingMode={trading_mode}

# Auto-accept non-brokerage account warning
AcceptNonBrokerageAccountWarning=yes

# Accept incoming API connections
AcceptIncomingConnectionAction=accept

# Gateway settings
OverrideTwsApiPort=4002
ReadOnlyLogin=no

# Keep alive
ExistingSessionDetectedAction=primaryoverride
"""
    config_path = ibc_dir / "config.ini"
    config_path.write_text(config)
    log.info("ibc_config_written", path=str(config_path))

    # StartGateway.bat
    bat = f"""@echo off
set IBC_INI={ibc_dir}\\config.ini
set TRADING_MODE={trading_mode}
set GATEWAY_DIR={gw_dir}

java -cp "{gw_dir}\\jars\\*;{ibc_dir}\\IBC.jar" ^
    ibcalpha.ibc.GatewayStart ^
    "%IBC_INI%" ^
    "%GATEWAY_DIR%" ^
    "%TRADING_MODE%"
"""
    bat_path = ibc_dir / "StartGateway.bat"
    bat_path.write_text(bat)
    log.info("ibc_bat_written", path=str(bat_path))

    return ibc_dir
