"""W82: TWS and IB Gateway listen on different ports; research collection accepts either.

The same account type is served by two possible endpoints: TWS paper 7497 or IB Gateway paper
4002, TWS live 7496 or IB Gateway live 4001. The history collector needs only a reachable
endpoint of the configured account type, so when the configured port is closed it tries the
other endpoint of the SAME account type (a paper setup never probes a live port). Pure: the
probe is injected, so tests and the supervisor job share one definition.
"""
from __future__ import annotations

import socket
from collections.abc import Callable

# configured port -> the other endpoint of the same account type
TWIN_PORTS: dict[int, int] = {7497: 4002, 4002: 7497, 7496: 4001, 4001: 7496}
GATEWAY_PORTS = frozenset({4001, 4002})
LIVE_PORTS = frozenset({7496, 4001})


def port_candidates(configured: int) -> list[int]:
    """The configured port first, then its twin (when it has one)."""
    p = int(configured)
    twin = TWIN_PORTS.get(p)
    return [p] if twin is None else [p, twin]


def pick_listening_port(configured: int, is_listening: Callable[[int], bool]) -> int | None:
    """The first candidate something is listening on, or None when both are closed."""
    for p in port_candidates(configured):
        if is_listening(p):
            return p
    return None


def endpoint_label(port: int) -> str:
    """Human label for logs: 'IB Gateway (paper)' / 'TWS (live)' / 'port 1234'."""
    p = int(port)
    if p in TWIN_PORTS:
        kind = "IB Gateway" if p in GATEWAY_PORTS else "TWS"
        return f"{kind} ({'live' if p in LIVE_PORTS else 'paper'})"
    return f"port {p}"


def tcp_listening(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    """True when a TCP connection to host:port is accepted."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False
