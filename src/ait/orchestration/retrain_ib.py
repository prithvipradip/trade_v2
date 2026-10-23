"""W89: the daily retrain's own IB session.

Why this exists. The supervisor's retrain used to build ``MarketDataService(None)`` (no IB client),
so every daily bar came from Yahoo and ``ModelTrainer`` stored it into ``daily_prices`` tagged
``YAHOO_UNADJ``, overwriting the IB rows (2026-10-22: 502 QQQ rows flipped, volume ~1.5x). The
retrain now connects to IB under its own client id and refuses to train on anything else.

What it never does: place or look at an order or a position, use the bot's client id (or its
+100..+104 fallback ladder) or the short-DTE collector's id, or connect to the other account type
(paper 7497 <-> 4002 only, live 7496 <-> 4001 only; see ``ait.data.ib_ports``).

Every external dependency is injected so this is exercised in tests without a broker.
``connect_for_retrain`` raises ``RetrainIBUnavailable`` with the reason; the retrain script turns that
into exit code 3 BEFORE any model or ``daily_prices`` write, which ``master.retrain_models`` already
reports as "model retrain FAILED - models stale".
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ait.orchestration.short_dte_job import reserved_client_ids
from ait.utils.logging import get_logger

log = get_logger("orchestration.retrain_ib")


class RetrainIBUnavailable(RuntimeError):
    """The retrain cannot get an IB session; the message says why."""


def refusal_reason(client_id: int, base_client_id: int, short_dte_client_id: int) -> str | None:
    """Why `client_id` may not be used by the retrain, or None when it is free."""
    taken = reserved_client_ids(base_client_id) | {int(short_dte_client_id)}
    if int(client_id) in taken:
        return (f"ml.retrain_ib_client_id={client_id} collides with ids already in use "
                f"{sorted(taken)} (the bot's id and fallback ladder, and the short-DTE collector)")
    return None


async def connect_for_retrain(
    env_cfg: Any, *, client_id: int, short_dte_client_id: int,
    listening: Callable[[int], bool],
    client_factory: Callable[[Any], Any] | None = None,
) -> Any:
    """Return a CONNECTED ``IBKRClient`` for the retrain, or raise ``RetrainIBUnavailable``.

    `env_cfg` is the bot's ``IBKREnvConfig`` (host, port, base client id). The retrain's copy keeps
    the host and account but takes `client_id` and whichever twin port is actually listening.
    """
    why = refusal_reason(client_id, env_cfg.ibkr_client_id, short_dte_client_id)
    if why:
        raise RetrainIBUnavailable(why)

    from ait.data.ib_ports import endpoint_label, pick_listening_port, port_candidates
    port = pick_listening_port(int(env_cfg.ibkr_port), listening)
    if port is None:
        tried = ", ".join(f"{p} ({endpoint_label(p)})" for p in port_candidates(int(env_cfg.ibkr_port)))
        raise RetrainIBUnavailable(
            f"nothing is listening on the IB port ({tried}): TWS / IB Gateway is down")

    if client_factory is None:
        from ait.broker.ibkr_client import IBKRClient as client_factory  # noqa: N813
    client = client_factory(env_cfg.model_copy(update={
        "ibkr_client_id": int(client_id), "ibkr_port": int(port)}))
    if not await client.connect():
        raise RetrainIBUnavailable(
            f"IB connection failed on port {port} as client id {client_id} (see the ibkr_* log lines)")
    log.info("retrain_ib_connected", port=port, client_id=client_id)
    return client


async def connect_from_settings(
    settings: Any, env_cfg: Any, *, listening: Callable[[int], bool],
    client_factory: Callable[[Any], Any] | None = None,
) -> Any:
    """``connect_for_retrain`` with the ids read from config: ``ml.retrain_ib_client_id`` and
    ``data_collection.short_dte_client_id`` (the one it must not collide with)."""
    return await connect_for_retrain(
        env_cfg, client_id=settings.ml.retrain_ib_client_id,
        short_dte_client_id=settings.data_collection.short_dte_client_id,
        listening=listening, client_factory=client_factory)
