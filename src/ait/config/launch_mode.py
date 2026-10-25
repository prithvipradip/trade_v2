"""Paper or live: the one decision, shared by the supervisor and the bot's own start-up.

``trading.mode`` is mostly a label. The ACCOUNT the bot trades is whichever one the IB endpoint (``IBKR_PORT``) and login
serve, so a bot labelled paper that is pointed at a live-account endpoint trades real money, and a bot labelled live
that is pointed at a paper endpoint is a mislabelled paper bot. This module refuses both mismatches, and it makes live a
deliberate act:

* live needs ``config.yaml`` ``trading.mode: live`` AND ``.env`` ``TRADING_MODE=live`` (either one saying anything else
  keeps paper),
* plus the environment interlock ``AIT_ALLOW_LIVE=1`` (environment-only on purpose, like ``AIT_ALLOW_UNDEFINED_RISK``: it
  is never a config edit that slips through review),
* plus a live-account IB endpoint (TWS live 7496 / Gateway live 4001) and, when ``IBKR_ACCOUNT`` is set, a live account id.

Paper must have a paper endpoint and, when set, a paper account id (IB paper ids start with ``D``). Stdlib and
``ait.data.ib_ports`` only: the supervisor imports this and must stay light. Docs: docs/RUNBOOK.md "Go-live day checklist".
"""
from __future__ import annotations

from dataclasses import dataclass

from ait.data.ib_ports import LIVE_PORTS, TWIN_PORTS


@dataclass(frozen=True)
class LaunchDecision:
    mode: str                 # "paper" or "live": what to start the bot as
    error: str | None = None  # set when the start must be REFUSED (the message is operator-facing)
    note: str | None = None   # set when the settings disagreed and paper was chosen


def port_kind(port: int) -> str:
    """'live' / 'paper' for a known TWS or Gateway port, else 'unknown' (a custom port cannot be classified)."""
    p = int(port)
    if p in LIVE_PORTS:
        return "live"
    return "paper" if p in TWIN_PORTS else "unknown"


def account_kind(account: str | None) -> str:
    """'paper' for an IB paper account id (starts with 'D', e.g. DU1234567), 'live' for any other id, 'unknown' when unset."""
    a = (account or "").strip().upper()
    if not a:
        return "unknown"
    return "paper" if a.startswith("D") else "live"


def decide(config_mode: str, env_mode: str | None, allow_live: bool, port: int, account: str | None) -> LaunchDecision:
    """What to start as, or why not to start at all. Pure: every input is passed in."""
    cfg = (config_mode or "paper").strip().lower()
    env = (env_mode or "paper").strip().lower()
    both_live = cfg == "live" and env == "live"
    mode = "live" if both_live else "paper"
    note = None
    if (cfg == "live") != (env == "live"):
        note = (f"settings disagree (config.yaml trading.mode={cfg}, .env TRADING_MODE={env}): live needs both, "
                f"so this start is paper")
    pk, ak = port_kind(port), account_kind(account)

    if mode == "paper":
        if pk == "live":
            return LaunchDecision(mode, f"REFUSING to start: the mode is paper but IBKR_PORT {int(port)} is a LIVE-account endpoint, "
                                         f"so orders would go to the live account. Point .env at the paper endpoint (4002 / 7497), "
                                         f"or enable live deliberately (docs/RUNBOOK.md \"Go-live day checklist\").", note)
        if ak == "live":
            return LaunchDecision(mode, f"REFUSING to start: the mode is paper but IBKR_ACCOUNT {account} is not a paper account id.", note)
        return LaunchDecision(mode, None, note)

    if not allow_live:
        return LaunchDecision(mode, "REFUSING to start LIVE: config.yaml and .env both say live but AIT_ALLOW_LIVE=1 is not set in "
                                    "the environment (.env). That interlock is how live trading is switched on deliberately.")
    if pk != "live":
        return LaunchDecision(mode, f"REFUSING to start LIVE: IBKR_PORT {int(port)} is not a live-account endpoint "
                                    f"(live is TWS 7496 / Gateway 4001). The bot would be a mislabelled paper bot.")
    if ak == "paper":
        return LaunchDecision(mode, f"REFUSING to start LIVE: IBKR_ACCOUNT {account} is a paper account id.")
    return LaunchDecision(mode, None, note)
