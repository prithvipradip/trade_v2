"""Roll-combo construction mechanics -- the piece between roll_selector's
target selection and an actual placed order. Nothing calls into this module
yet: it prepares a fully validated, qualified, priced Contract+Order pair
and hands it back, but NEVER calls ibkr.place_order() itself. Actually
submitting a roll -- and the harder problem it surfaces, sequencing the
booking so the reconciler's one-order-one-trade assumption doesn't break
(a roll's single fill needs to book TWO trades, one closing, one opening,
exactly once) -- is separate, later, more consequential work. See PLAN.md.

Generic mechanics (leg pricing, quote polling, contract qualification)
lifted from scripts/probe_w53_roll_atomicity.py, which proved this exact
4-leg combo shape live against TWS paper this session (a real whatIfOrder
margin preview succeeded for both the closing and opening legs). Probe-only
scaffolding (clientId 97, hardcoded SPY, a blocking poll loop, JSONL
evidence files) does not carry over -- this module fails safe (returns
None, never raises) like every other additive module in this package
(roll_selector.py, synthetic_delta.py, condor_defense.py).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ait.bot.state import TradeRecord
from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.execution.executor import _validate_roll_legs
from ait.execution.roll_selector import RollTarget
from ait.utils.logging import get_logger

log = get_logger("execution.roll_executor")


@dataclass
class PreparedRollCombo:
    """A fully validated, qualified, priced roll combo -- ready to place,
    not placed. `bag` is the qualified 4-leg Contract, `order` a configured
    combo_limit Order (orderId not yet assigned)."""

    bag: object
    order: object
    closing_legs: list[dict]
    opening_legs: list[dict]
    mid_net: float
    natural_net: float


def legs_for_roll(
    trade: TradeRecord, side: str, roll_target: RollTarget,
) -> tuple[list[dict], list[dict]] | None:
    """(closing_legs, opening_legs) in the {"strike","right","action",
    "expiry"} shape _validate_roll_legs/qualify_roll_legs expect. Closing
    legs are trade's own open legs on `side`, action reversed (the open
    SELL becomes a closing BUY, the open BUY becomes a closing SELL).
    Opening legs are roll_target's new short (SELL) and new long (BUY) at
    roll_target.expiry. Same construction probe_w53_roll_atomicity.py's
    cmd_roll (:658-667) already proved live -- given its own name/tests
    here instead of living inline in a probe command.

    Returns None if `side` isn't a clean single short+long pair on `trade`
    -- same invariant _at_risk_vertical (roll_selector.py:41) enforces,
    checked independently here since this needs the full leg dicts
    (including each leg's own expiry), not just the two strikes.
    """
    try:
        legs = json.loads(trade.legs) if trade.legs else []
    except (ValueError, TypeError):
        return None

    def _side_legs(action: str) -> list[dict]:
        return [
            leg for leg in legs
            if str(leg.get("action", "")).upper() == action
            and str(leg.get("right", leg.get("type", ""))).upper().startswith(side)
        ]

    shorts, longs = _side_legs("SELL"), _side_legs("BUY")
    if len(shorts) != 1 or len(longs) != 1:
        return None

    closing_legs = [
        {**shorts[0], "action": "BUY"},
        {**longs[0], "action": "SELL"},
    ]
    opening_legs = [
        {"strike": roll_target.new_short.strike, "right": side, "action": "SELL",
         "expiry": roll_target.expiry},
        {"strike": roll_target.new_long.strike, "right": side, "action": "BUY",
         "expiry": roll_target.expiry},
    ]
    return closing_legs, opening_legs


async def qualify_roll_legs(
    ibkr: IBKRClient, symbol: str, legs: list[dict],
) -> list[dict] | None:
    """legs: [{"strike","right","expiry","action"}] -> same dicts plus
    'conId'/'ratio'/'_contract' (the qualified ib_insync Option), or None
    if any leg fails to qualify -- fail-safe, never partial, matching
    roll_selector/synthetic_delta's own None-on-failure convention (the
    probe this was lifted from raised ProbeGuardError instead; production
    code here does not raise)."""
    contracts = [ContractBuilder.option(symbol, leg["expiry"], leg["strike"], leg["right"])
                 for leg in legs]
    qualified = await ibkr.qualify_contracts_batch(contracts)
    out = []
    for leg, q in zip(legs, qualified):
        if q is None:
            log.warning("roll_leg_qualification_failed", symbol=symbol, leg=leg)
            return None
        out.append({**leg, "conId": q.conId, "ratio": leg.get("ratio", 1), "_contract": q})
    return out


async def poll_leg_quotes(
    ibkr: IBKRClient, contracts: list, rounds: int = 10, delay: float = 0.3,
) -> dict[int, tuple[float | None, float | None]]:
    """Snapshot bid/ask per contract, keyed by conId. Same reqMktData/poll/
    cancel idiom as orchestrator._synthesize_combo_cost (orchestrator.py:4194)
    -- lifted verbatim from the probe (:254-282), already generic."""
    out: dict[int, tuple[float | None, float | None]] = {}
    for c in contracts:
        bid = ask = None
        try:
            ibkr.ib.reqMktData(c, "", False, False)
            for _ in range(rounds):
                await asyncio.sleep(delay)
                t = ibkr.ib.ticker(c)
                if not t:
                    continue
                b, a = t.bid, t.ask
                if b is not None and b == b and b >= 0:
                    bid = b
                if a is not None and a == a and a > 0:
                    ask = a
                if bid is not None and ask is not None:
                    break
        finally:
            try:
                ibkr.ib.cancelMktData(c)
            except Exception:  # noqa: BLE001
                pass
        out[c.conId] = (bid, ask)
    return out


def leg_net_price(
    legs: list[dict], quotes: dict[int, tuple[float | None, float | None]], side: str,
) -> float | None:
    """Signed net price for the always-BUY combo convention (OrderBuilder.
    combo_limit's docstring): a BUY leg's cost is positive, a SELL leg's
    proceeds are negative. `side` is "mid" (both legs at mid) or "natural"
    (BUY at ask, SELL at bid -- the marketable, guaranteed-fillable price).
    Returns None if any leg is missing the quote side it needs. Lifted
    verbatim from the probe (:308-330), already generic."""
    total = 0.0
    for leg in legs:
        bid, ask = quotes.get(leg["conId"], (None, None))
        ratio = float(leg.get("ratio", 1) or 1)
        is_buy = str(leg["action"]).upper() == "BUY"
        if side == "mid":
            if bid is None or ask is None:
                return None
            px = (bid + ask) / 2.0
        else:
            px = ask if is_buy else bid
            if px is None:
                return None
        total += ratio * (px if is_buy else -px)
    return round(total, 4)


async def prepare_roll_combo(
    ibkr: IBKRClient,
    trade: TradeRecord,
    side: str,
    roll_target: RollTarget,
    symbol: str,
    quantity: int,
    order_ref: str,
    debit_cap: float | None = None,
) -> PreparedRollCombo | None:
    """Build, validate, qualify and price a 4-leg roll combo -- READY to
    place, never placed. Never raises; every failure returns None with a
    logged reason, matching this package's fail-safe convention.

    Order, deliberately: legs_for_roll -> _validate_roll_legs (the
    EXISTING, already-tested validator, executor.py:114-181 -- reused, not
    duplicated) BEFORE any API call, exactly matching that function's own
    documented intent -- THEN qualification, THEN quotes/pricing.
    """
    derived = legs_for_roll(trade, side, roll_target)
    if derived is None:
        log.warning("roll_prep_no_clean_vertical", trade_id=trade.trade_id, side=side)
        return None
    closing_legs, opening_legs = derived

    err = _validate_roll_legs(trade, closing_legs, opening_legs)
    if err:
        log.warning("roll_prep_validation_failed", trade_id=trade.trade_id, error=err)
        return None

    qualified_closing = await qualify_roll_legs(ibkr, symbol, closing_legs)
    if qualified_closing is None:
        return None
    qualified_opening = await qualify_roll_legs(ibkr, symbol, opening_legs)
    if qualified_opening is None:
        return None
    all_legs = qualified_closing + qualified_opening

    bag = ContractBuilder.combo(
        symbol, [{"conId": leg["conId"], "action": leg["action"], "ratio": leg.get("ratio", 1)}
                 for leg in all_legs])

    quotes = await poll_leg_quotes(ibkr, [leg["_contract"] for leg in all_legs])
    mid_net = leg_net_price(all_legs, quotes, "mid")
    natural_net = leg_net_price(all_legs, quotes, "natural")
    if mid_net is None or natural_net is None:
        log.warning("roll_prep_missing_quote", trade_id=trade.trade_id)
        return None
    if debit_cap is not None and mid_net > debit_cap and natural_net > debit_cap:
        log.warning("roll_prep_over_debit_cap", trade_id=trade.trade_id,
                    mid_net=mid_net, natural_net=natural_net, debit_cap=debit_cap)
        return None

    order = OrderBuilder.combo_limit(action="BUY", quantity=quantity, limit_price=round(mid_net, 2))
    order.orderRef = order_ref

    return PreparedRollCombo(
        bag=bag, order=order, closing_legs=qualified_closing, opening_legs=qualified_opening,
        mid_net=mid_net, natural_net=natural_net,
    )
