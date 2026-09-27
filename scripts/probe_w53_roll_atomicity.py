#!/usr/bin/env python3
"""W53 follow-up: cross-expiry roll-combo atomicity probe (PAPER account only).

    python scripts/probe_w53_roll_atomicity.py preflight   # default; places nothing
    python scripts/probe_w53_roll_atomicity.py open        # places a 2-leg SPY put vertical
    python scripts/probe_w53_roll_atomicity.py roll        # ONE 4-leg cross-expiry BAG
    python scripts/probe_w53_roll_atomicity.py flatten     # closes whatever the probe has open
    python scripts/probe_w53_roll_atomicity.py status      # read-only snapshot

Why this exists: PLAN.md's W53 entries (roll-mechanics groundwork, `81bad54`) gate the
combo-placement/fill-handling build on one unproven question -- will IB accept, price and
fill ONE BAG that CLOSES a vertical at expiry A and OPENS a vertical at expiry B,
atomically? No cross-expiry combo has ever been placed by this bot. This script answers
that on the PAPER account by reusing the bot's own real primitives (ContractBuilder,
OrderBuilder, IBKRClient, the W53 _validate_roll_legs/select_roll_target pieces) rather
than hand-rolling ib_insync calls, and records every decision as JSONL evidence.

SAFETY, by design:
  - Refuses to run against a non-paper (non-"D") account (belt: IBKRClient.connect()
    already refuses; suspenders: _guard_account re-checks after connect).
  - `preflight` never calls placeOrder -- qualification, quotes and a whatIfOrder margin
    preview only.
  - Every order this script places carries orderRef="W53PROBE-<step>-<ts>", so the live
    bot's untagged-BAG sweep (reconciler.py:694-701, 767-797) can never adopt or cancel one.
  - Uses clientId 97 -- outside the bot's own ladder (1, then 101-105 on fallback) and
    every prior ad-hoc script's id (83/88/90/91/95).
  - Refuses any contract key already in data/HALT_UNTRACKED or already a leg of an open
    DB trade (both read via the same "SYMBOL:STRIKE:RIGHT:EXPIRY" key format
    reconciler.py uses).
  - `open`/`roll`/`flatten` refuse outside RTH and within 30 minutes of the open/close.
  - Hard caps: 1 contract, SPY only, wing width <=$5, roll net debit <=$2.00.
  - On any guard refusal or unexpected error during `open`/`roll`, main() attempts an
    emergency `flatten` automatically before exiting non-zero.

This script makes NO changes to src/ait. It is read/quote/order-only against IBKR and
read-only against data/ait_state.db (opened via `file:...?mode=ro`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

from ait.broker.contracts import ContractBuilder  # noqa: E402
from ait.broker.ibkr_client import IBKRClient  # noqa: E402
from ait.broker.orders import OrderBuilder  # noqa: E402
from ait.config.settings import IBKREnvConfig, Settings, load_settings  # noqa: E402
from ait.data.market_data import MarketDataService  # noqa: E402
from ait.data.options_chain import OptionContract, OptionsChainService  # noqa: E402
from ait.execution.executor import _validate_roll_legs  # noqa: E402
from ait.execution.roll_selector import select_roll_target  # noqa: E402
from ait.bot.state import TradeDirection, TradeRecord, TradeStatus  # noqa: E402
from ait.utils.logging import get_logger  # noqa: E402
from ait.utils.time import MARKET_OPEN, get_market_close, is_market_open, now_et  # noqa: E402

log = get_logger("scripts.probe_w53")

# --- Constants ---------------------------------------------------------

PROBE_CLIENT_ID = 97
PROBE_MARKET_DATA_TYPE = 4  # delayed-frozen: shows the last snapshot outside RTH
SYMBOL = "SPY"
QUANTITY = 1
MAX_WIDTH = 5.0
MAX_ROLL_NET_DEBIT = 2.00
RTH_BUFFER_MINUTES = 30
LADDER_STEPS = (0.0, 0.5, 1.0)  # fraction of the way from mid to natural
LADDER_STEP_SECONDS = 45
CANCEL_AFTER_SECONDS = 60  # after the final (natural) rung

OPEN_DTE_MIN = 5
OPEN_DTE_MAX = 10
OPEN_SHORT_DELTA_TARGET = 0.20
OPEN_SHORT_DELTA_TOLERANCE = 0.10
OPEN_WIDTH = 3.0

STATE_DIR = ROOT / "reports" / "w53_probe"
DB_PATH = ROOT / "data" / "ait_state.db"
HALT_UNTRACKED_PATH = ROOT / "data" / "HALT_UNTRACKED"

_OPEN_DB_STATUSES_EXCLUDED = ("closed", "cancelled", "rejected")


class ProbeGuardError(RuntimeError):
    """A hard safety guard refused the probe. Never bypass -- fix the condition
    that tripped it (wrong session, RTH window, a real collision, etc.)."""


# --- State / evidence files ---------------------------------------------

def _today_dir() -> Path:
    d = STATE_DIR / now_et().date().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return _today_dir() / "state.json"


def load_state() -> dict:
    p = _state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2, default=str))


def record_evidence(step_name: str, event: str, **fields) -> None:
    entry = {"ts": now_et().isoformat(), "event": event, **fields}
    with (_today_dir() / f"{step_name}.jsonl").open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    log.info(f"probe_{event}", step=step_name)


def _order_ref(step: str) -> str:
    return f"W53PROBE-{step}-{int(time.time())}"


def _contract_key(symbol: str, strike: float, right: str, expiry: str) -> str:
    return f"{symbol}:{float(strike)}:{str(right)[:1].upper()}:{expiry}"


# --- Guards --------------------------------------------------------------

def _read_halt_untracked_keys() -> set[str]:
    if not HALT_UNTRACKED_PATH.exists():
        return set()
    keys: set[str] = set()
    for line in HALT_UNTRACKED_PATH.read_text().splitlines():
        line = line.strip()
        if line.count(":") == 3:
            keys.add(line)
    return keys


def _read_open_db_leg_keys() -> set[str]:
    if not DB_PATH.exists():
        return set()
    keys: set[str] = set()
    placeholders = ",".join("?" for _ in _OPEN_DB_STATUSES_EXCLUDED)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(
            f"SELECT symbol, legs FROM trades WHERE status NOT IN ({placeholders})",
            _OPEN_DB_STATUSES_EXCLUDED,
        )
        for symbol, legs_json in cur.fetchall():
            try:
                legs = json.loads(legs_json or "[]")
            except (ValueError, TypeError):
                continue
            for leg in legs:
                try:
                    strike = float(leg["strike"])
                except (KeyError, TypeError, ValueError):
                    continue
                right = str(leg.get("right", leg.get("type", ""))).upper()
                expiry = str(leg.get("expiry", ""))
                keys.add(_contract_key(symbol, strike, right, expiry))
    finally:
        con.close()
    return keys


def guard_no_collision(legs: list[dict]) -> None:
    """Refuse if any of `legs` (each {"strike","right","expiry",...}) shares a
    contract key with an existing HALT_UNTRACKED entry or an open DB trade leg."""
    blocked = _read_halt_untracked_keys() | _read_open_db_leg_keys()
    for leg in legs:
        key = _contract_key(SYMBOL, leg["strike"], leg["right"], str(leg["expiry"]))
        if key in blocked:
            raise ProbeGuardError(
                f"contract key {key} collides with an existing HALT_UNTRACKED or "
                "open DB trade leg -- refusing")


def guard_account(ibkr: IBKRClient) -> None:
    acct = ibkr.session_account or ""
    if not acct.upper().startswith("D"):
        raise ProbeGuardError(
            f"session account {acct!r} is not paper (D-prefixed) -- refusing")


def guard_rth_window() -> None:
    now = now_et()
    if not is_market_open():
        raise ProbeGuardError(
            f"market is not open at {now.isoformat()} ET -- placing steps require RTH")
    close_t = get_market_close(now.date())
    open_dt = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                           second=0, microsecond=0)
    close_dt = now.replace(hour=close_t.hour, minute=close_t.minute,
                            second=0, microsecond=0)
    if now < open_dt + timedelta(minutes=RTH_BUFFER_MINUTES):
        raise ProbeGuardError(f"within {RTH_BUFFER_MINUTES}min of the open -- refusing")
    if now > close_dt - timedelta(minutes=RTH_BUFFER_MINUTES):
        raise ProbeGuardError(f"within {RTH_BUFFER_MINUTES}min of the close -- refusing")


def guard_no_working_step(state: dict, step: str) -> None:
    existing = state.get(step)
    if existing and existing.get("status") == "working":
        raise ProbeGuardError(
            f"a '{step}' step is already marked working in today's state file -- "
            "run 'status' or 'flatten' first")


# --- Connection / services ------------------------------------------------

async def connect_probe() -> IBKRClient:
    cfg = IBKREnvConfig(ibkr_client_id=PROBE_CLIENT_ID)
    ibkr = IBKRClient(cfg)
    ok = await ibkr.connect()
    if not ok:
        raise ProbeGuardError(
            "IBKRClient.connect() failed or refused (see logs -- this includes the "
            "built-in paper/live account gate)")
    guard_account(ibkr)
    # This Mac's .env sets AIT_MARKET_DATA_TYPE=3 (delayed, not delayed-frozen) --
    # IBKRClient.connect() already requested that for THIS session. Type 3 goes
    # silent outside RTH (no live delayed ticks to delay), which is exactly why
    # a Sunday preflight got bid=ask=vol=oi=0 on every contract in every chain.
    # Re-request type 4 (delayed-frozen) for the PROBE's own session only --
    # harmless to the live bot, which has its own separate clientId/session --
    # so preflight can price off Friday's last snapshot when the market is shut.
    ibkr.ib.reqMarketDataType(PROBE_MARKET_DATA_TYPE)
    return ibkr


def build_services(ibkr: IBKRClient) -> tuple[MarketDataService, OptionsChainService, Settings]:
    settings = load_settings(str(ROOT / "config.yaml"))
    market_data = MarketDataService(ibkr, polygon_api_key=settings.api_keys.polygon_api_key)
    chain_service = OptionsChainService(ibkr, market_data, settings.options)
    return market_data, chain_service, settings


# --- Quoting ---------------------------------------------------------------

async def poll_leg_quotes(
    ibkr: IBKRClient, contracts: list, rounds: int = 10, delay: float = 0.3,
) -> dict[int, tuple[float | None, float | None]]:
    """Snapshot bid/ask per contract, keyed by conId. Same reqMktData/poll/
    cancel idiom as orchestrator._synthesize_combo_cost (orchestrator.py:4194)."""
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


async def poll_bag_quote(
    ibkr: IBKRClient, bag, rounds: int = 15, delay: float = 0.1,
) -> tuple[float | None, float | None]:
    """Raw-Bag NBBO -- same idiom executor._execute_multi_leg uses (executor.py:772-799)."""
    bid = ask = None
    try:
        ibkr.ib.reqMktData(bag, "", False, False)
        for _ in range(rounds):
            await asyncio.sleep(delay)
            t = ibkr.ib.ticker(bag)
            b = t.bid if (t and t.bid == t.bid) else None
            a = t.ask if (t and t.ask == t.ask) else None
            if b is not None and a is not None:
                bid, ask = b, a
                break
    finally:
        try:
            ibkr.ib.cancelMktData(bag)
        except Exception:  # noqa: BLE001
            pass
    return bid, ask


def leg_net_price(
    legs: list[dict], quotes: dict[int, tuple[float | None, float | None]], side: str,
) -> float | None:
    """Signed net price for the always-BUY combo convention (OrderBuilder.combo_limit's
    docstring): a BUY leg's cost is positive, a SELL leg's proceeds are negative.
    `side` is "mid" (both legs at mid) or "natural" (BUY at ask, SELL at bid -- the
    marketable, guaranteed-fillable price). Returns None if any leg is missing the
    quote side it needs."""
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


async def whatif_margin(ibkr: IBKRClient, bag, order) -> dict:
    order.whatIf = True
    try:
        state = await ibkr.ib.whatIfOrderAsync(bag, order)
    finally:
        order.whatIf = False
    if state is None:
        return {"ok": False}
    return {
        "ok": True,
        "init_margin_change": getattr(state, "initMarginChange", None),
        "maint_margin_change": getattr(state, "maintMarginChange", None),
        "equity_with_loan_change": getattr(state, "equityWithLoanChange", None),
        "commission": getattr(state, "commission", None),
        "warning_text": getattr(state, "warningText", None),
    }


# --- Contract selection ----------------------------------------------------

async def qualify_legs(ibkr: IBKRClient, legs: list[dict]) -> list[dict]:
    """legs: [{"strike","right","expiry","action"}] -> same dicts plus
    'conId'/'ratio'/'_contract' (the qualified ib_insync Option)."""
    contracts = [ContractBuilder.option(SYMBOL, leg["expiry"], leg["strike"], leg["right"])
                 for leg in legs]
    qualified = await ibkr.qualify_contracts_batch(contracts)
    out = []
    for leg, q in zip(legs, qualified):
        if q is None:
            raise ProbeGuardError(f"contract qualification failed for leg {leg}")
        out.append({**leg, "conId": q.conId, "ratio": leg.get("ratio", 1), "_contract": q})
    return out


async def pick_vertical(
    chain_service: OptionsChainService, side: str, dte_min: int, dte_max: int,
    target_delta: float, tolerance: float, width: float,
) -> tuple[OptionContract, OptionContract]:
    """Same delta-target idiom roll_selector.select_roll_target uses
    (0.20 delta, iron_condor.py's own short-strike convention), but for a
    fresh open rather than a replacement -- picks the NEAREST expiry in the
    window, since the probe's vertical A is meant to be the near-dated leg
    that later gets rolled OUT."""
    chains = await chain_service.get_chain(SYMBOL, min_dte=dte_min, max_dte=dte_max)
    if not chains:
        raise ProbeGuardError(f"no chain returned for {SYMBOL} in [{dte_min},{dte_max}] DTE")
    chain = min(chains, key=lambda c: c.expiry)
    liquid = chain.filter_liquid()
    candidates = liquid.puts if side == "P" else liquid.calls
    if not candidates:
        raise ProbeGuardError(f"no liquid {side} contracts at expiry {chain.expiry}")
    short = min(candidates, key=lambda c: abs(abs(c.delta) - target_delta))
    if abs(abs(short.delta) - target_delta) > tolerance:
        raise ProbeGuardError(
            f"no strike within {tolerance} of {target_delta} delta (closest: "
            f"{short.strike} @ {short.delta})")
    target_long_strike = short.strike - width if side == "P" else short.strike + width
    long_leg = min(candidates, key=lambda c: abs(c.strike - target_long_strike))
    if long_leg.strike == short.strike:
        raise ProbeGuardError("no distinct long strike available for the wing")
    if abs(short.strike - long_leg.strike) > MAX_WIDTH:
        raise ProbeGuardError(
            f"wing width {abs(short.strike - long_leg.strike)} exceeds MAX_WIDTH={MAX_WIDTH}")
    return short, long_leg


# --- Order placement / monitoring ------------------------------------------

async def _monitor_and_ladder(
    ibkr: IBKRClient, trade, mid_net: float, natural_net: float, step_name: str,
) -> dict:
    start = time.time()
    step_idx = 0
    final_rung_at: float | None = None
    while True:
        await asyncio.sleep(1.0)
        status = trade.orderStatus.status
        record_evidence(step_name, "status_poll", status=status,
                         filled=trade.orderStatus.filled,
                         remaining=trade.orderStatus.remaining,
                         avg_fill_price=trade.orderStatus.avgFillPrice)
        if status == "Filled":
            break
        if status in ("Cancelled", "ApiCancelled", "Inactive"):
            break
        elapsed = time.time() - start
        target_step = min(int(elapsed // LADDER_STEP_SECONDS), len(LADDER_STEPS) - 1)
        if target_step > step_idx:
            step_idx = target_step
            frac = LADDER_STEPS[step_idx]
            new_limit = round(mid_net + frac * (natural_net - mid_net), 2)
            trade.order.lmtPrice = new_limit
            ibkr.ib.placeOrder(trade.contract, trade.order)  # modify in place, same orderId
            record_evidence(step_name, "reprice", step=step_idx, limit=new_limit)
            if step_idx == len(LADDER_STEPS) - 1:
                final_rung_at = time.time()
        if final_rung_at is not None and time.time() - final_rung_at > CANCEL_AFTER_SECONDS:
            await ibkr.cancel_order(trade)
            record_evidence(step_name, "cancel_timeout")
            await asyncio.sleep(2.0)
            break
    fills = [
        {"time": str(f.time), "price": f.execution.price, "shares": f.execution.shares,
         "side": f.execution.side, "perm_id": f.execution.permId, "exec_id": f.execution.execId}
        for f in trade.fills
    ]
    errors = ibkr.order_error_codes(trade.order.orderId)
    result = {"order_id": trade.order.orderId, "status": trade.orderStatus.status,
              "avg_fill_price": trade.orderStatus.avgFillPrice, "fills": fills, "errors": errors}
    record_evidence(step_name, "final", **result)
    return result


async def place_combo_ladder(
    ibkr: IBKRClient, legs: list[dict], step_name: str, order_ref: str,
    debit_cap: float | None = None,
) -> dict:
    """legs: qualified legs (conId + action [+ratio, _contract]). Prices via
    per-leg mid -> natural synthesis (orchestrator._synthesize_combo_cost's
    BUY-at-ask/SELL-at-bid idiom), ladders LADDER_STEPS 45s apart, cancels
    CANCEL_AFTER_SECONDS after the final (natural) rung if still unfilled."""
    bag = ContractBuilder.combo(
        SYMBOL, [{"conId": l["conId"], "action": l["action"], "ratio": l.get("ratio", 1)}
                 for l in legs])
    quotes = await poll_leg_quotes(ibkr, [l["_contract"] for l in legs])
    record_evidence(step_name, "leg_quotes",
                     quotes={str(k): v for k, v in quotes.items()})

    mid_net = leg_net_price(legs, quotes, "mid")
    natural_net = leg_net_price(legs, quotes, "natural")
    if mid_net is None or natural_net is None:
        raise ProbeGuardError(
            f"{step_name}: could not synthesize a per-leg net price (missing bid/ask "
            "on at least one leg) -- refusing to place")
    if debit_cap is not None and mid_net > debit_cap and natural_net > debit_cap:
        raise ProbeGuardError(
            f"{step_name}: net price mid={mid_net}/natural={natural_net} exceeds the "
            f"${debit_cap} debit cap -- refusing to place")

    bag_bid, bag_ask = await poll_bag_quote(ibkr, bag)
    record_evidence(step_name, "bag_quote", bid=bag_bid, ask=bag_ask,
                     mid_net=mid_net, natural_net=natural_net)

    order = OrderBuilder.combo_limit(action="BUY", quantity=QUANTITY, limit_price=round(mid_net, 2))
    order.orderRef = order_ref
    trade = await ibkr.place_order(bag, order)
    if trade is None:
        raise ProbeGuardError(f"{step_name}: place_order returned None")
    record_evidence(step_name, "order_placed", order_id=trade.order.orderId,
                     limit=round(mid_net, 2), order_ref=order_ref)

    return await _monitor_and_ladder(ibkr, trade, mid_net, natural_net, step_name)


# --- Commands ----------------------------------------------------------

async def cmd_preflight(_args: argparse.Namespace) -> dict:
    """Read-only: qualification, chain selection, _validate_roll_legs, quotes
    and a whatIfOrder margin preview for both the open combo and the roll
    combo. Never calls placeOrder."""
    ibkr = await connect_probe()
    try:
        _market_data, chain_service, settings = build_services(ibkr)
        record_evidence("preflight", "connected", client_id=ibkr.client_id,
                         account=ibkr.session_account,
                         market_data_type=PROBE_MARKET_DATA_TYPE)

        short_a, long_a = await pick_vertical(
            chain_service, "P", OPEN_DTE_MIN, OPEN_DTE_MAX,
            OPEN_SHORT_DELTA_TARGET, OPEN_SHORT_DELTA_TOLERANCE, OPEN_WIDTH)
        open_legs_raw = [
            {"strike": short_a.strike, "right": "P", "action": "SELL", "expiry": str(short_a.expiry)},
            {"strike": long_a.strike, "right": "P", "action": "BUY", "expiry": str(long_a.expiry)},
        ]
        guard_no_collision(open_legs_raw)
        record_evidence("preflight", "vertical_a_selected", short=short_a.strike,
                         long=long_a.strike, expiry=str(short_a.expiry),
                         short_delta=short_a.delta)

        stub_trade = TradeRecord(
            trade_id="W53PROBE", symbol=SYMBOL, strategy="w53_probe",
            direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
            entry_time=now_et().isoformat(), entry_price=0.0, quantity=QUANTITY,
            contract_type="spread", legs=json.dumps(open_legs_raw))
        roll_target = await select_roll_target(
            stub_trade, "P", chain_service, settings.assignment_response)
        if roll_target is None:
            record_evidence("preflight", "roll_target_none")
            raise ProbeGuardError(
                "select_roll_target returned None -- no valid roll target in the "
                "configured DTE window")
        record_evidence("preflight", "roll_target_selected",
                         new_short=roll_target.new_short.strike,
                         new_long=roll_target.new_long.strike, expiry=roll_target.expiry)

        closing_legs = [
            {"strike": short_a.strike, "right": "P", "action": "BUY", "expiry": str(short_a.expiry)},
            {"strike": long_a.strike, "right": "P", "action": "SELL", "expiry": str(long_a.expiry)},
        ]
        opening_legs = [
            {"strike": roll_target.new_short.strike, "right": "P", "action": "SELL",
             "expiry": roll_target.expiry},
            {"strike": roll_target.new_long.strike, "right": "P", "action": "BUY",
             "expiry": roll_target.expiry},
        ]
        validation_error = _validate_roll_legs(stub_trade, closing_legs, opening_legs)
        record_evidence("preflight", "roll_legs_validated", error=validation_error)
        if validation_error:
            raise ProbeGuardError(f"_validate_roll_legs refused the roll: {validation_error}")
        guard_no_collision(opening_legs)

        qualified_a = await qualify_legs(ibkr, open_legs_raw)
        qualified_closing = await qualify_legs(ibkr, closing_legs)
        qualified_opening = await qualify_legs(ibkr, opening_legs)
        all_roll_legs = qualified_closing + qualified_opening

        open_bag = ContractBuilder.combo(
            SYMBOL, [{"conId": l["conId"], "action": l["action"]} for l in qualified_a])
        roll_bag = ContractBuilder.combo(
            SYMBOL, [{"conId": l["conId"], "action": l["action"]} for l in all_roll_legs])

        open_quotes = await poll_leg_quotes(ibkr, [l["_contract"] for l in qualified_a])
        roll_quotes = await poll_leg_quotes(ibkr, [l["_contract"] for l in all_roll_legs])
        open_bag_bid, open_bag_ask = await poll_bag_quote(ibkr, open_bag)
        roll_bag_bid, roll_bag_ask = await poll_bag_quote(ibkr, roll_bag)
        record_evidence(
            "preflight", "quotes",
            open_leg_quotes={str(k): v for k, v in open_quotes.items()},
            roll_leg_quotes={str(k): v for k, v in roll_quotes.items()},
            open_bag=[open_bag_bid, open_bag_ask], roll_bag=[roll_bag_bid, roll_bag_ask])

        open_mid = leg_net_price(qualified_a, open_quotes, "mid")
        roll_mid = leg_net_price(all_roll_legs, roll_quotes, "mid")

        open_order = OrderBuilder.combo_limit("BUY", QUANTITY, round(open_mid, 2) if open_mid else 0.0)
        open_order.orderRef = _order_ref("preflight-open")
        roll_order = OrderBuilder.combo_limit("BUY", QUANTITY, round(roll_mid, 2) if roll_mid else 0.0)
        roll_order.orderRef = _order_ref("preflight-roll")

        open_whatif = await whatif_margin(ibkr, open_bag, open_order)
        roll_whatif = await whatif_margin(ibkr, roll_bag, roll_order)
        record_evidence("preflight", "whatif", open=open_whatif, roll=roll_whatif)

        preflight_result = {
            "ran_at": now_et().isoformat(),
            "vertical_a": open_legs_raw,
            "vertical_b": opening_legs,
            "open_mid": open_mid, "roll_mid": roll_mid,
            "open_bag_quote": [open_bag_bid, open_bag_ask],
            "roll_bag_quote": [roll_bag_bid, roll_bag_ask],
            "open_whatif": open_whatif, "roll_whatif": roll_whatif,
            "validation_error": validation_error,
        }
        state = load_state()
        state["preflight"] = preflight_result
        save_state(state)
        return preflight_result
    finally:
        await ibkr.disconnect()


async def cmd_open(_args: argparse.Namespace) -> dict:
    guard_rth_window()
    state = load_state()
    guard_no_working_step(state, "open")
    if state.get("open", {}).get("status") == "filled":
        raise ProbeGuardError(
            "an 'open' step already filled today -- run 'roll' or 'flatten', not 'open' again")

    ibkr = await connect_probe()
    try:
        _market_data, chain_service, _settings = build_services(ibkr)
        short_a, long_a = await pick_vertical(
            chain_service, "P", OPEN_DTE_MIN, OPEN_DTE_MAX,
            OPEN_SHORT_DELTA_TARGET, OPEN_SHORT_DELTA_TOLERANCE, OPEN_WIDTH)
        legs_raw = [
            {"strike": short_a.strike, "right": "P", "action": "SELL", "expiry": str(short_a.expiry)},
            {"strike": long_a.strike, "right": "P", "action": "BUY", "expiry": str(long_a.expiry)},
        ]
        guard_no_collision(legs_raw)
        qualified = await qualify_legs(ibkr, legs_raw)

        order_ref = _order_ref("open")
        state["open"] = {"legs": legs_raw, "order_ref": order_ref, "status": "working"}
        save_state(state)

        result = await place_combo_ladder(ibkr, qualified, "open", order_ref)
        state["open"] = {**result, "legs": legs_raw, "order_ref": order_ref,
                          "status": "filled" if result["status"] == "Filled" else "aborted"}
        save_state(state)
        if result["status"] != "Filled":
            raise ProbeGuardError(
                f"open step ended with status {result['status']!r}, not Filled -- "
                "nothing to roll")
        return state["open"]
    finally:
        await ibkr.disconnect()


async def cmd_roll(_args: argparse.Namespace) -> dict:
    guard_rth_window()
    state = load_state()
    open_state = state.get("open")
    if not open_state or open_state.get("status") != "filled":
        raise ProbeGuardError("no filled 'open' step in today's state -- run 'open' first")
    guard_no_working_step(state, "roll")
    if state.get("roll", {}).get("status") == "filled":
        raise ProbeGuardError("a 'roll' step already filled today")

    ibkr = await connect_probe()
    try:
        _market_data, chain_service, settings = build_services(ibkr)
        open_legs = open_state["legs"]
        stub_trade = TradeRecord(
            trade_id="W53PROBE", symbol=SYMBOL, strategy="w53_probe",
            direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
            entry_time=open_state.get("ran_at", now_et().isoformat()),
            entry_price=0.0, quantity=QUANTITY, contract_type="spread",
            legs=json.dumps(open_legs))

        roll_target = await select_roll_target(
            stub_trade, "P", chain_service, settings.assignment_response)
        if roll_target is None:
            raise ProbeGuardError("select_roll_target returned None -- nothing to roll to")

        closing_legs = [
            {**leg, "action": "BUY" if leg["action"] == "SELL" else "SELL"}
            for leg in open_legs
        ]
        opening_legs = [
            {"strike": roll_target.new_short.strike, "right": "P", "action": "SELL",
             "expiry": roll_target.expiry},
            {"strike": roll_target.new_long.strike, "right": "P", "action": "BUY",
             "expiry": roll_target.expiry},
        ]
        err = _validate_roll_legs(stub_trade, closing_legs, opening_legs)
        record_evidence("roll", "validate_roll_legs", error=err)
        if err:
            raise ProbeGuardError(f"_validate_roll_legs refused the roll: {err}")
        guard_no_collision(opening_legs)

        qualified_closing = await qualify_legs(ibkr, closing_legs)
        qualified_opening = await qualify_legs(ibkr, opening_legs)
        all_legs = qualified_closing + qualified_opening

        order_ref = _order_ref("roll")
        state["roll"] = {"closing_legs": closing_legs, "opening_legs": opening_legs,
                          "order_ref": order_ref, "status": "working"}
        save_state(state)

        result = await place_combo_ladder(
            ibkr, all_legs, "roll", order_ref, debit_cap=MAX_ROLL_NET_DEBIT)
        state["roll"] = {**result, "closing_legs": closing_legs, "opening_legs": opening_legs,
                          "order_ref": order_ref,
                          "status": "filled" if result["status"] == "Filled" else "aborted"}
        save_state(state)
        if result["status"] != "Filled":
            raise ProbeGuardError(f"roll step ended with status {result['status']!r}")
        return state["roll"]
    finally:
        await ibkr.disconnect()


async def cmd_flatten(_args: argparse.Namespace) -> dict:
    """Closes whatever vertical the probe currently has open (the rolled-to
    vertical B if 'roll' filled, else vertical A if only 'open' filled).
    Also the emergency path main() calls on any guard/error during
    open/roll -- safe to call repeatedly; a no-op once flat."""
    state = load_state()
    if state.get("roll", {}).get("status") == "filled":
        live_legs = state["roll"]["opening_legs"]
        source = "roll"
    elif state.get("open", {}).get("status") == "filled":
        live_legs = state["open"]["legs"]
        source = "open"
    else:
        return {"status": "nothing_to_flatten"}
    if state.get("flatten", {}).get("status") == "filled":
        return {"status": "already_flat"}

    guard_rth_window()
    ibkr = await connect_probe()
    try:
        closing_legs = [
            {**leg, "action": "BUY" if leg["action"] == "SELL" else "SELL"}
            for leg in live_legs
        ]
        qualified = await qualify_legs(ibkr, closing_legs)
        order_ref = _order_ref("flatten")
        state["flatten"] = {"source": source, "legs": closing_legs,
                             "order_ref": order_ref, "status": "working"}
        save_state(state)

        result = await place_combo_ladder(ibkr, qualified, "flatten", order_ref)
        state["flatten"] = {**result, "source": source, "legs": closing_legs,
                             "order_ref": order_ref,
                             "status": "filled" if result["status"] == "Filled" else "aborted"}
        save_state(state)
        return state["flatten"]
    finally:
        await ibkr.disconnect()


async def cmd_status(_args: argparse.Namespace) -> dict:
    """Read-only snapshot: today's state file plus a live broker check for
    any SPY option position or W53PROBE-tagged open order."""
    state = load_state()
    ibkr = await connect_probe()
    try:
        positions = [
            [p.contract.symbol, p.contract.strike, p.contract.right,
             str(p.contract.lastTradeDateOrContractMonth), p.position]
            for p in ibkr.ib.positions()
            if p.contract.symbol == SYMBOL and getattr(p.contract, "secType", "") == "OPT"
            and p.position != 0
        ]
        open_orders = [
            t.order.orderId for t in ibkr.ib.openTrades()
            if str(getattr(t.order, "orderRef", "") or "").startswith("W53PROBE")
        ]
        result = {"state": state, "probe_positions": positions, "probe_open_orders": open_orders}
        record_evidence("status", "snapshot", probe_positions=positions,
                         probe_open_orders=open_orders)
        return result
    finally:
        await ibkr.disconnect()


_HANDLERS = {
    "preflight": cmd_preflight,
    "open": cmd_open,
    "roll": cmd_roll,
    "flatten": cmd_flatten,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(_HANDLERS), nargs="?", default="preflight")
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]

    try:
        result = asyncio.run(handler(args))
    except (ProbeGuardError, Exception) as e:  # noqa: BLE001 -- top-level safety net
        kind = "GUARD REFUSED" if isinstance(e, ProbeGuardError) else "UNEXPECTED ERROR"
        log.error("probe_command_failed", command=args.command, kind=kind, error=str(e))
        print(f"{kind}: {e}", file=sys.stderr)
        if args.command in ("open", "roll"):
            print("attempting emergency flatten...", file=sys.stderr)
            try:
                flatten_result = asyncio.run(cmd_flatten(args))
                print(json.dumps(flatten_result, indent=2, default=str), file=sys.stderr)
            except Exception as flatten_err:  # noqa: BLE001
                print(f"emergency flatten ALSO failed: {flatten_err}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
