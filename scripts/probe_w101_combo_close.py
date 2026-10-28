#!/usr/bin/env python3
"""W101-P: does a condor close fill on the PAPER account, and at what concession? (PAPER only)

    python scripts/probe_w101_combo_close.py preflight                       # default; places nothing
    python scripts/probe_w101_combo_close.py open  --trial F1 --kind far     # open a 1-lot SPY condor
    python scripts/probe_w101_combo_close.py open  --trial M1 --kind far --quantity 2   # W112: a 2-lot condor
    python scripts/probe_w101_combo_close.py close --trial M1 --quantity 1  # ... close ONE lot, then the rest
    python scripts/probe_w101_combo_close.py close --trial F1                # THE EXPERIMENT
    python scripts/probe_w101_combo_close.py flatten --trial F1              # combo ladder, then leg-out
    python scripts/probe_w101_combo_close.py cancel                          # cancel W101PROBE orders only
    python scripts/probe_w101_combo_close.py status                          # read-only broker + state snapshot
    python scripts/probe_w101_combo_close.py report                          # read-only verdict from the evidence

Why this exists. On 2026-10-22/23 the live bot placed 176 closing combos for a QQQ condor on the
paper account (delayed market data); IB accepted all of them and none filled. W101-B changed the
pricing (intrinsic floor, escalation on every path, urgency rest). This probe answers, on SPY and
with one lot, whether such a close fills, at what concession over the legs' natural price, and how
fast, using the SAME rule the bot uses (`bot_rung_limit`), and it measures whether each limit was
truly marketable against what the market did ~15 minutes later (the "truth tail"), because delayed
quotes understate an in-the-money structure.

SAFETY, by design (this script PLACES REAL PAPER ORDERS; the operator runs it, bot stopped):
  - paper (`D...`) account only; its own clientId (96); any other id (a fallback) is refused,
    because a fallback id cannot cancel the orders placed under 96;
  - refuses to run while the supervisor holds `data/orchestrator.lock` (a flock test: the pid in
    the file is informational), or while `data/bot_heartbeat` is fresh in market hours;
  - SPY or QQQ only, one lot by default (`--quantity 2` or `3` for the W112 multi-lot ramp), exactly
    four legs, wing width <= 3; no combo limit is ever above the wing
    width (the structure's maximum loss); every order is tagged `W101PROBE-...` (the reconciler
    only ever adopts an order whose orderRef equals a trade id);
  - one probe order live at a time: each rung's cancel is CONFIRMED, and the executions are checked
    as well as the status (ib_insync marks an order Cancelled locally on some errors that do not
    prove it did not fill), before the next is placed;
  - leg-out buys the shorts back first and sells a long only after both shorts are confirmed flat;
  - a Ctrl-C or an error cancels the probe's working orders and exits; it never auto-flattens.

This script makes NO changes to src/ait. It is read/quote/order-only against IBKR and read-only
against data/ait_state.db (opened via `file:...?mode=ro`). The `report` command never connects.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

from ait.broker.contracts import ContractBuilder  # noqa: E402
from ait.broker.ibkr_client import IBKRClient  # noqa: E402
from ait.broker.orders import OrderBuilder  # noqa: E402
from ait.config.settings import IBKREnvConfig, load_settings  # noqa: E402
from ait.execution.reconciler import PositionReconciler  # noqa: E402
from ait.utils.logging import get_logger  # noqa: E402
from ait.utils.time import MARKET_OPEN, get_market_close, is_market_open, now_et  # noqa: E402

log = get_logger("scripts.probe_w101")

# --- Constants ---------------------------------------------------------------

PROBE_CLIENT_ID = 96            # 98 is validate_w55_synthetic_delta.py; the bot uses 1 and 101-105
SYMBOL = "SPY"                  # the underlying of THIS run; `use_symbol` changes it (state / --symbol)
PROBE_SYMBOLS = ("SPY", "QQQ")  # the only underlyings the probe will trade; also what its guards watch
SPOT_HINT: float | None = None  # --spot-hint: a rough price of the underlying, the parity search's starting point
QUANTITY = 1                    # lots of THIS run; `use_quantity` changes it (open / close / flatten read the trial)
MAX_QUANTITY = 3                # W112: the multi-lot ramp stops at three lots
MAX_WIDTH = 3.0
TAG = "W101PROBE"
SCHEMA = 1

# The bot's exit pricing, as shipped (src/ait/config/settings.py ExitConfig). `load_settings().exit`
# is read at run time and the values used are recorded in `run_start`; these are the fallbacks.
EXIT_CROSS = 0.10
REPRICE_STEP = 0.10
ESC_PCT = 0.05
NEAR_EXPIRY_DTE = 1
DELAYED_BAND_PCT = 0.0025

REST_S = 120.0                  # exit_stop_pending_rest_seconds
N_RUNGS = 7                     # rungs 0-3 follow the bot; 4-6 close the gap to the wing width
GAP_FRACTIONS = (0.25, 0.5, 1.0)
CANCEL_CONFIRM_TIMEOUT_S = 30.0
TRUTH_TAIL_S = 900.0            # keep sampling this long after the last rung
DELAY_S = 900.0                 # the delayed feed is ~15 minutes old
SAMPLE_EVERY_S = 5.0
TRUTH_MATCH_S = 30.0            # a truth sample must be this close to t + DELAY_S

OPEN_RUNG_FRACTIONS = (0.0, 0.5, 1.0, 1.5, 2.0)   # mid -> natural -> past natural, like W53
OPEN_REST_S = 45.0
OPEN_MIN_CREDIT = 0.05

MIN_NET_LIQ = 25_000.0
MARGIN_MULT = 3.0
HEARTBEAT_MAX_AGE_S = 900.0
WORTHLESS_ASK = 0.05            # a long leg offered at or below this is worthless enough to let expire

# minutes after the open / before the close during which a command may run
WINDOWS = {"open": (30, 45), "close": (30, 30), "flatten": (5, 5), "cancel": (5, 5)}
PIN_OPEN_LATEST = (14, 30)      # a 0DTE trial must be opened by this ET time ...
PIN_CLOSE_LATEST = (15, 15)     # ... and its close started by this one

PROBE_DIR = ROOT / "reports" / "w101_probe"
STATE_PATH = PROBE_DIR / "state.json"
LOCK_PATH = ROOT / "data" / "orchestrator.lock"
HEARTBEAT_PATH = ROOT / "data" / "bot_heartbeat"
DB_PATH = ROOT / "data" / "ait_state.db"
HALT_UNTRACKED_PATH = ROOT / "data" / "HALT_UNTRACKED"

_OPEN_DB_STATUSES_EXCLUDED = ("closed", "cancelled", "rejected")
_DONE = ("Filled", "Cancelled", "ApiCancelled")


def use_symbol(symbol: str) -> None:
    """Point this run at SPY or QQQ (the guards and contract builders read the module global)."""
    global SYMBOL
    if symbol not in PROBE_SYMBOLS:
        raise ProbeGuardError(f"the probe trades only {PROBE_SYMBOLS}, not {symbol!r}")
    SYMBOL = symbol


def use_quantity(quantity: int) -> None:
    """W112: the lot count of this run's orders (1 by default; the multi-lot ramp uses 2 or 3)."""
    global QUANTITY
    if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= MAX_QUANTITY:
        raise ProbeGuardError(f"the probe trades 1 to {MAX_QUANTITY} lots, not {quantity!r}")
    QUANTITY = quantity


class ProbeGuardError(RuntimeError):
    """A hard safety guard refused the probe. Never bypass: fix the condition that tripped it."""


# --- Evidence and state ------------------------------------------------------

class Evidence:
    """Append-only JSONL, one file per ET day: reports/w101_probe/<date>/events.jsonl.

    Every line carries the envelope {schema, ts, mono, run_id, cmd, trial, event, ...} so that
    `report` can compute the verdict later without a broker connection."""

    def __init__(self, cmd: str, trial: str | None = None, run_id: str | None = None) -> None:
        self.cmd = cmd
        self.trial = trial
        self.run_id = run_id or uuid.uuid4().hex[:8]

    def emit(self, event: str, **fields) -> None:
        day_dir = PROBE_DIR / now_et().date().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        entry = {"schema": SCHEMA, "ts": now_et().isoformat(), "mono": round(time.monotonic(), 3),
                 "run_id": self.run_id, "cmd": self.cmd, "trial": self.trial, "event": event, **fields}
        with (day_dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        log.info(f"probe_{event}", cmd=self.cmd, trial=self.trial)


def load_state() -> dict:
    """The one state file across days (a trial opened today may be closed tomorrow)."""
    if not STATE_PATH.exists():
        return {"schema": SCHEMA, "trials": {}}
    try:
        state = json.loads(STATE_PATH.read_text())
    except ValueError as e:
        raise ProbeGuardError(
            f"{STATE_PATH} is corrupt ({e}); inspect it by hand, then run 'status' and "
            "'cancel' before touching anything") from e
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ProbeGuardError(f"{STATE_PATH} has an unknown schema; refusing")
    state.setdefault("trials", {})
    return state


def save_state(state: dict) -> None:
    """Atomic: write a temp file, then os.replace."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, STATE_PATH)


def make_order_ref(trial: str, step: str, k: int, epoch: float | None = None) -> str:
    """e.g. W101PROBE-F1-close-r3-1790707000. orderRef is free text at IB (no uniqueness or
    matching semantics), so a per-rung suffix is safe and lets recovery find orders by prefix."""
    return f"{TAG}-{trial}-{step}-r{k}-{int(epoch if epoch is not None else time.time())}"


# --- Pure maths (no broker) --------------------------------------------------

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def natural_cost(close_legs: list[dict], quotes: dict) -> float | None:
    """The marketable cost of the CLOSING combo from the legs' own quotes, exactly as the bot's
    `_synthesize_combo_cost(credit_buyback=True)` states it: a leg we BUY costs its ask (no ask =
    None, its bid would understate the cost); a leg we SELL returns its bid, where 0.00 is a bid
    and IBKR's -1 ("nobody bids") counts as 0 on a credit buyback; NaN or missing = None.

    close_legs: [{conId, action (the CLOSING action), ratio}]; quotes: {conId: (bid, ask)}."""
    total = 0.0
    for leg in close_legs:
        bid, ask = quotes.get(leg["conId"], (None, None))
        ratio = float(leg.get("ratio", 1) or 1)
        if str(leg["action"]).upper() == "BUY":
            if not (_finite(ask) and ask > 0):
                return None
            total += ratio * ask
        else:
            if _finite(bid) and bid >= 0:
                b = bid
            elif _finite(bid) and bid == -1:
                b = 0.0
            else:
                return None
            total -= ratio * b
    return round(total, 2)


def sell_legs_without_bid(close_legs: list[dict], quotes: dict) -> list[int]:
    """conIds of the legs the CLOSE must sell that have no usable bid (IBKR's -1, 0, NaN or missing):
    nobody is buying them, so a combo that includes them has no counterparty for that leg."""
    out = []
    for leg in close_legs:
        if str(leg["action"]).upper() != "SELL":
            continue
        bid = quotes.get(leg["conId"], (None, None))[0]
        if not (_finite(bid) and bid > 0):
            out.append(int(leg["conId"]))
    return out


def mid_cost(close_legs: list[dict], quotes: dict) -> float | None:
    """Display only: every leg at its mid; None unless both sides of every leg are usable."""
    total = 0.0
    for leg in close_legs:
        bid, ask = quotes.get(leg["conId"], (None, None))
        if not (_finite(bid) and bid >= 0 and _finite(ask) and ask > 0):
            return None
        mid = (bid + ask) / 2.0
        ratio = float(leg.get("ratio", 1) or 1)
        total += ratio * (mid if str(leg["action"]).upper() == "BUY" else -mid)
    return round(total, 4)


def structure_intrinsic(open_legs: list[dict], spot: float) -> float | None:
    """The cost to close the structure at `spot`, through the one valuation function the bot uses.
    PITFALL (pinned by a test): it needs the OPENING legs with their original actions and an
    `iron_condor` strategy; reversed legs or a non-credit name silently flip the sign."""
    shim = SimpleNamespace(strategy="iron_condor", legs=json.dumps(open_legs), strike=None)
    return PositionReconciler._structure_intrinsic(shim, spot)


def intrinsic_floor(open_legs: list[dict], spot, dte, width: float, data_type: int, *,
                    cross: float = EXIT_CROSS, near_dte: int = NEAR_EXPIRY_DTE,
                    band_pct: float = DELAYED_BAND_PCT, must_get_out: bool = True) -> float | None:
    """Mirror of the bot's `_exit_intrinsic_floor` (orchestrator.py): intrinsic + cross, never above
    the width; None when out of the money or the spot is unusable. Near expiry, for a must-get-out
    exit, on delayed data (type 3/4) it is the worst case over spot +/- band_pct."""
    if not _finite(spot) or spot <= 0 or not width:
        return None
    band = 0.0
    if dte is not None and dte <= near_dte and must_get_out and data_type in (3, 4):
        band = band_pct
    lo, hi = spot * (1.0 - band), spot * (1.0 + band)
    cands = {float(spot), lo, hi}
    for leg in open_legs:
        try:
            k = float(leg["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= k <= hi:
            cands.add(k)
    vals = [v for v in (structure_intrinsic(open_legs, c) for c in cands) if v is not None]
    if not vals:
        return None
    intr = max(vals)
    if intr <= 0:
        return None
    return round(min(float(width), intr + cross), 2)


def parity_spot(strike: float, call_q: tuple, put_q: tuple, max_spread: float = 1.0) -> float | None:
    """The underlying implied by one call/put pair at `strike`: S = K + C - P (mid prices; carry and
    dividends are ignored, which is fine to a few cents for a short-dated at-the-money pair). None
    unless both sides of both quotes are usable and tight (a wide quote makes the estimate worthless)."""
    for bid, ask in (call_q, put_q):
        if not (_finite(bid) and bid >= 0 and _finite(ask) and ask > 0 and ask - bid <= max_spread):
            return None
    c = (call_q[0] + call_q[1]) / 2.0
    p = (put_q[0] + put_q[1]) / 2.0
    return round(strike + c - p, 4)


def bot_rung_limit(n: int, natural: float, cross: float = EXIT_CROSS,
                   step: float = REPRICE_STEP, pct: float = ESC_PCT) -> float:
    """The bot's leg-priced limit on its n-th consecutive unfilled placement
    (orchestrator `_close_multi_leg`, path B): (N + cross) + n * max(pct * (N + cross), step)."""
    base = natural + cross
    return base + n * max(pct * base, step)


def rung_limit(k: int, natural: float, prev: float | None, l3: float | None, width: float,
               floor: float | None, *, cross: float = EXIT_CROSS, step: float = REPRICE_STEP,
               pct: float = ESC_PCT) -> tuple[float | None, dict]:
    """The limit for rung k (None = a duplicate of the previous rung, skipped).

    Rungs 0-3 are the bot's rule on the CURRENT natural; rungs 4-6 close 25% / 50% / 100% of the
    remaining gap to the wing width from the rung-3 limit. Clamped to [floor, width], never below
    the previous rung, rounded to a cent. `cap_binds` records natural > width: such a trial proves
    nothing about combo matching (the order is non-marketable by construction)."""
    if k <= 3:
        raw, kind = bot_rung_limit(k, natural, cross, step, pct), "bot"
    else:
        anchor = l3 if l3 is not None else (prev if prev is not None else natural + cross)
        raw, kind = anchor + GAP_FRACTIONS[k - 4] * (width - anchor), "gap"
    lim = min(float(width), raw)
    floor_bound = False
    if floor is not None and lim < floor:
        lim, floor_bound = floor, True
    lim = round(lim, 2)
    info = {"k": k, "kind": kind, "natural": natural, "raw": round(raw, 4), "floor": floor,
            "floor_bound": floor_bound, "cap_bound": raw > width, "cap_binds": natural > width,
            "skipped_duplicate": False}
    if prev is not None and lim <= prev:           # never step down: a rung that does not rise is skipped
        info["skipped_duplicate"] = True
        return None, info
    return lim, info


def planned_ladder(natural: float, width: float, floor: float | None = None) -> list[float]:
    """The ladder for a natural that does not move (preflight display and property tests)."""
    out: list[float] = []
    prev = l3 = None
    for k in range(N_RUNGS):
        lim, _info = rung_limit(k, natural, prev, l3, width, floor)
        if lim is None:
            continue
        out.append(lim)
        prev = lim
        if k == 3:
            l3 = lim
    return out


def entry_rung_limit(k: int, mid: float, natural: float) -> float:
    """The entry ladder (as W53): mid + fraction * (natural - mid), fraction 0, .5, 1, 1.5, 2."""
    return round(mid + OPEN_RUNG_FRACTIONS[k] * (natural - mid), 2)


# --- Guards ------------------------------------------------------------------

def _contract_key(symbol: str, strike: float, right: str, expiry: str) -> str:
    """The reconciler's position-key format (copied from the W53 probe)."""
    return f"{symbol}:{float(strike)}:{str(right)[:1].upper()}:{expiry}"


def _read_halt_untracked_keys() -> set[str]:
    """Copied from the W53 probe."""
    if not HALT_UNTRACKED_PATH.exists():
        return set()
    keys: set[str] = set()
    for line in HALT_UNTRACKED_PATH.read_text().splitlines():
        line = line.strip()
        if line.count(":") == 3:
            keys.add(line)
    return keys


def _read_open_db_leg_keys() -> set[str]:
    """Copied from the W53 probe: legs of trades the bot still carries as open (DB read-only)."""
    if not DB_PATH.exists():
        return set()
    keys: set[str] = set()
    placeholders = ",".join("?" for _ in _OPEN_DB_STATUSES_EXCLUDED)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(
            f"SELECT symbol, legs FROM trades WHERE status NOT IN ({placeholders})",
            _OPEN_DB_STATUSES_EXCLUDED)
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
                keys.add(_contract_key(symbol, strike, right, str(leg.get("expiry", ""))))
    finally:
        con.close()
    return keys


def guard_no_collision(legs: list[dict]) -> None:
    """Refuse if any leg shares a contract key with HALT_UNTRACKED or an open DB trade leg."""
    blocked = _read_halt_untracked_keys() | _read_open_db_leg_keys()
    for leg in legs:
        key = _contract_key(SYMBOL, leg["strike"], leg["right"], str(leg["expiry"]))
        if key in blocked:
            raise ProbeGuardError(
                f"contract key {key} collides with HALT_UNTRACKED or an open DB trade leg")


def guard_account(ibkr) -> None:
    acct = ibkr.session_account or ""
    if not acct.upper().startswith("D"):
        raise ProbeGuardError(f"session account {acct!r} is not paper (D-prefixed) -- refusing")


def guard_client_id(ibkr) -> None:
    """A fallback client id (the ladder tries 196-200 when 96 is busy) could not cancel orders
    placed under 96; refuse rather than run with a session that cannot clean up after itself."""
    if ibkr.client_id != PROBE_CLIENT_ID:
        raise ProbeGuardError(
            f"connected as client id {ibkr.client_id}, not {PROBE_CLIENT_ID}: another session holds "
            f"{PROBE_CLIENT_ID} (a probe still running? a Gateway restart?). Free it or cancel in TWS")


def guard_bot_stopped(lock_path: Path = None, hb_path: Path = None, now_ts: float | None = None,
                      market_open: bool | None = None) -> None:
    """The supervisor flocks data/orchestrator.lock (the pid inside is informational and goes stale),
    so a non-blocking shared flock attempt answers "is a supervisor alive?" exactly. A bare ait.main
    running without the supervisor shows only through its heartbeat file (touched every ~30 s)."""
    lock_path = LOCK_PATH if lock_path is None else lock_path
    hb_path = HEARTBEAT_PATH if hb_path is None else hb_path
    try:
        import fcntl
    except ImportError as e:  # pragma: no cover - the probe is run on the Mac
        raise ProbeGuardError("fcntl is unavailable on this platform; cannot verify the bot is stopped") from e
    if lock_path.exists():
        fh = open(lock_path, "r")
        try:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as e:
                raise ProbeGuardError(
                    f"the supervisor holds {lock_path} -- stop the supervisor (and the bot) first") from e
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
    if market_open is None:
        market_open = is_market_open()
    if market_open and hb_path.exists():
        age = (time.time() if now_ts is None else now_ts) - hb_path.stat().st_mtime
        if age < HEARTBEAT_MAX_AGE_S:
            raise ProbeGuardError(
                f"{hb_path} was touched {age:.0f}s ago (< {HEARTBEAT_MAX_AGE_S:.0f}s): a bot process "
                "looks alive -- stop it first")


def guard_window(cmd: str, kind: str | None = None, *, step: str | None = None) -> None:
    """Per-command trading window. open/close keep clear of the open and the close; flatten and
    cancel only 5 minutes (so a 0DTE emergency close is never blocked). A 0DTE ('pin') trial must be
    opened by 14:30 ET and its close started by 15:15 ET."""
    if cmd not in WINDOWS:
        return
    now = now_et()
    if not is_market_open():
        raise ProbeGuardError(f"market is not open at {now.isoformat()} ET -- '{cmd}' needs RTH")
    after_open, before_close = WINDOWS[cmd]
    close_t = get_market_close(now.date())
    open_dt = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    close_dt = now.replace(hour=close_t.hour, minute=close_t.minute, second=0, microsecond=0)
    if now < open_dt + timedelta(minutes=after_open):
        raise ProbeGuardError(f"within {after_open}min of the open -- '{cmd}' refused")
    if now > close_dt - timedelta(minutes=before_close):
        raise ProbeGuardError(f"within {before_close}min of the close -- '{cmd}' refused")
    if kind in ("pin", "nobid"):
        latest = PIN_OPEN_LATEST if cmd == "open" else PIN_CLOSE_LATEST if cmd == "close" else None
        if latest and (now.hour, now.minute) > latest:
            raise ProbeGuardError(
                f"a 0DTE (pin) '{cmd}' must start by {latest[0]:02d}:{latest[1]:02d} ET -- refused")


def guard_caps(legs: list[dict]) -> None:
    """SPY or QQQ only, 1 to MAX_QUANTITY lots, exactly two puts and two calls, each wing <= MAX_WIDTH."""
    if SYMBOL not in PROBE_SYMBOLS or not 1 <= QUANTITY <= MAX_QUANTITY:
        raise ProbeGuardError(f"the probe is pinned to {PROBE_SYMBOLS}, 1 to {MAX_QUANTITY} lots")
    if len(legs) != 4:
        raise ProbeGuardError(f"a condor has exactly 4 legs, got {len(legs)}")
    for right in ("P", "C"):
        side = [l for l in legs if str(l["right"]).upper() == right]
        if len(side) != 2 or {str(l["action"]).upper() for l in side} != {"BUY", "SELL"}:
            raise ProbeGuardError(f"need one BUY and one SELL {right} leg, got {side}")
        width = abs(float(side[0]["strike"]) - float(side[1]["strike"]))
        if width <= 0 or width > MAX_WIDTH + 1e-9:
            raise ProbeGuardError(f"{right} wing width {width} outside (0, {MAX_WIDTH}]")
    if len({str(l["expiry"]) for l in legs}) != 1:
        raise ProbeGuardError("all four legs must share one expiry")


def guard_no_working(state: dict) -> None:
    for name, trial in state.get("trials", {}).items():
        for step in ("open", "close", "flatten"):
            st = (trial.get(step) or {}).get("status")
            if st in ("working", "placing"):
                raise ProbeGuardError(
                    f"trial {name!r} has a '{step}' step marked {st!r}: run 'status' and 'cancel' "
                    "(and check TWS) before anything else")


def guard_spy_book(positions, open_trades, owned_con_ids: set[int],
                   trade_con_ids: set[int] | None = None) -> None:
    """The broker's book must not collide with this trial, and must be known.

    `positions is None` means the broker would not answer: unknown, so refuse. A position on a
    contract the trial is about to trade (`trade_con_ids`) that the probe does not own is refused.
    Positions on OTHER contracts (the operator's own QQQ / SPY legs) are left alone: the probe only
    ever acts on its own trial's contracts, and its leg-out uses positions per conId. A working
    order on a probe underlying that is not a probe order is refused (it could interact)."""
    if positions is None:
        raise ProbeGuardError("could not read positions from the broker (unknown) -- refusing")
    for p in positions:
        c = p.contract
        if (trade_con_ids and c.conId in trade_con_ids and getattr(c, "secType", "") == "OPT" and p.position
                and c.conId not in owned_con_ids):
            raise ProbeGuardError(
                f"a {c.symbol} option position the probe does not own exists on a contract this trial "
                f"would trade ({c.strike}{c.right} {c.lastTradeDateOrContractMonth}, {p.position}) -- resolve it first")
    for t in open_trades or []:
        if t.contract.symbol in PROBE_SYMBOLS:
            ref = str(getattr(t.order, "orderRef", "") or "")
            if not ref.startswith(TAG):
                raise ProbeGuardError(
                    f"a working {t.contract.symbol} order that is not a probe order exists (id {t.order.orderId}, "
                    f"ref {ref!r}) -- resolve it first")


def guard_account_values(values: dict, init_margin: float | None) -> None:
    """NetLiquidation >= MIN_NET_LIQ (clear of PDT) and ExcessLiquidity >= MARGIN_MULT x the
    what-if initial margin of the structure. Missing numbers refuse."""
    try:
        net_liq = float(values["NetLiquidation"])
        excess = float(values["ExcessLiquidity"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProbeGuardError(f"account values unavailable ({e!r}) -- refusing") from e
    if net_liq < MIN_NET_LIQ:
        raise ProbeGuardError(f"NetLiquidation {net_liq:,.0f} < {MIN_NET_LIQ:,.0f} (PDT) -- refusing")
    if init_margin is not None and excess < MARGIN_MULT * abs(init_margin):
        raise ProbeGuardError(
            f"ExcessLiquidity {excess:,.0f} < {MARGIN_MULT:g}x the what-if initial margin "
            f"{abs(init_margin):,.0f} -- refusing")


# --- Broker edge: connection, quotes, errors --------------------------------

# Replaced by the fake clock in the tests; production uses the real ones.
_sleep = asyncio.sleep
_clock = time.monotonic


async def connect_probe(data_type: int = 3) -> IBKRClient:
    """Own client id (96), the paper/live gate, and the market-data type for THIS session only (the
    bot has its own session). In RTH types 3 and 4 are the same ~15-minute-delayed stream."""
    ibkr = IBKRClient(IBKREnvConfig(ibkr_client_id=PROBE_CLIENT_ID))
    if not await ibkr.connect():
        raise ProbeGuardError("IBKRClient.connect() failed or refused (see logs: this includes "
                              "the built-in paper/live account gate)")
    guard_account(ibkr)
    guard_client_id(ibkr)
    ibkr.ib.reqMarketDataType(data_type)
    return ibkr


class ErrorRecorder:
    """Every error the Gateway sends, with its code. `IBKRClient._on_error` keeps only a few
    order-related codes (it ignores 161, 10147, 10148, 399, 10091 and 10167, all of which matter
    here), so the probe listens on `ib.errorEvent` itself."""

    _QUIET = {2103, 2104, 2105, 2106, 2107, 2108, 2119, 2158}      # farm-status chatter

    def __init__(self, ib, ev: Evidence) -> None:
        self.items: list[dict] = []
        self._ib, self._ev = ib, ev
        ib.errorEvent += self._on

    def _on(self, req_id, code, text, contract=None) -> None:
        if code in self._QUIET:
            return
        item = {"t": _clock(), "req_id": req_id, "code": code, "text": str(text)[:300],
                "con_id": getattr(contract, "conId", None)}
        self.items.append(item)
        del self.items[:-500]
        self._ev.emit("order_error", **{k: v for k, v in item.items() if k != "t"})

    def for_order(self, order_id: int) -> list[dict]:
        return [i for i in self.items if i["req_id"] == order_id]

    def detach(self) -> None:
        try:
            self._ib.errorEvent -= self._on
        except Exception:  # noqa: BLE001
            pass


def _tick(t) -> dict:
    g = lambda a: (getattr(t, a, None) if t is not None else None)  # noqa: E731
    return {"bid": g("bid"), "ask": g("ask"), "last": g("last"), "mdt": g("marketDataType")}


class QuoteSampler:
    """Streams the four legs, the underlying and the BAG for the whole trial (and the truth tail),
    emitting a `quote_sample` every SAMPLE_EVERY_S. Subscriptions are always cancelled in stop()."""

    def __init__(self, ibkr, ev: Evidence, leg_contracts: dict, spot_contract, bag_contract,
                 parity: tuple | None = None) -> None:
        """`parity` = (call_contract, put_conId, strike): when the stock has no price, the underlying is
        estimated from this call (extra subscription) and that put (one of the legs)."""
        self._ibkr, self._ev = ibkr, ev
        self._legs, self._spot, self._bag = leg_contracts, spot_contract, bag_contract
        self._parity = parity
        self._tickers: dict = {}
        self.phase = "ladder"
        self._stop = False
        # (monotonic time, {conId: (bid, ask)}) every sample: lets the ladder price from the quotes
        # a delayed feed WOULD show (the quotes of `lag` seconds ago) while orders match live.
        self.history: deque = deque(maxlen=int(4 * 3600 / SAMPLE_EVERY_S))

    def start(self) -> None:
        ib = self._ibkr.ib
        for con_id, c in self._legs.items():
            self._tickers[("leg", con_id)] = ib.reqMktData(c, "", False, False)
        if self._spot is not None:
            self._tickers[("spot", 0)] = ib.reqMktData(self._spot, "", False, False)
        if self._bag is not None:
            self._tickers[("bag", 0)] = ib.reqMktData(self._bag, "", False, False)
        if self._parity is not None:
            self._tickers[("parity_call", 0)] = ib.reqMktData(self._parity[0], "", False, False)

    def snapshot(self) -> dict:
        legs = {cid: _tick(self._tickers.get(("leg", cid))) for cid in self._legs}
        return {"legs": legs, "spot": _tick(self._tickers.get(("spot", 0))),
                "bag": _tick(self._tickers.get(("bag", 0))), "t": _clock()}

    def quotes(self) -> dict:
        """{conId: (bid, ask)} as natural_cost() takes them."""
        return {cid: (v["bid"], v["ask"]) for cid, v in self.snapshot()["legs"].items()}

    def record(self) -> None:
        self.history.append((_clock(), self.quotes()))

    def span(self) -> float:
        """Seconds of history held."""
        return (self.history[-1][0] - self.history[0][0]) if len(self.history) > 1 else 0.0

    def lagged_quotes(self, lag_s: float):
        """The newest recorded quotes that are at least `lag_s` old -> (quotes, actual age), or
        None if the history does not reach back that far."""
        now = _clock()
        best = None
        for t, q in self.history:
            if now - t >= lag_s:
                best = (t, q)
            else:
                break
        return None if best is None else (best[1], round(now - best[0], 1))

    def spot(self) -> float | None:
        s = self.snapshot()["spot"]
        for key in ("last",):
            if _finite(s[key]) and s[key] > 0:
                return float(s[key])
        if _finite(s["bid"]) and _finite(s["ask"]) and s["ask"] > 0 and s["bid"] > 0:
            return (s["bid"] + s["ask"]) / 2.0
        if self._parity is not None:
            call_t = _tick(self._tickers.get(("parity_call", 0)))
            put_t = _tick(self._tickers.get(("leg", self._parity[1])))
            return parity_spot(self._parity[2], (call_t["bid"], call_t["ask"]), (put_t["bid"], put_t["ask"]))
        return None

    def bag_ticked(self) -> bool:
        b = self.snapshot()["bag"]
        return any(_finite(b[k]) and b[k] != 0 for k in ("bid", "ask"))

    async def run(self) -> None:
        while not self._stop:
            self.record()
            snap = self.snapshot()
            self._ev.emit("quote_sample", phase=self.phase, spot=snap["spot"],
                          legs={str(k): v for k, v in snap["legs"].items()}, bag=snap["bag"])
            await _sleep(SAMPLE_EVERY_S)

    def stop(self) -> None:
        self._stop = True
        ib = self._ibkr.ib
        for key, _t in list(self._tickers.items()):
            contract = (self._legs.get(key[1]) if key[0] == "leg" else
                        self._spot if key[0] == "spot" else
                        self._parity[0] if key[0] == "parity_call" else self._bag)
            try:
                ib.cancelMktData(contract)
            except Exception:  # noqa: BLE001
                pass
        self._tickers.clear()


async def wait_for_quotes(sampler: QuoteSampler, legs: list[dict], timeout: float = 30.0) -> bool:
    """Wait (at most `timeout`) until the closing natural can be stated from the legs' quotes."""
    t0 = _clock()
    while _clock() - t0 < timeout:
        if natural_cost(legs, sampler.quotes()) is not None:
            return True
        await _sleep(1.0)
    return natural_cost(legs, sampler.quotes()) is not None


# --- Orders: rest, cancel, confirm -------------------------------------------

def _fill_rows(trade) -> list[dict]:
    rows = []
    for f in getattr(trade, "fills", []) or []:
        ex, c = f.execution, f.contract
        rep = getattr(f, "commissionReport", None)
        rows.append({"exec_id": getattr(ex, "execId", ""), "perm_id": getattr(ex, "permId", 0),
                     "sec_type": getattr(c, "secType", ""), "con_id": getattr(c, "conId", None),
                     "side": getattr(ex, "side", ""), "price": getattr(ex, "price", None),
                     "shares": getattr(ex, "shares", None), "exec_time": str(getattr(f, "time", "")),
                     "commission": getattr(rep, "commission", None) if rep else None})
    return rows


async def executions_for(ibkr, trade) -> list[dict]:
    """What the broker says executed for this order, by permId or orderId (ClientId-filtered).
    The status alone is not enough: ib_insync marks an order Cancelled LOCALLY on some errors
    (10148, 161) that do not prove the order did not fill."""
    try:
        from ib_insync import ExecutionFilter
        fills = await ibkr.ib.reqExecutionsAsync(ExecutionFilter(clientId=PROBE_CLIENT_ID))
    except Exception as e:  # noqa: BLE001
        log.warning("probe_executions_unavailable", error=str(e)[:200])
        fills = []
    perm = getattr(trade.orderStatus, "permId", 0) or 0
    oid = trade.order.orderId
    rows = []
    for f in fills or []:
        ex = f.execution
        if (perm and getattr(ex, "permId", 0) == perm) or getattr(ex, "orderId", None) == oid:
            rows.append({"exec_id": ex.execId, "perm_id": ex.permId, "sec_type": f.contract.secType,
                         "con_id": f.contract.conId, "side": ex.side, "price": ex.price,
                         "shares": ex.shares, "exec_time": str(getattr(f, "time", ""))})
    return rows


async def cancel_and_confirm(ibkr, trade, ev: Evidence) -> str:
    """Request the cancel, wait up to CANCEL_CONFIRM_TIMEOUT_S for a terminal status, then ALWAYS
    check the executions. Returns: 'cancelled' | 'filled' (the cancel lost the race) |
    'cancel_race_filled' (Cancelled on paper, executions exist) | 'partial' | 'cancel_unconfirmed'."""
    oid = trade.order.orderId
    ev.emit("cancel_requested", order_id=oid)
    t0 = _clock()
    await ibkr.cancel_order(trade)
    status = trade.orderStatus.status
    while status not in _DONE and _clock() - t0 < CANCEL_CONFIRM_TIMEOUT_S:
        await _sleep(1.0)
        status = trade.orderStatus.status
    latency = round(_clock() - t0, 1)
    if status not in _DONE:
        ev.emit("cancel_confirmed", order_id=oid, final_status=status, latency_s=latency, confirmed=False)
        return "cancel_unconfirmed"
    execs = await executions_for(ibkr, trade)
    filled = float(getattr(trade.orderStatus, "filled", 0) or 0)
    remaining = float(getattr(trade.orderStatus, "remaining", 0) or 0)
    ev.emit("cancel_confirmed", order_id=oid, final_status=status, latency_s=latency, confirmed=True,
            executions=len(execs), filled=filled)
    if status == "Filled":
        return "filled"
    if filled > 0 and remaining > 0:
        return "partial"
    if filled > 0 or execs or getattr(trade, "fills", None):
        return "cancel_race_filled"
    return "cancelled"


async def place_and_rest(ibkr, contract, order, ref: str, rest_s: float, ev: Evidence,
                         errors: ErrorRecorder, k: int) -> dict:
    """Place ONE order, let it rest `rest_s` (polling once a second, emitting status changes), then
    cancel-and-confirm. Returns the rung outcome dict (kind: filled | rested_out |
    cancel_race_filled | partial | inactive | rejected | cancel_unconfirmed | place_failed)."""
    order.orderRef = ref
    trade = await ibkr.place_order(contract, order)
    if trade is None:
        return {"kind": "place_failed", "order_id": None}
    oid = trade.order.orderId
    ev.emit("order_placed", k=k, order_id=oid, order_ref=ref, action=order.action,
            qty=order.totalQuantity, limit=order.lmtPrice,
            sec_type=getattr(contract, "secType", ""), con_id=getattr(contract, "conId", None))
    t0 = _clock()
    last = None
    seen_fills = 0

    def _out(kind: str, **extra) -> dict:
        fills = _fill_rows(trade)
        bag = [f for f in fills if f["sec_type"] == "BAG"]
        price = (bag[0]["price"] if bag else getattr(trade.orderStatus, "avgFillPrice", None) or None)
        return {"kind": kind, "order_id": oid, "perm_id": getattr(trade.orderStatus, "permId", None),
                "t_to_fill_s": round(_clock() - t0, 1), "avg_fill_price": price, "fills": fills,
                "errors": errors.for_order(oid), "advanced_error": getattr(trade, "advancedError", ""),
                "trade_log": [str(getattr(e, "message", "")) for e in getattr(trade, "log", [])][-8:],
                **extra}

    while True:
        await _sleep(1.0)
        st = trade.orderStatus
        if st.status != last:
            last = st.status
            ev.emit("order_status", order_id=oid, status=st.status, filled=st.filled,
                    remaining=st.remaining, avg_fill_price=st.avgFillPrice,
                    why_held=getattr(st, "whyHeld", ""), mkt_cap_price=getattr(st, "mktCapPrice", None))
        rows = getattr(trade, "fills", []) or []
        for f in rows[seen_fills:]:
            ex = f.execution
            ev.emit("fill", order_id=oid, exec_id=ex.execId, perm_id=ex.permId, sec_type=f.contract.secType,
                    con_id=f.contract.conId, side=ex.side, price=ex.price, shares=ex.shares,
                    exec_time=str(getattr(f, "time", "")))
        seen_fills = len(rows)
        if st.status == "Filled":
            for _ in range(3):                      # let the commission reports arrive
                await _sleep(1.0)
            for f in getattr(trade, "fills", []) or []:
                rep = getattr(f, "commissionReport", None)
                if rep is not None and getattr(rep, "commission", None):
                    ev.emit("commission", order_id=oid, exec_id=f.execution.execId, commission=rep.commission)
            return _out("filled")
        if st.status == "Inactive":
            await cancel_and_confirm(ibkr, trade, ev)
            return _out("inactive")
        if st.status in ("Cancelled", "ApiCancelled"):
            # Not our cancel: the broker or a local error ended it. A fill may still have happened.
            execs = await executions_for(ibkr, trade)
            if execs or float(st.filled or 0) > 0:
                return _out("cancel_race_filled")
            return _out("rejected")
        if _clock() - t0 >= rest_s:
            break
    result = await cancel_and_confirm(ibkr, trade, ev)
    if result == "cancelled":
        return _out("rested_out")
    return _out(result)


# --- Trial state helpers -----------------------------------------------------

def trial_legs(tstate: dict, which: str) -> list[dict]:
    """The trial's four legs with the OPENING actions ('open') or the CLOSING ones ('close')."""
    key = "open_action" if which == "open" else "close_action"
    return [{"conId": l["conId"], "strike": l["strike"], "right": l["right"], "expiry": l["expiry"],
             "action": l[key], "ratio": 1} for l in tstate["legs"]]


def trial_closed(t: dict) -> bool:
    """W112: is the trial FLAT? A trial opened with a lot count tracks ``remaining`` (a staged close leaves
    some); an older one (no ``remaining``) is closed once any closing step filled."""
    if "remaining" in t:
        return int(t["remaining"]) <= 0
    return any((t.get(s) or {}).get("status") == "filled" for s in ("close", "flatten"))


def owned_con_ids(state: dict, *, active_only: bool = True) -> set[int]:
    """conIds of every trial whose open filled and which has not been closed/flattened."""
    out: set[int] = set()
    for t in state.get("trials", {}).values():
        if (t.get("open") or {}).get("status") != "filled":
            continue
        closed = trial_closed(t)
        if active_only and closed:
            continue
        out |= {int(l["conId"]) for l in t["legs"]}
    return out


def new_trial(name: str, kind: str, structure: dict, data_type: int) -> dict:
    return {"name": name, "kind": kind, "symbol": SYMBOL, "created": now_et().isoformat(),
            "expiry": structure["expiry"],
            "dte": structure["dte"], "width": structure["width"], "data_type": data_type,
            "quantity": QUANTITY, "remaining": QUANTITY,
            "legs": structure["legs"]}


# --- Structure selection (pure) ------------------------------------------------

def select_structure(kind: str, spot: float, expirations: list[str], strikes: list[float],
                     today, short_call: float | None = None) -> dict:
    """Pick the four legs. `expirations` are 'YYYYMMDD' strings, `strikes` the listed strikes.

    nobid: like pin, but the short call ~1.5% out of the money (see below).
    far: the nearest expiry with 3..7 days to go, shorts ~1.5% out of the money each side, width 3.
    pin: today's expiry (0DTE), short put at the lowest strike >= spot and <= spot + 1 (so the
         structure's natural stays below the width), short call at the lowest strike >= spot + 3,
         width 3: the shape of the 2026-10-23 incident (a short put at the money at the close)."""
    from datetime import datetime as _dt
    strikeset = {round(float(s), 2) for s in strikes}

    def listed(x: float) -> float:
        x = round(x, 2)
        if x not in strikeset:
            raise ProbeGuardError(f"strike {x} is not listed for the chosen expiry")
        return x

    dated = sorted((( _dt.strptime(e, "%Y%m%d").date() - today).days, e) for e in expirations)
    if kind == "far":
        cands = [(d, e) for d, e in dated if 3 <= d <= 7]
        if not cands:
            raise ProbeGuardError("no expiry with 3-7 days to go")
        dte, expiry = cands[0]
        puts_out = [s for s in strikeset if s <= spot * 0.985]
        calls_out = [s for s in strikeset if s >= spot * 1.015]
        if not puts_out or not calls_out:
            raise ProbeGuardError("no strikes ~1.5% out of the money")
        sp, sc = max(puts_out), min(calls_out)
    elif kind in ("pin", "nobid"):
        cands = [(d, e) for d, e in dated if d == 0]
        if not cands:
            raise ProbeGuardError("no expiry today (0DTE): a pin trial cannot be built")
        dte, expiry = cands[0]
        puts_itm = [s for s in strikeset if spot <= s <= spot + 1.0]
        # pin: a short call 3 out; nobid: ~1.5% out, so the long wing (3 further) loses its bid as the
        # afternoon goes on and the closing combo has to SELL a leg nobody bids for (the QQQ condor's state)
        calls_out = [s for s in strikeset if s >= (spot + 3.0 if kind == "pin" else spot * 1.015)]
        if short_call is not None:
            if kind != "nobid":
                raise ProbeGuardError("--short-call is only for the nobid kind")
            if round(float(short_call), 2) not in strikeset or short_call < spot:
                raise ProbeGuardError(f"--short-call {short_call} is not a listed strike at or above the spot")
            calls_out = [round(float(short_call), 2)]
        if not puts_itm or not calls_out:
            raise ProbeGuardError("no short put in [spot, spot+1] or no suitable short call")
        sp, sc = min(puts_itm), min(calls_out)
    else:
        raise ProbeGuardError(f"unknown kind {kind!r}")
    lp, lc = listed(sp - MAX_WIDTH), listed(sc + MAX_WIDTH)
    sp, sc = listed(sp), listed(sc)
    legs = [
        {"strike": lp, "right": "P", "expiry": expiry, "open_action": "BUY", "close_action": "SELL"},
        {"strike": sp, "right": "P", "expiry": expiry, "open_action": "SELL", "close_action": "BUY"},
        {"strike": sc, "right": "C", "expiry": expiry, "open_action": "SELL", "close_action": "BUY"},
        {"strike": lc, "right": "C", "expiry": expiry, "open_action": "BUY", "close_action": "SELL"},
    ]
    return {"kind": kind, "expiry": expiry, "dte": dte, "width": MAX_WIDTH, "legs": legs}


def open_net(open_legs: list[dict], quotes: dict, side: str) -> float | None:
    """The signed net price of the OPENING combo under the always-BUY convention (a credit is
    negative): side 'natural' = BUY legs at their ask, SELL legs at their bid; 'mid' = every leg at
    its mid. None if a needed quote side is missing."""
    total = 0.0
    for leg in open_legs:
        bid, ask = quotes.get(leg["conId"], (None, None))
        is_buy = str(leg["action"]).upper() == "BUY"
        if side == "mid":
            if is_buy and _finite(ask) and ask > 0 and not (_finite(bid) and bid >= 0):
                px = ask                    # a far wing we BUY may have no bid: its mid is its offer
            elif _finite(bid) and bid >= 0 and _finite(ask) and ask > 0:
                px = (bid + ask) / 2.0
            else:
                return None
        elif is_buy:
            if not (_finite(ask) and ask > 0):
                return None
            px = ask
        else:
            if not (_finite(bid) and bid >= 0):
                return None
            px = bid
        total += px if is_buy else -px
    return round(total, 4)


def two_sided(legs: list[dict], quotes: dict) -> bool:
    for leg in legs:
        bid, ask = quotes.get(leg["conId"], (None, None))
        if not (_finite(bid) and bid >= 0 and _finite(ask) and ask > 0):
            return False
    return True


# --- The combo ladder (open and close share it) --------------------------------

_BAND_RE = None


def parse_band_reject(errors: list[dict] | None) -> float | None:
    """IB's price-band reject: "We cannot accept an order at a limit price at or more aggressive than
    1.325". The bot handles the same reject (exit_band); the probe's ladder must too. Returns the
    band, or None if these errors are not one."""
    global _BAND_RE
    import re
    if _BAND_RE is None:
        _BAND_RE = re.compile(r"at or more aggressive than\s+(-?[0-9]+(?:\.[0-9]+)?)")
    for e in errors or []:
        m = _BAND_RE.search(str(e.get("text", "")))
        if m:
            return float(m.group(1))
    return None


_BAD_ENDS = ("inactive", "rejected", "cancel_unconfirmed", "place_failed")
_FILLED_ENDS = ("filled", "cancel_race_filled", "partial")


async def run_combo_ladder(ibkr, *, state: dict, tstate: dict, trial: str, step: str, bag,
                           next_limit, n_rungs: int, rest_s: float, ev: Evidence,
                           errors: ErrorRecorder, hard_cap: float | None) -> dict:
    """Rung by rung: price (via `next_limit(k, ctx) -> (limit | None, info)`), write the state AHEAD
    of the order, place, rest, cancel-and-confirm, and stop at the first fill or any trouble. At
    most one probe order is ever live. `hard_cap` is a last-line check that no combo limit exceeds
    the wing width (the structure's maximum loss)."""
    st = tstate.setdefault(step, {})
    st.clear()
    st.update({"status": "working", "rungs": [], "started": now_et().isoformat()})
    save_state(state)
    ctx = {"prev": None, "l3": None}
    outcome, filled_rung, fill_price = "unfilled", None, None
    for k in range(n_rungs):
        limit, info = await next_limit(k, ctx)
        if limit is None:
            if info.get("abort"):
                ev.emit("rung_abort", k=k, reason=info["abort"],
                        **{x: y for x, y in info.items() if x not in ("abort", "k")})
                outcome = "aborted_" + str(info["abort"])
                break
            ev.emit("rung_plan", **info, limit=None)          # a duplicate: nothing to place
            continue
        if ctx.get("band") is not None:                       # IB told us the most aggressive price it takes
            under = round(ctx["band"] - 0.01, 2)
            if limit > under:
                info["band_cap"] = under
                limit = under
        ev.emit("rung_plan", **info, limit=limit)
        if hard_cap is not None and limit > hard_cap + 1e-9:
            raise ProbeGuardError(f"rung {k}: limit {limit} exceeds the wing width {hard_cap}")
        ref = make_order_ref(trial, step, k)
        rung = {"k": k, "limit": limit, "ref": ref, "status": "placing"}
        st["rungs"].append(rung)
        save_state(state)                                      # write-ahead
        out = await place_and_rest(ibkr, bag, OrderBuilder.combo_limit(
            action="BUY", quantity=QUANTITY, limit_price=limit), ref, rest_s, ev, errors, k)
        rung.update({"status": out["kind"], "order_id": out.get("order_id")})
        save_state(state)
        ev.emit("rung_end", k=k, outcome=out["kind"], t_to_fill_s=out.get("t_to_fill_s"),
                avg_fill_price=out.get("avg_fill_price"), rest_s=rest_s, limit=limit,
                trade_log=out.get("trade_log"), errors=out.get("errors"),
                advanced_error=out.get("advanced_error"))
        if out["kind"] in _FILLED_ENDS:
            outcome, filled_rung, fill_price = out["kind"], k, out.get("avg_fill_price")
            break
        band = parse_band_reject(out.get("errors")) if out["kind"] == "rejected" else None
        if band is not None:
            # Not a failure of the ladder: the price was too aggressive for IB's band. Price under it.
            ev.emit("band_reject", k=k, limit=limit, band=band)
            ctx["band"] = band
            continue
        if out["kind"] in _BAD_ENDS:
            outcome = out["kind"]
            break
        ctx["prev"] = limit
        if k == 3:
            ctx["l3"] = limit
    status = ("filled" if outcome in ("filled", "cancel_race_filled") else
              "partial" if outcome == "partial" else "aborted")
    st.update({"status": status, "outcome": outcome, "filled_rung": filled_rung,
               "fill_price": fill_price, "ended": now_et().isoformat()})
    save_state(state)
    ev.emit("trial_end", step=step, outcome=outcome, filled_rung=filled_rung, fill_price=fill_price)
    return {"outcome": outcome, "filled_rung": filled_rung, "fill_price": fill_price}


def close_limit_fn(sampler: QuoteSampler, close_legs: list[dict], open_legs: list[dict],
                   width: float, dte: int, data_type: int, ev: Evidence, params: dict,
                   lag_s: float = 0.0, spot0: float | None = None):
    """The close ladder's pricing callback: the legs' natural at every rung, the bot's floor at the
    current spot, the bot's rung rule. A leg with no usable quote aborts the rung (the bot would take
    its no-quote path there, which is not what is being measured).

    `lag_s` > 0 emulates a DELAYED feed: the rung is priced from the quotes recorded `lag_s` ago, while
    the order still meets the live market. The live natural is recorded beside it (`natural_live`) so
    the report can tell whether the limit was truly marketable at the time it was placed."""
    seen = {"spot": spot0}

    async def next_limit(k: int, ctx: dict):
        snap = sampler.snapshot()
        live_q = {cid: (v["bid"], v["ask"]) for cid, v in snap["legs"].items()}
        nat_live = natural_cost(close_legs, live_q)
        lag_age = None
        if lag_s > 0:
            lagged = sampler.lagged_quotes(lag_s)
            if lagged is None:
                return None, {"abort": "no_lagged_quotes", "k": k}
            quotes, lag_age = lagged
        else:
            quotes = live_q
        nat = natural_cost(close_legs, quotes)
        if nat is None:
            return None, {"abort": "no_quotes", "k": k}
        spot = sampler.spot()
        if spot is not None:
            seen["spot"] = spot
        spot = seen["spot"]
        floor = intrinsic_floor(open_legs, spot, dte, width, data_type, cross=params["cross"],
                                near_dte=params["near_dte"], band_pct=params["band_pct"])
        limit, info = rung_limit(k, nat, ctx["prev"], ctx["l3"], width, floor, cross=params["cross"],
                                 step=params["step"], pct=params["pct"])
        info["cap_binds"] = bool(nat > width or (nat_live is not None and nat_live > width))
        info.update({"mid": mid_cost(close_legs, quotes), "spot": spot,
                     "intrinsic": (structure_intrinsic(open_legs, spot) if spot else None),
                     "width": width, "mdt": {str(c): v["mdt"] for c, v in snap["legs"].items()},
                     "bag_ticked": sampler.bag_ticked(), "natural_live": nat_live,
                     "lag_s": lag_s, "lag_age_s": lag_age,
                     "no_bid_sell_legs": sell_legs_without_bid(close_legs, live_q)})
        return limit, info
    return next_limit


def open_limit_fn(sampler: QuoteSampler, open_legs: list[dict], ev: Evidence):
    """The entry ladder's pricing callback: mid -> natural -> past natural, fresh quotes each rung.
    Refuses to open at a debit (a credit of at least OPEN_MIN_CREDIT is required)."""
    async def next_limit(k: int, ctx: dict):
        quotes = sampler.quotes()
        mid, nat = open_net(open_legs, quotes, "mid"), open_net(open_legs, quotes, "natural")
        if mid is None or nat is None:
            return None, {"abort": "no_quotes", "k": k}
        if nat > -OPEN_MIN_CREDIT:
            return None, {"abort": "credit_too_small", "k": k, "natural": nat}
        lim = entry_rung_limit(k, mid, nat)
        if ctx["prev"] is not None and lim <= ctx["prev"]:
            # Conceding credit raises the (negative) limit toward and past zero; a rung that does
            # not move (mid == natural) still steps one cent, so the ladder keeps climbing.
            lim = round(ctx["prev"] + 0.01, 2)
        return lim, {"k": k, "kind": "entry", "natural": nat, "mid": mid, "floor": None,
                     "raw": lim, "floor_bound": False, "cap_bound": False, "cap_binds": False,
                     "skipped_duplicate": False}
    return next_limit


# --- Commands: shared pieces ---------------------------------------------------

LEGOUT_RUNGS = 6
LEGOUT_REST_S = 60.0
LEGOUT_STEP = 0.05
LEGOUT_MAX_CONCESSION = 0.50


def load_params() -> dict:
    """The bot's exit settings as they are today (recorded in run_start so a config change shows)."""
    try:
        ex = load_settings(str(ROOT / "config.yaml")).exit
        return {"cross": float(ex.exit_cross_amount), "step": float(ex.exit_reprice_step),
                "pct": float(ex.exit_unfilled_escalation_pct), "rest": float(ex.exit_stop_pending_rest_seconds),
                "near_dte": int(ex.exit_near_expiry_dte), "band_pct": float(ex.exit_intrinsic_delayed_band_pct)}
    except Exception as e:  # noqa: BLE001 - fall back to the shipped defaults, and say so
        log.warning("probe_params_fallback", error=str(e)[:200])
        return {"cross": EXIT_CROSS, "step": REPRICE_STEP, "pct": ESC_PCT, "rest": REST_S,
                "near_dte": NEAR_EXPIRY_DTE, "band_pct": DELAYED_BAND_PCT}


async def qualify_legs(ibkr, legs: list[dict]) -> list[dict]:
    """legs [{strike, right, expiry, action, ...}] -> same dicts plus conId, ratio, _contract."""
    contracts = [ContractBuilder.option(SYMBOL, l["expiry"], l["strike"], l["right"]) for l in legs]
    qualified = await ibkr.qualify_contracts_batch(contracts)
    out = []
    for leg, q in zip(legs, qualified):
        if q is None:
            raise ProbeGuardError(f"contract qualification failed for leg {leg}")
        out.append({**leg, "conId": q.conId, "ratio": leg.get("ratio", 1), "_contract": q})
    return out


def bag_for(legs: list[dict]):
    return ContractBuilder.combo(
        SYMBOL, [{"conId": l["conId"], "action": l["action"], "ratio": l.get("ratio", 1)} for l in legs])


async def poll_spot(ibkr, stock, rounds: int = 12, delay: float = 0.5) -> float | None:
    """The underlying's price: last, else mid, else close (delayed data counts)."""
    price = None
    try:
        t = ibkr.ib.reqMktData(stock, "", False, False)
        for _ in range(rounds):
            await _sleep(delay)
            for cand in (t.last, (t.bid + t.ask) / 2 if _finite(t.bid) and _finite(t.ask) and t.bid > 0 and t.ask > 0 else None, t.close):
                if _finite(cand) and cand > 0:
                    price = float(cand)
                    break
            if price is not None:
                break
    finally:
        try:
            ibkr.ib.cancelMktData(stock)
        except Exception:  # noqa: BLE001
            pass
    return price


async def poll_quote(ibkr, contract, rounds: int = 10, delay: float = 0.3) -> tuple[float | None, float | None]:
    """One contract's raw (bid, ask): -1 is kept (it means 'nobody bids'), NaN becomes None."""
    bid = ask = None
    try:
        ibkr.ib.reqMktData(contract, "", False, False)
        for _ in range(rounds):
            await _sleep(delay)
            t = ibkr.ib.ticker(contract)
            if t is None:
                continue
            bid = t.bid if _finite(t.bid) else bid
            ask = t.ask if _finite(t.ask) else ask
            if bid is not None and ask is not None:
                break
    finally:
        try:
            ibkr.ib.cancelMktData(contract)
        except Exception:  # noqa: BLE001
            pass
    return bid, ask


def _last_known_spot() -> float | None:
    """The newest price of THIS run's underlying that an earlier run recorded (a starting guess for the
    parity search). Events without a symbol are SPY's (the probe's first underlying)."""
    for f in sorted(PROBE_DIR.glob("*/events.jsonl"), reverse=True):
        for line in reversed(f.read_text().splitlines()):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if (e.get("event") == "legs_selected" and _finite(e.get("spot")) and e["spot"] > 0
                    and e.get("symbol", "SPY") == SYMBOL):
                return float(e["spot"])
    return SPOT_HINT if _finite(SPOT_HINT) and SPOT_HINT and SPOT_HINT > 0 else None


async def estimate_spot_from_options(ibkr, expirations: list[str], strikes: list[float],
                                     guess: float | None, today) -> float | None:
    """The underlying from the options alone (put-call parity at the strike nearest the estimate,
    refined once), for when the stock has no price. Uses the nearest expiry that has not passed."""
    from datetime import datetime as _dt
    live = [e for e in sorted(expirations) if _dt.strptime(e, "%Y%m%d").date() >= today]
    if not live or not strikes:
        return None
    exp = live[0]
    est = guess if guess else sorted(strikes)[len(strikes) // 2]
    for _ in range(2):
        k = min(strikes, key=lambda x: abs(x - est))
        call, put = await qualify_legs(ibkr, [
            {"strike": k, "right": "C", "expiry": exp, "action": "BUY"},
            {"strike": k, "right": "P", "expiry": exp, "action": "BUY"}])
        cq = await poll_quote(ibkr, call["_contract"])
        pq = await poll_quote(ibkr, put["_contract"])
        got = parity_spot(k, cq, pq)
        if got is None:
            return None
        if abs(got - k) <= 1.0:
            return got
        est = got
    return est


async def fetch_chain(ibkr):
    """(stock contract, spot, expirations, strikes) from the broker's option-parameter listing. Not
    OptionsChainService: for a SPY 0DTE that subscribes to hundreds of strikes, its model greeks are
    unreliable at DTE 0 and its liquidity filter drops cheap wings."""
    stock = await ibkr.qualify_contract(ContractBuilder.stock(SYMBOL))
    if stock is None:
        raise ProbeGuardError("could not qualify the underlying")
    chains = await ibkr.ib.reqSecDefOptParamsAsync(SYMBOL, "", "STK", stock.conId)
    pick = [c for c in chains if getattr(c, "tradingClass", "") == SYMBOL]
    smart = [c for c in pick if getattr(c, "exchange", "") == "SMART"]
    chain = (smart or pick or [None])[0]
    if chain is None:
        raise ProbeGuardError("no SPY option chain listed")
    expirations, strikes = sorted(chain.expirations), sorted(float(s) for s in chain.strikes)
    spot = await poll_spot(ibkr, stock)
    if spot is None:
        # The stock feed is silent (no entitlement, or a competing session). The options are quoted:
        # derive the underlying from a call/put pair instead of refusing.
        spot = await estimate_spot_from_options(ibkr, expirations, strikes, _last_known_spot(), now_et().date())
        if spot is not None:
            log.warning("probe_spot_from_options", spot=spot, note="no stock price; put-call parity estimate")
    if spot is None:
        raise ProbeGuardError("no price for the underlying: the stock is silent and the options did not "
                              "give a usable call/put pair")
    return stock, spot, expirations, strikes


async def whatif_margin(ibkr, bag, order) -> dict:
    """Copied from the W53 probe: a what-if margin preview, nothing is placed."""
    order.whatIf = True
    try:
        state = await ibkr.ib.whatIfOrderAsync(bag, order)
    finally:
        order.whatIf = False
    if state is None:
        return {"ok": False}
    def _f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if abs(v) < 1e300 else None             # IB's "unset" is the largest double
    return {"ok": True, "init_margin": _f(getattr(state, "initMarginChange", None)),
            "maint_margin": _f(getattr(state, "maintMarginChange", None)),
            "commission": _f(getattr(state, "commission", None)),
            "warning": getattr(state, "warningText", None)}


async def fresh_positions(ibkr):
    return await ibkr.get_positions_fresh()


def _pos_of(positions, con_id: int) -> float:
    return sum(float(p.position) for p in positions or [] if p.contract.conId == con_id)


def positions_snapshot(positions, con_ids: set[int]) -> dict:
    return {str(c): _pos_of(positions, c) for c in sorted(con_ids)}


def assert_reduce_only(action: str, position: float) -> None:
    """A BUY may only reduce a short, a SELL only a long: leg-out must never open a position."""
    if action.upper() == "BUY" and not position < 0:
        raise ProbeGuardError(f"BUY would not reduce a short (position {position}) -- refusing")
    if action.upper() == "SELL" and not position > 0:
        raise ProbeGuardError(f"SELL would not reduce a long (position {position}) -- refusing")


async def _probe_orders_working(ibkr) -> list:
    trades = await ibkr.get_all_open_trades()
    if trades is None:
        raise ProbeGuardError("could not read the broker's working orders (unknown) -- refusing")
    return [t for t in trades
            if str(getattr(t.order, "orderRef", "") or "").startswith(TAG) and not t.isDone()]


# --- Commands ------------------------------------------------------------------

async def _prepare(ibkr, state: dict, ev: Evidence, owned: set[int], *, init_margin: float | None = None,
                   check_account: bool = True, trade_con_ids: set[int] | None = None) -> None:
    positions = await fresh_positions(ibkr)
    open_trades = await ibkr.get_all_open_trades()
    guard_spy_book(positions, open_trades, owned, trade_con_ids)
    if check_account:
        values = await ibkr.get_account_values()
        guard_account_values(values, init_margin)
        ev.emit("account", net_liq=values.get("NetLiquidation"), excess_liq=values.get("ExcessLiquidity"),
                buying_power=values.get("BuyingPower"), init_margin=init_margin)


async def cmd_preflight(args) -> dict:
    use_quantity(int(getattr(args, "quantity", 1) or 1))
    """Read-only: account, the structures the probe would build, whether the BAG ticks, a what-if
    margin preview, and the ladder it would walk. Never calls placeOrder."""
    ev = Evidence("preflight")
    params = load_params()
    ibkr = await connect_probe(args.data_type)
    errors = ErrorRecorder(ibkr.ib, ev)
    out: dict = {"structures": {}}
    try:
        ev.emit("run_start", argv=sys.argv[1:], client_id=ibkr.client_id, account=ibkr.session_account,
                data_type=args.data_type, params=params)
        values = await ibkr.get_account_values()
        guard_account_values(values, None)
        ev.emit("account", net_liq=values.get("NetLiquidation"), excess_liq=values.get("ExcessLiquidity"))
        stock, spot, expirations, strikes = await fetch_chain(ibkr)
        today = now_et().date()
        for kind in ("far", "pin"):
            try:
                structure = select_structure(kind, spot, expirations, strikes, today)
                guard_caps([{**l, "action": l["open_action"]} for l in structure["legs"]])
            except ProbeGuardError as e:
                out["structures"][kind] = {"selected": False, "why": str(e)}
                ev.emit("legs_selected", kind=kind, selected=False, why=str(e))
                continue
            legs = await qualify_legs(ibkr, [{**l, "action": l["open_action"]} for l in structure["legs"]])
            open_legs = [{k: v for k, v in l.items() if k != "_contract"} for l in legs]
            close_legs = [{**l, "action": "SELL" if l["action"] == "BUY" else "BUY"} for l in open_legs]
            bag = bag_for(close_legs)
            sampler = QuoteSampler(ibkr, ev, {l["conId"]: l["_contract"] for l in legs}, stock, bag)
            sampler.start()
            try:
                await _sleep(8.0)
                quotes = sampler.quotes()
                nat = natural_cost(close_legs, quotes)
                ticked = sampler.bag_ticked()
                ev.emit("bag_probe", kind=kind, ticked=ticked, **sampler.snapshot()["bag"])
            finally:
                sampler.stop()
            open_nat = open_net(open_legs, quotes, "natural")
            margin = None
            if open_nat is not None:
                order = OrderBuilder.combo_limit(action="BUY", quantity=QUANTITY, limit_price=round(open_nat, 2))
                margin = await whatif_margin(ibkr, bag_for(open_legs), order)
            info = {"selected": True, "expiry": structure["expiry"], "dte": structure["dte"],
                    "strikes": [l["strike"] for l in structure["legs"]], "two_sided": two_sided(close_legs, quotes),
                    "bag_ticks": ticked, "open_natural": open_nat, "close_natural": nat, "margin": margin,
                    "planned_close_ladder": (planned_ladder(nat, structure["width"],
                                                            intrinsic_floor(open_legs, spot, structure["dte"],
                                                                            structure["width"], args.data_type))
                                             if nat is not None else None)}
            out["structures"][kind] = info
            ev.emit("legs_selected", kind=kind, symbol=SYMBOL, spot=spot, **info)
        out["spot"] = spot
        return out
    finally:
        errors.detach()
        ev.emit("run_end", exit_code=0)
        await ibkr.disconnect()


async def bot_liveness(ibkr, tstate: dict, lots: int) -> str:
    """W112: what the BOT's own ``PositionReconciler.position_liveness`` says about this trial held at
    ``lots`` lots (the answer its exit path acts on: "live" lets a reverse combo go out, "partial" refuses
    it). Evidence only: any trouble is returned as text, never raised."""
    try:
        await fresh_positions(ibkr)                 # the bot reads its cached view: bring it up to date first
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._ibkr = ibkr
        legs = [{"strike": l["strike"], "right": l["right"], "expiry": str(l["expiry"]),
                 "action": l["open_action"]} for l in tstate["legs"]]
        trade = SimpleNamespace(trade_id=f"{TAG}-{tstate.get('name', '')}", symbol=tstate.get("symbol", SYMBOL),
                                quantity=lots, legs=json.dumps(legs), expiry=str(tstate["expiry"]),
                                strike=None, contract_type="iron_condor")
        return await rec.position_liveness(trade)
    except Exception as e:  # noqa: BLE001 - evidence, not a gate
        return f"error:{type(e).__name__}:{str(e)[:80]}"


async def cmd_open(args) -> dict:
    use_quantity(int(getattr(args, "quantity", 1) or 1))
    state = load_state()
    if args.trial in state["trials"]:
        raise ProbeGuardError(f"trial {args.trial!r} already exists -- pick a new name or inspect 'status'")
    guard_window("open", args.kind)
    guard_bot_stopped()
    guard_no_working(state)
    ev = Evidence("open", args.trial)
    params = load_params()
    ibkr = await connect_probe(args.data_type)
    errors = ErrorRecorder(ibkr.ib, ev)
    sampler = None
    task = None
    try:
        ev.emit("run_start", argv=sys.argv[1:], client_id=ibkr.client_id, account=ibkr.session_account,
                data_type=args.data_type, params=params)
        await _prepare(ibkr, state, ev, owned_con_ids(state))
        stock, spot, expirations, strikes = await fetch_chain(ibkr)
        structure = select_structure(args.kind, spot, expirations, strikes, now_et().date(),
                                     short_call=getattr(args, "short_call", None))
        raw_open = [{**l, "action": l["open_action"]} for l in structure["legs"]]
        guard_caps(raw_open)
        guard_no_collision(raw_open)
        legs = await qualify_legs(ibkr, raw_open)
        open_legs = [{k: v for k, v in l.items() if k != "_contract"} for l in legs]
        structure["legs"] = [{**l, "conId": legs[i]["conId"]} for i, l in enumerate(structure["legs"])]
        bag = bag_for(open_legs)
        ev.emit("legs_selected", kind=args.kind, symbol=SYMBOL, spot=spot, expiry=structure["expiry"], dte=structure["dte"],
                width=structure["width"], legs=structure["legs"])
        sampler = QuoteSampler(ibkr, ev, {l["conId"]: l["_contract"] for l in legs}, stock, bag)
        sampler.start()
        task = asyncio.ensure_future(sampler.run())
        t0 = _clock()
        while _clock() - t0 < 30 and open_net(open_legs, sampler.quotes(), "natural") is None:
            await _sleep(1.0)
        quotes = sampler.quotes()
        if open_net(open_legs, quotes, "natural") is None or open_net(open_legs, quotes, "mid") is None:
            raise ProbeGuardError("the legs did not all quote (an ask on every leg to buy, a bid on every "
                                  "leg to sell) within 30 s -- not opening")
        nat = open_net(open_legs, quotes, "natural")
        margin = await whatif_margin(ibkr, bag, OrderBuilder.combo_limit(
            action="BUY", quantity=QUANTITY, limit_price=round(nat, 2)))
        _check = margin.get("init_margin")
        await _prepare(ibkr, state, ev, owned_con_ids(state), init_margin=_check,
                       trade_con_ids={int(l["conId"]) for l in legs})
        tstate = new_trial(args.trial, args.kind, structure, args.data_type)
        state["trials"][args.trial] = tstate
        result = await run_combo_ladder(
            ibkr, state=state, tstate=tstate, trial=args.trial, step="open", bag=bag,
            next_limit=open_limit_fn(sampler, open_legs, ev), n_rungs=len(OPEN_RUNG_FRACTIONS),
            rest_s=OPEN_REST_S, ev=ev, errors=errors, hard_cap=None)
        positions = await fresh_positions(ibkr)
        ev.emit("trial_end", step="open_positions", positions=positions_snapshot(
            positions, {int(l["conId"]) for l in structure["legs"]}))
        if result["outcome"] in ("filled", "cancel_race_filled"):
            ev.emit("quantity_check", step="open", lots=QUANTITY,
                    positions=positions_snapshot(positions, {int(l["conId"]) for l in structure["legs"]}),
                    bot_liveness=await bot_liveness(ibkr, tstate, QUANTITY),
                    bot_liveness_one_lot_more=await bot_liveness(ibkr, tstate, QUANTITY + 1))
        if result["outcome"] not in ("filled", "cancel_race_filled"):
            raise ProbeGuardError(f"open ended {result['outcome']!r}: nothing to close "
                                  "(run 'status' and 'cancel' if an order may be left)")
        return {"trial": args.trial, **result}
    finally:
        if sampler:
            sampler.stop()
        if task:
            task.cancel()
        errors.detach()
        ev.emit("run_end")
        await ibkr.disconnect()


def _positions_match_open(tstate: dict, positions) -> tuple[bool, dict]:
    snap, ok = {}, True
    lots = float(tstate.get("remaining", 1))
    for l in tstate["legs"]:
        want = lots if l["open_action"] == "BUY" else -lots
        got = _pos_of(positions, int(l["conId"]))
        snap[str(l["conId"])] = got
        ok = ok and got == want
    return ok, snap


async def _run_close_like(ibkr, state, tstate, trial, step, ev, errors, params, *, rest_s, tail: bool,
                          emulate_delay_s: float = 0.0, require_nobid: bool = False,
                          exclude_unbid: bool = False) -> dict:
    """The close experiment, used by `close` (step 'close') and by `flatten` (step 'flatten')."""
    legs = await qualify_legs(ibkr, trial_legs(tstate, "open"))
    open_legs = [{k: v for k, v in l.items() if k != "_contract"} for l in legs]
    close_legs = [{**l, "action": "SELL" if l["action"] == "BUY" else "BUY"} for l in open_legs]
    stock = await ibkr.qualify_contract(ContractBuilder.stock(SYMBOL))
    spot0 = await poll_spot(ibkr, stock) if stock is not None else None   # before the sampler owns the stream
    bag = bag_for(close_legs)
    short_put = next((l for l in legs if l["right"] == "P" and l["action"] == "SELL"), None)
    parity = None
    if short_put is not None:
        call_at_k = (await qualify_legs(ibkr, [{"strike": short_put["strike"], "right": "C",
                                                "expiry": short_put["expiry"], "action": "BUY"}]))[0]
        parity = (call_at_k["_contract"], short_put["conId"], float(short_put["strike"]))
    sampler = QuoteSampler(ibkr, ev, {l["conId"]: l["_contract"] for l in legs}, stock, bag, parity=parity)
    sampler.start()
    task = asyncio.ensure_future(sampler.run())
    try:
        if not await wait_for_quotes(sampler, close_legs, 30.0):
            ev.emit("rung_abort", reason="no_quotes_at_start")
            tstate.setdefault(step, {}).update({"status": "aborted", "outcome": "aborted_no_quotes"})
            save_state(state)
            return {"outcome": "aborted_no_quotes", "filled_rung": None, "fill_price": None}
        await _sleep(2.5)                                         # the bot's own BAG-tick window
        ev.emit("bag_probe", ticked=sampler.bag_ticked(), **sampler.snapshot()["bag"])
        nobid_now = sell_legs_without_bid(close_legs, sampler.quotes())
        ev.emit("nobid_check", legs=nobid_now, required=require_nobid)
        if require_nobid and not nobid_now:
            tstate.setdefault(step, {}).update({"status": "aborted", "outcome": "aborted_nobid_condition_absent"})
            save_state(state)
            return {"outcome": "aborted_nobid_condition_absent", "filled_rung": None, "fill_price": None}
        order_bag, left_open = bag, []
        if exclude_unbid and nobid_now:
            # Leave a worthless, unbid LONG wing out of the closing combo: it expires on its own and
            # nobody would buy it. Only if it really is worthless; every short stays in the combo.
            q = sampler.quotes()
            valued = [c for c in nobid_now if not (_finite(q.get(c, (None, None))[1]) and q[c][1] <= WORTHLESS_ASK)]
            if valued:
                ev.emit("rung_abort", reason="unbid_leg_has_value", legs=valued)
                tstate.setdefault(step, {}).update({"status": "aborted", "outcome": "aborted_unbid_leg_has_value"})
                save_state(state)
                return {"outcome": "aborted_unbid_leg_has_value", "filled_rung": None, "fill_price": None}
            close_legs = [l for l in close_legs if l["conId"] not in nobid_now]
            if len(close_legs) < 2 or not any(l["action"] == "BUY" for l in close_legs):
                raise ProbeGuardError("excluding the unbid legs would leave nothing sensible to close")
            order_bag, left_open = bag_for(close_legs), list(nobid_now)
            ev.emit("excluded_legs", legs=left_open, remaining=[l["conId"] for l in close_legs])
        if emulate_delay_s > 0:
            ev.emit("emulate_delay_wait", seconds=emulate_delay_s, spot0=spot0)
            t_wait = _clock()
            while sampler.span() < emulate_delay_s and _clock() - t_wait < emulate_delay_s + 120:
                sampler.record()
                await _sleep(5.0)
            if sampler.span() < emulate_delay_s:
                ev.emit("rung_abort", reason="history_too_short", span=sampler.span())
                tstate.setdefault(step, {}).update({"status": "aborted", "outcome": "aborted_history_too_short"})
                save_state(state)
                return {"outcome": "aborted_history_too_short", "filled_rung": None, "fill_price": None}
        result = await run_combo_ladder(
            ibkr, state=state, tstate=tstate, trial=trial, step=step, bag=order_bag,
            next_limit=close_limit_fn(sampler, close_legs, open_legs, float(tstate["width"]),
                                      int(tstate["dte"]), int(tstate["data_type"]), ev, params,
                                      lag_s=emulate_delay_s, spot0=spot0),
            n_rungs=N_RUNGS, rest_s=rest_s, ev=ev, errors=errors, hard_cap=float(tstate["width"]))
        if left_open:                                              # run_combo_ladder resets the step's state
            tstate[step]["left_open"] = left_open
            save_state(state)
        if tail:
            sampler.phase = "tail"
            ev.emit("tail_start", seconds=TRUTH_TAIL_S)
            t0 = _clock()
            while _clock() - t0 < TRUTH_TAIL_S:
                await _sleep(5.0)
        return result
    finally:
        sampler.stop()
        task.cancel()


async def cmd_close(args) -> dict:
    state = load_state()
    tstate = state["trials"].get(args.trial)
    if not tstate or (tstate.get("open") or {}).get("status") != "filled":
        raise ProbeGuardError(f"trial {args.trial!r} has no filled 'open' step")
    if trial_closed(tstate):
        raise ProbeGuardError(f"trial {args.trial!r} is already closed")
    remaining = int(tstate.get("remaining", tstate.get("quantity", 1)))
    lots = int(getattr(args, "quantity", None) or remaining)
    if not 1 <= lots <= remaining:
        raise ProbeGuardError(f"trial {args.trial!r} holds {remaining} lot(s); cannot close {lots}")
    use_quantity(lots)
    use_symbol(tstate.get("symbol", "SPY"))
    guard_window("close", tstate["kind"])
    guard_bot_stopped()
    guard_no_working(state)
    ev = Evidence("close", args.trial)
    params = load_params()
    ibkr = await connect_probe(int(tstate["data_type"]))
    errors = ErrorRecorder(ibkr.ib, ev)
    try:
        ev.emit("run_start", argv=sys.argv[1:], client_id=ibkr.client_id, account=ibkr.session_account,
                data_type=tstate["data_type"], params=params)
        positions = await fresh_positions(ibkr)
        await _prepare(ibkr, state, ev, owned_con_ids(state), check_account=False)
        ok, snap = _positions_match_open(tstate, positions)
        if not ok:
            raise ProbeGuardError(f"the broker's positions do not match the trial's open legs: {snap}")
        result = await _run_close_like(ibkr, state, tstate, args.trial, "close", ev, errors, params,
                                       rest_s=args.rest or params["rest"], tail=not args.no_truth_tail,
                                       emulate_delay_s=float(getattr(args, "emulate_delay", 0) or 0),
                                       require_nobid=bool(getattr(args, "require_nobid", False)),
                                       exclude_unbid=bool(getattr(args, "exclude_unbid", False)))
        after = await fresh_positions(ibkr)
        ev.emit("trial_end", step="close_positions", positions=positions_snapshot(
            after, {int(l["conId"]) for l in tstate["legs"]}))
        if result["outcome"] in ("filled", "cancel_race_filled"):
            tstate["remaining"] = remaining - lots
            save_state(state)
            ev.emit("quantity_check", step="close", lots_closed=lots, remaining=tstate["remaining"],
                    positions=positions_snapshot(after, {int(l["conId"]) for l in tstate["legs"]}),
                    bot_liveness_at_remaining=(await bot_liveness(ibkr, tstate, tstate["remaining"])
                                               if tstate["remaining"] > 0 else None),
                    bot_liveness_at_old_size=await bot_liveness(ibkr, tstate, remaining))
        return {"trial": args.trial, "remaining": tstate.get("remaining"), **result}
    finally:
        errors.detach()
        ev.emit("run_end")
        await ibkr.disconnect()


async def single_leg_ladder(ibkr, leg: dict, action: str, position: float, ev: Evidence,
                            errors: ErrorRecorder, trial: str) -> dict:
    """Close ONE leg with single-option limits that creep toward the market. Reduce-only."""
    assert_reduce_only(action, position)
    last = {"kind": "unfilled"}
    for k in range(LEGOUT_RUNGS):
        bid, ask = await poll_quote(ibkr, leg["_contract"])
        if action == "BUY":
            if not (_finite(ask) and ask > 0):
                ev.emit("rung_abort", reason="no_ask", leg=leg["conId"], k=k)
                return {"kind": "no_quote"}
            lim = min(ask + LEGOUT_STEP * k, ask + LEGOUT_MAX_CONCESSION)
        else:
            b = bid if _finite(bid) and bid > 0 else 0.0
            lim = max(0.01, b - LEGOUT_STEP * k)
        lim = round(lim, 2)
        ev.emit("rung_plan", step="legout", k=k, leg=leg["conId"], action=action, bid=bid, ask=ask, limit=lim)
        order = OrderBuilder.limit(action, abs(int(position)) or QUANTITY, lim)
        last = await place_and_rest(ibkr, leg["_contract"], order,
                                    make_order_ref(trial, f"legout-{leg['conId']}", k), LEGOUT_REST_S,
                                    ev, errors, k)
        ev.emit("rung_end", step="legout", k=k, leg=leg["conId"], outcome=last["kind"],
                avg_fill_price=last.get("avg_fill_price"), t_to_fill_s=last.get("t_to_fill_s"))
        if last["kind"] in _FILLED_ENDS:
            return last
        if last["kind"] in _BAD_ENDS:
            return last
    return last


async def leg_out(ibkr, state: dict, tstate: dict, trial: str, ev: Evidence, errors: ErrorRecorder) -> dict:
    """Close the structure leg by leg, SHORTS FIRST: buy each short back (riskier side first) and
    confirm it is flat before anything else; only then sell the longs. If a short does not come
    back, nothing is sold and the operator is told. Positions come from the broker, not the state."""
    if await _probe_orders_working(ibkr):
        raise ProbeGuardError("a probe order is still working -- run 'cancel' before a leg-out")
    legs = await qualify_legs(ibkr, trial_legs(tstate, "open"))
    positions = await fresh_positions(ibkr)
    if positions is None:
        raise ProbeGuardError("positions unknown -- refusing a leg-out")
    for l in legs:
        l["pos"] = _pos_of(positions, l["conId"])
    stock = await ibkr.qualify_contract(ContractBuilder.stock(SYMBOL))
    spot = await poll_spot(ibkr, stock) if stock else None

    def _risk(l):
        if spot is None:
            return 0.0 if l["right"] == "P" else -1.0
        return (l["strike"] - spot) if l["right"] == "P" else (spot - l["strike"])

    shorts = sorted([l for l in legs if l["action"] == "SELL" and l["pos"] != 0], key=_risk, reverse=True)
    longs = [l for l in legs if l["action"] == "BUY" and l["pos"] != 0]
    ev.emit("legout_plan", shorts=[l["conId"] for l in shorts], longs=[l["conId"] for l in longs], spot=spot)
    tstate["legout"] = {"status": "working", "started": now_et().isoformat()}
    save_state(state)
    results = {}
    for leg in shorts:
        res = await single_leg_ladder(ibkr, leg, "BUY", leg["pos"], ev, errors, trial)
        results[str(leg["conId"])] = res["kind"]
        after = _pos_of(await fresh_positions(ibkr), leg["conId"])
        if after != 0:
            tstate["legout"].update({"status": "short_still_open", "results": results})
            save_state(state)
            ev.emit("trial_end", step="legout", outcome="short_still_open", results=results)
            raise ProbeGuardError(
                f"SHORT STILL OPEN: {leg['strike']}{leg['right']} {leg['expiry']} (position {after}). "
                "Nothing was sold. Close it by hand in TWS (buy it back), then run 'status'.")
    for leg in longs:
        res = await single_leg_ladder(ibkr, leg, "SELL", leg["pos"], ev, errors, trial)
        results[str(leg["conId"])] = res["kind"]
    after = await fresh_positions(ibkr)
    left = {str(l["conId"]): _pos_of(after, l["conId"]) for l in legs if _pos_of(after, l["conId"]) != 0}
    status = "filled" if not left else "longs_left"
    tstate["legout"].update({"status": status, "results": results, "left": left, "ended": now_et().isoformat()})
    save_state(state)
    ev.emit("trial_end", step="legout", outcome=status, results=results, left=left)
    return {"status": status, "results": results, "left": left}


async def cmd_flatten(args) -> dict:
    """Close what the trial holds: the combo ladder first (like `close`, urgent rest), then, if it
    ends unfilled, leg-out. Safe to repeat; a no-op once flat."""
    state = load_state()
    tstate = state["trials"].get(args.trial)
    if not tstate or (tstate.get("open") or {}).get("status") != "filled":
        return {"status": "nothing_to_flatten"}
    if trial_closed(tstate) or ("remaining" not in tstate and any(
            (tstate.get(s) or {}).get("status") == "filled" for s in ("legout",))):
        return {"status": "already_flat"}
    use_quantity(max(1, int(tstate.get("remaining", tstate.get("quantity", 1)))))
    use_symbol(tstate.get("symbol", "SPY"))
    guard_window("flatten")
    guard_bot_stopped()
    ev = Evidence("flatten", args.trial)
    params = load_params()
    ibkr = await connect_probe(int(tstate["data_type"]))
    errors = ErrorRecorder(ibkr.ib, ev)
    try:
        ev.emit("run_start", argv=sys.argv[1:], client_id=ibkr.client_id, account=ibkr.session_account,
                data_type=tstate["data_type"], params=params)
        if await _probe_orders_working(ibkr):
            raise ProbeGuardError("a probe order is still working -- run 'cancel' first")
        positions = await fresh_positions(ibkr)
        if positions is None:
            raise ProbeGuardError("positions unknown -- refusing")
        held = {int(l["conId"]) for l in tstate["legs"] if _pos_of(positions, int(l["conId"])) != 0}
        if not held:
            tstate["flatten"] = {"status": "filled", "outcome": "already_flat_at_broker"}
            tstate["remaining"] = 0
            save_state(state)
            return {"status": "already_flat"}
        if not args.legout_only:
            result = await _run_close_like(ibkr, state, tstate, args.trial, "flatten", ev, errors, params,
                                           rest_s=args.rest or LEGOUT_REST_S, tail=False)
            if result["outcome"] in ("filled", "cancel_race_filled"):
                tstate["remaining"] = 0
                save_state(state)
                return {"status": "flattened_by_combo", **result}
            if result["outcome"] in ("cancel_unconfirmed", "partial"):
                raise ProbeGuardError(f"combo flatten ended {result['outcome']!r}: run 'status' and 'cancel'")
        out = await leg_out(ibkr, state, tstate, args.trial, ev, errors)
        if out.get("status") == "filled":
            tstate["remaining"] = 0
            save_state(state)
        return {"status": "flattened_by_legout", "legout": out}
    finally:
        errors.detach()
        ev.emit("run_end")
        await ibkr.disconnect()


async def cmd_cancel(_args) -> dict:
    """Cancel the probe's working orders (by tag) and confirm each. Never touches another order."""
    guard_window("cancel")
    ev = Evidence("cancel")
    ibkr = await connect_probe(3)
    errors = ErrorRecorder(ibkr.ib, ev)
    try:
        working = await _probe_orders_working(ibkr)
        results = {}
        for t in working:
            results[str(t.order.orderId)] = await cancel_and_confirm(ibkr, t, ev)
        positions = await fresh_positions(ibkr)
        spy = [[p.contract.strike, p.contract.right, str(p.contract.lastTradeDateOrContractMonth), p.position]
               for p in positions or [] if p.contract.symbol in PROBE_SYMBOLS
               and getattr(p.contract, "secType", "") == "OPT" and p.position]
        ev.emit("trial_end", step="cancel", results=results, spy_positions=spy)
        return {"cancelled": results, "spy_positions": spy}
    finally:
        errors.detach()
        ev.emit("run_end")
        await ibkr.disconnect()


async def cmd_quotes(args) -> dict:
    """Read-only: the trial's four legs right now (bid, ask, feed type, age) and whether the closing
    combo would have to sell a leg nobody bids for. Use it to watch a `nobid` trial's wing."""
    import datetime as _dt
    state = load_state()
    tstate = state["trials"].get(args.trial)
    if not tstate:
        raise ProbeGuardError(f"no trial {args.trial!r}")
    use_symbol(tstate.get("symbol", "SPY"))
    ibkr = await connect_probe(int(tstate["data_type"]))
    try:
        legs = await qualify_legs(ibkr, trial_legs(tstate, "close"))
        rows, quotes = [], {}
        for leg in legs:
            ibkr.ib.reqMktData(leg["_contract"], "", False, False)
        await _sleep(6.0)
        now = _dt.datetime.now(_dt.timezone.utc)
        for leg in legs:
            t = ibkr.ib.ticker(leg["_contract"])
            bid, ask = (getattr(t, "bid", None), getattr(t, "ask", None))
            quotes[leg["conId"]] = (bid, ask)
            tm = getattr(t, "time", None)
            rows.append({"strike": leg["strike"], "right": leg["right"], "close_action": leg["action"],
                         "bid": bid, "ask": ask, "marketDataType": getattr(t, "marketDataType", None),
                         "age_s": (None if tm is None else round((now - tm.astimezone(_dt.timezone.utc)).total_seconds()))})
        for leg in legs:
            try:
                ibkr.ib.cancelMktData(leg["_contract"])
            except Exception:  # noqa: BLE001
                pass
        nobid = sell_legs_without_bid(legs, quotes)
        natural = natural_cost(legs, quotes)
        return {"trial": args.trial, "symbol": SYMBOL, "legs": rows, "close_sell_legs_without_bid": len(nobid),
                "natural_cost_to_close": natural}
    finally:
        await ibkr.disconnect()


async def cmd_status(_args) -> dict:
    """Read-only: the state file, the broker's SPY option positions and probe orders, and the
    pre-restart checklist."""
    state = load_state()
    ev = Evidence("status")
    ibkr = await connect_probe(3)
    try:
        positions = await fresh_positions(ibkr)
        working = await _probe_orders_working(ibkr)
        spy = [{"conId": p.contract.conId, "strike": p.contract.strike, "right": p.contract.right,
                "expiry": str(p.contract.lastTradeDateOrContractMonth), "position": p.position}
               for p in positions or [] if p.contract.symbol in PROBE_SYMBOLS
               and getattr(p.contract, "secType", "") == "OPT" and p.position]
        halt = sorted(k for k in _read_halt_untracked_keys() if k.split(":")[0] in PROBE_SYMBOLS)
        pending = [n for n, t in state["trials"].items()
                   if any((t.get(s) or {}).get("status") in ("working", "placing", "partial", "short_still_open")
                          for s in ("open", "close", "flatten", "legout"))
                   or ((t.get("open") or {}).get("status") == "filled"
                       and not trial_closed(t)
                       and not any((t.get(s) or {}).get("status") in ("filled", "longs_left")
                                   for s in ("close", "flatten", "legout") if "remaining" not in t))]
        checklist = {"no_spy_option_positions": not spy, "no_probe_orders_working": not working,
                     "no_spy_keys_in_halt_untracked": not halt, "no_trial_open_or_working": not pending}
        result = {"trials": {n: {s: (t.get(s) or {}).get("status") for s in ("open", "close", "flatten", "legout")}
                             for n, t in state["trials"].items()},
                  "spy_positions": spy, "probe_option_positions": spy, "probe_orders_working": [t.order.orderId for t in working],
                  "halt_untracked_spy_keys": halt, "halt_untracked_probe_keys": halt, "unfinished_trials": pending,
                  "ready_to_restart_the_bot": all(checklist.values()), "checklist": checklist,
                  "tws_reminder": "confirm in TWS Activity that nothing else is open before restarting the bot"}
        ev.emit("status", **{k: v for k, v in result.items() if k != "tws_reminder"})
        return result
    finally:
        await ibkr.disconnect()


# --- Report (pure: reads the evidence, never connects) ---------------------------

def load_events(root: Path | None = None) -> list[dict]:
    root = PROBE_DIR if root is None else root
    events: list[dict] = []
    for f in sorted(root.glob("*/events.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("schema") != SCHEMA:
                raise ProbeGuardError(f"{f}: evidence schema {e.get('schema')!r} != {SCHEMA}")
            events.append(e)
    return events


def _ts(e: dict):
    from datetime import datetime
    return datetime.fromisoformat(e["ts"])


def _feed_of(plan: dict) -> str:
    """'emulated' when the rung was priced from lagged quotes; 'live' when every leg's quote was
    real-time (marketDataType 1); otherwise 'delayed'. Decides how `truly marketable` is judged."""
    if plan.get("lag_s"):
        return "emulated"
    mdt = [v for v in (plan.get("mdt") or {}).values() if v is not None]
    return "live" if mdt and all(v == 1 for v in mdt) else "delayed"


def analyze_run(events: list[dict], legs: list[dict]) -> dict:
    """One close/flatten run of one trial -> the facts the verdict needs.

    `events`: that run's events, in order. `legs`: the trial's legs [{conId, close_action, ...}]."""
    close_legs = [{"conId": int(l["conId"]), "action": l["close_action"], "ratio": 1} for l in legs]
    plans = {e["k"]: e for e in events if e["event"] == "rung_plan" and e.get("limit") is not None}
    placed = {e["k"]: e for e in events if e["event"] == "order_placed"}
    ends = {e["k"]: e for e in events if e["event"] == "rung_end" and e.get("step") is None}
    samples = [e for e in events if e["event"] == "quote_sample"]
    start = _ts(placed[min(placed)]) if placed else None
    rungs, any_truth, flags = [], [], []
    filled_rung, fill_price = None, None
    for k in sorted(plans):
        p, end = plans[k], ends.get(k, {})
        outcome = end.get("outcome")
        true_nat = marketable = None
        feed = _feed_of(p)
        if feed in ("live", "emulated") and p.get("natural_live") is not None:
            # the limit met the live market: compare it with the live natural at that moment
            true_nat = p["natural_live"]
            marketable = bool(p["limit"] >= true_nat)
        elif k in placed and samples:
            target = _ts(placed[k]).timestamp() + DELAY_S
            best = min(samples, key=lambda s: abs(_ts(s).timestamp() - target))
            if abs(_ts(best).timestamp() - target) <= TRUTH_MATCH_S:
                q = {int(c): (v.get("bid"), v.get("ask")) for c, v in best["legs"].items()}
                true_nat = natural_cost(close_legs, q)
                if true_nat is not None:
                    marketable = bool(p["limit"] >= true_nat)
        any_truth.append(marketable)
        rungs.append({"k": k, "limit": p["limit"], "natural": p.get("natural"), "mid": p.get("mid"),
                      "natural_live": p.get("natural_live"), "feed": feed,
                      "outcome": outcome, "true_natural": true_nat, "truly_marketable": marketable,
                      "cap_binds": bool(p.get("cap_binds")), "floor_bound": bool(p.get("floor_bound"))})
        if outcome in ("filled", "cancel_race_filled") and filled_rung is None:
            filled_rung, fill_price = k, end.get("avg_fill_price")
        if outcome in _BAD_ENDS:
            flags.append(outcome)
    aborts = [e for e in events if e["event"] == "rung_abort"]
    flags += [f"aborted_{e.get('reason')}" for e in aborts]
    width = next((p.get("width") for p in plans.values() if p.get("width")), None)
    last = rungs[-1]["limit"] if rungs else None
    n0 = rungs[0]["natural"] if rungs else None
    nf = next((r["natural"] for r in rungs if r["k"] == filled_rung), None)
    mid0 = rungs[0]["mid"] if rungs else None
    nlive = next((r["natural_live"] for r in rungs if r["k"] == filled_rung), None)
    t_fill = None
    if filled_rung is not None and filled_rung in ends and start is not None:
        t_fill = round((_ts(ends[filled_rung]) - start).total_seconds(), 1)

    def _bag_quoted(bag) -> bool:
        return bool(bag) and any(_finite(bag.get(x)) and bag.get(x) != 0 for x in ("bid", "ask"))

    return {
        "bag_ticked": any(e.get("ticked") for e in events if e["event"] == "bag_probe")
                      or any(_bag_quoted(s.get("bag")) for s in samples),
        "priceable": bool(rungs) and all(r["natural"] is not None for r in rungs),
        "rungs": rungs, "filled_rung": filled_rung, "fill_price": fill_price,
        "t_from_start_s": t_fill, "flags": sorted(set(flags)),
        "cap_binds": any(r["cap_binds"] for r in rungs),
        "reached_width": bool(width and last is not None and last >= width - 0.005),
        "any_truly_marketable": (True if any(m is True for m in any_truth) else
                                 False if any(m is False for m in any_truth) else None),
        "concession_vs_first_natural": (round(fill_price - n0, 4) if fill_price is not None and n0 is not None else None),
        "concession_vs_natural_at_fill": (round(fill_price - nf, 4) if fill_price is not None and nf is not None else None),
        "concession_vs_first_mid": (round(fill_price - mid0, 4) if fill_price is not None and mid0 is not None else None),
        "concession_vs_live_natural": (round(fill_price - nlive, 4) if fill_price is not None and nlive is not None else None),
        "feed": rungs[0]["feed"] if rungs else None,
        "unbid_legs_in_combo": sorted({c for p in plans.values() for c in (p.get("no_bid_sell_legs") or [])}
                                      - {c for e in events if e["event"] == "excluded_legs" for c in e.get("legs", [])}),
        "excluded_legs": sorted({c for e in events if e["event"] == "excluded_legs" for c in e.get("legs", [])}),
        "band_rejects": sum(1 for e in events if e["event"] == "band_reject"),
        "emulated_delay_s": next((p.get("lag_s") for p in plans.values() if p.get("lag_s")), 0.0),
    }


def classify_run(r: dict) -> str:
    """A: filled by rung 3 (the shipped escalation is enough). B: filled at rung 4-6. C: never
    filled although the legs priced, the width was reached and a limit was truly marketable (whether
    the BAG ticked is recorded and only changes the verdict's name). C': never filled and no limit
    was truly marketable (stale quotes). D: anything that cannot be read."""
    if r["filled_rung"] is not None:
        return "A" if r["filled_rung"] <= 3 else "B"
    if r["flags"] or r["cap_binds"] or not r["priceable"] or not r["reached_width"]:
        return "D"
    if r["any_truly_marketable"] is None:
        return "D"
    return "C" if r["any_truly_marketable"] else "C'"


VERDICT_NAMES = {"A": "BOT_LADDER_SUFFICIENT", "B": "NEEDS_DEEPER_LADDER", "C": "NO_MATCH_WITHOUT_COMBO_QUOTE",
                 "C'": "LIMITS_NEVER_MARKETABLE", "D": "INCONCLUSIVE"}


def verdict(classes: list[str], c_bag_ticked: bool = False) -> dict:
    """The decision map. C needs at least two independent runs. If a C run's BAG ticked, the combo
    did not fill although it was quoted AND marketable: NO_MATCH_AT_MARKETABLE_PRICE."""
    counts = {c: classes.count(c) for c in ("A", "B", "C", "C'", "D")}
    if not classes:
        name = "D"
    elif counts["C"] >= 2:
        name = "C"
    elif counts["B"] or counts["A"]:
        name = "B" if counts["B"] else "A"
    elif counts["C'"] and not counts["C"]:      # a lone C beside a C' is conflicting evidence
        name = "C'"
    else:
        name = "D"
    label = VERDICT_NAMES[name]
    if name == "C" and c_bag_ticked:
        label = "NO_MATCH_AT_MARKETABLE_PRICE"
    return {"verdict": label, "code": name, "counts": counts,
            "mixed": sum(1 for c in counts.values() if c) > 1}


def unbid_leg_finding(runs: list[dict]) -> dict:
    """Did a closing combo that included a leg nobody bids for fail to fill, and did leaving that leg out
    fix it? `blocked`: runs whose combo kept an unbid sold leg; `excluded`: runs that left it out."""
    blocked = [r for r in runs if r.get("unbid_legs_in_combo")]
    excluded = [r for r in runs if r.get("excluded_legs")]
    f = lambda rs: {"runs": len(rs), "filled": sum(1 for r in rs if r["filled_rung"] is not None)}  # noqa: E731
    b, x = f(blocked), f(excluded)
    if b["runs"] and b["filled"] == 0 and x["filled"] >= 1:
        verdict_ = "UNBID_LEG_BLOCKS_THE_COMBO"
    elif b["runs"] and b["filled"] == 0:
        verdict_ = "UNBID_LEG_BLOCKED_THE_COMBO_NOT_YET_CONFIRMED_BY_EXCLUSION"
    elif b["runs"]:
        verdict_ = "UNBID_LEG_DID_NOT_PREVENT_FILLS"
    else:
        verdict_ = None
    return {"verdict": verdict_, "combo_kept_unbid_leg": b, "combo_left_it_out": x}


def leg_out_verdict(events: list[dict]) -> str | None:
    outs = [e.get("outcome") for e in events if e["event"] == "trial_end" and e.get("step") == "legout"]
    if not outs:
        return None
    # `longs_left`: every short was bought back and every sellable long sold; only a long nobody bids
    # for remained (worthless, left to expire). That is a successful leg-out.
    return "LEG_OUT_VIABLE" if any(o in ("filled", "longs_left") for o in outs) else "LEG_OUT_FAILED"


async def cmd_report(_args) -> dict:
    events = load_events()
    legs_by_trial: dict[str, list] = {}
    for e in events:
        if e["event"] == "legs_selected" and e.get("trial") and e.get("legs"):
            legs_by_trial[e["trial"]] = e["legs"]
    runs: dict[tuple, list] = {}
    for e in events:
        if e.get("cmd") in ("close", "flatten") and e.get("trial"):
            runs.setdefault((e["trial"], e["run_id"], e["cmd"]), []).append(e)
    per_run = []
    for (trial, run_id, cmd), evs in sorted(runs.items()):
        legs = legs_by_trial.get(trial)
        if not legs or not any(x["event"] == "rung_plan" and "natural" in x for x in evs):
            continue                                    # no combo ladder in this run (e.g. a leg-out)
        r = analyze_run(evs, legs)
        r.update({"trial": trial, "run_id": run_id, "cmd": cmd, "class": classify_run(r)})
        per_run.append(r)
    closes = [r for r in per_run if r["cmd"] == "close"]
    result = verdict([r["class"] for r in closes],
                     c_bag_ticked=any(r["bag_ticked"] for r in closes if r["class"] == "C"))
    result["leg_out"] = leg_out_verdict(events)
    result["unbid_leg"] = unbid_leg_finding(closes)
    result["headline"] = (
        "A closing combo that contains a leg nobody bids for does not fill, even far above the market; "
        "leaving that leg out lets it fill." if result["unbid_leg"]["verdict"] == "UNBID_LEG_BLOCKS_THE_COMBO"
        else result["verdict"])
    result["runs"] = per_run
    return result


# --- Entry point -----------------------------------------------------------------

_HANDLERS = {"preflight": cmd_preflight, "open": cmd_open, "close": cmd_close, "flatten": cmd_flatten,
             "cancel": cmd_cancel, "status": cmd_status, "report": cmd_report, "quotes": cmd_quotes}
_PLACES_ORDERS = ("open", "close", "flatten")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")
    for name in _HANDLERS:
        sp = sub.add_parser(name)
        if name in ("preflight", "open", "close"):
            sp.add_argument("--quantity", type=int, default=None, choices=range(1, MAX_QUANTITY + 1),
                            metavar=f"1..{MAX_QUANTITY}",
                            help="lots: open/preflight default 1; close default = every lot still held "
                                 "(a smaller number closes that many and leaves the rest)")
        if name in ("preflight", "open"):
            sp.add_argument("--data-type", type=int, default=3, choices=(1, 2, 3, 4),
                            help="market-data type for this session (default 3 = the bot's)")
            sp.add_argument("--spot-hint", type=float, default=None, metavar="PRICE",
                            help="a rough price of the underlying: where the options-parity search starts when the "
                                 "stock has no price and no earlier run recorded one")
            sp.add_argument("--symbol", default="SPY", choices=PROBE_SYMBOLS,
                            help="the underlying (default SPY); close/flatten/quotes read it from the trial")
        if name in ("open", "close", "flatten", "quotes"):
            sp.add_argument("--trial", required=True, help="trial name, e.g. F1 or P1")
        if name == "open":
            sp.add_argument("--kind", required=True, choices=("far", "pin", "nobid"))
            sp.add_argument("--short-call", type=float, default=None, metavar="STRIKE",
                            help="nobid only: the short call strike (the long call is 3 further). Pick the "
                                 "highest strike that still has a bid; the next 3 up should have none")
        if name in ("close", "flatten"):
            sp.add_argument("--rest", type=float, default=None, help="seconds each rung rests")
        if name == "close":
            sp.add_argument("--no-truth-tail", action="store_true", help="skip the 15-minute truth tail")
            sp.add_argument("--require-nobid", action="store_true",
                            help="refuse to start unless a leg the close must SELL has no bid")
            sp.add_argument("--exclude-unbid", action="store_true",
                            help="leave a worthless long leg that nobody bids for OUT of the closing combo "
                                 "(it expires); refused if such a leg still has value")
            sp.add_argument("--emulate-delay", type=float, default=0.0, metavar="SECONDS",
                            help="price each rung from the quotes recorded SECONDS ago (900 = a delayed "
                                 "feed) while orders meet the live market; waits for that much history first")
        if name == "flatten":
            sp.add_argument("--legout-only", action="store_true", help="skip the combo ladder")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["preflight"])
    handler = _HANDLERS[args.command]
    try:
        if getattr(args, "symbol", None):
            use_symbol(args.symbol)
        if getattr(args, "spot_hint", None):
            globals()["SPOT_HINT"] = float(args.spot_hint)
        result = asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr)
        _cancel_after_failure(args)
        return 130
    except Exception as e:  # noqa: BLE001 - top-level safety net
        kind = "GUARD REFUSED" if isinstance(e, ProbeGuardError) else "UNEXPECTED ERROR"
        log.error("probe_command_failed", command=args.command, kind=kind, error=str(e))
        print(f"{kind}: {e}", file=sys.stderr)
        _cancel_after_failure(args)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cancel_after_failure(args) -> None:
    """After a failure in a command that may have orders working: cancel the probe's orders
    (never flatten: a leg-out on stale data at an unplanned moment is riskier than leaving a
    one-lot defined-risk condor for the operator to close deliberately)."""
    if args.command not in _PLACES_ORDERS:
        return
    print("cancelling the probe's working orders (positions are NOT touched)...", file=sys.stderr)
    try:
        print(json.dumps(asyncio.run(cmd_cancel(args)), indent=2, default=str), file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"cancel FAILED: {e}\nCancel any W101PROBE order in TWS by hand, then run 'status'.", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
