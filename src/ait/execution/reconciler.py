"""Position reconciliation — sync local state with IBKR on restart.

When the bot restarts (crash, manual stop, etc.), this module:
1. Loads local state from SQLite
2. Fetches live positions from IBKR
3. Reconciles differences and flags discrepancies
4. Updates local state to match reality
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime

from ait.broker.ibkr_client import IBKRClient
from ait.bot.state import StateManager, TradeStatus
from ait.strategies.base import CREDIT_STRATEGIES
from ait.utils.logging import get_logger
from ait.utils.time import now_et, parse_trade_ts

log = get_logger("execution.reconciler")

# ---------------------------------------------------------------------------
# W3 (string-contracts-4): the exit_reason_detailed sentinels this module
# books with realized_pnl = 0.0 for outcomes it could NOT determine.
#
# These rows are status='closed' but they are NOT trading results: a $0
# placeholder for an UNKNOWN outcome is not a scratch trade. Before W3 all
# three PASSED the not-a-real-close filter every consumer applied (which knew
# only about never_filled / pending / migrated), so the first unbookable exit
# would have counted as a real LOSING close in PF, win-rate, drawdown,
# status.py's P&L and the go-live scorecard that authorizes real money.
# ait.reporting.go_live now excludes them everywhere.
#
# The STRING VALUES are a written contract other consumers match on and are
# deliberately unchanged — this module still WRITES exactly what it always
# wrote; only who COUNTS the rows changed. Never edit a value here without
# editing ait.reporting.go_live.NOT_REAL_CLOSE_PATTERNS with it:
# tests/test_w3_scorecard_truth.py asserts every constant below is excluded
# by is_real_close().
UNKNOWN_EXIT_REASON = "reconciler_unknown_exit"
UNKNOWN_EXPIRED_EXIT_REASON = "reconciler_unknown_exit_expired_needs_review"
NON_FINITE_EXIT_REASON = "reconciler_unknown_exit_needs_review"
NEVER_FILLED_EXIT_REASON = "stale_pending_never_filled"

#: Every $0 sentinel this module can book. Not a real close, by construction.
NOT_REAL_CLOSE_EXIT_REASONS = (
    UNKNOWN_EXIT_REASON,
    UNKNOWN_EXPIRED_EXIT_REASON,
    NON_FINITE_EXIT_REASON,
    NEVER_FILLED_EXIT_REASON,
)


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    matched: int  # Positions that match between local and IBKR
    new_from_ibkr: int  # Positions in IBKR not in local state
    stale_local: int  # Local positions not found in IBKR
    discrepancies: list[str]  # Human-readable descriptions
    promoted: int = 0  # PENDING/CLOSING trades found live in IBKR -> FILLED
    # R12 F4.1: CLOSING trades KEPT closing because a working close order
    # still exists at the broker: trade_id -> broker orderId. The
    # orchestrator must call executor.adopt_exit_order(order_id, trade_id,
    # reason) for each so the exit tracker survives the restart.
    closing_exit_orders: dict[str, int] = field(default_factory=dict)


class PositionReconciler:
    """Reconciles local trade state with IBKR live positions."""

    def __init__(self, ibkr_client: IBKRClient, state: StateManager) -> None:
        self._ibkr = ibkr_client
        self._state = state
        # W22: optional — set via set_telegram(). None means the declare/
        # reject prompt is simply skipped (logged), same as an unconfigured
        # TelegramNotifier no-op's; INST-3's freeze itself never depends on
        # this being set.
        self._telegram = None

    def set_telegram(self, telegram) -> None:
        """W22: wire in the TelegramNotifier so untracked-position prompts
        can be sent. Mirrors TradingOrchestrator.set_notification_callback's
        shape — a setter rather than a constructor arg, so existing
        `PositionReconciler(ibkr, state)` call sites (tests included) don't
        need to change."""
        self._telegram = telegram

    @staticmethod
    def _normalize_expiry(expiry: str | None) -> str:
        """Normalize expiry date to YYYY-MM-DD format.

        IBKR returns lastTradeDateOrContractMonth in YYYYMMDD format,
        while local trades may store YYYY-MM-DD. This normalizes both.
        """
        if not expiry:
            return ""
        # Strip any whitespace
        expiry = expiry.strip()
        # If already YYYY-MM-DD, return as-is
        if re.match(r"^\d{4}-\d{2}-\d{2}$", expiry):
            return expiry
        # Convert YYYYMMDD to YYYY-MM-DD
        if re.match(r"^\d{8}$", expiry):
            return f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:8]}"
        return expiry

    @staticmethod
    def _make_position_key(symbol: str, strike: float, right: str, expiry: str) -> str:
        """Build a normalized position key: symbol:strike:right:expiry.

        Drops secType/contract_type since it's redundant with right (C/P).
        """
        normalized_expiry = PositionReconciler._normalize_expiry(expiry)
        return f"{symbol}:{strike}:{right}:{normalized_expiry}"

    # A PENDING order older than this is definitively dead: the fill-timeout
    # is minutes (limit_order_timeout_bars × 5 min, ~15 min), so 30 min with
    # no confirmed fill means the order never filled (or filled-and-expired).
    # These orphans otherwise persist forever (the in-memory fill tracker is
    # lost on restart) and, with the pending-aware dedup guard, actively BLOCK
    # all new trades on the symbol + correlated symbols.
    STALE_PENDING_MINUTES = 30

    def _ensure_open_position_row(self, trade) -> None:
        """R16: give a rescued trade the open_positions row its monitors need.

        Every promotion path here flips trades.status via CAS but nothing ever
        created the open_positions row that only the executor's fill paths
        write (executor.py:1550, 1594). Without that row
        update_high_water_mark / update_position_mark are UPDATEs that match
        zero rows and no-op SILENTLY, and get_high_water_mark returns 0.0 — so
        a promoted trade is monitored for stops and take-profit but its
        trailing profit-lock, peak-P&L journaling and partial-exit ledger are
        dead for its entire life. insert_open_position is INSERT OR IGNORE,
        so this is idempotent and safe to call on every promotion.
        """
        try:
            self._state.insert_open_position(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                contract_type=getattr(trade, "contract_type", "") or "",
                quantity=abs(int(getattr(trade, "quantity", 0) or 0)),
                entry_price=abs(float(getattr(trade, "entry_price", 0.0) or 0.0)),
                legs=getattr(trade, "legs", "[]") or "[]",
            )
        except Exception as e:  # noqa: BLE001 — a promotion must not be undone
            log.warning("promoted_open_position_insert_failed",
                        trade_id=getattr(trade, "trade_id", "?"), error=str(e))

    def _adopt_or_flag_quantity(self, trade, broker_qty_map: dict[str, int],
                                frozen_keys: list[str]):
        """W50 (trade-life-promotion-ignores-broker-quantity): called right
        before a promotion writes the open_positions row.

        Promotion used to adopt the LOCAL quantity unconditionally — a fill
        or partial-exit this process missed while it was down left
        trade.quantity wrong, and nothing in this file ever called
        update_trade_quantity (grep confirmed zero call sites here before
        this wave). The later exit then reverses trade.quantity blind
        (orchestrator.py's combo_limit(quantity=trade.quantity) calls),
        closing what exists and opening the difference inverted — the R12
        F4.3 reversal incident class, resurrected on the restart path.

        Policy (operator-confirmed): every leg at the SAME broker quantity
        -> adopt it (a clean fill/partial-exit while down; safe to
        normalize). Legs at INCONSISTENT quantities -> do not guess which
        number is real; freeze new entries via the same frozen_keys /
        HALT_UNTRACKED path the untracked-position branches below already
        use, and leave trade.quantity as-is. That is not a silent gap:
        position_liveness's own quantity-awareness then reads this trade as
        "partial" at the next exit attempt, and the exit path already
        refuses a "partial" reverse combo (orchestrator.py, `if liveness ==
        "partial"`) — this only adds an IMMEDIATE alert instead of waiting
        for a stop to fire before anyone finds out.

        `broker_qty_map` is a plain leg-key -> abs(quantity) mapping (what
        _live_ibkr_leg_keys returns directly, or reconcile()'s own
        ibkr_map reduced to just the quantity) — a shape both callers can
        produce without each inventing their own.

        Returns the trade to use for the open_positions row: the same
        object unless the quantity was adopted, in which case an updated
        copy — the in-memory dataclass does not observe the DB write below.
        """
        leg_keys = self._trade_leg_keys(trade)
        if not leg_keys or leg_keys == {f"{trade.symbol}:STK"}:
            return trade  # unkeyable shape — nothing to compare against
        broker_qtys = {
            k: abs(int(broker_qty_map[k])) if k in broker_qty_map else 0
            for k in leg_keys
        }
        distinct = set(broker_qtys.values())
        local_qty = abs(int(trade.quantity))
        if len(distinct) == 1:
            adopted = distinct.pop()
            # adopted == 0 cannot actually happen here: this method only
            # runs from inside the ibkr_map key-match loop, so at least one
            # of `trade`'s own legs (the one that matched) is confirmed
            # nonzero — kept as a guard anyway, not a promise.
            if adopted and adopted != local_qty:
                self._state.update_trade_quantity(trade.trade_id, adopted)
                log.warning("reconcile_promotion_adopted_broker_quantity",
                            trade_id=trade.trade_id, symbol=trade.symbol,
                            local_quantity=local_qty, adopted_quantity=adopted)
                import dataclasses as _dc
                return _dc.replace(trade, quantity=adopted)
            return trade
        # Non-uniform legs: a genuinely broken structure, not a clean
        # partial. Freeze new entries and alert a human; leave
        # trade.quantity untouched.
        frozen_keys.append(
            f"{trade.symbol} {trade.trade_id} quantity mismatch at "
            f"promotion: local={local_qty}, broker legs={broker_qtys}")
        log.critical("reconcile_promotion_quantity_inconsistent",
                     trade_id=trade.trade_id, symbol=trade.symbol,
                     local_quantity=local_qty, broker_leg_quantities=broker_qtys)
        return trade

    async def _underlying_last(self, symbol: str) -> float | None:
        """Best-effort current/last price of the underlying via IBKR."""
        try:
            if not self._ibkr.connected:
                return None
            from ib_insync import Stock
            contract = Stock(symbol, "SMART", "USD")
            q = await self._ibkr.ib.qualifyContractsAsync(contract)
            if not q:
                return None
            ticker = self._ibkr.ib.reqMktData(q[0], "", False, False)
            try:
                import asyncio as _aio
                await _aio.sleep(2.0)
                for cand in (ticker.last, ticker.close,
                             (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None):
                    if cand and cand == cand and cand > 0:
                        return float(cand)
            finally:
                try:
                    self._ibkr.ib.cancelMktData(q[0])
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            log.warning("underlying_last_failed", symbol=symbol, error=str(e))
        return None

    @staticmethod
    def _structure_intrinsic(trade, settle_px: float) -> float | None:
        """Per-unit intrinsic value of the trade's option structure at expiry.

        Convention matches executor P&L: the returned number is the COST TO
        SETTLE for credit structures / VALUE RECEIVED for debit structures —
        i.e. (short intrinsics - long intrinsics) from the counterparty view
        equals (long intrinsics - short intrinsics) from ours; we return the
        absolute structure value with legs signed by action so both P&L
        formulas in the caller work with the same number.
        """
        try:
            legs = json.loads(trade.legs) if trade.legs else []
        except (ValueError, TypeError):
            legs = []
        if not legs and getattr(trade, "strike", None):
            right = "C" if "call" in trade.strategy else "P"
            # R15 #1: the synthesized action must follow the strategy's cash
            # flow. Hardcoded "SELL" sign-flipped every single-leg DEBIT trade
            # (long_call/long_put store legs="[]" + strike): a long put bought
            # at 3.00 expiring ITM worth 5.00 booked as an amplified LOSS.
            from ait.strategies.base import CREDIT_STRATEGIES as _CS_IL
            action = "SELL" if trade.strategy in _CS_IL else "BUY"
            legs = [{"strike": trade.strike, "right": right, "action": action}]
        if not legs:
            return None
        value = 0.0
        for leg in legs:
            try:
                k = float(leg["strike"])
                right = str(leg["right"]).upper()
                action = str(leg.get("action", "SELL")).upper()
            except (KeyError, TypeError, ValueError):
                return None
            intrinsic = max(0.0, settle_px - k) if right == "C" else max(0.0, k - settle_px)
            # Long legs add value to us; short legs are what we owe.
            value += intrinsic if action == "BUY" else -intrinsic
        # For CREDIT structures the caller computes (credit - cost_to_close):
        # cost_to_close = shorts owed - longs held = -value. For DEBIT:
        # value received = value. Return the caller-appropriate magnitude.
        from ait.strategies.base import CREDIT_STRATEGIES as _CS
        return -value if trade.strategy in _CS else value

    def _trade_leg_keys(self, trade) -> set[str]:
        """Normalized position keys for every leg of a local trade."""
        keys: set[str] = set()
        try:
            legs = json.loads(trade.legs) if trade.legs else []
        except (ValueError, TypeError):
            legs = []
        for leg in legs:
            try:
                keys.add(self._make_position_key(
                    symbol=trade.symbol,
                    strike=float(leg["strike"]),
                    right=str(leg["right"]),
                    expiry=str(leg.get("expiry", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        if not keys and getattr(trade, "strike", None):
            right = "C" if "call" in trade.strategy else "P"
            keys.add(self._make_position_key(
                symbol=trade.symbol, strike=trade.strike,
                right=right, expiry=trade.expiry or "",
            ))
        if not keys:
            keys.add(f"{trade.symbol}:STK")
        return keys

    async def position_liveness(self, trade) -> str:
        """R14: does this trade's structure still exist at the broker, and is
        it WHOLE? Returns one of:

          "live"    — every leg is present at >= trade.quantity; a reverse
                      combo closes exactly it.
          "partial" — SOME legs present, some gone (manual half-flatten,
                      assignment) OR every leg present but SHORT of
                      trade.quantity (a fill/partial-exit this process missed
                      while it was down — W50). Reversing all legs at
                      trade.quantity would SELL the already-flat/short ones
                      and OPEN new inverted positions. Needs a human.
          "gone"    — authoritatively absent. Reversing would REBUILD the
                      position inverted (the 07-13 end-state, reachable by an
                      operator with no bug at all).
          "unknown" — we cannot tell. NEVER block an exit on this: a wedged
                      data feed must not silently disable stops.

        The exit path must ask BEFORE placing a reverse combo.

        Why this is async: a cached-empty position list is ambiguous between
        "flat" and "stream wedged" (ib_insync's startup reqPositions timeout
        only logs; `connected` stays True). Believing an empty cache would
        refuse every exit — a stop-loss disabled exactly when the broker link
        is flaky. So before ever returning "gone" we re-request positions from
        the broker authoritatively; only a FRESH answer can retire a position.

        W50 (trade-life-promotion-ignores-broker-quantity): this used to be a
        pure KEY-SET comparison — every leg present, at ANY quantity, read as
        "live", so a trade whose broker legs were all present but at HALF the
        locally recorded quantity (a partial fill/exit the process missed
        while down, then reconciler-promoted without adopting the broker
        figure) passed here identically to a fully intact position. The
        caller's reverse combo prices at `trade.quantity` regardless
        (orchestrator.py's combo_limit calls), so it reversed more than
        actually existed. Quantity is now part of the "whole" test.
        """
        keys = self._trade_leg_keys(trade)
        if not keys or keys == {f"{trade.symbol}:STK"}:
            return "unknown"  # shape we can't key — don't block the exit on it
        try:
            needed = abs(int(trade.quantity))
        except (TypeError, ValueError):
            needed = 0  # unreadable quantity: presence-only, same as pre-W50

        live = self._live_ibkr_leg_keys()
        if live is None:
            return "unknown"

        present = keys & live.keys()
        if present == keys:
            if needed and any(live[k] < needed for k in keys):
                return "partial"
            return "live"
        if present:
            return "partial"

        # The cached view says GONE — the one answer we refuse to take on
        # trust, because it is also what a wedged position stream looks like.
        # Re-request from the broker before acting on it.
        fresh = await self._ibkr.get_positions_fresh()
        if fresh is None:
            log.warning("liveness_unconfirmed_broker_silent",
                        trade_id=trade.trade_id, symbol=trade.symbol,
                        note="cache says gone, broker won't confirm — treating "
                             "as unknown so the exit still goes out")
            return "unknown"

        fresh_map: dict[str, int] = {}
        for pos in fresh:
            if not getattr(pos, "position", 0):
                continue  # R15b: zeroed row = closed, not live
            try:
                if pos.contract.secType == "OPT":
                    fresh_map[self._make_position_key(
                        symbol=pos.contract.symbol,
                        strike=pos.contract.strike,
                        right=pos.contract.right,
                        expiry=pos.contract.lastTradeDateOrContractMonth,
                    )] = abs(int(pos.position))
            except (AttributeError, TypeError, ValueError):
                continue

        present = keys & fresh_map.keys()
        if present == keys:
            if needed and any(fresh_map[k] < needed for k in keys):
                return "partial"
            return "live"
        if present:
            return "partial"
        return "gone"

    async def _book_stale_trade(self, trade, ibkr_portfolio) -> None:
        """Close a trade whose position is gone at the broker, attributing the
        best P&L we can defend. Shared by reconcile()'s stale-local loop and
        the exit path's confirmed-vanished booking, so both book identically.
        """
        # Try to get realized P&L from IBKR portfolio items
        exit_price = 0.0
        realized_pnl = 0.0
        exit_reason = "reconciler_ibkr_realized"
        found_ibkr_pnl = False
        # LEG-AWARE attribution (deep-audit MP-F4): matching by symbol
        # alone grabbed the FIRST leg's realizedPNL as the whole
        # structure's P&L (~1/4 of an iron condor), and could book one
        # trade's P&L onto another trade on the same underlying. Sum the
        # realizedPNL of exactly THIS trade's legs.
        trade_keys = self._trade_leg_keys(trade)
        leg_pnls = []
        leg_pnl_unusable = False
        for item in ibkr_portfolio:
            if item.contract.secType != "OPT" or item.position != 0:
                continue
            key = self._make_position_key(
                symbol=item.contract.symbol,
                strike=item.contract.strike,
                right=item.contract.right,
                expiry=item.contract.lastTradeDateOrContractMonth,
            )
            if key not in trade_keys:
                continue
            # R19: NaN is TRUTHY, so `if item.realizedPNL:` admitted IBKR's
            # NaN sentinel; one NaN leg makes sum(leg_pnls) NaN and
            # close_trade() then writes realized_pnl=NaN into trades (and the
            # DuckDB dual-write) labelled as broker-derived — poisoning every
            # SUM / profit-factor / win-rate over the column, permanently and
            # silently. portfolio.py:227 already isnan-guards marketPrice on
            # these same PortfolioItem objects; realizedPNL had no guard.
            _leg_pnl = item.realizedPNL
            try:
                _usable = _leg_pnl is not None and math.isfinite(float(_leg_pnl))
            except (TypeError, ValueError):
                _usable = False
            if not _usable:
                if _leg_pnl is not None:
                    # One unknown leg makes the STRUCTURE's P&L unknown —
                    # summing the rest would label a knowingly-incomplete
                    # number "broker-derived". Refuse the whole attribution.
                    leg_pnl_unusable = True
                    log.warning("reconcile_leg_pnl_unusable", trade_id=trade.trade_id,
                                leg=key, value=str(_leg_pnl),
                                note="non-finite realizedPNL from IBKR")
                continue
            if _leg_pnl:
                leg_pnls.append(float(_leg_pnl))
        if leg_pnls and not leg_pnl_unusable:
            realized_pnl = float(sum(leg_pnls))
            found_ibkr_pnl = True
        elif leg_pnl_unusable:
            # Fall through to the intrinsic / needs-review path below.
            log.critical("reconcile_ibkr_pnl_non_finite", trade_id=trade.trade_id,
                         symbol=trade.symbol, usable_legs=str(leg_pnls),
                         note="a leg's broker P&L is not finite — NOT booked as "
                              "broker-derived; using the review path instead")

        if not found_ibkr_pnl and trade.entry_price > 0:
            # No IBKR data. Only assume "expired worthless" when the
            # position has actually reached expiry; a position that
            # vanished mid-life was closed at an unknown price and
            # recording a full win/loss would be fiction.
            expired = False
            if trade.expiry:
                try:
                    expired = datetime.fromisoformat(trade.expiry).date() <= datetime.now().date()
                except ValueError:
                    pass

            multiplier = 100 if trade.contract_type != "stock" else 1
            if expired:
                # ITM-AWARE expiry booking (audit R2, goal-align #1): the
                # old branch booked ANY expired credit as a full-premium
                # win — an option that expired IN the money (e.g. short
                # put after a selloff over a down weekend) was recorded as
                # a max win instead of a loss. Value the structure at
                # expiry from intrinsic using the underlying's price; if
                # the underlying price can't be fetched, refuse to invent
                # a number and fall through to the review path.
                settle_px = await self._underlying_last(trade.symbol)
                intrinsic = self._structure_intrinsic(trade, settle_px) if settle_px else None
                if intrinsic is not None:
                    if trade.strategy in CREDIT_STRATEGIES:
                        # cost to settle = what the shorts owe minus longs
                        realized_pnl = (trade.entry_price - intrinsic) * multiplier * trade.quantity
                    else:
                        realized_pnl = (intrinsic - trade.entry_price) * multiplier * trade.quantity
                    exit_price = intrinsic
                    exit_reason = "reconciler_expired_intrinsic"
                    log.info("reconcile_expired_intrinsic",
                             trade_id=trade.trade_id, settle_px=settle_px,
                             intrinsic=round(intrinsic, 2),
                             realized_pnl=round(realized_pnl, 2))
                else:
                    realized_pnl = 0.0
                    exit_reason = UNKNOWN_EXPIRED_EXIT_REASON
                    log.critical(
                        "reconcile_expired_unbookable",
                        trade_id=trade.trade_id, symbol=trade.symbol,
                        note="expired but underlying price unavailable — "
                             "P&L NOT booked, needs manual review",
                    )
            else:
                # Unknown exit — record neutral P&L and flag for review
                # rather than inventing a number.
                realized_pnl = 0.0
                exit_reason = UNKNOWN_EXIT_REASON
                log.warning(
                    "reconcile_unknown_exit",
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    strategy=trade.strategy,
                    note="position gone from IBKR before expiry; true exit price unrecoverable",
                )

        # R19: last gate before the write — a non-finite number (NaN/inf from
        # any branch above, incl. a NaN settle price feeding the intrinsic
        # calc) must never enter the trades table. Book the neutral
        # needs-review row instead: a $0 flagged for a human is recoverable,
        # a NaN aggregate is not.
        if not math.isfinite(realized_pnl) or not math.isfinite(exit_price):
            log.critical("reconcile_non_finite_booking_blocked",
                         trade_id=trade.trade_id, symbol=trade.symbol,
                         realized_pnl=str(realized_pnl), exit_price=str(exit_price),
                         prior_reason=exit_reason)
            realized_pnl = 0.0
            exit_price = 0.0
            exit_reason = NON_FINITE_EXIT_REASON

        log.info("reconcile_closing_stale", trade_id=trade.trade_id,
                 exit_price=exit_price, realized_pnl=realized_pnl,
                 exit_reason=exit_reason)

        self._state.close_trade(
            trade_id=trade.trade_id,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            exit_reason_detailed=exit_reason,
        )

    async def book_vanished_trade(self, trade) -> bool:
        """R14 (the strand hole): CLOSE a trade whose position is confirmed
        gone at the broker.

        reconcile()'s stale-local loop cannot do this for the case that
        matters most. Its zero-options guard — "IBKR shows no option positions
        while local option trades are open; refusing to mass-close" — fires on
        exactly the state left by manually flattening the LAST open position,
        so the booking loop never runs and the trade stays FILLED forever:
        the monitor re-demands the exit every pass, the exit is refused every
        pass, and no code path ever closes the row.

        That guard is right to distrust an empty snapshot in bulk. This path
        is safe where it isn't, because the caller has already resolved the
        ambiguity against a FRESH broker query for THIS trade specifically
        (position_liveness -> "gone"), and books one trade rather than the book.
        """
        try:
            portfolio = self._ibkr.get_portfolio()
        except Exception as e:  # noqa: BLE001
            log.error("book_vanished_portfolio_failed",
                      trade_id=trade.trade_id, error=str(e))
            portfolio = []
        try:
            await self._book_stale_trade(trade, portfolio)
            return True
        except Exception as e:  # noqa: BLE001
            log.error("book_vanished_failed",
                      trade_id=trade.trade_id, error=str(e))
            return False

    def _live_ibkr_leg_keys(self) -> dict[str, int] | None:
        """Position keys currently live at IBKR, mapped to their ABS
        quantity, or None if unavailable.

        W50 (trade-life-promotion-ignores-broker-quantity): was `set[str]`;
        position_liveness needs the quantity to detect a leg present but
        SHORT of trade.quantity, not just presence/absence. Callers that
        only need presence (e.g. _sweep_stale_pending's `.isdisjoint(...)`)
        are unaffected — a dict iterates as its keys, same as a set would.
        """
        try:
            if not self._ibkr.connected:
                return None
            live: dict[str, int] = {}
            for pos in self._ibkr.get_positions():
                if not getattr(pos, "position", 0):
                    continue  # R15b: zeroed row = closed, not live
                if pos.contract.secType == "OPT":
                    live[self._make_position_key(
                        symbol=pos.contract.symbol,
                        strike=pos.contract.strike,
                        right=pos.contract.right,
                        expiry=pos.contract.lastTradeDateOrContractMonth,
                    )] = abs(int(pos.position))
                else:
                    live[f"{pos.contract.symbol}:STK"] = abs(int(pos.position))
            return live
        except Exception as e:  # noqa: BLE001
            log.warning("live_leg_keys_failed", error=str(e))
            return None

    # R12 F4.1: orderStatus values that mean "still working at the broker".
    # Unknown/empty statuses are treated as working — the safe direction is
    # keeping a CLOSING trade CLOSING (self-heals via the adopted tracker's
    # stale/zombie handling) rather than promoting past a live close order.
    _WORKING_ORDER_STATUSES = {
        "pendingsubmit", "apipending", "presubmitted", "submitted", "",
    }

    async def _fetch_working_orders(self) -> list[tuple[int, set[str], str, str]] | None:
        """R12 F4.1: working orders at the broker as (orderId, keys, symbol, order_ref).

        Keys are leg position-keys for plain option orders; combo (BAG)
        orders only expose leg conIds, so they match at symbol level via the
        "SYMBOL:BAG" marker. Uses reqAllOpenOrders (all clientIds) so exit
        orders placed by a previous session/clientId are seen. Returns None
        when the broker view is unavailable (callers must then NOT promote
        CLOSING trades — unknown is not "no working order").

        R17: order_ref is the trade_id stamped on the order at placement
        (executor.py / orchestrator.py _close_*). Orders placed before this
        change carry no order_ref — _working_exit_order_for falls back to
        the symbol heuristic for those, same as before.
        """
        try:
            if hasattr(self._ibkr, "get_all_open_trades"):
                trades = await self._ibkr.get_all_open_trades()
            else:  # pragma: no cover — legacy client without the helper
                trades = self._ibkr.get_open_orders()
            if trades is None:
                return None
            entries: list[tuple[int, set[str], str, str]] = []
            for t in trades:
                status = str(getattr(t.orderStatus, "status", "") or "").lower()
                if status not in self._WORKING_ORDER_STATUSES:
                    continue
                c = t.contract
                keys: set[str] = set()
                sec_type = str(getattr(c, "secType", "") or "")
                if sec_type == "OPT":
                    keys.add(self._make_position_key(
                        symbol=c.symbol, strike=c.strike, right=c.right,
                        expiry=c.lastTradeDateOrContractMonth))
                elif sec_type == "BAG":
                    keys.add(f"{c.symbol}:BAG")
                else:
                    keys.add(f"{c.symbol}:STK")
                order_ref = str(getattr(t.order, "orderRef", "") or "")
                entries.append((int(t.order.orderId or 0), keys, str(c.symbol), order_ref))
            return entries
        except Exception as e:  # noqa: BLE001
            log.warning("working_orders_fetch_failed", error=str(e))
            return None

    def _working_exit_order_for(
        self, trade, working: list[tuple[int, set[str], str, str]]
    ) -> int | None:
        """R12 F4.1: orderId of a working order touching this trade's legs
        (leg-key match for plain options; symbol match for BAG combos —
        combo legs are opaque conIds at this layer, and keeping CLOSING on a
        same-symbol combo is the safe direction).

        R17: the symbol-only BAG match could misassign between two
        same-symbol multi-leg positions both mid-close after a restart. Try
        an exact order_ref==trade_id match first (stamped at placement);
        only fall back to the symbol heuristic for orders with no order_ref
        (pre-deploy orders still in flight) — and never claim an order
        that's tagged for a DIFFERENT trade.
        """
        for order_id, _keys, _symbol, order_ref in working:
            if order_ref and order_ref == trade.trade_id:
                return order_id

        trade_keys = self._trade_leg_keys(trade)
        for order_id, keys, symbol, order_ref in working:
            if order_ref:
                continue  # tagged for a different trade — never claim it
            if not keys.isdisjoint(trade_keys):
                return order_id
            if f"{symbol}:BAG" in keys and symbol == trade.symbol:
                return order_id
        return None

    def _cancel_working_entry_order(self, trade) -> bool:
        """R16: cancel a still-WORKING broker order for a stale PENDING trade.

        Returns True when a matching open order was found (cancel requested —
        the caller must NOT book "never filled" this cycle; the next sweep
        sees either the cancel or a late fill).

        W45 (bot-day-08): Pass 1/2 below use the in-session openTrades cache,
        which in practice sees orders under OTHER clientIds too (ib_insync's
        Wrapper.openOrder caches every clientId unconditionally, and
        openTrades() applies no clientId filter — verified directly against
        the installed ib_insync==0.9.86) — this docstring used to claim the
        opposite ("same-clientId orders only"), which was simply wrong. That
        protection is real but UNENFORCED: it depends on a disconnect never
        having wiped the cache (it does, on every reconnect) without a
        stash-refetch since (which only fires on a clientId CHANGE, not a
        same-clientId reconnect). Pass 3 backs it with
        IBKRClient.foreign_open_order_ids — populated from an authoritative
        all-clientId snapshot at connect time and kept accurate by
        sweep_stale_pending()'s periodic prune (see there) — so correctness
        no longer depends on that cache accident holding.

        R19: the matcher used to accept ANY active same-symbol BAG order, and
        for plain options a strike SUBSTRING test ("45.0" matches
        "SPY:745.0:P:...", expiry ignored) — so a sweep for one stale PENDING
        could cancel a DIFFERENT trade's working order, including a live EXIT
        combo (a stop/TP in flight, silently killed, and re-killed every ~30s
        cycle). Every entry/exit order has carried its trade_id in orderRef
        since R17, so: match orderRef == trade_id exactly; fall back to the
        symbol/leg heuristic ONLY for orders with no orderRef (pre-R17 orders
        still in flight), logging when it does; and NEVER touch an order
        tagged for another trade. Same contract as _working_exit_order_for.
        """
        try:
            ib = getattr(self._ibkr, "ib", None)
            try:
                open_trades = list(ib.openTrades()) if ib is not None else []
            except TypeError:
                # Not a real ib_insync view (older fake/mock): no readable
                # order book = no visible working order — pre-R16 behavior.
                # The live_keys gate upstream still guards genuinely-filled
                # orders from fiction.
                return False
            keys = self._trade_leg_keys(trade)
            _TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
            active = []
            for bt in open_trades:
                # Belt-and-braces: only genuinely ACTIVE orders count as
                # working (some fakes/caches keep terminal trades in the view).
                _st = getattr(getattr(bt, "orderStatus", None), "status", "")
                if _st in _TERMINAL:
                    continue
                active.append(bt)

            # Pass 1 — the exact answer: this trade's own tagged order.
            for bt in active:
                ref = str(getattr(getattr(bt, "order", None), "orderRef", "") or "")
                if ref and ref == trade.trade_id:
                    ib.cancelOrder(bt.order)
                    log.info("sweep_cancelled_by_order_ref",
                             trade_id=trade.trade_id, order_ref=ref)
                    return True

            # Pass 2 — legacy untagged orders only.
            for bt in active:
                ref = str(getattr(getattr(bt, "order", None), "orderRef", "") or "")
                if ref:
                    continue  # tagged for a DIFFERENT trade — never cancel it
                c = bt.contract
                sym = getattr(c, "symbol", "")
                if sym != trade.symbol:
                    continue
                sec = getattr(c, "secType", "")
                matched = sec == "BAG"
                if not matched and sec == "OPT" and keys:
                    # Full position-key equality (strike + right + expiry),
                    # never a substring of the key.
                    try:
                        matched = self._make_position_key(
                            symbol=sym,
                            strike=float(getattr(c, "strike", 0.0) or 0.0),
                            right=str(getattr(c, "right", "")),
                            expiry=str(getattr(c, "lastTradeDateOrContractMonth", "")),
                        ) in keys
                    except (TypeError, ValueError):
                        continue
                if matched:
                    ib.cancelOrder(bt.order)
                    log.warning("sweep_cancelled_untagged_order",
                                trade_id=trade.trade_id, symbol=sym,
                                sec_type=sec,
                                note="order carries no orderRef (pre-R17) — "
                                     "matched by symbol/leg heuristic")
                    return True
        except Exception as e:  # noqa: BLE001 — sweep must survive broker hiccups
            log.warning("sweep_cancel_check_failed",
                        trade_id=trade.trade_id, error=str(e))
            return True  # unknown broker state: defer booking, never fiction

        # W45 (bot-day-08) Pass 3 — neither pass above found a match in the
        # LOCAL openTrades cache. That cache can be blind (wiped by any
        # disconnect, never repopulated on a same-clientId reconnect), so
        # before concluding "no working order" at all, check the explicit
        # signal: IBKRClient.foreign_open_order_ids, populated from an
        # authoritative all-clientId snapshot at connect time and kept
        # accurate by sweep_stale_pending()'s periodic prune (self._ibkr's
        # stash, not something passed in here — the 30s hot path, which
        # cannot afford a fresh broker round-trip every tick, still checks
        # against whatever the last successful prune left, which is
        # slightly-stale-at-worst but never fabricated). This can't say
        # WHICH trade a stashed id belongs to (bare order ids carry no
        # symbol/orderRef, and a still-PENDING trade has no persisted
        # order-id column to match against) — so it defers every
        # stale-PENDING candidate this pass rather than guessing, not just
        # this one. Bounded cost: the stash is only non-empty in the window
        # after a clientId-changing reconnect until pruned back to empty, and
        # a deferred booking just retries next sweep.
        try:
            if self._ibkr.foreign_open_order_ids:
                log.warning(
                    "sweep_deferred_foreign_order_uncertainty",
                    trade_id=trade.trade_id,
                    foreign_ids=sorted(self._ibkr.foreign_open_order_ids),
                    note="a known foreign-session order id is still stashed "
                         "(broker-confirmed open at the last successful "
                         "check); deferring this stale-PENDING booking "
                         "rather than risk it belongs to this trade",
                )
                return True
        except Exception as e:  # noqa: BLE001 — sweep must survive broker hiccups
            log.warning("sweep_foreign_check_failed",
                        trade_id=trade.trade_id, error=str(e))
            return True  # unknown broker state: defer booking, never fiction
        return False

    def _request_fresh_positions(self) -> None:
        """R19: best-effort authoritative re-query kicked off from the SYNC
        sweep (the 30s orchestrator path cannot await).

        reqPositionsAsync also repopulates ib_insync's position cache, so the
        deferred booking decision has real data to work with on the next
        cycle instead of re-reading the same wedged-empty list forever. Purely
        a read; nothing is booked off it here.
        """
        try:
            import asyncio
            fresh_fn = getattr(self._ibkr, "get_positions_fresh", None)
            if not callable(fresh_fn):
                return
            loop = asyncio.get_running_loop()
        except (RuntimeError, ImportError):
            return  # no running loop (sync test/tool) — nothing to schedule
        except Exception:  # noqa: BLE001
            return
        try:
            task = loop.create_task(fresh_fn())
            # Keep a reference so the task is not garbage-collected mid-flight.
            self._fresh_refresh_task = task
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except Exception as e:  # noqa: BLE001 — a refresh must never break the sweep
            log.warning("sweep_fresh_refresh_failed", error=str(e))

    def _fresh_leg_keys(self, fresh) -> set[str]:
        """R19: position keys from an AUTHORITATIVE get_positions_fresh() list."""
        keys: set[str] = set()
        for pos in fresh or []:
            if not getattr(pos, "position", 0):
                continue  # zeroed row = closed, not live
            try:
                c = pos.contract
                if getattr(c, "secType", "") == "OPT":
                    keys.add(self._make_position_key(
                        symbol=c.symbol, strike=c.strike, right=c.right,
                        expiry=c.lastTradeDateOrContractMonth))
                else:
                    keys.add(f"{c.symbol}:STK")
            except (AttributeError, TypeError, ValueError):
                continue
        return keys

    def declared_position_risk_rows(self) -> list[dict]:
        """W30 (R26 F3, finding #3): a declared manual position has no local
        trade row, so orchestrator._sync_risk_manager_positions — the SOLE
        producer of RiskManager._open_positions — never saw it: invisible to
        the aggregate capital-at-risk cap, the symbol-concentration cap, the
        per-expiry cap, correlation, and the open-position count. Synthesizes
        one row per declared key from the (cached) broker position list,
        priced via the same conservative structural/Reg-T proxy a real
        position's aggregate-risk backfill already uses.
        """
        from ait.execution import manual_positions as _mp
        from ait.risk.manager import RiskManager
        rows: list[dict] = []
        try:
            positions = self._ibkr.get_positions()
        except Exception as e:  # noqa: BLE001 — a sync failure must not break the risk sync
            log.warning("declared_position_risk_rows_failed", error=str(e))
            return rows
        for pos in positions:
            qty = getattr(pos, "position", 0)
            if not qty:
                continue
            c = pos.contract
            if getattr(c, "secType", "") != "OPT":
                continue
            key = self._make_position_key(
                symbol=c.symbol, strike=c.strike, right=c.right,
                expiry=c.lastTradeDateOrContractMonth)
            if not _mp.is_declared(self._state, key):
                continue
            action = "SELL" if qty < 0 else "BUY"
            max_loss = RiskManager._structural_max_loss(
                [{"strike": c.strike, "right": c.right, "action": action}], qty)
            if max_loss <= 0:
                # A long option (or an unpriceable structure) — the premium
                # already paid is the conservative floor; never zero for a
                # real position (the R16 aggregate-risk lesson).
                try:
                    max_loss = abs(float(getattr(pos, "avgCost", 0) or 0)) * abs(qty)
                except (TypeError, ValueError):
                    max_loss = 0.0
            rows.append({
                "symbol": c.symbol,
                "strategy": "declared_manual",
                "expiry": self._normalize_expiry(c.lastTradeDateOrContractMonth),
                "quantity": qty,
                "delta": 0, "gamma": 0, "theta": 0, "vega": 0,
                "max_loss": max_loss,
                "market_value": max_loss,
            })
        return rows

    async def sweep_stale_pending(self) -> int:
        """R19: async entry point for the stale-PENDING sweep that can resolve
        an empty position cache authoritatively before anything is booked.

        The sync _sweep_stale_pending() defers (books nothing) when the cached
        list is empty; this wrapper answers the ambiguity the way
        position_liveness and the zero-options guard do — one fresh broker
        query — and hands the answer down through _fresh_keys_for_sweep.

        The answer travels on the instance rather than as an argument so
        _sweep_stale_pending keeps the zero-arg signature the orchestrator's
        30s cycle calls it with; it is consumed exactly once, so a
        confirmation can never be re-used on a later cycle.

        W45 (bot-day-08): also refreshes the foreign-order-id stash
        (IBKRClient.foreign_open_order_ids) against authoritative broker
        truth before the sweep runs. That stash is add-only at connect time
        and otherwise never shrinks on its own; _cancel_working_entry_order's
        Pass 3 (below) defers a booking whenever it's non-empty, so pruning
        it here against a fresh get_all_open_order_ids() snapshot is what
        keeps Pass 3 from deferring forever once a stashed order has actually
        resolved (filled/cancelled/expired) at the broker. Deliberately NOT
        threaded down as a parameter (unlike fresh_keys): Pass 3 just reads
        the stash directly, so a call with no fresh prune this cycle (the
        orchestrator's 30s hot path, which cannot afford this round-trip
        every tick) still checks against the last successfully-pruned state
        rather than skipping the check entirely — slightly stale is still
        grounded in a real broker observation, never fabricated.
        """
        fresh_keys = None
        try:
            fresh_fn = getattr(self._ibkr, "get_positions_fresh", None)
            if callable(fresh_fn):
                fresh = await fresh_fn()
                if fresh is not None:
                    fresh_keys = self._fresh_leg_keys(fresh)
        except Exception as e:  # noqa: BLE001 — unknown stays unknown
            log.warning("sweep_fresh_positions_failed", error=str(e))
            fresh_keys = None
        self._fresh_keys_for_sweep = fresh_keys

        try:
            ids_fn = getattr(self._ibkr, "get_all_open_order_ids", None)
            prune_fn = getattr(self._ibkr, "prune_foreign_open_orders", None)
            if callable(ids_fn) and callable(prune_fn):
                broker_open_ids = await ids_fn()
                prune_fn(broker_open_ids)
        except Exception as e:  # noqa: BLE001 — a failed refresh just leaves
            # the stash as it was; never treat a failed prune as "resolved".
            log.warning("sweep_foreign_stash_refresh_failed", error=str(e))

        try:
            return self._sweep_stale_pending()
        finally:
            self._fresh_keys_for_sweep = None

    def _refund_entry_budget(self, trade) -> None:
        """W26 (data-integrity-07): give back the intraday entry slot a phantom took.

        `stats.trades_taken += 1` fires when an order is PLACED
        (orchestrator.py, inside `if trade_id:`) and had NO decrement anywhere,
        so an order that never filled consumed a trade slot permanently.
        _get_trade_budget returns `min(budget, max_trades) - taken`, and the
        first-hour budget is 2 — one phantom halved first-hour capacity for the
        rest of the session.

        Two correctness points, both of which would turn this fix into a worse bug
        if got wrong:

        * THE TRADE'S OWN DATE, never today. reconcile() runs on every startup, so
          a phantom placed Monday is typically swept Tuesday morning. Decrementing
          TODAY would credit a slot that today never spent, and `taken = -1` makes
          the budget expression return max_trades + 1 — a fail-open on the daily
          cap, from a data-integrity fix.
        * FLOOR AT 0. A negative counter inflates the budget the same way. (W25
          shipped the mirror-image lesson: there a floor of 1 manufactured a
          position where 0 was correct. A floor is only ever safe in the
          direction that cannot create permission.)

        Never raises: the sweep's job is booking, and a stats hiccup must not
        abort it or leave the trade wrongly open.
        """
        try:
            entry_date = parse_trade_ts(trade.entry_time).date()
        except (ValueError, TypeError) as e:
            log.warning("budget_refund_skipped_unparseable_entry_time",
                        trade_id=getattr(trade, "trade_id", None),
                        entry_time=getattr(trade, "entry_time", None), error=str(e))
            return
        try:
            stats = self._state.get_daily_stats(entry_date)
            before = int(stats.trades_taken or 0)
            if before <= 0:
                # Nothing to give back. Not an error: the counter may legitimately
                # be 0 on a restored/backfilled row, and driving it negative is
                # the one outcome this must never produce.
                log.info("budget_refund_noop", trade_id=trade.trade_id,
                         date=entry_date.isoformat(), trades_taken=before)
                return
            stats.trades_taken = before - 1
            self._state.update_daily_stats(stats)
            log.info("budget_refunded_never_filled",
                     trade_id=trade.trade_id, symbol=trade.symbol,
                     date=entry_date.isoformat(),
                     trades_taken_before=before, trades_taken_after=before - 1)
        except Exception as e:  # noqa: BLE001 — must never abort the sweep
            log.warning("budget_refund_failed", trade_id=trade.trade_id,
                        date=entry_date.isoformat(), error=str(e))

    def _sweep_stale_pending(self) -> int:
        """Close PENDING trades too old to still be live working orders.

        LIVENESS-GATED (audit R2/C3): the old sweep closed any 30-min-old
        PENDING as "never filled, $0" WITHOUT checking IBKR — front-running
        the promotion rescue in reconcile(). A genuinely-filled order whose
        in-memory tracker died in a restart was booked as fiction while the
        real position lived on unmanaged. Now: if any of the trade's legs are
        live at IBKR, PROMOTE to FILLED instead of closing; if IBKR positions
        can't be read this cycle, don't sweep at all (pending dedup still
        protects — no fiction gets written).

        R19: only DISCONNECTED/exception used to stop the sweep, so an
        EMPTY-but-connected ib.positions() cache (Gateway stream reset,
        reqPositions timeout that ib_insync only logs) read as "no legs live"
        — the exact ambiguity every other 'gone' decision in this file refuses
        to take on trust (position_liveness, the zero-options guard,
        ibkr_client.get_positions_fresh's own docstring). A filled-but-
        tracker-lost PENDING was then booked $0 "never filled" while the real
        position lived unmanaged. An empty cache now requires an authoritative
        answer — the one the async sweep_stale_pending() wrapper leaves in
        _fresh_keys_for_sweep — or nothing is booked this cycle.
        """
        # Consume the authoritative answer exactly once (None = we don't have
        # one; a confirmation must never be re-used on a later cycle).
        fresh_keys = getattr(self, "_fresh_keys_for_sweep", None)
        self._fresh_keys_for_sweep = None
        closed = 0
        try:
            now = now_et()
            stale = []
            for t in self._state.get_open_trades():
                if t.status != TradeStatus.PENDING:
                    continue
                try:
                    age_min = (now - parse_trade_ts(t.entry_time)).total_seconds() / 60
                except (ValueError, TypeError) as _pe:
                    # W21: this guard is what would make a half-migrated corpus
                    # SILENT. Mixing naive and aware entry_time raises TypeError
                    # here, and the bare `continue` then drops the trade from the
                    # sweep entirely — stale-PENDING reconciliation degrades to a
                    # no-op with nothing in the log to say so. Keep the guard (a
                    # single bad row must not abort the sweep) but never keep
                    # quiet about which row it was.
                    log.warning("sweep_entry_time_unparseable",
                                trade_id=t.trade_id, entry_time=t.entry_time,
                                error=str(_pe))
                    continue
                if age_min >= self.STALE_PENDING_MINUTES:
                    stale.append((t, age_min))
            if not stale:
                return 0
            live_keys = self._live_ibkr_leg_keys()
            if live_keys is None:
                log.warning("sweep_skipped_no_ibkr_positions")
                return 0
            if not live_keys:
                # R19: empty-but-connected == ambiguous. Only a FRESH broker
                # answer may retire a position here.
                if fresh_keys is not None:
                    live_keys = fresh_keys
                    log.info("sweep_using_fresh_positions", legs=len(live_keys))
                elif callable(getattr(self._ibkr, "get_positions_fresh", None)):
                    log.warning(
                        "sweep_deferred_empty_position_cache",
                        stale_pendings=len(stale),
                        note="cached position list is EMPTY while stale PENDING "
                             "trades exist — ambiguous between flat and a wedged "
                             "stream; booking deferred until a fresh query "
                             "confirms. Requesting one now.",
                    )
                    self._request_fresh_positions()
                    return 0
                else:
                    # No authoritative-query surface on this client at all
                    # (legacy client / test fake): there is nothing to wait
                    # for, so keep the pre-R19 behaviour rather than wedging
                    # every orphan open forever — but never silently.
                    log.warning(
                        "sweep_unconfirmed_empty_position_cache",
                        stale_pendings=len(stale),
                        note="client exposes no get_positions_fresh — booking "
                             "off an UNCONFIRMED empty position cache",
                    )
            for t, age_min in stale:
                if not self._trade_leg_keys(t).isdisjoint(live_keys):
                    # Legs live at the broker — the order FILLED. Rescue it.
                    # R12 Tier-A #1: CAS PENDING->FILLED (a concurrent writer
                    # may have moved the trade since get_open_trades()).
                    if self._state.transition(
                            t.trade_id, (TradeStatus.PENDING,), TradeStatus.FILLED):
                        # W50: adopt broker quantity when we have it — this
                        # is a PENDING rescue, so a partial fill this process
                        # missed is exactly as dangerous here as in
                        # reconcile()'s own PENDING->FILLED branch. Only when
                        # live_keys carries quantities (the common case,
                        # sourced from _live_ibkr_leg_keys); the R19 empty-
                        # cache fallback (fresh_keys, a bare set[str] with no
                        # quantity signal) has nothing to adopt or flag
                        # against, so this is skipped rather than guessed.
                        # No frozen_keys list here — this function doesn't
                        # own the HALT_UNTRACKED write (reconcile() does,
                        # once per pass); a non-uniform find is logged
                        # loudly and left for the next reconcile() pass to
                        # freeze, while position_liveness's own quantity
                        # check (not this freeze) is what actually keeps the
                        # later exit from reversing the wrong amount.
                        _t = t
                        if isinstance(live_keys, dict):
                            _t = self._adopt_or_flag_quantity(t, live_keys, [])
                        # R16: rescued trades had no open_positions row, so
                        # HWM/trailing-lock/partial-exit tracking was dead.
                        self._ensure_open_position_row(_t)
                    log.warning("sweep_promoted_filled_pending",
                                trade_id=t.trade_id, symbol=t.symbol,
                                strategy=t.strategy, age_min=int(age_min))
                    continue
                # R16: a WORKING entry order at the broker means "not filled
                # YET", not "never filled" — booking $0 while the order rests
                # lets it fill later as an untracked position. Cancel the
                # order first; book only once no working order remains.
                if self._cancel_working_entry_order(t):
                    log.warning("sweep_cancelled_working_entry_order",
                                trade_id=t.trade_id, symbol=t.symbol,
                                note="entry order was still WORKING at the "
                                     "broker; cancelled — booking deferred "
                                     "to the next sweep cycle")
                    continue
                # R16: CAS narrowed to PENDING — a concurrent promotion to
                # FILLED (second process / broker-lag race) must always win
                # over a $0 "never filled" booking.
                # W26 (data-integrity-07): the return value was DISCARDED and
                # `closed += 1` ran unconditionally, so a lost CAS was counted as
                # a close. It also gates the budget give-back below: handing a
                # trade slot back for an order that actually FILLED would be
                # strictly worse than the bug being fixed.
                _booked = self._state.close_trade(
                    trade_id=t.trade_id, exit_price=0.0, realized_pnl=0.0,
                    exit_reason_detailed=NEVER_FILLED_EXIT_REASON,
                    from_statuses=(TradeStatus.PENDING,),
                )
                if not _booked:
                    log.warning("sweep_close_lost_cas",
                                trade_id=t.trade_id, symbol=t.symbol,
                                note="close CAS refused — the row left PENDING "
                                     "concurrently (promoted to FILLED); neither "
                                     "counted nor refunded")
                    continue
                closed += 1
                self._refund_entry_budget(t)
                log.warning("reconcile_stale_pending_closed",
                            trade_id=t.trade_id, symbol=t.symbol,
                            strategy=t.strategy, age_min=int(age_min))
        except Exception as e:  # noqa: BLE001
            log.warning("sweep_stale_pending_failed", error=str(e))
        return closed

    async def reconcile(self) -> ReconciliationResult:
        """Perform full reconciliation between local state and IBKR.

        This should be called on every bot startup.
        """
        log.info("reconciliation_starting")

        # NOTE (audit R2/C3): the stale-pending sweep used to run HERE —
        # before the IBKR promotion loop — so a filled-but-tracker-lost
        # PENDING was closed as "$0 never filled" before promotion could
        # rescue it. The sweep is now liveness-gated AND runs after the
        # promotion pass (end of this method).

        # Get IBKR positions
        ibkr_positions = self._ibkr.get_positions()
        ibkr_portfolio = self._ibkr.get_portfolio()

        # Build IBKR position map using normalized keys
        ibkr_map: dict[str, dict] = {}
        for pos in ibkr_positions:
            if not getattr(pos, "position", 0):
                continue  # R15b: a zeroed row must not "match" local legs
            if pos.contract.secType == "OPT":
                key = self._make_position_key(
                    symbol=pos.contract.symbol,
                    strike=pos.contract.strike,
                    right=pos.contract.right,
                    expiry=pos.contract.lastTradeDateOrContractMonth,
                )
            else:
                # For stocks, just use symbol
                key = f"{pos.contract.symbol}:STK"
            ibkr_map[key] = {
                "symbol": pos.contract.symbol,
                "sec_type": pos.contract.secType,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "contract": pos.contract,
            }
        # W50: plain key -> abs(quantity), the shape _adopt_or_flag_quantity
        # takes — computed once rather than re-derived from ibkr_map on
        # every promoted trade.
        ibkr_qty_map = {k: abs(int(v["quantity"])) for k, v in ibkr_map.items()}

        # Get local open trades. Multi-leg positions live in IBKR as 2-4
        # SEPARATE option positions, so each local trade maps to a SET of
        # leg keys (from its legs JSON) — a single synthesized key can never
        # match a condor's 4 legs and historically caused every multi-leg
        # trade to be declared stale and force-closed at the first sweep.
        local_trades = self._state.get_open_trades()
        local_entries: list[tuple] = []  # (trade, set_of_leg_keys)
        local_map: dict[str, list] = {}  # any leg key -> [trade, ...]
        for trade in local_trades:
            keys = self._trade_leg_keys(trade)
            local_entries.append((trade, keys))
            for key in keys:
                local_map.setdefault(key, []).append(trade)

        result = ReconciliationResult(
            matched=0,
            new_from_ibkr=0,
            stale_local=0,
            discrepancies=[],
        )

        # R12 F4.1: broker working orders are needed BEFORE promoting any
        # CLOSING trade — a live close order means the exit is in flight, and
        # promoting past it re-triggers a second close (fills of both =
        # position REVERSAL). Fetched once, and only when a CLOSING trade
        # exists locally.
        working_orders: list[tuple[int, set[str], str, str]] | None = None
        if any(t.status == TradeStatus.CLOSING for t in local_trades):
            working_orders = await self._fetch_working_orders()

        # W22: collected across this pass, written to HALT_UNTRACKED ONCE
        # after the loop — fixes the old per-iteration write_text, which
        # overwrote itself on every untracked key and left the file naming
        # only the LAST one (live-confirmed twice with a real two-leg
        # spread: both legs froze, the file named only one).
        frozen_keys: list[str] = []

        # Check IBKR positions against local state
        promoted_ids: set[str] = set()
        # R16: a CLOSING trade kept because the broker order view is
        # unavailable is ONE fact about ONE trade, but the branch below sits
        # inside the per-IBKR-key match loop and — unlike the adopted branch,
        # which records the trade in result.closing_exit_orders — recorded
        # nothing, so all 4 legs of a condor re-entered it and the EOD recon
        # Telegram counted 4 identical BREAKs. Dedup like promoted_ids.
        kept_unknown_ids: set[str] = set()
        for key, ibkr_pos in ibkr_map.items():
            if key in local_map:
                result.matched += 1
                matching_trades = local_map[key]
                # Recover orphaned entries: a trade still marked PENDING (or
                # CLOSING) whose legs are LIVE in IBKR actually filled — the
                # in-memory fill tracker was lost on restart. Promote it to
                # FILLED so the portfolio monitor manages it (check_positions
                # only evaluates FILLED/PARTIAL).
                for _mt_idx, local_trade in enumerate(matching_trades):
                    if local_trade.trade_id in promoted_ids:
                        continue
                    if local_trade.trade_id in result.closing_exit_orders:
                        continue  # already resolved as kept-CLOSING
                    if local_trade.trade_id in kept_unknown_ids:
                        continue  # R16: already reported once for this trade
                    if local_trade.status == TradeStatus.PENDING:
                        # R12 Tier-A #1: CAS PENDING->FILLED.
                        if self._state.update_trade_status(
                                local_trade.trade_id, TradeStatus.FILLED,
                                from_statuses=(TradeStatus.PENDING,)):
                            promoted_ids.add(local_trade.trade_id)
                            result.promoted += 1
                            # W50: adopt broker quantity (or flag + freeze)
                            # BEFORE the open_positions row is written, so
                            # HWM/partial-exit tracking and the exit path
                            # both see the right number from the start.
                            # Keep matching_trades in sync so the quantity-
                            # mismatch check below reads the adopted figure,
                            # not a now-stale one.
                            local_trade = self._adopt_or_flag_quantity(
                                local_trade, ibkr_qty_map, frozen_keys)
                            matching_trades[_mt_idx] = local_trade
                            # R16: promotions never inserted the
                            # open_positions row the HWM/partial-exit
                            # ledger lives in.
                            self._ensure_open_position_row(local_trade)
                            log.warning("reconcile_promoted_to_filled",
                                        trade_id=local_trade.trade_id,
                                        symbol=local_trade.symbol,
                                        strategy=local_trade.strategy,
                                        prior_status=local_trade.status.value)
                    elif local_trade.status == TradeStatus.CLOSING:
                        # R12 F4.1: only promote CLOSING->FILLED when NO
                        # working order exists on the trade's legs. If one
                        # does (or the broker view is unavailable), keep
                        # CLOSING and report the orderId so the orchestrator
                        # re-registers the exit tracker via
                        # executor.adopt_exit_order().
                        _oid = (self._working_exit_order_for(local_trade, working_orders)
                                if working_orders else None)
                        if _oid is not None:
                            result.closing_exit_orders[local_trade.trade_id] = _oid
                            log.warning("reconcile_closing_kept_working_order",
                                        trade_id=local_trade.trade_id,
                                        symbol=local_trade.symbol,
                                        order_id=_oid)
                        elif working_orders is None:
                            # Unknown broker order state — promoting could
                            # duplicate a live close. Keep CLOSING; the next
                            # reconcile retries with a working broker view.
                            # R16: report ONCE per trade, not once per leg key
                            # (a condor matched here 4x = 4 identical BREAKs).
                            kept_unknown_ids.add(local_trade.trade_id)
                            result.discrepancies.append(
                                f"CLOSING trade {local_trade.trade_id} kept: "
                                f"broker open-order view unavailable")
                            log.warning("reconcile_closing_kept_orders_unknown",
                                        trade_id=local_trade.trade_id,
                                        symbol=local_trade.symbol)
                        else:
                            if self._state.transition(
                                    local_trade.trade_id,
                                    (TradeStatus.CLOSING,), TradeStatus.FILLED):
                                promoted_ids.add(local_trade.trade_id)
                                result.promoted += 1
                                # W50: same adoption/flag as the PENDING
                                # branch above — see _adopt_or_flag_quantity.
                                # This is the MORE dangerous of the two: a
                                # CLOSING trade rescued here is about to be
                                # re-offered to the exit monitor, and a stale
                                # quantity here is exactly the reversal risk
                                # trade-life-promotion-ignores-broker-quantity
                                # describes.
                                local_trade = self._adopt_or_flag_quantity(
                                    local_trade, ibkr_qty_map, frozen_keys)
                                matching_trades[_mt_idx] = local_trade
                                # R16: see _ensure_open_position_row — a
                                # CLOSING->FILLED rescue is exactly the case
                                # where the row went missing at restart.
                                self._ensure_open_position_row(local_trade)
                                log.warning("reconcile_promoted_to_filled",
                                            trade_id=local_trade.trade_id,
                                            symbol=local_trade.symbol,
                                            strategy=local_trade.strategy,
                                            prior_status=local_trade.status.value)
                # Check quantity matches
                local_qty = sum(abs(t.quantity) for t in matching_trades)
                if abs(ibkr_pos["quantity"]) != local_qty:
                    msg = (
                        f"Quantity mismatch for {key}: "
                        f"IBKR={ibkr_pos['quantity']}, local_total={local_qty}"
                    )
                    result.discrepancies.append(msg)
                    log.warning("reconcile_quantity_mismatch", position=key,
                                ibkr_qty=ibkr_pos["quantity"], local_qty=local_qty)
            else:
                result.new_from_ibkr += 1
                # ASSIGNMENT DETECTION (audit R2): an untracked STOCK position
                # is the signature of a short option being assigned — 100
                # shares/contract appear, consume buying power, and previously
                # only produced this log line. Flag it unmistakably so the
                # orchestrator can alert; without handling it corrupts both
                # the position count and every margin-based check.
                if ibkr_pos.get("sec_type") == "OPT":
                    # W22: a DECLARED manual position (operator confirmed via
                    # Telegram) is ignored for management — never re-enters
                    # the debounce/freeze machinery below — but stays counted
                    # in result.new_from_ibkr above like any other broker
                    # position, and in every exposure/delta calculation
                    # elsewhere (those read ib.portfolio() directly, never
                    # filtered by local-trades membership).
                    from ait.execution import manual_positions as _mp
                    if _mp.declared_quantity_matches(
                            self._state, key, ibkr_pos["quantity"]):
                        log.info("declared_manual_position_seen", position=key,
                                 qty=ibkr_pos["quantity"])
                        continue

                    # INST-3: a live OPTION at the broker with no local record
                    # = the mid-placement crash window. It has no stop, no TP,
                    # no expiry handling. Freeze new entries until a human
                    # resolves it (delete data/HALT_UNTRACKED to resume) —
                    # or, W22, declares it manual via the Telegram prompt.
                    #
                    # R5 follow-up (2026-07-09): DEBOUNCE — the Gateway's
                    # morning position stream produced a phantom row twice
                    # (QQQ 500C Aug-21, 07-08 09:17 and 07-09 09:43) that a
                    # direct broker query minutes later disproved both times.
                    # Halting on a single sighting froze entries for hours on
                    # a position that never existed. Require the same key to
                    # be sighted on TWO DISTINCT reconcile runs (>=2 min and
                    # <=8 h apart) before freezing; a real crash-window
                    # position persists at the broker and halts on the next
                    # periodic reconcile.
                    # W22: durable now (bot_state), not an in-memory dict —
                    # the old _untracked_sightings dict reset on every
                    # restart, and this box restarts often; a genuine
                    # crash-window position could evade the freeze
                    # indefinitely on a frequently-restarted box.
                    _gap_min = _mp.record_sighting(self._state, key)
                    # W22: once a key is CONFIRMED, do not re-run the gap
                    # check on later passes — record_sighting resets
                    # last_seen every call, so two reconcile() calls landing
                    # within 2 minutes of each other would otherwise re-read
                    # as "1st sighting, deferred" and silently drop an
                    # already-live freeze for a cycle. The debounce's job is
                    # proving persistence; once proven, it stays proven
                    # until declared or gone — independent of whether the
                    # Telegram prompt below actually sends.
                    if not _mp.is_confirmed(self._state, key) and (
                            _gap_min is None or _gap_min < 2 or _gap_min > 480):
                        # R12 F3.1/chaos#4B follow-through: the debounce was
                        # meant to delay the FREEZE, but it also swallowed the
                        # discrepancy — the EOD recon Telegram reported 0
                        # BREAKS on the day an untracked option first showed.
                        # Report the break same-day; only the HALT file still
                        # waits for the second sighting.
                        result.discrepancies.append(
                            f"UNTRACKED OPTION POSITION (1st sighting, freeze "
                            f"deferred): {key}, qty={ibkr_pos['quantity']} — "
                            f"freezes if still present next reconcile")
                        log.warning("untracked_option_first_sighting_debounced",
                                    position=key, qty=ibkr_pos["quantity"],
                                    note="will freeze if still present next reconcile")
                        continue
                    frozen_keys.append(key)
                    _mp.mark_confirmed(self._state, key)
                    # getattr-guarded: several existing tests construct this
                    # class via __new__ (bypassing __init__) to avoid a full
                    # IBKR client — _telegram would not exist on those at all.
                    _telegram = getattr(self, "_telegram", None)
                    if _telegram is not None:
                        await _mp.prompt_if_needed(
                            self._state, _telegram, key, ibkr_pos["quantity"])
                    msg = (
                        f"UNTRACKED OPTION POSITION (no local record): {key}, "
                        f"qty={ibkr_pos['quantity']} — NEW ENTRIES FROZEN "
                        f"(data/HALT_UNTRACKED). Resolve manually, reply to the "
                        f"Telegram prompt if it's yours, or delete the file."
                    )
                    log.critical("reconcile_untracked_option_freeze",
                                 position=key, qty=ibkr_pos["quantity"])
                elif ibkr_pos.get("sec_type") == "STK":
                    # W41 (R26 model-vs-reality-ex-div-assignment-void, piece
                    # 1): this branch logged and reported a discrepancy but
                    # never joined `frozen_keys` — the ONLY branch of the
                    # three that doesn't. An assigned/exercised short option
                    # leaves a live, unmanaged stock position (no stop, no
                    # exit) while new entries stayed completely UNFROZEN.
                    # Fixed there; two follow-ups explicitly left open for
                    # "W51's deeper pass" are resolved here:
                    #
                    # (1) No debounce, deliberately (kept from W41's own
                    # reasoning, not revisited): the OPT branch's two-sighting
                    # rule exists for an empirically observed position-STREAM
                    # flicker (R5, 2026-07-09, QQQ 500C sighted-then-disproved
                    # twice). Assignment is a discrete overnight broker event,
                    # not a stream artifact, and nothing suggests a STK
                    # sighting flickers the same way — freezing on the first
                    # sighting is correct here, not a gap.
                    # (2) DOES now route through the same Telegram
                    # declare-yes/no flow the OPT branch uses (kind="stock"
                    # so the prompt correctly says "stock", not "option") —
                    # for consistency and so a genuinely manual stock
                    # position (a declared hedge, a transferred-in holding)
                    # doesn't re-freeze every single reconcile pass forever.
                    from ait.execution import manual_positions as _mp
                    if _mp.declared_quantity_matches(
                            self._state, key, ibkr_pos["quantity"]):
                        log.info("declared_manual_position_seen", position=key,
                                 qty=ibkr_pos["quantity"])
                        continue
                    frozen_keys.append(key)
                    _telegram = getattr(self, "_telegram", None)
                    if _telegram is not None:
                        await _mp.prompt_if_needed(
                            self._state, _telegram, key, ibkr_pos["quantity"],
                            kind="stock")
                    msg = (
                        f"UNTRACKED STOCK POSITION (possible ASSIGNMENT): {key}, "
                        f"qty={ibkr_pos['quantity']}, avg_cost={ibkr_pos['avg_cost']} "
                        f"— needs manual liquidation/review — NEW ENTRIES FROZEN "
                        f"(data/HALT_UNTRACKED). Resolve manually, reply to the "
                        f"Telegram prompt if it's yours, or delete the file."
                    )
                    log.critical("reconcile_untracked_stock_assignment",
                                 position=key, qty=ibkr_pos["quantity"])
                else:
                    msg = (
                        f"New position from IBKR: {key}, "
                        f"qty={ibkr_pos['quantity']}, avg_cost={ibkr_pos['avg_cost']}"
                    )
                    log.warning("reconcile_new_ibkr_position", position=key)
                result.discrepancies.append(msg)

        # W22: write HALT_UNTRACKED as ONE complete snapshot of this pass's
        # frozen keys — a declared or since-closed key that was frozen on a
        # PRIOR pass and isn't in frozen_keys this time is correctly dropped
        # here, no separate cleanup needed. Empty list -> no undeclared,
        # unresolved untracked option remains -> delete the file.
        # W30 (R26 F3, findings #1/#2): both the HALT_UNTRACKED clear below
        # and reconcile_declared_positions() take an IRREVERSIBLE action —
        # unlinking an operator kill-switch file, deleting declared-position
        # state — on "absent from ibkr_map". ibkr_map is built from
        # get_positions()'s CACHE, which reads back EMPTY on any disconnect
        # or Gateway hiccup, identically to a genuinely flat book
        # (get_positions_fresh's own docstring already makes this contract).
        # R26 caught a single reconcile() pass emitting halt_untracked_cleared
        # AND reconcile_skipping_stale_close (the mass-close guard further
        # below, refusing to trust this SAME empty snapshot) together —
        # resolve the ambiguity the same way here instead of contradicting it.
        # Only an EMPTY cache is ambiguous: a non-empty ibkr_map is positive
        # proof the client is connected and answering for real.
        positions_confirmed_flat = True
        fresh_declared_keys: set[str] | None = None
        if not ibkr_positions:
            _declared_fresh_fn = getattr(self._ibkr, "get_positions_fresh", None)
            if callable(_declared_fresh_fn):
                _declared_fresh = await _declared_fresh_fn()
                if _declared_fresh is None:
                    positions_confirmed_flat = False
                    log.warning(
                        "declared_position_reconcile_unconfirmed",
                        note="position cache is empty and a fresh re-query "
                             "could not confirm the book is flat — skipping "
                             "the HALT_UNTRACKED clear and the declared-"
                             "position cleanup this pass",
                    )
                else:
                    fresh_declared_keys = self._fresh_leg_keys(_declared_fresh)
            # else: no authoritative-query surface on this client at all
            # (legacy client / test fake) — nothing to confirm against; keep
            # the pre-W30 cache-trusting behaviour rather than wedging every
            # freeze/declared-position cleanup forever.

        from pathlib import Path as _P
        _halt_path = _P("data/HALT_UNTRACKED")
        try:
            if frozen_keys:
                _halt_path.write_text(
                    "untracked positions (resolve at IBKR, or answer the "
                    "Telegram prompt, then this file clears itself):\n"
                    + "\n".join(f"  {k}" for k in frozen_keys))
                log.info("halt_untracked_written", keys=frozen_keys)
            elif _halt_path.exists() and positions_confirmed_flat:
                # 2026-09-16 investigation note: this file went missing once
                # with no reconcile() call in the window and no unlink call
                # logged here either — meaning THIS branch, the only unlink
                # path anywhere in the tree, did not fire. Cause never
                # identified. This log line exists so a recurrence is a log
                # lookup: if it fires, the code did it (and this proves
                # when); if the file vanishes with NEITHER this line NOR a
                # reconciliation_starting/complete pair anywhere near it,
                # the cause is external to this codebase, same as that day.
                _halt_path.unlink()
                log.info("halt_untracked_cleared",
                         reason="all previously-frozen keys now declared or gone")
        except Exception as e:  # noqa: BLE001
            log.warning("halt_untracked_write_failed", error=str(e))

        # W22: confirm + clean up any DECLARED position the operator has
        # since closed at the broker — the "confirms when it disappears"
        # half of the design.
        # W30: skipped outright when the pass above could not confirm the
        # book is flat (positions_confirmed_flat is False) — see above.
        _telegram = getattr(self, "_telegram", None)
        if _telegram is not None and positions_confirmed_flat:
            from ait.execution import manual_positions as _mp
            _declared_keys = (fresh_declared_keys if fresh_declared_keys is not None
                              else set(ibkr_map.keys()))
            try:
                await _mp.reconcile_declared_positions(
                    self._state, _telegram, _declared_keys)
            except Exception as e:  # noqa: BLE001
                log.warning("manual_position_cleanup_failed", error=str(e))
        else:
            # W39 (R26 `w22-leaked-msgid-keeps-a-dead-position-pending-forever`,
            # residual). Skipping is CORRECT — W30 made this fail closed so an
            # ambiguous position cache can never be taken as proof a position
            # is gone — but it was silent, and the skip is the only state in
            # which a prompt for a closed position lingers. Both consumers
            # (pending_keys, and the digest's awaiting_reply) read the msgid
            # row this sweep would have cleared, so while it is skipped the
            # operator sees "awaiting your reply" for something that may be
            # long gone, with nothing to explain why. Say so instead.
            from ait.execution import manual_positions as _mp
            _stale = _mp.pending_keys(self._state)
            if _stale:
                log.info(
                    "manual_position_cleanup_skipped",
                    reason=("no_telegram" if _telegram is None
                            else "positions_not_confirmed_flat"),
                    pending=len(_stale),
                    note="prompts stay pending until a pass can confirm the "
                         "broker set authoritatively",
                )

        # Check local positions not in IBKR (may have been closed while bot was down).
        # SAFETY: if IBKR reports ZERO OPTION positions while we hold open
        # local OPTION trades, the position list is almost certainly
        # stale/unsynced (fresh connect, Gateway reset, reqPositions timeout
        # that ib_insync only logs) — refuse to mass-close on it. Checking
        # options specifically: an unrelated stock position (e.g. an
        # assignment artifact) must not disarm this guard.
        ibkr_has_options = any(
            v.get("sec_type") == "OPT" for v in ibkr_map.values()
        )
        local_has_options = any(
            t.contract_type != "stock" for t, _ in local_entries
        )
        if local_has_options and not ibkr_has_options:
            # R14 #10: the cached snapshot shows zero options while we hold open
            # option trades — almost always a stale/unsynced list, so the bulk
            # stale-local loop below must NOT run on it (that is the mass-close
            # this guard has always prevented). But the blunt refusal also
            # STRANDS a book that is GENUINELY flat — every position closed
            # while the bot was down / after hours, no exit ever pending so the
            # _execute_exit gone-path never fired — leaving those rows FILLED
            # forever. Resolve the ambiguity the way position_liveness does: an
            # AUTHORITATIVE re-query. Only when a FRESH result confirms zero
            # options do we book, and then only the FILLED/PARTIAL option trades
            # (targeted, via the same booker) — PENDING orphans still belong to
            # the stale-pending sweep, and stocks are never auto-closed here.
            # None (broker won't answer) or any live option keeps the refusal.
            # getattr-guarded so an older IBKR fake without the method simply
            # reads as "can't confirm" = refuse, i.e. the prior behaviour.
            fresh_fn = getattr(self._ibkr, "get_positions_fresh", None)
            fresh = await fresh_fn() if callable(fresh_fn) else None
            fresh_has_options = (
                any(getattr(p.contract, "secType", "") == "OPT"
                    and getattr(p, "position", 0) for p in fresh)
                if fresh is not None else None
            )
            if fresh is not None and not fresh_has_options:
                booked = 0
                for _t, _keys in local_entries:
                    if getattr(_t, "contract_type", "") == "stock":
                        continue
                    _bookable = _t.status in (TradeStatus.FILLED, TradeStatus.PARTIAL)
                    if not _bookable and _t.status == TradeStatus.CLOSING:
                        # R16: a CLOSING trade whose exit filled while the bot
                        # was down (or was manually flattened) used to fall
                        # through this filter and wedge in CLOSING forever —
                        # counted against dedup/max-position/risk caps on every
                        # cycle. Book it ONLY when the broker view is available
                        # AND shows no working exit order (a working order
                        # means the exit is still in flight: keep + adopt).
                        _woid = (self._working_exit_order_for(_t, working_orders)
                                 if working_orders else None)
                        _bookable = working_orders is not None and _woid is None
                        if _bookable:
                            log.warning("reconcile_booking_wedged_closing",
                                        trade_id=_t.trade_id,
                                        symbol=_t.symbol)
                    if _bookable:
                        result.stale_local += 1
                        result.discrepancies.append(
                            f"Local position not in IBKR (fresh-confirmed flat): "
                            f"{_t.trade_id}")
                        await self._book_stale_trade(_t, ibkr_portfolio)
                        booked += 1
                log.critical(
                    "reconcile_zero_options_confirmed_fresh",
                    open_local_trades=len(local_entries), booked=booked,
                    note="fresh broker re-query CONFIRMS zero option positions "
                         "— booked the genuinely-flat FILLED/PARTIAL trades",
                )
            else:
                log.warning(
                    "reconcile_skipping_stale_close",
                    open_local_trades=len(local_entries),
                    fresh_available=fresh is not None,
                    reason="IBKR shows no option positions while local option "
                           "trades are open, and a fresh re-query did not "
                           "confirm the book is flat; refusing to mass-close",
                )
            # In every branch the bulk stale-local loop stays disabled on a
            # zero-options snapshot: booking (if any) happened above; PENDING
            # orphans are handled by the stale-pending sweep at the end.
            local_entries = []

        # A multi-leg trade is stale only when NONE of its legs exist in IBKR;
        # any surviving leg means the position (or part of it) is still live.
        for trade, keys in local_entries:
            # R15 #2: PENDING rows belong to the liveness-gated, 30-min-aged
            # stale-pending sweep — NEVER to this booking loop. A crash-restart
            # one minute after placing an entry used to book the fresh PENDING
            # as reconciler_unknown_exit while the entry order still WORKED at
            # the broker: the fill then landed untracked (the exact class the
            # untracked-option freeze exists to catch).
            if trade.status == TradeStatus.PENDING:
                continue
            if not keys.isdisjoint(ibkr_map):
                # Some legs alive. If OTHERS are missing, this is a partial
                # assignment/exercise — flag it loudly; the position needs
                # human attention (we never auto-close half a structure).
                missing = [k for k in keys if k not in ibkr_map]
                if missing:
                    msg = (
                        f"PARTIAL legs missing for {trade.trade_id} "
                        f"({trade.strategy}): {missing} — possible assignment"
                    )
                    result.discrepancies.append(msg)
                    log.warning(
                        "reconcile_partial_legs_missing",
                        trade_id=trade.trade_id,
                        strategy=trade.strategy,
                        missing_legs=missing,
                    )
                continue
            result.stale_local += 1
            msg = f"Local position not in IBKR (likely closed): {trade.trade_id}"
            result.discrepancies.append(msg)
            log.warning("reconcile_stale_local", trade_id=trade.trade_id,
                        symbol=trade.symbol, strategy=trade.strategy)

            await self._book_stale_trade(trade, ibkr_portfolio)

        # Sweep long-dead PENDING orphans LAST — after promotion had its
        # chance to rescue filled-but-tracker-lost orders (audit R2/C3).
        # R19: via the async wrapper so an empty position cache is resolved
        # against a FRESH broker query instead of being believed.
        await self.sweep_stale_pending()

        # Update portfolio value in IBKR for reconciliation
        for item in ibkr_portfolio:
            log.debug(
                "ibkr_portfolio_item",
                symbol=item.contract.symbol,
                position=item.position,
                market_value=item.marketValue,
                unrealized_pnl=item.unrealizedPNL,
            )

        log.info(
            "reconciliation_complete",
            matched=result.matched,
            new_from_ibkr=result.new_from_ibkr,
            stale_local=result.stale_local,
            discrepancies=len(result.discrepancies),
        )

        return result
