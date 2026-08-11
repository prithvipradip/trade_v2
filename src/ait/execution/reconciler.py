"""Position reconciliation — sync local state with IBKR on restart.

When the bot restarts (crash, manual stop, etc.), this module:
1. Loads local state from SQLite
2. Fetches live positions from IBKR
3. Reconciles differences and flags discrepancies
4. Updates local state to match reality
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from ait.broker.ibkr_client import IBKRClient
from ait.bot.state import StateManager, TradeStatus
from ait.strategies.base import CREDIT_STRATEGIES
from ait.utils.logging import get_logger

log = get_logger("execution.reconciler")


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

          "live"    — every leg is present; a reverse combo closes exactly it.
          "partial" — SOME legs present, some gone (manual half-flatten,
                      assignment). Reversing all legs would SELL the already-
                      flat ones and OPEN new inverted positions. Needs a human.
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
        """
        keys = self._trade_leg_keys(trade)
        if not keys or keys == {f"{trade.symbol}:STK"}:
            return "unknown"  # shape we can't key — don't block the exit on it

        live = self._live_ibkr_leg_keys()
        if live is None:
            return "unknown"

        present = keys & live
        if present == keys:
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

        fresh_keys: set[str] = set()
        for pos in fresh:
            if not getattr(pos, "position", 0):
                continue  # R15b: zeroed row = closed, not live
            try:
                if pos.contract.secType == "OPT":
                    fresh_keys.add(self._make_position_key(
                        symbol=pos.contract.symbol,
                        strike=pos.contract.strike,
                        right=pos.contract.right,
                        expiry=pos.contract.lastTradeDateOrContractMonth,
                    ))
            except (AttributeError, TypeError, ValueError):
                continue

        present = keys & fresh_keys
        if present == keys:
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
        for item in ibkr_portfolio:
            if item.contract.secType != "OPT" or item.position != 0:
                continue
            key = self._make_position_key(
                symbol=item.contract.symbol,
                strike=item.contract.strike,
                right=item.contract.right,
                expiry=item.contract.lastTradeDateOrContractMonth,
            )
            if key in trade_keys and item.realizedPNL:
                leg_pnls.append(item.realizedPNL)
        if leg_pnls:
            realized_pnl = float(sum(leg_pnls))
            found_ibkr_pnl = True

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
                    exit_reason = "reconciler_unknown_exit_expired_needs_review"
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
                exit_reason = "reconciler_unknown_exit"
                log.warning(
                    "reconcile_unknown_exit",
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    strategy=trade.strategy,
                    note="position gone from IBKR before expiry; true exit price unrecoverable",
                )

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

    def _live_ibkr_leg_keys(self) -> set[str] | None:
        """Position keys currently live at IBKR, or None if unavailable."""
        try:
            if not self._ibkr.connected:
                return None
            live: set[str] = set()
            for pos in self._ibkr.get_positions():
                if not getattr(pos, "position", 0):
                    continue  # R15b: zeroed row = closed, not live
                if pos.contract.secType == "OPT":
                    live.add(self._make_position_key(
                        symbol=pos.contract.symbol,
                        strike=pos.contract.strike,
                        right=pos.contract.right,
                        expiry=pos.contract.lastTradeDateOrContractMonth,
                    ))
                else:
                    live.add(f"{pos.contract.symbol}:STK")
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

    async def _fetch_working_orders(self) -> list[tuple[int, set[str], str]] | None:
        """R12 F4.1: working orders at the broker as (orderId, keys, symbol).

        Keys are leg position-keys for plain option orders; combo (BAG)
        orders only expose leg conIds, so they match at symbol level via the
        "SYMBOL:BAG" marker. Uses reqAllOpenOrders (all clientIds) so exit
        orders placed by a previous session/clientId are seen. Returns None
        when the broker view is unavailable (callers must then NOT promote
        CLOSING trades — unknown is not "no working order").
        """
        try:
            if hasattr(self._ibkr, "get_all_open_trades"):
                trades = await self._ibkr.get_all_open_trades()
            else:  # pragma: no cover — legacy client without the helper
                trades = self._ibkr.get_open_orders()
            if trades is None:
                return None
            entries: list[tuple[int, set[str], str]] = []
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
                entries.append((int(t.order.orderId or 0), keys, str(c.symbol)))
            return entries
        except Exception as e:  # noqa: BLE001
            log.warning("working_orders_fetch_failed", error=str(e))
            return None

    def _working_exit_order_for(
        self, trade, working: list[tuple[int, set[str], str]]
    ) -> int | None:
        """R12 F4.1: orderId of a working order touching this trade's legs
        (leg-key match for plain options; symbol match for BAG combos —
        combo legs are opaque conIds at this layer, and keeping CLOSING on a
        same-symbol combo is the safe direction)."""
        trade_keys = self._trade_leg_keys(trade)
        for order_id, keys, symbol in working:
            if not keys.isdisjoint(trade_keys):
                return order_id
            if f"{symbol}:BAG" in keys and symbol == trade.symbol:
                return order_id
        return None

    def _cancel_working_entry_order(self, trade) -> bool:
        """R16: cancel a still-WORKING broker order for a stale PENDING trade.

        Returns True when a matching open order was found (cancel requested —
        the caller must NOT book "never filled" this cycle; the next sweep
        sees either the cancel or a late fill). Uses the in-session
        openTrades cache: same-clientId orders only, which is exactly the
        population the sweep can own.
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
            for bt in open_trades:
                # Belt-and-braces: only genuinely ACTIVE orders count as
                # working (some fakes/caches keep terminal trades in the view).
                _st = getattr(getattr(bt, "orderStatus", None), "status", "")
                if _st in _TERMINAL:
                    continue
                c = bt.contract
                sym = getattr(c, "symbol", "")
                if sym != trade.symbol:
                    continue
                sec = getattr(c, "secType", "")
                if sec == "BAG" or (sec == "OPT" and keys and any(
                        str(getattr(c, "strike", "")) in k for k in keys)):
                    ib.cancelOrder(bt.order)
                    return True
        except Exception as e:  # noqa: BLE001 — sweep must survive broker hiccups
            log.warning("sweep_cancel_check_failed",
                        trade_id=trade.trade_id, error=str(e))
            return True  # unknown broker state: defer booking, never fiction
        return False

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
        """
        closed = 0
        try:
            now = datetime.now()
            stale = []
            for t in self._state.get_open_trades():
                if t.status != TradeStatus.PENDING:
                    continue
                try:
                    age_min = (now - datetime.fromisoformat(t.entry_time)).total_seconds() / 60
                except (ValueError, TypeError):
                    continue
                if age_min >= self.STALE_PENDING_MINUTES:
                    stale.append((t, age_min))
            if not stale:
                return 0
            live_keys = self._live_ibkr_leg_keys()
            if live_keys is None:
                log.warning("sweep_skipped_no_ibkr_positions")
                return 0
            for t, age_min in stale:
                if not self._trade_leg_keys(t).isdisjoint(live_keys):
                    # Legs live at the broker — the order FILLED. Rescue it.
                    # R12 Tier-A #1: CAS PENDING->FILLED (a concurrent writer
                    # may have moved the trade since get_open_trades()).
                    if self._state.transition(
                            t.trade_id, (TradeStatus.PENDING,), TradeStatus.FILLED):
                        # R16: rescued trades had no open_positions row, so
                        # HWM/trailing-lock/partial-exit tracking was dead.
                        self._ensure_open_position_row(t)
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
                self._state.close_trade(
                    trade_id=t.trade_id, exit_price=0.0, realized_pnl=0.0,
                    exit_reason_detailed="stale_pending_never_filled",
                    from_statuses=(TradeStatus.PENDING,),
                )
                closed += 1
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
        working_orders: list[tuple[int, set[str], str]] | None = None
        if any(t.status == TradeStatus.CLOSING for t in local_trades):
            working_orders = await self._fetch_working_orders()

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
                for local_trade in matching_trades:
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
                    # INST-3: a live OPTION at the broker with no local record
                    # = the mid-placement crash window. It has no stop, no TP,
                    # no expiry handling. Freeze new entries until a human
                    # resolves it (delete data/HALT_UNTRACKED to resume).
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
                    from datetime import datetime as _dtd
                    if not hasattr(self, "_untracked_sightings"):
                        self._untracked_sightings: dict[str, _dtd] = {}
                    _now = _dtd.now()
                    _prev = self._untracked_sightings.get(key)
                    self._untracked_sightings[key] = _now
                    _gap_min = (_now - _prev).total_seconds() / 60 if _prev else None
                    if _gap_min is None or _gap_min < 2 or _gap_min > 480:
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
                    try:
                        from pathlib import Path as _P
                        _P("data/HALT_UNTRACKED").write_text(
                            f"untracked {key} qty={ibkr_pos['quantity']} — "
                            f"resolve at IBKR then delete this file")
                    except Exception:  # noqa: BLE001
                        pass
                    msg = (
                        f"UNTRACKED OPTION POSITION (no local record): {key}, "
                        f"qty={ibkr_pos['quantity']} — NEW ENTRIES FROZEN "
                        f"(data/HALT_UNTRACKED). Resolve manually, then delete the file."
                    )
                    log.critical("reconcile_untracked_option_freeze",
                                 position=key, qty=ibkr_pos["quantity"])
                elif ibkr_pos.get("sec_type") == "STK":
                    msg = (
                        f"UNTRACKED STOCK POSITION (possible ASSIGNMENT): {key}, "
                        f"qty={ibkr_pos['quantity']}, avg_cost={ibkr_pos['avg_cost']} "
                        f"— needs manual liquidation/review"
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
        self._sweep_stale_pending()

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
