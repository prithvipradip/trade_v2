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
from dataclasses import dataclass
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

    def _sweep_stale_pending(self) -> int:
        """Close PENDING trades too old to still be live working orders.

        Age-based, so RECENT pendings (awaiting a legitimate fill) are left
        alone and still dedup correctly; only long-dead orphans are cleared.
        Runs before the IBKR-empty mass-close guard because an unfilled order
        opened no position — closing it as never-filled risks nothing.
        """
        closed = 0
        try:
            now = datetime.now()
            for t in self._state.get_open_trades():
                if t.status != TradeStatus.PENDING:
                    continue
                try:
                    age_min = (now - datetime.fromisoformat(t.entry_time)).total_seconds() / 60
                except (ValueError, TypeError):
                    continue
                if age_min >= self.STALE_PENDING_MINUTES:
                    self._state.close_trade(
                        trade_id=t.trade_id, exit_price=0.0, realized_pnl=0.0,
                        exit_reason_detailed="stale_pending_never_filled",
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

        # Clear long-dead PENDING orphans first (see _sweep_stale_pending).
        self._sweep_stale_pending()

        # Get IBKR positions
        ibkr_positions = self._ibkr.get_positions()
        ibkr_portfolio = self._ibkr.get_portfolio()

        # Build IBKR position map using normalized keys
        ibkr_map: dict[str, dict] = {}
        for pos in ibkr_positions:
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
            keys: set[str] = set()
            try:
                legs = json.loads(trade.legs) if trade.legs else []
            except (ValueError, TypeError):
                legs = []
            if legs:
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
            if not keys and trade.strike:
                right = "C" if "call" in trade.strategy else "P"
                keys.add(self._make_position_key(
                    symbol=trade.symbol,
                    strike=trade.strike,
                    right=right,
                    expiry=trade.expiry or "",
                ))
            if not keys:
                # Stock positions
                keys.add(f"{trade.symbol}:STK")
            local_entries.append((trade, keys))
            for key in keys:
                local_map.setdefault(key, []).append(trade)

        result = ReconciliationResult(
            matched=0,
            new_from_ibkr=0,
            stale_local=0,
            discrepancies=[],
        )

        # Check IBKR positions against local state
        promoted_ids: set[str] = set()
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
                    if (local_trade.status in (TradeStatus.PENDING, TradeStatus.CLOSING)
                            and local_trade.trade_id not in promoted_ids):
                        self._state.update_trade_status(local_trade.trade_id, TradeStatus.FILLED)
                        promoted_ids.add(local_trade.trade_id)
                        result.promoted += 1
                        log.warning("reconcile_promoted_to_filled",
                                    trade_id=local_trade.trade_id, symbol=local_trade.symbol,
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
                msg = (
                    f"New position from IBKR: {key}, "
                    f"qty={ibkr_pos['quantity']}, avg_cost={ibkr_pos['avg_cost']}"
                )
                result.discrepancies.append(msg)
                log.warning("reconcile_new_ibkr_position", position=key)

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
            log.warning(
                "reconcile_skipping_stale_close",
                open_local_trades=len(local_entries),
                reason="IBKR shows no option positions while local option "
                       "trades are open; refusing to mass-close",
            )
            local_entries = []

        # A multi-leg trade is stale only when NONE of its legs exist in IBKR;
        # any surviving leg means the position (or part of it) is still live.
        for trade, keys in local_entries:
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

            # Try to get realized P&L from IBKR portfolio items
            exit_price = 0.0
            realized_pnl = 0.0
            exit_reason = "reconciler_ibkr_realized"
            found_ibkr_pnl = False
            for item in ibkr_portfolio:
                sym = item.contract.symbol
                if sym == trade.symbol and item.position == 0 and item.realizedPNL != 0:
                    realized_pnl = item.realizedPNL
                    found_ibkr_pnl = True
                    break

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
                    if trade.strategy in CREDIT_STRATEGIES:
                        # Credit expired worthless = full premium kept
                        realized_pnl = trade.entry_price * multiplier * trade.quantity
                    else:
                        # Debit expired worthless = full loss
                        realized_pnl = -trade.entry_price * multiplier * trade.quantity
                    exit_reason = "reconciler_estimate_expired_worthless"
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
