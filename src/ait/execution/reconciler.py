"""Position reconciliation — sync local state with IBKR on restart.

When the bot restarts (crash, manual stop, etc.), this module:
1. Loads local state from SQLite
2. Fetches live positions from IBKR
3. Reconciles differences and flags discrepancies
4. Updates local state to match reality
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from ait.broker.ibkr_client import IBKRClient
from ait.bot.state import StateManager, TradeStatus
from ait.utils.logging import get_logger

log = get_logger("execution.reconciler")


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    matched: int  # Positions that match between local and IBKR
    new_from_ibkr: int  # Positions in IBKR not in local state
    stale_local: int  # Local positions not found in IBKR
    discrepancies: list[str]  # Human-readable descriptions


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

    async def reconcile(self) -> ReconciliationResult:
        """Perform full reconciliation between local state and IBKR.

        This should be called on every bot startup.
        """
        log.info("reconciliation_starting")

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

        # Get local open trades and build map with matching normalized keys
        local_trades = self._state.get_open_trades()
        local_map: dict[str, dict] = {}
        for trade in local_trades:
            if trade.strike:
                right = "C" if "call" in trade.strategy else "P"
                key = self._make_position_key(
                    symbol=trade.symbol,
                    strike=trade.strike,
                    right=right,
                    expiry=trade.expiry or "",
                )
            else:
                # Stock positions
                key = f"{trade.symbol}:STK"
            local_map[key] = {"trade": trade}

        result = ReconciliationResult(
            matched=0,
            new_from_ibkr=0,
            stale_local=0,
            discrepancies=[],
        )

        # Check IBKR positions against local state
        for key, ibkr_pos in ibkr_map.items():
            if key in local_map:
                result.matched += 1
                # Check quantity matches
                local_trade = local_map[key]["trade"]
                if abs(ibkr_pos["quantity"]) != abs(local_trade.quantity):
                    msg = (
                        f"Quantity mismatch for {key}: "
                        f"IBKR={ibkr_pos['quantity']}, local={local_trade.quantity}"
                    )
                    result.discrepancies.append(msg)
                    log.warning("reconcile_quantity_mismatch", position=key,
                                ibkr_qty=ibkr_pos["quantity"], local_qty=local_trade.quantity)
            else:
                result.new_from_ibkr += 1
                msg = (
                    f"New position from IBKR: {key}, "
                    f"qty={ibkr_pos['quantity']}, avg_cost={ibkr_pos['avg_cost']}"
                )
                result.discrepancies.append(msg)
                log.warning("reconcile_new_ibkr_position", position=key)

        # Check local positions not in IBKR (may have been closed while bot was down)
        for key, local_data in local_map.items():
            if key not in ibkr_map:
                result.stale_local += 1
                trade = local_data["trade"]
                msg = f"Local position not in IBKR (likely closed): {key}"
                result.discrepancies.append(msg)
                log.warning("reconcile_stale_local", position=key, trade_id=trade.trade_id)

                # Mark as closed in local state
                self._state.close_trade(
                    trade_id=trade.trade_id,
                    exit_price=0,  # Unknown exit price
                    realized_pnl=0,  # Unknown P&L
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
