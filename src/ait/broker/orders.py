"""Order builders for all order types.

Creates IBKR-compatible orders with proper formatting for options trades.
Supports market, limit, bracket (with stop-loss/take-profit), and combo orders.
"""

from __future__ import annotations

from ib_insync import (
    LimitOrder,
    MarketOrder,
    Order,
    StopOrder,
    BracketOrder,
    TagValue,
)

from ait.utils.logging import get_logger

log = get_logger("broker.orders")


class OrderBuilder:
    """Build IBKR-compatible orders for options trading."""

    @staticmethod
    def market(action: str, quantity: int) -> MarketOrder:
        """Simple market order.

        Use sparingly for options — prefer limit orders to avoid slippage.
        """
        return MarketOrder(action.upper(), quantity)

    @staticmethod
    def limit(action: str, quantity: int, limit_price: float) -> LimitOrder:
        """Limit order — preferred for options to control entry price.

        Args:
            action: "BUY" or "SELL"
            quantity: Number of contracts
            limit_price: Maximum (buy) or minimum (sell) price per contract
        """
        order = LimitOrder(action.upper(), quantity, limit_price)
        order.tif = "DAY"  # Good for the day only
        return order

    @staticmethod
    def limit_gtc(action: str, quantity: int, limit_price: float) -> LimitOrder:
        """Good-till-cancelled limit order."""
        order = LimitOrder(action.upper(), quantity, limit_price)
        order.tif = "GTC"
        return order

    @staticmethod
    def stop(action: str, quantity: int, stop_price: float) -> StopOrder:
        """Stop order — triggers market order when price crosses stop level."""
        return StopOrder(action.upper(), quantity, stop_price)

    @staticmethod
    def bracket(
        action: str,
        quantity: int,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> BracketOrder:
        """Bracket order — entry + take profit + stop loss.

        This creates 3 linked orders:
        1. Entry: limit order at limit_price
        2. Take profit: limit order at take_profit_price (closes position)
        3. Stop loss: stop order at stop_loss_price (closes position)

        When either the take profit or stop loss fills, the other is cancelled.
        """
        bracket = BracketOrder(
            action.upper(),
            quantity,
            limit_price,
            take_profit_price,
            stop_loss_price,
        )
        return bracket

    @staticmethod
    def combo_limit(
        action: str,
        quantity: int,
        limit_price: float,
    ) -> LimitOrder:
        """Limit order for combo/spread contracts.

        For spreads, the limit_price is the NET debit (positive) or credit (negative).
        Example: Bull call spread at $2.50 net debit → limit_price = 2.50
        Example: Iron condor at $1.00 net credit → limit_price = -1.00
        """
        order = LimitOrder(action.upper(), quantity, limit_price)
        order.tif = "DAY"
        return order

    @staticmethod
    def adaptive_market(action: str, quantity: int) -> Order:
        """Adaptive market order — IBKR's smart routing for better fills.

        Uses IBKR's adaptive algorithm to seek price improvement
        before resorting to market order.
        """
        order = Order()
        order.action = action.upper()
        order.totalQuantity = quantity
        order.orderType = "MKT"
        order.algoStrategy = "Adaptive"
        order.algoParams = [
            TagValue("adaptivePriority", "Normal"),
        ]
        order.tif = "DAY"
        return order

    @staticmethod
    def passive_limit(
        action: str,
        quantity: int,
        bid: float,
        ask: float,
        improvement_pct: float = 0.20,
    ) -> LimitOrder:
        """Passive limit order with price improvement.

        Places the order between the mid-price and the natural side,
        giving us price improvement while still being more likely to fill
        than sitting at the bid/ask.

        For BUY: price = mid - (improvement_pct * half-spread)
                 (below mid, toward bid — passive buyer)
        For SELL: price = mid + (improvement_pct * half-spread)
                 (above mid, toward ask — passive seller)

        Args:
            action: "BUY" or "SELL"
            quantity: Number of contracts
            bid: Current best bid
            ask: Current best ask
            improvement_pct: 0.0 = at mid, 1.0 = at natural side (bid/ask)
        """
        mid = (bid + ask) / 2
        half_spread = (ask - bid) / 2

        if action.upper() == "BUY":
            # Passive: below mid, toward bid
            price = mid - half_spread * improvement_pct
        else:
            # Passive: above mid, toward ask
            price = mid + half_spread * improvement_pct

        # Round to nearest penny, ensure within spread
        price = round(price, 2)
        price = max(bid, min(ask, price))

        order = LimitOrder(action.upper(), quantity, price)
        order.tif = "DAY"
        return order

    @staticmethod
    def calculate_spread_limit(
        mid_price: float,
        action: str,
        aggression: float = 0.0,
    ) -> float:
        """Calculate limit price for a spread based on mid-price.

        Args:
            mid_price: Mid-point between bid and ask of the spread
            action: "BUY" (debit) or "SELL" (credit)
            aggression: 0.0 = at mid, 1.0 = at ask/bid (aggressive fill)

        Returns:
            Adjusted limit price
        """
        # For a BUY, being more aggressive means paying more (toward ask)
        # For a SELL, being more aggressive means accepting less (toward bid)
        # This is a simple model — real aggression depends on bid-ask width
        adjustment = mid_price * 0.02 * aggression  # Up to 2% adjustment
        if action.upper() == "BUY":
            return round(mid_price + adjustment, 2)
        else:
            return round(mid_price - adjustment, 2)
