"""Contract builders for stocks and options.

Creates properly formatted IBKR contracts for qualification and trading.
Handles the complexity of options contract specification (expiry, strike, right).
"""

from __future__ import annotations

from datetime import date, datetime

from ib_insync import Contract, Option, Stock, ComboLeg, Bag

from ait.utils.logging import get_logger

log = get_logger("broker.contracts")


class ContractBuilder:
    """Build IBKR-compatible contracts for stocks and options."""

    @staticmethod
    def stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
        """Create a stock contract."""
        return Stock(symbol, exchange, currency)

    @staticmethod
    def option(
        symbol: str,
        expiry: date | str,
        strike: float,
        right: str,  # "C" or "P"
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Option:
        """Create an option contract.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            expiry: Expiration date (date object or "YYYYMMDD" string)
            strike: Strike price
            right: "C" for call, "P" for put
            exchange: Exchange (default SMART for best routing)
            currency: Currency (default USD)
        """
        if isinstance(expiry, date):
            expiry_str = expiry.strftime("%Y%m%d")
        else:
            # Normalize string format: "2026-04-17" → "20260417"
            expiry_str = expiry.replace("-", "")

        # Validate right
        right = right.upper()
        if right not in ("C", "P"):
            raise ValueError(f"Option right must be 'C' or 'P', got '{right}'")

        return Option(symbol, expiry_str, strike, right, exchange, currency=currency)

    @staticmethod
    def combo(
        symbol: str,
        legs: list[dict],
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Bag:
        """Create a combo/spread contract for multi-leg orders.

        Each leg dict should have:
            - conId: int (contract ID from qualified option)
            - action: "BUY" or "SELL"
            - ratio: int (usually 1)

        This is used for spreads, iron condors, etc. to ensure
        atomic execution (all legs fill or none do).
        """
        bag = Bag()
        bag.symbol = symbol
        bag.exchange = exchange
        bag.currency = currency
        bag.comboLegs = []

        for leg in legs:
            combo_leg = ComboLeg()
            combo_leg.conId = leg["conId"]
            combo_leg.ratio = leg.get("ratio", 1)
            combo_leg.action = leg["action"]
            combo_leg.exchange = exchange
            bag.comboLegs.append(combo_leg)

        return bag

    @staticmethod
    def vertical_spread(
        symbol: str,
        expiry: date | str,
        long_strike: float,
        short_strike: float,
        right: str,
        long_con_id: int,
        short_con_id: int,
        exchange: str = "SMART",
    ) -> Bag:
        """Create a vertical spread (bull call or bear put).

        For a bull call spread: buy lower strike call, sell higher strike call.
        For a bear put spread: buy higher strike put, sell lower strike put.
        """
        return ContractBuilder.combo(
            symbol=symbol,
            legs=[
                {"conId": long_con_id, "action": "BUY", "ratio": 1},
                {"conId": short_con_id, "action": "SELL", "ratio": 1},
            ],
            exchange=exchange,
        )

    @staticmethod
    def iron_condor(
        symbol: str,
        put_long_con_id: int,
        put_short_con_id: int,
        call_short_con_id: int,
        call_long_con_id: int,
        exchange: str = "SMART",
    ) -> Bag:
        """Create an iron condor (4-leg combo).

        Legs (from lowest to highest strike):
        1. BUY put (lowest strike) — protection
        2. SELL put (lower-mid strike) — collect premium
        3. SELL call (upper-mid strike) — collect premium
        4. BUY call (highest strike) — protection
        """
        return ContractBuilder.combo(
            symbol=symbol,
            legs=[
                {"conId": put_long_con_id, "action": "BUY", "ratio": 1},
                {"conId": put_short_con_id, "action": "SELL", "ratio": 1},
                {"conId": call_short_con_id, "action": "SELL", "ratio": 1},
                {"conId": call_long_con_id, "action": "BUY", "ratio": 1},
            ],
            exchange=exchange,
        )

    @staticmethod
    def straddle_legs(
        symbol: str,
        expiry: date | str,
        strike: float,
    ) -> tuple[Option, Option]:
        """Create the two legs of a straddle (call + put at same strike).

        Returns individual contracts — for a straddle, you place two
        separate orders or use combo().
        """
        call = ContractBuilder.option(symbol, expiry, strike, "C")
        put = ContractBuilder.option(symbol, expiry, strike, "P")
        return call, put
