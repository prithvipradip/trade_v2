"""Event-driven long straddle strategy.

Buys ATM long straddle 1 day before scheduled macro events (FOMC,
CPI, NFP, GDP, PCE). Profits from:
  1. IV expansion pre-event (vega gain even before announcement)
  2. Directional move on announcement (delta gain in either direction)
  3. Exits same-day or day-after to avoid IV crush erosion

Why this exists:
  Our short-premium strategies (iron condors, strangles) are wrong-side
  of macro events. The bot used to flatten before events to avoid
  losses. This strategy profits from the SAME volatility expansion
  the short-vol bot has to dodge.

Risk: capped at the debit paid (max loss = premium).
"""

from __future__ import annotations

from datetime import date

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection
from ait.utils.logging import get_logger

log = get_logger("strategies.event_straddle")


class EventStraddle:
    """Long ATM straddle entered 0-1 days before a macro event."""

    name = "event_straddle"
    direction_bias = None  # Volatility play, direction-neutral

    # Symbols to consider — most liquid, tightest spreads
    PREFERRED_SYMBOLS = {"SPY", "QQQ", "IWM"}

    def generate_signals(
        self,
        symbol: str,
        chains: list[OptionsChain],
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        economic_cal=None,
    ) -> list[Signal]:
        """Generate event-straddle signals.

        Only fires when:
          - Symbol is liquid (SPY/QQQ/IWM)
          - A scheduled macro event is 0-1 days away
          - IV rank < 70 (not already overpriced)
          - Multiple expiries available
        """
        if symbol not in self.PREFERRED_SYMBOLS:
            return []
        if economic_cal is None:
            return []

        try:
            days_to_event = economic_cal.days_until_next_event()
        except Exception as e:
            log.debug("event_straddle_no_calendar", symbol=symbol, error=str(e))
            return []
        if days_to_event is None or days_to_event > 1 or days_to_event < 0:
            return []  # not in the event window — silent by design (every scan)

        # Past the event-window gate: from here, a bail is worth logging so a
        # misfire on event day is never invisible again (it was — the strategy
        # had never once fired because of a silent chain-DTE mismatch).
        log.info("event_straddle_window_open",
                 symbol=symbol, days_to_event=days_to_event, iv_rank=iv_rank)

        # Skip if IV already inflated — the move is priced in
        if iv_rank > 70:
            log.info("event_straddle_skip_high_iv", symbol=symbol, iv_rank=iv_rank)
            return []

        if not chains:
            log.info("event_straddle_skip_no_chains", symbol=symbol)
            return []

        # Pick the NEAREST available expiry. The orchestrator fetches chains in
        # the configured dte_range (e.g. 14-45), so demanding dte <= 14 here
        # found nothing and bailed silently — the strategy never fired. Take
        # the soonest expiry available; an event straddle on the front-month
        # still captures the event-day IV expansion and move.
        today = date.today()
        candidates = sorted(
            [c for c in chains if c.expiry is not None and (c.expiry - today).days > 0],
            key=lambda c: c.expiry,
        )
        chain = candidates[0] if candidates else None
        if chain is None:
            log.info("event_straddle_skip_no_expiry", symbol=symbol,
                     chains_seen=len(chains))
            return []

        price = chain.underlying_price
        if price <= 0:
            log.info("event_straddle_skip_no_price", symbol=symbol)
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        liquid_puts = self._filter_liquid(chain.puts)
        if not liquid_calls or not liquid_puts:
            log.info("event_straddle_skip_illiquid", symbol=symbol,
                     dte=(chain.expiry - date.today()).days,
                     n_calls=len(chain.calls), n_puts=len(chain.puts),
                     liquid_calls=len(liquid_calls), liquid_puts=len(liquid_puts))
            return []

        # ATM strike — closest to current price
        common_strikes = (
            set(c.strike for c in liquid_calls)
            & set(p.strike for p in liquid_puts)
        )
        if not common_strikes:
            log.info("event_straddle_skip_no_common_strike", symbol=symbol)
            return []
        atm_strike = min(common_strikes, key=lambda s: abs(s - price))

        atm_call = next((c for c in liquid_calls if c.strike == atm_strike), None)
        atm_put = next((p for p in liquid_puts if p.strike == atm_strike), None)
        if not atm_call or not atm_put:
            return []

        # Net debit = call + put premiums
        debit = atm_call.mid + atm_put.mid
        if debit <= 0:
            return []

        # Breakeven move = debit / price; needs ~1% move at typical setup
        breakeven_pct = debit / price

        max_loss = debit * 100
        # Conservative profit estimate: 50% of debit if directional move
        max_profit_est = debit * 100 * 0.5

        legs = [
            {"strike": atm_strike, "right": "C", "action": "BUY",
             "expiry": atm_call.expiry.isoformat()},
            {"strike": atm_strike, "right": "P", "action": "BUY",
             "expiry": atm_put.expiry.isoformat()},
        ]

        log.info("event_straddle_setup",
                 symbol=symbol, days_to_event=days_to_event,
                 strike=atm_strike, debit=debit,
                 breakeven_move_pct=f"{breakeven_pct:.2%}")

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.65,  # baseline confidence; overridden by vol_mag if available
                contract=atm_call,
                action="BUY",
                quantity=1,
                legs=legs,
                entry_price=debit,
                max_loss=max_loss,
                max_profit=max_profit_est,
                stop_loss=round(debit * 0.7, 2),  # 30% stop (event plays move fast)
                take_profit=round(debit * 1.5, 2),  # 50% target
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=atm_call.expiry,
            )
        ]

    @staticmethod
    def _filter_liquid(contracts: list[OptionContract]) -> list[OptionContract]:
        out = []
        for c in contracts:
            if not c.bid or not c.ask:
                continue
            if c.bid <= 0 or c.ask <= 0:
                continue
            if (c.ask - c.bid) / max(c.mid, 0.01) > 0.20:
                continue
            out.append(c)
        return out
