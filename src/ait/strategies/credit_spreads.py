"""Vertical credit spreads: the put credit spread (bullish) and the call credit spread (bearish).

W108 4b: the live builders of two structures that existed only in the research engine
(``backtesting/engine.py`` ``_build_credit_position``, "put_credit_spread" / "call_credit_spread").
Registered in the selector but NOT enabled: ``options.strategies`` must name them, and that is a
recorded operator decision taken after a pre-registered test (PLAN.md W108 NEXT PHASES, 4d).

Structure = research's: sell the strike nearest the condor's short delta, buy the wing
``max(wing floor, wing_k x expected move x backtest.spread_tuning.wing_expected_move_fraction)``
further out. Economics and budget = the live condor's: the wing is budget-fitted through
``IronCondor.affordable_width`` and the credit must clear the condor's two floors, which the executor
enforces on every credit entry anyway (``TradeExecutor._credit_floor_minimum``).

Put credit spread  = sell OTM put, buy a lower put   (net credit; profits if price stays above the short)
Call credit spread = sell OTM call, buy a higher call (net credit; profits if price stays below the short)
"""

from __future__ import annotations

import pandas as pd

from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING
from ait.config.runtime_env import contract_float
from ait.data.options_chain import OptionContract, OptionsChain
from ait.execution.exit_policy import DEFAULT_TAKE_PROFIT
from ait.risk.width_sizing import fitted_credit_width
from ait.strategies.base import Signal, SignalDirection, Strategy
from ait.strategies.iron_condor import SHORT_DELTA_TARGET, IronCondor
from ait.utils.logging import get_logger

log = get_logger("strategies.credit_spreads")


def wing_expected_move_fraction() -> float:
    """``backtest.spread_tuning.wing_expected_move_fraction`` from the LOADED settings.

    W108 4b: the research engine's credit-spread wing is ``wing_k x expected move x`` this number
    (``Backtester._spr``), so live reads the same key instead of re-typing 0.5. The block may not
    appear in ``config.yaml`` (research-only keys, ``test_w88b_tunables``), so the live bot runs at
    the schema default; a research process (frozen overlay settings) sees its overlay value, which is
    what a parity replay of this builder needs. Resolution: ``resolve_config_value`` (loaded
    settings, else the schema default with a logged warning).
    """
    from ait.config.settings import BacktestConfig, load_settings, resolve_config_value
    try:
        settings = load_settings()
    except Exception as e:  # noqa: BLE001 — no config here: the schema default, said out loud
        log.warning("credit_spread_settings_unavailable", error=str(e))
        settings = None
    tuning = resolve_config_value(None, "backtest", "spread_tuning", BacktestConfig, settings)
    return float(tuning.wing_expected_move_fraction)


class _VerticalCreditSpread(Strategy):
    """W108 4b: one builder, two sides. A subclass fixes the option right and the direction it serves."""

    #: "P" (put credit spread) or "C" (call credit spread)
    RIGHT: str = ""
    #: the ONLY market direction this structure is built on (4d: the direction model decides)
    DIRECTION: SignalDirection = SignalDirection.NEUTRAL

    #: W108 4b (as IronCondor W31): why the last generate_signals() emitted nothing, None once it
    #: emitted. Write-only in live; tests and a future sizing self-test read it.
    last_entry_reason: str | None = None

    def __init__(self, wing_fraction: float | None = None) -> None:
        # W108 4b: None = read backtest.spread_tuning.wing_expected_move_fraction on first use and
        # keep it (the live bot's settings are frozen at start, so one read is the whole truth).
        self._wing_fraction = wing_fraction

    @property
    def direction_bias(self) -> SignalDirection:
        return self.DIRECTION

    def wing_fraction(self) -> float:
        if self._wing_fraction is None:
            self._wing_fraction = wing_expected_move_fraction()
        return self._wing_fraction

    def natural_wing_width(self, expected_move: float) -> float:
        """The wing before any budget fit: ``max(wing floor, wing_k x EM x fraction)``, the floor and
        ``wing_k`` from the same contract keys the condor and the engine read
        (``AIT_IC_WING_FLOOR`` / ``AIT_IC_WING_K``)."""
        return max(contract_float("AIT_IC_WING_FLOOR"),
                   contract_float("AIT_IC_WING_K") * expected_move * self.wing_fraction())

    def wing_width(self, expected_move: float) -> float:
        """Target wing width (dollars per share) before the strike snap.

        Research's rule (``engine.py`` credit spreads) is the natural width above. Then the condor's
        budget fit: the widest wing the per-trade budget can carry is ``IronCondor.affordable_width``
        (the ONE authority for the budget <-> width algebra, W31), and a budget-capped wing may go below
        the floor, never below ``IC_BUDGET_CAPPED_MIN_WING``. The algebra holds for one vertical exactly
        as for the condor: max loss = (width - credit) x 100 with credit >= ratio floor x width.
        ``generate_signals`` picks the strike through ``fitted_credit_width`` (the rule research uses),
        which keeps a snapped wing at or below this cap when a listed strike allows it.
        """
        width = self.natural_wing_width(expected_move)
        if self.risk_budget and self.risk_budget > 0:
            affordable = IronCondor.affordable_width(self.risk_budget)
            if affordable < width:
                width = max(IC_BUDGET_CAPPED_MIN_WING, affordable)
        return width

    def _refuse(self, symbol: str, reason: str, chain: OptionsChain, **fields) -> list[Signal]:
        self._log_entry_quality(symbol, "rejected", reason, chain, **fields)
        return []

    def _log_entry_quality(self, symbol: str, status: str, reason: str, chain: OptionsChain,
                           short: OptionContract | None = None, credit: float | None = None,
                           width: float | None = None, wing_target: float | None = None) -> None:
        """One structured line per built / refused spread (the condor's ``condor_entry_quality``)."""
        self.last_entry_reason = reason if status == "rejected" else None
        em = float(getattr(chain, "expected_move", 0.0) or 0.0)
        spot = float(getattr(chain, "underlying_price", 0.0) or 0.0)
        log.info(
            "credit_spread_entry_quality",
            strategy=self.name, symbol=symbol, status=status, reason=reason,
            atm_iv=round(float(getattr(chain, "atm_iv", 0.0) or 0.0), 4),
            expected_move=round(em, 3),
            short_strike=(short.strike if short else None),
            short_delta=(round(short.delta, 3) if short else None),
            short_em_mult=(round(abs(spot - short.strike) / em, 3) if short and em > 0 else None),
            wing_target=(round(wing_target, 2) if wing_target is not None else None),
            credit=(round(credit, 2) if credit is not None else None),
            width=(round(width, 2) if width is not None else None),
            credit_width_ratio=(round(credit / width, 3) if credit is not None and width else None),
        )

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # W108 4b: directional, like the debit spreads (BullCallSpread / BearPutSpread): built only
        # when the direction model calls this side. Silent as theirs: it is the normal case.
        if market_direction != self.DIRECTION:
            self.last_entry_reason = "direction_mismatch"
            return []

        side = chain.puts if self.RIGHT == "P" else chain.calls
        liquid = self._filter_liquid(side)
        if len(liquid) < 2:
            return self._refuse(symbol, "insufficient_liquid_contracts", chain)

        # The condor's degraded-greeks gates (R12-B / W19): a chain with no usable IV carries guessed
        # deltas, and strike-by-delta on guessed deltas is the 0.49-delta incident.
        if getattr(chain, "iv_is_guessed", False) or not any(abs(c.delta) > 0 for c in liquid):
            return self._refuse(symbol, "degraded_greeks", chain)

        # Short strike: the condor's target delta (research: backtest.delta_short, the same 0.20),
        # within _find_strike_by_delta's tolerance and the condor's delta band (R12-B: enforce the
        # DESIGNED trade; a credit floor rewards a short that drifted toward the money).
        short = self._find_strike_by_delta(liquid, SHORT_DELTA_TARGET)
        if short is None:
            return self._refuse(symbol, "short_outside_delta_tolerance", chain)
        if not (contract_float("AIT_IC_DELTA_MIN") <= abs(short.delta)
                <= contract_float("AIT_IC_DELTA_MAX")):
            return self._refuse(symbol, "delta_band", chain, short=short)

        price = chain.underlying_price
        if not (short.strike < price if self.RIGHT == "P" else short.strike > price):
            return self._refuse(symbol, "short_not_out_of_the_money", chain, short=short)

        # Wing from the expected move (chain.expected_move = spot x ATM IV x sqrt(DTE/365), the
        # engine's formula). No expected move, no wing rule: refuse, as the condor's em_unavailable.
        em = float(getattr(chain, "expected_move", 0.0) or 0.0)
        if em <= 0:
            return self._refuse(symbol, "em_unavailable", chain, short=short)
        wing_target = self.wing_width(em)
        beyond = [c for c in liquid
                  if (c.strike < short.strike if self.RIGHT == "P" else c.strike > short.strike)]
        if not beyond:
            return self._refuse(symbol, "no_wing", chain, short=short, wing_target=wing_target)
        # The budget fit chooses among the LISTED widths exactly as research's does: the widest at or
        # below the budget cap (so the snap cannot land wider than the budget carries), the narrowest
        # when none fits (the max_loss re-check below then refuses it). Not capped: the strike nearest
        # the target (the condor's R6 idiom).
        listed = {abs(short.strike - c.strike) for c in beyond}
        # a budget-capped wing never goes below IC_BUDGET_CAPPED_MIN_WING ($1) while a listed strike allows it
        listed = {w for w in listed if w >= IC_BUDGET_CAPPED_MIN_WING - 1e-9} or listed
        wing_target = fitted_credit_width(
            self.natural_wing_width(em), self.risk_budget, listed,
            min_credit_ratio=contract_float("AIT_IC_MIN_CREDIT_WIDTH"))
        long_leg = min(beyond, key=lambda c: abs(abs(short.strike - c.strike) - wing_target))
        width = abs(short.strike - long_leg.strike)

        credit = short.mid - long_leg.mid
        if credit <= 0:
            return self._refuse(symbol, "non_positive_credit", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)
        # The condor's two economics floors, on the mid credit (research's condor applies them at the
        # same point). The executor re-applies exactly these at placement and at every reprice
        # (_credit_floor_minimum), so a looser floor here would only build refusals.
        if credit < contract_float("AIT_IC_MIN_CREDIT"):
            return self._refuse(symbol, "credit_floor", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)
        if credit / width < contract_float("AIT_IC_MIN_CREDIT_WIDTH"):
            return self._refuse(symbol, "credit_width_ratio", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)

        max_loss = (width - credit) * 100
        if max_loss <= 0:
            return self._refuse(symbol, "non_positive_max_loss", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)
        # The condor's R16 guard: the snap can land wider than the budget-fitted target, so the BUILT
        # structure is re-checked against the budget the target came from.
        if self.risk_budget and self.risk_budget > 0 and max_loss > self.risk_budget:
            return self._refuse(symbol, "max_loss_over_budget", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)

        self._log_entry_quality(symbol, "generated", "all_gates_passed", chain, short=short,
                                credit=credit, width=width, wing_target=wing_target)
        long_entry = {"contract": long_leg, "action": "BUY", "ratio": 1}
        short_entry = {"contract": short, "action": "SELL", "ratio": 1}
        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=self.DIRECTION,
                confidence=confidence,
                contract=short,  # primary reference leg (the condor's is its short put)
                action="SELL",
                quantity=1,
                legs=([long_entry, short_entry] if self.RIGHT == "P" else [short_entry, long_entry]),
                entry_price=credit,
                max_loss=max_loss,
                max_profit=credit * 100,
                # Informational only (exits are the monitor's: the touch stop on the short strike, the
                # shared take-profit ladder, the DTE close). The buy-back price at the flat short
                # target; no flat stop, since the live credit loss limit is AIT_CREDIT_LOSS_LIMIT (0 = off).
                take_profit=round(credit * (1.0 - DEFAULT_TAKE_PROFIT[1]), 2),
                stop_loss=None,
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=chain.expiry,
            )
        ]


class PutCreditSpread(_VerticalCreditSpread):
    """Put credit spread: sell an OTM put near the short delta, buy a lower put. Bullish, defined risk.

    Max loss = width - credit. Max profit = credit.
    """

    RIGHT = "P"
    DIRECTION = SignalDirection.BULLISH

    @property
    def name(self) -> str:
        return "put_credit_spread"


class CallCreditSpread(_VerticalCreditSpread):
    """Call credit spread: sell an OTM call near the short delta, buy a higher call. Bearish, defined risk.

    Max loss = width - credit. Max profit = credit.
    """

    RIGHT = "C"
    DIRECTION = SignalDirection.BEARISH

    @property
    def name(self) -> str:
        return "call_credit_spread"
