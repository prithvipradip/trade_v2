"""Iron condor strategy — premium selling in range-bound markets.

Sell OTM put spread + sell OTM call spread simultaneously.
Profits when underlying stays between the short strikes.

Best in: High IV (expensive premium), neutral/range-bound markets.
Risk: Defined — max loss = width of wider spread - net credit.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection, Strategy
from ait.utils.logging import get_logger
from ait.config.runtime_env import contract_float  # R19: ONE authority for env-contract defaults

log = get_logger("strategies.iron_condor")


class IronCondor(Strategy):
    """Iron condor — sell premium on both sides."""

    @property
    def name(self) -> str:
        return "iron_condor"

    @property
    def direction_bias(self) -> SignalDirection | None:
        return None  # Direction neutral

    def _log_entry_quality(
        self,
        symbol: str,
        status: str,
        reason: str,
        chain: OptionsChain,
        short_put: OptionContract | None = None,
        short_call: OptionContract | None = None,
        credit: float | None = None,
        width: float | None = None,
    ) -> None:
        """R12-B entry-quality telemetry: one structured line per generated/
        rejected condor (short deltas, EM multiples, credit/width, atm_iv).

        Evidence: the 07-07 SPY condor filled its short call at 0.49 DELTA
        (designed 0.20) and left NO trace of the gap between the designed and
        the filled structure — this line is that audit trail.
        """
        em = float(getattr(chain, "expected_move", 0.0) or 0.0)
        spot = chain.underlying_price

        def _em_mult(dist: float) -> float | None:
            return round(dist / em, 3) if em > 0 else None

        log.info(
            "condor_entry_quality",
            symbol=symbol,
            status=status,
            reason=reason,
            atm_iv=round(float(getattr(chain, "atm_iv", 0.0) or 0.0), 4),
            expected_move=round(em, 3),
            short_put_delta=(round(short_put.delta, 3) if short_put else None),
            short_call_delta=(round(short_call.delta, 3) if short_call else None),
            put_em_mult=(_em_mult(spot - short_put.strike) if short_put else None),
            call_em_mult=(_em_mult(short_call.strike - spot) if short_call else None),
            credit=(round(credit, 2) if credit is not None else None),
            width=(round(width, 2) if width is not None else None),
            credit_width_ratio=(
                round(credit / width, 3)
                if (credit is not None and width) else None
            ),
        )

    def _vol_scaled_width(self, price, short_put, short_call_hint, chain) -> float:
        """R6: wing width target = wing_k x price x IV x sqrt(DTE/365), min $2.

        Mirrors the backtest's wing sizing (config backtest.wing_k) so live
        condors are the structure the walk-forward evidence describes. IV
        from the short strikes (fallback 0.20), DTE from the chain expiry.
        """
        import math
        iv_vals = []
        for c in (short_put, short_call_hint):
            v = float(getattr(c, "implied_vol", 0) or 0) if c is not None else 0.0
            if v > 0:
                iv_vals.append(v)
        if iv_vals:
            iv = sum(iv_vals) / len(iv_vals)
        else:
            # R12-B (vol agent): the flat 0.20 fallback mis-sized wings by
            # +/-50% vs the chain's real ATM vol. Contract IV stays primary;
            # the chain's interpolated ATM IV is the fallback; 0.20 survives
            # only as the last resort when the chain carries no IV at all.
            _chain_atm_iv = float(getattr(chain, "atm_iv", 0.0) or 0.0)
            iv = _chain_atm_iv if _chain_atm_iv > 0 else 0.20
        dte = 21
        try:
            from datetime import date as _date, datetime as _dt
            exp = getattr(chain, "expiry", None)
            if exp:
                exp_d = _dt.fromisoformat(str(exp)[:10]).date()
                dte = max(1, (exp_d - _date.today()).days)
        except Exception:  # noqa: BLE001
            pass
        import os as _os_w
        k = contract_float("AIT_IC_WING_K")
        width = max(2.0, k * price * iv * math.sqrt(dte / 365))
        # R7-SOON (user-approved): budget-aware cap — the $2.1k launch
        # account cannot hold the $10-22 wings the vol formula produces on
        # index underlyings (max_loss $400-2,100 vs a ~$147 per-trade
        # budget). Worst-case max_loss = (width - credit)*100 with credit >=
        # min_ratio*width, so affordable width <= budget / (100*(1-ratio)).
        # Width floor relaxes to $1 when budget-capped (GLD/TLT strike grid
        # supports it); the credit floor + credit/width gates still decide
        # whether the narrow structure is economically WORTH trading — at
        # launch scale the bot should trade less, only when premium is rich.
        if self.risk_budget and self.risk_budget > 0:
            min_ratio = contract_float("AIT_IC_MIN_CREDIT_WIDTH")
            affordable = self.risk_budget / (100.0 * max(0.5, 1.0 - min_ratio))
            if affordable < width:
                width = max(1.0, affordable)
        return width

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # Iron condors work in ANY direction — they profit from theta decay
        # regardless of whether the market goes up, down, or sideways.
        # Only skip in extreme low IV where premium isn't worth the risk.
        # IV rank floor — read from env so we can backtest different values.
        # Default 15 (further loosened from 20 — current market IV ranks are
        # 7-25%, so 20 was still blocking most trades. Trade quality drops
        # but data collection rate matters more right now).
        # Research suggests IV rank > 50 is ideal but trades too rarely.
        import os
        iv_floor = float(os.environ.get("AIT_IRON_CONDOR_IV_FLOOR", "15"))
        if iv_rank < iv_floor:
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        liquid_puts = self._filter_liquid(chain.puts)

        if len(liquid_calls) < 2 or len(liquid_puts) < 2:
            return []

        price = chain.underlying_price

        # R12-B degraded-greeks gate (vol agent): when the feed returns NO
        # deltas at all (modelGreeks missing chain-wide), closest-match strike
        # selection is meaningless — this is the exact failure mode that let
        # the 07-07 SPY condor fill its short call at 0.49 DELTA (designed
        # 0.20). Reject before selecting anything.
        if (not any(abs(p.delta) > 0 for p in liquid_puts)
                and not any(abs(c.delta) > 0 for c in liquid_calls)):
            self._log_entry_quality(symbol, "rejected", "degraded_greeks", chain)
            return []

        # Put side (below price):
        # Sell put at delta ~0.20, buy put 1-2 strikes lower
        # (R12-B: _find_strike_by_delta now enforces target +/- 0.07)
        short_put = self._find_strike_by_delta(liquid_puts, 0.20)
        if not short_put:
            self._log_entry_quality(
                symbol, "rejected", "short_put_outside_delta_tolerance", chain)
            return []

        # Call side (above price):
        # Sell call at delta ~0.20, buy call 1-2 strikes higher
        short_call = self._find_strike_by_delta(liquid_calls, 0.20)
        if not short_call:
            self._log_entry_quality(
                symbol, "rejected", "short_call_outside_delta_tolerance",
                chain, short_put)
            return []

        # R12-B DELTA BAND (vol agent): enforce the DESIGNED trade. Credit
        # floors monotonically reward closer-to-ATM shorts, so without an
        # upper delta bound a mis-selected 0.49-delta short PASSES every gate
        # more easily than the intended 0.20 ("the gate stack's equilibrium,
        # not an accident"). Both shorts must sit in |delta| [0.15, 0.30].
        _band_lo = float(os.environ.get("AIT_IC_DELTA_MIN", "0.15"))
        _band_hi = float(os.environ.get("AIT_IC_DELTA_MAX", "0.30"))
        _spd, _scd = abs(short_put.delta), abs(short_call.delta)
        if _spd == 0 and _scd == 0:
            # Deltas 0/missing on BOTH shorts — the degraded-greeks failure
            # mode that produced the 0.49-delta fill.
            self._log_entry_quality(
                symbol, "rejected", "degraded_greeks", chain, short_put, short_call)
            return []
        if not (_band_lo <= _spd <= _band_hi and _band_lo <= _scd <= _band_hi):
            self._log_entry_quality(
                symbol, "rejected", "delta_band", chain, short_put, short_call)
            return []

        # R12-B EM SANITY GATE (vol agent): each short strike must sit within
        # [0.6, 1.3] x expected move of spot — a delta-independent second
        # opinion on strike placement (a 0.49-delta short sits ~0 EMs from
        # spot; a 0.05-delta short sits far outside; both are NOT the designed
        # ~0.20-delta / ~1-EM condor). Reject when EM is unavailable: we
        # cannot verify the structure we'd be selling.
        _em = float(getattr(chain, "expected_move", 0.0) or 0.0)
        _em_lo = float(os.environ.get("AIT_IC_EM_MIN", "0.6"))
        _em_hi = float(os.environ.get("AIT_IC_EM_MAX", "1.3"))
        if _em <= 0:
            self._log_entry_quality(
                symbol, "rejected", "em_unavailable", chain, short_put, short_call)
            return []
        _put_dist = price - short_put.strike
        _call_dist = short_call.strike - price
        if not (_em_lo * _em <= _put_dist <= _em_hi * _em
                and _em_lo * _em <= _call_dist <= _em_hi * _em):
            self._log_entry_quality(
                symbol, "rejected", "em_gate", chain, short_put, short_call)
            return []

        # R6 (user-approved): wing width was a strike-grid artifact ("2nd
        # strike beyond" = $2 on QQQ/IWM, $10 on NVDA) while the backtest
        # sizes wings as wing_k x price x IV x sqrt(DTE/365) — live traded a
        # structurally different condor than the one the evidence describes.
        # Target the vol-scaled width, snapped to the nearest listed strike.
        long_put_candidates = sorted(
            [p for p in liquid_puts if p.strike < short_put.strike],
            key=lambda p: p.strike, reverse=True,
        )
        if not long_put_candidates:
            self._log_entry_quality(
                symbol, "rejected", "no_put_wing", chain, short_put, short_call)
            return []
        _wing_target = self._vol_scaled_width(price, short_put, short_call_hint=None, chain=chain)
        long_put = min(
            long_put_candidates,
            key=lambda pc: abs((short_put.strike - pc.strike) - _wing_target),
        )

        # Vol-scaled call wing (see put side, R6)
        long_call_candidates = sorted(
            [c for c in liquid_calls if c.strike > short_call.strike],
            key=lambda c: c.strike,
        )
        if not long_call_candidates:
            self._log_entry_quality(
                symbol, "rejected", "no_call_wing", chain, short_put, short_call)
            return []
        _wing_target = self._vol_scaled_width(price, short_put, short_call_hint=short_call, chain=chain)
        long_call = min(
            long_call_candidates,
            key=lambda cc: abs((cc.strike - short_call.strike) - _wing_target),
        )

        # Verify structure: long_put < short_put < price < short_call < long_call
        if not (long_put.strike < short_put.strike < price < short_call.strike < long_call.strike):
            self._log_entry_quality(
                symbol, "rejected", "structure_invalid", chain, short_put, short_call)
            return []

        # Calculate pricing
        put_credit = short_put.mid - long_put.mid
        call_credit = short_call.mid - long_call.mid
        total_credit = put_credit + call_credit

        if total_credit <= 0:
            self._log_entry_quality(
                symbol, "rejected", "non_positive_credit", chain, short_put, short_call)
            return []

        # R6 (user-approved): cost floor — at a 50% take-profit the gross
        # must clear ~3x the round-trip cost (8 legs x ~$0.65 + entry/exit
        # crossing = $10-13); QQQ condors collecting $0.72 were structurally
        # cost-dominated regardless of hit rate.
        min_credit = contract_float("AIT_IC_MIN_CREDIT")
        if total_credit < min_credit:
            self._log_entry_quality(
                symbol, "rejected", "credit_floor", chain, short_put, short_call,
                credit=total_credit,
                width=max(short_put.strike - long_put.strike,
                          long_call.strike - short_call.strike))
            return []

        # R7: credit-to-width ratio gate — the absolute floor alone lets a
        # $0.75 credit on a $10-wide condor through (7.5% of width = terrible
        # risk/reward, ~$925 risked for a $37 TP). Practitioner floor ~20%.
        put_w = short_put.strike - long_put.strike
        call_w = long_call.strike - short_call.strike
        _mw = max(put_w, call_w)
        min_ratio = contract_float("AIT_IC_MIN_CREDIT_WIDTH")
        if _mw > 0 and (total_credit / _mw) < min_ratio:
            self._log_entry_quality(
                symbol, "rejected", "credit_width_ratio", chain,
                short_put, short_call, credit=total_credit, width=_mw)
            return []

        # Max loss = wider spread width - total credit
        put_width = short_put.strike - long_put.strike
        call_width = long_call.strike - short_call.strike
        max_width = max(put_width, call_width)
        max_loss = (max_width - total_credit) * 100

        if max_loss <= 0:
            self._log_entry_quality(
                symbol, "rejected", "non_positive_max_loss", chain,
                short_put, short_call, credit=total_credit, width=max_width)
            return []

        # R16: snap-to-strike can land WIDER than the budget-affordable target
        # _vol_scaled_width capped above (one strike up on a $1 grid, several
        # dollars on a $5 grid), so the emitted structure's max_loss could
        # exceed the very risk_budget the width cap was derived from — e.g. a
        # $3k budget (affordable width 1.0) on a $5 grid emitted max_loss $329
        # vs the $90 budget. Nothing inside the strategy re-checked it; the
        # only backstop was RiskManager gate 6b (manager.py:304, max_loss vs
        # max_position_risk_pct x NLV), which agrees with risk_budget ONLY
        # because both currently read 3% — divergence there would let an
        # over-budget structure execute. Re-check the built structure against
        # the budget the width target came from, with its own telemetry reason
        # so these show up as budget rejections, not as silent fall-through
        # candidates rejected downstream every scan.
        if self.risk_budget and self.risk_budget > 0 and max_loss > self.risk_budget:
            self._log_entry_quality(
                symbol, "rejected", "max_loss_over_budget", chain,
                short_put, short_call, credit=total_credit, width=max_width)
            return []

        max_profit = total_credit * 100

        # Exit at 50% of max profit (take profit early)
        take_profit = round(total_credit * 0.50, 2)
        # Stop loss at 2x credit received
        stop_loss = round(total_credit * 2.0, 2)

        # R12-B: entry-quality telemetry for the ACCEPTED condor.
        self._log_entry_quality(
            symbol, "generated", "all_gates_passed", chain,
            short_put, short_call, credit=total_credit, width=max_width)

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=confidence,
                contract=short_put,  # Primary reference leg
                action="SELL",
                quantity=1,
                legs=[
                    {"contract": long_put, "action": "BUY", "ratio": 1},
                    {"contract": short_put, "action": "SELL", "ratio": 1},
                    {"contract": short_call, "action": "SELL", "ratio": 1},
                    {"contract": long_call, "action": "BUY", "ratio": 1},
                ],
                entry_price=total_credit,
                max_loss=max_loss,
                max_profit=max_profit,
                stop_loss=stop_loss,
                take_profit=take_profit,
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=chain.expiry,
            )
        ]
