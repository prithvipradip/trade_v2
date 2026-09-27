"""W19 (w17-07): backfilled Greeks must be countable, and must not disarm the
gate that exists to catch a degraded chain.

TWO DEFECTS, ONE ROOT. The 2026-09-14 backfill change gave every contract a
non-zero delta. That (a) made `iv_solved_from_price` a success-only counter
with no way to see failures, and (b) made iron_condor's degraded_greeks gate
UNREACHABLE — it rejected only when NO delta was non-zero. The gate's own
comment names the incident it exists for: the 07-07 SPY condor that filled a
"0.20-delta" short call at 0.49 DELTA.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from ait.data.options_chain import (
    MIN_PHYSICAL_IV,
    OptionContract,
    OptionsChain,
    OptionsChainService,
)
from ait.strategies.base import SignalDirection


def _c(strike, right, iv=0.20, delta=0.2, bid=1.0, ask=1.2):
    return OptionContract(
        symbol="SPY", expiry=date.today() + timedelta(days=30), strike=strike,
        right=right, bid=bid, ask=ask, last=(bid + ask) / 2, volume=100,
        open_interest=200, implied_vol=iv, delta=delta)


def _chain(calls, puts, spot=100.0):
    return OptionsChain(symbol="SPY", underlying_price=spot,
                        expiry=date.today() + timedelta(days=30),
                        calls=calls, puts=puts)


class TestSolverFailuresAreCountable:
    def test_a_below_intrinsic_quote_is_surfaced_as_a_BAD_QUOTE(self):
        """py_vollib raises BelowIntrinsicException when mid < intrinsic — the
        QUOTE is impossible, not the solver. Previously indistinguishable from
        a convergence miss, and silent in both cases. The module logs through
        structlog (not stdlib logging, so caplog cannot see it), so assert on
        the emitted event directly."""
        pytest.importorskip("py_vollib")
        import ait.data.options_chain as OC
        seen = []
        real = OC.log.warning
        OC.log.warning = lambda ev, **kw: seen.append((ev, kw))
        try:
            # S=100, K=60 call: intrinsic 40, but the market shows 5.0.
            bad = _c(60.0, "C", iv=None, delta=0.0, bid=4.9, ask=5.1)
            chain = _chain([bad, _c(100.0, "C", iv=0.22)], [_c(100.0, "P", iv=0.20)])
            OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
        finally:
            OC.log.warning = real
        events = [e for e, _ in seen]
        assert "iv_solve_quote_outside_arbitrage_bounds" in events, \
            f"an impossible quote must be surfaced, not silently absorbed; saw {events}"

    def test_the_contract_still_gets_a_usable_iv(self):
        """Counting the failure must not change the fallback behaviour."""
        pytest.importorskip("py_vollib")
        bad = _c(60.0, "C", iv=None, delta=0.0, bid=4.9, ask=5.1)
        chain = _chain([bad, _c(100.0, "C", iv=0.22)], [_c(100.0, "P", iv=0.20)])
        OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
        assert bad.implied_vol and bad.implied_vol >= MIN_PHYSICAL_IV


class TestGuessedIvIsSurfaced:
    def test_a_chain_with_no_usable_iv_anywhere_is_flagged(self):
        pytest.importorskip("py_vollib")
        # W35: these contracts used to carry a solvable 1.0 x 1.2 quote, so
        # both were SOLVED from price and the chain was not a guess at all —
        # the flag was simply computed before the solve (R26
        # iv-is-guessed-computed-before-the-solve). "No usable IV anywhere"
        # means no delivered IV AND nothing to solve one from.
        chain = _chain([_c(100.0, "C", iv=0.0, delta=0.0, bid=0.0, ask=0.0)],
                       [_c(95.0, "P", iv=0.0, delta=0.0, bid=0.0, ask=0.0)])
        OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
        assert chain.iv_is_guessed is True, \
            "no usable IV anywhere means every backfilled delta is a guess"

    def test_a_normal_chain_is_not_flagged(self):
        """The narrowness IS the fix. A few solved contracts per expiry is
        routine — live logs show 2-55 — and must not trip anything."""
        pytest.importorskip("py_vollib")
        chain = _chain([_c(100.0, "C", iv=0.22), _c(105.0, "C", iv=0.0, delta=0.0)],
                       [_c(95.0, "P", iv=0.20)])
        OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
        assert chain.iv_is_guessed is False

    def test_the_flag_survives_chain_filtering(self):
        """filter_by_delta/filter_liquid build NEW chains. A filtered chain
        that drops this flag would silently disarm the gate all over again —
        the same fragility atm_iv already has."""
        pytest.importorskip("py_vollib")
        chain = _chain([_c(100.0, "C", iv=0.0, delta=0.0, bid=0.0, ask=0.0)],
                       [_c(95.0, "P", iv=0.0, delta=0.0, bid=0.0, ask=0.0)])
        OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
        assert chain.iv_is_guessed is True
        assert chain.filter_liquid().iv_is_guessed is True
        assert chain.filter_by_delta(0.05, 0.95).iv_is_guessed is True


class TestDegradedGreeksGateIsReArmed:
    def _strategy(self):
        from ait.strategies.iron_condor import IronCondor
        return IronCondor.__new__(IronCondor)

    def _wide_chain(self, delta: float, iv_is_guessed: bool) -> OptionsChain:
        """2 calls + 2 puts, all liquid, so only the gate under test (not the
        `len(liquid_*) < 2` guard) can be the reason for a rejection."""
        chain = _chain(
            [_c(100.0, "C", delta=delta), _c(105.0, "C", delta=delta)],
            [_c(95.0, "P", delta=-delta), _c(90.0, "P", delta=-delta)],
        )
        chain.iv_is_guessed = iv_is_guessed
        return chain

    def test_a_guessed_iv_chain_is_rejected(self):
        """CENSUS-4 (2026-09-27): mutation-verified the ORIGINAL claim (byte-
        offset ordering between the two checks) has NO observable effect --
        both return [] with the identical 'degraded_greeks' reason, so their
        relative order is unobservable. What actually matters, and what this
        drives for real through IronCondor.generate_signals: the guessed-IV
        check fires BEFORE strike selection, so a guessed chain with
        technically-nonzero (but wildly off-target) deltas still reports
        degraded_greeks -- not a later, unrelated rejection reason."""
        strategy = self._strategy()

        # Good deltas (0.20, on-target): guessed -> rejected; the SAME chain
        # unflagged -> an actual signal (the gate, not incidental data, is
        # what's rejecting it).
        guessed = self._wide_chain(delta=0.20, iv_is_guessed=True)
        sigs = strategy.generate_signals(
            "SPY", guessed, SignalDirection.NEUTRAL, confidence=0.6, iv_rank=50.0)
        assert sigs == []
        assert strategy.last_entry_reason == "degraded_greeks"

        unguessed = self._wide_chain(delta=0.20, iv_is_guessed=False)
        strategy.generate_signals(
            "SPY", unguessed, SignalDirection.NEUTRAL, confidence=0.6, iv_rank=50.0)
        assert strategy.last_entry_reason != "degraded_greeks", (
            "an unflagged, on-target chain must clear the guessed-IV gate "
            "(it may still be rejected downstream by an unrelated economic "
            "gate -- em_gate, credit floor, etc. -- on this synthetic chain, "
            "which is not what this test is checking)"
        )

        # 0.9-magnitude deltas: nonzero, so the LEGACY check alone would NOT
        # catch this -- it would fall through to strike selection, which (0.9
        # is far outside the 0.20+/-0.07 tolerance) would reject with
        # short_put_outside_delta_tolerance instead. Guessed must still win.
        guessed_offtarget = self._wide_chain(delta=0.9, iv_is_guessed=True)
        sigs = strategy.generate_signals(
            "SPY", guessed_offtarget, SignalDirection.NEUTRAL,
            confidence=0.6, iv_rank=50.0)
        assert sigs == []
        assert strategy.last_entry_reason == "degraded_greeks", (
            "the guessed-IV check must fire before strike selection even "
            "when deltas are technically nonzero"
        )

    def test_the_legacy_no_delta_check_is_retained(self):
        """Keep it: a chain that somehow arrives with all-zero deltas and a
        usable (unguessed) IV is still meaningless for delta selection."""
        strategy = self._strategy()
        chain = self._wide_chain(delta=0.0, iv_is_guessed=False)
        sigs = strategy.generate_signals(
            "SPY", chain, SignalDirection.NEUTRAL, confidence=0.6, iv_rank=50.0)
        assert sigs == []
        assert strategy.last_entry_reason == "degraded_greeks", (
            "an unguessed chain with all-zero deltas must still be caught by "
            "the legacy check"
        )
