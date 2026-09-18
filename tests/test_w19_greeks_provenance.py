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

    def test_a_guessed_iv_chain_is_rejected(self):
        """Before W19 this was unreachable: backfill gives every contract a
        non-zero delta, so 'no deltas at all' never happened."""
        import inspect
        from ait.strategies import iron_condor as IC
        src = inspect.getsource(IC.IronCondor.generate_signals)
        assert "iv_is_guessed" in src, "the gate must consult the guessed-IV flag"
        assert src.index("iv_is_guessed") < src.index("not any(abs(p.delta) > 0"), \
            "the guessed-IV check must run BEFORE the unreachable legacy check"

    def test_the_legacy_no_delta_check_is_retained(self):
        """Keep it: a chain that somehow arrives with all-zero deltas and a
        usable IV is still meaningless for delta selection."""
        import inspect
        from ait.strategies import iron_condor as IC
        src = inspect.getsource(IC.IronCondor.generate_signals)
        assert src.count("degraded_greeks") >= 2
