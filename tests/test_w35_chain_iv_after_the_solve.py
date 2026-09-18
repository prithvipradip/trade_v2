"""W35 (R26 fix-set F8, the two findings that reach strike selection).

`OptionsChainService._calculate_greeks` solves each contract's IV from its own
quote — the normal path on this feed, where contracts arrive with
`implied_vol=0`. Two chain-level facts were computed BEFORE that solve and
never again:

  * `iv_is_guessed` was set from the PRE-solve IV set, so a chain whose every
    contract was genuinely solved from real prices was still flagged "guessed"
    — and `iron_condor.py` refuses a flagged chain.
  * `atm_iv` / `expected_move` are computed in `OptionsChain.__post_init__`
    (before the solve) and were never refreshed, so wings were sized and the
    expected-move gate evaluated from 0.0 while the contracts carried a correct
    solved sigma.

A third, smaller one falls out of the same ordering: the fallback sigma for a
quote-less contract was the PRE-solve median, so in a chain that needed solving
it was the flat 0.30 guess even though real solved IVs existed a few lines later.

Quotes are generated with py_vollib itself at a known sigma, so the expected
answers are exact. Every test drives the real `_calculate_greeks`.
"""
from __future__ import annotations

import math
from datetime import timedelta

import pytest

pytest.importorskip("py_vollib")
from py_vollib.black_scholes import black_scholes  # noqa: E402

import ait.data.options_chain as OC  # noqa: E402
from ait.data.options_chain import (  # noqa: E402
    OptionContract,
    OptionsChain,
    OptionsChainService,
)
from ait.utils.time import today_et  # noqa: E402

SPOT, SIGMA, DTE, R = 500.0, 0.22, 30, 0.05
EXPIRY = today_et() + timedelta(days=DTE)
T = DTE / 365.0


def _priced(strike: float, right: str, *, sigma: float = SIGMA,
            quoted: bool = True) -> OptionContract:
    """A contract as this feed delivers it: real two-sided quote, no IV, no
    Greeks."""
    px = black_scholes("c" if right == "C" else "p", SPOT, strike, T, R, sigma)
    bid, ask = (round(px - 0.02, 4), round(px + 0.02, 4)) if quoted else (0.0, 0.0)
    return OptionContract(
        symbol="SPY", expiry=EXPIRY, strike=strike, right=right, bid=bid,
        ask=ask, last=px if quoted else 0.0, volume=100, open_interest=200,
        implied_vol=0.0, delta=0.0)


def _chain(calls, puts) -> OptionsChain:
    return OptionsChain(symbol="SPY", underlying_price=SPOT, expiry=EXPIRY,
                        calls=calls, puts=puts)


def _solve(chain: OptionsChain) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []
    real = OC.log.warning
    OC.log.warning = lambda ev, **kw: seen.append((ev, kw))
    try:
        OptionsChainService.__new__(OptionsChainService)._calculate_greeks(chain)
    finally:
        OC.log.warning = real
    return seen


def _solved_chain():
    return _chain([_priced(k, "C") for k in (480.0, 500.0, 520.0)],
                  [_priced(k, "P") for k in (480.0, 500.0, 520.0)])


class TestAGenuinelySolvedChainIsNotAGuess:
    def test_every_contract_solved_means_not_guessed(self):
        chain = _solved_chain()
        events = [e for e, _ in _solve(chain)]
        ivs = [c.implied_vol for c in chain.calls + chain.puts]
        assert all(abs(iv - SIGMA) < 0.01 for iv in ivs), ivs   # really solved
        assert chain.iv_is_guessed is False, (
            "six contracts solved from real prices, none from 0.30 — and the "
            "chain is still called a guess; iron_condor refuses such a chain")
        assert "chain_iv_is_a_guess" not in events

    def test_a_chain_nothing_can_be_solved_from_is_still_a_guess(self):
        """The W19 gate must survive: no delivered IV AND no usable quote."""
        chain = _chain([_priced(500.0, "C", quoted=False)],
                       [_priced(480.0, "P", quoted=False)])
        events = [e for e, _ in _solve(chain)]
        assert chain.iv_is_guessed is True
        assert "chain_iv_is_a_guess" in events
        assert chain.filter_by_delta(0.05, 0.95).iv_is_guessed is True


class TestAtmIvAndExpectedMoveFollowTheSolve:
    def test_they_are_refreshed_from_the_solved_ivs(self):
        chain = _solved_chain()
        assert chain.atm_iv == 0.0 and chain.expected_move == 0.0   # as built
        _solve(chain)
        assert chain.atm_iv == pytest.approx(SIGMA, abs=0.005), chain.atm_iv
        assert chain.expected_move == pytest.approx(
            SPOT * SIGMA * math.sqrt(DTE / 365.0), rel=0.03), chain.expected_move

    def test_the_condor_no_longer_sizes_wings_from_the_flat_fallback(self):
        """iron_condor.py: `iv = chain.atm_iv if chain.atm_iv > 0 else 0.20`.
        With atm_iv stuck at 0.0 every solved chain took the 0.20 fallback."""
        chain = _solved_chain()
        _solve(chain)
        assert chain.atm_iv > 0

    def test_a_delivered_atm_iv_is_not_disturbed_when_nothing_was_solved(self):
        calls = [_priced(k, "C") for k in (480.0, 500.0, 520.0)]
        puts = [_priced(k, "P") for k in (480.0, 500.0, 520.0)]
        for c in calls + puts:
            c.implied_vol = 0.31                 # the feed delivered IVs
        chain = _chain(calls, puts)
        before = chain.atm_iv
        _solve(chain)
        assert before == pytest.approx(0.31) and chain.atm_iv == pytest.approx(0.31)

    def test_backfilled_guesses_do_not_leak_into_atm_iv(self):
        """The refresh must read SOLVED/DELIVERED IVs only. A quote-less
        ATM strike filled with a fallback sigma is not an observation."""
        calls = [_priced(480.0, "C"), _priced(500.0, "C", quoted=False),
                 _priced(520.0, "C")]
        puts = [_priced(480.0, "P"), _priced(520.0, "P")]
        chain = _chain(calls, puts)
        _solve(chain)
        assert chain.atm_iv == pytest.approx(SIGMA, abs=0.005), chain.atm_iv


class TestTheFallbackSigmaComesFromTheSolvedChain:
    def test_a_quote_less_contract_gets_the_solved_median_not_030(self):
        dead = _priced(540.0, "C", quoted=False)
        chain = _chain([_priced(480.0, "C"), _priced(500.0, "C"), dead],
                       [_priced(480.0, "P"), _priced(500.0, "P")])
        _solve(chain)
        assert dead.implied_vol == pytest.approx(SIGMA, abs=0.01), (
            f"backfilled from {dead.implied_vol}: the flat 0.30 guess, in a "
            f"chain whose other contracts all solved to ~{SIGMA}")
        assert dead.delta != 0.0
