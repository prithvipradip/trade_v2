"""R16 — data-dimension fixes from the 2026-08 audit round.

Pins the corrected behavior for:
  [1] chain-def selection prefers exchange=='SMART' AND tradingClass==symbol
      (reqSecDefOptParams lists the adjusted '2IWM'/'2SPY'/'2QQQ' class first
      on SMART — the old first-SMART pick served a 1-3 strike garbage chain),
      plus tradingClass=symbol pinned on Option() contracts so class-ambiguous
      strikes stop dying at qualification ('Ambiguous contract').
  [2] placeholder-IV filter: Yahoo's ~1e-5 off-hours impliedVolatility rows
      are treated as MISSING (floor MIN_PHYSICAL_IV), not real vol.
  [3] None-greeks guard: ib_insync maps IB's "not yet computed" sentinels to
      None — _calculate_greeks / filter_by_delta must skip/coerce instead of
      raising TypeError and killing the whole symbol's scan.
  [4] chain snapshot hygiene: every reqMktData is paired with cancelMktData
      in a finally, at reduced concurrency.
  [5] dead Polygon key: repeated auth failures disable the client instead of
      burning an HTTPS roundtrip on every get_historical cache miss forever.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from ait.broker.contracts import ContractBuilder
from ait.data.market_data import MarketDataService
from ait.data.options_chain import (
    MIN_PHYSICAL_IV,
    OptionContract,
    OptionsChain,
    OptionsChainService,
    atm_iv,
)


def _contract(strike, right, iv=0.20, delta=0.2, **kw):
    return OptionContract(
        symbol=kw.pop("symbol", "SPY"),
        expiry=kw.pop("expiry", date.today() + timedelta(days=30)),
        strike=strike,
        right=right,
        bid=kw.pop("bid", 1.0),
        ask=kw.pop("ask", 1.2),
        last=kw.pop("last", 1.1),
        volume=kw.pop("volume", 100),
        open_interest=kw.pop("open_interest", 200),
        implied_vol=iv,
        delta=delta,
        **kw,
    )


def _chain(calls, puts, spot=100.0):
    return OptionsChain(
        symbol="SPY",
        underlying_price=spot,
        expiry=date.today() + timedelta(days=30),
        calls=calls,
        puts=puts,
    )


def _svc() -> OptionsChainService:
    """Bare service — the methods under test touch only self._ibkr (set per test)."""
    return OptionsChainService.__new__(OptionsChainService)


# ---------------------------------------------------------------------------
# [1] chain-def selection prefers tradingClass == symbol
# ---------------------------------------------------------------------------

def _std_def(symbol="IWM"):
    exp1 = (date.today() + timedelta(days=20)).strftime("%Y%m%d")
    exp2 = (date.today() + timedelta(days=40)).strftime("%Y%m%d")
    return SimpleNamespace(
        exchange="SMART", tradingClass=symbol,
        expirations=[exp1, exp2],
        strikes=[float(s) for s in range(240, 361, 5)],
    )


def _adj_def_in_window():
    exp = (date.today() + timedelta(days=20)).strftime("%Y%m%d")
    return SimpleNamespace(
        exchange="SMART", tradingClass="2IWM",
        expirations=[exp], strikes=[266.0, 286.0],
    )


class TestChainDefSelection:
    def _service(self, chain_defs):
        svc = _svc()
        ib = SimpleNamespace(
            reqSecDefOptParamsAsync=AsyncMock(return_value=chain_defs),
        )
        svc._ibkr = SimpleNamespace(
            connected=True,
            ib=ib,
            qualify_contract=AsyncMock(
                return_value=SimpleNamespace(symbol="IWM", secType="STK", conId=9)
            ),
        )
        captured = []

        # Bug-fix note: strike discovery moved from _get_ibkr_chain (which
        # used to pre-filter chain_def.strikes -- a COMBINED, all-expiries
        # set -- and pass the guess down) into _fetch_ibkr_expiry itself
        # (per-expiry, via reqContractDetailsAsync). This class tests
        # chain_def/expirations SELECTION only, which is unaffected by that
        # move -- captured now records exp_date, the one signal that
        # survives it. Strike-discovery itself is covered by
        # TestFetchExpiryPerExpiryStrikes below.
        async def _fake_fetch(symbol, underlying, exp_date, price):
            captured.append(exp_date)
            return None

        svc._fetch_ibkr_expiry = _fake_fetch
        return svc, captured

    async def test_prefers_standard_class_over_first_smart(self):
        # Adjusted '2IWM' def listed FIRST — the pre-R16 code picked it.
        svc, captured = self._service([_adj_def_in_window(), _std_def()])
        await svc._get_ibkr_chain("IWM", 300.0, 14, 45)
        # _std_def() has 2 in-window expirations, _adj_def_in_window() has 1
        # -- the count alone distinguishes which def actually got used.
        std_expiries = {
            datetime.strptime(e, "%Y%m%d").date() for e in _std_def().expirations
        }
        assert captured, "standard-class def has in-window expiries — fetch must run"
        assert set(captured) == std_expiries

    async def test_falls_back_to_any_smart_when_no_class_match(self):
        svc, captured = self._service([_adj_def_in_window()])
        await svc._get_ibkr_chain("IWM", 300.0, 14, 45)
        # Fallback keeps the old behavior alive rather than returning nothing.
        adj_expiry = datetime.strptime(
            _adj_def_in_window().expirations[0], "%Y%m%d"
        ).date()
        assert captured == [adj_expiry]

    async def test_option_contracts_pin_trading_class(self):
        """_fetch_ibkr_expiry must request contract details with
        Option(..., tradingClass=symbol), once per side."""
        svc = _svc()
        seen = []

        async def _req_contract_details(opt):
            seen.append(opt)
            return []  # nothing listed -- proves the pin without needing real strikes

        svc._ibkr = SimpleNamespace(
            connected=True,
            ib=SimpleNamespace(
                reqContractDetailsAsync=_req_contract_details,
                reqMktData=MagicMock(),
                cancelMktData=MagicMock(),
            ),
        )
        out = await svc._fetch_ibkr_expiry(
            "IWM", MagicMock(), date.today() + timedelta(days=20), 300.0
        )
        assert out is None  # nothing listed for either side
        assert len(seen) == 2, "one reqContractDetailsAsync call per side (C, P)"
        assert all(c.tradingClass == "IWM" for c in seen)
        assert {c.right for c in seen} == {"C", "P"}
        assert all(c.strike == 0.0 for c in seen), \
            "strike=0.0 means 'every real strike at this expiry' to IB's API"

    def test_contract_builder_defaults_trading_class_to_symbol(self):
        opt = ContractBuilder.option("SPY", "2026-09-04", 682.0, "C")
        assert opt.tradingClass == "SPY"
        # Explicit override stays available (index weeklies etc.)
        opt2 = ContractBuilder.option(
            "SPX", "2026-09-04", 6800.0, "P", trading_class="SPXW"
        )
        assert opt2.tradingClass == "SPXW"


# ---------------------------------------------------------------------------
# [1b] the actual bug fix: strike discovery is per-expiry, not guessed from
# reqSecDefOptParamsAsync's ALL-expiries combined strike list. Confirmed
# live 2026-09-28: SPY at ~$765 spot generated hundreds of invalid-strike
# qualify attempts per near-dated expiry ("No security definition"/"Unknown
# contract"), enough to stall a chain fetch outright — and this was already
# happening silently on the live bot's own entry scan (~8,800 error-200
# lines in logs/ait.log), not just the probe script that surfaced it.
# ---------------------------------------------------------------------------

def _fake_contract_details_ib(strikes_by_expiry: dict[str, list[float]]):
    """A fake ib whose reqContractDetailsAsync returns ONLY the strikes
    registered for the SPECIFIC (expiry, right) requested — the real IB API
    behavior this fix relies on, unlike reqSecDefOptParamsAsync's combined
    all-expiries set. `strikes_by_expiry` keys are `lastTradeDateOrContractMonth`
    strings (e.g. "20261006"); the same strikes are listed for both rights."""

    class _FakeIB:
        def __init__(self):
            self.contract_details_calls: list = []
            self.qualify_calls: list = []
            self.md_reqs: list = []

        async def reqContractDetailsAsync(self, opt):
            self.contract_details_calls.append(opt)
            strikes = strikes_by_expiry.get(opt.lastTradeDateOrContractMonth, [])
            return [
                SimpleNamespace(contract=SimpleNamespace(
                    symbol=opt.symbol,
                    lastTradeDateOrContractMonth=opt.lastTradeDateOrContractMonth,
                    strike=s, right=opt.right,
                    conId=int(s * 10) + (1 if opt.right == "C" else 2),
                ))
                for s in strikes
            ]

        async def qualifyContractsAsync(self, *batch):  # pragma: no cover
            # Must never be called post-fix — reqContractDetailsAsync's
            # results are already qualified. Recorded, not raised, so the
            # "never called" test gets a clear assertion failure instead of
            # an opaque crash if this regresses.
            self.qualify_calls.extend(batch)
            return [SimpleNamespace(conId=0) for _ in batch]

        def reqMktData(self, q, *args):
            self.md_reqs.append(q)

        def cancelMktData(self, q):
            pass

        def ticker(self, q):
            return SimpleNamespace(
                bid=1.0, ask=1.2, last=1.1, volume=10.0,
                modelGreeks=SimpleNamespace(
                    impliedVol=0.2, delta=0.3, gamma=0.01, theta=-0.02, vega=0.05),
            )

    svc = _svc()
    fake_ib = _FakeIB()
    svc._ibkr = SimpleNamespace(connected=True, ib=fake_ib)
    return svc, fake_ib


class TestFetchExpiryPerExpiryStrikes:
    async def test_narrow_and_wide_expiries_never_cross_contaminate(self):
        """The literal bug: a near-dated 'weekly' with few real strikes must
        never end up with a contract at a strike that only exists on a
        different, wider 'monthly' expiry."""
        narrow_exp = "20261006"
        wide_exp = "20261012"
        svc, fake_ib = _fake_contract_details_ib({
            narrow_exp: [99.0, 100.0, 101.0],
            wide_exp: [90.0, 95.0, 99.0, 100.0, 101.0, 105.0, 110.0],
        })

        narrow_chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 6), 100.0)
        wide_chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 12), 100.0)

        narrow_strikes = {c.strike for c in narrow_chain.calls + narrow_chain.puts}
        wide_strikes = {c.strike for c in wide_chain.calls + wide_chain.puts}
        # The bug this guards against: a combined-strikes guess would have
        # let the narrow expiry reach for 90/95/105/110 -- strikes that only
        # exist on the wide expiry. Exact equality proves neither expiry's
        # result contains a strike outside its own registered set.
        assert narrow_strikes == {99.0, 100.0, 101.0}
        assert wide_strikes == {90.0, 95.0, 99.0, 100.0, 101.0, 105.0, 110.0}

    async def test_qualify_contracts_async_is_never_called(self):
        """reqContractDetailsAsync's results are already qualified (real
        conId) — the old per-strike qualify step must be gone entirely."""
        svc, fake_ib = _fake_contract_details_ib({"20261006": [99.0, 100.0]})
        await svc._fetch_ibkr_expiry("SPY", MagicMock(), date(2026, 10, 6), 100.0)
        assert fake_ib.qualify_calls == []

    async def test_price_band_excludes_a_real_but_out_of_band_strike(self):
        """A strike IBKR genuinely lists for this expiry, but outside the
        ±20% ATM band, must still be excluded — the band filter moved
        location (into this method) but not behavior."""
        svc, fake_ib = _fake_contract_details_ib({
            "20261006": [80.0, 100.0, 130.0],  # 130 is 30% above spot 100
        })
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 6), 100.0)
        strikes = {c.strike for c in chain.calls + chain.puts}
        assert strikes == {80.0, 100.0}
        assert 130.0 not in strikes

    async def test_quarter_dollar_strike_still_excluded(self):
        """deep-audit DATA-M7: allow $0.50 strikes, skip finer (adjusted-
        class-only) increments — preserved at the new call site."""
        svc, fake_ib = _fake_contract_details_ib({
            "20261006": [99.0, 99.25, 99.5],
        })
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 6), 100.0)
        strikes = {c.strike for c in chain.calls + chain.puts}
        assert strikes == {99.0, 99.5}
        assert 99.25 not in strikes

    async def test_one_side_failing_does_not_block_the_other(self):
        """A real Exception from reqContractDetailsAsync on one side (e.g.
        no listings for puts on this expiry) must not abort the whole
        expiry — matches this method's existing per-side fail-open."""
        exp = "20261006"

        class _FlakyIB:
            def __init__(self):
                self.md_reqs = []

            async def reqContractDetailsAsync(self, opt):
                if opt.right == "P":
                    raise RuntimeError("no security definition")
                return [SimpleNamespace(contract=SimpleNamespace(
                    symbol=opt.symbol, lastTradeDateOrContractMonth=exp,
                    strike=100.0, right="C", conId=1))]

            def reqMktData(self, q, *args):
                self.md_reqs.append(q)

            def cancelMktData(self, q):
                pass

            def ticker(self, q):
                return SimpleNamespace(
                    bid=1.0, ask=1.2, last=1.1, volume=10.0,
                    modelGreeks=SimpleNamespace(
                        impliedVol=0.2, delta=0.3, gamma=0.01, theta=-0.02, vega=0.05))

        svc = _svc()
        svc._ibkr = SimpleNamespace(connected=True, ib=_FlakyIB())
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 6), 100.0)
        assert chain is not None
        assert len(chain.calls) == 1
        assert len(chain.puts) == 0

    async def test_resulting_contracts_carry_the_real_conid(self):
        svc, fake_ib = _fake_contract_details_ib({"20261006": [100.0]})
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date(2026, 10, 6), 100.0)
        all_contracts = chain.calls + chain.puts
        assert len(all_contracts) == 2  # one call, one put at strike 100.0
        assert all(c.con_id != 0 for c in all_contracts)
        # conId encodes strike*10 + side offset in the fake -- confirms the
        # REAL per-expiry contract (not a placeholder) reached OptionContract.
        for c in all_contracts:
            expected = int(c.strike * 10) + (1 if c.right == "C" else 2)
            assert c.con_id == expected


# ---------------------------------------------------------------------------
# [2] placeholder-IV filter
# ---------------------------------------------------------------------------

class TestPlaceholderIVFilter:
    def test_atm_iv_ignores_placeholder_rows(self):
        # Placeholders at 99/101 bracket spot tighter than the real 95/105 —
        # the pre-R16 `> 0` filter interpolated them to atm_iv ≈ 1e-5.
        chain = _chain(
            calls=[
                _contract(95.0, "C", iv=0.20),
                _contract(99.0, "C", iv=1e-5),
                _contract(101.0, "C", iv=1e-5),
                _contract(105.0, "C", iv=0.20),
            ],
            puts=[],
        )
        assert chain.atm_iv == pytest.approx(0.20)

    def test_atm_iv_all_placeholders_reads_as_no_iv(self):
        chain = _chain(
            calls=[_contract(99.0, "C", iv=1e-5), _contract(101.0, "C", iv=1e-5)],
            puts=[_contract(100.0, "P", iv=0.004)],
        )
        # Every IV below MIN_PHYSICAL_IV → no usable IV at all → 0.0
        assert chain.atm_iv == 0.0
        assert atm_iv(chain) == 0.0

    def test_min_physical_iv_floor_value(self):
        # The finding's placeholder magnitude must be under the floor and
        # real equity-ETF vol comfortably above it.
        assert 1e-5 < MIN_PHYSICAL_IV <= 0.01
        assert 0.10 > MIN_PHYSICAL_IV

    def test_yahoo_chain_coerces_placeholder_iv_to_missing(self, monkeypatch):
        exp = date.today() + timedelta(days=30)
        exp_str = exp.strftime("%Y-%m-%d")

        def _rows(iv_a, iv_b):
            return pd.DataFrame([
                {"strike": 98.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1,
                 "volume": 10, "openInterest": 50, "impliedVolatility": iv_a},
                {"strike": 102.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1,
                 "volume": 10, "openInterest": 50, "impliedVolatility": iv_b},
            ])

        class _FakeTicker:
            options = (exp_str,)

            def option_chain(self, _e):
                return SimpleNamespace(
                    calls=_rows(1e-5, 0.21), puts=_rows(0.22, 1e-5)
                )

        monkeypatch.setitem(
            sys.modules, "yfinance", SimpleNamespace(Ticker=lambda s: _FakeTicker())
        )
        chains = _svc()._fetch_yahoo_chain_sync("SPY", 100.0, 14, 45)
        assert len(chains) == 1
        ivs = {(c.right, c.strike): c.implied_vol
               for grp in (chains[0].calls, chains[0].puts) for c in grp}
        assert ivs[("C", 98.0)] == 0.0      # placeholder → missing
        assert ivs[("C", 102.0)] == pytest.approx(0.21)  # real IV untouched
        assert ivs[("P", 98.0)] == pytest.approx(0.22)
        assert ivs[("P", 102.0)] == 0.0

    def test_bs_backfill_never_uses_placeholder_sigma(self):
        pytest.importorskip("py_vollib")
        # delta==0 contract with placeholder IV: sigma must never be the
        # degenerate 1e-5 (which pins delta to ~0) — 2026-09-14: now solved
        # from the contract's own real quote (bid=1.0/ask=1.2 from the
        # _contract default) rather than the chain median, since a real
        # two-sided market is available; either source clears the floor.
        bad = _contract(105.0, "C", iv=1e-5, delta=0.0)
        chain = _chain(
            calls=[_contract(95.0, "C", iv=0.20), bad],
            puts=[_contract(100.0, "P", iv=0.20)],
        )
        _svc()._calculate_greeks(chain)
        assert abs(bad.delta) > 0.05  # sigma=1e-5 would leave ~0


# ---------------------------------------------------------------------------
# [3] None-greeks guard
# ---------------------------------------------------------------------------

class TestNoneGreeksGuard:
    def test_calculate_greeks_survives_none_iv(self):
        pytest.importorskip("py_vollib")
        # Pre-R16: sigma compare on None raised TypeError OUTSIDE the try.
        bad = _contract(100.0, "C", iv=None, delta=0.0)
        chain = _chain(
            calls=[bad, _contract(95.0, "C", iv=0.20)],
            puts=[_contract(100.0, "P", iv=0.20)],
        )
        _svc()._calculate_greeks(chain)  # must not raise
        # 2026-09-14: implied_vol is now actually corrected (solved from the
        # contract's own real quote, bid=1.0/ask=1.2 from the _contract
        # default) rather than permanently left at 0.0 — see the FIFTH
        # FINDING dead-import fix.
        assert bad.implied_vol > 0
        assert bad.delta != 0.0  # BS backfill ran

    def test_calculate_greeks_survives_none_delta(self):
        pytest.importorskip("py_vollib")
        bad = _contract(105.0, "P", iv=0.20, delta=None, gamma=None, theta=None, vega=None)
        chain = _chain(calls=[_contract(95.0, "C", iv=0.20)], puts=[bad])
        _svc()._calculate_greeks(chain)  # must not raise
        assert bad.delta is not None and bad.delta < 0  # put delta backfilled

    def test_delta_present_but_iv_missing_still_solves_iv(self):
        """2026-09-14 FIFTH FINDING: live-observed pattern — IBKR delivers a
        real delta while impliedVol is still its own '-1 not computed'
        sentinel (independent per-field sentinels, confirmed from
        ib_insync's tickOptionComputation handler). The old `if c.delta !=
        0: continue` gate skipped IV entirely whenever delta was already
        populated, leaving implied_vol stuck at 0.0 forever despite a real
        quote being available to solve it from."""
        pytest.importorskip("py_vollib")
        real_delta = 0.42  # as if genuinely reported by IBKR
        bad = _contract(100.0, "C", iv=None, delta=real_delta)
        chain = _chain(
            calls=[bad, _contract(95.0, "C", iv=0.20)],
            puts=[_contract(100.0, "P", iv=0.20)],
        )
        _svc()._calculate_greeks(chain)
        assert bad.implied_vol > 0, "IV must be solved even when delta already arrived"
        assert bad.delta == real_delta, "a real IBKR delta must never be overwritten"

    def test_iv_missing_no_quote_falls_back_to_chain_median_and_persists(self):
        """No two-sided quote to solve IV from -> chain median, but (unlike
        before this fix) actually WRITTEN onto implied_vol, not just used as
        scratch sigma for the delta/gamma/theta/vega backfill."""
        pytest.importorskip("py_vollib")
        bad = _contract(100.0, "C", iv=None, delta=0.0, bid=0.0, ask=0.0)
        chain = _chain(
            calls=[bad, _contract(95.0, "C", iv=0.20)],
            puts=[_contract(100.0, "P", iv=0.20)],
        )
        _svc()._calculate_greeks(chain)
        assert bad.implied_vol == pytest.approx(0.20)  # the chain median
        assert bad.delta != 0.0  # backfilled using that median as sigma

    def test_filter_by_delta_skips_none_delta(self):
        # Pre-R16: abs(None) TypeError aborted the symbol's scan.
        good = _contract(95.0, "C", iv=0.20, delta=0.25)
        bad = _contract(100.0, "C", iv=0.20, delta=None)
        chain = _chain(calls=[good, bad], puts=[])
        out = chain.filter_by_delta(0.10, 0.30)  # must not raise
        assert good in out.calls
        assert bad not in out.calls

    async def test_ibkr_ingest_coerces_none_model_greeks(self):
        """_fetch_ibkr_expiry stores 0.0, never None, for sentinel greeks."""
        svc, fake_ib = _make_md_service(
            model_greeks=SimpleNamespace(
                impliedVol=None, delta=None, gamma=None, theta=None, vega=None
            )
        )
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), 100.0
        )
        assert chain is not None
        for c in chain.calls + chain.puts:
            for v in (c.implied_vol, c.delta, c.gamma, c.theta, c.vega):
                assert v == 0.0


# ---------------------------------------------------------------------------
# [4] snapshot hygiene: every reqMktData paired with cancelMktData
# ---------------------------------------------------------------------------

def _make_md_service(model_greeks="default", ticker_raises=False, strikes=(100.0,)):
    """Service + fake ib for _fetch_ibkr_expiry (fast: sleep patched out).

    `strikes` stands in for what reqContractDetailsAsync would really
    return for a given (expiry, right) — the SAME set for both C and P here
    (these tests exercise the reqMktData/cancel/greeks plumbing downstream
    of strike discovery, not strike discovery itself — see
    TestFetchExpiryPerExpiryStrikes for that).
    """
    if model_greeks == "default":
        model_greeks = SimpleNamespace(
            impliedVol=0.2, delta=0.3, gamma=0.01, theta=-0.02, vega=0.05
        )

    class _FakeIB:
        def __init__(self):
            self.md_reqs = []
            self.md_req_args = []
            self.md_cancels = []
            self.contract_details_calls = []

        async def reqContractDetailsAsync(self, opt):
            self.contract_details_calls.append(opt)
            return [
                SimpleNamespace(contract=SimpleNamespace(
                    symbol=opt.symbol,
                    lastTradeDateOrContractMonth=opt.lastTradeDateOrContractMonth,
                    strike=s, right=opt.right,
                    conId=int(s * 10) + (1 if opt.right == "C" else 2),
                ))
                for s in strikes
            ]

        def reqMktData(self, q, *args):
            self.md_reqs.append(q)
            self.md_req_args.append(args)  # (genericTickList, snapshot, regulatorySnapshot)

        def cancelMktData(self, q):
            self.md_cancels.append(q)

        def ticker(self, q):
            if ticker_raises:
                raise RuntimeError("boom")
            return SimpleNamespace(
                bid=1.0, ask=1.2, last=1.1, volume=10.0, modelGreeks=model_greeks
            )

    svc = _svc()
    fake_ib = _FakeIB()
    svc._ibkr = SimpleNamespace(connected=True, ib=fake_ib)
    return svc, fake_ib


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Keep the md_wait pacing out of test wall-time."""
    import asyncio as _asyncio
    import ait.data.options_chain as oc_mod

    real_sleep = _asyncio.sleep

    async def _instant(_secs):
        await real_sleep(0)

    monkeypatch.setattr(
        oc_mod, "asyncio", SimpleNamespace(sleep=_instant, get_running_loop=_asyncio.get_running_loop)
    )


class TestSnapshotCancelPairing:
    async def test_every_request_is_cancelled(self):
        strikes = [95.0 + i for i in range(30)]  # 60 contracts → several md batches
        # price=110 keeps the whole 95-124 range inside the ±20% band
        # _fetch_ibkr_expiry now applies itself (moved there from
        # _get_ibkr_chain by this fix) — unrelated to what this test checks.
        svc, fake_ib = _make_md_service(strikes=strikes)
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), 110.0
        )
        assert chain is not None
        assert len(fake_ib.md_reqs) == 60
        assert len(fake_ib.md_cancels) == 60
        assert {id(q) for q in fake_ib.md_reqs} == {id(q) for q in fake_ib.md_cancels}

    async def test_cancel_runs_even_when_harvest_raises(self):
        svc, fake_ib = _make_md_service(ticker_raises=True, strikes=(100.0, 101.0))
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), 100.0
        )
        assert chain is None  # harvest failed
        assert len(fake_ib.md_cancels) == len(fake_ib.md_reqs) > 0

    async def test_requests_are_streaming_not_snapshot(self):
        """R16's modelGreeks gap: snapshot=True's ~11s server-side collection
        window never fit inside the 2-3s per-batch wait, so modelGreeks
        silently never arrived. Confirmed live 2026-09-14 (isolated clientId
        test against the real account): streaming delivers modelGreeks
        within ~4s where snapshot didn't even at 11s."""
        svc, fake_ib = _make_md_service()
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), 100.0
        )
        assert chain is not None
        assert fake_ib.md_req_args  # at least one request happened
        for args in fake_ib.md_req_args:
            assert args[1] is False, "options quotes must stream, not snapshot"


# ---------------------------------------------------------------------------
# [5] Polygon dead-key breaker
# ---------------------------------------------------------------------------

class TestPolygonAuthBreaker:
    def _service(self, error_text):
        svc = MarketDataService.__new__(MarketDataService)

        def _raise(**_kw):
            raise Exception(error_text)

        svc._polygon_client = SimpleNamespace(list_aggs=_raise)
        svc._polygon_auth_failures = 0
        return svc

    async def test_disables_after_consecutive_auth_failures(self):
        svc = self._service("Unknown API Key")
        for _ in range(MarketDataService._POLYGON_AUTH_FAILURE_LIMIT):
            assert await svc._get_polygon_historical("SPY", 30) is None
        assert svc._polygon_client is None  # breaker tripped → Yahoo-only

    async def test_transient_errors_do_not_trip_breaker(self):
        svc = self._service("connection reset by peer")
        for _ in range(MarketDataService._POLYGON_AUTH_FAILURE_LIMIT + 2):
            assert await svc._get_polygon_historical("SPY", 30) is None
        assert svc._polygon_client is not None


# ---------------------------------------------------------------------------
# [6] get_chain_for_expiry must never silently return a DIFFERENT expiry's
# chain (the dormant landmine registered while scoping W53) — nothing calls
# it today, but a caller trusting its return value must be able to trust the
# expiry actually matches.
# ---------------------------------------------------------------------------

class TestGetChainForExpiry:
    def _svc_with_chains(self, chains):
        svc = _svc()

        async def _get_chain(symbol, min_dte=None, max_dte=None):
            return chains

        svc.get_chain = _get_chain
        return svc

    async def test_returns_none_when_exact_expiry_missing(self):
        other = _chain([], [])  # expiry = today + 30 (the helper's default)
        svc = self._svc_with_chains([other])
        requested = date.today() + timedelta(days=45)

        result = await svc.get_chain_for_expiry("SPY", requested)

        assert result is None, (
            "a missing exact expiry must return None, not a different "
            "expiry's chain mislabeled as the one asked for"
        )

    async def test_returns_the_matching_chain_when_present(self):
        target = date.today() + timedelta(days=45)
        other = _chain([], [])  # expiry = today + 30, a decoy
        wanted = OptionsChain(symbol="SPY", underlying_price=100.0,
                              expiry=target, calls=[], puts=[])
        svc = self._svc_with_chains([other, wanted])

        result = await svc.get_chain_for_expiry("SPY", target)

        assert result is wanted
