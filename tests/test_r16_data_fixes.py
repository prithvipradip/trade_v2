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
from datetime import date, timedelta
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

        async def _fake_fetch(symbol, underlying, exp_date, strikes, price):
            captured.append((exp_date, list(strikes)))
            return None

        svc._fetch_ibkr_expiry = _fake_fetch
        return svc, captured

    async def test_prefers_standard_class_over_first_smart(self):
        # Adjusted '2IWM' def listed FIRST — the pre-R16 code picked it.
        svc, captured = self._service([_adj_def_in_window(), _std_def()])
        await svc._get_ibkr_chain("IWM", 300.0, 14, 45)
        assert captured, "standard-class def has in-window expiries — fetch must run"
        for _exp, strikes in captured:
            # Standard-class grid brackets spot; the 2IWM mini-grid (266/286)
            # sits entirely below it.
            assert any(s > 300.0 for s in strikes)
            assert 266.0 not in strikes  # outside ±20% of 300 anyway, but pin it

    async def test_falls_back_to_any_smart_when_no_class_match(self):
        svc, captured = self._service([_adj_def_in_window()])
        await svc._get_ibkr_chain("IWM", 300.0, 14, 45)
        # Fallback keeps the old behavior alive rather than returning nothing.
        assert len(captured) == 1
        assert captured[0][1] == [266.0, 286.0]

    async def test_option_contracts_pin_trading_class(self):
        """_fetch_ibkr_expiry must build Option(..., tradingClass=symbol)."""
        svc = _svc()
        seen = []

        async def _qualify(*batch):
            seen.extend(batch)
            return [SimpleNamespace(conId=0) for _ in batch]  # nothing qualifies

        svc._ibkr = SimpleNamespace(
            connected=True,
            ib=SimpleNamespace(
                qualifyContractsAsync=_qualify,
                reqMktData=MagicMock(),
                cancelMktData=MagicMock(),
            ),
        )
        out = await svc._fetch_ibkr_expiry(
            "IWM", MagicMock(), date.today() + timedelta(days=20), [290.0, 300.0], 300.0
        )
        assert out is None  # no leg qualified
        assert seen and all(c.tradingClass == "IWM" for c in seen)

    def test_contract_builder_defaults_trading_class_to_symbol(self):
        opt = ContractBuilder.option("SPY", "2026-09-04", 682.0, "C")
        assert opt.tradingClass == "SPY"
        # Explicit override stays available (index weeklies etc.)
        opt2 = ContractBuilder.option(
            "SPX", "2026-09-04", 6800.0, "P", trading_class="SPXW"
        )
        assert opt2.tradingClass == "SPXW"


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
        # delta==0 contract with placeholder IV: sigma must come from the
        # chain median (0.20), not 1e-5 (which pins delta to ~0).
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
        assert bad.implied_vol == 0.0  # coerced to missing
        assert bad.delta != 0.0  # BS backfill ran with the chain median

    def test_calculate_greeks_survives_none_delta(self):
        pytest.importorskip("py_vollib")
        bad = _contract(105.0, "P", iv=0.20, delta=None, gamma=None, theta=None, vega=None)
        chain = _chain(calls=[_contract(95.0, "C", iv=0.20)], puts=[bad])
        _svc()._calculate_greeks(chain)  # must not raise
        assert bad.delta is not None and bad.delta < 0  # put delta backfilled

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
            "SPY", MagicMock(), date.today() + timedelta(days=20), [100.0], 100.0
        )
        assert chain is not None
        for c in chain.calls + chain.puts:
            for v in (c.implied_vol, c.delta, c.gamma, c.theta, c.vega):
                assert v == 0.0


# ---------------------------------------------------------------------------
# [4] snapshot hygiene: every reqMktData paired with cancelMktData
# ---------------------------------------------------------------------------

def _make_md_service(model_greeks="default", ticker_raises=False):
    """Service + fake ib for _fetch_ibkr_expiry (fast: sleep patched out)."""
    if model_greeks == "default":
        model_greeks = SimpleNamespace(
            impliedVol=0.2, delta=0.3, gamma=0.01, theta=-0.02, vega=0.05
        )

    class _FakeIB:
        def __init__(self):
            self.md_reqs = []
            self.md_cancels = []

        async def qualifyContractsAsync(self, *batch):
            return [
                SimpleNamespace(conId=i + 1, strike=c.strike, right=c.right)
                for i, c in enumerate(batch)
            ]

        def reqMktData(self, q, *_a):
            self.md_reqs.append(q)

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
        svc, fake_ib = _make_md_service()
        strikes = [95.0 + i for i in range(30)]  # 60 contracts → several md batches
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), strikes, 100.0
        )
        assert chain is not None
        assert len(fake_ib.md_reqs) == 60
        assert len(fake_ib.md_cancels) == 60
        assert {id(q) for q in fake_ib.md_reqs} == {id(q) for q in fake_ib.md_cancels}

    async def test_cancel_runs_even_when_harvest_raises(self):
        svc, fake_ib = _make_md_service(ticker_raises=True)
        chain = await svc._fetch_ibkr_expiry(
            "SPY", MagicMock(), date.today() + timedelta(days=20), [100.0, 101.0], 100.0
        )
        assert chain is None  # harvest failed
        assert len(fake_ib.md_cancels) == len(fake_ib.md_reqs) > 0


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
