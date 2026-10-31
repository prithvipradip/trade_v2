"""Review fixes on the W106-W108 branch (PR 8).

* the executor's leg-shape guard knows the two live credit verticals (class guard: every multi-leg
  strategy the selector can build has a shape);
* the sizer's selling set and the selector's selling set agree (class guard);
* the research loaders judge a pinned window against ITS end, not against today;
* a credit-spread wing snapped to a listed strike never lands wider than the budget carries when a
  narrower listed strike fits;
* an option fetch whose bars were all dropped for want of an underlying close is asked again;
* the walk-forward's VIX staleness check survives a tz-aware symbol index.

House rules: every test executes the real code path; no source-text assertions.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import ait.data.market_data as md
from ait.backtesting.walkforward import WalkForwardBacktester
from ait.config.settings import PositionSizingConfig
from ait.data.historical import HistoricalDataStore
from ait.execution.executor import _EXPECTED_LEG_SHAPE, _validate_combo_legs
from ait.risk import scalers
from ait.risk.scalers import ScalerInputs, scaler_factors
from ait.strategies.credit_spreads import PutCreditSpread
from ait.strategies.iron_condor import IronCondor
from ait.strategies.base import SignalDirection
from ait.strategies.selector import STRATEGY_MAP, StrategySelector
from tests.test_research_data_coverage import _fake_yahoo
from tests.test_w108_4b_credit_spreads import _bs_chain, _legs, pinned_contract  # noqa: F401 — autouse fixture


# --------------------------------------------------------------------------- executor leg shape

def _legs_of(*actions):
    return [{"action": a, "conId": 1000 + i, "ratio": 1} for i, a in enumerate(actions)]


@pytest.mark.parametrize("name", ["put_credit_spread", "call_credit_spread"])
def test_a_credit_vertical_needs_one_sell_and_one_buy(name):
    assert _validate_combo_legs(name, _legs_of("SELL", "BUY")) is None
    assert "leg shape" in _validate_combo_legs(name, _legs_of("SELL", "SELL"))
    assert "leg shape" in _validate_combo_legs(name, _legs_of("BUY", "BUY"))


# the single-leg structures the selector can build (no combo, so no shape to check)
_SINGLE_LEG = {"long_call", "long_put"}


def test_every_multi_leg_strategy_the_selector_can_build_has_a_leg_shape():
    missing = sorted(set(STRATEGY_MAP) - _SINGLE_LEG - set(_EXPECTED_LEG_SHAPE))
    assert not missing, f"no entry in executor._EXPECTED_LEG_SHAPE for {missing}"


# --------------------------------------------------------------------------- one selling set

def test_the_selector_sellers_are_sizer_sellers_too():
    assert StrategySelector.SELLING_STRATEGIES <= scalers.SELLING_STRATEGIES


def test_the_iv_rank_factor_rises_with_rank_for_a_credit_vertical():
    sizing = PositionSizingConfig()

    def factor(rank):
        return scaler_factors("put_credit_spread", ScalerInputs(confidence=0.7, implied_vol=0.2, iv_rank=rank),
                              sizing, 0.5).iv_rank

    assert factor(90.0) > factor(10.0)


# --------------------------------------------------------------------------- the window ends where the query ends

TODAY = date(2026, 10, 31)


def _seed(tmp_path, *, start: str, end: str, symbol: str | None = None):
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    idx = pd.bdate_range(start=start, end=end)
    if symbol is None:                                    # the VIX family, stored /100
        store.save_daily_iv_bars("VIX", pd.DataFrame(
            {"Open": 0.150, "High": 0.160, "Low": 0.140, "Close": 0.155}, index=idx))
    else:
        store.save(symbol, pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1e6}, index=idx))
    return db


def test_a_pinned_vix_window_is_judged_against_its_end_not_today(tmp_path, monkeypatch):
    db = _seed(tmp_path, start="2024-01-02", end="2025-12-31")
    monkeypatch.setattr(md, "today_et", lambda: TODAY)
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=730, db_path=db, research=True,
                             start_date=date(2024, 1, 2), end_date=date(2025, 12, 31))
    assert calls["n"] == 0, "Yahoo was asked although the store covers the pinned window"
    assert df.index.max() == pd.Timestamp("2025-12-31") and df["Close"].iloc[-1] == pytest.approx(15.5)


def test_a_pinned_vix_window_without_a_start_is_anchored_to_its_end(tmp_path, monkeypatch):
    """No start_date: `days` before the END, not before today (today - 760 would hold only ~315 bars)."""
    db = _seed(tmp_path, start="2023-01-02", end="2025-12-31")
    monkeypatch.setattr(md, "today_et", lambda: TODAY)
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=730, db_path=db, research=True, end_date=date(2025, 12, 31))
    assert calls["n"] == 0
    assert df.index.max() == pd.Timestamp("2025-12-31") and len(df) > 500


def test_a_pinned_symbol_window_is_judged_against_its_end_not_today(tmp_path, monkeypatch):
    db = _seed(tmp_path, start="2024-01-02", end="2025-12-31", symbol="QQQ")
    monkeypatch.setattr(md, "today_et", lambda: TODAY)
    out = md._load_research_daily("QQQ", 730, db, "", date(2024, 1, 2), date(2025, 12, 31))
    assert out is not None and out.index.max() == pd.Timestamp("2025-12-31")


def test_without_an_end_date_a_short_store_still_falls_back(tmp_path, monkeypatch):
    """Unpinned = the window ends today, so a store that stopped in 2025 IS short."""
    db = _seed(tmp_path, start="2024-01-02", end="2025-12-31")
    monkeypatch.setattr(md, "today_et", lambda: TODAY)
    calls = _fake_yahoo(monkeypatch, rows=300)
    md.load_daily_ohlcv("^VIX", days=730, db_path=db, research=True, start_date=date(2024, 1, 2))
    assert calls["n"] == 1


def test_a_pinned_symbol_window_still_falls_back_when_it_is_really_short(tmp_path, monkeypatch):
    db = _seed(tmp_path, start="2025-10-01", end="2025-12-31", symbol="QQQ")     # ~65 bars of ~500
    monkeypatch.setattr(md, "today_et", lambda: TODAY)
    out = md._load_research_daily("QQQ", 730, db, "", date(2024, 1, 2), date(2025, 12, 31))
    assert out is None


# --------------------------------------------------------------------------- the wing snap respects the budget

def test_a_budget_capped_wing_is_not_snapped_wider_than_the_budget_carries():
    chain = _bs_chain(600.0, 0.18, 21)                    # $1 strikes: 5 and 6 wide are both listed
    cap = 5.6                                             # budget / (100 x (1 - 0.10)), between 5 and 6
    budget = IronCondor.budget_for_width(cap)
    strategy = PutCreditSpread()
    strategy.risk_budget = budget
    sigs = strategy.generate_signals("SPY", chain, SignalDirection.BULLISH, 0.6, 40.0)
    assert len(sigs) == 1
    short, long_ = _legs(sigs[0])
    width = abs(short.strike - long_.strike)
    assert width == pytest.approx(5.0), "the nearest strike to 5.6 is 6 wide; the budget carries only 5"
    assert sigs[0].max_loss <= budget


def test_an_uncapped_wing_still_snaps_to_the_nearest_strike():
    chain = _bs_chain(600.0, 0.18, 21)
    strategy = PutCreditSpread()
    strategy.risk_budget = None
    sig = strategy.generate_signals("SPY", chain, SignalDirection.BULLISH, 0.6, 40.0)[0]
    short, long_ = _legs(sig)
    target = strategy.wing_width(chain.expected_move)
    widths = [abs(short.strike - c.strike) for c in chain.puts if c.strike < short.strike]
    assert abs(short.strike - long_.strike) == pytest.approx(min(widths, key=lambda w: abs(w - target)))


# --------------------------------------------------------------------------- the option fetch log

def test_bars_dropped_for_a_missing_underlying_close_leave_the_contract_open():
    from tests.test_w106_option_fetch_log import _FakeClient, _contract, _run, _temp_store
    store = _temp_store()
    t_ms = int(pd.Timestamp(date(2024, 9, 18), tz="UTC").timestamp() * 1000)
    bars = [{"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 1000, "vw": 5.1, "n": 200, "t": t_ms}]
    assert _run(store, _FakeClient(bars=bars)) == 0
    assert store.option_fetch_is_final(_contract()["option_ticker"], date(2024, 9, 1)) is False


def test_an_empty_answer_is_still_final():
    from tests.test_w106_option_fetch_log import _FakeClient, _contract, _run, _temp_store
    store = _temp_store()
    _run(store, _FakeClient(bars=[]))
    assert store.option_fetch_is_final(_contract()["option_ticker"], date(2024, 9, 1)) is True


# --------------------------------------------------------------------------- the walk-forward's VIX staleness check

def _frame(end: str, close: float, tz=None) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=300, tz=tz)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0}, index=idx)


class _Loader:
    def __init__(self, stored_end: str):
        self.calls: list[bool] = []
        self._stored_end = stored_end

    def __call__(self, symbol, *, research, **kw):
        self.calls.append(research)
        return _frame(self._stored_end, 30.0) if research else _frame("2026-10-27", 18.0)


def test_a_tz_aware_symbol_index_does_not_break_the_staleness_check(tmp_path):
    wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db")
    data = {"SPY": _frame("2026-10-27", 1.0, tz="America/New_York")}
    stale = _Loader(stored_end="2026-08-25")
    vix = wf._load_vix_context(stale, 400, data)
    assert stale.calls == [True, False] and vix["Close"].iloc[-1] == 18.0
    current = _Loader(stored_end="2026-10-27")
    assert wf._load_vix_context(current, 400, data)["Close"].iloc[-1] == 30.0
    assert current.calls == [True]
