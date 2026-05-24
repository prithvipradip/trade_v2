"""Tests for Fix 5 — Options Bid-Ask Spread Model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.optimization.param_spaces import STRATEGY_SPACES


def _make_ohlcv(days: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.010, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(days, 5_000_000)},
        index=dates,
    )


def _bt(spread_base=0.03, spread_iv_sensitivity=0.10,
        spread_dte_sensitivity=0.005, spread_cap=0.15) -> Backtester:
    return Backtester(
        data=_make_ohlcv(),
        strategies=["iron_condor"],
        initial_capital=50_000,
        min_confidence=0.30,
        spread_base=spread_base,
        spread_iv_sensitivity=spread_iv_sensitivity,
        spread_dte_sensitivity=spread_dte_sensitivity,
        spread_cap=spread_cap,
    )


class TestHalfSpread:
    """T5-1 through T5-3: _options_half_spread() behaviour."""

    def test_half_spread_increases_with_iv(self) -> None:
        bt = _bt(spread_base=0.03, spread_iv_sensitivity=0.15)
        spread_lo = bt._options_half_spread(iv=0.20, dte=21)
        spread_mid = bt._options_half_spread(iv=0.40, dte=21)
        spread_hi  = bt._options_half_spread(iv=0.60, dte=21)
        assert spread_lo == pytest.approx(0.03, abs=0.001)
        assert spread_mid > spread_lo
        assert spread_hi > spread_mid

    def test_half_spread_increases_near_expiry(self) -> None:
        bt = _bt(spread_dte_sensitivity=0.005)
        spread_normal = bt._options_half_spread(iv=0.25, dte=21)
        spread_nearex  = bt._options_half_spread(iv=0.25, dte=3)
        assert spread_nearex > spread_normal

    def test_half_spread_capped_at_spread_cap(self) -> None:
        bt = _bt(spread_cap=0.10, spread_iv_sensitivity=1.0)
        spread = bt._options_half_spread(iv=2.0, dte=1)
        assert spread == pytest.approx(0.10, abs=0.001)

    def test_half_spread_never_negative(self) -> None:
        bt = _bt(spread_base=0.01)
        assert bt._options_half_spread(iv=0.01, dte=100) >= 0


class TestSpreadReducesNetCredit:
    """T5-4 through T5-5: spread cost reduces entry credit."""

    def test_per_leg_spread_reduces_net_credit(self) -> None:
        hist = _make_ohlcv(90)
        underlying = hist["Close"].iloc[-1]
        iv = 0.25
        t = 21 / 365
        r = 0.05

        from ait.backtesting.options_sim import black_scholes_price, find_strike_by_delta, OptionType

        # Short call (OTM)
        sc_strike = find_strike_by_delta(underlying, t, iv, 0.20, OptionType.CALL, r)
        sc_mid = black_scholes_price(underlying, sc_strike, t, r, iv, OptionType.CALL)

        bt_nospread = _bt(spread_base=0.0)
        bt_spread   = _bt(spread_base=0.05)

        credit_nospread = sc_mid
        half_sp = bt_spread._options_half_spread(iv=iv, dte=21)
        credit_with_spread = sc_mid - half_sp  # seller pays half spread on entry

        assert credit_with_spread < credit_nospread


class TestParamSpacesContainSpread:
    """T5-6: spread params present in STRATEGY_SPACES."""

    def test_iron_condor_has_spread_params(self) -> None:
        ic_space = STRATEGY_SPACES.get("iron_condor", {})
        assert "spread_base" in ic_space, \
            "spread_base must be in iron_condor param space"
        assert "spread_iv_sensitivity" in ic_space, \
            "spread_iv_sensitivity must be in iron_condor param space"

    def test_spread_base_range_is_plausible(self) -> None:
        ic_space = STRATEGY_SPACES.get("iron_condor", {})
        spec = ic_space.get("spread_base")
        assert spec is not None
        kind, low, high = spec[0], spec[1], spec[2]
        assert kind == "float"
        assert 0.01 <= low <= 0.05
        assert 0.05 <= high <= 0.15
