"""Tests for Fix 4 — Realistic IV from IBKR Historical Data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester


def _make_ohlcv(days: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(days, 5_000_000)},
        index=dates,
    )


def _make_backtester(**kwargs) -> Backtester:
    df = _make_ohlcv()
    return Backtester(
        data=df,
        strategies=["iron_condor"],
        initial_capital=50_000,
        min_confidence=0.30,  # low so trades fire
        **kwargs,
    )


class TestGetIV:
    """T4-1 through T4-4: _get_iv() priority chain."""

    def test_uses_stored_iv_when_column_present(self) -> None:
        bt = _make_backtester(iv_floor=0.20)
        hist = _make_ohlcv(60)
        hist["implied_vol"] = 0.35
        result = bt._get_iv(hist)
        assert result == pytest.approx(0.35, abs=0.001)

    def test_respects_iv_floor_when_stored_iv_is_low(self) -> None:
        bt = _make_backtester(iv_floor=0.20)
        hist = _make_ohlcv(60)
        hist["implied_vol"] = 0.10  # below floor
        result = bt._get_iv(hist)
        assert result >= 0.20

    def test_fallback_when_column_absent(self) -> None:
        bt = _make_backtester(iv_floor=0.20)
        hist = _make_ohlcv(60)
        assert "implied_vol" not in hist.columns
        result = bt._get_iv(hist)
        assert result >= bt._iv_floor
        assert result > 0

    def test_fallback_when_value_is_nan(self) -> None:
        bt = _make_backtester(iv_floor=0.20)
        hist = _make_ohlcv(60)
        hist["implied_vol"] = float("nan")
        result = bt._get_iv(hist)
        # NaN → synthetic fallback; must be >= floor and finite
        assert result >= bt._iv_floor
        assert np.isfinite(result)

    def test_vix_proxy_produces_plausible_iv(self) -> None:
        bt = _make_backtester(iv_floor=0.20)
        hist = _make_ohlcv(60)
        # Inject vix-like market context
        market_ctx = {"vix_close": 22.0}
        result = bt._get_iv(hist, market_context=market_ctx)
        # Should be at least the floor and something reasonable
        assert result >= 0.20
        assert result < 2.0


class TestIVEffect:
    """T4-5: Premium should increase with IV."""

    def test_iron_condor_premium_higher_in_high_iv_regime(self) -> None:
        hist = _make_ohlcv(90)
        underlying = hist["Close"].iloc[-1]

        bt_lo = _make_backtester(iv_floor=0.20)
        bt_hi = _make_backtester(iv_floor=0.45)

        hist_lo = hist.copy()
        hist_lo["implied_vol"] = 0.20
        hist_hi = hist.copy()
        hist_hi["implied_vol"] = 0.45

        iv_lo = bt_lo._get_iv(hist_lo)
        iv_hi = bt_hi._get_iv(hist_hi)

        t = 21 / 365
        r = 0.05

        from ait.backtesting.options_sim import black_scholes_price, find_strike_by_delta, OptionType

        call_strike_lo = find_strike_by_delta(underlying, t, iv_lo, 0.20, OptionType.CALL, r)
        call_strike_hi = find_strike_by_delta(underlying, t, iv_hi, 0.20, OptionType.CALL, r)

        credit_lo = black_scholes_price(underlying, call_strike_lo, t, r, iv_lo, OptionType.CALL)
        credit_hi = black_scholes_price(underlying, call_strike_hi, t, r, iv_hi, OptionType.CALL)

        assert credit_hi > credit_lo, "Higher IV should produce more premium"
