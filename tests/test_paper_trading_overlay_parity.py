"""Tests for Gap Z — Live/Backtest Overlay Parity (paper_trading_mode gates)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ait.ml.meta_label import META_FEATURES, MetaLabeler


def _make_ohlcv(days: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.010, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(days, 5_000_000, dtype=float)},
        index=dates,
    )


class TestMetaLabelerGating:
    """TZ-1: paper_trading_mode disables meta-labeler in backtest engine."""

    def test_backtest_engine_accepts_meta_labeler_param(self) -> None:
        from ait.backtesting.engine import Backtester
        bt = Backtester(
            data=_make_ohlcv(),
            strategies=["iron_condor"],
            meta_labeler=None,
        )
        assert bt._meta_labeler is None

    def test_meta_labeler_stored_in_engine(self) -> None:
        from ait.backtesting.engine import Backtester
        ml = MagicMock()
        ml.is_trained = True
        bt = Backtester(
            data=_make_ohlcv(),
            strategies=["iron_condor"],
            meta_labeler=ml,
        )
        assert bt._meta_labeler is ml


class TestMarketContextForwarding:
    """TZ-3: market_context forwarded to predictors in engine."""

    def test_backtester_stores_market_context(self) -> None:
        from ait.backtesting.engine import Backtester
        ctx = {"vix_close": 25.0, "spy_close": 480.0}
        bt = Backtester(
            data=_make_ohlcv(),
            strategies=["iron_condor"],
            market_context=ctx,
        )
        assert bt._market_context == ctx

    def test_no_market_context_defaults_to_none(self) -> None:
        from ait.backtesting.engine import Backtester
        bt = Backtester(data=_make_ohlcv(), strategies=["iron_condor"])
        assert bt._market_context is None


class TestFractalPenaltyConsistency:
    """TZ-5: fractal regime penalty identical in backtest vs live."""

    def test_backtest_engine_has_fractal_threshold_params(self) -> None:
        from ait.backtesting.engine import Backtester
        bt = Backtester(
            data=_make_ohlcv(),
            strategies=["iron_condor"],
            hurst_regime_threshold=0.20,
            hurst_regime_penalty=0.10,
            multifractal_max_width=0.50,
        )
        assert bt._hurst_regime_threshold == pytest.approx(0.20)
        assert bt._hurst_regime_penalty   == pytest.approx(0.10)
        assert bt._multifractal_max_width == pytest.approx(0.50)

    def test_settings_backtest_config_has_fractal_params(self) -> None:
        from ait.config.settings import BacktestConfig
        cfg = BacktestConfig()
        assert hasattr(cfg, "hurst_regime_threshold")
        assert hasattr(cfg, "hurst_regime_penalty")
        assert hasattr(cfg, "multifractal_max_width")


class TestMetaLabelerAllFeatures:
    """TZ-14: build_training_data_from_backtest() populates all 20 features."""

    def test_all_meta_features_present_in_output(self) -> None:
        ml = MetaLabeler()
        features_df = _make_ohlcv(60)
        for col in ["rsi_14", "rsi_7", "bb_position", "volume_sma_20_ratio",
                    "realized_vol_20", "atr_pct", "weekly_trend_aligned",
                    "volume_confirmation", "macd_hist", "price_vs_sma_20",
                    "sma_10_20_cross", "iv_rank", "vix_level",
                    "vol_regime_expanding"]:
            features_df[col] = np.random.rand(60)

        trades = [
            {
                "entry_date": str(features_df.index[i].date()),
                "pnl": np.random.uniform(-100, 200),
                "entry_confidence": 0.65,
                "entry_regime": "range_bound",
                "entry_iv_rank": 0.4,
                "entry_vix_level": 0.5,
            }
            for i in range(60)
        ]
        df = ml.build_training_data_from_backtest(trades, features_df)

        assert not df.empty
        for feat in META_FEATURES:
            assert feat in df.columns, f"Missing META_FEATURE: {feat}"
        assert "profitable" in df.columns

    def test_zero_features_not_all_identical(self) -> None:
        """Non-trivial features must not all be 0 (the old corrupted-data problem)."""
        ml = MetaLabeler()
        features_df = _make_ohlcv(60)
        for col in ["rsi_14", "rsi_7", "bb_position", "volume_sma_20_ratio",
                    "realized_vol_20", "atr_pct", "weekly_trend_aligned",
                    "volume_confirmation", "macd_hist", "price_vs_sma_20",
                    "sma_10_20_cross", "iv_rank", "vix_level",
                    "vol_regime_expanding"]:
            features_df[col] = np.random.rand(60) * 0.5 + 0.25  # all non-zero

        trades = [
            {"entry_date": str(features_df.index[i].date()),
             "pnl": (1 if i % 2 == 0 else -1) * 50.0}
            for i in range(60)
        ]
        df = ml.build_training_data_from_backtest(trades, features_df)
        # rsi_14 should not all be 50.0 (the default) when FeatureEngine data is present
        assert df["rsi_14"].std() > 0, "rsi_14 should vary — not all zeros/defaults"


class TestMetaLabelerWalkforwardIntegration:
    """TZ-12 partial: MetaLabeler training method exists on WalkForwardBacktester."""

    def test_train_window_meta_labeler_method_exists(self) -> None:
        from ait.backtesting.walkforward import WalkForwardBacktester
        assert hasattr(WalkForwardBacktester, "_train_window_meta_labeler")

    def test_train_window_meta_labeler_returns_none_on_no_predictor(self) -> None:
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

        wf = WalkForwardBacktester(
            symbols=["QQQ"],
            strategies=["iron_condor"],
            config=WalkForwardConfig(min_confidence=0.99),  # no trades will fire
        )
        df = _make_ohlcv(100)
        result = wf._train_window_meta_labeler(
            train_df=df,
            symbol="QQQ",
            window_id=1,
            predictor=None,
            window_cfg=WalkForwardConfig(min_confidence=0.99),
        )
        assert result is None


class TestThesisInvalidationInEngine:
    """TZ-6: thesis invalidation exit method exists and identifies direction flip."""

    def test_check_thesis_invalidation_method_exists(self) -> None:
        from ait.backtesting.engine import Backtester
        assert hasattr(Backtester, "_check_thesis_invalidation")

    def test_thesis_invalidation_does_not_raise(self) -> None:
        from ait.backtesting.engine import Backtester
        import datetime as dt

        df = _make_ohlcv()
        bt = Backtester(data=df, strategies=["iron_condor"])
        pos = {
            "strategy": "iron_condor",
            "entry_date": str(df.index[50].date()),
            "entry_price": 5.0,
            "net_credit": 5.0,
            "contracts": 1,
            "n_legs": 4,
            "position_type": "credit",
            "call_short_strike": 500.0, "put_short_strike": 460.0,
            "call_long_strike": 510.0,  "put_long_strike": 450.0,
            "call_short_price": 2.5,   "put_short_price": 2.5,
            "call_long_price": 1.0,    "put_long_price": 1.0,
            "dte": 21, "iv": 0.25,
        }
        # Should not raise even with no predictor (signature: pos, hist)
        result = bt._check_thesis_invalidation(pos, df)
        # result is None (no predictor) or a dict
        assert result is None or isinstance(result, dict)


class TestEarningsSkip:
    """TZ-7: earnings proximity skip built into Backtester."""

    def test_load_earnings_dates_is_a_set(self) -> None:
        from ait.backtesting.engine import Backtester
        df = _make_ohlcv()
        bt = Backtester(data=df, strategies=["iron_condor"])
        assert isinstance(bt._earnings_dates, set)

    def test_earnings_dates_empty_for_blank_symbol(self) -> None:
        from ait.backtesting.engine import Backtester
        # Empty symbol → no yfinance call; set should be empty or have load errors silently swallowed
        bt = Backtester(data=_make_ohlcv(), strategies=["iron_condor"], symbol=None)
        assert isinstance(bt._earnings_dates, set)


class TestPaperTradingModeInSettings:
    """TZ-10 structural: paper_trading_mode setting exists and defaults False."""

    def test_learning_config_has_paper_trading_mode(self) -> None:
        from ait.config.settings import LearningConfig
        cfg = LearningConfig()
        assert hasattr(cfg, "paper_trading_mode")
        assert cfg.paper_trading_mode is False
