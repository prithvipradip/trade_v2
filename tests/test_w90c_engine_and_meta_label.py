"""W90c: the research engine's pricing / gate constants and the meta-label fallbacks are settings.

``backtest.engine_tuning`` (IV-skew coefficients 0.10 / 0.02, the 11-row realized-vol and IV-rank-rise look-backs, the 20-bar
minimum history), ``ml.meta_label_neutrals`` (the value of a missing meta-label input: RSI 50, vol 0.20, ... shared by live and
research), ``meta_label.{min_feature_coverage,max_artifact_age_days}`` (15 / 45), the meta-labeler's regime fallback (the
engine's regime thresholds, D3) and the IV-rank comparison's live floor / cap (the env contract). Defaults equal the old
literals; each setting is shown to MOVE behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.backtesting.pricing import OptionType, ic_leg_marks, skewed_leg_iv
from ait.config.settings import (
    EngineTuningConfig,
    MetaLabelConfig,
    MetaLabelNeutralsConfig,
    load_settings,
)
from tests.test_r20_research_validity import _engine_kwargs, synth_df, synth_features  # noqa: F401  (fixtures)

OLD_ENGINE = {"put_skew_coefficient": 0.10, "call_skew_coefficient": 0.02, "iv_rank_rise_lookback_rows": 11,
              "vol_gate_lookback_rows": 11, "min_history_bars": 20}
OLD_NEUTRALS = {"rsi": 50.0, "bb_position": 0.5, "volume_ratio": 1.0, "realized_vol": 0.20, "atr_pct": 0.01,
                "weekly_trend_aligned": 0.5, "volume_confirmation": 0.0, "macd_hist": 0.0, "px_vs_mid": 0.0, "cross": 0.5,
                "vix_level": 0.5, "iv_rank": 0.5, "sentiment_score": 0.0, "primary_confidence": 0.60, "hour_of_day": 10}


def _moved(**sections):
    """The loaded settings with ``{section: {field: value}}`` applied (``backtest.engine_tuning`` etc. by dotted key)."""
    s = load_settings()
    upd = {}
    for dotted, vals in sections.items():
        top, _, sub = dotted.partition(".")
        cur = getattr(s, top)
        upd[top] = cur.model_copy(update={sub: getattr(cur, sub).model_copy(update=vals)}) if sub else cur.model_copy(update=vals)
    return s.model_copy(update=upd)


def _freeze(monkeypatch, settings):
    import ait.config.settings as st
    monkeypatch.setattr(st, "_FROZEN", settings)


class TestSchemaAndYaml:
    def test_defaults_equal_the_old_literals_in_schema_and_files(self):
        s = load_settings()
        for name, old in OLD_ENGINE.items():
            assert getattr(EngineTuningConfig(), name) == old == getattr(s.backtest.engine_tuning, name), name
        for name, old in OLD_NEUTRALS.items():
            assert getattr(MetaLabelNeutralsConfig(), name) == old == getattr(s.ml.meta_label_neutrals, name), name
        assert (MetaLabelConfig().min_feature_coverage, MetaLabelConfig().max_artifact_age_days) == (15, 45)
        assert (s.meta_label.min_feature_coverage, s.meta_label.max_artifact_age_days) == (15, 45)

    def test_the_neutrals_are_shared_so_they_live_in_config_yaml_not_the_overlay(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        assert "meta_label_neutrals" in yaml.safe_load((root / "config.yaml").read_text())["ml"]
        assert "meta_label_neutrals" not in (yaml.safe_load((root / "config_research.yaml").read_text()).get("ml") or {})


class TestSkewCoefficients:
    def test_put_and_call_terms_are_the_arguments(self):
        args = (0.20, 95.0, 100.0)
        put = lambda c: skewed_leg_iv(*args[:3], OptionType.PUT, 1.0, 0.0, 0.01, c, 0.02)       # noqa: E731
        call = lambda c: skewed_leg_iv(0.20, 105.0, 100.0, OptionType.CALL, 1.0, 0.0, 0.01, 0.10, c)  # noqa: E731
        assert put(0.10) == pytest.approx(0.20 + np.log(100 / 95.0) * 0.10)
        assert put(0.30) > put(0.10) and call(0.10) > call(0.02)
        assert call(0.02) == pytest.approx(0.20 + np.log(105 / 100.0) * 0.02)
        # the call coefficient never touches a put and vice versa
        assert skewed_leg_iv(0.2, 95.0, 100.0, OptionType.PUT, 1.0, 0.0, 0.01, 0.1, 0.9) == put(0.10)

    def test_ic_leg_marks_price_with_them(self):
        base = ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, 0.04, 0.05, 0.2, 1.0, 0.0736, 0.05, 0.10, 0.02)
        steeper = ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, 0.04, 0.05, 0.2, 1.0, 0.0736, 0.05, 0.60, 0.02)
        assert steeper[1] > base[1] and steeper[3] > base[3]            # puts get dearer
        assert steeper[0] == base[0] and steeper[2] == base[2]           # calls unchanged
        wider = ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, 0.04, 0.05, 0.2, 1.0, 0.0736, 0.05, 0.10, 0.50)
        assert wider[0] > base[0] and wider[2] > base[2]                 # the call coefficient reaches the call legs
        assert wider[1] == base[1] and wider[3] == base[3]               # ... and only them

    def test_the_engine_prices_with_the_loaded_coefficients(self, synth_df, synth_features):
        pos = {"short_call_strike": 104.0, "short_put_strike": 96.0, "long_call_strike": 107.0,
               "long_put_strike": 93.0, "entry_iv": 0.2}
        kw = _engine_kwargs(synth_df, synth_features, entry_dte=14)
        base = Backtester(**kw)
        moved = Backtester(**kw, settings=_moved(**{"backtest.engine_tuning": {"put_skew_coefficient": 0.60}}))
        calls = Backtester(**kw, settings=_moved(**{"backtest.engine_tuning": {"call_skew_coefficient": 0.50}}))
        assert calls._get_leg_iv(0.21, 105.0, 100.0, OptionType.CALL) > base._get_leg_iv(0.21, 105.0, 100.0, OptionType.CALL)
        assert calls._get_leg_iv(0.21, 95.0, 100.0, OptionType.PUT) == base._get_leg_iv(0.21, 95.0, 100.0, OptionType.PUT)
        assert (base._eng.put_skew_coefficient, moved._eng.put_skew_coefficient) == (0.10, 0.60)
        assert moved._get_leg_iv(0.21, 95.0, 100.0, OptionType.PUT) > base._get_leg_iv(0.21, 95.0, 100.0, OptionType.PUT)
        assert moved._get_leg_iv(0.21, 105.0, 100.0, OptionType.CALL) == base._get_leg_iv(0.21, 105.0, 100.0, OptionType.CALL)
        assert moved._reprice_ic_legs(pos, 100.0, 3, None) != base._reprice_ic_legs(pos, 100.0, 3, None)

    def test_a_whole_run_prices_entries_with_them(self, synth_df, synth_features):
        """The entry build (not just the re-mark) uses the coefficients: the same run with steeper put skew books more
        credit for the short put, so the entry credits differ."""
        kw = _engine_kwargs(synth_df, synth_features)
        base = Backtester(**kw).run().trades
        put = Backtester(**kw, settings=_moved(**{"backtest.engine_tuning": {"put_skew_coefficient": 0.60}})).run().trades
        call = Backtester(**kw, settings=_moved(**{"backtest.engine_tuning": {"call_skew_coefficient": 0.60}})).run().trades
        credits = lambda ts: [round(t["mid_credit"], 6) for t in ts]       # noqa: E731
        assert base and credits(put) != credits(base) and credits(call) != credits(base)

    def test_the_replay_params_carry_them(self):
        from ait.backtesting.replay import PricingParams
        assert (PricingParams().put_skew_coefficient, PricingParams().call_skew_coefficient) == (0.10, 0.02)
        b = load_settings().backtest.model_copy(update={"engine_tuning": EngineTuningConfig(put_skew_coefficient=0.4,
                                                                                            call_skew_coefficient=0.07)})
        p = PricingParams.from_backtest_config(b)
        assert (p.put_skew_coefficient, p.call_skew_coefficient) == (0.4, 0.07)
        bare = PricingParams.from_backtest_config(SimpleNamespace(skew_factor=1.0, skew_slope_per_pct_otm=0.0, iv_floor=0.1))
        assert (bare.put_skew_coefficient, bare.call_skew_coefficient) == (0.10, 0.02)      # a stub config degrades to the schema

    def test_the_replay_model_mids_move_with_them(self):
        from ait.backtesting import replay as rp
        legs = [SimpleNamespace(strike=95.0, option_type=OptionType.PUT, side="short", right="P"),
                SimpleNamespace(strike=90.0, option_type=OptionType.PUT, side="long", right="P")]
        trade = SimpleNamespace(legs=legs, condor_roles=lambda: None)         # not a condor: the per-leg primitives
        a = rp.model_leg_mids(trade, 100.0, 0.2, [0.05, 0.05], rp.PricingParams())
        b = rp.model_leg_mids(trade, 100.0, 0.2, [0.05, 0.05], rp.PricingParams(put_skew_coefficient=0.8))
        assert b[0] > a[0] and b[1] > a[1]

    def test_the_replay_condor_path_uses_them_too(self):
        from ait.backtesting import replay as rp
        spec = [("short", "C", 104.0), ("short", "P", 96.0), ("long", "C", 107.0), ("long", "P", 93.0)]
        legs = [SimpleNamespace(strike=k, side=s, right=r, option_type=OptionType.CALL if r == "C" else OptionType.PUT)
                for s, r, k in spec]
        roles = {f"{s}_{'call' if r == 'C' else 'put'}": SimpleNamespace(strike=k) for s, r, k in spec}
        trade = SimpleNamespace(legs=legs, condor_roles=lambda: roles)
        a = rp.model_leg_mids(trade, 100.0, 0.2, [0.05] * 4, rp.PricingParams())
        put = rp.model_leg_mids(trade, 100.0, 0.2, [0.05] * 4, rp.PricingParams(put_skew_coefficient=0.8))
        call = rp.model_leg_mids(trade, 100.0, 0.2, [0.05] * 4, rp.PricingParams(call_skew_coefficient=0.8))
        assert put[1] > a[1] and put[0] == a[0] and call[0] > a[0] and call[1] == a[1]


class TestEngineGates:
    def _funnel(self, synth_df, synth_features, settings=None, **kw):
        extra = {"settings": settings} if settings is not None else {}
        return Backtester(**_engine_kwargs(synth_df, synth_features, **kw), **extra).run().entry_funnel

    def test_the_realized_vol_gate_lookback(self, synth_df, synth_features):
        base = self._funnel(synth_df, synth_features, max_entry_vol_annual=0.0001)
        assert base.get("vol_gate", 0) > 0                                    # the gate evaluates and vetoes
        off = self._funnel(synth_df, synth_features, max_entry_vol_annual=0.0001,
                           settings=_moved(**{"backtest.engine_tuning": {"vol_gate_lookback_rows": 252}}) )
        assert off.get("vol_gate", 0) < base["vol_gate"]                      # a longer window is reached later (fewer days)

    def test_the_iv_rank_rise_lookback(self, synth_df, synth_features):
        base = self._funnel(synth_df, synth_features, iv_rank_rise_threshold=-1.0, iv_rank_rise_gate=True)
        assert base.get("iv_rank_rise", 0) > 0
        long = self._funnel(synth_df, synth_features, iv_rank_rise_threshold=-1.0, iv_rank_rise_gate=True,
                            settings=_moved(**{"backtest.engine_tuning": {"iv_rank_rise_lookback_rows": 252}}))
        assert long.get("iv_rank_rise", 0) < base["iv_rank_rise"]

    def test_the_minimum_history(self, synth_df, synth_features):
        base = Backtester(**_engine_kwargs(synth_df, synth_features)).run().entry_funnel["days_evaluated"]
        late = Backtester(**_engine_kwargs(synth_df, synth_features),
                          settings=_moved(**{"backtest.engine_tuning": {"min_history_bars": 120}})).run().entry_funnel["days_evaluated"]
        assert base == len(synth_df) - 20 and late == len(synth_df) - 120

    def test_the_context_bars_still_win_when_larger(self, synth_df, synth_features):
        days = Backtester(**_engine_kwargs(synth_df, synth_features, context_bars=60)).run().entry_funnel["days_evaluated"]
        assert days == len(synth_df) - 60


class _CapturingMeta:
    is_trained = True

    def __init__(self):
        self.contexts = []

    def predict(self, ctx):
        self.contexts.append(dict(ctx))
        return None


class TestEngineMetaContext:
    def test_a_missing_feature_reads_the_neutral_value(self, synth_df, synth_features):
        # the engine reads the LAST feature row up to the decision day; blank the indicator columns so every one is missing
        blank = synth_features.drop(columns=[c for c in ("rsi_14", "rsi_7", "bb_position", "volume_sma_20_ratio",
                                                         "realized_vol_20", "atr_pct", "weekly_trend_aligned", "volume_confirmation",
                                                         "macd_hist", "price_vs_sma_20", "sma_10_20_cross", "vix_level")
                                             if c in synth_features.columns])
        cap = _CapturingMeta()
        Backtester(**_engine_kwargs(synth_df, blank), meta_labeler=cap).run()
        assert cap.contexts, "the meta gate never ran"
        ctx = cap.contexts[-1]
        for key, want in (("rsi_14", 50.0), ("rsi_7", 50.0), ("bb_position", 0.5), ("volume_sma_20_ratio", 1.0),
                          ("realized_vol_20", 0.20), ("atr_pct", 0.01), ("weekly_trend_aligned", 0.5),
                          ("volume_confirmation", 0.0), ("macd_hist", 0.0), ("price_vs_sma_20", 0.0),
                          ("sma_10_20_cross", 0.5), ("vix", 0.5), ("sentiment_score", 0.0), ("hour_of_day", 10)):
            assert ctx[key] == want, key
        cap2 = _CapturingMeta()
        s = _moved(**{"ml": {"meta_label_neutrals": MetaLabelNeutralsConfig(rsi=33.0, realized_vol=0.31, cross=0.9, hour_of_day=14)}})
        Backtester(**_engine_kwargs(synth_df, blank), meta_labeler=cap2, settings=s).run()
        c2 = cap2.contexts[-1]
        assert (c2["rsi_14"], c2["rsi_7"], c2["realized_vol_20"], c2["sma_10_20_cross"], c2["hour_of_day"]) == (33.0, 33.0, 0.31, 0.9, 14)


    def test_every_neutral_reaches_the_engine_context(self, synth_df, synth_features):
        """vix / iv_rank / sentiment / price-vs-SMA neutrals, one at a time, on features with those columns blanked."""
        drop = ["vix_level", "iv_rank", "price_vs_sma_20"]
        blank = synth_features.drop(columns=[c for c in drop if c in synth_features.columns])
        blank = blank.assign(vol_regime_expanding=1.0)                      # so the regime flags read the px-vs-SMA neutral
        nz = MetaLabelNeutralsConfig(vix_level=0.31, iv_rank=0.72, sentiment_score=0.25, px_vs_mid=0.5)
        cap = _CapturingMeta()
        Backtester(**_engine_kwargs(synth_df, blank), meta_labeler=cap,
                   settings=_moved(**{"ml": {"meta_label_neutrals": nz}})).run()
        ctx = cap.contexts[-1]
        assert (ctx["vix"], ctx["iv_rank"], ctx["sentiment_score"], ctx["price_vs_sma_20"]) == (0.31, 0.72, 0.25, 0.5)
        assert ctx["regime_trending_up"] == 1.0 and ctx["regime_high_vol"] == 0.0     # 0.5 > regime_up: a neutral moves the class


class TestMetaLabeler:
    def _trades(self):
        return [{"entry_date": "2024-02-05", "pnl": 10.0}, {"entry_date": "2024-02-06", "pnl": -5.0, "entry_regime": "range_bound"}]

    def _empty_features(self):
        return pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2024-01-02"]))

    def test_the_backtest_builder_uses_the_neutrals(self):
        from ait.ml.meta_label import MetaLabeler
        df = MetaLabeler().build_training_data_from_backtest(self._trades(), self._empty_features())
        row = df.iloc[0]
        assert (row["rsi_14"], row["realized_vol_20"], row["atr_pct"], row["primary_confidence"], row["hour_of_day"]) == (50.0, 0.2, 0.01, 0.6, 10)
        nz = MetaLabelNeutralsConfig(rsi=33.0, realized_vol=0.31, atr_pct=0.02, primary_confidence=0.7, hour_of_day=14, vix_level=0.4)
        row = MetaLabeler(neutrals=nz).build_training_data_from_backtest(self._trades(), self._empty_features()).iloc[0]
        assert (row["rsi_14"], row["rsi_7"], row["realized_vol_20"], row["atr_pct"], row["primary_confidence"],
                row["hour_of_day"], row["vix"]) == (33.0, 33.0, 0.31, 0.02, 0.7, 14, 0.4)

    def test_sentiment_neutral_reaches_the_backtest_builder(self):
        from ait.ml.meta_label import MetaLabeler
        row = MetaLabeler(neutrals=MetaLabelNeutralsConfig(sentiment_score=0.25)).build_training_data_from_backtest(
            self._trades(), self._empty_features()).iloc[0]
        assert row["sentiment_score"] == 0.25

    def test_the_journal_builder_uses_the_neutrals(self, tmp_path):
        """build_training_data (the live journal): a snapshot that lacks a key, and an unparseable entry time, read the neutrals."""
        import sqlite3

        from ait.ml.meta_label import MetaLabeler
        db = tmp_path / "state.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE trades (trade_id TEXT, realized_pnl REAL, entry_time TEXT, status TEXT, exit_reason_detailed TEXT)")
        con.execute("CREATE TABLE trade_context (trade_id TEXT, entry_direction TEXT, entry_confidence REAL, entry_regime TEXT, "
                    "entry_vix REAL, entry_iv_rank REAL, entry_sentiment_score REAL, entry_signals TEXT)")
        con.execute("INSERT INTO trades VALUES ('t1', 12.0, 'not-a-timestamp', 'closed', 'take_profit')")
        con.execute("INSERT INTO trade_context VALUES ('t1', 'neutral', 0.7, 'range_bound', 18.0, 0.4, 0.0, '{\"bb_position\": 0.9}')")
        con.commit()
        con.close()
        sm = SimpleNamespace(_db_path=str(db))
        base = MetaLabeler().build_training_data(sm).iloc[0]
        assert (base["rsi_14"], base["realized_vol_20"], base["sma_10_20_cross"], base["hour_of_day"], base["bb_position"]) == (50.0, 0.2, 0.5, 10, 0.9)
        nz = MetaLabelNeutralsConfig(rsi=33.0, realized_vol=0.31, cross=0.9, hour_of_day=14)
        moved = MetaLabeler(neutrals=nz).build_training_data(sm).iloc[0]
        assert (moved["rsi_14"], moved["rsi_7"], moved["realized_vol_20"], moved["sma_10_20_cross"], moved["hour_of_day"]) == (33.0, 33.0, 0.31, 0.9, 14)
        assert moved["bb_position"] == 0.9                                     # a key the snapshot has wins over the neutral

    def test_the_artifact_age_limit_is_the_setting(self, tmp_path, monkeypatch):
        import os
        from datetime import datetime, timedelta

        import ait.ml.meta_label as ml
        from structlog.testing import capture_logs
        monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
        art = tmp_path / "meta_label.pkl"
        art.write_bytes(b"not-a-real-pickle")
        old = (datetime.now() - timedelta(days=20)).timestamp()
        os.utime(art, (old, old))
        with capture_logs() as strict:
            assert ml.MetaLabeler(meta_label_config=MetaLabelConfig(max_artifact_age_days=10)).load_model() is False
        with capture_logs() as lax:
            ml.MetaLabeler(meta_label_config=MetaLabelConfig(max_artifact_age_days=30)).load_model()
        assert any(e["event"] == "meta_label_artifact_stale" for e in strict)
        assert not any(e["event"] == "meta_label_artifact_stale" for e in lax)

    def test_the_live_snapshot_neutrals_follow_the_loaded_settings(self, monkeypatch):
        from ait.ml.feature_spec import spec_from_config
        from ait.ml.meta_label import live_technical_features
        _freeze(monkeypatch, _moved(**{"ml": {"meta_label_neutrals": MetaLabelNeutralsConfig(rsi=41.0, realized_vol=0.27)}}))
        out = live_technical_features(pd.Series(dtype=float), spec_from_config(None).names)
        assert (out["rsi_14"], out["realized_vol_20"]) == (41.0, 0.27)

    def test_the_column_order_is_unchanged(self):
        from ait.ml.meta_label import META_FEATURES, MetaLabeler
        df = MetaLabeler().build_training_data_from_backtest(self._trades(), self._empty_features())
        assert list(df.columns) == [*META_FEATURES, "profitable"]

    def test_the_regime_fallback_follows_the_engines_thresholds(self):
        """D3: with no stored regime the meta-labeler classes a day like the engine (backtest.regime_*_threshold: -0.05 / 0.02),
        not by a private +-0.02."""
        from ait.ml.meta_label import MetaLabeler
        feats = pd.DataFrame({"vol_regime_expanding": [1.0], "price_vs_sma_20": [-0.03]}, index=pd.to_datetime(["2024-02-05"]))
        trades = [{"entry_date": "2024-02-05", "pnl": 1.0}]
        default = MetaLabeler().build_training_data_from_backtest(trades, feats).iloc[0]
        assert (default["regime_trending_down"], default["regime_high_vol"]) == (0.0, 1.0)          # -0.03 is inside (-0.05, 0.02)
        tight = MetaLabeler(regime_thresholds=(-0.02, 0.02)).build_training_data_from_backtest(trades, feats).iloc[0]
        assert (tight["regime_trending_down"], tight["regime_high_vol"]) == (1.0, 0.0)
        up = pd.DataFrame({"vol_regime_expanding": [1.0], "price_vs_sma_20": [0.03]}, index=pd.to_datetime(["2024-02-05"]))
        assert MetaLabeler().build_training_data_from_backtest(trades, up).iloc[0]["regime_trending_up"] == 1.0
        assert MetaLabeler(regime_thresholds=(-0.05, 0.05)).build_training_data_from_backtest(trades, up).iloc[0]["regime_trending_up"] == 0.0

    def test_the_regime_thresholds_come_from_the_loaded_backtest_section(self, monkeypatch):
        from ait.ml.meta_label import MetaLabeler
        _freeze(monkeypatch, _moved(backtest={"regime_down_threshold": -0.2, "regime_up_threshold": 0.15}))
        m = MetaLabeler()
        assert (m._regime_down, m._regime_up) == (-0.2, 0.15)

    def test_coverage_floor_and_artifact_age_follow_the_config(self, monkeypatch):
        from ait.ml.meta_label import MetaLabeler
        m = MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=7, max_artifact_age_days=9))
        assert (m._min_feature_coverage, m._max_artifact_age_days) == (7, 9)
        _freeze(monkeypatch, _moved(meta_label={"min_feature_coverage": 11, "max_artifact_age_days": 100}))
        d = MetaLabeler()
        assert (d._min_feature_coverage, d._max_artifact_age_days) == (11, 100)

    def test_the_arm_guard_uses_the_coverage_setting(self):
        from ait.ml.meta_label import MIN_TRADES_FOR_TRAINING, MetaLabeler
        rng = np.random.default_rng(1)
        n = MIN_TRADES_FOR_TRAINING + 20
        df = pd.DataFrame({"primary_confidence": rng.uniform(0.5, 0.9, n), "vix": rng.uniform(10, 30, n),
                           "iv_rank": rng.uniform(0, 1, n), "rsi_14": rng.uniform(20, 80, n),
                           "profitable": rng.integers(0, 2, n)})
        assert MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=15)).train(df) == {}
        assert MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=4)).train(df) != {}

    def test_live_snapshot_defaults_missing_columns_to_the_neutrals(self):
        """D2: the live entry snapshot used to read 0 for an absent column; it now reads the same neutral as everything else."""
        from ait.ml.feature_spec import spec_from_config
        from ait.ml.meta_label import live_technical_features
        names = spec_from_config(None).names
        out = live_technical_features(pd.Series({"atr_pct": 0.05}), names)
        assert (out["rsi_14"], out["rsi_7"], out["realized_vol_20"], out["sma_10_20_cross"], out["bb_position"]) == (50.0, 50.0, 0.2, 0.5, 0.5)
        assert out["atr_pct"] == 0.05                                       # a present column wins
        assert (out["macd_hist"], out["price_vs_sma_20"], out["volume_confirmation"]) == (0.0, 0.0, 0.0)
        moved = live_technical_features(pd.Series(dtype=float), names, MetaLabelNeutralsConfig(rsi=40.0, realized_vol=0.3))
        assert (moved["rsi_14"], moved["realized_vol_20"]) == (40.0, 0.3)


class TestIvRankCompare:
    def test_live_floor_and_cap_are_the_env_contract(self, monkeypatch):
        from ait.backtesting import iv_rank_compare as cmp
        assert (cmp.live_floor(), cmp.live_cap()) == (15.0, 85.0)
        monkeypatch.setenv("AIT_IRON_CONDOR_IV_FLOOR", "22")
        monkeypatch.setenv("AIT_IV_RANK_CAP", "70")
        assert (cmp.live_floor(), cmp.live_cap()) == (22.0, 70.0)
        r = pd.Series([10.0, 20.0, 30.0, 75.0, 90.0])
        assert list(cmp.live_level_blocks(r)) == [True, True, False, True, True]
        assert list(cmp.side(r)) == ["below", "below", "in", "above", "above"]
        assert list(cmp.live_level_blocks(r, floor=5.0, cap=95.0)) == [False] * 5       # explicit wins

    def test_the_rise_rows_follow_the_engine_look_back(self, monkeypatch):
        from ait.backtesting import iv_rank_compare as cmp
        assert cmp.rise_rows() == 10
        _freeze(monkeypatch, _moved(**{"backtest.engine_tuning": {"iv_rank_rise_lookback_rows": 6}}))
        assert cmp.rise_rows() == 5
        s = pd.Series(np.arange(20, dtype=float))
        assert cmp.rise(s).iloc[10] == 5.0 and cmp.rise(s, 3).iloc[10] == 3.0

    def test_the_default_threshold_is_the_backtest_setting(self, monkeypatch):
        from ait.backtesting import iv_rank_compare as cmp
        idx = pd.bdate_range("2024-01-02", periods=60)
        f = pd.Series(np.linspace(0.0, 0.5, 60), index=idx)             # a steady climb: 0.083 per 10 rows
        s = pd.Series(50.0, index=idx)
        _freeze(monkeypatch, _moved(backtest={"iv_rank_rise_threshold": 0.05}))
        assert cmp.veto_vs_live(f, s)["research_veto_days"] > 0
        _freeze(monkeypatch, _moved(backtest={"iv_rank_rise_threshold": 0.5}))
        assert cmp.veto_vs_live(f, s)["research_veto_days"] == 0
