"""W78: per-horizon range models, the decision frame, and concurrency at the fill instant.

Real code paths on synthetic fixtures (never reads data/replay or historical.db):

  * ``ait.ml.range_spec``                 horizon_for / bucket_for / the live mirror
  * ``ait.backtesting.range_bank``        lazy, cached, RNG-safe per-horizon models
  * ``WalkForwardBacktester``             horizon from the study's own DTE; bucket routing;
                                          per-bucket artifact directory
  * ``StrategyOptimizer``                 the per-trial predictor hook
  * ``Backtester`` (realistic)            decisions use the PREVIOUS session; a position that
                                          was still alive at the fill counts toward the limit
"""
from __future__ import annotations

import ast
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.engine as engine_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.range_bank import RangeModelBank
from ait.backtesting.walkforward import WalkForwardBacktester
from ait.config.settings import OptionsConfig, load_settings
from ait.execution.exit_policy import EXPIRY_APPROACHING_DTE
from ait.ml import range_spec as rs
from ait.optimization.optimizer import StrategyOptimizer

import tests.test_w74_engine_realism as W74


# --------------------------------------------------------------------------
# range_spec (pure)
# --------------------------------------------------------------------------

class TestRangeSpec:
    def test_close_rules_apply_by_default(self):
        assert rs.horizon_for(14) == 14 - int(EXPIRY_APPROACHING_DTE)
        assert rs.horizon_for(30) == 25
        assert rs.horizon_for(6) == 1            # floor: never zero bars
        assert rs.horizon_for(3) == 1

    def test_guard_exempt_entries_live_to_expiry(self):
        assert rs.horizon_for(3, guard_dte=5) == 3
        assert rs.horizon_for(5, guard_dte=5) == 5
        assert rs.horizon_for(6, guard_dte=5) == 1           # above the guard: the rules apply
        assert rs.horizon_for(0, guard_dte=0) == 1           # 0 DTE: one bar is the floor

    def test_hold_fraction_rounds_up_so_the_model_never_asks_less(self):
        assert rs.horizon_for(10, guard_dte=10, hold_fraction=0.25) == 3     # 2.5 -> 3, not 2
        assert rs.horizon_for(7, guard_dte=7, hold_fraction=0.3) == 3        # 2.1 -> 3

    def test_hold_fraction_cuts_the_life_and_replaces_the_day_cap(self):
        assert rs.horizon_for(10, guard_dte=10, hold_fraction=0.5) == 5
        assert rs.horizon_for(10, guard_dte=10, hold_fraction=0.01) == 1
        # with a fraction set, max_hold_days is not applied (it is replaced, as in the engine)
        assert rs.horizon_for(10, guard_dte=10, hold_fraction=1.0, max_hold_days=2) == 10

    def test_max_hold_days_caps_never_extends(self):
        assert rs.horizon_for(30, max_hold_days=4) == 4
        assert rs.horizon_for(8, max_hold_days=30) == 3

    def test_identical_to_the_pre_w78_formula_over_the_whole_grid(self):
        for dte in range(0, 61):
            for cap in (1, 4, 9, 30, 120):
                old = min(cap, max(1, dte - int(EXPIRY_APPROACHING_DTE)))
                assert rs.horizon_for(dte, max_hold_days=cap) == old, (dte, cap)

    def test_bucket_for_rounds_up_and_caps(self):
        assert rs.bucket_for(1) == 1 and rs.bucket_for(4) == 5 and rs.bucket_for(8) == 9
        assert rs.bucket_for(10) == 14 and rs.bucket_for(22) == 30
        assert rs.bucket_for(999) == rs.RANGE_HORIZON_BUCKETS[-1]
        prev = 0
        for h in range(1, 60):
            b = rs.bucket_for(h)
            assert b >= min(h, rs.RANGE_HORIZON_BUCKETS[-1]) and b >= prev   # never easier, monotone
            prev = b

    def test_shipped_horizon_is_a_bucket_so_the_default_study_is_unchanged(self):
        live = rs.live_range_spec()[1]
        assert live == rs.horizon_for(int(load_settings().options.dte_range[0]))
        assert live in rs.RANGE_HORIZON_BUCKETS and rs.bucket_for(live) == live

    def test_live_spec_is_the_shared_function(self):
        s = SimpleNamespace(options=SimpleNamespace(dte_range=[21, 45]))
        assert rs.live_range_spec(s) == (rs.LIVE_RANGE_THRESHOLD_PCT, rs.horizon_for(21))

    def test_module_stays_import_light(self):
        tree = ast.parse(inspect.getsource(rs))
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert top <= {"__future__", "math"}


# --------------------------------------------------------------------------
# RangeModelBank
# --------------------------------------------------------------------------

class TestRangeModelBank:
    def test_trains_once_per_horizon_and_caches(self):
        calls = []
        bank = RangeModelBank(lambda h: (calls.append(h), (f"p{h}", "ok", 0.05))[1], lambda e: 9)
        assert bank.get(3) == ("p3", "ok", 0.05)
        assert bank.get(3) == ("p3", "ok", 0.05)
        assert bank.get(5)[0] == "p5"
        assert calls == [3, 5]
        assert bank.trained_horizons == (3, 5)

    def test_seeded_model_is_never_retrained(self):
        calls = []
        bank = RangeModelBank(lambda h: calls.append(h) or ("x", "ok", 0.1), lambda e: 9)
        bank.seed(9, ("base", "ok", 0.07))
        assert bank.get(9)[0] == "base" and calls == []

    def test_global_rng_state_is_restored_around_training(self):
        def noisy(h):
            np.random.rand(25)
            random.random()
            return ("p", "ok", 0.05)
        bank = RangeModelBank(noisy, lambda e: 3)
        np.random.seed(1234)
        random.seed(1234)
        np_before, py_before = np.random.get_state(), random.getstate()
        bank.get(3)
        np_after, py_after = np.random.get_state(), random.getstate()
        assert py_after == py_before
        assert np_after[0] == np_before[0] and (np_after[1] == np_before[1]).all()
        assert np_after[2:] == np_before[2:]

    def test_failed_training_is_cached_not_retried(self):
        n = {"calls": 0}

        def boom(h):
            n["calls"] += 1
            raise ValueError("no")
        bank = RangeModelBank(boom, lambda e: 3)
        first = bank.get(3)
        assert first[0] is None and first[1] == "train_exception:ValueError"
        assert bank.get(3) == first and n["calls"] == 1

    def test_none_result_becomes_a_skipped_status(self):
        bank = RangeModelBank(lambda h: None, lambda e: 3)
        assert bank.get(3) == (None, "skipped", 0.05)

    def test_lookup_by_entry_dte_goes_through_the_bucket_function(self):
        seen = []
        bank = RangeModelBank(lambda h: (f"p{h}", "ok", 0.05),
                              lambda e, g, f: (seen.append(e), 5 if e == 7 else 9)[1])
        assert bank.predictor_for(7) == "p5"
        assert bank.predictor_for(None) == "p9"
        assert seen == [7, None]


# --------------------------------------------------------------------------
# walk-forward horizon + routing (partial engine, as test_r21b does)
# --------------------------------------------------------------------------

def _wf(**cfg):
    wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
    wf._settings = load_settings()
    base = dict(max_hold_days=30, train_window_models=True)
    base.update(cfg)
    wf._config = SimpleNamespace(**base)
    return wf


class TestWalkForwardHorizon:
    def test_default_is_the_shipped_horizon(self):
        assert _wf()._range_label_horizon() == rs.live_range_spec()[1]

    def test_study_entry_dte_is_honoured(self):
        # pre-W78 this read options.dte_range[0] and ignored the study's own DTE
        assert _wf(entry_dte=18)._range_label_horizon() == 18 - int(EXPIRY_APPROACHING_DTE)
        assert _wf(entry_dte=3, expiry_guard_dte=5)._range_label_horizon() == 3

    def test_explicit_dte_argument_beats_the_config(self):
        assert _wf(entry_dte=18)._range_label_horizon(10) == 5

    def test_partial_config_without_new_fields_still_works(self):
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)     # the test_r21b form
        wf._settings = load_settings()
        wf._config = SimpleNamespace(max_hold_days=30)
        assert isinstance(wf._range_label_horizon(), int)
        assert wf._range_label_horizon() == rs.live_range_spec()[1]

    def test_base_dte_keeps_its_exact_horizon_others_round_up(self):
        wf = _wf(max_hold_days=4)                       # base horizon 4 (the cap), not a bucket
        assert wf._range_bucket() == 4
        assert wf._range_bucket(14) == 4                # same horizon -> the base model
        wf = _wf()
        base = wf._range_bucket()
        assert base == 9 and wf._range_bucket(14) == 9
        assert wf._range_bucket(10) == 5                # h=5
        assert wf._range_bucket(30) == 30               # h=25 -> 30
        assert wf._range_bucket(45) == 30               # h=40 -> capped at the last bucket

    def test_oos_gate_follows_the_window_configs_own_entry_dte(self):
        wf = _wf()
        made = []
        bank = RangeModelBank(lambda h: (made.append(h), (f"model{h}", "ok", 0.05))[1], wf._range_bucket)
        bank.seed(wf._range_bucket(), ("base", "ok", 0.06))
        # no DTE on the window config -> the base model, nothing trained
        assert wf._pick_range_model(bank, SimpleNamespace()) == (9, ("base", "ok", 0.06))
        assert wf._pick_range_model(bank, SimpleNamespace(entry_dte=None))[0] == 9
        # a window whose best trial chose 30 DTE (reachable 25 bars -> bucket 30) gets that model
        bucket, res = wf._pick_range_model(bank, SimpleNamespace(entry_dte=30))
        assert (bucket, res[0]) == (30, "model30") and made == [30]

    def test_non_base_bucket_trains_with_its_own_artifact_directory(self):
        wf = _wf()
        seen = {}
        wf._train_window_range_model = lambda df, sym, wid, max_hold_days=30, model_subdir=None: (
            seen.update(h=max_hold_days, sub=model_subdir) or ("p", "ok", 0.05))
        assert wf._train_range_bucket(pd.DataFrame(), "QQQ", 1, 5) == ("p", "ok", 0.05)
        assert seen == {"h": 5, "sub": "h5"}

    def test_disabled_window_models_never_train_extra_buckets(self):
        wf = _wf(train_window_models=False)
        wf._train_window_range_model = lambda *a, **k: pytest.fail("must not train")
        assert wf._train_range_bucket(pd.DataFrame(), "QQQ", 1, 5) == (None, "disabled_by_config", 0.05)

    def test_inprocess_training_saves_non_base_horizons_in_a_subdirectory(self, monkeypatch):
        import ait.ml.range_predictor as rp_mod
        made = []

        class Fake:
            is_trained = True

            def __init__(self, **kw):
                made.append(kw)

            def train(self, *a, **k):
                return {"xgb": 0.6}
        monkeypatch.setattr(rp_mod, "RangePredictor", Fake)
        wf = _wf()
        wf._make_intraday_store = lambda: None
        wf._train_window_range_model_inprocess(pd.DataFrame(), "QQQ", 1, 9, 0.05)
        wf._train_window_range_model_inprocess(pd.DataFrame(), "QQQ", 1, 5, 0.04, model_subdir="h5")
        assert made[0]["model_dir"] == rp_mod.RESEARCH_MODEL_DIR
        assert made[1]["model_dir"] == rp_mod.RESEARCH_MODEL_DIR / "h5"
        assert made[0]["horizon_days"] == 9 and made[1]["horizon_days"] == 5


# --------------------------------------------------------------------------
# optimizer hook
# --------------------------------------------------------------------------

class TestOptimizerProvider:
    def _run(self, monkeypatch, **kw):
        sink = []

        def factory(**kwargs):
            sink.append(kwargs)
            return SimpleNamespace(run=lambda: SimpleNamespace(trades=[]))
        monkeypatch.setattr(engine_mod, "Backtester", factory)
        idx = pd.bdate_range("2024-01-02", periods=120)
        df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)
        opt = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1, **kw)
        opt._data = {"SPY": df}
        opt._run_backtest({})
        assert len(sink) == 1
        return sink[0]

    def test_without_a_provider_every_trial_gets_the_single_model(self, monkeypatch):
        got = self._run(monkeypatch, range_predictor="single")
        assert got["range_predictor"] == "single"

    def test_provider_supplies_the_trial_model_and_is_asked_per_trial(self, monkeypatch):
        asked = []
        got = self._run(monkeypatch, range_predictor="single",
                        range_predictor_provider=lambda e, g, f: (asked.append((e, g, f)), "per-trial")[1])
        assert got["range_predictor"] == "per-trial"
        assert asked == [(None, None, None)]   # nothing searched or set -> the base bucket


# --------------------------------------------------------------------------
# engine: decision frame
# --------------------------------------------------------------------------

def _engine(df, **kw):
    base = dict(intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                entry_window_start_et="10:30", entry_window_end_et="15:30",
                scan_interval_minutes=60, engine_model="realistic", entry_dte=1,
                expiry_guard_dte=1, max_contracts_per_trade=1)
    base.update(kw)
    return W74._bt(df, **base)


class TestDecisionFrame:
    def _spy_get_direction(self, monkeypatch):
        calls = []
        orig = Backtester._get_direction

        def spy(self, hist, market_context=None):
            i = max(20, self._context_bars) + self._funnel["days_evaluated"] - 1
            calls.append((pd.Timestamp(hist.index[-1]), self._data.index[i], self._data.index[i - 1]))
            return orig(self, hist, market_context=market_context)
        monkeypatch.setattr(Backtester, "_get_direction", spy)
        return calls

    def test_realistic_intraday_decides_on_the_previous_session(self, monkeypatch):
        calls = self._spy_get_direction(monkeypatch)
        _engine(W74._gap_market()).run()
        assert len(calls) > 20
        assert all(last == prev and last < today for last, today, prev in calls)

    def test_legacy_and_daily_close_realistic_still_see_the_decision_day(self, monkeypatch):
        calls = self._spy_get_direction(monkeypatch)
        df = W74._gap_market()
        W74._bt(df, intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                entry_window_start_et="10:30", entry_window_end_et="15:30",
                scan_interval_minutes=60, entry_dte=1).run()                     # legacy
        assert calls and all(last == today for last, today, _ in calls)
        calls.clear()
        W74._bt(df, condor_any_direction=True, engine_model="realistic", entry_dte=1,
                expiry_guard_dte=1).run()                                         # no intraday store
        assert calls and all(last == today for last, today, _ in calls)

    def test_downstream_gates_read_the_same_frame(self, monkeypatch):
        seen = {"select": [], "range": []}
        orig_sel = Backtester._select_strategy

        def spy_sel(self, direction, hist, confidence, features_df):
            i = max(20, self._context_bars) + self._funnel["days_evaluated"] - 1
            seen["select"].append(pd.Timestamp(hist.index[-1]) == self._data.index[i - 1])
            return orig_sel(self, direction, hist, confidence, features_df)
        monkeypatch.setattr(Backtester, "_select_strategy", spy_sel)

        class RP:
            def predict(inner, hist, **k):
                seen["range"].append(pd.Timestamp(hist.index[-1]))
                return SimpleNamespace(probability_in_range=0.9)
        df = W74._gap_market()
        bt = _engine(df, range_predictor=RP(), range_min_confidence=0.5, gates_mode="enforce")
        bt.run()
        assert seen["select"] and all(seen["select"])
        assert seen["range"] and max(seen["range"]) < df.index[-1]

    def test_iv_rank_gate_uses_the_previous_sessions_rank(self, monkeypatch):
        df = W74._gap_market()
        poison = df.index[40]                              # one session whose rank breaches the cap
        rank = pd.Series(50.0, index=df.index)
        rank.loc[poison] = 99.0
        rejected = []
        orig = Backtester._reject

        def spy(self, reason):
            i = max(20, self._context_bars) + self._funnel["days_evaluated"] - 1
            if reason == "iv_rank_cap":
                rejected.append(self._data.index[i])
            return orig(self, reason)
        monkeypatch.setattr(Backtester, "_reject", spy)
        _engine(df, iv_rank_gate=True, iv_rank_series=rank).run()
        nxt = df.index[41]
        assert nxt in rejected and poison not in rejected

    def test_no_decision_input_reads_hist_after_the_frame_is_set(self):
        src = inspect.getsource(engine_mod.Backtester.run)
        tree = ast.parse("class _C:\n" + "\n".join("    " + l for l in src.splitlines()))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
        frame_line = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
                          and any(getattr(t, "id", "") == "hist_dec" for t in n.targets))
        # `hist` may still be read for PRICING (build position, intraday partials); a decision
        # read is one of these specific calls/subscripts, which must use hist_dec.
        bad = []
        for n in ast.walk(fn):
            if getattr(n, "lineno", 0) <= frame_line:
                continue
            txt = ast.unparse(n) if isinstance(n, (ast.Call, ast.Subscript)) else ""
            if ("_get_direction(hist" in txt or "_select_strategy(" in txt or ".predict(" in txt
                    or "recent_close" in txt or "_close_series" in txt) and "hist," in txt.replace("hist_dec,", ""):
                bad.append((n.lineno, txt[:80]))
            if isinstance(n, ast.Subscript) and ast.unparse(n) == "hist['Close']":
                bad.append((n.lineno, "hist['Close']"))
        assert not bad, bad


# --------------------------------------------------------------------------
# engine: concurrency at the fill instant
# --------------------------------------------------------------------------

def _overlaps(trades):
    def ts(v):
        t = pd.Timestamp(str(v))
        if t.tzinfo is None:
            t = t.tz_localize("America/New_York")
        if len(str(v)) <= 10:
            t = t.normalize() + pd.Timedelta(hours=16)
        return t
    iv = sorted((ts(t.get("fill_time") or t.get("entry_time") or t["entry_date"]),
                 ts(t.get("exit_time") or t["exit_date"])) for t in trades)
    return sum(1 for i in range(1, len(iv)) if iv[i][0] < max(x for _, x in iv[:i]))


class TestConcurrencyAtFill:
    def _run(self, dte, **kw):
        df = W74._calm_market()
        bt = W74._bt(df, intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                     entry_window_start_et="10:30", entry_window_end_et="15:30",
                     scan_interval_minutes=60, engine_model="realistic", entry_dte=dte,
                     expiry_guard_dte=dte, max_concurrent_positions=1, max_contracts_per_trade=1,
                     credit_tp_ladder=W74._no_take_profit(), **kw)
        return bt, bt.run()

    @pytest.mark.parametrize("dte", [1, 3, 7])
    def test_no_entry_opens_while_another_position_is_alive(self, dte):
        bt, res = self._run(dte)
        assert len(res.trades) >= 10
        assert _overlaps(res.trades) == 0

    def test_the_blocked_entries_are_counted_under_their_own_reason(self):
        bt, _ = self._run(3)
        assert bt._funnel.get("max_concurrent_intraday", 0) > 0

    def test_a_higher_limit_lets_the_same_days_trade(self):
        _, one = self._run(3)
        df = W74._calm_market()
        three = W74._bt(df, intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                        entry_window_start_et="10:30", entry_window_end_et="15:30",
                        scan_interval_minutes=60, engine_model="realistic", entry_dte=3,
                        expiry_guard_dte=3, max_concurrent_positions=3, max_contracts_per_trade=1,
                        credit_tp_ladder=W74._no_take_profit()).run()
        assert len(three.trades) >= len(one.trades)

    def test_legacy_engine_keeps_its_ordering(self):
        # a market whose sessions end with a ramp, so held positions touch / stop out LATE in
        # the day; in the legacy engine that late exit frees the slot for a 10:30 entry
        df = W74._gap_market()
        bt = W74._bt(df, intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                     entry_window_start_et="10:30", entry_window_end_et="15:30",
                     scan_interval_minutes=60, entry_dte=10, expiry_guard_dte=10,
                     max_concurrent_positions=1, max_contracts_per_trade=1,
                     credit_tp_ladder=W74._no_take_profit(), credit_loss_limit_mult=1.0)
        res = bt.run()
        assert bt._funnel.get("max_concurrent_intraday", 0) == 0
        # the scenario is real: the legacy exit-before-entry ordering DOES overlap here, so a
        # check that leaked into legacy would have changed its trades
        assert _overlaps(res.trades) > 0

    def test_exit_instant_maps_date_only_exits_to_the_session_close(self):
        bt = Backtester.__new__(Backtester)
        a = bt._exit_instant({"exit_date": "2024-03-04", "exit_time": "2024-03-04"})
        assert (a.hour, a.minute) == (16, 0)
        b = bt._exit_instant({"exit_date": "2024-03-04", "exit_time": "2024-03-04T19:00:00+00:00"})
        assert (b.hour, b.minute) == (14, 0)           # 19:00Z = 14:00 ET (EST)
        assert bt._exit_instant({"exit_date": "2024-07-03"}).hour == 13   # day before July 4: 13:00 early close
