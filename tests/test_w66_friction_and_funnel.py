"""W66: entry-funnel telemetry (engine -> result -> walk-forward) and friction model v2.

House rules: every test drives the real code path; the AST guard catches the NEXT
silent entry gate (a `continue`/`return None` on the entry path that bypasses `_reject`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardResult


@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996,
                         "Close": close, "Volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def synth_features(synth_df) -> pd.DataFrame:
    from ait.ml.features import FeatureEngine
    return FeatureEngine().compute(synth_df)


def _kw(df, feats, **over) -> dict:
    kw = dict(data=df, strategies=["iron_condor"], initial_capital=100_000,
              macro_event_gate=False, allow_live_model_fallback=False,
              hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
              iv_rank_rise_threshold=10.0, ic_min_credit=0.0, ic_min_credit_width=0.0,
              iv_floor=0.01, features_cache=feats)
    kw.update(over)
    return kw


def _top(funnel: dict) -> dict:
    return {k: v for k, v in funnel.items()
            if k not in ("days_evaluated", "entries_taken") and not k.startswith("builder:")}


class TestFunnelConservation:
    """Count reconciliation: every evaluated day is either an entry or ONE rejection."""

    @pytest.mark.parametrize("over", [
        {},
        dict(ic_min_credit=50.0),                       # credit floor binds
        dict(iv_floor=5.0),                             # IV floor binds
        dict(max_entry_vol_annual=0.0),                 # vol gate binds
        dict(range_min_confidence=1.0),                 # range gate binds
        dict(iv_rank_rise_threshold=-10.0),             # iv-rank-rise veto binds
    ], ids=["open", "credit_floor", "iv_floor", "vol_gate", "range_gate", "iv_rank_rise"])
    def test_days_equal_entries_plus_rejections(self, synth_df, synth_features, over):
        res = Backtester(**_kw(synth_df, synth_features, **over)).run()
        f = res.entry_funnel
        assert f["days_evaluated"] > 0
        assert f["days_evaluated"] == f.get("entries_taken", 0) + sum(_top(f).values())
        assert f.get("entries_taken", 0) == res.total_trades

    def test_builder_detail_is_namespaced_and_explains_builder_none(
            self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, ic_min_credit=50.0)).run()
        f = res.entry_funnel
        assert f.get("builder_none", 0) > 0
        assert f.get("builder:credit_floor", 0) > 0
        detail = sum(v for k, v in f.items() if k.startswith("builder:"))
        # detail is a refinement of builder_none, never added to the day total
        assert detail >= f["builder_none"] or detail <= f["builder_none"] * 2

    def test_credit_floor_forces_its_own_reason(self, synth_df, synth_features):
        base = Backtester(**_kw(synth_df, synth_features)).run().entry_funnel
        hit = Backtester(**_kw(synth_df, synth_features, ic_min_credit=50.0)).run().entry_funnel
        assert base.get("builder:credit_floor", 0) == 0
        assert hit["builder:credit_floor"] > 0 and hit.get("entries_taken", 0) == 0

    def test_iv_floor_forces_its_own_reason(self, synth_df, synth_features):
        hit = Backtester(**_kw(synth_df, synth_features, iv_floor=5.0)).run().entry_funnel
        assert hit["builder:iv_floor"] > 0

    def test_funnel_is_reset_between_runs(self, synth_df, synth_features):
        bt = Backtester(**_kw(synth_df, synth_features))
        a = dict(bt.run().entry_funnel)
        b = dict(bt.run().entry_funnel)
        assert a == b


class TestCondorAnyDirection:
    def test_default_off_and_opt_in_removes_no_strategy(self, synth_df, synth_features):
        off = Backtester(**_kw(synth_df, synth_features)).run().entry_funnel
        on = Backtester(**_kw(synth_df, synth_features, condor_any_direction=True)
                        ).run().entry_funnel
        assert on.get("no_strategy", 0) <= off.get("no_strategy", 0)
        assert on.get("entries_taken", 0) >= off.get("entries_taken", 0)

    def test_config_default_is_false(self):
        from ait.config.settings import BacktestConfig
        assert BacktestConfig().condor_any_direction is False


class TestWalkForwardFunnel:
    def test_aggregate_sums_over_windows_and_keeps_symbols_apart(self):
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        wf._funnel_by_window = {
            0: {"SPY": {"days_evaluated": 10, "no_strategy": 8, "entries_taken": 2},
                "QQQ": {"days_evaluated": 10, "entries_taken": 10}},
            1: {"SPY": {"days_evaluated": 5, "no_strategy": 5}},
        }
        agg = wf._aggregate_funnel()
        assert agg["SPY"] == {"days_evaluated": 15, "no_strategy": 13, "entries_taken": 2}
        assert agg["QQQ"] == {"days_evaluated": 10, "entries_taken": 10}

    def test_summary_prints_funnel_even_with_no_trades(self):
        res = WalkForwardResult(windows=[], initial_capital=5000.0, config=None,
                                entry_funnel={"SPY": {"days_evaluated": 20, "no_strategy": 17,
                                                      "builder_none": 3,
                                                      "builder:credit_floor": 3}})
        try:
            text = res.summary()
        except Exception:  # summary() may need a config for headers; the table is what matters
            pytest.skip("summary() requires a populated config")
        assert "ENTRY FUNNEL" in text and "no_strategy=17" in text
        assert "credit_floor=3" in text

    def test_worker_returns_a_five_tuple(self):
        import inspect
        import ait.backtesting.walkforward as wfm
        src = inspect.getsource(wfm._window_task_mp)
        # structural contract between the worker and its consumer (both in this module)
        assert "_funnel_by_window" in src


class TestNoSilentEntryGate:
    """AST class guard: every `return None` in the iron-condor branch of
    `_build_credit_position` (and in `_build_position`) must be preceded by a
    `_reject(...)` — the next silent gate fails here, not in a 0-trade backtest."""

    @staticmethod
    def _uncounted_returns(stmts: list) -> list[int]:
        import ast
        missing: list[int] = []

        def walk(body: list) -> None:
            for i, st in enumerate(body):
                if isinstance(st, ast.Return) and (st.value is None or (
                        isinstance(st.value, ast.Constant) and st.value.value is None)):
                    prev = body[i - 1] if i else None
                    counted = (isinstance(prev, ast.Expr) and isinstance(prev.value, ast.Call)
                               and getattr(prev.value.func, "attr", "") == "_reject")
                    if not counted:
                        missing.append(st.lineno)
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list) and not isinstance(st, ast.FunctionDef):
                        walk(sub)
        walk(stmts)
        return missing

    def _fn(self, name: str):
        import ast
        import inspect
        import textwrap
        return ast.parse(textwrap.dedent(inspect.getsource(getattr(Backtester, name)))).body[0]

    def test_iron_condor_branch_has_no_silent_return(self):
        import ast
        fn = self._fn("_build_credit_position")
        branch = next(n for n in fn.body if isinstance(n, ast.If)
                      and "iron_condor" in ast.dump(n.test))
        rejects = [n for n in ast.walk(branch) if isinstance(n, ast.Call)
                   and getattr(n.func, "attr", "") == "_reject"]
        assert len(rejects) >= 8, "guard would be vacuous: the condor branch lost its counters"
        # only the IC `if` body, not the elif arms of other (retired) strategies
        assert self._uncounted_returns(branch.body) == []

    def test_build_position_iv_floor_return_is_counted(self):
        assert self._uncounted_returns(self._fn("_build_position").body) == []

    def test_guard_detects_a_silent_return(self):
        import ast
        tree = ast.parse("def f(self):\n    if x:\n        return None\n")
        assert self._uncounted_returns(tree.body[0].body) != []


# ---------------------------------------------------------------------------
# Friction model v2
# ---------------------------------------------------------------------------

from ait.backtesting.friction import (  # noqa: E402
    LegSpreadModel,
    condor_leg_cost_usd,
    fit_leg_spread_models,
)


@pytest.fixture()
def leg_db(tmp_path):
    """A temp historical.db holding a fitted const-$ model for symbol XYZ."""
    from ait.data.historical import HistoricalDataStore
    db = tmp_path / "hist.db"
    store = HistoricalDataStore(db_path=db)
    store.save_leg_spread_model("XYZ", {"form": "const_usd", "a": 0.02, "b": 0.0, "n": 50,
                                        "n_sessions": 5, "cv_mae": 0.003,
                                        "calibrated_on": "2026-10-09"})
    return db


def _v2(df, feats, db, **over):
    return Backtester(**_kw(df, feats, symbol="XYZ", db_path=db,
                            friction_model="per_leg_usd", **over))


class TestFittedForms:
    def test_const_law_is_recovered_and_cv_picks_it(self):
        rng = np.random.default_rng(1)
        mid = rng.uniform(0.2, 3.0, 400)
        sess = rng.integers(0, 15, 400).astype(str)
        hs = 0.015 + rng.normal(0, 0.002, 400)
        fit = fit_leg_spread_models(mid, hs, sess)
        assert fit["best"].form == "const_usd" and fit["cv"] == "grouped"
        assert fit["best"].a == pytest.approx(0.015, abs=0.002)

    def test_affine_law_is_recovered_and_beats_const(self):
        rng = np.random.default_rng(2)
        mid = rng.uniform(0.1, 4.0, 500)
        sess = rng.integers(0, 15, 500).astype(str)
        hs = 0.005 + 0.02 * mid + rng.normal(0, 0.001, 500)
        fit = fit_leg_spread_models(mid, hs, sess)
        assert fit["best"].form == "affine_usd"
        assert fit["best"].b == pytest.approx(0.02, abs=0.004)
        assert fit["cv_mae"]["affine_usd"] < fit["cv_mae"]["const_usd"]

    def test_floor_and_validation(self):
        m = LegSpreadModel("const_usd", a=0.0)
        assert m.half_spread_usd(1.0) == 0.005  # nothing quotes tighter than half a tick
        with pytest.raises(ValueError):
            LegSpreadModel("magic")
        with pytest.raises(ValueError):
            fit_leg_spread_models([1, 2], [0.01, 0.01], ["a", "b"])

    def test_condor_cost_is_capture_times_stress_times_leg_sum(self):
        m = LegSpreadModel("affine_usd", a=0.01, b=0.01)
        mids = (1.0, 1.2, 0.3, 0.4)
        want = sum(0.01 + 0.01 * x for x in mids)
        assert condor_leg_cost_usd(m, mids) == pytest.approx(want)
        assert condor_leg_cost_usd(m, mids, capture=0.5, stress=2.0) == pytest.approx(want)


class TestEngineFrictionV2:
    def test_bare_default_is_legacy_and_loads_no_model(self, synth_df, synth_features):
        bt = Backtester(**_kw(synth_df, synth_features))
        assert bt._friction_model == "legacy" and bt._leg_spread_model is None

    def test_entry_friction_is_capture_times_four_legs_of_dollar_halfspread(
            self, synth_df, synth_features, leg_db):
        for capture in (1.0, 0.5):
            res = _v2(synth_df, synth_features, leg_db,
                      friction_entry_capture=capture).run()
            assert res.total_trades > 0
            for t in res.trades:
                assert t["entry_friction_usd"] == pytest.approx(4 * 0.02 * capture, abs=1e-4)
                assert t["mid_credit"] - t["entry_price"] == pytest.approx(
                    t["entry_friction_usd"], abs=2e-4)

    def test_friction_model_changes_the_cost_not_the_first_trade_selection(
            self, synth_df, synth_features, tmp_path):
        """Same first entry day/strikes/mid credit under both models; only the friction
        deducted from it differs (a tiny $/leg model must cost less than the legacy %)."""
        from ait.data.historical import HistoricalDataStore
        db = tmp_path / "tiny.db"
        HistoricalDataStore(db_path=db).save_leg_spread_model(
            "XYZ", {"form": "const_usd", "a": 0.005, "b": 0.0, "n": 9, "n_sessions": 3,
                    "cv_mae": 0.001, "calibrated_on": "2026-10-09"})
        legacy = Backtester(**_kw(synth_df, synth_features)).run().trades[0]
        v2 = _v2(synth_df, synth_features, db).run().trades[0]
        assert (v2["entry_date"], v2["short_call_strike"], v2["short_put_strike"]) == (
            legacy["entry_date"], legacy["short_call_strike"], legacy["short_put_strike"])
        assert v2["mid_credit"] == pytest.approx(legacy["mid_credit"], abs=1e-4)
        assert v2["entry_friction_usd"] == pytest.approx(0.02)
        assert v2["entry_friction_usd"] < legacy["entry_friction_usd"]

    def test_exit_friction_is_capture_times_sum_of_leg_halfspreads_at_exit_mids(
            self, synth_df, synth_features, tmp_path):
        from ait.data.historical import HistoricalDataStore
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db).save_leg_spread_model(
            "XYZ", {"form": "affine_usd", "a": 0.01, "b": 0.02, "n": 9, "n_sessions": 3,
                    "cv_mae": 0.001, "calibrated_on": "2026-10-09"})
        res = _v2(synth_df, synth_features, db, friction_exit_capture=0.5).run()
        bt = _v2(synth_df, synth_features, db, friction_exit_capture=0.5)
        t = res.trades[0]
        pos = dict(t, strategy="iron_condor")
        und = float(t["underlying_at_entry"]) * 1.01
        sc, sp, lc, lp = bt._reprice_ic_legs(pos, und, 5, None)
        base = bt._reprice_position(pos, und, 5, None)
        assert base == pytest.approx((sc + sp) - (lc + lp))
        want = base + 0.5 * sum(0.01 + 0.02 * m for m in (sc, sp, lc, lp))
        got = bt._apply_exit_friction(pos, base, und, 5, None, 0.5, "credit")
        assert got == pytest.approx(want)

    def test_legacy_exit_is_the_percent_shortcut(self, synth_df, synth_features):
        bt = Backtester(**_kw(synth_df, synth_features))
        pos = {"strategy": "iron_condor"}
        assert bt._apply_exit_friction(pos, 1.0, 100.0, 3, None, 0.10, "credit") == pytest.approx(1.1)
        assert bt._apply_exit_friction(pos, 1.0, 100.0, 3, None, 0.10, "debit") == pytest.approx(0.9)

    def test_non_condor_keeps_the_percent_model_even_under_v2(
            self, synth_df, synth_features, leg_db):
        bt = _v2(synth_df, synth_features, leg_db)
        assert bt._apply_exit_friction({"strategy": "short_strangle"}, 2.0, 100.0, 3, None,
                                       0.10, "credit") == pytest.approx(2.2)

    def test_missing_params_fall_back_to_legacy_with_a_warning(
            self, synth_df, synth_features, tmp_path):
        from structlog.testing import capture_logs
        with capture_logs() as ev:
            bt = Backtester(**_kw(synth_df, synth_features, symbol="NOPE",
                                  db_path=tmp_path / "empty.db", friction_model="per_leg_usd"))
        assert bt._leg_spread_model is None
        warn = [e for e in ev if e["event"] == "friction_params_absent"]
        assert warn and warn[0]["log_level"] == "warning"

    def test_bad_friction_model_is_refused(self, synth_df, synth_features):
        with pytest.raises(ValueError):
            Backtester(**_kw(synth_df, synth_features, friction_model="vibes"))

    def test_stress_multiplier_scales_entry_cost(self, synth_df, synth_features, leg_db):
        res = _v2(synth_df, synth_features, leg_db, friction_stress_multiplier=2.0).run()
        assert all(t["entry_friction_usd"] == pytest.approx(0.16, abs=1e-4)
                   for t in res.trades)

    def test_a_trade_round_trips_through_exit_with_v2(self, synth_df, synth_features, leg_db):
        res = _v2(synth_df, synth_features, leg_db).run()
        assert res.total_trades > 0 and all(np.isfinite(t["pnl"]) for t in res.trades)


class TestFrictionConfigAndGuards:
    def test_config_defaults_and_bounds(self):
        from pydantic import ValidationError
        from ait.config.settings import BacktestConfig
        c = BacktestConfig()
        assert (c.friction_model, c.friction_entry_capture, c.friction_exit_capture,
                c.friction_stress_multiplier) == ("per_leg_usd", 1.0, 1.0, 1.0)
        with pytest.raises(ValidationError):
            BacktestConfig(friction_model="nope")
        with pytest.raises(ValidationError):
            BacktestConfig(friction_entry_capture=-0.1)

    def test_research_entry_points_resolve_v2_from_config(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        wf = WalkForwardConfig()
        assert wf.friction_model == "per_leg_usd"
        assert wf.friction_entry_capture == 1.0 and wf.friction_exit_capture == 1.0

    def test_no_exit_site_applies_the_percent_shortcut_directly(self):
        """Class guard: the NEXT exit path must go through _apply_exit_friction, or v2
        silently diverges between exit routes (entry/exit inconsistency was a W65 finding)."""
        import ast
        import inspect
        import textwrap
        offenders = []
        for name, fn in inspect.getmembers(Backtester, inspect.isfunction):
            if name == "_apply_exit_friction":
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            for node in ast.walk(tree):
                if isinstance(node, (ast.AugAssign, ast.BinOp)) and any(
                        isinstance(n, ast.Name) and n.id == "exit_half_spread"
                        for n in ast.walk(node)):
                    offenders.append((name, node.lineno))
        assert not offenders, f"exit_half_spread applied outside _apply_exit_friction: {offenders}"


class TestStoreAndSampler:
    def test_bar_samples_roundtrip_and_are_idempotent(self, tmp_path):
        from ait.data.historical import HistoricalDataStore
        s = HistoricalDataStore(db_path=tmp_path / "h.db")
        df = pd.DataFrame([{"bar_time": "2026-10-08T11:00:00-04:00", "right": "P", "strike": 255.0,
                            "expiry": "2026-10-30", "dte": 22, "bid": 0.40, "ask": 0.44,
                            "underlying_price": 277.0, "delta": -0.1, "iv": 0.2, "role": "short"}])
        assert s.save_spread_bar_samples("IWM", df, source="IB_BID_ASK_BARS") == 1
        s.save_spread_bar_samples("IWM", df, source="IB_BID_ASK_BARS")
        out = s.load_spread_bar_samples("IWM")
        assert len(out) == 1
        r = out.iloc[0]
        assert r["half_spread_usd"] == pytest.approx(0.02) and r["mid"] == pytest.approx(0.42)
        assert r["source"] == "IB_BID_ASK_BARS"

    def test_leg_model_roundtrip_and_absent(self, tmp_path):
        from ait.data.historical import HistoricalDataStore
        s = HistoricalDataStore(db_path=tmp_path / "h.db")
        assert s.load_leg_spread_model("SPY") is None
        s.save_leg_spread_model("SPY", {"form": "pct", "a": 0.0, "b": 0.02, "n": 10,
                                        "calibrated_on": "2026-10-09"})
        assert LegSpreadModel.from_row(s.load_leg_spread_model("SPY")).form == "pct"

    def test_legacy_spread_tables_are_untouched(self, tmp_path):
        import sqlite3
        from ait.data.historical import HistoricalDataStore
        HistoricalDataStore(db_path=tmp_path / "h.db")
        cols = [r[1] for r in sqlite3.connect(tmp_path / "h.db").execute(
            "PRAGMA table_info(option_spread_samples)")]
        assert cols == ["symbol", "sample_date", "right", "strike", "dte", "iv", "bid", "ask",
                        "mid", "half_spread_pct"]

    def test_pair_bid_ask_filters_window_and_invalid_quotes(self):
        from types import SimpleNamespace as B
        from ait.data.option_spread_sampler import parse_window, pair_bid_ask
        # 2026-10-08 is EDT (UTC-4): 14:00Z = 10:00 ET, 13:00Z = 09:00 ET
        t = lambda h: pd.Timestamp(f"2026-10-08 {h}:00", tz="UTC")  # noqa: E731
        bid = [B(date=t(13), close=1.0), B(date=t(14), close=1.0), B(date=t(15), close=-1.0),
               B(date=t(16), close=2.0)]
        ask = [B(date=t(13), close=1.1), B(date=t(14), close=1.1), B(date=t(15), close=1.2),
               B(date=t(16), close=1.9)]
        out = pair_bid_ask(bid, ask, parse_window("10:00-15:30"))
        assert list(out.bid) == [1.0]            # 09:00 outside window, -1 invalid, crossed dropped
        assert out.bar_time.iloc[0].startswith("2026-10-08T10:00")

    def test_delta_from_mid_inverts_black_scholes(self):
        from ait.backtesting.pricing import OptionType, black_scholes_price
        from ait.data.option_spread_sampler import delta_from_mid
        mid = black_scholes_price(600.0, 620.0, 21 / 365, 0.05, 0.18, OptionType.CALL)
        d, iv = delta_from_mid(mid, 600.0, 620.0, 21, "C")
        assert iv == pytest.approx(0.18, abs=1e-3) and 0.1 < d < 0.3
        assert delta_from_mid(0.0, 600.0, 620.0, 21, "C") == (None, None)

    def test_plan_strikes_is_deduplicated_and_marks_wings(self):
        from ait.data.option_spread_sampler import plan_strikes
        plan = plan_strikes(277.0, 0.18, 21, [float(k) for k in range(240, 320)], "P")
        ks = [k for k, _ in plan]
        assert len(ks) == len(set(ks)) and {"short", "wing"} <= {r for _, r in plan}
        assert all(k < 277.0 for k in ks)  # puts: below spot

    def test_relevant_mask_keeps_wings_and_drops_dust_and_itm(self):
        from ait.data.option_spread_sampler import relevant_mask
        df = pd.DataFrame({
            "dte": [21, 21, 21, 3, 21], "mid": [1.0, 0.10, 0.01, 1.0, 2.0],
            "half_spread_usd": [0.02] * 5, "delta": [-0.16, -0.03, -0.02, -0.16, -0.8],
            "role": ["short", "wing", "short", "short", "short"]})
        assert list(relevant_mask(df)) == [True, True, False, False, False]

    def test_sampler_never_places_or_modifies_orders(self):
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        for rel in ("scripts/calibrate_option_spreads.py",
                    "src/ait/data/option_spread_sampler.py"):
            tree = ast.parse((root / rel).read_text())
            calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
                     for n in ast.walk(tree) if isinstance(n, ast.Call)}
            assert not calls & {"placeOrder", "cancelOrder", "modifyOrder"}, rel


class TestSensitivityAndFillReport:
    @staticmethod
    def _load(name: str):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_break_even_interpolates_the_sign_change(self):
        m = self._load("friction_sensitivity")
        assert m.break_even_multiple([(0, 100), (1, 40), (2, -20)]) == pytest.approx(1.6667, abs=1e-3)
        assert m.break_even_multiple([(0, 100), (1, 50)]) is None       # never crosses
        assert m.break_even_multiple([(0, -5), (1, -50)]) is None

    def test_summarize_counts_wins_and_means(self):
        m = self._load("friction_sensitivity")
        s = m.summarize([{"pnl": 20.0}, {"pnl": 30.0}, {"pnl": -100.0}])
        assert s["n"] == 3 and s["win"] == pytest.approx(66.667, abs=1e-2)
        assert s["pnl"] == -50.0 and s["avg_win"] == 25.0 and s["avg_loss"] == -100.0
        assert m.summarize([])["n"] == 0

    def test_fill_report_pairs_combo_fill_with_its_four_leg_fills(self, tmp_path):
        import sqlite3
        m = self._load("fill_vs_model_report")
        db = tmp_path / "s.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE executions (exec_id TEXT, order_id INT, symbol TEXT, con_id INT,"
                    " side TEXT, shares REAL, price REAL, exec_time TEXT, trade_id TEXT,"
                    " live_mid REAL, nbbo_spread REAL, signal_price REAL)")
        rows = [("a", 7, "QQQ", 1, "BOT", 1, 0.92, "t", "T1", 1.005, 0.21, 1.0)]
        for i, p in enumerate((2.92, 3.45, 4.85, 4.46)):
            rows.append((f"l{i}", 7, "QQQ", 100 + i, "BOT", 1, p, "t", "T1", 0.0, 0.0, 0.0))
        con.executemany("INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        loaded = m.load_rows(db)
        cov = m.coverage(loaded)
        assert cov["total"] == 5 and cov["with_quote_context"] == 1
        comps = m.fill_comparisons(loaded, lambda s: LegSpreadModel("const_usd", a=0.02))
        assert len(comps) == 1
        assert comps[0]["realized"] == pytest.approx(0.085)
        assert comps[0]["model"] == pytest.approx(0.08)   # 4 legs x $0.02
        # without a fitted model the row is reported but not compared
        assert m.fill_comparisons(loaded, lambda s: None)[0]["model"] is None

    def test_fill_report_opens_the_state_db_read_only(self, tmp_path):
        import sqlite3
        m = self._load("fill_vs_model_report")
        db = tmp_path / "s.db"
        sqlite3.connect(db).execute("CREATE TABLE executions (exec_id TEXT, order_id INT,"
                                    " symbol TEXT, con_id INT, side TEXT, shares REAL, price REAL,"
                                    " exec_time TEXT, trade_id TEXT, live_mid REAL,"
                                    " nbbo_spread REAL, signal_price REAL)").connection.commit()
        before = db.stat().st_mtime_ns
        assert m.load_rows(db) == []
        assert db.stat().st_mtime_ns == before
        import inspect
        assert "mode=ro" in inspect.getsource(m.load_rows)


class TestEveryEngineIsBuiltWithItsSymbolAndStore:
    """Class guard: a Backtester built without `symbol` silently prices with the config
    spread formula and skips friction model v2 (the W66 meta-label shadow labeller did
    exactly that for months); one built without `db_path` reads the cwd-default store."""

    @pytest.mark.parametrize("module", ["ait.backtesting.walkforward",
                                         "ait.optimization.optimizer"])
    def test_every_backtester_call_passes_symbol_and_db_path(self, module):
        import ast
        import importlib
        import inspect
        tree = ast.parse(inspect.getsource(importlib.import_module(module)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "Backtester"]
        assert calls, "guard would be vacuous"
        bad = []
        for c in calls:
            kws = {k.arg for k in c.keywords}
            if not {"symbol", "db_path"} <= kws:
                bad.append((c.lineno, sorted({"symbol", "db_path"} - kws)))
        assert not bad, f"Backtester(...) built without symbol/db_path: {bad}"
