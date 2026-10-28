"""W114: the sizing scalers and their inputs are ONE implementation (ait.risk.scalers), called by live and research.

* a golden captured from the sizer BEFORE the refactor pins every count the live sizer produced;
* the six factors, the losing-day streak and the sizing confidence are pinned as units;
* the live sizer's count equals the engine's for the same inputs, scalers off and on;
* the research engine feeds the scalers the five inputs live feeds them.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import PositionSizingConfig, load_settings
from ait.risk.lot_sizing import lots_for_budget
from ait.risk.position_sizer import PositionSizer
from ait.risk.scalers import (
    LOSING_DAYS_LOOKBACK_CALENDAR_DAYS,
    RANGE_GATED_STRATEGIES,
    ScalerInputs,
    lot_scaler,
    recent_losing_days,
    scaler_factors,
    sizing_confidence,
    tier_factor,
)

GOLDEN = json.loads((pathlib.Path(__file__).parent / "data" / "sizer_factors_golden.json").read_text())
SIZING = PositionSizingConfig()
MIN_CONF = 0.50


def _settings(scaled):
    s = load_settings("config.yaml").model_copy(deep=True)
    s.positions.max_contracts_per_trade = GOLDEN["max_lots"]
    s.position_sizing.scaled_strategies = list(scaled)
    return s


def _row_kwargs(key):
    strat, c, iv, rk, d, v = key.split("|")
    kw = dict(account_value=GOLDEN["nlv"], option_price=GOLDEN["option_price"], confidence=float(c),
              implied_vol=float(iv), strategy=strat, underlying_price=0,
              recent_losing_days=int(d), vix=None if v == "None" else float(v))
    if rk != "None":
        kw["iv_rank"] = float(rk)
    return kw


# --------------------------------------------------------------------------------------------------
# the golden: nothing the live sizer produced has changed
# --------------------------------------------------------------------------------------------------
class TestTheSizerMatchesTheGoldenCapturedBeforeTheRefactor:
    def test_scalers_off_every_count_and_reported_factor_is_identical(self):
        s = _settings([])
        sizer = PositionSizer(s.positions, s.risk, s.position_sizing)
        for key, row in GOLDEN["rows"].items():
            r = sizer.calculate(**_row_kwargs(key))
            o = row["off"]
            assert (r.contracts, round(r.confidence_adjustment, 12), round(r.volatility_adjustment, 12)) == \
                   (o["contracts"], o["conf_adj"], o["vol_adj"]), key
            assert r.scaler == 1.0, key

    def test_scalers_on_identical_except_exactly_where_the_product_exceeds_one(self):
        strategies = sorted({k.split("|")[0] for k in GOLDEN["rows"]})
        s = _settings(strategies)
        sizer = PositionSizer(s.positions, s.risk, s.position_sizing)
        clamped = 0
        for key, row in GOLDEN["rows"].items():
            r = sizer.calculate(**_row_kwargs(key))
            on, off = row["on"], row["off"]
            assert round(r.scaler, 12) == on["scaler"], key                  # the product is unchanged
            if on["scaler"] > 1.0:
                clamped += 1
                assert r.contracts == off["contracts"], key                  # the clamp: no more than unscaled
            else:
                assert r.contracts == on["contracts"], key
        assert clamped >= 10                                                  # the grid does exercise the clamp

    def test_an_unknown_iv_rank_is_neutral_where_it_used_to_be_rank_50(self):
        s = _settings(["iron_condor"])
        sizer = PositionSizer(s.positions, s.risk, s.position_sizing)
        base = dict(account_value=100_000.0, option_price=1.5, confidence=0.72, implied_vol=0.3,
                    strategy="iron_condor", underlying_price=0)
        assert sizer.calculate(**base).scaler == sizer.calculate(**base, iv_rank=50.0).scaler
        assert sizer.calculate(**base).factors.iv_rank == 1.0


# --------------------------------------------------------------------------------------------------
# the units
# --------------------------------------------------------------------------------------------------
class TestTheSixFactors:
    @staticmethod
    def f(strategy="long_call", **kw):
        inputs = dict(confidence=0.75, implied_vol=0.10, iv_rank=50.0, recent_losing_days=0, vix=10.0)
        inputs.update(kw)
        return scaler_factors(strategy, ScalerInputs(**inputs), SIZING, MIN_CONF)

    def test_confidence(self):
        assert self.f(confidence=1.0).conf == 1.0
        assert self.f(confidence=0.5).conf == pytest.approx(0.5)
        assert self.f(confidence=0.0).conf == SIZING.confidence_size_floor

    def test_volatility(self):
        assert self.f(implied_vol=0.20).vol == 1.0
        assert self.f(implied_vol=0.60).vol == SIZING.iv_floor_size_factor
        assert self.f(implied_vol=0.40).vol == pytest.approx(0.75)

    def test_iv_rank_selling_and_buying_run_opposite_ways_and_are_clamped(self):
        assert self.f("iron_condor", iv_rank=0.0).iv_rank == SIZING.iv_rank_adj_min
        assert self.f("iron_condor", iv_rank=100.0).iv_rank == SIZING.iv_rank_adj_max
        assert self.f("long_call", iv_rank=0.0).iv_rank == SIZING.iv_rank_adj_max
        assert self.f("long_call", iv_rank=100.0).iv_rank == SIZING.iv_rank_adj_min
        assert self.f("iron_condor", iv_rank=500.0).iv_rank == SIZING.iv_rank_adj_max

    def test_an_unknown_rank_is_neutral_even_when_the_band_is_not_symmetric(self):
        lopsided = SIZING.model_copy(update={"iv_rank_adj_min": 0.7, "iv_rank_adj_max": 1.2})
        unknown = scaler_factors("iron_condor", ScalerInputs(0.75, 0.1, iv_rank=None), lopsided, MIN_CONF)
        mid = scaler_factors("iron_condor", ScalerInputs(0.75, 0.1, iv_rank=50.0), lopsided, MIN_CONF)
        assert unknown.iv_rank == 1.0 and mid.iv_rank == pytest.approx(0.95)

    def test_strategy_drawdown_and_vix(self):
        assert self.f("iron_condor").strategy == SIZING.strategy_risk["iron_condor"]
        assert self.f("not_a_strategy").strategy == 1.0
        assert [self.f(recent_losing_days=d).drawdown for d in (0, 2, 3, 4, 5, 9)] == [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]
        assert [self.f(vix=v).vix for v in (None, 0.0, 24.9, 25.0, 29.9, 30.0)] == [1.0, 1.0, 1.0, 0.75, 0.75, 0.5]

    def test_the_product_and_the_listed_strategy_rule(self):
        factors = self.f("iron_condor", confidence=1.0, implied_vol=0.4, iv_rank=100.0, recent_losing_days=3, vix=26.0)
        expected = 1.0 * 0.75 * 1.2 * SIZING.strategy_risk["iron_condor"] * 0.5 * 0.75
        assert factors.product == pytest.approx(expected)
        listed = SIZING.model_copy(update={"scaled_strategies": ["iron_condor"]})
        assert lot_scaler("iron_condor", factors, listed) == pytest.approx(expected)
        assert lot_scaler("iron_condor", factors, SIZING) == 1.0
        assert lot_scaler("long_call", factors, listed) == 1.0

    def test_tier_factor_edges(self):
        assert tier_factor(2.9, [[3, 0.5]]) == 1.0 and tier_factor(3.0, [[3, 0.5]]) == 0.5


class TestTheClampAtOne:
    @pytest.mark.parametrize("scaler", [1.0, 1.2, 1.5, 7.0])
    def test_a_scaler_above_one_never_raises_the_count(self, scaler):
        assert lots_for_budget(150.0, 100_000.0, 0.05, max_lots=50, scaler=scaler) == 33

    def test_throttling_still_works_and_never_goes_below_one_lot(self):
        assert lots_for_budget(150.0, 100_000.0, 0.05, max_lots=50, scaler=0.5) == 16
        assert lots_for_budget(150.0, 100_000.0, 0.05, max_lots=50, scaler=0.0) == 1
        assert lots_for_budget(150.0, 100_000.0, 0.05, max_lots=50, scaler=-3.0) == 1

    def test_viability_is_still_judged_on_the_unscaled_budget(self):
        assert lots_for_budget(300.0, 5_000.0, 0.05, scaler=0.01) == 0


def _day(trades_taken, pnl):
    return SimpleNamespace(trades_taken=trades_taken, total_pnl=pnl)


class TestTheLosingDayStreak:
    TODAY = date(2026, 10, 28)

    def stats(self, table):
        return lambda d: table.get((self.TODAY - d).days)

    def test_consecutive_losing_days_that_had_an_entry_skipping_idle_days(self):
        table = {1: _day(1, -50.0), 2: _day(0, 0.0), 3: _day(2, -10.0), 4: _day(1, 5.0), 5: _day(1, -99.0)}
        assert recent_losing_days(self.TODAY, self.stats(table)) == 2

    def test_a_flat_day_ends_the_streak(self):
        table = {1: _day(1, -50.0), 2: _day(1, 0.0), 3: _day(1, -50.0)}
        assert recent_losing_days(self.TODAY, self.stats(table)) == 1

    def test_today_is_excluded(self):
        assert recent_losing_days(self.TODAY, self.stats({0: _day(3, -500.0)})) == 0

    def test_the_lookback_is_ten_calendar_days(self):
        assert LOSING_DAYS_LOOKBACK_CALENDAR_DAYS == 10
        assert recent_losing_days(self.TODAY, self.stats({10: _day(1, -1.0)})) == 1
        assert recent_losing_days(self.TODAY, self.stats({11: _day(1, -1.0)})) == 0

    def test_an_unknown_day_is_skipped(self):
        assert recent_losing_days(self.TODAY, lambda d: None) == 0

    def test_the_live_manager_counts_with_the_shared_rule(self):
        from ait.risk.manager import RiskManager
        import ait.risk.manager as mgr
        state = MagicMock()
        table = {1: _day(1, -50.0), 2: _day(1, -20.0), 3: _day(1, 10.0)}
        state.get_daily_stats = lambda d: table.get((self.TODAY - d).days, _day(0, 0.0))
        rm = RiskManager.__new__(RiskManager)
        rm._state = state
        orig = mgr.today_et
        mgr.today_et = lambda: self.TODAY
        try:
            assert rm._count_recent_losing_days() == 2
        finally:
            mgr.today_et = orig

    def test_a_store_that_raises_means_no_streak_not_a_crash(self):
        from ait.risk.manager import RiskManager
        rm = RiskManager.__new__(RiskManager)
        rm._state = MagicMock()
        rm._state.get_daily_stats.side_effect = RuntimeError("db locked")
        assert rm._count_recent_losing_days() == 0


class TestTheConfidenceAStructureIsSizedOn:
    KW = dict(directional_conf=0.81, neutral_conf=0.60)

    def test_a_range_gated_structure_with_the_gates_off_is_sized_on_the_neutral_baseline(self):
        for strat in RANGE_GATED_STRATEGIES:
            assert sizing_confidence(strat, gates_on=False, range_prob=0.93, **self.KW) == 0.60

    def test_with_the_gates_on_it_is_sized_on_the_range_probability(self):
        assert sizing_confidence("iron_condor", gates_on=True, range_prob=0.93, **self.KW) == 0.93

    def test_with_the_gates_on_and_no_probability_it_falls_back_to_the_directional_value(self):
        assert sizing_confidence("iron_condor", gates_on=True, range_prob=None, **self.KW) == 0.81

    def test_everything_else_is_sized_on_the_directional_value(self):
        for gates in (True, False):
            assert sizing_confidence("bull_call_spread", gates_on=gates, range_prob=0.93, **self.KW) == 0.81

    def test_the_range_gated_set_is_the_one_the_orchestrator_uses(self):
        import ait.bot.orchestrator as orch
        assert orch.RANGE_GATED_STRATEGIES is RANGE_GATED_STRATEGIES == frozenset({"iron_condor", "short_strangle"})


# --------------------------------------------------------------------------------------------------
# live == research
# --------------------------------------------------------------------------------------------------
def _engine(scaled, budget=0.05):
    from tests.test_w74_engine_realism import _bt
    s = load_settings("config.yaml").model_copy(deep=True)
    s.position_sizing.scaled_strategies = list(scaled)
    return _bt(settings=s, position_size_pct=budget, initial_capital=100_000.0)


class TestLiveAndResearchCountTheSameLots:
    GRID = list(itertools.product(
        ("iron_condor", "short_strangle", "bull_call_spread"), (0.45, 0.60, 0.93), (0.10, 0.35, 0.70),
        (None, 0.0, 100.0), (0, 3, 6), (None, 16.0, 27.0, 32.0)))

    def test_counts_agree_with_scalers_off_and_on(self):
        for scaled in ([], ["iron_condor", "short_strangle", "bull_call_spread"]):
            s = _settings(scaled)
            sizer = PositionSizer(s.positions, s.risk, s.position_sizing)
            bt = _engine(scaled)
            bt._sizing_min_conf = s.risk.min_confidence
            for strat, conf, iv, rank, days, vix in self.GRID:
                live = sizer.calculate(account_value=100_000.0, option_price=1.5, confidence=conf, implied_vol=iv,
                                       strategy=strat, underlying_price=0, iv_rank=rank,
                                       recent_losing_days=days, vix=vix).contracts
                bt._size_ctx = {"strategy": strat, "confidence": conf, "iv": iv, "iv_rank": rank,
                                "losing_days": days, "vix": vix}
                research = min(bt._lots(150.0, 100_000.0), GOLDEN["max_lots"])
                assert research == live, (scaled, strat, conf, iv, rank, days, vix)
            bt._size_ctx = None

    def test_without_inputs_the_engine_applies_no_scaler(self):
        bt = _engine(["iron_condor"])
        bt._size_ctx = None
        assert bt._lots(150.0, 100_000.0) == lots_for_budget(150.0, 100_000.0, 0.05)

    def test_a_strategy_that_is_not_listed_is_sized_by_the_budget_alone(self):
        bt = _engine(["iron_condor"])
        bt._size_ctx = {"strategy": "bull_call_spread", "confidence": 0.45, "iv": 0.7, "iv_rank": 100.0,
                        "losing_days": 9, "vix": 40.0}
        assert bt._lots(150.0, 100_000.0) == lots_for_budget(150.0, 100_000.0, 0.05)

    def test_the_condor_site_hands_the_scalers_the_short_put_leg_iv(self):
        bt = _engine(["iron_condor"])
        bt._size_ctx = {"strategy": "iron_condor", "confidence": 0.75, "iv": 0.2, "leg_iv": 0.7, "iv_rank": 50.0,
                        "losing_days": 0, "vix": 10.0}
        with_leg = bt._lots(150.0, 1_000_000.0)
        bt._size_ctx = {**bt._size_ctx, "leg_iv": None}
        assert with_leg < bt._lots(150.0, 1_000_000.0)


class TestTheEnginesOwnInputs:
    def test_the_losing_streak_is_read_off_the_runs_own_book(self):
        from ait.backtesting.engine import Backtester
        today = date(2026, 3, 20)
        d = lambda n: (today - timedelta(days=n)).isoformat()                       # noqa: E731
        closed = [
            {"entry_date": d(5), "exit_date": d(1), "pnl": -40.0},
            {"entry_date": d(4), "exit_date": d(2), "pnl": -10.0},
            {"entry_date": d(9), "exit_date": d(8), "pnl": 90.0},
        ]
        open_positions = [{"entry_date": d(1)}]
        got = Backtester._losing_days_at(today, closed, open_positions)
        entries = {d(5): 1, d(4): 1, d(9): 1, d(1): 1}
        # live's rule on the same facts: a day counts when it HAD AN ENTRY; P&L lands on the exit date
        def stats(day):
            key = day.isoformat()
            pnl = sum(c["pnl"] for c in closed if c["exit_date"] == key)
            return SimpleNamespace(trades_taken=entries.get(key, 0), total_pnl=pnl)
        assert got == recent_losing_days(today, stats)
        assert got == 1       # d(1): an entry and -40 -> a loss; d(2), d(3) had no entry -> skipped; d(4): entry, P&L 0 -> ends

    def test_a_win_on_a_day_with_an_entry_ends_it(self):
        from ait.backtesting.engine import Backtester
        today = date(2026, 3, 20)
        d = lambda n: (today - timedelta(days=n)).isoformat()                       # noqa: E731
        closed = [{"entry_date": d(3), "exit_date": d(1), "pnl": -40.0},
                  {"entry_date": d(2), "exit_date": d(2), "pnl": 60.0},
                  {"entry_date": d(6), "exit_date": d(3), "pnl": -5.0}]
        # d(1) has no entry (skipped); d(2) entry + a win -> ends: 0 losing days
        assert Backtester._losing_days_at(today, closed, []) == 0

    def test_the_vix_level_is_the_raw_close_at_the_decision_bar(self):
        import pandas as pd
        bt = _engine([])
        idx = pd.date_range("2026-03-02", periods=5, freq="B")
        bt._market_context = {"vix": pd.DataFrame({"Close": [14.0, 15.0, 16.5, 18.0, 20.0]}, index=idx)}
        assert bt._decision_vix(pd.DataFrame(index=idx[:3])) == 16.5
        bt._market_context = {}
        assert bt._decision_vix(pd.DataFrame(index=idx[:3])) is None
        bt._market_context = {"vix": 21.5}
        assert bt._decision_vix(pd.DataFrame(index=idx[:3])) == 21.5


class _RangeModel:
    def predict(self, hist, **kw):
        return SimpleNamespace(probability_in_range=0.91)


def _run(scaled, gates_mode):
    from tests.test_w74_engine_realism import _bt
    s = load_settings("config.yaml").model_copy(deep=True)
    s.position_sizing.scaled_strategies = list(scaled)
    return _bt(settings=s, gates_mode=gates_mode, range_predictor=_RangeModel(), range_min_confidence=0.5).run().trades


class TestTheEngineRunFeedsTheScalers:
    def test_with_the_list_empty_a_trade_carries_no_scaler_fields_and_is_sized_by_the_budget_alone(self):
        trades = _run([], "observe")
        assert trades and all("size_scaler" not in t and "size_confidence" not in t for t in trades)

    def test_observe_sizes_a_condor_on_the_neutral_baseline_while_the_entry_keeps_the_range_probability(self):
        trades = _run(["iron_condor"], "observe")
        assert trades
        assert {t["size_confidence"] for t in trades} == {0.6}
        assert {t["entry_confidence"] for t in trades} == {0.91}

    def test_enforce_sizes_a_condor_on_the_range_probability(self):
        trades = _run(["iron_condor"], "enforce")
        assert trades and {t["size_confidence"] for t in trades} == {0.91}

    def test_the_scaler_throttles_the_count_and_is_recorded_on_the_trade(self):
        off, on = _run([], "observe"), _run(["iron_condor"], "observe")
        assert len(off) == len(on)
        assert all(t["size_scaler"] < 1.0 for t in on)
        assert all(a["contracts"] >= b["contracts"] for a, b in zip(off, on))
        assert sum(a["contracts"] for a in off) > sum(b["contracts"] for b in on)

    def test_scalers_on_does_not_change_which_trades_are_taken(self):
        off, on = _run([], "observe"), _run(["iron_condor"], "observe")
        assert [(t["entry_date"], t["strategy"]) for t in off] == [(t["entry_date"], t["strategy"]) for t in on]

    def test_the_iv_rank_array_is_built_when_scalers_are_on_even_without_the_gate(self):
        assert _engine([])._iv_rank_arr is None
        assert _engine(["iron_condor"])._iv_rank_arr is not None


class TestTheResidualInputDifferencesAreDeclared:
    def test_every_scaler_input_has_a_declared_row(self):
        from ait.backtesting.gate_parity import SIZING_INPUT_DIFFERENCES
        assert {r[0] for r in SIZING_INPUT_DIFFERENCES} == {
            "confidence", "implied_vol", "iv_rank", "vix", "recent_losing_days", "learning_multiplier"}
        assert all(len(r) == 3 and all(r) for r in SIZING_INPUT_DIFFERENCES)


class TestTheManagerHandsTheSizerTheRank:
    pytestmark = pytest.mark.asyncio

    @staticmethod
    async def _seen(rank, scale):
        from tests.lot_request_rig import build_manager, make_request, shipped_settings
        rm = build_manager(shipped_settings(), 25_000.0, [])
        calls = []
        orig = rm._sizer.calculate

        def spy(**kw):
            calls.append(kw)
            return orig(**kw)
        rm._sizer.calculate = spy
        req = make_request("condor_3w", vix=16.0, confidence=0.70, scale_to_budget=scale)
        req.iv_rank = rank
        await rm.validate_trade(req)
        return calls

    @pytest.mark.parametrize("scale", [False, True])
    async def test_a_known_rank_reaches_the_sizer_on_both_paths(self, scale):
        calls = await self._seen(63.0, scale)
        assert calls and all(c["iv_rank"] == 63.0 for c in calls)

    @pytest.mark.parametrize("scale", [False, True])
    async def test_an_unknown_rank_reaches_it_as_unknown_not_as_zero(self, scale):
        calls = await self._seen(None, scale)
        assert calls and all(c["iv_rank"] is None for c in calls)

    async def test_the_scale_to_budget_path_asks_for_a_count_with_the_rank_too(self):
        calls = await self._seen(63.0, True)
        assert len(calls) >= 2          # _target_lots, then step 9


class TestThePoolAppliesTheEntrysScaler:
    def test_a_trade_with_a_scaler_is_throttled_and_one_without_is_not(self):
        from tests.test_lot_sizing_parity import _pool, _trade
        plain = _trade("QQQ", "2024-01-02", "2024-01-10", 100.0, contracts=100)
        scaled = {**_trade("QQQ", "2024-01-02", "2024-01-10", 100.0, contracts=100), "size_scaler": 0.5}
        assert _pool([plain], equity=5_000.0).accepted[0]["contracts"] == 2
        assert _pool([scaled], equity=5_000.0).accepted[0]["contracts"] == 1

    def test_a_scaler_above_one_is_clamped_in_the_pool_too(self):
        from tests.test_lot_sizing_parity import _pool, _trade
        hot = {**_trade("QQQ", "2024-01-02", "2024-01-10", 100.0, contracts=100), "size_scaler": 1.2}
        assert _pool([hot], equity=5_000.0).accepted[0]["contracts"] == 2


class TestTheWalkForwardBuildsTheRankSeriesWhenScalersAreOn:
    @staticmethod
    def _wf(scaled, gate=False):
        import pandas as pd
        from ait.backtesting.walkforward import WalkForwardBacktester
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        wf._config = SimpleNamespace(iv_rank_gate=gate)
        wf._settings = SimpleNamespace(position_sizing=SimpleNamespace(scaled_strategies=scaled))
        wf._iv_rank_series = {}
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        data = {"QQQ": pd.DataFrame({"implied_vol": [0.2 + 0.001 * i for i in range(80)]}, index=idx)}
        wf._ensure_iv_rank_series(data)
        return wf._iv_rank_series

    def test_off_when_neither_the_gate_nor_a_scaler_needs_it(self):
        assert self._wf([]) == {}

    def test_on_for_the_gate_as_before(self):
        assert "QQQ" in self._wf([], gate=True)

    def test_on_for_a_scaler(self):
        assert "QQQ" in self._wf(["iron_condor"])


class TestTheCondorBuilderHandsTheScalersTheShortPutIv:
    def test_the_size_context_a_condor_build_exposes_carries_the_short_put_leg_iv(self):
        from ait.backtesting.pricing import OptionType, skewed_leg_iv
        from tests.test_w74_engine_realism import _bt
        s = load_settings("config.yaml").model_copy(deep=True)
        s.position_sizing.scaled_strategies = ["iron_condor"]
        bt = _bt(settings=s)
        seen = []
        orig = bt._lots

        def spy(risk, capital):
            seen.append(dict(bt._size_ctx))
            return orig(risk, capital)
        bt._lots = spy
        bt.run()
        assert seen
        for ctx in seen:
            assert ctx["leg_iv"] is not None and ctx["leg_iv"] != ctx["iv"]
            assert ctx["leg_iv"] > ctx["iv"]                     # the OTM put carries the skew premium


class TestTheRunHandsTheScalersWhatItMeasured:
    @staticmethod
    def _capture(bt):
        seen = []
        orig = bt._build_position

        def spy(*a, **k):
            if bt._size_in is not None:
                seen.append(dict(bt._size_in))
            return orig(*a, **k)
        bt._build_position = spy
        return seen

    def test_an_unmeasurable_rank_reaches_the_scalers_as_unknown(self):
        import numpy as np
        bt = _engine(["iron_condor"])
        bt._iv_rank_arr = np.full(len(bt._data), np.nan)
        seen = self._capture(bt)
        bt.run()
        assert seen and all(c["iv_rank"] is None for c in seen)

    def test_a_measured_rank_reaches_them_as_that_number(self):
        import numpy as np
        bt = _engine(["iron_condor"])
        bt._iv_rank_arr = np.full(len(bt._data), 63.0)
        seen = self._capture(bt)
        bt.run()
        assert seen and {c["iv_rank"] for c in seen} == {63.0}

    def test_the_runs_losing_streak_is_what_the_scalers_get(self, monkeypatch):
        from ait.backtesting.engine import Backtester
        monkeypatch.setattr(Backtester, "_losing_days_at", staticmethod(lambda today, closed, open_: 7))
        bt = _engine(["iron_condor"])
        seen = self._capture(bt)
        trades = bt.run().trades
        assert seen and {c["losing_days"] for c in seen} == {7}
        assert all(t["size_scaler"] <= 0.25 * 1.0 + 1e-9 for t in trades)      # drawdown tier 5+ quarters the size
