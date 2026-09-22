"""Post-event blackout study: the research-only gate must be inert by default,
must block exactly the days it says, and must reach every window worker.

Prompted by 2026-09-17: a QQQ condor entered the morning after FOMC (IV
crushed, credit 4.99 on a 46-wide) and stopped out for -$503 two sessions
later. One trade is not evidence; scripts/post_event_study.py sweeps a
post-FOMC blackout of 1/2/3 days across the walk-forward history under a
pre-registered rule. These tests cover the injection path only — the same
shape as tests/test_take_profit_study_injection.py.

The live pre-event gate (ait.data.economic_calendar, 2026-2027 only) is not
touched; the study passes its own FOMC dates back to 2022 because a buffer
tested on the 2026 windows alone would be a test of one year.
"""
from __future__ import annotations

import pickle
from datetime import date, timedelta

import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

from tests.test_r20_research_validity import (  # noqa: F401 — shared fixtures
    _engine_kwargs,
    _recorder_factory,
    _run_window,
    synth_df,
    synth_features,
)

FOMC = date(2026, 9, 16)


class TestTheGateIsInertByDefault:
    def test_engine_default_is_off(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._post_event_blackout_days is None
        for d in (FOMC - timedelta(days=1), FOMC, FOMC + timedelta(days=1)):
            assert bt._post_event_blocked(d) is False

    def test_dates_without_a_window_do_nothing(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        post_event_dates=(FOMC,))
        assert bt._post_event_blocked(FOMC + timedelta(days=1)) is False

    def test_config_default_is_off(self):
        cfg = WalkForwardConfig()
        assert cfg.post_event_blackout_days is None and cfg.post_event_dates is None


class TestTheGateBlocksExactlyItsWindow:
    @pytest.mark.parametrize("n,blocked_offsets,open_offsets", [
        (1, (0, 1), (-1, 2, 3)),
        (2, (0, 1, 2), (-1, 3)),
        (0, (0,), (-1, 1)),
    ])
    def test_days_after_the_event(self, synth_df, synth_features, n, blocked_offsets, open_offsets):
        """Day 0 is the event day (the decision lands intraday); the 09-17
        trade was day +1 after the 09-16 FOMC."""
        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        post_event_blackout_days=n, post_event_dates=(FOMC,))
        for off in blocked_offsets:
            assert bt._post_event_blocked(FOMC + timedelta(days=off)), (n, off)
        for off in open_offsets:
            assert not bt._post_event_blocked(FOMC + timedelta(days=off)), (n, off)

    def test_only_the_most_recent_event_counts(self, synth_df, synth_features):
        earlier = FOMC - timedelta(days=49)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        post_event_blackout_days=2, post_event_dates=(earlier, FOMC))
        assert bt._days_since_event(FOMC + timedelta(days=1)) == 1
        assert bt._days_since_event(earlier + timedelta(days=10)) == 10
        assert not bt._post_event_blocked(earlier + timedelta(days=10))

    def test_iso_strings_are_accepted(self, synth_df, synth_features):
        """The walk-forward config carries ISO strings (they pickle cleanly)."""
        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        post_event_blackout_days=1, post_event_dates=("2026-09-16",))
        assert bt._post_event_blocked(date(2026, 9, 17))

    def test_the_gate_is_applied_at_entry(self, synth_df, synth_features):
        """Drive the real entry path: with every simulated day inside a
        blackout, a run that otherwise opens condors opens none."""
        kw = _engine_kwargs(synth_df, synth_features)
        plain = Backtester(**kw).run()
        n_plain = len(plain.trades)
        assert n_plain > 0, "the synthetic series must produce entries for this test to mean anything"
        days = sorted({d.date() for d in synth_df.index})
        every_day = tuple(days)
        gated = Backtester(**kw, post_event_blackout_days=0, post_event_dates=every_day).run()
        assert len(gated.trades) == 0, (n_plain, len(gated.trades))


class TestWalkForwardThreading:
    def test_refused_with_per_window_optimization(self):
        with pytest.raises(ValueError, match="post_event_blackout_days"):
            WalkForwardConfig(post_event_blackout_days=1, optimize_per_window=True)

    def test_negative_window_is_refused(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(post_event_blackout_days=-1)

    def test_survives_the_process_pool_boundary(self):
        cfg = pickle.loads(pickle.dumps(WalkForwardConfig(
            post_event_blackout_days=2, post_event_dates=(FOMC, "2026-10-28"))))
        assert cfg.post_event_blackout_days == 2
        assert cfg.post_event_dates == ("2026-09-16", "2026-10-28")

    def test_oos_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(train_window_models=False,
                                post_event_blackout_days=2, post_event_dates=(FOMC,))
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink, "no OOS Backtester was constructed"
        assert sink[-1].get("post_event_blackout_days") == 2
        assert sink[-1].get("post_event_dates") == ("2026-09-16",)

    def test_meta_label_shadow_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(post_event_blackout_days=1, post_event_dates=(FOMC,))
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(
            train_df=synth_df, symbol="SPY", window_id=1,
            predictor=None, window_cfg=cfg)
        assert sink, "shadow Backtester was not constructed"
        assert sink[-1].get("post_event_blackout_days") == 1
