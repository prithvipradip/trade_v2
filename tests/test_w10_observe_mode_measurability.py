"""W10 (R24): the pre-registered observe-mode verdict must be SCOREABLE.

PLAN 2026-09-01 pre-registered the rule for re-arming or retiring the ML
entry gate: "compare closes where p_in_range was above vs below 0.65". The
R24 audit found that rule cannot be executed, because the entry-time range
probability was never persisted anywhere durable:

  * it was written into trade_context.entry_confidence ONLY while the gates
    were armed, and since 2026-09-01 that column holds the observe-mode
    neutral baseline (0.60) instead — the live rows prove it: the 08-24 trade
    carries 0.8977 (a model probability), the 09-01 trades carry exactly 0.60;
  * for SPY/QQQ no probability was ever produced at all (predict() exits at
    the min-edge gate), and "no prediction" must not be confused with a low
    probability;
  * the surviving copy was a log line, and logs rotate in ~9 trading days
    while the verdict is scored at 50-100 closes.

These tests EXECUTE the real StateManager and the real orchestrator helper.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.config.settings import load_settings


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _orch(gates: bool):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    s = load_settings()
    s.ml.entry_gates_enabled = gates
    o._settings = s
    return o


def _pred(p=0.80, thr=0.05, hor=9, ver="range-v-test"):
    return SimpleNamespace(probability_in_range=p, threshold_pct=thr,
                           horizon_days=hor, model_version=ver)


class TestTheColumnsExist:
    def test_range_columns_are_migrated(self, tmp_path):
        st = _state(tmp_path)
        with sqlite3.connect(st._db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trade_context)")}
        for c in ("range_probability", "range_threshold_pct",
                  "range_horizon_days", "range_model_version",
                  "entry_gates_mode", "directional_confidence"):
            assert c in cols, f"{c} missing — the verdict stays unscoreable"


class TestPredictionIsPersistedSeparately:
    def _save(self, st, **kw):
        st.save_trade_context(
            trade_id="T-1", direction="neutral", confidence=0.60,
            regime="neutral", vix=16.0, iv_rank=34.9, sentiment_score=0.0,
            **kw)
        return st.get_trade_context("T-1")

    def test_probability_survives_independently_of_confidence(self, tmp_path):
        """The whole point: `confidence` is the number that drove risk
        validation and changes meaning with gate mode; the probability needs
        its own column."""
        ctx = self._save(_state(tmp_path), range_probability=0.802,
                         range_threshold_pct=0.05, range_horizon_days=9,
                         range_model_version="range-v-1",
                         entry_gates_mode="observe")
        assert ctx["entry_confidence"] == pytest.approx(0.60)
        assert ctx["range_probability"] == pytest.approx(0.802)
        assert ctx["range_horizon_days"] == 9
        assert ctx["entry_gates_mode"] == "observe"

    def test_absent_prediction_is_null_not_zero(self, tmp_path):
        """A model that failed its edge gate produced NOTHING. Storing 0.0
        would read as 'certain the price leaves the range' and would land on
        the wrong side of the 0.65 split."""
        ctx = self._save(_state(tmp_path), range_probability=None,
                         entry_gates_mode="observe")
        assert ctx["range_probability"] is None

    def test_spec_is_recorded_with_the_number(self, tmp_path):
        """A probability is meaningless without the horizon/threshold it was
        produced under — the live model was retrained from h=30 to h=9."""
        ctx = self._save(_state(tmp_path), range_probability=0.71,
                         range_threshold_pct=0.05, range_horizon_days=9)
        assert (ctx["range_threshold_pct"], ctx["range_horizon_days"]) == (0.05, 9)

    def test_backwards_compatible_call_still_works(self, tmp_path):
        ctx = self._save(_state(tmp_path))
        assert ctx["range_probability"] is None
        assert ctx["entry_confidence"] == pytest.approx(0.60)


class TestOrchestratorSuppliesTheContext:
    def test_prediction_is_carried_from_scan_to_entry(self):
        o = _orch(gates=False)
        o._entry_range_pred = {"SPY": _pred(0.82, 0.05, 9, "range-v-9")}
        ctx = o._entry_range_context("SPY")
        assert ctx["range_probability"] == pytest.approx(0.82)
        assert ctx["range_horizon_days"] == 9
        assert ctx["range_model_version"] == "range-v-9"
        assert ctx["entry_gates_mode"] == "observe"

    def test_missing_prediction_yields_nulls_and_the_mode(self):
        """QQQ today: the model exists but fails its edge gate, so predict()
        returns None. That fact must be recorded, not silently dropped."""
        o = _orch(gates=False)
        o._entry_range_pred = {"SPY": _pred()}
        ctx = o._entry_range_context("QQQ")
        assert ctx["range_probability"] is None
        assert ctx["entry_gates_mode"] == "observe"

    def test_mode_reflects_armed_gates(self):
        o = _orch(gates=True)
        o._entry_range_pred = {"SPY": _pred()}
        assert o._entry_range_context("SPY")["entry_gates_mode"] == "armed"

    def test_no_stash_at_all_is_safe(self):
        o = _orch(gates=False)
        ctx = o._entry_range_context("SPY")
        assert ctx["range_probability"] is None

    def test_context_keys_match_the_save_signature(self):
        """Guards the seam: these are splatted into save_trade_context."""
        import inspect
        o = _orch(gates=False)
        params = inspect.signature(StateManager.save_trade_context).parameters
        for k in o._entry_range_context("SPY"):
            assert k in params, f"{k} is not a save_trade_context parameter"


class TestTheVerdictCanNowBeScored:
    def test_above_below_split_is_computable_from_the_db(self, tmp_path):
        """End-to-end: the pre-registered rule, executed against stored rows."""
        st = _state(tmp_path)
        rows = [("T-hi", 0.80), ("T-lo", 0.55), ("T-none", None)]
        for tid, p in rows:
            st.save_trade_context(
                trade_id=tid, direction="neutral", confidence=0.60,
                regime="neutral", vix=16.0, iv_rank=30.0, sentiment_score=0.0,
                range_probability=p, range_threshold_pct=0.05,
                range_horizon_days=9, entry_gates_mode="observe")
        with sqlite3.connect(st._db_path) as conn:
            above = conn.execute(
                "SELECT COUNT(*) FROM trade_context "
                "WHERE range_probability >= 0.65").fetchone()[0]
            below = conn.execute(
                "SELECT COUNT(*) FROM trade_context "
                "WHERE range_probability < 0.65").fetchone()[0]
            unscored = conn.execute(
                "SELECT COUNT(*) FROM trade_context "
                "WHERE range_probability IS NULL").fetchone()[0]
        assert (above, below, unscored) == (1, 1, 1)
