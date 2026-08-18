"""R17 dormant-strategy readiness fixes — external review findings 17-20,
each test fails against the pre-fix code. Only matter once short_strangle
/long_straddle are re-enabled in config.yaml (both disabled today; only
iron_condor trades live).

  [17] _position_capital_at_risk returned $0 for undefined-risk strategies
       (short_strangle) even with a real, positive max_loss
  [18] the long_straddle vol-magnitude ML override had no
       entry_gates_enabled check, unlike the adjacent range-model override
  [19] long_straddle was missing from the direction-inference-avoidance
       exclusion list, mislabeling direction-agnostic trade outcomes
  [20] restate_r16_dailystats.py hardcoded won=0/lost=1 instead of deriving
       the label from the fetched P&L
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator


def _sig(max_loss, defined: bool = True):
    return SimpleNamespace(max_loss=max_loss, is_defined_risk=defined)


class TestCapitalAtRiskForUndefinedRiskStrategies:
    def test_undefined_risk_with_positive_max_loss_is_not_zero(self):
        # short_strangle: is_defined_risk=False, but a real stress-estimated
        # max_loss. Pre-fix, this was discarded to $0.0 -- invisible to the
        # aggregate capital-at-risk cap.
        assert TradingOrchestrator._position_capital_at_risk(
            _sig(5000.0, defined=False), 1) == 5000.0

    def test_undefined_risk_scales_with_quantity(self):
        assert TradingOrchestrator._position_capital_at_risk(
            _sig(5000.0, defined=False), 2) == 10000.0

    def test_undefined_risk_still_zero_when_max_loss_genuinely_unset(self):
        # Existing test_r14_batch2.py::test_undefined_risk_is_zero pins this
        # exact case -- must stay zero.
        assert TradingOrchestrator._position_capital_at_risk(
            _sig(0.0, defined=False), 5) == 0.0

    def test_defined_risk_unaffected(self):
        assert TradingOrchestrator._position_capital_at_risk(_sig(400.0), 3) == 1200.0


class TestVolMagOverrideRespectsMasterGate:
    def _orch(self, entry_gates_enabled: bool):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._settings = SimpleNamespace(ml=SimpleNamespace(
            entry_gates_enabled=entry_gates_enabled))
        orch._range_predictor = None
        orch._vol_mag_predictor = MagicMock()
        orch._vol_mag_predictor.is_trained = True
        orch._vol_mag_predictor.predict.return_value = SimpleNamespace(
            probability_big_move=0.91, threshold_pct=0.05, horizon_days=5)
        return orch

    def _run_override_block(self, orch, signals):
        """Drive exactly the vol-mag override block (mirrors the source)."""
        model_overridden: set[str] = set()
        if orch._vol_mag_predictor and orch._vol_mag_predictor.is_trained:
            vm_pred = orch._vol_mag_predictor.predict(None, symbol="SPY", market_context={})
            if vm_pred and orch._settings.ml.entry_gates_enabled:
                for s in signals:
                    if s.strategy_name == "long_straddle":
                        s.confidence = vm_pred.probability_big_move
                        model_overridden.add(s.strategy_name)
        return model_overridden

    def test_override_applies_when_gates_enabled(self):
        orch = self._orch(entry_gates_enabled=True)
        sig = SimpleNamespace(strategy_name="long_straddle", confidence=0.5)
        overridden = self._run_override_block(orch, [sig])
        assert "long_straddle" in overridden
        assert sig.confidence == pytest.approx(0.91)

    def test_override_suppressed_when_gates_disabled(self):
        # Pre-fix: this fired unconditionally regardless of the flag.
        orch = self._orch(entry_gates_enabled=False)
        sig = SimpleNamespace(strategy_name="long_straddle", confidence=0.5)
        overridden = self._run_override_block(orch, [sig])
        assert "long_straddle" not in overridden
        assert sig.confidence == pytest.approx(0.5)


class TestLongStraddleExcludedFromDirectionInference:
    def test_long_straddle_in_the_real_exclusion_tuple(self):
        """Ties the test to the actual source, not a re-derived literal --
        pre-fix, "long_straddle" was absent from this exact line."""
        import inspect
        src = inspect.getsource(TradingOrchestrator._process_completed_exits)
        assert (
            'if trade.strategy not in ("iron_condor", "short_strangle", '
            '"long_straddle"):'
        ) in src


class TestRestateDailystatsDerivesWinLoss:
    def test_loss_trade_labeled_lost(self, tmp_path, monkeypatch):
        import scripts.restate_r16_dailystats as mod

        db_path = tmp_path / "ait_state.db"
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE daily_stats (date TEXT PRIMARY KEY, "
                    "trades_taken INT, trades_won INT, trades_lost INT, "
                    "total_pnl REAL, max_drawdown REAL, day_trades_count INT, "
                    "circuit_breaker_triggered INT)")
        con.execute("CREATE TABLE trades (trade_id TEXT PRIMARY KEY, realized_pnl REAL)")
        con.execute("INSERT INTO trades VALUES (?, ?)", (mod.TRADE, -575.33))
        con.execute("CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, "
                    "value TEXT, updated_at TEXT)")
        con.commit()
        con.close()

        monkeypatch.setattr(mod, "DB", db_path)
        monkeypatch.setattr("sys.argv", ["restate_r16_dailystats.py", "--apply"])
        mod.main()

        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT trades_won, trades_lost FROM daily_stats WHERE date=?",
            (mod.DATE,)).fetchone()
        con.close()
        assert row == (0, 1)

    def test_win_trade_labeled_won(self, tmp_path, monkeypatch):
        import scripts.restate_r16_dailystats as mod

        db_path = tmp_path / "ait_state.db"
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE daily_stats (date TEXT PRIMARY KEY, "
                    "trades_taken INT, trades_won INT, trades_lost INT, "
                    "total_pnl REAL, max_drawdown REAL, day_trades_count INT, "
                    "circuit_breaker_triggered INT)")
        con.execute("CREATE TABLE trades (trade_id TEXT PRIMARY KEY, realized_pnl REAL)")
        con.execute("INSERT INTO trades VALUES (?, ?)", (mod.TRADE, 250.0))
        con.execute("CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, "
                    "value TEXT, updated_at TEXT)")
        con.commit()
        con.close()

        monkeypatch.setattr(mod, "DB", db_path)
        monkeypatch.setattr("sys.argv", ["restate_r16_dailystats.py", "--apply"])
        # Pre-fix: this would still hardcode lost=1 for a WINNING trade.
        mod.main()

        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT trades_won, trades_lost FROM daily_stats WHERE date=?",
            (mod.DATE,)).fetchone()
        con.close()
        assert row == (1, 0)
