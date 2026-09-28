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


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Real orchestrator, fakes only at the I/O boundary -- the same rig
    test_w2_orchestrator_gates.py uses to drive _scan_symbol for real."""
    from tests.test_hot_path_smoke import build_smoke_orchestrator
    monkeypatch.chdir(tmp_path)     # sandbox FIRST: __init__ writes data/*.db
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


async def _scan_one(rig, monkeypatch, signal) -> list:
    """Drive the real _scan_symbol with signal generation pinned, and return
    the candidates it collected."""
    from tests.test_hot_path_smoke import SYMBOL, VIX
    monkeypatch.setattr(rig.orch._strategy_selector, "generate_all_signals",
                        lambda **kw: [signal])
    rig.orch._risk_budget = 3000.0
    collected: list = []
    await rig.orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)
    await rig.drain_notifications()
    return collected


class _VolMagStub:
    """Stands in for VolMagnitudePredictor at the model boundary (same idiom
    as test_w2_orchestrator_gates.py's _RangeStub)."""

    def __init__(self) -> None:
        self.is_trained = True
        self.calls = 0

    def predict(self, hist, **kw):
        self.calls += 1
        return SimpleNamespace(probability_big_move=0.91, threshold_pct=0.05,
                               horizon_days=5)


class TestVolMagOverrideRespectsMasterGate:
    """CENSUS-4 (2026-09-27) flagged _run_override_block, this class's own
    former helper, as a second offender in this file: it hand-copied
    (mirrored) the vol-mag override block from orchestrator.py's
    _scan_symbol instead of calling it. Rewritten to drive the real
    _scan_symbol via the same real-orchestrator rig
    test_w2_orchestrator_gates.py's TestRangeGateFailsClosed already
    established for the sibling range-model override."""

    async def test_override_applies_when_gates_enabled(self, rig, monkeypatch):
        from tests.test_hot_path_smoke import iron_condor_signal
        rig.orch._settings.ml.entry_gates_enabled = True
        stub = _VolMagStub()
        rig.orch._vol_mag_predictor = stub
        sig = iron_condor_signal(confidence=0.5)
        sig.strategy_name = "long_straddle"

        collected = await _scan_one(rig, monkeypatch, sig)

        assert collected, "long_straddle never reached the ranking pass"
        assert stub.calls == 1, (
            "the vol-mag model was never consulted -- a silently-skipped "
            "block would look identical to a correctly-applied one"
        )
        assert sig.confidence == pytest.approx(0.91)
        assert "vol_mag_prediction" in rig.log.names()

    async def test_override_suppressed_when_gates_disabled(self, rig, monkeypatch):
        # Pre-fix (R17): this fired unconditionally regardless of the flag.
        from tests.test_hot_path_smoke import iron_condor_signal
        rig.orch._settings.ml.entry_gates_enabled = False
        stub = _VolMagStub()
        rig.orch._vol_mag_predictor = stub
        sig = iron_condor_signal(confidence=0.5)
        sig.strategy_name = "long_straddle"

        await _scan_one(rig, monkeypatch, sig)

        assert sig.confidence == pytest.approx(0.5), (
            "R17: the vol-mag override must not apply while gates are OFF"
        )


class TestLongStraddleExcludedFromDirectionInference:
    """CENSUS-4 (2026-09-27): the class docstring's own admission --
    "flagship offender" -- said it outright: "ties the test to the actual
    source, not a re-derived literal." Rewritten to drive the real
    TradingOrchestrator._process_completed_exits (same harness
    test_r15_review_fixes.py's TestCommissionTrueUp already proves out,
    reused here rather than rebuilt), and to assert on whether
    orch._trainer.drift_detector.record_outcome actually gets called --
    _exit_step silently swallows any exception a drift call raises
    (orchestrator.py:1082-1088), so a positive control (a DIRECTIONAL
    strategy that MUST record an outcome) is not optional here: without it,
    a drift step that errors out looks identical to one that correctly
    didn't run."""

    def _orch(self, strategy: str, realized_pnl: float = 100.0):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        state = MagicMock()
        stats = SimpleNamespace(total_pnl=0.0, trades_won=0, trades_lost=0,
                                trades_taken=0)
        state.get_daily_stats.return_value = stats
        state.total_commission.return_value = 0.0
        state.count_executions_with_commission.return_value = 0
        orch._state = state
        trade = SimpleNamespace(
            trade_id="T-DIR", symbol="SPY", strategy=strategy,
            contract_type=strategy, quantity=1, entry_time=None,
            legs="[]", direction=SimpleNamespace(value="long"))
        orch._find_trade_by_id = lambda tid: trade
        orch._circuit_breaker = MagicMock()
        orch._thompson = MagicMock()
        orch._pdt_guard = MagicMock()
        orch._trainer = MagicMock()
        orch._exit_attempts = {}

        async def _notify(m):
            pass

        orch._send_notification = _notify
        ex = {"trade_id": "T-DIR", "realized_pnl": realized_pnl,
              "exit_reason": "take_profit", "exit_price": 1.0}
        return orch, ex

    async def _run(self, strategy: str) -> MagicMock:
        orch, ex = self._orch(strategy)
        await orch._process_completed_exits([ex])
        return orch._trainer.drift_detector.record_outcome

    async def test_iron_condor_is_excluded(self):
        record_outcome = await self._run("iron_condor")
        record_outcome.assert_not_called()

    async def test_short_strangle_is_excluded(self):
        record_outcome = await self._run("short_strangle")
        record_outcome.assert_not_called()

    async def test_long_straddle_is_excluded(self):
        record_outcome = await self._run("long_straddle")
        record_outcome.assert_not_called()

    async def test_a_directional_strategy_control_still_records(self):
        """The mandatory positive control: proves the drift step actually
        runs (and _exit_step's exception-swallowing isn't hiding a silent
        failure) for a strategy that SHOULD feed the drift detector."""
        record_outcome = await self._run("long_call")
        record_outcome.assert_called_once()
        assert record_outcome.call_args.kwargs["trade_id"] == "SPY-bullish"
        assert record_outcome.call_args.kwargs["actual_direction"] == "bullish"


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
