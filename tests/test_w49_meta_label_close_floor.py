"""W49 (R24/R26 register, `policy-vs-impl-7`) — the meta-labeler's close-count
floor is now a real, configurable, ENFORCED number, not three disagreeing
documents and no code check at all.

`MetaLabelConfig.min_closes_to_enable` (config/settings.py) replaces the bare
comment in config.yaml ("Re-enable after 50+ clean trades" — never an
enforced value; config.yaml said 50, PLAN.md said 100+, the code enforced
neither). `reporting.go_live.real_close_count` is the single count
authority — the same real-close, IC-only population `compute_go_live_verdict`
already grades for the go-live scorecard's own pace line.
`TradingOrchestrator._meta_label_arm_sufficient` is the new enforcement
point, re-checked every scan at the ONLY place live entries are actually
influenced (the meta-label reject check inside `_scan_symbol`) — not at
`__init__`/training time, which would go stale for the rest of the day as
closes accrue.

Group A tests the count helper directly (real StateManager). Group B tests
the new orchestrator method directly, including the fail-closed error path.
Group C proves the WIRING, not just the helper in isolation: drives the real
`_scan_symbol` through the build_smoke_orchestrator rig
(test_hot_path_smoke.py) with a fake trained meta-labeler that always
rejects, and shows the reject is skipped below the floor and respected
above it.

No inspect.getsource.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.ml.meta_label import MetaSignal
from ait.reporting.go_live import real_close_count
from ait.utils.time import now_et
from tests.test_hot_path_smoke import SYMBOL, VIX, build_smoke_orchestrator, iron_condor_signal


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed_closed_trade(
    st: StateManager, trade_id: str, strategy: str = "iron_condor",
    exit_reason_detailed: str = "stop_loss", pnl: float = 10.0,
) -> None:
    trade = TradeRecord(
        trade_id=trade_id, symbol="QQQ", strategy=strategy,
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time=now_et().isoformat(), entry_price=2.0, quantity=1,
        contract_type="iron_condor",
    )
    st.record_trade(trade)
    st.close_trade(trade_id, exit_price=1.0, realized_pnl=pnl,
                   exit_reason_detailed=exit_reason_detailed)


# ---------------------------------------------------------------------------
# Group A -- the count helper
# ---------------------------------------------------------------------------

class TestRealCloseCount:
    def test_counts_real_iron_condor_closes_only(self, tmp_path):
        st = _state(tmp_path)
        _seed_closed_trade(st, "T-1", strategy="iron_condor", exit_reason_detailed="stop_loss")
        _seed_closed_trade(st, "T-2", strategy="iron_condor", exit_reason_detailed="take_profit")
        # Reconciler sentinel -- not a real close.
        _seed_closed_trade(st, "T-3", strategy="iron_condor",
                           exit_reason_detailed="stale_pending_never_filled")
        # A real close, but not a GO_LIVE_VERDICT_STRATEGIES member.
        _seed_closed_trade(st, "T-4", strategy="short_strangle", exit_reason_detailed="stop_loss")

        assert real_close_count(st._db_path) == 2

    def test_strategies_param_overrides_the_default_population(self, tmp_path):
        st = _state(tmp_path)
        _seed_closed_trade(st, "T-1", strategy="short_strangle")

        assert real_close_count(st._db_path) == 0
        assert real_close_count(st._db_path, strategies=frozenset({"short_strangle"})) == 1


# ---------------------------------------------------------------------------
# Group B -- the orchestrator method, direct
# ---------------------------------------------------------------------------

def _bare_orch(st: StateManager, min_closes: int) -> TradingOrchestrator:
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._settings = SimpleNamespace(meta_label=SimpleNamespace(min_closes_to_enable=min_closes))
    return o


class TestMetaLabelArmSufficient:
    def test_below_the_floor_is_false(self, tmp_path):
        st = _state(tmp_path)
        _seed_closed_trade(st, "T-1")
        orch = _bare_orch(st, min_closes=100)

        assert orch._meta_label_arm_sufficient() is False

    def test_at_the_floor_is_true(self, tmp_path):
        st = _state(tmp_path)
        for i in range(3):
            _seed_closed_trade(st, f"T-{i}")
        orch = _bare_orch(st, min_closes=3)

        assert orch._meta_label_arm_sufficient() is True

    def test_fails_closed_when_the_count_cannot_be_read(self, tmp_path, monkeypatch):
        st = _state(tmp_path)
        orch = _bare_orch(st, min_closes=1)

        def _boom(*a, **k):
            raise RuntimeError("db unreadable")
        monkeypatch.setattr("ait.reporting.go_live.real_close_count", _boom)

        assert orch._meta_label_arm_sufficient() is False


# ---------------------------------------------------------------------------
# Group C -- the wiring, through a real _scan_symbol
# ---------------------------------------------------------------------------

class TestEntryGateRespectsTheFloor:
    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def test_reject_is_skipped_below_the_floor(self, rig, monkeypatch):
        orch = rig.orch
        orch._settings.learning.paper_trading_mode = False
        orch._settings.meta_label.min_closes_to_enable = 100
        _seed_closed_trade(orch._state, "T-only-one")  # far below 100
        orch._meta_labeler = SimpleNamespace(
            is_trained=True,
            predict=lambda ctx: MetaSignal(take_trade=False, probability=0.05, features_used=11),
        )
        monkeypatch.setattr(orch._strategy_selector, "generate_all_signals",
                            lambda **kw: [iron_condor_signal()])
        collected: list = []

        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)

        assert collected, (
            "below the close-count floor the meta-labeler must not be "
            "consulted at all -- the candidate should reach collection "
            "exactly as if no meta-labeler existed")
        assert "meta_label_reject" not in rig.log.names()
        assert "meta_label_gate_insufficient_closes" in rig.log.names()

    async def test_reject_is_respected_at_the_floor(self, rig, monkeypatch):
        orch = rig.orch
        orch._settings.learning.paper_trading_mode = False
        orch._settings.meta_label.min_closes_to_enable = 2
        _seed_closed_trade(orch._state, "T-a")
        _seed_closed_trade(orch._state, "T-b")
        orch._meta_labeler = SimpleNamespace(
            is_trained=True,
            predict=lambda ctx: MetaSignal(take_trade=False, probability=0.05, features_used=11),
        )
        monkeypatch.setattr(orch._strategy_selector, "generate_all_signals",
                            lambda **kw: [iron_condor_signal()])
        collected: list = []

        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)

        assert not collected, (
            "at/above the close-count floor a rejecting meta-labeler must "
            "still block the candidate")
        assert "meta_label_reject" in rig.log.names()
