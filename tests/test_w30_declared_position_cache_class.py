"""W30 (R26 F3): the W22 cached-snapshot class — six findings, one seam.

Root cause: an IRREVERSIBLE action (unlinking data/HALT_UNTRACKED, deleting
declared-position state) taken on ib.positions()'s CACHE, which reads back
EMPTY on any disconnect/Gateway hiccup exactly like a genuinely flat book
(IBKRClient.get_positions_fresh's own docstring). Findings #1/#2 share that
seam; #3 is a design gap (declared positions invisible to every risk gate);
#4/#5 are adjacent leaks in the same declared-position state machine; #6 is
reconcile_now() discarding its result / running with no connection guard or
shutdown drain.

Uses the same rig as tests/test_w22_reconciler_declared_positions.py — a
real PositionReconciler + real StateManager (tmp_path sqlite) with only the
IBKR boundary faked (tests.fakes). No Gateway, no live bot, no live DB.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution import manual_positions as mp
from ait.execution.reconciler import PositionReconciler, ReconciliationResult
from ait.risk.manager import RiskManager
from tests.fakes import FakeIB, FakeIBKRClient, _option_position
from tests.test_w22_manual_positions import FakeTelegram

KEY_SHORT = "QQQ:712.0:C:2026-09-18"  # _option_position("QQQ", 712.0, "C", "20260918", ...)


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _rec(state, ib=None, telegram=None) -> PositionReconciler:
    r = PositionReconciler(FakeIBKRClient(ib or FakeIB()), state)
    if telegram is not None:
        r.set_telegram(telegram)
    return r


def _backdate_last_seen(state, key, minutes_ago):
    from datetime import timedelta

    from ait.utils.time import now_et
    state.set_state(f"manual_position_last_seen_{key}",
                    (now_et() - timedelta(minutes=minutes_ago)).isoformat())


def _halt_text(tmp_path) -> str | None:
    p = tmp_path / "data" / "HALT_UNTRACKED"
    return p.read_text() if p.exists() else None


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    """The reconciler writes data/HALT_UNTRACKED relative to cwd — run each
    test from its own tmp_path so tests never touch the real repo's file."""
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


def _risk_manager() -> RiskManager:
    pos_config = MagicMock(max_open_positions=5, max_position_pct=0.05,
                           max_portfolio_delta=1.0, max_portfolio_risk_pct=0.20)
    risk_config = MagicMock(min_confidence=0.50, max_position_risk_pct=0.10,
                            max_credit_positions=6, credit_vix_halt=28.0,
                            credit_cap_vix_tiers=[[20.0, 6], [25.0, 4], [999.0, 2]],
                            max_symbol_concentration_pct=0.20)
    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=200_000.0)
    cb = MagicMock(is_tripped=False)
    pdt = MagicMock()
    sizer = MagicMock()
    corr = MagicMock()
    return RiskManager(pos_config, risk_config, account, cb, pdt, sizer,
                       correlation_guard=corr, state=None)


# ===========================================================================
# The seam fix — findings #1 (freeze unlink) and #2 (root cause)
# ===========================================================================

class TestSeamRequiresAuthoritativeConfirmation:
    """An EMPTY ibkr_map must never authorize an irreversible action without
    a fresh, authoritative confirmation from the broker."""

    def test_halt_file_survives_when_fresh_query_cannot_confirm(self, tmp_path):
        st = _state(tmp_path)
        (tmp_path / "data" / "HALT_UNTRACKED").write_text(f"{KEY_SHORT}\n")
        ib = FakeIB()  # positions_list defaults to [] — empty cache
        rec = _rec(st, ib)
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=None)

        asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is not None, (
            "the only unlink path in the tree fired on an UNCONFIRMED empty "
            "snapshot — the exact ambiguity get_positions_fresh exists to "
            "resolve")

    def test_declared_position_state_survives_when_fresh_query_cannot_confirm(
            self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        st.set_state(f"manual_position_prompt_msgid_{KEY_SHORT}", "555")
        st.set_state(f"manual_position_last_seen_{KEY_SHORT}", "2026-09-16T09:00:00-04:00")
        st.set_state(f"manual_position_confirmed_{KEY_SHORT}", "1")
        tg = FakeTelegram()
        ib = FakeIB()  # empty cache
        rec = _rec(st, ib, telegram=tg)
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=None)

        asyncio.run(rec.reconcile())

        assert mp.is_declared(st, KEY_SHORT) is True
        assert st.get_state(f"manual_position_prompt_msgid_{KEY_SHORT}") == "555"
        assert tg.sent == [], "a declared position must not be un-tracked on a guess"

    def test_halt_file_clears_once_a_fresh_query_confirms_flat(self, tmp_path):
        st = _state(tmp_path)
        (tmp_path / "data" / "HALT_UNTRACKED").write_text(f"{KEY_SHORT}\n")
        ib = FakeIB()
        rec = _rec(st, ib)
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=[])

        asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is None, (
            "a genuinely flat book, CONFIRMED fresh, must still lift the freeze")

    def test_declared_position_cleaned_up_once_a_fresh_query_confirms_gone(
            self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        tg = FakeTelegram()
        ib = FakeIB()
        rec = _rec(st, ib, telegram=tg)
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=[])

        asyncio.run(rec.reconcile())

        assert mp.is_declared(st, KEY_SHORT) is None
        assert len(tg.sent) == 1

    def test_legacy_client_without_a_fresh_query_surface_keeps_old_behavior(
            self, tmp_path):
        """A client exposing no get_positions_fresh at all has nothing to
        confirm against — W30 must not wedge a legacy client's freeze/
        declared-position cleanup open forever."""
        st = _state(tmp_path)
        (tmp_path / "data" / "HALT_UNTRACKED").write_text(f"{KEY_SHORT}\n")
        ib = FakeIB()
        rec = _rec(st, ib)  # FakeIBKRClient defines no get_positions_fresh

        asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is None


# ===========================================================================
# Finding #3 [design]: invisible to every risk gate
# ===========================================================================

class TestDeclaredPositionRiskVisibility:
    def test_declared_naked_short_produces_a_priced_risk_row(self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -10.0)]
        rec = _rec(st, ib)

        rows = rec.declared_position_risk_rows()

        assert len(rows) == 1
        assert rows[0]["symbol"] == "QQQ"
        assert rows[0]["quantity"] == -10.0
        assert rows[0]["max_loss"] > 0

    def test_declared_naked_short_is_visible_to_the_aggregate_risk_cap(self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -10.0)]
        rec = _rec(st, ib)

        rm = _risk_manager()
        rm.update_positions(rec.declared_position_risk_rows())

        assert rm._aggregate_open_risk() > 0
        assert len(rm._open_positions) == 1

    def test_undeclared_position_is_not_synthesized(self, tmp_path):
        """Only DECLARED keys get a row — an ordinary untracked position must
        still go through the freeze path, never silently risk-accounted."""
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -10.0)]
        rec = _rec(st, ib)

        assert rec.declared_position_risk_rows() == []


# ===========================================================================
# Finding #4: never-answered keys leak
# ===========================================================================

class TestNeverAnsweredKeysDoNotLeak:
    def test_vanished_never_answered_key_is_fully_cleared(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib, telegram=tg)

        asyncio.run(rec.reconcile())  # 1st sighting, deferred
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())  # 2nd sighting -> confirmed, frozen, prompted
        assert len(tg.buttoned) == 1
        assert st.get_state(f"manual_position_prompt_msgid_{KEY_SHORT}") != ""

        ib.positions_list = []  # disappears, still never answered
        asyncio.run(rec.reconcile())

        assert mp.is_declared(st, KEY_SHORT) is None
        assert st.get_state(f"manual_position_prompt_msgid_{KEY_SHORT}") == ""
        assert st.get_state(f"manual_position_last_seen_{KEY_SHORT}") == ""
        assert not mp.is_confirmed(st, KEY_SHORT)

    def test_after_vanish_a_fresh_sighting_restarts_the_debounce_and_reprompts(
            self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib, telegram=tg)

        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())  # confirmed + 1st prompt
        assert len(tg.buttoned) == 1

        ib.positions_list = []
        asyncio.run(rec.reconcile())  # vanished, never answered -> swept

        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        result = asyncio.run(rec.reconcile())  # the identical contract reappears
        assert any("1st sighting" in d for d in result.discrepancies), (
            "the debounce must restart from scratch, not immediately re-freeze "
            "off a stale manual_position_confirmed_ flag")
        assert _halt_text(tmp_path) is None

        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())  # 2nd sighting of the NEW occurrence

        assert KEY_SHORT in _halt_text(tmp_path)
        assert len(tg.buttoned) == 2, (
            "a new prompt must be sent, not suppressed by the stale msgid")

    def test_a_no_answered_key_is_left_untouched_on_vanish(self, tmp_path):
        """Only NEVER-answered keys are widened by W30 — an explicit 'no' is
        deliberately kept as-is (matches test_w22's own contract)."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "no")
        ib = FakeIB()  # already gone
        rec = _rec(st, ib, telegram=tg)

        asyncio.run(rec.reconcile())

        assert mp.is_declared(st, KEY_SHORT) is False
        assert tg.sent == []


# ===========================================================================
# Finding #5: declared key is contract identity only
# ===========================================================================

class TestDeclaredQuantityMustMatch:
    @staticmethod
    def _declare_qty_minus_one(st, ib, tg) -> PositionReconciler:
        rec = _rec(st, ib, telegram=tg)
        asyncio.run(rec.reconcile())  # 1st sighting
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())  # 2nd sighting -> confirmed + prompted (qty=-1 stored)
        assert len(tg.buttoned) == 1
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        return rec

    def test_matching_quantity_stays_declared(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = self._declare_qty_minus_one(st, ib, tg)

        result = asyncio.run(rec.reconcile())  # still -1

        assert _halt_text(tmp_path) is None
        assert len(tg.buttoned) == 1  # no new prompt
        assert not any(KEY_SHORT in d for d in result.discrepancies)

    def test_quantity_increase_forces_freeze_and_reprompt(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = self._declare_qty_minus_one(st, ib, tg)

        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -3.0)]
        asyncio.run(rec.reconcile())

        halt = _halt_text(tmp_path)
        assert halt is not None and KEY_SHORT in halt, (
            "an undeclared size increase on an already-approved contract must freeze")
        assert len(tg.buttoned) == 2, "the size change must trigger a NEW prompt"

    def test_pre_w30_declaration_without_a_recorded_quantity_is_grandfathered(
            self, tmp_path):
        """A key declared before quantity tracking existed (direct DB write,
        no prompt ever sent) must not be treated as an automatic mismatch —
        matches test_w22_reconciler_declared_positions.py's own
        TestDeclaredPositionNeverFreezes contract."""
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib)

        result = asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is None
        assert not any(KEY_SHORT in d for d in result.discrepancies)


# ===========================================================================
# Finding #6: reconcile_now() discards its result / no connection guard
# ===========================================================================

class _FakeExecutor:
    def __init__(self) -> None:
        self.adopted: list[tuple] = []

    def adopt_exit_order(self, order_id, trade_id, reason) -> None:
        self.adopted.append((order_id, trade_id, reason))


class _FakeReconciler:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.call_count = 0

    async def reconcile(self):
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._result


class _FakeOrchestratorSelf:
    """Duck-typed stand-in for TradingOrchestrator — reconcile_now only ever
    touches self._ibkr, self._reconciler and self._executor."""

    def __init__(self, connected: bool, result=None, error: Exception | None = None) -> None:
        self._ibkr = SimpleNamespace(connected=connected)
        self._executor = _FakeExecutor()
        self._reconciler = _FakeReconciler(result=result, error=error)


class TestReconcileNowAdoptsAndGuardsConnection:
    def test_skips_the_broker_call_when_disconnected(self):
        fake = _FakeOrchestratorSelf(connected=False)

        asyncio.run(TradingOrchestrator.reconcile_now(fake))

        assert fake._reconciler.call_count == 0

    def test_adopts_kept_closing_exit_orders(self):
        recon = ReconciliationResult(
            matched=0, new_from_ibkr=0, stale_local=0, discrepancies=[],
            closing_exit_orders={"T-1": 555})
        fake = _FakeOrchestratorSelf(connected=True, result=recon)

        asyncio.run(TradingOrchestrator.reconcile_now(fake))

        assert fake._executor.adopted == [(555, "T-1", "reconcile_readopt")]

    def test_a_reconcile_failure_is_logged_not_raised(self):
        fake = _FakeOrchestratorSelf(connected=True, error=RuntimeError("boom"))

        asyncio.run(TradingOrchestrator.reconcile_now(fake))  # must not raise


class TestBackgroundTaskDrain:
    def test_drain_waits_for_pending_background_tasks(self):
        done_flag = {"done": False}

        async def _slow():
            await asyncio.sleep(0.05)
            done_flag["done"] = True

        async def _run():
            mp._fire_and_forget(_slow())
            assert not done_flag["done"]
            await mp.drain_background_tasks(timeout=2.0)
            assert done_flag["done"]

        asyncio.run(_run())

    def test_drain_is_a_noop_with_nothing_pending(self):
        asyncio.run(mp.drain_background_tasks(timeout=1.0))
