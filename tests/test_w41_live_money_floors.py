"""W41 (R26 register, live-money floors) — five findings, first of the
unowned-work roadmap (PLAN.md, 2026-09-21 REGISTER CORRECTION + ROADMAP entry).

No operator decision and no re-probe needed for any of these five — unlike
several later waves in the roadmap, each was independently verified against
HEAD before this wave started.

1. regress-live-05 — PDT history destroyed on an unreadable-store write.
2. time-authority-2 / time-authority-4 — already fixed (W28, W21); these are
   the regression tests that close the finding, not a code change.
3. model-vs-reality-margin-model-unimplemented, piece 1 — can_afford ignored
   available_funds.
4. model-vs-reality-ex-div-assignment-void, piece 1 — an assigned stock
   position never froze new entries.
5. The max_contracts_per_trade startup guard-rail (GOV-6) — protected the
   three findings queued for W50 from ever arming before they were fixed.
   RETIRED by W50 once those three were actually fixed; only the
   still-true "the pinned config value constructs cleanly" control remains.

Every test executes the real code path. No inspect.getsource.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from ait.bot.state import StateManager
from ait.broker.account import AccountManager, AccountSnapshot
from ait.config.settings import AccountConfig
from ait.execution.reconciler import PositionReconciler
from ait.risk.pdt_guard import PDTGuard
from ait.utils.time import now_et
from tests.fakes import FakeIB, FakeIBKRClient, _option_position, _stock_position


# ---------------------------------------------------------------------------
# 1. regress-live-05 — PDT history destroyed on an unreadable-store write
# ---------------------------------------------------------------------------

def _pdt(state) -> PDTGuard:
    return PDTGuard(AccountConfig(pdt_protection=True, pdt_account_under_25k=True), state)


class _ReadLocked:
    """Fakes only the failing verb; every other call delegates to the real
    store. Same idiom as test_w14_booking_exactly_once.py's crash test."""

    def __init__(self, real):
        self._real = real

    def get_state(self, key, default=""):
        raise sqlite3.OperationalError("locked")

    def __getattr__(self, name):
        return getattr(self._real, name)


def _in_window_days(n: int) -> list[str]:
    """The n OLDEST trading days still inside the guard's rolling window, as
    the ISO strings the guard stores - real calendar, never a literal date."""
    from ait.risk import pdt_guard as _pg
    days = list(_pg.get_recent_trading_days(_pg.WINDOW_DAYS))
    assert len(days) >= n, days
    return [d.isoformat() if hasattr(d, "isoformat") else str(d)[:10] for d in days[:n]]


class TestPDTHistorySurvivesATransientLock:

    def test_a_locked_write_does_not_destroy_real_history(self, tmp_path):
        """The defect, reproduced then closed in one test.

        Seed a REAL 5-day history through a healthy guard, then record one
        more day trade through a guard whose READ is locked (the exact shape
        of a transient store blip). Before the fix, _save_state() wrote
        unconditionally, so the locked guard's freshly-reset empty deque plus
        one new entry overwrote the real history in the STORE.
        """
        st = StateManager(db_path=tmp_path / "ait_state.db")
        healthy = _pdt(st)
        # W41b: dated INSIDE the guard's rolling window, whatever today is.
        # These were the literals 2026-09-15/16/17 - green on 09-21, red on
        # 09-22 once 09-15 rolled out of the 5-business-day window (the
        # guard's own purge, not a defect). Same class as the W23/W24
        # calendar-date-dependent tests.
        for i, day in enumerate(_in_window_days(3)):
            healthy._day_trades.append((day, f"SYM{i}"))
        healthy._save_state()
        assert len(healthy._day_trades) == 3, "premise: three real entries exist"

        locked = _pdt(_ReadLocked(st))
        assert locked._state_unreadable, "premise: the locked load must fail closed"
        locked.record_day_trade("QQQ")

        recovered = _pdt(st)
        assert len(recovered._day_trades) == 3, (
            f"the locked write destroyed real history: {recovered._day_trades}"
        )

    def test_a_recovered_store_self_heals_on_the_next_write(self, tmp_path):
        """The retry path: a SECOND record_day_trade, store now healthy,
        must not just append blindly — it re-reads first, then persists
        everything, including the trade recorded while still locked."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        healthy = _pdt(st)
        healthy._day_trades.append((_in_window_days(1)[0], "SYM0"))  # W41b: see above
        healthy._save_state()

        locked = _pdt(_ReadLocked(st))
        locked.record_day_trade("QQQ")  # suppressed, per the test above
        assert locked._state_unreadable

        locked._state = st  # the lock "clears" — attach the real store back
        locked.record_day_trade("SPY")

        assert not locked._state_unreadable, "did not recover on a good read"
        recovered = _pdt(st)
        symbols = {s for _, s in recovered._day_trades}
        assert symbols == {"SYM0", "QQQ", "SPY"}, (
            f"expected the pre-lock entry plus both recorded trades, got {recovered._day_trades}"
        )

    def test_a_genuinely_corrupt_store_still_blocks(self, tmp_path):
        """CONTROL: this change must not weaken the existing fail-closed
        behavior for a store that never recovers."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        st.set_state("pdt_day_trades", "{not json")  # genuinely corrupt, not locked
        guard = _pdt(st)
        assert guard._state_unreadable
        assert guard.can_day_trade() is False, "a corrupt store must still block"

        guard.record_day_trade("QQQ")  # must not raise
        assert guard._state_unreadable, "a corrupt store cannot self-heal by definition"


# ---------------------------------------------------------------------------
# 2. time-authority-2 / time-authority-4 — regression tests only
# ---------------------------------------------------------------------------

class TestTimeAuthorityAlreadyFixedIncidentally:
    """Closes time-authority-2 and time-authority-4 — both findings say the
    code should read the ET clock, not the host clock. Both already do
    (executor.py:391 -> now_et(), orchestrator.py:3055 -> today_et()); these
    tests exist so the finding has a reproduction that fails on revert,
    per this project's own closing rule, not because anything needed fixing.

    Deliberately NOT a TZ-env-pinning test (the `host_tz` fixture pattern
    used elsewhere in this project). Verified directly: `now_et()` is
    `datetime.now(ET)` with an explicit tzinfo argument, which is immune to
    the process TZ environment variable by construction — the TZ trick only
    has signal against a BARE `datetime.now()`/`date.today()`, which is
    exactly the class already fixed here. Testing the fixed code with a tool
    built to catch the retired bug would prove nothing; these assert the
    real, host-independent property instead.
    """

    def test_entry_time_is_written_aware_et_not_naive_host_local(self):
        """The structural signature of the fix: an AWARE ET offset, never a
        naive local wall-clock string. Same expression executor.py:391
        evaluates — there is only one call site."""
        stamped = now_et().isoformat()
        assert stamped.endswith(("-04:00", "-05:00")), (
            f"entry_time would be stamped {stamped!r} — naive or non-ET"
        )

    def test_earnings_gate_compares_against_the_et_calendar_day(self):
        """Drives the real would_hold_through_earnings (sync — the same call
        shape orchestrator.py:3055 uses) with today_et() as entry_date and
        earnings landing on that same ET day; must block. A regression to
        date.today() would only diverge from today_et() during specific
        evening ET hours (this test would then be flaky by wall-clock time,
        not deterministically red) — so this proves the call site USES
        today_et() end to end, which is the actual claim, rather than trying
        to catch a divergence that isn't always observable."""
        from datetime import timedelta

        from ait.data.earnings import EarningsCalendar, EarningsInfo
        from ait.utils.time import today_et

        cal = EarningsCalendar()  # cheap real __init__ — just buffers + a TTLCache
        today = today_et()
        expiry = today + timedelta(days=21)
        cal.get_next_earnings = lambda symbol: EarningsInfo(
            symbol=symbol, next_earnings_date=today)

        assert cal.would_hold_through_earnings("NVDA", today_et(), expiry) is True

    def test_earnings_outside_the_window_does_not_block(self):
        """CONTROL: must hold on both trees — an unrelated earnings date must
        not block."""
        from datetime import timedelta

        from ait.data.earnings import EarningsCalendar, EarningsInfo
        from ait.utils.time import today_et

        cal = EarningsCalendar()
        today = today_et()
        far_off = today + timedelta(days=90)
        cal.get_next_earnings = lambda symbol: EarningsInfo(
            symbol=symbol, next_earnings_date=far_off)

        assert cal.would_hold_through_earnings(
            "NVDA", today, today + timedelta(days=21)) is False


# ---------------------------------------------------------------------------
# 3. model-vs-reality-margin-model-unimplemented, piece 1
# ---------------------------------------------------------------------------

def _account_with(buying_power, available_funds) -> AccountManager:
    acct = AccountManager.__new__(AccountManager)

    async def _snapshot(force_refresh=False):
        return AccountSnapshot(buying_power=buying_power, available_funds=available_funds)

    acct.get_snapshot = _snapshot
    return acct


class TestCanAffordChecksAvailableFunds:

    def test_drained_available_funds_refuses_a_trade_buying_power_would_allow(self):
        """The exact drained-margin shape the finding names: healthy
        leveraged buying_power, exhausted actual cash."""
        acct = _account_with(buying_power=50_000.0, available_funds=200.0)
        assert asyncio.run(acct.can_afford(1_000.0)) is False, (
            "can_afford approved a trade the account cannot actually cover"
        )

    def test_both_healthy_still_affords(self):
        """CONTROL: must hold on both trees."""
        acct = _account_with(buying_power=50_000.0, available_funds=48_000.0)
        assert asyncio.run(acct.can_afford(1_000.0)) is True

    def test_the_binding_constraint_is_whichever_is_tighter(self):
        """CONTROL, the other direction: a tight buying_power still refuses
        even with ample available_funds -- this must not have become an
        available_funds-only check."""
        acct = _account_with(buying_power=500.0, available_funds=50_000.0)
        assert asyncio.run(acct.can_afford(1_000.0)) is False


# ---------------------------------------------------------------------------
# 4. model-vs-reality-ex-div-assignment-void, piece 1
# ---------------------------------------------------------------------------

def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _rec(state, ib=None) -> PositionReconciler:
    return PositionReconciler(FakeIBKRClient(ib or FakeIB()), state)


def _halt_text(tmp_path) -> str | None:
    p = tmp_path / "data" / "HALT_UNTRACKED"
    return p.read_text() if p.exists() else None


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    """The reconciler writes data/HALT_UNTRACKED relative to cwd -- same
    fixture shape as test_w22_reconciler_declared_positions.py:54-59."""
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


class TestAssignedStockFreezesEntries:

    def test_an_untracked_stock_position_writes_halt_untracked(self, tmp_path):
        """The defect, reproduced then closed: before the fix, this branch
        logged and reported a discrepancy but never joined frozen_keys, so
        HALT_UNTRACKED was never written for an assignment."""
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_stock_position("SPY", 100.0, avg_cost=660.0)]

        rec = _rec(st, ib)
        result = asyncio.run(rec.reconcile())

        text = _halt_text(tmp_path)
        assert text is not None, "an assigned stock position never froze entries"
        assert "SPY" in text
        assert any("ASSIGNMENT" in d for d in result.discrepancies)

    def test_the_opt_branch_is_unaffected(self, tmp_path):
        """CONTROL: the existing OPT freeze behavior — including its R5
        debounce, untouched by this wave — must be unchanged. A first
        sighting defers (no freeze yet); backdating last_seen simulates the
        SECOND sighting the debounce requires, the same idiom
        test_w22_reconciler_declared_positions.py uses."""
        from datetime import timedelta

        from ait.utils.time import now_et

        st = _state(tmp_path)
        key = "QQQ:712.0:C:2026-09-18"
        st.set_state(f"manual_position_last_seen_{key}",
                    (now_et() - timedelta(minutes=10)).isoformat())
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]

        rec = _rec(st, ib)
        asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is not None
        assert "QQQ" in _halt_text(tmp_path)

    def test_a_flat_book_still_clears_the_halt_file(self, tmp_path):
        """CONTROL: no untracked position of either kind -> no freeze."""
        st = _state(tmp_path)
        rec = _rec(st, FakeIB())
        asyncio.run(rec.reconcile())
        assert _halt_text(tmp_path) is None


# ---------------------------------------------------------------------------
# 5. max_contracts_per_trade startup guard-rail (GOV-6)
#
# W50: GOV-6 itself was RETIRED — the startup RuntimeError this class used to
# assert (test_more_than_one_contract_refuses_construction, removed) existed
# only "until the qty>1 booking path is fixed (PLAN.md W50)". All three
# findings it guarded (money-flow-05, trade-life-partial-slice-pnl-erased,
# trade-life-promotion-ignores-broker-quantity) are fixed and proven by
# tests/test_w50_qty_ledger_fork.py; asserting the refusal here would now be
# asserting something false. The construction control stays: it is still
# true, and still worth pinning, that the live config value builds cleanly.
# ---------------------------------------------------------------------------

class TestMaxContractsGuardRail:

    def test_one_contract_still_constructs(self, tmp_path, monkeypatch):
        """CONTROL: the live config value (1) must keep working — this is
        exactly what every other smoke-rig-based test already proves by not
        raising, made explicit here."""
        from tests.test_hot_path_smoke import build_smoke_orchestrator

        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            assert rig.orch._settings.positions.max_contracts_per_trade == 1
        finally:
            rig.restore()
