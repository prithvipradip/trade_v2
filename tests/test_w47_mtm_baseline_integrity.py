"""W47 bot-day-09 + money-flow-02 (R24/R26 register) — one fix, one key.

Both findings are the same `mtm_sod_<date>` block in
`TradingOrchestrator._monitor_positions_fast` (orchestrator.py):

- money-flow-02: a marks-missing position returned `unrealized_pnl=0.0`
  (indistinguishable from a real zero), so the MTM daily-loss brake summed a
  PHANTOM GAIN equal to the book's actual carry during an outage and
  structurally could not trip.
- bot-day-09: the lazy SOD-key capture (the only path that runs when the
  prior close's `_prestamp_mtm_baseline` never got to fire — e.g. the bot was
  down across the whole POST_MARKET window) silently stamped whatever the
  CURRENT tick's unrealized P&L was, with no log and no page.

Fix: `PositionStatus.marks_missing` (portfolio.py) now leaves the function it
was always computed in; a marks-missing tick skips the MTM evaluation AND the
lazy stamp entirely (next clean tick re-evaluates — a pure per-tick
evaluate-and-maybe-trip costs nothing to skip once). The lazy stamp itself now
prefers reconstructing the baseline from each position's last PERSISTED real
mark (`StateManager.get_position_mark` — the same numbers master.py's digest
already trusts) over the current tick's number, and pages once when it can't
(`TradingOrchestrator._capture_mtm_baseline`).

Group A drives `_capture_mtm_baseline` directly against a real StateManager
(sqlite, tmp_path) — the reconstruction/degraded-fallback/paging-gate logic
itself. Group B drives the real `_monitor_positions_fast` through the
build_smoke_orchestrator rig (test_hot_path_smoke.py) with `check_positions`
swapped for a canned PositionStatus list — proof the skip is actually WIRED
into the fast monitor, not just correct in isolation.

No inspect.getsource.
"""

from __future__ import annotations

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.portfolio import PositionStatus
from tests.test_hot_path_smoke import build_smoke_orchestrator


def _status(
    trade_id: str, unrealized_pnl: float = 0.0, marks_missing: bool = False,
) -> PositionStatus:
    return PositionStatus(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor", quantity=1,
        entry_price=2.0, current_price=1.5, unrealized_pnl=unrealized_pnl,
        pnl_pct=0.0, high_water_mark=0.0, dte=10, should_exit=False,
        exit_reason="", marks_missing=marks_missing,
    )


def _bare_orch(tmp_path) -> TradingOrchestrator:
    """A `_capture_mtm_baseline`-sized orchestrator, same idiom
    test_r14_batch2.py's TestMtmBaselinePrestamp uses for its sibling
    `_prestamp_mtm_baseline` — __new__ plus only the attributes the method
    under test touches. A REAL StateManager (not a MagicMock): the whole
    point is proving get_position_mark's sqlite round-trip, not a mock's
    say-so."""
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    orch._state = StateManager(db_path=tmp_path / "ait_state.db")
    orch._alert_gate_last = {}
    return orch


def _recorder(sink: list[str]):
    async def _notify(msg: str) -> bool:
        sink.append(msg)
        return True
    return _notify


def _seed_persisted_mark(
    state: StateManager, trade_id: str, unrealized_pnl: float, pnl_pct: float = 0.0,
) -> None:
    """update_position_mark is an UPDATE against an existing open_positions
    row -- with no matching row it silently affects zero rows (sqlite raises
    nothing), so get_position_mark then reads back None, not the value just
    "written". insert_open_position first, exactly as the real entry-fill
    path does before the first tick ever marks a new position."""
    state.insert_open_position(trade_id, "SPY", "iron_condor", 1, 2.0)
    state.update_position_mark(trade_id, unrealized_pnl, pnl_pct)


class TestCaptureMtmBaselineReconstruction:
    async def test_zero_open_positions_returns_fallback_untouched(self, tmp_path):
        orch = _bare_orch(tmp_path)
        pages: list[str] = []
        orch._send_notification = _recorder(pages)

        result = await orch._capture_mtm_baseline([], -999.0)

        assert result == -999.0
        assert pages == [], "a clean start (no positions) must never page"

    async def test_all_positions_have_persisted_marks_reconstructs_from_them(self, tmp_path):
        orch = _bare_orch(tmp_path)
        pages: list[str] = []
        orch._send_notification = _recorder(pages)
        _seed_persisted_mark(orch._state, "T-1", 120.0, 0.24)
        _seed_persisted_mark(orch._state, "T-2", -30.0, -0.05)

        result = await orch._capture_mtm_baseline(
            [_status("T-1"), _status("T-2")], fallback_unrealized=-500.0,
        )

        assert result == pytest.approx(90.0), (
            "must reconstruct from the persisted marks (120 + -30), not fall "
            "back to the current tick's number")
        assert pages == [], "a full, real history must not page as degraded"

    async def test_one_position_with_no_persisted_mark_falls_back_and_pages(self, tmp_path):
        orch = _bare_orch(tmp_path)
        pages: list[str] = []
        orch._send_notification = _recorder(pages)
        _seed_persisted_mark(orch._state, "T-1", 120.0, 0.24)
        # T-2 never marked -- a genuinely new position, nothing to
        # reconstruct from.

        result = await orch._capture_mtm_baseline(
            [_status("T-1"), _status("T-2")], fallback_unrealized=-500.0,
        )

        assert result == -500.0, (
            "a partial history must fall back wholesale, not silently mix a "
            "real mark with a missing one treated as zero")
        assert len(pages) == 1
        assert "MTM BASELINE DEGRADED" in pages[0]

    async def test_degraded_page_is_gated_across_consecutive_ticks(self, tmp_path):
        orch = _bare_orch(tmp_path)
        pages: list[str] = []
        orch._send_notification = _recorder(pages)
        positions = [_status("T-1")]  # never marked

        await orch._capture_mtm_baseline(positions, -100.0)
        await orch._capture_mtm_baseline(positions, -100.0)

        assert len(pages) == 1, "the degraded page must not resend every tick"


class TestMonitorPositionsFastMarksMissingWiring:
    """Proof the skip is wired into the real fast monitor, driven through the
    same rig test_hot_path_smoke.py and scripts/smoke_deploy.py both use."""

    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def test_marks_missing_skips_both_evaluation_and_the_lazy_stamp(self, rig, monkeypatch):
        orch = rig.orch
        calls = {"n": 0}
        real_check = orch._circuit_breaker.check_daily_loss_mtm

        def _spy(*a, **k):
            calls["n"] += 1
            return real_check(*a, **k)
        monkeypatch.setattr(orch._circuit_breaker, "check_daily_loss_mtm", _spy)

        async def _fake_check_positions():
            return [_status("T-1", marks_missing=True)]
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        await orch._monitor_positions_fast()
        await rig.drain_notifications()

        assert calls["n"] == 0, "the MTM brake must not evaluate on a marks-missing tick"
        from ait.utils.time import now_et
        sod_key = f"mtm_sod_{now_et().date().isoformat()}"
        assert orch._state.get_state(sod_key, "") == "", (
            "the lazy SOD stamp must not fire on a marks-missing tick either "
            "-- it would lock in a phantom baseline")
        assert "mtm_check_skipped_marks_missing" in rig.log.names()
        rig.log.assert_clean("_monitor_positions_fast (marks missing)")

    async def test_clean_marks_still_evaluates_and_stamps_as_before(self, rig, monkeypatch):
        orch = rig.orch

        async def _fake_check_positions():
            return [_status("T-1", unrealized_pnl=42.0, marks_missing=False)]
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        await orch._monitor_positions_fast()
        await rig.drain_notifications()

        from ait.utils.time import now_et
        sod_key = f"mtm_sod_{now_et().date().isoformat()}"
        assert orch._state.get_state(sod_key, "") != "", (
            "a clean tick (no marks-missing positions) must still stamp the "
            "baseline, exactly as before this wave")
        assert "mtm_check_skipped_marks_missing" not in rig.log.names()
        rig.log.assert_clean("_monitor_positions_fast (clean)")

    async def test_clean_tick_reconstructs_from_a_real_persisted_mark(self, rig, monkeypatch):
        """The orchestrator-level proof that a real prior mark, not the
        current tick's number, wins -- Group A proves the helper; this
        proves _monitor_positions_fast actually calls it with real state."""
        orch = rig.orch
        _seed_persisted_mark(orch._state, "T-1", 77.0, 0.15)

        async def _fake_check_positions():
            # Deliberately a DIFFERENT current-tick number than the
            # persisted mark, so reconstruction vs. fallback is unambiguous.
            return [_status("T-1", unrealized_pnl=-9_000.0, marks_missing=False)]
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        await orch._monitor_positions_fast()
        await rig.drain_notifications()

        from ait.utils.time import now_et
        sod_key = f"mtm_sod_{now_et().date().isoformat()}"
        assert orch._state.get_state(sod_key, "") == "77.0", (
            "the baseline must be the persisted real mark, not the current "
            "tick's unrelated number")
        assert "mtm_baseline_reconstructed_from_persisted_marks" in rig.log.names()
        assert "mtm_baseline_degraded_post_open" not in rig.log.names()
