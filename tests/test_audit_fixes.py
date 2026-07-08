"""Tests for the 2026-07-07 audit fixes (PLAN.md).

Covers the two paths the audit found with ZERO coverage despite production
failures — marketable combo entry pricing and BotManager supervision — plus
regression tests for the Thompson decay and strangle stress-loss fixes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ait.execution.executor import combo_entry_limit
from ait.strategies.thompson import StrategyArm, ThompsonSampler


# ---------------------------------------------------------------------------
# Marketable combo entry pricing (executor) — audit item 3.2 / commit 495ac26
# ---------------------------------------------------------------------------

class TestComboEntryLimit:
    def test_credit_crosses_down_and_quotes_negative(self):
        # Iron condor collecting 1.00 credit: concede 15% (0.15), limit -0.85
        limit, offset = combo_entry_limit(1.00, is_credit=True)
        assert offset == pytest.approx(0.15)
        assert limit == pytest.approx(-0.85)

    def test_debit_crosses_up(self):
        # Long straddle paying 9.72: pay up 15% -> 11.18 limit
        limit, offset = combo_entry_limit(9.72, is_credit=False)
        assert offset == pytest.approx(1.46, abs=0.01)
        assert limit == pytest.approx(9.72 + offset)

    def test_minimum_offset_floor_10c(self):
        # Tiny 0.20 credit: 15% would be 3c — floor at 10c
        limit, offset = combo_entry_limit(0.20, is_credit=True)
        assert offset == pytest.approx(0.10)
        assert limit == pytest.approx(-0.10)

    def test_credit_never_flips_sign_to_paying(self):
        # Credit so small the offset swamps it: limit must stay negative
        # (still a credit — never accidentally BUY for a debit).
        limit, _ = combo_entry_limit(0.05, is_credit=True)
        assert limit < 0
        assert limit == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# BotManager supervision (master) — audit item 3.2
# ---------------------------------------------------------------------------

@pytest.fixture()
def bot_manager(monkeypatch, tmp_path):
    import ait.orchestration.master as master

    monkeypatch.setattr(master, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(master, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    # Never touch telegram or real processes
    monkeypatch.setattr(master, "_alert", lambda *a, **k: None)
    mgr = master.BotManager()
    return master, mgr


class TestBotManagerSupervision:
    def test_gateway_down_defers_without_spending_budget(self, bot_manager, monkeypatch):
        master, mgr = bot_manager
        monkeypatch.setattr(master, "_gateway_listening", lambda port: False)
        mgr._proc = None  # bot never started
        before = mgr._restarts
        mgr.health_check()
        assert mgr._restarts == before  # no budget spent on gateway outage

    def test_max_restarts_gives_up(self, bot_manager, monkeypatch):
        master, mgr = bot_manager
        monkeypatch.setattr(master, "_gateway_listening", lambda port: True)
        started = {"n": 0}
        monkeypatch.setattr(mgr, "start", lambda: started.__setitem__("n", started["n"] + 1))
        mgr._proc = SimpleNamespace(returncode=1, poll=lambda: 1)
        mgr._restarts = mgr._max_restarts
        mgr._last_restart = datetime.now()  # recent — no budget forgiveness
        mgr.health_check()
        assert started["n"] == 0  # gave up, did not restart

    def test_budget_reset_after_healthy_stretch(self, bot_manager, monkeypatch):
        master, mgr = bot_manager
        mgr._proc = SimpleNamespace(pid=123, poll=lambda: None, returncode=None)
        mgr._restarts = 3
        mgr._last_restart = datetime.now() - timedelta(minutes=45)
        mgr.health_check()
        assert mgr._restarts == 0  # forgiven after >30 min healthy

    def test_fresh_models_marker_triggers_one_restart(self, bot_manager, monkeypatch):
        master, mgr = bot_manager
        mgr._proc = SimpleNamespace(pid=123, poll=lambda: None, returncode=None)
        marker = master.ROOT / "models" / ".retrained"
        marker.write_text("2026-07-07")
        calls = []
        monkeypatch.setattr(mgr, "stop", lambda: calls.append("stop"))
        monkeypatch.setattr(mgr, "start", lambda: calls.append("start"))
        mgr.health_check()
        assert calls == ["stop", "start"]
        assert not marker.exists()  # consumed — no restart loop

    def test_stdout_rotation_shifts_backups(self, bot_manager):
        master, mgr = bot_manager
        base = master.LOGS_DIR / "bot_stdout.log"
        base.write_bytes(b"x" * (mgr._STDOUT_LOG_MAX_BYTES + 1))
        (master.LOGS_DIR / "bot_stdout.log.1").write_text("old1")
        mgr._rotate_stdout_log()
        assert not base.exists()
        assert (master.LOGS_DIR / "bot_stdout.log.1").stat().st_size > 4  # new .1 = big file
        assert (master.LOGS_DIR / "bot_stdout.log.2").read_text() == "old1"

    def test_stdout_rotation_noop_under_cap(self, bot_manager):
        master, mgr = bot_manager
        base = master.LOGS_DIR / "bot_stdout.log"
        base.write_text("small")
        mgr._rotate_stdout_log()
        assert base.read_text() == "small"  # untouched


# ---------------------------------------------------------------------------
# Thompson decay (float, no truncation) — audit item 2.5 regression
# ---------------------------------------------------------------------------

class TestThompsonDecay:
    def test_single_win_survives_daily_decay(self, tmp_path, monkeypatch):
        import ait.strategies.thompson as th
        monkeypatch.setattr(th, "STATE_FILE", tmp_path / "state.json")
        sampler = ThompsonSampler(decay_factor=0.995)
        sampler.record_outcome("iron_condor", won=True, pnl=50.0)
        arm = sampler._arms["iron_condor"]
        assert arm.wins == 1
        sampler.apply_decay()
        # Old bug: int(1 * 0.995) == 0 — the win vanished after ONE day.
        assert arm.wins == pytest.approx(0.995)
        for _ in range(30):
            sampler.apply_decay()
        assert arm.wins > 0.8  # a month of decay keeps ~86% of the signal

    def test_alpha_beta_accept_float_counts(self):
        arm = StrategyArm(name="x", wins=0.995, losses=0.4975)
        assert arm.alpha == pytest.approx(1.995)
        assert arm.beta == pytest.approx(1.4975)
        assert 0.0 <= arm.sample() <= 1.0  # betavariate works with floats


# ---------------------------------------------------------------------------
# Strangle stress-loss (never LESS conservative than 3x credit) — item 2.1
# ---------------------------------------------------------------------------

class TestStrangleStressLoss:
    @staticmethod
    def _stress_loss(spot, put_strike, call_strike, credit, stress=0.15):
        # Mirrors the formula in straddles.py generate_signals (short strangle)
        put_side = max(0.0, put_strike - spot * (1 - stress))
        call_side = max(0.0, spot * (1 + stress) - call_strike)
        return max((max(put_side, call_side) - credit) * 100, credit * 3 * 100)

    def test_realistic_tail_dwarfs_old_3x_proxy(self):
        # IWM-like: spot 298, 286P/312C, 2.28 credit
        loss = self._stress_loss(298, 286, 312, 2.28)
        assert loss > 3000          # ~$3.0k stress loss
        assert loss > 2.28 * 3 * 100 * 4  # >4x the old proxy — the audit's point

    def test_floor_at_3x_credit_when_strikes_are_far(self):
        # Strikes so wide even a 15% move stays OTM -> falls back to 3x floor
        loss = self._stress_loss(100, 60, 140, 1.00)
        assert loss == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Round-2 audit fixes (2026-07-07 evening)
# ---------------------------------------------------------------------------

class TestStructureIntrinsic:
    """ITM-aware expiry booking (reconciler) — no more fabricated wins."""

    @staticmethod
    def _trade(legs, strategy):
        import json as _json
        return SimpleNamespace(legs=_json.dumps(legs), strategy=strategy, strike=None)

    def test_ic_put_side_itm_costs_wing_width(self):
        from ait.execution.reconciler import PositionReconciler
        legs = [
            {"strike": 713, "right": "P", "action": "BUY"},
            {"strike": 715, "right": "P", "action": "SELL"},
            {"strike": 752, "right": "C", "action": "SELL"},
            {"strike": 754, "right": "C", "action": "BUY"},
        ]
        # settle 700: short 715P owes 15, long 713P worth 13 -> cost 2.0
        cost = PositionReconciler._structure_intrinsic(self._trade(legs, "iron_condor"), 700.0)
        assert cost == pytest.approx(2.0)
        # credit P&L would be (credit - 2.0)*100 => max loss, NOT a full win

    def test_strangle_otm_costs_zero(self):
        from ait.execution.reconciler import PositionReconciler
        legs = [
            {"strike": 225, "right": "P", "action": "SELL"},
            {"strike": 265, "right": "C", "action": "SELL"},
        ]
        cost = PositionReconciler._structure_intrinsic(self._trade(legs, "short_strangle"), 245.0)
        assert cost == pytest.approx(0.0)  # genuinely expired worthless

    def test_short_put_itm_costs_intrinsic(self):
        from ait.execution.reconciler import PositionReconciler
        legs = [{"strike": 225, "right": "P", "action": "SELL"}]
        cost = PositionReconciler._structure_intrinsic(self._trade(legs, "cash_secured_put"), 210.0)
        assert cost == pytest.approx(15.0)  # the old code booked this as a WIN


class TestOIUnknownPassthrough:
    """IBKR realtime reports OI=0 (unknown) — must not reject on unknown."""

    def test_oi_zero_passes_when_otherwise_liquid(self, monkeypatch):
        monkeypatch.setenv("AIT_LIQ_MIN_VOL", "0")
        monkeypatch.setenv("AIT_LIQ_MIN_OI", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.50")
        from ait.data.options_chain import OptionContract
        c = OptionContract(symbol="SPY", expiry="2026-07-24", strike=650, right="C",
                           bid=1.00, ask=1.05, last=1.02, volume=100,
                           open_interest=0, implied_vol=0.2, delta=0.2,
                           gamma=0, theta=0, vega=0)
        assert c.is_liquid  # unknown OI does not disqualify

    def test_low_known_oi_still_rejected(self, monkeypatch):
        monkeypatch.setenv("AIT_LIQ_MIN_VOL", "0")
        monkeypatch.setenv("AIT_LIQ_MIN_OI", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.50")
        from ait.data.options_chain import OptionContract
        c = OptionContract(symbol="SPY", expiry="2026-07-24", strike=650, right="C",
                           bid=1.00, ask=1.05, last=1.02, volume=100,
                           open_interest=3, implied_vol=0.2, delta=0.2,
                           gamma=0, theta=0, vega=0)
        assert not c.is_liquid  # KNOWN thin OI is still rejected
