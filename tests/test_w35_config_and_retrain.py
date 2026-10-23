"""W35 (R26 fix-set F8, config homes and the retrain spec) + the deploy-order
hazard registered in W33.

  * range.pkl flapped daily between two specs: the live bot builds its range
    model at ``live_range_spec()`` (threshold 0.05, horizon 9 — the reachable
    hold), while the supervisor's 07:30 retrain script hardcoded
    ``horizon_days=30`` — so every retrain rebuilt the artifact at a spec the
    live gate's confidence floor was never calibrated for.
  * a long-lived process re-reads config.yaml at runtime (commission estimate,
    chain liquidity, the supervisor's first page) against the settings MODEL
    it imported at start. ``extra=forbid`` then turns any new yaml key into a
    ValidationError inside every running process until it restarts — W33 hit
    this live on 2026-09-18. A live process now freezes its settings.
  * the last-15-minutes entry blackout was a bare ``900`` with no config home.
  * the Gateway start script had two unrelated env authorities.

Real subprocesses wherever the claim is about a separate, long-lived process.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ait.orchestration.master as M
from ait.config.settings import load_settings

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _run(code: str, cwd: Path, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    e["PYTHONPATH"] = str(SRC)
    e["PYTHONIOENCODING"] = "utf-8"
    e.update(env)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd,
                          env=e, capture_output=True, text=True,
                          encoding="utf-8", timeout=180)


# --------------------------------------------------------------------------
# the retrain builds the range model the live gate actually asks about
# --------------------------------------------------------------------------
class TestTheRetrainUsesTheLiveRangeSpec:
    def test_the_supervisors_own_script_builds_the_live_spec(self, monkeypatch, tmp_path):
        """Capture the script retrain_models() really launches and EXECUTE it
        (training and I/O stubbed). HEAD built RangePredictor(0.05, 30)."""
        from ait.ml.range_spec import live_range_spec
        from ait.ml.trainer import ModelTrainer

        monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
        launched: dict = {}

        class _Captured(Exception):
            pass

        def _popen(argv, **kw):
            launched["argv"] = argv
            raise _Captured()

        monkeypatch.setattr(M.subprocess, "Popen", _popen)
        M.retrain_models()                      # swallows the exception itself
        script = launched["argv"][-1]

        built: dict = {}

        async def _no_training(self, symbols):
            built["trainer"] = self
            return {}

        monkeypatch.setattr(ModelTrainer, "train_all_symbols", _no_training)
        # W89: the script now opens its own IB session and pre-fetches every series, so the
        # IB boundary is faked (a connected client + a service that returns one bar frame).
        import pandas as pd
        from unittest.mock import AsyncMock

        class _Svc:
            def __init__(self, *a, **kw):
                pass

            async def get_historical(self, symbol, days=0, **kw):
                return pd.DataFrame({"Close": [1.0]})

        fake_ib = MagicMock()
        fake_ib.disconnect = AsyncMock()
        monkeypatch.setattr("ait.orchestration.retrain_ib.connect_from_settings",
                            AsyncMock(return_value=fake_ib))
        monkeypatch.setattr("ait.data.market_data.MarketDataService", _Svc)
        monkeypatch.setattr("ait.data.historical.HistoricalDataStore", MagicMock())
        monkeypatch.chdir(REPO)
        exec(compile(script, "<supervisor retrain script>", "exec"),
             {"__name__": "__main__"})

        trainer = built["trainer"]
        want = live_range_spec(load_settings(REPO / "config.yaml"))
        got = (trainer._range_predictor._spec_threshold,
               trainer._range_predictor._spec_horizon)
        assert got == want, (
            f"the daily retrain rebuilds range.pkl at {got}; the live gate and "
            f"the research that calibrated its floor use {want}")
        assert trainer.check_range_spec() is True


# --------------------------------------------------------------------------
# a live process reads its config once
# --------------------------------------------------------------------------
_LIVE_PROCESS = """
    from ait.config import settings as S
    s = S.load_settings("config.yaml")
    {freeze}
    # an operator (or a deploy) adds a key this process's model has never seen
    text = open("config.yaml", encoding="utf-8").read()
    text = text.replace("\\nexit:\\n", "\\nexit:\\n  a_key_from_a_later_wave: 1\\n", 1)
    assert "a_key_from_a_later_wave" in text
    open("config.yaml", "w", encoding="utf-8").write(text)
    try:
        again = S.load_settings("config.yaml")
        print("RELOAD_OK", again.exit.exit_cross_amount == s.exit.exit_cross_amount)
    except Exception as e:
        print("RELOAD_FAILED", type(e).__name__)
"""


class TestALiveProcessIsImmuneToConfigEdits:
    def _cfg(self, tmp_path):
        shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")

    def test_the_hazard_is_real_without_the_freeze(self, tmp_path):
        """Control — this is what hit the live bot's window on 2026-09-18."""
        self._cfg(tmp_path)
        r = _run(_LIVE_PROCESS.format(freeze="pass"), tmp_path)
        assert "RELOAD_FAILED ValidationError" in r.stdout, r.stdout + r.stderr[-400:]

    def test_a_frozen_process_keeps_running_on_its_snapshot(self, tmp_path):
        self._cfg(tmp_path)
        r = _run(_LIVE_PROCESS.format(freeze="S.freeze_settings(s)"), tmp_path)
        assert "RELOAD_OK True" in r.stdout, r.stdout + r.stderr[-400:]

    def test_each_caller_gets_its_own_copy(self, monkeypatch):
        from ait.config import settings as S
        s = load_settings(REPO / "config.yaml")
        monkeypatch.setattr(S, "_FROZEN", None, raising=False)
        S.freeze_settings(s)
        try:
            a = S.load_settings(REPO / "config.yaml")
            a.exit.exit_cross_amount = 0.49
            b = S.load_settings(REPO / "config.yaml")
            assert b.exit.exit_cross_amount == s.exit.exit_cross_amount
        finally:
            S._FROZEN = None

    def test_the_bot_entry_point_freezes(self, tmp_path):
        self._cfg(tmp_path)
        r = _run("""
            import argparse, asyncio
            import ait.main as main_mod
            from ait.config import settings as S

            class _Stop(Exception):
                pass
            def _stop(*a, **k):
                raise _Stop()
            main_mod.setup_logging = _stop          # first call after the load
            try:
                asyncio.run(main_mod.run_bot(argparse.Namespace(
                    config="config.yaml", paper=True)))
            except _Stop:
                pass
            print("FROZEN", S._FROZEN is not None,
                  getattr(S._FROZEN.trading, "mode", None))
        """, tmp_path)
        assert "FROZEN True paper" in r.stdout, r.stdout + r.stderr[-600:]

    def test_the_supervisor_entry_point_freezes(self, tmp_path):
        self._cfg(tmp_path)
        r = _run(f"""
            from pathlib import Path
            import ait.orchestration.master as M
            from ait.config import settings as S
            M.LOGS_DIR = Path(r"{tmp_path}")
            M._acquire_singleton_lock = lambda: False   # stop right after startup
            M.main()
            print("FROZEN", S._FROZEN is not None)
        """, tmp_path)
        assert "FROZEN True" in r.stdout, r.stdout + r.stderr[-600:]


# --------------------------------------------------------------------------
# the entry blackout before the close has a config home
# --------------------------------------------------------------------------
class TestTheEntryBlackoutIsConfigured:
    def _sched(self, minutes, remaining_s, monkeypatch):
        from ait.bot.scheduler import MarketScheduler
        s = load_settings(REPO / "config.yaml")
        s.trading.no_new_entries_minutes_before_close = minutes
        sched = MarketScheduler(s)
        monkeypatch.setattr(sched, "seconds_until_close", lambda: remaining_s)
        return sched

    def test_the_default_is_the_old_fifteen_minutes(self, monkeypatch):
        assert load_settings(
            REPO / "config.yaml").trading.no_new_entries_minutes_before_close == 15
        assert self._sched(15, 899, monkeypatch).should_avoid_new_trades() is True
        assert self._sched(15, 901, monkeypatch).should_avoid_new_trades() is False

    def test_widening_it_is_a_config_edit_not_a_code_edit(self, monkeypatch):
        assert self._sched(30, 1500, monkeypatch).should_avoid_new_trades() is True

    def test_a_scheduler_built_bare_still_works(self, monkeypatch):
        from ait.bot.scheduler import MarketScheduler
        sched = MarketScheduler()
        monkeypatch.setattr(sched, "seconds_until_close", lambda: 600)
        assert sched.should_avoid_new_trades() is True

    def test_the_live_orchestrator_hands_its_settings_to_the_scheduler(
            self, tmp_path, monkeypatch):
        """The real __init__, via the hot-path smoke rig (sandboxed cwd: the
        real constructor creates its stores cwd-relative)."""
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            want = rig.orch._settings.trading.no_new_entries_minutes_before_close
            assert rig.orch._scheduler._no_new_entries_s == want * 60
        finally:
            rig.restore()


# --------------------------------------------------------------------------
# one authority for where the Gateway start script lives
# --------------------------------------------------------------------------
class TestOneGatewayScriptAuthority:
    def test_relocating_ibc_moves_recovery_with_boot(self, tmp_path):
        """IBC_DIR drove the boot launcher; the recovery launcher had its own
        default, so a relocated IBC booted but could never be recovered."""
        r = _run("""
            import ait.orchestration.master as M
            import ait.orchestration.gateway as G
            print("SAME", str(M.GATEWAY_START_SCRIPT) == str(G.IBC_GATEWAY_SCRIPT))
            print(M.GATEWAY_START_SCRIPT)
        """, tmp_path, IBC_DIR=str(tmp_path / "ibc"), AIT_GATEWAY_START_SCRIPT="")
        assert "SAME True" in r.stdout, r.stdout + r.stderr[-400:]

    def test_an_explicit_override_still_wins(self, tmp_path):
        r = _run("""
            import ait.orchestration.master as M
            print(M.GATEWAY_START_SCRIPT)
        """, tmp_path, IBC_DIR=str(tmp_path / "ibc"),
            AIT_GATEWAY_START_SCRIPT=r"D:\custom\start.bat")
        assert r"D:\custom\start.bat" in r.stdout, r.stdout + r.stderr[-400:]
