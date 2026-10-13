"""W76: the supervisor-run short-DTE collection job.

Everything external is injected (clock, calendar, broker port probe, subprocess), so the
whole job runs here without a broker, a sleeping process or a wall clock.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from ait.config.settings import DataCollectionConfig
from ait.orchestration import short_dte_job as J

ET = ZoneInfo("US/Eastern")
MON = date(2026, 10, 12)


def _cfg(**over):
    return DataCollectionConfig(short_dte_enabled=True, **over)


# --------------------------------------------------------------------------
# config model
# --------------------------------------------------------------------------

class TestConfig:
    def test_defaults_are_off_and_conservative(self):
        c = DataCollectionConfig()
        assert c.short_dte_enabled is False and c.short_dte_time_et == "15:30"
        assert c.short_dte_symbols == ["QQQ"] and c.short_dte_client_id == 91
        assert c.short_dte_max_dte == 6 and c.short_dte_duration == "7 D"

    @pytest.mark.parametrize("bad", [
        dict(short_dte_time_et="25:00"), dict(short_dte_time_et="4:10"), dict(short_dte_time_et="16:60"),
        dict(short_dte_symbols=[]), dict(short_dte_symbols=["QQQ", "qqq"]), dict(short_dte_symbols=["A B"]),
        dict(short_dte_duration="week"), dict(short_dte_client_id=0), dict(short_dte_max_dte=0),
        dict(short_dte_max_dte=30), dict(short_dte_timeout_minutes=1), dict(short_dte_idle_minutes=1),
        dict(unknown_key=1),
    ])
    def test_bad_values_are_refused(self, bad):
        with pytest.raises(Exception):
            DataCollectionConfig(**bad)

    def test_symbols_are_normalised(self):
        assert DataCollectionConfig(short_dte_symbols=[" qqq ", "spy"]).short_dte_symbols == ["QQQ", "SPY"]

    def test_the_section_is_part_of_settings_and_default_off(self):
        from ait.config.settings import Settings
        assert Settings().data_collection.short_dte_enabled is False


# --------------------------------------------------------------------------
# the job
# --------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    """A collector process: optionally writes output/status, exits after ``polls`` polls."""

    instances: list = []

    def __init__(self, cmd, cwd=None, stdout=None, stderr=None, env=None, *, script):
        self.cmd, self.cwd, self.env, self.stdout = cmd, cwd, env, stdout
        self.polls = 0
        self.terminated = self.killed = False
        self.script = script
        FakeProc.instances.append(self)
        script.on_start(self)

    def poll(self):
        if self.terminated or self.killed:
            return -15
        self.polls += 1
        self.script.on_poll(self)
        return None if self.polls <= self.script.polls else self.script.rc

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class Script:
    """What the fake collector does."""

    def __init__(self, *, rc=0, polls=2, status=None, chatter=True, silent=False, status_path=None):
        self.rc, self.polls, self.status = rc, polls, status
        self.chatter, self.silent = chatter, silent

    def on_start(self, proc):
        proc.status_path = Path(proc.cmd[proc.cmd.index("--status-file") + 1])
        if self.chatter and not self.silent:
            proc.stdout.write("progress line 0\n")
            proc.stdout.flush()

    def on_poll(self, proc):
        if self.chatter and not self.silent:
            proc.stdout.write(f"progress line {proc.polls}\n")
            proc.stdout.flush()
        if proc.polls == self.polls and self.status is not None:
            proc.status_path.write_text(json.dumps(self.status))


def _status(stored=500, planned=85, ok=True, error=None, bands=None):
    return {"schema": "ait.short_dte_collection/1", "ok": ok, "error": error, "symbols": {
        "QQQ": {"planned": planned, "stored": stored, "requested": planned * 2, "errors": 0,
                "unqualified": 0, "bands": bands or {"0-2": {"sessions": 3, "needed": 15},
                                                     "3-6": {"sessions": 5, "needed": 15}}}}}


def _run(tmp_path, script, *, cfg=None, trading=True, gateway=True, base_id=1, today=MON,
         poll_s=5.0):
    clock = FakeClock()
    FakeProc.instances = []
    data, logs = tmp_path / "data", tmp_path / "logs"
    data.mkdir(exist_ok=True)
    out = J.run_collection(
        cfg or _cfg(), root=tmp_path, data_dir=data, logs_dir=logs, min_sessions=15,
        ibkr_port=7497, base_client_id=base_id, is_trading_day=lambda d: trading,
        gateway_listening=lambda p: gateway, today=today,
        popen=lambda cmd, **kw: FakeProc(cmd, script=script, **kw), python="PY",
        clock=clock, sleep=clock.sleep, poll_s=poll_s)
    return out, data / J.STATUS_FILE_NAME, logs


class TestGuards:
    def test_a_non_trading_day_starts_nothing(self, tmp_path):
        out, *_ = _run(tmp_path, Script(status=_status()), trading=False)
        assert out.status == "skipped_not_trading_day" and not out.alert_worthy
        assert FakeProc.instances == []

    def test_gateway_down_starts_nothing_and_is_alert_worthy(self, tmp_path):
        out, *_ = _run(tmp_path, Script(status=_status()), gateway=False)
        assert out.status == "skipped_gateway_down" and out.alert_worthy
        assert "7497" in out.detail and FakeProc.instances == []

    @pytest.mark.parametrize("cid,base", [(1, 1), (101, 1), (105, 1), (91, 91), (191, 91)])
    def test_a_client_id_inside_the_bots_ladder_is_refused(self, tmp_path, cid, base):
        out, *_ = _run(tmp_path, Script(status=_status()),
                       cfg=_cfg(short_dte_client_id=cid), base_id=base)
        assert out.status == "refused_client_id" and out.alert_worthy
        assert FakeProc.instances == []

    def test_a_client_id_just_outside_the_ladder_is_allowed(self, tmp_path):
        out, *_ = _run(tmp_path, Script(status=_status()), cfg=_cfg(short_dte_client_id=106), base_id=1)
        assert out.status == "ok"

    def test_reserved_ids_are_the_bots_base_and_its_fallback_ladder(self):
        assert J.reserved_client_ids(326) == {326, 426, 427, 428, 429, 430}


class TestOutcomes:
    def test_a_healthy_run_is_ok_and_reports_band_progress(self, tmp_path):
        out, status_path, logs = _run(tmp_path, Script(status=_status(stored=612, planned=85)))
        assert out.status == "ok" and out.stored == 612 and out.planned == 85
        assert out.bands == {"QQQ": {"0-2": "3/15", "3-6": "5/15"}}
        assert not out.alert_worthy
        assert Path(out.log_path).exists() and "progress line" in Path(out.log_path).read_text()

    def test_zero_samples_is_an_empty_run_and_alerts(self, tmp_path):
        out, *_ = _run(tmp_path, Script(status=_status(stored=0, planned=85)))
        assert out.status == "empty" and out.alert_worthy

    def test_a_nonzero_exit_is_failed_with_the_log_tail(self, tmp_path):
        out, *_ = _run(tmp_path, Script(rc=1, status=None))
        assert out.status == "failed" and out.alert_worthy
        assert "exit=1" in out.detail and "progress line" in out.detail

    def test_a_status_written_with_ok_false_is_failed_with_its_error(self, tmp_path):
        out, *_ = _run(tmp_path, Script(rc=1, status=_status(ok=False, error="could not connect to IBKR: refused")))
        assert out.status == "failed" and "could not connect" in out.detail

    def test_a_nonzero_exit_is_failed_even_when_a_good_looking_status_exists(self, tmp_path):
        """A crash AFTER the status file was written (or a partial run that wrote ok=true)
        must not be reported as a healthy collection."""
        out, *_ = _run(tmp_path, Script(rc=3, status=_status(stored=612, planned=85)))
        assert out.status == "failed" and out.alert_worthy and "exit=3" in out.detail

    def test_exit_zero_without_a_status_file_is_failed_not_ok(self, tmp_path):
        out, *_ = _run(tmp_path, Script(rc=0, status=None))
        assert out.status == "failed"

    def test_a_stale_status_file_is_never_mistaken_for_this_runs_result(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / J.STATUS_FILE_NAME).write_text(json.dumps(_status(stored=9999)))   # yesterday's
        out, *_ = _run(tmp_path, Script(rc=0, status=None))
        assert out.status == "failed" and out.stored == 0

    def test_ok_false_status_with_exit_zero_is_still_failed(self, tmp_path):
        out, *_ = _run(tmp_path, Script(rc=0, status=_status(ok=False, error="symbol QQQ: boom")))
        assert out.status == "failed" and "boom" in out.detail


class TestTimeouts:
    def test_a_silent_hang_is_killed_by_the_idle_timeout(self, tmp_path):
        out, *_ = _run(tmp_path, Script(polls=10**5, silent=True), cfg=_cfg(short_dte_idle_minutes=2))
        assert out.status == "idle_timeout" and out.alert_worthy
        proc = FakeProc.instances[0]
        assert proc.terminated

    def test_a_chatty_run_that_never_ends_hits_the_absolute_cap(self, tmp_path):
        out, *_ = _run(tmp_path, Script(polls=10**5, chatter=True),
                       cfg=_cfg(short_dte_timeout_minutes=10, short_dte_idle_minutes=120))
        assert out.status == "timeout" and "10 min" in out.detail
        assert FakeProc.instances[0].terminated

    def test_a_run_that_finishes_inside_both_limits_is_not_killed(self, tmp_path):
        out, *_ = _run(tmp_path, Script(polls=50, status=_status()),
                       cfg=_cfg(short_dte_timeout_minutes=150, short_dte_idle_minutes=20))
        assert out.status == "ok" and not FakeProc.instances[0].terminated


class TestCommand:
    def test_the_collector_is_started_with_the_configured_arguments(self, tmp_path):
        cfg = _cfg(short_dte_symbols=["QQQ", "SPY"], short_dte_client_id=95, short_dte_max_dte=5,
                   short_dte_duration="5 D")
        _run(tmp_path, Script(status=_status()), cfg=cfg)
        cmd = FakeProc.instances[0].cmd
        assert cmd[0] == "PY" and cmd[1] == "-u" and cmd[2].endswith("calibrate_option_spreads.py")
        flat = " ".join(cmd)
        assert "--source ib-bars --short-dte" in flat
        assert "--symbols QQQ SPY" in flat and "--client-id 95" in flat
        assert "--short-dte-max 5" in flat and "--short-dte-duration 5 D" in flat
        assert "--min-sessions 15" in flat and "--status-file" in flat

    def test_the_port_is_not_passed_it_comes_from_the_environment(self, tmp_path):
        _run(tmp_path, Script(status=_status()))
        assert "--port" not in FakeProc.instances[0].cmd

    def test_it_runs_in_the_repo_root_with_pinned_threads(self, tmp_path):
        _run(tmp_path, Script(status=_status()))
        p = FakeProc.instances[0]
        assert p.cwd == str(tmp_path) and p.env["OMP_NUM_THREADS"] == "1"


# --------------------------------------------------------------------------
# the supervisor wrapper + registration
# --------------------------------------------------------------------------

class TestSupervisor:
    @pytest.fixture
    def M(self, monkeypatch):
        from ait.orchestration import master
        monkeypatch.setattr(master, "_ALERT_GATE_LAST", {})
        return master

    def _wire(self, M, monkeypatch, outcome, *, enabled=True):
        sent = []
        monkeypatch.setattr(M, "_alert", lambda m: sent.append(m) or True)
        monkeypatch.setattr(M, "_data_collection_cfg", lambda: _cfg())
        monkeypatch.setattr("ait.orchestration.short_dte_job.run_collection", lambda *a, **k: outcome)
        return sent

    def test_a_failure_alerts_once_per_day(self, M, monkeypatch):
        sent = self._wire(M, monkeypatch, J.JobOutcome("failed", "exit=1: boom", log_path="x.log"))
        M.collect_short_dte_friction()
        M.collect_short_dte_friction()
        assert len(sent) == 1 and "failed" in sent[0] and "boom" in sent[0]

    def test_a_healthy_run_sends_nothing(self, M, monkeypatch):
        sent = self._wire(M, monkeypatch, J.JobOutcome("ok", "stored 5", stored=5, planned=3))
        M.collect_short_dte_friction()
        assert sent == []

    def test_a_non_trading_day_sends_nothing(self, M, monkeypatch):
        sent = self._wire(M, monkeypatch, J.JobOutcome("skipped_not_trading_day", "holiday"))
        M.collect_short_dte_friction()
        assert sent == []

    def test_it_never_raises_into_the_supervisor(self, M, monkeypatch):
        monkeypatch.setattr(M, "_data_collection_cfg", lambda: (_ for _ in ()).throw(RuntimeError("cfg")))
        M.collect_short_dte_friction()          # must not raise

    def _register(self, M, monkeypatch, cfg):
        monkeypatch.setattr(M, "_data_collection_cfg", lambda: cfg)
        sch = M.BlockingScheduler(timezone="US/Eastern")
        M._register_recurring_jobs(sch, bot=MagicMock(), web=MagicMock())
        return sch

    def test_disabled_registers_nothing(self, M, monkeypatch):
        sch = self._register(M, monkeypatch, DataCollectionConfig())
        assert sch.get_job("short_dte_collection") is None

    def test_enabled_registers_one_weekday_et_job_at_the_configured_time(self, M, monkeypatch):
        sch = self._register(M, monkeypatch, _cfg(short_dte_time_et="15:50"))
        job = sch.get_job("short_dte_collection")
        assert job is not None and job.max_instances == 1
        assert str(job.trigger.timezone) == "US/Eastern"
        fields = {f.name: str(f) for f in job.trigger.fields}
        assert fields["hour"] == "15" and fields["minute"] == "50"
        assert fields["day_of_week"] == "mon-fri"
        nxt = job.trigger.get_next_fire_time(None, datetime(2026, 10, 12, 12, 0, tzinfo=ET))
        assert (nxt.hour, nxt.minute) == (15, 50) and nxt.tzinfo is not None

    def test_the_default_runs_before_the_close_so_dte_zero_can_be_captured(self, M, monkeypatch):
        """W76b, MEASURED 2026-10-13: after the session ends IB no longer serves that day's
        expiring contracts (a 16:10 run listed expiries from the NEXT day), so the run must start
        before 16:00 ET; 15:30 is the first moment the entry window's last bar has ended."""
        sch = self._register(M, monkeypatch, _cfg())
        job = sch.get_job("short_dte_collection")
        nxt = job.trigger.get_next_fire_time(None, datetime(2026, 10, 9, 0, 0, tzinfo=ET))   # Friday
        assert (nxt.hour, nxt.minute) == (15, 30) and nxt.weekday() == 4
        assert (nxt.hour, nxt.minute) < (16, 0)

    def test_enabling_it_adds_exactly_one_cron_job(self, M, monkeypatch):
        from apscheduler.triggers.cron import CronTrigger
        off = self._register(M, monkeypatch, DataCollectionConfig())
        on = self._register(M, monkeypatch, _cfg())
        n = lambda s: sum(isinstance(j.trigger, CronTrigger) for j in s.get_jobs())   # noqa: E731
        assert n(on) == n(off) + 1


# --------------------------------------------------------------------------
# the script's status file, end to end against a fake broker
# --------------------------------------------------------------------------

class TestStatusFile:
    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "calibrate_status_cli", Path(__file__).resolve().parents[1] / "scripts"
            / "calibrate_option_spreads.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_write_status_is_atomic_and_complete(self, tmp_path):
        from ait.data.short_dte_collector import STATUS_SCHEMA, write_status
        p = tmp_path / "sub" / "s.json"
        got = write_status(p, ok=True, error=None, symbols={"QQQ": {"stored": 3}},
                           min_sessions=15, duration_s=12.34, short_dte_max=6)
        on_disk = json.loads(p.read_text())
        assert on_disk == got and on_disk["schema"] == STATUS_SCHEMA
        assert on_disk["duration_s"] == 12.3 and not list(p.parent.glob("*.tmp"))

    def test_symbol_status_counts_active_only_with_a_fit_and_enough_sessions(self):
        from ait.data.short_dte_collector import symbol_status
        blk = symbol_status(None, {}, {"0-2": {"form": "pct", "n": 80}},
                            {"0-2": 16, "3-6": 20}, 15)
        assert blk["bands"]["0-2"]["active"] is True and blk["bands"]["0-2"]["form"] == "pct"
        assert blk["bands"]["3-6"]["active"] is False       # enough sessions but no fit yet
        assert blk["bands"]["3-6"]["fitted"] is False

    def test_a_connect_failure_leaves_a_failed_status_for_the_supervisor(self, tmp_path, monkeypatch):
        mod = self._mod()
        import ib_insync

        class Refusing:
            async def connectAsync(self, *a, **k):
                raise ConnectionRefusedError("[Errno 61] Connect call failed")

            def disconnect(self):
                pass
        monkeypatch.setattr(ib_insync, "IB", Refusing)
        sf = tmp_path / "status.json"
        args = mod._parse_args(["--source", "ib-bars", "--short-dte", "--symbols", "QQQ",
                                "--db-path", str(tmp_path / "h.db"), "--status-file", str(sf)])
        assert asyncio.run(mod._main(args)) == 1
        st = json.loads(sf.read_text())
        assert st["ok"] is False and "could not connect" in st["error"]

    def test_an_exception_mid_run_is_reported_in_the_status_and_fails_the_run(self, tmp_path, monkeypatch):
        from tests.test_w76_short_dte_friction import FakeIB
        mod = self._mod()
        import ib_insync

        class Exploding(FakeIB):
            async def connectAsync(self, *a, **k):
                return None

            def disconnect(self):
                pass

            async def reqSecDefOptParamsAsync(self, *a, **k):
                raise RuntimeError("secdef unavailable")
        monkeypatch.setattr(ib_insync, "IB", Exploding)
        sf = tmp_path / "status.json"
        args = mod._parse_args(["--source", "ib-bars", "--short-dte", "--symbols", "QQQ",
                                "--db-path", str(tmp_path / "h.db"), "--status-file", str(sf),
                                "--bars-pause-secs", "0"])
        assert asyncio.run(mod._main(args)) == 1
        st = json.loads(sf.read_text())
        assert st["ok"] is False and "QQQ: RuntimeError: secdef unavailable" in st["error"]

    def test_a_full_fake_run_writes_coverage_and_band_progress(self, tmp_path, monkeypatch):
        from tests.test_w76_short_dte_friction import FakeIB
        mod = self._mod()
        import ib_insync

        class Connected(FakeIB):
            async def connectAsync(self, *a, **k):
                return None

            def disconnect(self):
                pass
        monkeypatch.setattr(ib_insync, "IB", Connected)
        # the fake chain's "today" is the W76 test's TODAY; pin the collector's clock to it
        import ait.data.short_dte_collector as sc
        from tests.test_w76_short_dte_friction import TODAY
        monkeypatch.setattr(sc, "today_et", lambda: TODAY)
        sf = tmp_path / "status.json"
        args = mod._parse_args(["--source", "ib-bars", "--short-dte", "--symbols", "QQQ",
                                "--db-path", str(tmp_path / "h.db"), "--status-file", str(sf),
                                "--bars-pause-secs", "0", "--min-sessions", "2"])
        assert asyncio.run(mod._main(args)) == 0
        st = json.loads(sf.read_text())
        blk = st["symbols"]["QQQ"]
        assert st["ok"] is True and blk["stored"] > 0 and blk["planned"] > 0
        assert set(blk["coverage"]) == {str(d) for d in range(7)}
        assert set(blk["bands"]) == {"0-2", "3-6"}
        assert all(b["needed"] == 2 for b in blk["bands"].values())
