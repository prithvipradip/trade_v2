"""Shared per-machine logs: each box publishes its own bot events to the
orphan `shared-logs` branch; the code branch and the live tree are never
touched. See src/ait/orchestration/shared_logs.py for the why.

Executing tests only — the git ones drive real `git` against a throwaway bare
"origin" in tmp_path, two clones standing in for the two boxes.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from ait.orchestration import shared_logs as sl

DAY = date(2026, 9, 28)


def _ev(ts: str, event: str, **kw) -> str:
    return json.dumps({"event": event, "level": kw.pop("level", "info"),
                       "logger": kw.pop("logger", "ait.bot.orchestrator"),
                       "timestamp": ts, **kw})


def _write_logs(logs: Path, files: dict[str, list[str]]) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (logs / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestOwner:
    def test_unset_means_publishing_off(self):
        assert sl.resolve_owner({}) is None

    @pytest.mark.parametrize("raw", ["prithvi", "ahmed", "box-2"])
    def test_valid_names(self, raw):
        assert sl.resolve_owner({"AIT_LOG_OWNER": raw}) == raw

    @pytest.mark.parametrize("raw", ["", "Prithvi", "../etc", "a", "has space", "x" * 40])
    def test_malformed_names_are_treated_as_unset(self, raw):
        assert sl.resolve_owner({"AIT_LOG_OWNER": raw}) is None


class TestScrub:
    @pytest.mark.parametrize("raw", [
        'GET /v1/quote?apiKey=abcd1234efgh&symbol=SPY',
        '{"api_key": "sk-live-123456"}',
        'token=zzzz9999',
        'Authorization: Bearer eyJhbGciOiJ',
        '"password": "hunter22"',
    ])
    def test_secrets_are_redacted(self, raw):
        out = sl.scrub(raw)
        assert "[REDACTED]" in out
        for secret in ("abcd1234efgh", "sk-live-123456", "zzzz9999", "eyJhbGciOiJ", "hunter22"):
            assert secret not in out

    def test_account_ids_are_redacted(self):
        out = sl.scrub('{"account": "DUN603821", "live": "U21959335", "paper": "DUO442667"}')
        assert "DUN603821" not in out and "U21959335" not in out and "DUO442667" not in out
        assert out.count("[ACCOUNT]") == 3

    def test_ordinary_trading_fields_survive(self):
        line = _ev("2026-09-28T14:09:25Z", "trade_validated", symbol="QQQ", max_risk=4981.0)
        assert sl.scrub(line) == line


class TestSelectDay:
    def test_et_day_across_rotations_in_order_deduped(self, tmp_path):
        logs = tmp_path / "logs"
        a = _ev("2026-09-28T13:35:00Z", "scan_symbol_timing", symbol="SPY")
        b = _ev("2026-09-28T14:09:25Z", "order_placed", symbol="QQQ")
        c = _ev("2026-09-28T19:59:00Z", "skipping_new_trades")
        _write_logs(logs, {
            "ait.log.2": [a, "Error 10091, reqId 1: Part of requested market data ...", b],
            "ait.log.1": [b, "Warning 10167, reqId 2: delayed", "Traceback (most recent call last):"],
            "ait.log": [c],
        })
        out = sl.select_day_events(logs, DAY)
        assert [json.loads(x)["event"] for x in out] == [
            "scan_symbol_timing", "order_placed", "skipping_new_trades"]

    def test_day_is_the_et_calendar_day_not_utc(self, tmp_path):
        logs = tmp_path / "logs"
        late_et = _ev("2026-09-29T02:30:00Z", "post_market_complete")   # 22:30 ET on the 28th
        next_et = _ev("2026-09-29T04:30:00Z", "waiting_for_phase")      # 00:30 ET on the 29th
        _write_logs(logs, {"ait.log": [late_et, next_et]})
        assert [json.loads(x)["event"] for x in sl.select_day_events(logs, DAY)] == ["post_market_complete"]
        assert [json.loads(x)["event"] for x in sl.select_day_events(logs, date(2026, 9, 29))] == ["waiting_for_phase"]

    def test_http_library_loggers_are_dropped(self, tmp_path):
        logs = tmp_path / "logs"
        _write_logs(logs, {"ait.log": [
            _ev("2026-09-28T14:00:00Z", "request", logger="urllib3.connectionpool",
                url="https://api.example/q?apikey=LEAKED123"),
            _ev("2026-09-28T14:00:01Z", "scan_symbol_timing"),
        ]})
        out = sl.select_day_events(logs, DAY)
        assert len(out) == 1 and "LEAKED123" not in out[0]

    def test_secrets_inside_kept_events_are_scrubbed(self, tmp_path):
        logs = tmp_path / "logs"
        _write_logs(logs, {"ait.log": [
            _ev("2026-09-28T14:00:00Z", "polygon_disabled_auth_failures",
                detail="apiKey=SECRET9999 rejected", account="DUN603821")]})
        out = sl.select_day_events(logs, DAY)[0]
        assert "SECRET9999" not in out and "DUN603821" not in out


class TestExportAndRead:
    def test_round_trip_with_coverage_header(self, tmp_path):
        logs = tmp_path / "logs"
        _write_logs(logs, {"ait.log": [
            _ev("2026-09-28T13:35:00Z", "iv_rank_true_percentile", symbol="SPY", rank=25.0, level="debug"),
            _ev("2026-09-28T13:36:00Z", "condor_rejected_iv_rank_floor", symbol="IWM"),
            _ev("2026-09-28T14:09:25Z", "trade_validated", symbol="QQQ"),
            _ev("2026-09-28T14:09:26Z", "order_placed", symbol="QQQ"),
            _ev("2026-09-28T14:10:00Z", "high_memory_usage", level="warning"),
        ]})
        dest = tmp_path / "out" / "prithvi_logs" / "2026-09-28.jsonl.gz"
        meta = sl.export_day(logs, DAY, dest, "prithvi")
        assert meta["events"] == 5 and meta["first_event"] == "2026-09-28T13:35:00Z"
        rmeta, events = sl.read_day(dest)
        assert rmeta["owner"] == "prithvi" and len(events) == 5
        s = sl.summarize(events)
        assert s["iv_ranks_last"] == {"SPY": 25.0}
        assert s["rejections"] == {"condor_rejected_iv_rank_floor": 1}
        assert s["orders"]["trade_validated"] == 1 and s["orders"]["order_placed"] == 1
        assert s["warnings_errors"] == ["high_memory_usage"]


# ------------------------------------------------------------------- git side

def _run(args, cwd):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (args, r.stderr)
    return r.stdout


def _box(tmp_path: Path, origin: Path, name: str) -> Path:
    repo = tmp_path / name / "repo"
    repo.parent.mkdir(parents=True)
    _run(["git", "clone", "-q", str(origin), str(repo)], tmp_path)
    _run(["git", "config", "user.email", f"{name}@example.invalid"], repo)
    _run(["git", "config", "user.name", name], repo)
    return repo


@pytest.fixture
def origin(tmp_path):
    """A bare 'origin' holding one code commit on feature/event-straddle."""
    bare = tmp_path / "origin.git"
    _run(["git", "init", "-q", "--bare", "-b", "feature/event-straddle", str(bare)], tmp_path)
    seed = tmp_path / "seed"
    _run(["git", "clone", "-q", str(bare), str(seed)], tmp_path)
    _run(["git", "config", "user.email", "s@example.invalid"], seed)
    _run(["git", "config", "user.name", "seed"], seed)
    (seed / "config.yaml").write_text("risk: {}\n")
    _run(["git", "add", "config.yaml"], seed)
    _run(["git", "commit", "-q", "-m", "code"], seed)
    _run(["git", "push", "-q", "origin", "HEAD:feature/event-straddle"], seed)
    return bare


def _logs_for(repo: Path, event_ts: str, n: int = 2) -> Path:
    logs = repo / "logs"
    _write_logs(logs, {"ait.log": [_ev(event_ts, "scan_symbol_timing", i=i) for i in range(n)]})
    return logs


class TestPublish:
    def test_first_publish_creates_the_orphan_branch(self, tmp_path, origin):
        repo = _box(tmp_path, origin, "prithvi")
        logs = _logs_for(repo, "2026-09-28T14:00:00Z")
        res = sl.publish(repo, "prithvi", [DAY], logs_dir=logs)
        assert res.ok and res.published == ["2026-09-28"], res
        files = _run(["git", "ls-tree", "-r", "--name-only", "shared-logs"], origin).split()
        assert "prithvi_logs/2026-09-28.jsonl.gz" in files and "README.md" in files
        # orphan: shares no history with the code branch
        code = _run(["git", "rev-list", "feature/event-straddle"], origin).split()
        shared = _run(["git", "rev-list", "shared-logs"], origin).split()
        assert not set(code) & set(shared)

    def test_the_live_tree_is_untouched(self, tmp_path, origin):
        repo = _box(tmp_path, origin, "prithvi")
        head_before = _run(["git", "rev-parse", "HEAD"], repo)
        branch_before = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
        logs = _logs_for(repo, "2026-09-28T14:00:00Z")
        assert sl.publish(repo, "prithvi", [DAY], logs_dir=logs).ok
        assert _run(["git", "rev-parse", "HEAD"], repo) == head_before
        assert _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo) == branch_before
        # only the (git-ignored-in-the-real-repo) logs/ dir is new; no tracked change
        assert _run(["git", "status", "--porcelain", "--untracked-files=no"], repo) == ""
        assert _run(["git", "ls-remote", "--heads", "origin", "feature/event-straddle"], repo).split()[0] \
            == head_before.strip()

    def test_two_boxes_publish_to_their_own_folders(self, tmp_path, origin):
        a = _box(tmp_path, origin, "prithvi")
        b = _box(tmp_path, origin, "ahmed")
        assert sl.publish(a, "prithvi", [DAY], logs_dir=_logs_for(a, "2026-09-28T14:00:00Z")).ok
        # b's worktree predates nothing of a's; its push must rebase over a's
        assert sl.publish(b, "ahmed", [DAY], logs_dir=_logs_for(b, "2026-09-28T15:00:00Z")).ok
        files = set(_run(["git", "ls-tree", "-r", "--name-only", "shared-logs"], origin).split())
        assert {"prithvi_logs/2026-09-28.jsonl.gz", "ahmed_logs/2026-09-28.jsonl.gz"} <= files

    def test_republish_only_when_the_day_grew(self, tmp_path, origin):
        repo = _box(tmp_path, origin, "prithvi")
        logs = _logs_for(repo, "2026-09-28T14:00:00Z", n=2)
        assert sl.publish(repo, "prithvi", [DAY], logs_dir=logs).published == ["2026-09-28"]
        again = sl.publish(repo, "prithvi", [DAY], logs_dir=logs)
        assert again.ok and again.published == [] and "already published" in again.skipped[0]
        _logs_for(repo, "2026-09-28T14:00:00Z", n=5)
        assert sl.publish(repo, "prithvi", [DAY], logs_dir=logs).published == ["2026-09-28"]

    def test_days_without_events_are_skipped(self, tmp_path, origin):
        repo = _box(tmp_path, origin, "prithvi")
        logs = _logs_for(repo, "2026-09-28T14:00:00Z")
        res = sl.publish(repo, "prithvi", [date(2026, 9, 25), DAY], logs_dir=logs)
        assert res.ok and res.published == ["2026-09-28"] and res.skipped == ["2026-09-25: no events"]

    def test_never_raises(self, tmp_path):
        not_a_repo = tmp_path / "nope"
        not_a_repo.mkdir()
        res = sl.publish(not_a_repo, "prithvi", [DAY], logs_dir=_logs_for(not_a_repo, "2026-09-28T14:00:00Z"))
        assert res.ok is False and res.error

    def test_invalid_owner_refused(self, tmp_path, origin):
        repo = _box(tmp_path, origin, "prithvi")
        res = sl.publish(repo, "../x", [DAY], logs_dir=_logs_for(repo, "2026-09-28T14:00:00Z"))
        assert res.ok is False and "invalid owner" in res.error


class TestSupervisorJob:
    def test_no_owner_no_subprocess(self, monkeypatch):
        import ait.orchestration.master as m
        monkeypatch.delenv("AIT_LOG_OWNER", raising=False)
        calls = []
        monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a))
        m.publish_shared_logs()
        assert calls == []

    def test_with_owner_runs_the_publisher_with_a_timeout(self, monkeypatch):
        import ait.orchestration.master as m
        monkeypatch.setenv("AIT_LOG_OWNER", "prithvi")
        seen = {}

        def _run(args, **kw):
            seen["args"], seen["kw"] = args, kw
            return subprocess.CompletedProcess(args, 0, stdout='{"ok": true}', stderr="")
        monkeypatch.setattr(m.subprocess, "run", _run)
        m.publish_shared_logs()
        assert seen["args"][-1].endswith("publish_shared_logs.py")
        assert seen["kw"]["timeout"] == 600

    def test_a_publisher_failure_never_raises(self, monkeypatch):
        import ait.orchestration.master as m
        monkeypatch.setenv("AIT_LOG_OWNER", "prithvi")

        def _boom(*a, **k):
            raise subprocess.TimeoutExpired("x", 600)
        monkeypatch.setattr(m.subprocess, "run", _boom)
        m.publish_shared_logs()   # must not raise

    def test_it_is_scheduled_after_the_close(self):
        import ait.orchestration.master as m
        jobs = {}

        class _S:
            def add_job(self, fn, trigger=None, **kw):
                jobs[kw.get("id")] = (fn, str(trigger))
        m._register_recurring_jobs(_S(), bot=None, web=None)
        fn, trig = jobs["shared_logs_publish"]
        assert fn is m.publish_shared_logs and "hour='16'" in trig and "minute='45'" in trig
