"""W94: the bot and the supervisor fall back to the other endpoint of the SAME IB account type.

2026-10-23: .env said IBKR_PORT=7497 (TWS) while IB Gateway listened on 4002; the supervisor deferred
the bot behind a hint. Covers: effective_port, the endpoint-state record, scripts/sync_ibkr_port.py
(on a temp .env with dummy secrets), the supervisor's _ib_port / one-note-per-process behaviour, the
health-check start path, and an AST class guard that nothing in master.py reads IBKR_PORT outside
_configured_ib_port. No inspect.getsource.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import ait.orchestration.master as M
from ait.data.ib_endpoint import (
    ENDPOINT_FILE_NAME,
    endpoint_mismatch_text,
    read_endpoint_state,
    write_endpoint_state,
)
from ait.data.ib_ports import TWIN_PORTS, effective_port

REPO = Path(__file__).resolve().parents[1]
LIVE = {7496, 4001}


# ---------------------------------------------------------------------------
# effective_port
# ---------------------------------------------------------------------------

class TestEffectivePort:
    def test_configured_up_wins(self):
        assert effective_port(7497, lambda p: p == 7497) == 7497

    def test_twin_when_configured_is_down(self):
        assert effective_port(7497, lambda p: p == 4002) == 4002
        assert effective_port(4002, lambda p: p == 7497) == 7497

    def test_nothing_up_reports_the_configured_port(self):
        assert effective_port(7497, lambda p: False) == 7497

    def test_both_up_configured_wins(self):
        assert effective_port(7497, lambda p: True) == 7497

    @pytest.mark.parametrize("configured", sorted(TWIN_PORTS))
    def test_never_crosses_account_type(self, configured):
        got = effective_port(configured, lambda p: True)
        assert (got in LIVE) == (configured in LIVE)
        got = effective_port(configured, lambda p: p not in {configured})
        assert (got in LIVE) == (configured in LIVE)

    def test_unknown_port_is_left_alone(self):
        assert effective_port(9999, lambda p: False) == 9999


# ---------------------------------------------------------------------------
# endpoint record
# ---------------------------------------------------------------------------

class TestEndpointState:
    def test_written_on_change_only_and_since_is_stable(self, tmp_path):
        f = tmp_path / ENDPOINT_FILE_NAME
        assert write_endpoint_state(f, 7497, 4002) is True
        first = read_endpoint_state(f)
        assert first["fallback"] is True and first["using"] == 4002
        assert write_endpoint_state(f, 7497, 4002) is False
        assert read_endpoint_state(f)["since"] == first["since"]
        assert write_endpoint_state(f, 4002, 4002) is True
        assert read_endpoint_state(f)["fallback"] is False

    def test_mismatch_text(self, tmp_path):
        f = tmp_path / ENDPOINT_FILE_NAME
        write_endpoint_state(f, 7497, 4002)
        txt = endpoint_mismatch_text(read_endpoint_state(f))
        assert "7497" in txt and "4002" in txt and "sync_ibkr_port" in txt
        write_endpoint_state(f, 4002, 4002)
        assert endpoint_mismatch_text(read_endpoint_state(f)) == ""
        assert endpoint_mismatch_text(None) == ""

    def test_unreadable_file_is_none(self, tmp_path):
        f = tmp_path / ENDPOINT_FILE_NAME
        f.write_text("{not json")
        assert read_endpoint_state(f) is None

    def test_status_surface_reports_it(self, tmp_path, monkeypatch):
        spec = importlib.util.spec_from_file_location("status_mod", REPO / "status.py")
        st = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(st)
        (tmp_path / "data").mkdir()
        write_endpoint_state(tmp_path / "data" / ENDPOINT_FILE_NAME, 7497, 4002)
        monkeypatch.setattr(st, "ROOT", tmp_path)
        out = st._ib_endpoint()
        assert out["record"]["using"] == 4002 and "fix .env" in out["mismatch"]


# ---------------------------------------------------------------------------
# scripts/sync_ibkr_port.py (temp .env, dummy secrets)
# ---------------------------------------------------------------------------

def _load_sync():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import sync_ibkr_port
        return sync_ibkr_port
    finally:
        sys.path.pop(0)


ENV = ("IBKR_ACCOUNT=DU0000000\r\n"
       "IBKR_PORT=7497                          # 7497=TWS paper, 4002=Gateway paper\r\n"
       "POLYGON_API_KEY=dummy-secret-123\n"
       "TELEGRAM_BOT_TOKEN=dummy-token-456\n")


class TestSyncScript:
    def _env(self, tmp_path, text=ENV):
        p = tmp_path / ".env"
        p.write_bytes(text.encode())
        return p

    def test_dry_run_changes_nothing_and_prints_no_secret(self, tmp_path, capsys):
        sync = _load_sync()
        p = self._env(tmp_path)
        before = p.read_bytes()
        rc = sync.main(["--env-file", str(p)], listening=lambda port: port == 4002)
        out = capsys.readouterr().out
        assert rc == 0 and p.read_bytes() == before
        assert "7497" in out and "4002" in out and "dry run" in out
        assert "dummy" not in out and "DU0000000" not in out

    def test_apply_changes_only_the_port_value(self, tmp_path, capsys):
        sync = _load_sync()
        p = self._env(tmp_path)
        mode = p.stat().st_mode
        rc = sync.main(["--env-file", str(p), "--apply"], listening=lambda port: port == 4002)
        assert rc == 0
        want = ENV.replace("IBKR_PORT=7497", "IBKR_PORT=4002", 1)
        assert p.read_bytes() == want.encode()          # CRLF, comment, every other line intact
        assert p.stat().st_mode == mode
        assert "dummy" not in capsys.readouterr().out
        assert not list(tmp_path.glob("*.sync_tmp"))

    def test_in_sync_is_a_noop(self, tmp_path, capsys):
        sync = _load_sync()
        p = self._env(tmp_path)
        before = p.read_bytes()
        assert sync.main(["--env-file", str(p), "--apply"], listening=lambda port: port == 7497) == 0
        assert p.read_bytes() == before and "in sync" in capsys.readouterr().out

    def test_nothing_listening_changes_nothing(self, tmp_path, capsys):
        sync = _load_sync()
        p = self._env(tmp_path)
        before = p.read_bytes()
        assert sync.main(["--env-file", str(p), "--apply"], listening=lambda port: False) == 0
        assert p.read_bytes() == before and "nothing is listening" in capsys.readouterr().out

    def test_refuses_a_live_target_for_a_paper_port(self, tmp_path, capsys):
        sync = _load_sync()
        p = self._env(tmp_path)
        before = p.read_bytes()
        assert sync.main(["--env-file", str(p), "--apply", "--port", "7496"],
                         listening=lambda port: True) == 3
        assert p.read_bytes() == before and "REFUSED" in capsys.readouterr().out

    def test_missing_line_or_file_is_an_error(self, tmp_path):
        sync = _load_sync()
        p = self._env(tmp_path, "FOO=1\n")
        assert sync.main(["--env-file", str(p)], listening=lambda port: True) == 2
        assert sync.main(["--env-file", str(tmp_path / "nope")], listening=lambda port: True) == 2


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------

@pytest.fixture
def sup(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "_IB_ENDPOINT_LAST", None)
    monkeypatch.setattr(M, "_ALERT_GATE_LAST", {})
    return tmp_path


class TestSupervisorPort:
    def test_uses_the_twin_and_notes_it_once(self, sup, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002), \
             patch.object(M, "_alert", return_value=True) as alert:
            assert M._ib_port() == 4002
            assert M._ib_port() == 4002
        assert alert.call_count == 1 and "4002" in alert.call_args[0][0]
        rec = read_endpoint_state(sup / ENDPOINT_FILE_NAME)
        assert rec["fallback"] is True and rec["configured"] == 7497

    def test_configured_port_up_is_silent_and_recorded_in_sync(self, sup, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "4002")
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002), \
             patch.object(M, "_alert", return_value=True) as alert:
            assert M._ib_port() == 4002
        assert alert.call_count == 0
        assert read_endpoint_state(sup / ENDPOINT_FILE_NAME)["fallback"] is False

    def test_both_down_reports_the_configured_port_and_does_not_clear_a_mismatch(self, sup, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        write_endpoint_state(sup / ENDPOINT_FILE_NAME, 7497, 4002)
        with patch.object(M, "_gateway_listening", return_value=False), \
             patch.object(M, "_alert", return_value=True) as alert:
            assert M._ib_port() == 7497
        assert alert.call_count == 0
        assert read_endpoint_state(sup / ENDPOINT_FILE_NAME)["fallback"] is True

    def test_a_failing_note_never_breaks_the_port_answer(self, sup, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002), \
             patch.object(M, "_alert", side_effect=RuntimeError("telegram down")):
            assert M._ib_port() == 4002

    def test_not_written_into_the_real_data_dir_under_pytest(self, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        monkeypatch.setattr(M, "_IB_ENDPOINT_LAST", None)
        before = (REPO / "data" / ENDPOINT_FILE_NAME).exists()
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002), \
             patch.object(M, "_alert", return_value=True) as alert:
            assert M._ib_port() == 4002
        assert (REPO / "data" / ENDPOINT_FILE_NAME).exists() == before
        assert alert.call_count == 0

    def test_health_check_starts_the_bot_instead_of_deferring(self, sup, monkeypatch):
        """The 2026-10-23 incident: configured 7497 down, 4002 up -> no deferral."""
        monkeypatch.setenv("IBKR_PORT", "7497")
        m = M.BotManager.__new__(M.BotManager)
        m._proc = None
        m._restarts = 0
        m._max_restarts = 5
        m._last_restart = None
        m._gateway_down_since = None
        m._last_gateway_alert = None
        m._last_gateway_escalation = None
        m._gateway_recoveries = 0
        m._last_gateway_recovery = None
        m._log_handle = None
        started = []
        with patch.object(M, "_gateway_listening", side_effect=lambda p, **k: p == 4002), \
             patch.object(M, "_alert", return_value=True) as alert, \
             patch.object(M, "_in_post_market_window", return_value=False), \
             patch.object(M.BotManager, "start", lambda self: started.append(True)), \
             patch.object(M.Path, "exists", return_value=False):
            m.health_check()
        assert started, "the bot must be started when the twin endpoint is up"
        assert not any("unreachable" in str(c) for c in alert.call_args_list)


class TestNoDirectPortReadsInMaster:
    def test_only_the_one_reader_touches_ibkr_port(self):
        """Class guard: every site goes through _ib_port()/_configured_ib_port()."""
        tree = ast.parse((REPO / "src/ait/orchestration/master.py").read_text())
        offenders = []

        class V(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_FunctionDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                for a in node.args:
                    if isinstance(a, ast.Constant) and a.value == "IBKR_PORT":
                        if "_configured_ib_port" not in self.stack:
                            offenders.append((node.lineno, "/".join(self.stack)))
                self.generic_visit(node)

        V().visit(tree)
        assert not offenders, f"read IBKR_PORT outside _configured_ib_port: {offenders}"


# ---------------------------------------------------------------------------
# IBKRClient.connect / reconnect follow the same fallback
# ---------------------------------------------------------------------------

import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import ait.broker.ibkr_client as BC  # noqa: E402


def _client(configured_port, accounts=("DU0000001",)):
    c = BC.IBKRClient.__new__(BC.IBKRClient)
    c._config = SimpleNamespace(ibkr_host="127.0.0.1", ibkr_port=configured_port,
                                ibkr_client_id=1, ibkr_account="")
    ib = MagicMock()
    ib.connectAsync = AsyncMock(return_value=None)
    ib.managedAccounts = lambda: list(accounts)
    ib.isConnected = lambda: True
    c._ib = ib
    c._connected = False
    c._reconnect_attempts = 0
    c._ever_connected = False
    c._last_client_id = None
    c.foreign_open_order_ids = set()
    return c, ib


class TestIBKRClientPortFallback:
    async def test_connects_on_the_twin_when_the_configured_port_is_closed(self, monkeypatch):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: p == 4002)
        c, ib = _client(7497)
        assert await c.connect() is True
        assert ib.connectAsync.await_args.kwargs["port"] == 4002

    async def test_configured_port_wins_when_it_is_up(self, monkeypatch):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: True)
        c, ib = _client(7497)
        assert await c.connect() is True
        assert ib.connectAsync.await_args.kwargs["port"] == 7497

    async def test_nothing_listening_still_tries_the_configured_port(self, monkeypatch):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: False)
        c, ib = _client(7497)
        await c.connect()
        assert ib.connectAsync.await_args.kwargs["port"] == 7497

    async def test_a_live_port_never_resolves_to_a_paper_port(self, monkeypatch):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: p in (7497, 4002))
        c, ib = _client(4001)           # live Gateway configured, only paper endpoints are up
        await c.connect()
        assert ib.connectAsync.await_args.kwargs["port"] == 4001

    async def test_fallback_is_logged_as_a_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: p == 4002)
        events = []
        monkeypatch.setattr(BC.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        c, _ = _client(7497)
        await c.connect()
        assert any(ev == "ibkr_port_fallback" and kw["using"] == 4002 and kw["configured"] == 7497
                   for ev, kw in events)

    async def test_reconnect_follows_a_mid_day_switch(self, monkeypatch):
        """_reconnect_locked calls connect(), so the next attempt resolves afresh."""
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        up = {4002}
        monkeypatch.setattr("ait.data.ib_ports.tcp_listening", lambda p, host="127.0.0.1", **k: p in up)
        c, ib = _client(7497)
        c._reconnect_delay = 0
        c._max_reconnect_attempts = 1
        monkeypatch.setattr(BC.asyncio, "sleep", AsyncMock())
        ib.isConnected = lambda: False
        assert await c._reconnect_locked() is True
        assert ib.connectAsync.await_args.kwargs["port"] == 4002


class TestNoCircularImport:
    @pytest.mark.parametrize("module", ["ait.broker.ibkr_client", "ait.data.market_data",
                                        "ait.data.ib_ports", "ait.main"])
    def test_each_module_imports_first_in_a_fresh_interpreter(self, module):
        """W94 live-check catch: a top-level `from ait.data.ib_ports import` in ibkr_client was a
        circular import whenever ibkr_client was the first module loaded."""
        import subprocess
        r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, 'src'); "
                            f"import {module}"], cwd=REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-800:]
