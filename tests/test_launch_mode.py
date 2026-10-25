"""Paper or live is decided by one function (ait/config/launch_mode.py) used by the supervisor and by the bot's own start.

The supervisor used to hardcode ``--paper``. That label does not choose the account: the IB endpoint does, so a
"paper" bot pointed at a live-account port traded real money. These tests pin the decision table and run the real
``BotManager.start`` and ``ait.main.run_bot`` paths against it.
"""
from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ait.orchestration.master as M
from ait.config.launch_mode import account_kind, decide, port_kind


class TestClassification:
    @pytest.mark.parametrize("port,kind", [(4002, "paper"), (7497, "paper"), (4001, "live"), (7496, "live"), (4999, "unknown")])
    def test_ports(self, port, kind):
        assert port_kind(port) == kind

    @pytest.mark.parametrize("acct,kind", [("DU1234567", "paper"), ("du1", "paper"), ("U1234567", "live"), ("", "unknown"), (None, "unknown")])
    def test_accounts(self, acct, kind):
        assert account_kind(acct) == kind


class TestDecision:
    # (config, env TRADING_MODE, AIT_ALLOW_LIVE, port, account) -> (mode, refused)
    @pytest.mark.parametrize("args,mode,refused", [
        (("paper", None, False, 4002, "DU1"), "paper", False),            # the shipped setup
        (("paper", None, False, 7497, ""), "paper", False),
        (("paper", None, False, 4001, ""), "paper", True),                # paper label, live-account endpoint
        (("paper", None, False, 4002, "U123"), "paper", True),            # paper label, live account id
        (("live", None, True, 4001, "U1"), "paper", True),                # config says live, .env does not: paper, and the endpoint is live
        (("live", None, True, 4002, "DU1"), "paper", False),              # disagreement on a paper endpoint just runs paper
        (("paper", "live", True, 4002, "DU1"), "paper", False),
        (("live", "live", False, 4001, "U1"), "live", True),              # both live but no interlock
        (("live", "live", True, 4002, "U1"), "live", True),               # live label on a paper endpoint
        (("live", "live", True, 4001, "DU1"), "live", True),              # live label with a paper account id
        (("live", "live", True, 4001, "U1"), "live", False),              # the one fully enabled case
        (("live", "live", True, 7496, ""), "live", False),                # TWS live, account not set
        (("live", "live", True, 4999, "U1"), "live", True),               # an unclassifiable port cannot prove live
    ])
    def test_table(self, args, mode, refused):
        d = decide(*args)
        assert d.mode == mode
        assert bool(d.error) is refused, d

    def test_disagreement_is_noted(self):
        d = decide("live", "paper", True, 4002, "DU1")
        assert d.mode == "paper" and d.note and "live needs both" in d.note

    def test_case_and_blanks_are_forgiven(self):
        assert decide(" LIVE ", "Live", True, 4001, "U1").error is None


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Drive the real BotManager.start with everything outside it stubbed: returns (run, popen, pages)."""
    monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_scan_foreign_bot_pids", lambda: [])
    monkeypatch.setattr("ait.orchestration.gateway.ensure_gateway", lambda port, **kw: True)
    popen = MagicMock(return_value=SimpleNamespace(pid=4242, poll=lambda: None, returncode=None))
    monkeypatch.setattr(M.subprocess, "Popen", popen)
    pages: list[str] = []
    monkeypatch.setattr(M, "_alert", lambda msg, **kw: (pages.append(msg), True)[1])
    M._ALERT_GATE_LAST.clear()
    for k in ("TRADING_MODE", "AIT_ALLOW_LIVE", "IBKR_ACCOUNT"):
        monkeypatch.delenv(k, raising=False)

    def run(config_mode, port, env=None):
        monkeypatch.setattr(M, "_ib_port", lambda: port)
        monkeypatch.setattr("ait.config.settings.load_settings", lambda *a, **k: SimpleNamespace(trading=SimpleNamespace(mode=config_mode)))
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        M.BotManager().start()

    yield run, popen, pages
    M._ALERT_GATE_LAST.clear()


class TestSupervisorLaunch:
    def test_paper_setup_still_starts_with_paper(self, started):
        run, popen, pages = started
        run("paper", 4002)
        assert popen.called and popen.call_args.args[0][-1] == "--paper" and not pages

    def test_fully_enabled_live_starts_without_the_paper_flag(self, started):
        run, popen, pages = started
        run("live", 4001, {"TRADING_MODE": "live", "AIT_ALLOW_LIVE": "1", "IBKR_ACCOUNT": "U1234567"})
        assert popen.called and "--paper" not in popen.call_args.args[0] and not pages

    def test_live_in_config_only_starts_as_paper_on_a_paper_endpoint(self, started):
        run, popen, pages = started
        run("live", 4002, {"IBKR_ACCOUNT": "DU1234567"})
        assert popen.call_args.args[0][-1] == "--paper"

    def test_live_in_config_only_on_a_live_endpoint_is_refused(self, started):
        run, popen, pages = started
        run("live", 4001, {"AIT_ALLOW_LIVE": "1"})
        assert not popen.called
        assert len(pages) == 1 and "REFUSING" in pages[0] and "LIVE-account endpoint" in pages[0]

    def test_both_live_without_the_interlock_is_refused(self, started):
        run, popen, pages = started
        run("live", 4001, {"TRADING_MODE": "live"})
        assert not popen.called and "AIT_ALLOW_LIVE" in pages[0]

    def test_the_refusal_page_is_throttled(self, started):
        run, popen, pages = started
        for _ in range(3):
            run("paper", 4001)
        assert not popen.called and len(pages) == 1


class TestBotOwnStart:
    def _run_bot(self, monkeypatch, mode, port, env=None, paper=False):
        import ait.main as main
        from ait.config.settings import load_settings
        s = load_settings()
        s.trading.mode = mode
        s.ibkr.ibkr_port = port
        s.ibkr.ibkr_account = (env or {}).get("IBKR_ACCOUNT", "")
        monkeypatch.setattr(main, "load_settings", lambda *a, **k: s)
        for k in ("TRADING_MODE", "AIT_ALLOW_LIVE"):
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        return main, s, argparse.Namespace(config="config.yaml", paper=paper, dashboard_only=False)

    def test_a_live_endpoint_without_full_live_is_refused_before_anything_connects(self, monkeypatch):
        main, s, args = self._run_bot(monkeypatch, "paper", 4001)
        with pytest.raises(SystemExit) as e:
            asyncio.run(main.run_bot(args))
        assert e.value.code == 1

    def test_the_paper_flag_does_not_bypass_the_endpoint_check(self, monkeypatch):
        main, s, args = self._run_bot(monkeypatch, "live", 4001, paper=True)
        with pytest.raises(SystemExit):
            asyncio.run(main.run_bot(args))

    def test_a_config_live_bot_with_no_env_vote_is_forced_to_paper(self, monkeypatch):
        main, s, args = self._run_bot(monkeypatch, "live", 4002)
        monkeypatch.setattr(main, "setup_logging", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
        with pytest.raises(RuntimeError, match="stop here"):
            asyncio.run(main.run_bot(args))
        assert s.trading.mode == "paper"
