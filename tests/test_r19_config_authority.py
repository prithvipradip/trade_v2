"""R19: ONE authority for the runtime env contract.

User audit question: "have we hardcoded any config in code instead of config
files?" Answer for the env contract was yes, in 17 reader sites — and FOUR
disagreed with the contract they were supposed to mirror:

    AIT_IC_WING_K            contract 1.6  -> readers 1.0  (x2)
    AIT_IC_MIN_CREDIT_WIDTH  contract 0.10 -> readers 0.20 (x4)
    AIT_SKIP_MACRO_EVENTS    contract "1"  -> readers "0"  (x3, FAIL-OPEN)
    AIT_CREDIT_LOSS_LIMIT    absent        -> readers split 0 vs 1.25

Live was safe only because both entry points call apply_runtime_env_defaults()
first. Any path that skipped it ran pre-promotion economics with macro-event
protection OFF — the wing_k four-sources incident recurring one layer down.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ait.config.runtime_env import (CONTRACT_DEFAULTS, apply_runtime_env_defaults,
                                    contract_flag, contract_float, contract_str)

SRC = Path(__file__).resolve().parents[1] / "src" / "ait"
# executor.py's AIT_ALLOW_UNDEFINED_RISK read is the INST-5 executor backstop:
# it deliberately treats "anything but 1" as refuse, which is already
# fail-safe and matches the contract default.
_ALLOWED_RAW_READS = {"executor.py"}


class TestContractIsTheSingleAuthority:
    def test_bare_process_resolves_promoted_values(self, monkeypatch):
        """The regression that matters: NO applier called, values still right."""
        for k in CONTRACT_DEFAULTS:
            monkeypatch.delenv(k, raising=False)
        assert contract_float("AIT_IC_WING_K") == pytest.approx(1.6)
        assert contract_float("AIT_IC_MIN_CREDIT_WIDTH") == pytest.approx(0.10)
        assert contract_flag("AIT_SKIP_MACRO_EVENTS") is True   # fail-CLOSED
        assert contract_float("AIT_CREDIT_LOSS_LIMIT") == pytest.approx(0.0)

    def test_applier_and_readers_share_one_table(self, monkeypatch):
        for k in CONTRACT_DEFAULTS:
            monkeypatch.delenv(k, raising=False)
        apply_runtime_env_defaults()
        import os
        for key, declared in CONTRACT_DEFAULTS.items():
            assert os.environ[key] == declared
            assert contract_str(key) == declared

    def test_explicit_override_still_wins(self, monkeypatch):
        monkeypatch.setenv("AIT_IC_WING_K", "0.8")
        assert contract_float("AIT_IC_WING_K") == pytest.approx(0.8)

    def test_malformed_override_falls_back_not_crashes(self, monkeypatch):
        # a bad value must never take down a trading cycle
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "not-a-number")
        assert contract_float("AIT_IC_MIN_CREDIT_WIDTH") == pytest.approx(0.10)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(KeyError):
            contract_str("AIT_NOT_IN_CONTRACT")


class TestNoReaderReintroducesItsOwnFallback:
    """The rule this file exists to enforce: no production module may carry a
    private default for a contract key. A promotion must edit exactly one
    place (CONTRACT_DEFAULTS) and take effect everywhere."""

    def test_no_raw_environ_reads_of_contract_keys(self):
        pattern = re.compile(
            r'environ\.get\(\s*"(' + "|".join(
                k for k in CONTRACT_DEFAULTS if k.startswith("AIT_")) + r')"')
        offenders = []
        for path in SRC.rglob("*.py"):
            if path.name in _ALLOWED_RAW_READS:
                continue
            if path.name == "runtime_env.py":
                continue          # the authority itself
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in pattern.finditer(text):
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line} reads {m.group(1)} directly")
        assert not offenders, (
            "contract keys must be read via contract_float/flag/str so one "
            "table governs every reader:\n  " + "\n  ".join(offenders))
