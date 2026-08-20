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
        # R19c: config-backed keys seed the CONFIG value (which today equals
        # the contract semantically but may differ in string form, e.g. yaml
        # 0.10 -> "0.1"). Compare resolved SEMANTICS, not string formatting.
        for k in CONTRACT_DEFAULTS:
            monkeypatch.delenv(k, raising=False)
        apply_runtime_env_defaults()
        import os
        for key, declared in CONTRACT_DEFAULTS.items():
            assert os.environ[key] == contract_str(key)
            try:
                assert float(os.environ[key]) == pytest.approx(float(declared))
            except ValueError:  # non-numeric (TRUE etc.)
                assert os.environ[key] == declared

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


class TestDefaultDivergenceReport:
    """R19 second shadowing layer: pydantic Field defaults vs config.yaml.

    Every module that builds a config model BARE (Backtester(),
    WalkForwardConfig(), the optimizer, the trainer) gets the Field default,
    NOT the yaml operating value — so research can validate a different bot
    than the one that trades. wing_k was ALIGNED (a different wing size is a
    different strategy, not a safer fallback); the rest stay deliberately
    STRICTER in code, but must never diverge SILENTLY again.
    """

    def test_wing_k_default_tracks_the_live_contract(self):
        from ait.config.settings import BacktestConfig
        from ait.config.runtime_env import CONTRACT_DEFAULTS
        assert BacktestConfig().wing_k == float(CONTRACT_DEFAULTS["AIT_IC_WING_K"])

    def test_report_lists_yaml_overrides(self):
        from ait.config.settings import load_settings, default_divergences
        d = dict((n, (c, a)) for n, c, a in
                 default_divergences(load_settings("config.yaml")))
        # known, deliberate overrides — the report must SEE them
        assert "options.max_bid_ask_spread_pct" in d
        assert "risk.min_confidence" in d
        # and wing_k must NOT appear (it is now aligned)
        assert "backtest.wing_k" not in d

    def test_report_never_contains_secrets(self):
        """The report writes to the structured log. A live Finnhub key already
        reached 11 log files once (R13 #12); this report must not be the
        second route."""
        from ait.config.settings import load_settings, default_divergences
        secret_markers = ("key", "token", "secret", "password", "chat_id")
        for name, code, active in default_divergences(load_settings("config.yaml")):
            low = name.lower()
            assert not any(m in low for m in secret_markers), (
                f"{name} would log a secret VALUE")
            assert not low.startswith(("api_keys.", "ibkr."))


class TestConfigBeatsDefault:
    """R19b (user question: "the config must have priority over the default,
    right?"). It did NOT: resolution went env -> hardcoded default and skipped
    config.yaml entirely, so editing backtest.wing_k changed nothing on the
    live/default path. Precedence is now: explicit env > config.yaml >
    CONTRACT_DEFAULTS — at the reader AND in the applier (which seeds env for
    live processes and would otherwise mask config forever).
    """

    def _fresh(self, monkeypatch, yaml_value):
        import ait.config.runtime_env as m
        monkeypatch.delenv("AIT_IC_WING_K", raising=False)
        monkeypatch.setattr(m, "_yaml_cache",
                            {"backtest": {"wing_k": yaml_value}} if yaml_value
                            is not None else {})
        return m

    def test_config_yaml_beats_table_default(self, monkeypatch):
        m = self._fresh(monkeypatch, 2.2)
        assert m.contract_float("AIT_IC_WING_K") == pytest.approx(2.2)

    def test_env_beats_config_yaml(self, monkeypatch):
        m = self._fresh(monkeypatch, 2.2)
        monkeypatch.setenv("AIT_IC_WING_K", "0.8")
        assert m.contract_float("AIT_IC_WING_K") == pytest.approx(0.8)

    def test_applier_seeds_config_value_not_table_default(self, monkeypatch):
        # In a LIVE process the applier populates env; if it seeded the table
        # default, config.yaml could never take effect anywhere. It must seed
        # the config value.
        import os
        m = self._fresh(monkeypatch, 2.2)
        m.apply_runtime_env_defaults()
        assert os.environ["AIT_IC_WING_K"] == "2.2"

    def test_missing_config_home_falls_to_table_default(self, monkeypatch):
        m = self._fresh(monkeypatch, None)
        assert m.contract_float("AIT_IC_WING_K") == pytest.approx(1.6)


class TestConfigIsTheOperatingSource:
    """R19c (user policy: "config must be the ONLY place with values that
    manage trading"). Every trading-economics contract key must have a
    config.yaml home; env is an override, CONTRACT_DEFAULTS a safety net.
    Named exceptions (each documented in runtime_env.py): the OpenMP crash
    guards, AIT_MARKET_DATA_TYPE (broker entitlement, flips at U6), and
    AIT_ALLOW_UNDEFINED_RISK (INST-5 interlock, env-only on purpose).
    """

    _EXEMPT = {"KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS",
               "AIT_MARKET_DATA_TYPE", "AIT_ALLOW_UNDEFINED_RISK"}

    def test_every_trading_key_has_a_config_home(self):
        from ait.config.runtime_env import CONTRACT_DEFAULTS, CONFIG_BACKED
        missing = [k for k in CONTRACT_DEFAULTS
                   if k not in self._EXEMPT and k not in CONFIG_BACKED]
        assert not missing, (
            f"trading keys without a config.yaml home: {missing} — add the "
            "field to settings.py + config.yaml + CONFIG_BACKED, or document "
            "an exemption")

    def test_config_homes_exist_in_yaml_and_settings(self):
        import yaml as _yaml
        from ait.config.runtime_env import CONFIG_BACKED, REPO_ROOT
        from ait.config.settings import load_settings
        y = _yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
        st = load_settings(str(REPO_ROOT / "config.yaml"))
        for key, (section, field) in CONFIG_BACKED.items():
            assert field in (y.get(section) or {}), (
                f"{key}: config.yaml [{section}] lacks '{field}'")
            assert hasattr(getattr(st, section), field), (
                f"{key}: settings.{section} lacks '{field}' (extra=forbid "
                "would reject the yaml key)")

    def test_yaml_bool_normalizes_to_contract_flag(self, monkeypatch):
        import ait.config.runtime_env as m
        monkeypatch.delenv("AIT_SKIP_MACRO_EVENTS", raising=False)
        monkeypatch.setattr(m, "_yaml_cache", {"risk": {"skip_macro_events": False}})
        assert m.contract_flag("AIT_SKIP_MACRO_EVENTS") is False
        monkeypatch.setattr(m, "_yaml_cache", {"risk": {"skip_macro_events": True}})
        assert m.contract_flag("AIT_SKIP_MACRO_EVENTS") is True

    def test_config_values_match_live_operating_state(self):
        """The yaml declarations added in R19c must equal what live actually
        runs today — this change is about WHERE values live, not WHAT they are."""
        from ait.config.settings import load_settings
        st = load_settings("config.yaml")
        assert st.backtest.wing_k == pytest.approx(1.6)
        assert st.backtest.ic_min_credit_width == pytest.approx(0.10)
        assert st.backtest.ic_min_credit == pytest.approx(0.70)
        assert st.backtest.credit_loss_limit == pytest.approx(0.0)
        assert st.risk.skip_macro_events is True


class TestR20LiveRegisterMigrations:
    """R20: two live-side register items moved into config as PURE
    relocations — semantics identical, ownership changed."""

    def test_vix_tier_semantics_unchanged(self):
        from ait.config.settings import RiskConfig
        tiers = RiskConfig().credit_cap_vix_tiers
        def cap(v):
            return next((int(c) for ceil, c in tiers if v < float(ceil)), 2)
        # the exact historical mapping the literal implemented
        assert (cap(15.0), cap(19.9), cap(20.0), cap(24.9), cap(25.0), cap(40.0)) \
            == (6, 6, 4, 4, 2, 2)

    def test_manager_reads_config_not_literals(self):
        import inspect
        from ait.risk import manager
        src = inspect.getsource(manager)
        assert "credit_cap_vix_tiers" in src
        assert "max_symbol_concentration_pct" in src
        # the old literals must be gone from the validation path
        assert "6 if request.vix < 20" not in src
        assert "account_value * 0.20" not in src

    def test_entry_signals_capture_shape(self):
        """R20: trade_context.entry_signals must carry the 11 technical
        META_FEATURES + hour_of_day (was '{}' on every trade ever taken)."""
        import json
        import pandas as pd
        from ait.bot.orchestrator import TradingOrchestrator as T
        from ait.ml.meta_label import META_FEATURES
        o = T.__new__(T)
        tech = [f for f in META_FEATURES if f not in (
            "primary_confidence", "regime_trending_up", "regime_trending_down",
            "regime_high_vol", "regime_range_bound", "vix", "iv_rank",
            "sentiment_score", "hour_of_day")]
        o._entry_feature_snap = {"QQQ": pd.Series({t: 1.0 for t in tech})}
        d = json.loads(o._entry_signals_json("QQQ"))
        assert set(d) == set(tech) | {"hour_of_day"}
        # degradation paths never block an entry
        assert o._entry_signals_json("SPY") == "{}"
        assert T.__new__(T)._entry_signals_json("QQQ") == "{}"
