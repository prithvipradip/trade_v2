"""W70 — one wing floor for live AND research; research-engine knobs follow config;
the earnings window matches live; and a guard against config keys nothing reads.

Why this exists (all verified in the W70 investigation):
  * live's minimum wing was a LITERAL 2.0 in strategies/iron_condor.py while research
    read backtest.wing_floor_dollars, so the key only ever aligned research TO a number
    live did not read. Now both read AIT_IC_WING_FLOOR (config home
    backtest.wing_floor_dollars) through the same env-contract call.
  * the four backtest.spread_* keys were read by nothing; the engine and the walk-forward
    carried their own literals.
  * the research earnings gate blocked [earnings-1, earnings+2]; live blocks
    [earnings-2, earnings+1] (earnings.buffer_days_before/after).

Behavioral tests drive the real builder / engine; class guards are AST walks.
"""
from __future__ import annotations

import ast
import inspect
import sys
import types
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ait.config import runtime_env
from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING
from ait.config.runtime_env import CONFIG_BACKED, CONTRACT_DEFAULTS, contract_float
from ait.config.settings import (
    BacktestConfig,
    EarningsConfig,
    ExitConfig,
    Settings,
    load_settings,
)
from ait.strategies.iron_condor import IronCondor
from ait.utils.time import today_et

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"
CONFIG_PATH = ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# 1. LIVE and research read the same wing floor
# --------------------------------------------------------------------------- #

@pytest.fixture
def clean_contract(monkeypatch):
    """No env override and no config.yaml value: the contract default decides."""
    monkeypatch.delenv("AIT_IC_WING_FLOOR", raising=False)
    monkeypatch.delenv("AIT_IC_MIN_CREDIT_WIDTH", raising=False)
    monkeypatch.delenv("AIT_IC_WING_K", raising=False)
    monkeypatch.setattr(runtime_env, "_config_yaml", lambda: {})


def _set_config_floor(monkeypatch, value):
    monkeypatch.delenv("AIT_IC_WING_FLOOR", raising=False)
    monkeypatch.setattr(runtime_env, "_config_yaml",
                        lambda: {"backtest": {"wing_floor_dollars": value}})


def _low_vol_inputs():
    """A chain whose vol-scaled width is ~$1.4 (wing_k 1.6 x $100 x 3% IV x sqrt(30/365)),
    i.e. UNDER every plausible floor, so the floor is what decides."""
    expiry = (today_et() + timedelta(days=30)).isoformat()
    leg = SimpleNamespace(implied_vol=0.03)
    chain = SimpleNamespace(expiry=expiry, atm_iv=0.03)
    return 100.0, leg, leg, chain


class TestLiveBuilderReadsTheContractKey:
    def test_contract_default_and_config_home(self, clean_contract):
        assert CONTRACT_DEFAULTS["AIT_IC_WING_FLOOR"] == "2.0"
        assert CONFIG_BACKED["AIT_IC_WING_FLOOR"] == ("backtest", "wing_floor_dollars")
        assert contract_float("AIT_IC_WING_FLOOR") == 2.0
        assert BacktestConfig().wing_floor_dollars == 2.0     # model default == contract

    def test_live_default_is_still_2_dollars(self, clean_contract):
        price, put, call, chain = _low_vol_inputs()
        assert IronCondor()._vol_scaled_width(price, put, call, chain) == pytest.approx(2.0)

    def test_changing_the_config_value_changes_the_live_wing(self, monkeypatch, clean_contract):
        price, put, call, chain = _low_vol_inputs()
        _set_config_floor(monkeypatch, 3.5)
        assert IronCondor()._vol_scaled_width(price, put, call, chain) == pytest.approx(3.5)
        _set_config_floor(monkeypatch, 5.0)
        assert IronCondor()._vol_scaled_width(price, put, call, chain) == pytest.approx(5.0)

    def test_env_override_beats_config(self, monkeypatch, clean_contract):
        price, put, call, chain = _low_vol_inputs()
        _set_config_floor(monkeypatch, 3.5)
        monkeypatch.setenv("AIT_IC_WING_FLOOR", "4.0")
        assert IronCondor()._vol_scaled_width(price, put, call, chain) == pytest.approx(4.0)

    def test_a_wider_vol_scaled_width_is_untouched_by_the_floor(self, monkeypatch, clean_contract):
        """The floor is a floor: a vol-scaled width above it is not changed."""
        expiry = (today_et() + timedelta(days=30)).isoformat()
        leg = SimpleNamespace(implied_vol=0.20)         # 1.6 x 100 x .20 x .2867 ~ $9.2
        chain = SimpleNamespace(expiry=expiry, atm_iv=0.20)
        _set_config_floor(monkeypatch, 3.5)
        w = IronCondor()._vol_scaled_width(100.0, leg, leg, chain)
        assert w == pytest.approx(1.6 * 100 * 0.20 * (30 / 365) ** 0.5, rel=1e-6)
        assert w > 3.5

    def test_the_budget_cap_can_still_shrink_below_the_floor(self, clean_contract):
        """Documented behavior: the floor binds only when the budget does not.
        budget $150 -> affordable = 150 / (100 x 0.9) = $1.67 < the $2 floor."""
        price, put, call, chain = _low_vol_inputs()
        ic = IronCondor()
        ic.risk_budget = 150.0
        assert ic._vol_scaled_width(price, put, call, chain) == pytest.approx(150 / 90)

    def test_the_budget_capped_minimum_is_the_fixed_invariant(self, clean_contract):
        price, put, call, chain = _low_vol_inputs()
        ic = IronCondor()
        ic.risk_budget = 50.0                              # affordable $0.56
        assert ic._vol_scaled_width(price, put, call, chain) == IC_BUDGET_CAPPED_MIN_WING == 1.0

    def test_class_guard_the_live_builder_carries_no_wing_literal(self):
        """Re-typing `max(2.0, ...)` (or the $1 relaxed floor) in the live builder would
        silently fork live from research again."""
        tree = ast.parse((SRC / "strategies" / "iron_condor.py").read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_vol_scaled_width")
        bad = []
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "max":
                for a in call.args:
                    # float literals only: `max(1, dte)` is the unrelated 1-day DTE minimum
                    if (isinstance(a, ast.Constant) and isinstance(a.value, float)
                            and a.value in (1.0, 2.0)):
                        bad.append(f"line {call.lineno}: {ast.unparse(call)}")
        assert not bad, "wing literal re-introduced in the live builder:\n  " + "\n  ".join(bad)


class TestResearchReadsTheSameKey:
    def _df(self):
        idx = pd.bdate_range("2024-01-02", periods=60)
        return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                             "volume": 1_000_000, "implied_vol": 0.2}, index=idx)

    def test_engine_walkforward_optimizer_and_cli_follow_the_contract(
            self, monkeypatch, clean_contract):
        from ait.backtesting.engine import Backtester
        from ait.backtesting.walkforward import WalkForwardConfig
        from ait.optimization.optimizer import StrategyOptimizer

        def values():
            return (
                Backtester(data=self._df(), strategies=[IronCondor()])._wing_floor_dollars,
                WalkForwardConfig().wing_floor_dollars,
                StrategyOptimizer(["SPY"], ["iron_condor"])._wing_floor_dollars,
            )
        assert values() == (2.0, 2.0, 2.0)
        _set_config_floor(monkeypatch, 3.5)
        assert values() == (3.5, 3.5, 3.5)
        monkeypatch.setenv("AIT_IC_WING_FLOOR", "6.0")
        assert values() == (6.0, 6.0, 6.0)

    def test_explicit_argument_still_wins(self, clean_contract):
        from ait.backtesting.engine import Backtester
        from ait.backtesting.walkforward import WalkForwardConfig
        assert Backtester(data=self._df(), strategies=[IronCondor()],
                          wing_floor_dollars=5.0)._wing_floor_dollars == 5.0
        assert WalkForwardConfig(wing_floor_dollars=5.0).wing_floor_dollars == 5.0

    def test_run_backtest_cli_default_is_resolved_not_a_literal(self):
        tree = ast.parse((ROOT / "run_backtest.py").read_text())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                    and n.args and getattr(n.args[0], "value", None) == "--wing-floor"):
                default = next(k.value for k in n.keywords if k.arg == "default")
                assert isinstance(default, ast.Constant) and default.value is None
                return
        pytest.fail("--wing-floor argument not found")

    def test_engine_budget_relaxation_uses_the_invariant(self):
        src = (SRC / "backtesting" / "engine.py").read_text()
        assert "IC_BUDGET_CAPPED_MIN_WING" in src
        tree = ast.parse(src)
        bad = [c.lineno for c in ast.walk(tree)
               if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "max"
               and any(isinstance(a, ast.Constant) and a.value == 1.0 for a in c.args)
               and "round" in ast.unparse(c)]
        assert not bad, f"engine re-hardcoded the $1 budget-capped floor at lines {bad}"


class TestWingFloorKeyIsSelfExplaining:
    def test_bounds(self):
        for bad in (0.9, 10.5, 0.0, 50.0):
            with pytest.raises(ValidationError):
                BacktestConfig(wing_floor_dollars=bad)
        BacktestConfig(wing_floor_dollars=1.0)
        BacktestConfig(wing_floor_dollars=10.0)

    def test_description_and_yaml_say_it_controls_live(self):
        desc = BacktestConfig.model_fields["wing_floor_dollars"].description
        assert "LIVE ENTRIES AND RESEARCH" in desc and "AIT_IC_WING_FLOOR" in desc
        lines = CONFIG_PATH.read_text().splitlines()
        idx = next(i for i, l in enumerate(lines) if l.strip().startswith("wing_floor_dollars:"))
        block = "\n".join(lines[max(0, idx - 14):idx])
        assert "LIVE ENTRIES AND RESEARCH" in block
        assert "AIT_IC_WING_FLOOR" in block and "IMPACT:" in block

    def test_shipped_yaml_value(self):
        assert load_settings(str(CONFIG_PATH)).backtest.wing_floor_dollars == 2.0
        assert yaml.safe_load(CONFIG_PATH.read_text())["backtest"]["wing_floor_dollars"] == 2.0


# --------------------------------------------------------------------------- #
# 2. Research-engine knobs follow config
# --------------------------------------------------------------------------- #

NEW_BACKTEST_DEFAULTS = {
    "slippage_pct": 0.01, "delta_short": 0.20, "delta_long": 0.30,
    "hurst_hard_veto_multiplier": 1.5, "pct_from_60d_high_threshold": -1.0,
    "skew_factor": 1.0, "context_bars": 0,
}


class TestResearchKnobs:
    def test_new_fields_default_to_the_former_engine_literals(self):
        b = BacktestConfig()
        assert {k: getattr(b, k) for k in NEW_BACKTEST_DEFAULTS} == NEW_BACKTEST_DEFAULTS
        yb = yaml.safe_load(CONFIG_PATH.read_text())["backtest"]
        assert {k: yb[k] for k in NEW_BACKTEST_DEFAULTS} == NEW_BACKTEST_DEFAULTS

    def _bt(self, **settings_kw):
        from ait.backtesting.engine import Backtester
        idx = pd.bdate_range("2024-01-02", periods=60)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                           "volume": 1_000_000, "implied_vol": 0.2}, index=idx)
        return Backtester(data=df, strategies=[IronCondor()], settings=Settings(**settings_kw))

    def test_bare_engine_is_unchanged_at_default_settings(self, clean_contract):
        bt = self._bt()
        assert (bt._slippage_pct, bt._delta_short, bt._delta_long, bt._skew_factor,
                bt._context_bars, bt._hurst_hard_veto_multiplier,
                bt._pct_from_60d_high_threshold) == (0.01, 0.20, 0.30, 1.0, 0, 1.5, -1.0)
        assert (bt._spread_base, bt._spread_iv_sensitivity, bt._spread_dte_sensitivity,
                bt._spread_cap) == (0.03, 0.10, 0.005, 0.15)
        assert (bt._max_hold_days, bt._trailing_stop_pct, bt._breakeven_trigger_pct,
                bt._max_entry_vol_annual) == (30, 0.25, 0.30, 0.80)

    def test_engine_follows_config_including_the_formerly_unread_spread_keys(self, clean_contract):
        bt = self._bt(
            backtest=BacktestConfig(
                slippage_pct=0.02, delta_short=0.15, delta_long=0.35, skew_factor=0.5,
                context_bars=40, hurst_hard_veto_multiplier=0.0,
                spread_base=0.07, spread_iv_sensitivity=0.2, spread_dte_sensitivity=0.01,
                spread_cap=0.25, max_hold_days=45, max_entry_vol_annual=0.9),
            exit=ExitConfig(trailing_stop_pct=0.35, breakeven_trigger_pct=0.4))
        assert (bt._slippage_pct, bt._delta_short, bt._delta_long, bt._skew_factor,
                bt._context_bars, bt._hurst_hard_veto_multiplier) == (
            0.02, 0.15, 0.35, 0.5, 40, 0.0)
        assert (bt._spread_base, bt._spread_iv_sensitivity, bt._spread_dte_sensitivity,
                bt._spread_cap) == (0.07, 0.2, 0.01, 0.25)
        assert (bt._max_hold_days, bt._trailing_stop_pct, bt._breakeven_trigger_pct,
                bt._max_entry_vol_annual) == (45, 0.35, 0.4, 0.9)

    def test_walkforward_config_follows_config(self, monkeypatch, clean_contract):
        from ait.backtesting.walkforward import WalkForwardConfig
        import ait.config.settings as sm
        cfg = Settings(backtest=BacktestConfig(spread_base=0.06, spread_cap=0.2,
                                               max_entry_vol_annual=0.7, delta_iv_scale=0.5),
                       exit=ExitConfig(trailing_stop_pct=0.3, breakeven_trigger_pct=0.35))
        monkeypatch.setattr(sm, "load_settings", lambda *a, **k: cfg)
        w = WalkForwardConfig()
        assert (w.spread_base, w.spread_cap, w.max_entry_vol_annual, w.delta_iv_scale,
                w.trailing_stop_pct, w.breakeven_trigger_pct) == (0.06, 0.2, 0.7, 0.5, 0.3, 0.35)

    def test_stop_and_target_follow_the_config(self, clean_contract):
        """Decision D, CLOSED in W93 (it was 'registered, NOT changed': the bare engine kept 0.50 / 1.00 although the config
        baselines are 0.35 / 0.50). The bare engine now resolves backtest.stop_loss_pct / profit_target_pct like every sibling
        knob; an explicit argument still wins."""
        bt = self._bt(backtest=BacktestConfig(stop_loss_pct=0.35, profit_target_pct=0.5))
        assert (bt._stop_loss_pct, bt._profit_target_pct) == (0.35, 0.50)
        moved = self._bt(backtest=BacktestConfig(stop_loss_pct=0.21, profit_target_pct=0.66))
        assert (moved._stop_loss_pct, moved._profit_target_pct) == (0.21, 0.66)
        from ait.backtesting.engine import Backtester
        idx = pd.bdate_range("2024-01-02", periods=60)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                           "volume": 1_000_000, "implied_vol": 0.2}, index=idx)
        explicit = Backtester(data=df, strategies=[IronCondor()], stop_loss_pct=0.50, profit_target_pct=1.00,
                              settings=Settings(backtest=BacktestConfig(stop_loss_pct=0.21, profit_target_pct=0.66)))
        assert (explicit._stop_loss_pct, explicit._profit_target_pct) == (0.50, 1.00)


# --------------------------------------------------------------------------- #
# 3. Earnings window parity with live
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_yfinance(monkeypatch):
    mod = types.ModuleType("yfinance")

    class _T:
        def __init__(self, symbol):
            self.earnings_dates = pd.DataFrame(
                {"x": [1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-10-15")]))
    mod.Ticker = _T
    monkeypatch.setitem(sys.modules, "yfinance", mod)


class TestEarningsWindowIsLives:
    ED = date(2026, 10, 15)

    def _live_blocked(self, before, after):
        """live: EarningsCalendar refuses when -buffer_after <= days_until <= buffer_before."""
        return {self.ED + timedelta(days=d) for d in range(-before, after + 1)
                if -after <= (self.ED - (self.ED + timedelta(days=d))).days <= before}

    def test_default_window_is_live_two_before_one_after(self, fake_yfinance):
        from ait.backtesting.engine import Backtester
        got = Backtester._load_earnings_dates("XYZ", 2, 1)
        assert got == {date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 15),
                       date(2026, 10, 16)}
        assert got == self._live_blocked(2, 1)

    def test_not_the_old_mirror_image(self, fake_yfinance):
        """The pre-W70 engine blocked [-1, +2] (10-14 .. 10-17)."""
        from ait.backtesting.engine import Backtester
        got = Backtester._load_earnings_dates("XYZ", 2, 1)
        assert date(2026, 10, 17) not in got and date(2026, 10, 13) in got

    def test_both_zero_is_off_and_blank_symbol_is_empty(self, fake_yfinance):
        from ait.backtesting.engine import Backtester
        assert Backtester._load_earnings_dates("XYZ", 0, 0) == set()
        assert Backtester._load_earnings_dates(None, 2, 1) == set()

    def test_engine_reads_the_live_config_keys(self, fake_yfinance, clean_contract):
        from ait.backtesting.engine import Backtester
        idx = pd.bdate_range("2024-01-02", periods=60)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                           "volume": 1_000_000, "implied_vol": 0.2}, index=idx)
        s = Settings(earnings=EarningsConfig(buffer_days_before=3, buffer_days_after=0))
        bt = Backtester(data=df, strategies=[IronCondor()], symbol="XYZ", settings=s)
        assert bt._earnings_dates == {self.ED + timedelta(days=d) for d in (-3, -2, -1, 0)}

    def test_the_private_skip_days_parameter_is_gone(self):
        from ait.backtesting.engine import Backtester
        params = inspect.signature(Backtester.__init__).parameters
        assert "earnings_skip_days" not in params
        assert {"earnings_buffer_days_before", "earnings_buffer_days_after"} <= set(params)


# --------------------------------------------------------------------------- #
# 4. Phantom-key guard: a config field nothing reads is a lie in config.yaml
# --------------------------------------------------------------------------- #

# Fields that look tunable and are not (W69 fixed dashboard.* and learning.*; these
# remain). Each is REGISTERED in PLAN.md W70 and decided later: do not "fix" by editing
# this dict, either wire the key or delete it.
EXEMPT: dict[str, str] = {
    "trading.trading_hours_only":
        "RTH is enforced unconditionally (scheduler is_market_open + executor refusal unless "
        "env AIT_ALLOW_AFTER_HOURS=1); this yaml flag is a second, unconnected switch.",
    "options.delta_range":
        "long-option strike targeting (long_options.py uses a literal 0.40); live is IC-only.",
    "meta_label.retrain_with_primary":
        "the live trainer has no meta-labeler retrain; the gate itself is off (enabled: false).",
    "exit.auto_hedge":
        "no hedging code exists anywhere in src/; docs/archive/AUDIT_R12.md's 'one flip from live' is false.",
    # Read, but only inside settings.py through a derived property, which this scan skips:
    "trading.first_hour_end_et": "read via TradingConfig.first_hour_end_hour (property)",
    "trading.midday_end_et": "read via TradingConfig.midday_end_hour (property)",
    "risk.weekend_gap_cutoff_et": "read via RiskConfig.weekend_gap_cutoff_minutes (property)",
}

# helper aliases whose second positional string argument is a field of a fixed section
HELPER_SECTION = {"_bt": "backtest"}


def _scan_sources():
    paths = [p for p in (list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                         + [ROOT / n for n in ("run_backtest.py", "run_optimizer.py",
                                               "run_orchestrator.py", "status.py",
                                               "status_server.py", "web_logs.py")])
             if p.exists() and not (p.name == "settings.py" and "config" in p.parts)]
    attrs, strs, pairs, chains = set(), set(), set(), set()
    for p in paths:
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                attrs.add(n.attr)
                if isinstance(n.value, ast.Attribute):
                    chains.add((n.value.attr, n.attr))
            elif isinstance(n, ast.Call):
                consts = [a.value for a in n.args
                          if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                strs.update(consts)
                for a, b in zip(consts, consts[1:]):
                    pairs.add((a, b))
                fn = getattr(n.func, "id", "")
                if fn in HELPER_SECTION and consts:
                    pairs.add((HELPER_SECTION[fn], consts[-1]))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and False:
                pass
    return attrs, strs, pairs, chains


def _config_fields():
    out = {}
    for section, finfo in Settings.model_fields.items():
        model = finfo.annotation
        fields = getattr(model, "model_fields", None)
        if not fields or section in ("ibkr", "api_keys"):
            continue
        for f in fields:
            out[(section, f)] = None
        # nested models (learning.analyzer.*) are checked as their own section
        for f, info in fields.items():
            sub = getattr(info.annotation, "model_fields", None)
            if sub:
                for g in sub:
                    out[(f"{section}.{f}", g)] = None
                out.pop((section, f), None)
    return out


# Fields whose NAME collides with a WalkForwardConfig / engine parameter. A bare
# attribute read of `.spread_base` (on WalkForwardConfig) once made the BacktestConfig
# key look "read" for months while nothing read it. These demand a section-qualified
# read: resolve_config_value(..., "backtest", "spread_base", ...), X.backtest.spread_base,
# or the engine's `_bt(v, "spread_base")` helper.
STRICT = {("backtest", "spread_base"), ("backtest", "spread_iv_sensitivity"),
          ("backtest", "spread_dte_sensitivity"), ("backtest", "spread_cap")}


class TestNoPhantomConfigKeys:
    def test_every_config_field_has_a_reader_or_a_registered_reason(self):
        attrs, strs, pairs, chains = _scan_sources()
        homes = set(CONFIG_BACKED.values())            # read through the env contract
        unread = []
        for (section, f) in _config_fields():
            dotted = f"{section}.{f}"
            if dotted in EXEMPT:
                continue
            leaf = section.split(".")[-1]
            qualified = ((section, f) in pairs or (leaf, f) in pairs
                         or (section, f) in chains or (leaf, f) in chains
                         or (section, f) in homes)
            if (leaf, f) in STRICT:
                ok = qualified
            else:
                ok = qualified or f in attrs or f in strs
            if not ok:
                unread.append(dotted)
        assert not unread, (
            "config fields that NOTHING reads (a tunable that does nothing is a lie): "
            + ", ".join(sorted(unread)) + ". Wire each (prefer resolve_config_value("
            "..., '<section>', '<field>', ...)) or delete it; do not add it to EXEMPT "
            "without registering it in PLAN.md.")

    def test_the_spread_keys_are_read_through_a_section_qualified_call(self):
        """The exact incident: these four were read by nothing for as long as the engine
        and the walk-forward carried their own literals."""
        _a, _s, pairs, chains = _scan_sources()
        for f in ("spread_base", "spread_iv_sensitivity", "spread_dte_sensitivity",
                  "spread_cap"):
            assert ("backtest", f) in pairs or ("backtest", f) in chains, f

    def test_exempt_entries_are_still_phantoms(self):
        """An EXEMPT entry that gained a reader must be removed: the dict never rots."""
        attrs, strs, pairs, chains = _scan_sources()
        stale = []
        for dotted, why in EXEMPT.items():
            if "property" in why:
                continue                     # read inside settings.py by design
            section, f = dotted.rsplit(".", 1)
            if (section, f) in pairs or (section, f) in chains or f in attrs or f in strs:
                stale.append(dotted)
        assert not stale, f"now read; drop from EXEMPT: {stale}"
