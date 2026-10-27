"""W64 — retune the live risk knobs at the $5k account, give research live's construction.

At $5k the startup coherence check fired CAP-GRID INCOHERENT: the 1.33% budget ($66)
could not carry ANY condor (smallest emitting budget ~$136) and the 2% breaker ($100)
sat under a hardcoded 1.5x rule. Separately research funded 5% per trade (live 1.33%),
built the full wing_k x expected-move wing with no budget cap (live caps it to the
budget, $2-wide at this size) and traded >1 contract (live: 1).

These tests drive the real settings, the real orchestrator self-test, the real builder
and the real engine.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.config.settings import (
    PositionConfig, RiskConfig, Settings, load_settings, resolve_max_contracts_per_trade,
    resolve_position_size_pct,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ait"


@pytest.fixture(scope="module")
def real_settings():
    return load_settings(str(ROOT / "config.yaml"))


# --------------------------------------------------------------------------
# The retuned values, through the real file + real model
# --------------------------------------------------------------------------
class TestRetunedConfig:
    def test_live_values(self, real_settings):
        s = real_settings
        assert s.risk.max_position_risk_pct == pytest.approx(0.05)   # W108: 0.20 (W100) -> 0.05
        assert s.risk.max_daily_loss_pct == pytest.approx(0.045)
        assert s.risk.breaker_to_max_loss_ratio == pytest.approx(1.0)

    def test_research_values(self):
        # W68: budget_aware_wings is a research-only knob; it lives in the research overlay
        # (config_research.yaml), not in the live config.yaml. W108: backtest.position_size_pct is
        # retired: research sizes from risk.max_position_risk_pct, the same budget live funds.
        from pathlib import Path
        from ait.config.research import load_research_settings
        s = load_research_settings(str(Path(__file__).resolve().parent.parent / "config.yaml"))
        assert not hasattr(s.backtest, "position_size_pct")
        assert s.backtest.budget_aware_wings is True

    def test_code_defaults_match_config_so_bare_models_do_not_skew(self, real_settings):
        assert RiskConfig().max_position_risk_pct == real_settings.risk.max_position_risk_pct
        assert RiskConfig().max_daily_loss_pct == real_settings.risk.max_daily_loss_pct

    def test_credit_floor_unchanged(self):
        from ait.config.runtime_env import contract_float
        assert contract_float("AIT_IC_MIN_CREDIT") == pytest.approx(0.70)


# --------------------------------------------------------------------------
# The links between the per-trade budget and the aggregate / per-symbol caps
# (W108: the sizer's own cap, positions.max_position_pct, is retired; one budget remains)
# --------------------------------------------------------------------------
class TestPerTradeKnobsAreLinked:
    def _with(self, real_settings, risk_pct=None, portfolio_pct=None):
        d = real_settings.model_dump(exclude={"ibkr", "api_keys"})
        if risk_pct is not None:
            d["risk"]["max_position_risk_pct"] = risk_pct
        if portfolio_pct is not None:
            d["positions"]["max_portfolio_risk_pct"] = portfolio_pct
        return Settings(**d)

    def test_budget_above_the_aggregate_cap_is_a_startup_error(self, real_settings):
        with pytest.raises(ValueError, match="max_portfolio_risk_pct"):
            self._with(real_settings, risk_pct=0.25)

    def test_equal_is_allowed(self, real_settings):
        self._with(real_settings, risk_pct=0.05, portfolio_pct=0.05)

    def test_the_retired_sizer_cap_is_not_a_key_any_more(self, real_settings):
        d = real_settings.model_dump(exclude={"ibkr", "api_keys"})
        d["positions"]["max_position_pct"] = 0.20
        with pytest.raises(ValueError):          # extra="forbid": a local config naming it is an error
            Settings(**d)


# --------------------------------------------------------------------------
# The coherence self-test, through the REAL orchestrator method and builder
# --------------------------------------------------------------------------
async def _coherence_alerts(settings, nlv: float) -> list[str]:
    from ait.bot.orchestrator import TradingOrchestrator

    o = object.__new__(TradingOrchestrator)
    o._settings = settings
    sent: list[str] = []

    async def _notify(msg, *a, **k):
        sent.append(msg)

    async def _noop(*a, **k):
        return None

    o._send_notification = _notify
    o._note_loop_error = _noop
    ok = await o._launch_size_coherence_check(nlv)
    assert ok is True
    return sent


class TestCoherenceAtFiveK:
    async def test_retuned_values_are_coherent_at_5k(self, real_settings):
        assert await _coherence_alerts(real_settings, 5000.0) == []

    async def test_negative_control_the_old_values_were_not(self, real_settings):
        s = real_settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.0133
        s.risk.max_daily_loss_pct = 0.02
        alerts = await _coherence_alerts(s, 5000.0)
        assert len(alerts) == 1 and "ZERO trades" in alerts[0]

    async def test_ratio_is_read_from_config_not_a_literal(self, real_settings):
        strict = real_settings.model_copy(deep=True)
        strict.risk.breaker_to_max_loss_ratio = 1.5
        alerts = await _coherence_alerts(strict, 5000.0)
        assert len(alerts) == 1
        assert "1.5x (risk.breaker_to_max_loss_ratio)" in alerts[0]

    async def test_ratio_one_is_quiet_at_ten_k_too(self, real_settings):
        assert await _coherence_alerts(real_settings, 10_000.0) == []


# --------------------------------------------------------------------------
# Resolvers: explicit > study value > live budget
# --------------------------------------------------------------------------
class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestResolvers:
    def test_position_size_precedence(self, real_settings):
        assert resolve_position_size_pct(0.07, real_settings) == pytest.approx(0.07)
        assert resolve_position_size_pct(None, real_settings) == pytest.approx(0.05)
        # W108: backtest.position_size_pct is retired; the loaded risk budget is the only default
        study = _Stub(backtest=_Stub(position_size_pct=0.10),
                      risk=_Stub(max_position_risk_pct=0.03))
        assert resolve_position_size_pct(None, study) == pytest.approx(0.03)

    def test_position_size_survives_missing_settings(self):
        assert resolve_position_size_pct(None, None) == pytest.approx(
            RiskConfig().max_position_risk_pct)
        assert resolve_position_size_pct(None, _Stub()) == pytest.approx(
            RiskConfig().max_position_risk_pct)

    def test_max_contracts(self, real_settings):
        assert resolve_max_contracts_per_trade(None, real_settings) == 1
        assert resolve_max_contracts_per_trade(4, real_settings) == 4
        assert resolve_max_contracts_per_trade(None, None) == PositionConfig().max_contracts_per_trade


class TestEntryPointsInheritLive:
    def test_walkforward_config(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        c = WalkForwardConfig()
        assert c.position_size_pct == pytest.approx(0.05)
        assert c.budget_aware_wings is True
        assert c.max_contracts_per_trade == 1
        e = WalkForwardConfig(position_size_pct=0.03, budget_aware_wings=False,
                              max_contracts_per_trade=3)
        assert (e.position_size_pct, e.budget_aware_wings, e.max_contracts_per_trade) == (
            0.03, False, 3)

    def test_optimizer(self):
        from ait.optimization.optimizer import StrategyOptimizer
        o = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1)
        assert o._position_size_pct == pytest.approx(0.05)
        assert o._budget_aware_wings is True and o._max_contracts_per_trade == 1
        o2 = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1,
                               position_size_pct=0.03, budget_aware_wings=False,
                               max_contracts_per_trade=2)
        assert (o2._position_size_pct, o2._budget_aware_wings,
                o2._max_contracts_per_trade) == (0.03, False, 2)

    def test_optimizer_trials_carry_the_parity_switches(self):
        from ait.optimization.optimizer import StrategyOptimizer
        src = (SRC / "optimization" / "optimizer.py").read_text()
        assert '"budget_aware_wings"' in src and '"max_contracts_per_trade"' in src
        # and they are real Backtester parameters
        import inspect
        from ait.backtesting.engine import Backtester
        sig = inspect.signature(Backtester.__init__).parameters
        assert "budget_aware_wings" in sig and "max_contracts_per_trade" in sig
        assert StrategyOptimizer  # imported for the check above


# --------------------------------------------------------------------------
# Engine: live's budget-aware wing and contract cap
# --------------------------------------------------------------------------
def _df():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=400, freq="B")
    close = 277 * np.cumprod(1 + rng.normal(0.0003, 0.008, 400))
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                         "Close": close, "Volume": 5_000_000}, index=idx)


def _build(capital, pct, aware, iv=0.14, cap=None):
    from ait.backtesting.engine import Backtester
    bt = Backtester(data=_df(), strategies=["iron_condor"], initial_capital=capital,
                    position_size_pct=pct, budget_aware_wings=aware,
                    max_contracts_per_trade=cap, wing_floor_dollars=2.0)
    return bt, bt._build_credit_position(
        "iron_condor", 277.0, iv, 21 / 365, 0.05, 21, date(2026, 3, 2), float(capital))


class TestEngineBudgetAwareWings:
    def test_small_account_now_builds_the_narrow_condor_live_builds(self):
        _, pos = _build(5000, 0.05, aware=True)          # $250 budget, the shipped value
        assert pos is not None
        wing = pos["long_call_strike"] - pos["short_call_strike"]
        assert wing == pytest.approx(3.0)                # $250 / 90 = 2.78 -> $3
        assert pos["contracts"] == 1
        assert pos["max_loss_per_share"] * 100 <= 5000 * 0.05   # within the budget

    def test_smaller_budget_builds_the_two_wide_condor(self):
        _, pos = _build(5000, 0.03, aware=True)          # $150 -> $1.67 -> $2
        assert pos is not None
        assert pos["long_call_strike"] - pos["short_call_strike"] == pytest.approx(2.0)

    def test_negative_control_without_it_the_same_account_builds_nothing(self):
        _, pos = _build(5000, 0.05, aware=False)
        assert pos is None   # full wing_k x EM wing is far over a $250 budget

    def test_wing_equals_the_one_authority_when_the_cap_binds(self):
        from ait.strategies.iron_condor import IronCondor
        for pct in (0.03, 0.05):
            _, pos = _build(5000, pct, aware=True)
            if pos is None:
                continue
            wing = pos["long_call_strike"] - pos["short_call_strike"]
            assert wing == pytest.approx(
                float(max(1.0, round(IronCondor.affordable_width(5000 * pct)))))

    def test_wing_grows_with_the_budget(self):
        _, a = _build(5000, 0.03, aware=True)
        _, b = _build(5000, 0.05, aware=True)
        assert a is not None and b is not None
        assert (b["long_call_strike"] - b["short_call_strike"]) > (
            a["long_call_strike"] - a["short_call_strike"])

    def test_budget_gate_uses_mid_credit_but_the_trade_records_friction(self):
        """Live checks the budget on the MID credit; the engine used to deduct
        half-spread friction first and refuse. Budget $140: the friction-
        inclusive max loss ($144) exceeds it, the mid-credit one does not."""
        _, pos = _build(5000, 0.028, aware=True, iv=0.16)
        assert pos is not None                           # mid-credit gate passes
        assert pos["max_loss_per_share"] * 100 > 5000 * 0.028   # friction retained in the record
        assert pos["contracts"] == 1

    def test_big_account_is_not_capped(self):
        _, pos = _build(100_000, 0.03, aware=True)
        assert pos is not None
        assert pos["long_call_strike"] - pos["short_call_strike"] > 10  # vol-scaled wing

    def test_default_engine_is_unchanged(self):
        """Bare Backtester keeps its explicit harness defaults (R20b)."""
        from ait.backtesting.engine import Backtester
        bt = Backtester(data=_df(), strategies=["iron_condor"])
        assert bt._budget_aware_wings is False and bt._max_contracts_per_trade is None
        assert bt._position_size_pct == pytest.approx(0.05)


class TestEngineContractCap:
    def _positions(self, cap):
        from ait.backtesting.engine import Backtester
        from ait.strategies.base import SignalDirection
        bt = Backtester(data=_df(), strategies=["iron_condor"], initial_capital=100_000,
                        max_contracts_per_trade=cap, wing_floor_dollars=2.0)
        bt._get_iv = lambda *a, **k: 0.30            # clear the credit-entry IV floor
        bt._build_credit_position = lambda *a, **k: {"contracts": 4, "strategy": "iron_condor"}
        df = _df()
        return bt._build_position("iron_condor", SignalDirection.NEUTRAL, df.iloc[-1], df,
                                  date(2026, 3, 2), 100_000.0)

    def test_cap_applies(self):
        assert self._positions(1)["contracts"] == 1
        assert self._positions(2)["contracts"] == 2

    def test_none_is_uncapped(self):
        assert self._positions(None)["contracts"] == 4


# --------------------------------------------------------------------------
# run_backtest's parity manifest carries the new rows and reports parity
# --------------------------------------------------------------------------
def _load_run_backtest(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    spec = importlib.util.spec_from_file_location("rb_w64", ROOT / "run_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rb_w64"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestParityManifest:
    def test_rows_present_and_in_parity(self, monkeypatch):
        rb = _load_run_backtest(monkeypatch)
        m = rb.build_parity_manifest(rb.parse_args())
        for key in ("per_trade_budget_pct", "budget_aware_wings", "max_contracts_per_trade"):
            assert key in m["live"] and key in m["backtest"]
            assert m["live"][key] == m["backtest"][key]
        assert not [w for w in m["parity_warnings"]
                    if w.startswith(("per_trade_budget_pct", "budget_aware_wings",
                                     "max_contracts_per_trade"))]


# --------------------------------------------------------------------------
# Class guards: the NEXT stray copy of these facts fails here
# --------------------------------------------------------------------------
class TestClassGuards:
    def test_no_numeric_breaker_multiple_in_the_coherence_check(self):
        tree = ast.parse((SRC / "bot" / "orchestrator.py").read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_launch_size_coherence_check"):
                bad = [n.lineno for n in ast.walk(node)
                       if isinstance(n, ast.Constant) and n.value == 1.5]
                assert not bad, f"hardcoded 1.5 multiple at lines {bad}"
                return
        pytest.fail("_launch_size_coherence_check not found")

    @pytest.mark.parametrize("path,cls", [
        ("backtesting/walkforward.py", "WalkForwardConfig"),
    ])
    def test_no_literal_position_size_default_in_research_entry_points(self, path, cls):
        tree = ast.parse((SRC / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for st in node.body:
                    if (isinstance(st, ast.AnnAssign)
                            and getattr(st.target, "id", "") == "position_size_pct"):
                        assert isinstance(st.value, ast.Constant) and st.value.value is None
                        return
        pytest.fail(f"{cls}.position_size_pct not found")

    def test_optimizer_position_size_default_is_none(self):
        tree = ast.parse((SRC / "optimization" / "optimizer.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                args = node.args
                names = [a.arg for a in args.args + args.kwonlyargs]
                if "position_size_pct" in names:
                    defaults = dict(zip(reversed(names), reversed(
                        args.defaults + args.kw_defaults)))
                    d = defaults["position_size_pct"]
                    assert isinstance(d, ast.Constant) and d.value is None
                    return
        pytest.fail("StrategyOptimizer.__init__ position_size_pct not found")
