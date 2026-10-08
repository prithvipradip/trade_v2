"""W63 — ONE capital number (config.yaml `account.capital`), shared-pool walk-forward.

Capital used to live in six places that disagreed: .env AIT_SIMULATED_CAPITAL
(10000, live only), backtest.initial_capital (100k, ablations only), two
`--capital 50000` CLI defaults, the supervisor's hardcoded weekly-health
`--capital 50000`, WalkForwardConfig 100k and the dashboard's 50k. Walk-forward
also gave every symbol its OWN full-capital sleeve (BT-H3), so three symbols
traded as if the account held 3x its money.

These tests drive the real resolvers, the real AccountManager refresh path, the
real allocator and a real walk-forward window.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

import ait.config.runtime_env as RE
from ait.broker.account import AccountManager
from ait.backtesting.pool import allocate_pool

SRC = Path(__file__).resolve().parent.parent / "src" / "ait"
ROOT = SRC.parent.parent


def _yaml(monkeypatch, capital=5000, simulate=True):
    monkeypatch.setattr(RE, "_yaml_cache", {
        "account": {"capital": capital, "simulate_live_capital": simulate}})


# --------------------------------------------------------------------------
# Resolution precedence: env override > config (flag on) > off
# --------------------------------------------------------------------------
class TestSimulatedCapitalResolution:
    def test_config_drives_it_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 5000, True)
        assert RE.simulated_capital() == 5000.0
        assert RE.simulated_capital_source() == "config"

    def test_flag_off_means_no_simulation(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 5000, False)
        assert RE.simulated_capital() is None
        assert RE.simulated_capital_source() == "off"

    def test_env_overrides_config(self, monkeypatch):
        _yaml(monkeypatch, 5000, True)
        monkeypatch.setenv("AIT_SIMULATED_CAPITAL", "12345")
        assert RE.simulated_capital() == 12345.0
        assert RE.simulated_capital_source() == "env"

    @pytest.mark.parametrize("raw", ["0", "0.0", "-5", "garbage"])
    def test_env_zero_or_junk_means_off_even_if_config_says_on(self, monkeypatch, raw):
        _yaml(monkeypatch, 5000, True)
        monkeypatch.setenv("AIT_SIMULATED_CAPITAL", raw)
        assert RE.simulated_capital() is None

    def test_research_capital_reads_config(self, monkeypatch):
        _yaml(monkeypatch, 7777, False)  # the simulate flag must not affect research
        assert RE.research_capital() == 7777.0

    def test_research_capital_survives_unreadable_config(self, monkeypatch):
        monkeypatch.setattr(RE, "_yaml_cache", {})
        assert RE.research_capital() == 5000.0

    def test_the_real_config_yaml_says_5000(self):
        """Pins the operator decision through the real file + real model."""
        from ait.config.settings import load_settings
        s = load_settings(ROOT / "config.yaml")
        assert s.account.capital == 5000.0
        assert s.account.simulate_live_capital is True
        assert not hasattr(s.backtest, "initial_capital")

    def test_a_stale_backtest_initial_capital_key_is_a_startup_error(self):
        from pydantic import ValidationError
        from ait.config.settings import BacktestConfig
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=100_000)


# --------------------------------------------------------------------------
# Live: the REAL AccountManager scales from config; W29's real NLV survives
# --------------------------------------------------------------------------
REAL_NLV = 201475.35
_BROKER = {"NetLiquidation": str(REAL_NLV), "BuyingPower": "402950.70",
           "AvailableFunds": "180000", "ExcessLiquidity": "175000",
           "CashBalance": "50000"}


async def _snap():
    client = MagicMock()
    client.get_account_values = AsyncMock(return_value=dict(_BROKER))
    return await AccountManager(client).get_snapshot()


class TestLiveScalingIsConfigDriven:
    async def test_config_scales_the_risk_stack_view(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 5000, True)
        snap = await _snap()
        assert snap.net_liquidation == pytest.approx(5000.0)
        assert snap.buying_power < float(_BROKER["BuyingPower"])

    async def test_w29_unscaled_nlv_still_captured(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 5000, True)
        snap = await _snap()
        assert snap.real_net_liquidation == pytest.approx(REAL_NLV)

    async def test_flag_off_leaves_the_account_alone(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 5000, False)
        assert (await _snap()).net_liquidation == pytest.approx(REAL_NLV)

    async def test_a_capital_above_real_nlv_never_inflates(self, monkeypatch):
        monkeypatch.delenv("AIT_SIMULATED_CAPITAL", raising=False)
        _yaml(monkeypatch, 9_999_999, True)
        assert (await _snap()).net_liquidation == pytest.approx(REAL_NLV)


# --------------------------------------------------------------------------
# allocate_pool: one account, symbols compete
# --------------------------------------------------------------------------
def _ic(symbol, entry, exit_, pnl, contracts=1, per_share=1.0):
    return {"symbol": symbol, "entry_date": entry, "exit_date": exit_,
            "contracts": contracts, "pnl": pnl,
            "max_loss_per_share": per_share,
            "max_loss": per_share * 100 * contracts,
            "credit": 30.0 * contracts, "entry_commission": 2.0 * contracts}


class TestAllocatePool:
    KW = dict(position_size_pct=0.05, max_portfolio_risk_pct=0.20,
              max_open_positions=10)

    def test_unaffordable_trade_is_rejected_and_counted(self):
        # $5k * 5% = $250 budget; a $300 max-loss condor cannot be sized.
        r = allocate_pool([_ic("SPY", "2026-01-05", "2026-01-20", 40, per_share=3.0)],
                          5000, **self.KW)
        assert r.accepted == []
        assert r.rejections["unaffordable"] == 1

    def test_affordable_trade_passes_unchanged(self):
        r = allocate_pool([_ic("SPY", "2026-01-05", "2026-01-20", 40, per_share=2.0)],
                          5000, **self.KW)
        assert len(r.accepted) == 1 and r.accepted[0]["contracts"] == 1
        assert r.final_equity == pytest.approx(5040.0)

    def test_engine_contracts_are_capped_to_the_pool_budget(self):
        # Engine chose 4 contracts ($100 risk each); pool budget is $250 -> 2.
        t = _ic("SPY", "2026-01-05", "2026-01-20", 80, contracts=4)
        r = allocate_pool([t], 5000, **self.KW)
        got = r.accepted[0]
        assert got["contracts"] == 2 and got["pool_contracts_engine"] == 4
        assert got["pnl"] == pytest.approx(40.0)          # scaled 2/4
        assert got["max_loss"] == pytest.approx(200.0)
        assert got["entry_commission"] == pytest.approx(4.0)

    def test_symbols_compete_for_one_aggregate_risk_cap(self):
        # 20% of $5k = $1000 aggregate. Five overlapping $250-risk condors on
        # three different symbols: only four fit; the fifth is portfolio_risk.
        trades = [_ic(s, "2026-01-05", "2026-01-20", 10, contracts=1, per_share=2.5)
                  for s in ("SPY", "QQQ", "IWM", "SPY", "QQQ")]
        r = allocate_pool(trades, 5000, **self.KW)
        assert len(r.accepted) == 4
        assert r.rejections["portfolio_risk"] == 1

    def test_a_sleeve_model_would_have_taken_all_five(self):
        """Negative control: independent full-capital sleeves (the old BT-H3
        behaviour) each pass the per-symbol gate -- that is the bug."""
        per_symbol = [allocate_pool([_ic(s, "2026-01-05", "2026-01-20", 10,
                                         per_share=2.5)], 5000, **self.KW)
                      for s in ("SPY", "QQQ", "IWM", "SPY", "QQQ")]
        assert sum(len(r.accepted) for r in per_symbol) == 5

    def test_exits_free_the_risk_budget(self):
        a = _ic("SPY", "2026-01-05", "2026-01-06", 10, per_share=2.5)
        b = _ic("QQQ", "2026-01-07", "2026-01-20", 10, per_share=2.5)
        r = allocate_pool([a, b], 5000, position_size_pct=0.05,
                          max_portfolio_risk_pct=0.06, max_open_positions=10)
        assert len(r.accepted) == 2          # b enters after a closed

    def test_same_day_exit_does_not_free_that_days_entries(self):
        a = _ic("SPY", "2026-01-05", "2026-01-05", 10, per_share=2.5)
        b = _ic("QQQ", "2026-01-05", "2026-01-20", 10, per_share=2.5)
        r = allocate_pool([a, b], 5000, position_size_pct=0.05,
                          max_portfolio_risk_pct=0.06, max_open_positions=10)
        assert len(r.accepted) == 1 and r.rejections["portfolio_risk"] == 1

    def test_realised_pnl_compounds_the_pool(self):
        a = _ic("SPY", "2026-01-05", "2026-01-06", 500, per_share=2.0)
        b = _ic("QQQ", "2026-02-01", "2026-02-10", 10, per_share=2.0)
        r = allocate_pool([a, b], 5000, **self.KW)
        assert r.final_equity == pytest.approx(5510.0)

    def test_open_position_cap(self):
        trades = [_ic(s, "2026-01-05", "2026-01-20", 1, per_share=0.1)
                  for s in ("A", "B", "C")]
        r = allocate_pool(trades, 5000, position_size_pct=0.05,
                          max_portfolio_risk_pct=0.50, max_open_positions=2)
        assert len(r.accepted) == 2 and r.rejections["max_open_positions"] == 1

    def test_input_order_across_symbols_does_not_matter(self):
        trades = [_ic("QQQ", "2026-01-07", "2026-01-20", 10, per_share=2.5),
                  _ic("SPY", "2026-01-05", "2026-01-06", 10, per_share=2.5)]
        r = allocate_pool(trades, 5000, position_size_pct=0.05,
                          max_portfolio_risk_pct=0.06, max_open_positions=10)
        assert len(r.accepted) == 2
        assert [t["symbol"] for t in r.accepted] == ["SPY", "QQQ"]

    def test_private_size_key_is_honoured_then_stripped(self):
        t = _ic("SPY", "2026-01-05", "2026-01-20", 40, contracts=4, per_share=1.0)
        t["_pool_size_pct"] = 0.01             # $50 budget -> 0 contracts
        assert allocate_pool([t], 5000, **self.KW).rejections["unaffordable"] == 1
        t["_pool_size_pct"] = 0.05
        out = allocate_pool([t], 5000, **self.KW).accepted[0]
        assert "_pool_size_pct" not in out and out["contracts"] == 2

    def test_input_trades_are_never_mutated(self):
        t = _ic("SPY", "2026-01-05", "2026-01-20", 80, contracts=4)
        before = dict(t)
        allocate_pool([t], 5000, **self.KW)
        assert t == before


# --------------------------------------------------------------------------
# Walk-forward: the window's capital base is the pool, whatever the symbol count
# --------------------------------------------------------------------------
def _ohlcv(days=500, start=100.0):
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    close = start * np.cumprod(1 + np.random.normal(0.0005, 0.015, days))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, days)}, index=dates)


class TestWalkForwardUsesOnePool:
    def _run(self, monkeypatch, symbols, trade):
        from datetime import date
        from ait.backtesting.engine import Backtester
        from ait.backtesting.result import BacktestResult
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

        monkeypatch.setattr(Backtester, "run",
                            lambda self_bt: BacktestResult(trades=[dict(trade)]))
        cfg = WalkForwardConfig(train_days=300, test_days=50, gap_days=5,
                                initial_capital=5000.0)
        bt = WalkForwardBacktester(symbols, ["iron_condor"], config=cfg)
        data = {s: _ohlcv(500) for s in symbols}
        wr, _, _ = bt._run_single_window(
            window_id=1, train_start=date(2023, 1, 1), train_end=date(2023, 10, 28),
            test_start=date(2023, 11, 2), test_end=date(2023, 12, 22),
            data=data, vix_full=pd.DataFrame(), learner=None)
        return wr

    def test_three_symbols_one_5k_pool(self, monkeypatch):
        trade = _ic("X", "2023-11-06", "2023-11-20", 30, per_share=1.0)
        wr = self._run(monkeypatch, ["SPY", "QQQ", "IWM"], trade)
        # Pre-W63: 5_000 x 3 sleeves = 15_000 "capital" for the window.
        assert wr.backtest_result.initial_capital == pytest.approx(5000.0)
        assert wr.backtest_result.final_capital == pytest.approx(
            5000.0 + sum(t["pnl"] for t in wr.backtest_result.trades))

    def test_rejections_are_surfaced_on_the_window(self, monkeypatch):
        # $1k aggregate cap, $400 per condor, three symbols overlapping.
        trade = _ic("X", "2023-11-06", "2023-11-20", 30, per_share=4.0)
        wr = self._run(monkeypatch, ["SPY", "QQQ", "IWM"], trade)
        assert sum(wr.pool_rejections.values()) >= 1
        assert len(wr.backtest_result.trades) + sum(wr.pool_rejections.values()) == 3


def test_walkforward_config_default_is_the_configured_pool(monkeypatch):
    from ait.backtesting.walkforward import WalkForwardConfig
    _yaml(monkeypatch, 5000, False)
    assert WalkForwardConfig().initial_capital == 5000.0
    assert WalkForwardConfig(initial_capital=123.0).initial_capital == 123.0


# --------------------------------------------------------------------------
# Class guards: the NEXT stray copy of this fact fails here
# --------------------------------------------------------------------------
def _py_files():
    return [p for p in SRC.rglob("*.py") if "deprecated" not in p.parts]


class TestNoSecondHomeForCapital:
    def test_no_raw_read_of_simulated_capital_outside_runtime_env(self):
        offenders = []
        for p in _py_files():
            if p.name == "runtime_env.py":
                continue
            for node in ast.walk(ast.parse(p.read_text())):
                if (isinstance(node, ast.Constant)
                        and node.value == "AIT_SIMULATED_CAPITAL"):
                    offenders.append(f"{p.relative_to(SRC)}:{node.lineno}")
        assert not offenders, f"raw AIT_SIMULATED_CAPITAL read: {offenders}"

    def test_supervisor_passes_no_capital_literal_to_run_backtest(self):
        src = (SRC / "orchestration" / "master.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and node.value == "--capital":
                pytest.fail(f"master.py:{node.lineno} hardcodes --capital")

    @pytest.mark.parametrize("script", ["run_backtest.py", "run_optimizer.py"])
    def test_cli_capital_defaults_to_config_not_a_literal(self, script):
        tree = ast.parse((ROOT / script).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                    and node.args and getattr(node.args[0], "value", None) == "--capital"):
                default = next(k.value for k in node.keywords if k.arg == "default")
                assert isinstance(default, ast.Constant) and default.value is None

    def test_no_numeric_initial_capital_literal_in_research_entry_points(self):
        """walkforward.py's config default must stay None (resolved at runtime)."""
        src = (SRC / "backtesting" / "walkforward.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.ClassDef) and node.name == "WalkForwardConfig"):
                for st in node.body:
                    if (isinstance(st, ast.AnnAssign)
                            and getattr(st.target, "id", "") == "initial_capital"):
                        assert isinstance(st.value, ast.Constant) and st.value.value is None
                        return
        pytest.fail("WalkForwardConfig.initial_capital not found")
