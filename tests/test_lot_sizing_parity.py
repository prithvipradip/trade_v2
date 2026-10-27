"""W108: ONE lot-sizing rule for live and research.

Before this wave live (``PositionSizer``: scalers x 0.4 condor factor, floor to 1, a per-account scale cap)
and research (ten inline ``int(capital * pct / risk)`` sites, a "1 lot if it costs <= 25% of capital"
fallback, then the pool's own ``floor``) counted lots with different code, so a research result about a lot
count could not be reproduced live. ``ait.risk.lot_sizing.lots_for_budget`` is now the only arithmetic:

* viability is decided by the UNSCALED budget (``equity * risk.max_position_risk_pct``); one lot above it
  is refused (0 lots);
* the count is ``floor(budget * scaler / risk)``, never below 1 once viable;
* a scaler only throttles the count above 1 lot (``position_sizing.scaled_strategies``; empty today);
* ``max_lots`` caps the result.

These tests drive the SAME table through the shared function, the live sizer, the engine and the pool, and
pin the pool's three new book-level gates (live gates 4b, 5 and 6c) and the registry of its rejections.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from ait.backtesting.gate_parity import POOL_REJECT_REASONS
from ait.backtesting.pool import allocate_pool
from ait.config.settings import PositionConfig, PositionSizingConfig, RiskConfig, load_settings
from ait.risk.lot_sizing import lots_for_budget
from ait.risk.position_sizer import PositionSizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"

BUDGET_PCT = 0.05

# (equity, dollars one lot can lose) -> expected lots at a 5% budget, no scaler, no cap
TABLE = [
    (5_000, 200.0, 1),        # the 3-wide QQQ condor at the shipped budget ($250)
    (5_000, 250.0, 1),        # exactly the budget: viable
    (5_000, 99.0, 2),         # 2.53 lots: floor is 2, a round would say 3
    (5_000, 250.01, 0),       # a cent above: refused
    (5_000, 1_000.0, 0),      # the 11-wide that the 20% budget used to build
    (100_000, 4_999.0, 1),
    (100_000, 1_200.0, 4),
    (100_000, 500.0, 10),
    (100_000, 100.0, 50),
]


class TestTheSharedRule:
    @pytest.mark.parametrize("equity, risk, expected", TABLE)
    def test_the_table(self, equity, risk, expected):
        assert lots_for_budget(risk, equity, BUDGET_PCT) == expected

    @pytest.mark.parametrize("risk", [None, 0, -5.0, float("nan")])
    def test_an_unknown_risk_sizes_nothing(self, risk):
        assert lots_for_budget(risk, 5_000, BUDGET_PCT) == 0

    @pytest.mark.parametrize("equity, pct", [(0, 0.05), (-1, 0.05), (5_000, 0), (float("nan"), 0.05)])
    def test_no_budget_sizes_nothing(self, equity, pct):
        assert lots_for_budget(100.0, equity, pct) == 0

    def test_the_cap_binds(self):
        assert lots_for_budget(100.0, 100_000, BUDGET_PCT, max_lots=3) == 3
        assert lots_for_budget(100.0, 100_000, BUDGET_PCT, max_lots=1) == 1

    def test_a_scaler_throttles_above_one_lot_and_never_below_it(self):
        # unscaled: floor(5000/100) = 50 lots; a 0.5 scaler halves it
        assert lots_for_budget(100.0, 100_000, BUDGET_PCT, scaler=0.5) == 25
        # a 0.01 scaler would give 0.5 lots: still 1, because ONE lot fits the unscaled budget
        assert lots_for_budget(200.0, 5_000, BUDGET_PCT, scaler=0.01) == 1

    def test_a_scaler_never_makes_a_refused_lot_viable_or_a_viable_lot_refused(self):
        assert lots_for_budget(300.0, 5_000, BUDGET_PCT, scaler=5.0) == 0     # one lot > budget stays refused
        assert lots_for_budget(250.0, 5_000, BUDGET_PCT, scaler=0.0) == 1      # viable stays viable

    def test_the_default_scaler_is_one(self):
        assert lots_for_budget(100.0, 100_000, BUDGET_PCT) == lots_for_budget(
            100.0, 100_000, BUDGET_PCT, scaler=1.0)


def _live_sizer(scaled=(), max_lots=100):
    pc = PositionConfig(max_contracts_per_trade=max_lots)
    rc = RiskConfig(min_confidence=0.5, max_position_risk_pct=BUDGET_PCT)
    return PositionSizer(pc, rc, PositionSizingConfig(scaled_strategies=list(scaled)))


def _live_lots(sizer, equity, risk, strategy="iron_condor", confidence=0.6):
    return sizer.calculate(
        account_value=equity, option_price=risk / 100.0, confidence=confidence,
        implied_vol=0.19, strategy=strategy, underlying_price=0, iv_rank=55.0, vix=16.0).contracts


class TestLiveSizerUsesTheRule:
    @pytest.mark.parametrize("equity, risk, expected", TABLE)
    def test_same_lots_as_the_shared_function(self, equity, risk, expected):
        assert _live_lots(_live_sizer(), equity, risk) == expected

    def test_the_condor_factor_and_confidence_do_not_act_when_the_list_is_empty(self):
        """The 0.4 condor factor x a 0.6 confidence factor used to cut a $200 condor's budget to $60 and
        lean on the floor. With scalers off the budget alone decides: 100 lots at $100 each on $100k."""
        assert _live_lots(_live_sizer(), 100_000, 100.0) == 50

    def test_a_listed_strategy_is_throttled_by_its_multipliers(self):
        scaled = _live_lots(_live_sizer(scaled=["iron_condor"]), 100_000, 100.0)
        unscaled = _live_lots(_live_sizer(), 100_000, 100.0)
        assert 1 <= scaled < unscaled

    def test_a_listed_strategy_is_still_refused_only_when_one_lot_exceeds_the_unscaled_budget(self):
        sizer = _live_sizer(scaled=["iron_condor"])
        assert _live_lots(sizer, 5_000, 200.0) == 1       # 0.4 x 0.6 would give $60 of budget: floor of 1 holds
        assert _live_lots(sizer, 5_000, 300.0) == 0

    def test_another_strategy_is_not_scaled_by_a_list_that_names_one(self):
        sizer = _live_sizer(scaled=["iron_condor"])
        assert _live_lots(sizer, 100_000, 100.0, strategy="long_call") == 50

    def test_the_hard_cap_still_applies(self):
        assert _live_lots(_live_sizer(max_lots=3), 100_000, 100.0) == 3

    def test_the_budget_is_the_risk_gate_number_not_a_second_cap(self):
        sizer = _live_sizer()
        sizer._risk_config = RiskConfig(min_confidence=0.5, max_position_risk_pct=0.10)
        assert _live_lots(sizer, 5_000, 450.0) == 1          # $500 budget carries $450
        sizer._risk_config = RiskConfig(min_confidence=0.5, max_position_risk_pct=0.05)
        assert _live_lots(sizer, 5_000, 450.0) == 0


class TestEngineUsesTheRule:
    @pytest.mark.parametrize("equity, risk, expected", TABLE)
    def test_same_lots_as_the_shared_function(self, equity, risk, expected):
        from tests.test_w74_engine_realism import _bt
        bt = _bt(position_size_pct=BUDGET_PCT, initial_capital=float(equity))
        assert bt._lots(risk, float(equity)) == expected

    def test_no_inline_lot_arithmetic_is_left_in_the_engine(self):
        src = (SRC / "backtesting" / "engine.py").read_text()
        assert "int(capital *" not in src, "a builder still counts lots itself"
        assert "max_cost_fraction_of_capital" not in src, "the 25% one-lot fallback is back"
        assert src.count("self._lots(") >= 9


def _trade(symbol, entry, exit_, risk, *, expiry="2024-03-15", contracts=100, pnl=0.0):
    return {"symbol": symbol, "entry_date": entry, "exit_date": exit_, "expiry_date": expiry,
            "contracts": contracts, "max_loss_per_share": risk / 100.0, "pnl": pnl,
            "max_loss": risk * contracts, "credit": 0.0, "entry_commission": 0.0}


def _pool(trades, equity=5_000.0, **limits):
    base = dict(max_portfolio_risk_pct=1.0, max_open_positions=50)
    base.update(limits)
    return allocate_pool(trades, equity, BUDGET_PCT, **base)


class TestPoolUsesTheRule:
    @pytest.mark.parametrize("equity, risk, expected", TABLE)
    def test_same_lots_as_the_shared_function(self, equity, risk, expected):
        res = _pool([_trade("QQQ", "2024-01-02", "2024-01-10", risk)], equity=float(equity))
        if expected == 0:
            assert not res.accepted and res.rejections["unaffordable"] == 1
        else:
            assert res.accepted[0]["contracts"] == min(expected, 100)


class TestPoolBookLevelGates:
    def test_the_gates_are_off_unless_asked_for(self):
        trades = [_trade("QQQ", "2024-01-02", "2024-01-20", 200.0),
                  _trade("QQQ", "2024-01-03", "2024-01-20", 200.0)]
        assert len(_pool(trades).accepted) == 2

    def test_symbol_concentration_gate_6c(self):
        trades = [_trade("QQQ", "2024-01-02", "2024-01-20", 200.0, contracts=1),
                  _trade("QQQ", "2024-01-03", "2024-01-20", 200.0, contracts=1)]
        res = _pool(trades, max_symbol_concentration_pct=0.07)       # $350 cap: 200 + 200 > 350
        assert len(res.accepted) == 1 and res.rejections["symbol_concentration"] == 1
        # the cap is per symbol: a second symbol is unaffected
        other = [trades[0], _trade("XOM", "2024-01-03", "2024-01-20", 200.0, contracts=1)]
        assert len(_pool(other, max_symbol_concentration_pct=0.07).accepted) == 2

    def test_the_symbol_cap_is_inclusive_at_the_boundary(self):
        """Live 6c refuses only when exposure + new EXCEEDS the cap: exactly equal is allowed."""
        trades = [_trade("QQQ", "2024-01-02", "2024-01-20", 200.0, contracts=1),
                  _trade("QQQ", "2024-01-03", "2024-01-20", 200.0, contracts=1)]
        assert len(_pool(trades, max_symbol_concentration_pct=0.08).accepted) == 2    # $400 == 200 + 200

    def test_a_closed_position_frees_the_symbol(self):
        trades = [_trade("QQQ", "2024-01-02", "2024-01-05", 200.0, contracts=1),
                  _trade("QQQ", "2024-01-09", "2024-01-20", 200.0, contracts=1)]
        assert len(_pool(trades, max_symbol_concentration_pct=0.05).accepted) == 2

    def test_per_expiry_cap_gate_4b(self):
        syms = ["XOM", "KO", "PG", "JNJ"]          # in no SECTOR_GROUPS cluster
        trades = [_trade(s, f"2024-01-0{i + 2}", "2024-01-20", 100.0, contracts=1, expiry="2024-02-16")
                  for i, s in enumerate(syms)]
        res = _pool(trades, max_positions_per_expiry=3)
        assert len(res.accepted) == 3 and res.rejections["per_expiry_cap"] == 1
        other_expiry = trades[:3] + [_trade("JNJ", "2024-01-06", "2024-01-20", 100.0, contracts=1,
                                            expiry="2024-02-23")]
        assert len(_pool(other_expiry, max_positions_per_expiry=3).accepted) == 4

    def test_a_trade_with_no_expiry_is_skipped_tolerantly_like_live(self):
        trades = [dict(_trade(s, f"2024-01-0{i + 2}", "2024-01-20", 100.0, contracts=1), expiry_date=None)
                  for i, s in enumerate(["XOM", "KO", "PG", "JNJ"])]
        assert len(_pool(trades, max_positions_per_expiry=1).accepted) == 4

    def test_cluster_cap_gate_5(self):
        trades = [_trade("SPY", "2024-01-02", "2024-01-20", 100.0, contracts=1),
                  _trade("QQQ", "2024-01-03", "2024-01-20", 100.0, contracts=1),
                  _trade("IWM", "2024-01-04", "2024-01-20", 100.0, contracts=1),
                  _trade("XOM", "2024-01-05", "2024-01-20", 100.0, contracts=1)]
        res = _pool(trades, max_correlated_positions=2)
        assert [t["symbol"] for t in res.accepted] == ["SPY", "QQQ", "XOM"]
        assert res.rejections["correlation_cluster"] == 1

    def test_the_same_symbol_counts_toward_its_own_cluster(self):
        trades = [_trade("XOM", "2024-01-02", "2024-01-20", 100.0, contracts=1),
                  _trade("XOM", "2024-01-03", "2024-01-20", 100.0, contracts=1)]
        res = _pool(trades, max_correlated_positions=1)
        assert len(res.accepted) == 1 and res.rejections["correlation_cluster"] == 1

    def test_the_limits_come_from_the_loaded_settings(self):
        from ait.backtesting.walkforward import WalkForwardBacktester
        s = load_settings()
        limits = WalkForwardBacktester.__new__(WalkForwardBacktester)
        limits._settings = s
        got = limits._pool_limits()
        assert got["max_symbol_concentration_pct"] == s.risk.max_symbol_concentration_pct
        assert got["max_positions_per_expiry"] == s.risk.max_positions_per_expiry
        assert got["max_correlated_positions"] == s.risk.max_correlated_positions


class TestTheRegistry:
    def test_every_pool_rejection_is_classified(self):
        tree = ast.parse((SRC / "backtesting" / "pool.py").read_text())
        recorded = {n.slice.value for n in ast.walk(tree)
                    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "rejections" and isinstance(n.slice, ast.Constant)}
        assert recorded, "found no rejection reasons in pool.py"
        assert recorded <= set(POOL_REJECT_REASONS), recorded - set(POOL_REJECT_REASONS)
        assert set(POOL_REJECT_REASONS) <= recorded | {"no_contracts"}

    def test_the_shipped_scaled_list_names_known_strategies(self):
        from ait.backtesting.engine import DEBIT_STRATEGIES
        from ait.strategies.base import CREDIT_STRATEGIES
        from ait.strategies.selector import STRATEGY_MAP
        known = set(STRATEGY_MAP) | set(CREDIT_STRATEGIES) | set(DEBIT_STRATEGIES)
        shipped = load_settings().position_sizing.scaled_strategies
        assert set(shipped) <= known, set(shipped) - known

    def test_scalers_are_off_in_the_shipped_config(self):
        assert load_settings().position_sizing.scaled_strategies == []


class TestTheRetiredSettings:
    def test_one_budget_number_serves_live_and_research(self):
        s = load_settings()
        assert s.risk.max_position_risk_pct == pytest.approx(0.05)
        assert not hasattr(s.positions, "max_position_pct")
        assert not hasattr(s.backtest, "position_size_pct")
        assert "max_cost_fraction_of_capital" not in type(s.backtest.spread_tuning).model_fields

    def test_the_lot_function_is_a_floor_not_a_round(self):
        # 5000 * 0.05 / 99 = 2.53 lots: two, never three
        assert lots_for_budget(99.0, 5_000, BUDGET_PCT) == int(math.floor(250 / 99.0)) == 2
