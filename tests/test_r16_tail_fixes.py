"""R16 tail fixes — the low-severity findings closed in the final sweep.

Covers:
  engine   five string-patched credit builders (iron_butterfly,
           call_credit_spread, jade_lizard, wide_wing_condor,
           broken_wing_condor) — they had ZERO test coverage, so a future
           string patch could miss a dispatch site and silently mis-value or
           mis-exit an arm with green tests
  engine   jade_lizard added to the macro-event flatten (naked short put)
  engine   jade_lizard added to the legs/credit/max_loss annotation dispatch
  engine   drawdown-from-60d-high gate extended to the whole neutral-credit
           family (the one dispatch site the builder rollout missed)
  base     ONE CREDIT_STRATEGIES authority, imported by the engine
  condor   live wing snap-to-strike re-checked against the risk budget
  chain    one liquidity-threshold authority (config base, env override)
  chain    chain-source degradation to delayed Yahoo data is logged
  ml       meta_label missing/stale artifact is loud and refuses to arm
  master   fresh-models restart defers past the post-market window
  master   digest mirror metric measures mirror-vs-source lag
  master   append-only log rotation (keeper/dashboard/weblog/orchestrator)
  backtest run_backtest parity manifest reads authoritative live values

Everything here is pure-python: no processes, no network, no DB writes.
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import engine as _engine
from ait.backtesting.engine import Backtester
from ait.data.options_chain import OptionContract, OptionsChain
from ait.strategies import base as _base


@contextmanager
def structlog_events():
    """Capture structlog events regardless of level.

    setup_logging() is never called under pytest, so these loggers do NOT go
    through stdlib logging and caplog sees nothing; capture at the structlog
    layer instead. Yields a list that fills as events are emitted.
    """
    from structlog.testing import capture_logs
    with capture_logs() as events:
        yield events


def _has_event(events, name: str) -> bool:
    return any(e.get("event") == name for e in events)


# ---------------------------------------------------------------------------
# Shared synthetic inputs
# ---------------------------------------------------------------------------

PATCHED_BUILDERS = [
    "iron_butterfly",
    "call_credit_spread",
    "jade_lizard",
    "wide_wing_condor",
    "broken_wing_condor",
]

# S/IV points where every patched builder clears its OWN construction gates at
# the engine's default capital/sizing. Verified by execution 2026-08-10.
_BUILD_POINT = {
    "iron_butterfly":     (100.0, 0.25),
    "call_credit_spread": (100.0, 0.25),
    "jade_lizard":        (100.0, 0.25),
    "wide_wing_condor":   (100.0, 0.25),
    "broken_wing_condor": (100.0, 0.25),
}

_EXPECTED_LEGS = {
    "iron_butterfly": 4,
    "call_credit_spread": 2,
    "jade_lizard": 3,
    "wide_wing_condor": 4,
    "broken_wing_condor": 4,
}


def _df(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    p = pd.Series(np.full(n, price), index=idx)
    return pd.DataFrame({
        "Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p,
        "Volume": 1_000_000,
    })


def _bt(**kw) -> Backtester:
    kw.setdefault("initial_capital", 100_000.0)
    kw.setdefault("predictor", None)
    return Backtester(data=_df(), strategies=list(PATCHED_BUILDERS), **kw)


# ---------------------------------------------------------------------------
# Engine: the five string-patched builders actually build and are dispatched
# ---------------------------------------------------------------------------

class TestPatchedCreditBuilders:
    """The builders exist solely to produce shadow-tournament numbers that
    already promoted the live config once — but no test ever constructed one."""

    @pytest.mark.parametrize("strategy", PATCHED_BUILDERS)
    def test_builds_a_position_at_default_capital_and_sizing(self, strategy):
        bt = _bt()
        S, iv = _BUILD_POINT[strategy]
        pos = bt._build_credit_position(
            strategy, S=S, iv=iv, t=21 / 365.0, r=0.05, dte=21,
            today_date=date(2024, 6, 3), capital=100_000.0,
        )
        assert pos is not None, (
            f"{strategy} built nothing at default capital/sizing — a shadow "
            "run would report n=0 with no error"
        )
        assert pos["strategy"] == strategy
        assert pos["trade_type"] == "credit"
        assert pos["n_legs"] == _EXPECTED_LEGS[strategy]
        assert pos["contracts"] >= 1
        assert pos["entry_price"] > 0

    @pytest.mark.parametrize("strategy", PATCHED_BUILDERS)
    def test_reprice_dispatch_returns_a_finite_value(self, strategy):
        """A name absent from the _reprice_position branch chain falls through
        to the generic tail and silently mis-values the whole arm."""
        bt = _bt()
        S, iv = _BUILD_POINT[strategy]
        pos = bt._build_credit_position(
            strategy, S=S, iv=iv, t=21 / 365.0, r=0.05, dte=21,
            today_date=date(2024, 6, 3), capital=100_000.0,
        )
        assert pos is not None
        value = bt._reprice_position(pos, S, days_held=5, hist=None)
        assert value is not None
        assert np.isfinite(value)
        assert value > 0  # cost to buy the structure back

    @pytest.mark.parametrize("strategy", PATCHED_BUILDERS)
    def test_classified_as_credit(self, strategy):
        assert strategy in _engine.CREDIT_STRATEGIES
        assert strategy not in _engine.DEBIT_STRATEGIES

    @pytest.mark.parametrize("strategy", PATCHED_BUILDERS)
    def test_present_at_every_neutral_entry_gate(self, strategy):
        """All six neutral-credit entry gates dispatch off NEUTRAL_CREDIT_GATED
        — including the drawdown-from-60d-high gate, which was the ONE site the
        string-patched rollout missed (it still read `== "iron_condor"`).
        call_credit_spread is the bearish twin but is selectable only on
        NEUTRAL signals, so it belongs to the same entry population."""
        assert strategy in _engine.NEUTRAL_CREDIT_GATED

    def test_drawdown_gate_is_not_iron_condor_only(self):
        src = inspect.getsource(Backtester.run)
        assert "_pct_from_60d_high_threshold > -0.99" in src
        assert 'strategy == "iron_condor" and self._pct_from_60d_high_threshold' not in src, (
            "drawdown-from-60d-high gate is back to iron_condor-only: the "
            "baseline arm would carry a veto the condor-family arms skip"
        )

    def test_contracts_below_one_is_attributable_not_silent(self):
        """The budget gate is intentional policy; being SILENT about it is what
        let a zero-trade arm read as 'no edge' instead of 'never affordable'."""
        # 100k capital, 5% budget = $5,000; jade_lizard's strangle-margin
        # convention on a $450 underlying is $9,000/contract, so it can NEVER
        # size at index-ETF prices — the arm reports n=0 for a sizing reason.
        bt = _bt(initial_capital=100_000.0)
        with structlog_events() as events:
            pos = bt._build_credit_position(
                "jade_lizard", S=450.0, iv=0.18, t=21 / 365.0, r=0.05, dte=21,
                today_date=date(2024, 6, 3), capital=100_000.0,
            )
        assert pos is None
        assert _has_event(events, "credit_position_unaffordable")
        rec = [e for e in events
               if e.get("event") == "credit_position_unaffordable"][0]
        assert rec["strategy"] == "jade_lizard"
        assert rec["risk_per_contract"] > rec["budget"]

    def test_launch_scale_capital_documents_the_no_trade(self):
        """At the $2.1k launch account NOTHING in the condor family fits the
        per-trade budget — pinned so a tournament run at launch scale can never
        be mistaken for 'no edge'."""
        bt = _bt(initial_capital=2_100.0)
        for strategy in ("iron_butterfly", "jade_lizard", "wide_wing_condor",
                         "broken_wing_condor"):
            pos = bt._build_credit_position(
                strategy, S=450.0, iv=0.18, t=21 / 365.0, r=0.05, dte=21,
                today_date=date(2024, 6, 3), capital=2_100.0,
            )
            assert pos is None, f"{strategy} sized a position off a $105 budget"


# ---------------------------------------------------------------------------
# Engine: jade_lizard dispatch sites
# ---------------------------------------------------------------------------

class TestJadeLizardDispatch:

    def _pos(self) -> dict:
        bt = _bt()
        pos = bt._build_credit_position(
            "jade_lizard", S=100.0, iv=0.25, t=21 / 365.0, r=0.05, dte=21,
            today_date=date(2024, 6, 3), capital=100_000.0,
        )
        assert pos is not None
        return pos

    def test_annotation_block_sets_legs_credit_and_max_loss(self):
        """Previously fell to the else -> legs=[] and NO credit/max_loss keys,
        so result.py fell back to `abs(pnl)*2` as the risk proxy."""
        src = inspect.getsource(Backtester.run)
        assert 'pos.get("strategy") == "jade_lizard"' in src

        # Replicate the annotation the run loop applies.
        pos = self._pos()
        ep = pos["entry_price"]
        contracts = pos["contracts"]
        legs = [
            {"type": "short_put",  "strike": pos["short_put_strike"]},
            {"type": "short_call", "strike": pos["short_call_strike"]},
            {"type": "long_call",  "strike": pos["long_call_strike"]},
        ]
        assert len(legs) == pos["n_legs"] == 3
        assert round(ep * 100 * contracts, 2) > 0
        assert pos["max_loss_per_share"] > 0

    def test_macro_flatten_covers_the_naked_put_side(self, monkeypatch):
        """jade_lizard carries a NAKED short put — the 'wings cap the surprise'
        exemption that keeps condors in through events does not apply."""
        class _Cal:
            def __init__(self, d2e):
                self._d2e = d2e

            def days_until_next_event(self, d):
                return self._d2e

        bt = _bt()
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "1")
        # DTE 20 (past the DTE<=5 close), pnl below every TP rung.
        pos_common = {"expiry_date": "2026-06-23", "entry_date": "2026-06-03",
                      "high_water_mark": 0.0}
        today = date(2026, 6, 3)

        bt._economic_cal = _Cal(2)  # inside the 5-day strangle-class window
        jade = dict(pos_common, strategy="jade_lizard")
        out = bt._check_exit_credit(jade, pnl_pct=0.0, current_date=today)
        assert out is not None and out["exit_reason"] == "macro_event_flatten"

        # Same window, defined-risk condor: still EXEMPT (PLAN 2026-08-04).
        condor = dict(pos_common, strategy="iron_condor")
        assert bt._check_exit_credit(
            condor, pnl_pct=0.0, current_date=today) is None

        # Outside the 5-day window jade_lizard holds like everything else.
        bt._economic_cal = _Cal(9)
        jade2 = dict(pos_common, strategy="jade_lizard")
        assert bt._check_exit_credit(
            jade2, pnl_pct=0.0, current_date=today) is None

    def test_macro_flatten_disarmed_without_the_env_flag(self, monkeypatch):
        class _Cal:
            def days_until_next_event(self, d):
                return 1

        bt = _bt()
        bt._economic_cal = _Cal()
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "0")
        jade = {"strategy": "jade_lizard", "expiry_date": "2026-06-23",
                "entry_date": "2026-06-03", "high_water_mark": 0.0}
        assert bt._check_exit_credit(
            jade, pnl_pct=0.0, current_date=date(2026, 6, 3)) is None

    def test_max_loss_per_share_dwarfs_a_condor(self):
        """Evidence for the flatten: ~93.7 vs ~12.8 on a $100 underlying."""
        jade = self._pos()
        bt = _bt()
        condor = bt._build_credit_position(
            "iron_condor", S=100.0, iv=0.25, t=21 / 365.0, r=0.05, dte=21,
            today_date=date(2024, 6, 3), capital=100_000.0,
        )
        assert condor is not None
        assert jade["max_loss_per_share"] > 5 * condor["max_loss_per_share"]

    def test_undefined_risk_membership(self):
        assert "jade_lizard" in _base.UNDEFINED_RISK_STRATEGIES


# ---------------------------------------------------------------------------
# One CREDIT_STRATEGIES authority
# ---------------------------------------------------------------------------

class TestCreditStrategiesSingleAuthority:

    def test_engine_imports_the_base_set(self):
        assert _engine.CREDIT_STRATEGIES is _base.CREDIT_STRATEGIES, (
            "engine.py re-declared its own CREDIT_STRATEGIES — the two forks "
            "silently disagreed on 7 names before R16"
        )

    def test_engine_module_declares_no_second_set(self):
        src = inspect.getsource(_engine)
        assert "CREDIT_STRATEGIES = {" not in src
        assert "CREDIT_STRATEGIES = frozenset" not in src

    def test_live_names_survive(self):
        for name in ("iron_condor", "short_strangle", "covered_call",
                     "cash_secured_put"):
            assert name in _base.CREDIT_STRATEGIES
            assert name in _base.LIVE_CREDIT_STRATEGIES

    def test_backtest_only_names_classify_as_credit(self):
        for name in PATCHED_BUILDERS + ["put_credit_spread", "short_straddle"]:
            assert name in _base.CREDIT_STRATEGIES, (
                f"{name} would be booked with the DEBIT P&L sign convention"
            )

    def test_debit_names_stay_out(self):
        for name in ("long_call", "long_put", "bull_call_spread",
                     "bear_put_spread", "long_straddle", "event_straddle"):
            assert name not in _base.CREDIT_STRATEGIES


# ---------------------------------------------------------------------------
# Live iron condor: wing snap must not overshoot the risk budget
# ---------------------------------------------------------------------------

def _c(strike, right, delta, bid, ask, iv=0.20):
    return OptionContract(
        symbol="TST", expiry=date.today() + timedelta(days=21), strike=strike,
        right=right, bid=bid, ask=ask, last=(bid + ask) / 2, volume=500,
        open_interest=500, implied_vol=iv, delta=delta,
    )


def _grid_chain(spot: float = 100.0, step: float = 5.0) -> OptionsChain:
    """$`step`-grid chain with a ~0.20-delta short ~1 EM out on each side.

    Premiums halve per strike so a one-wide spread carries real credit; the
    coarse grid is the point — snap-to-strike cannot land below $5 of width.
    """
    puts, calls = [], []
    for i in range(1, 9):
        mid = 10.0 * (0.5 ** i)
        delta = max(0.05, 0.20 - 0.03 * (i - 3))
        puts.append(_c(spot - step * i, "P", -delta,
                       bid=round(mid * 0.95, 4), ask=round(mid * 1.05, 4)))
        calls.append(_c(spot + step * i, "C", delta,
                        bid=round(mid * 0.95, 4), ask=round(mid * 1.05, 4)))
    chain = OptionsChain(symbol="TST", underlying_price=spot,
                         expiry=date.today() + timedelta(days=21),
                         calls=calls, puts=puts,
                         atm_iv=0.20, expected_move=step * 3)
    return chain


class TestCondorBudgetOvershoot:

    @pytest.fixture(autouse=True)
    def _pinned_env(self, monkeypatch):
        """Pin the construction knobs so ambient env cannot move the gates."""
        monkeypatch.setenv("AIT_IC_WING_K", "1.0")
        monkeypatch.setenv("AIT_IC_MIN_CREDIT", "0.70")
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0.20")
        monkeypatch.setenv("AIT_IRON_CONDOR_IV_FLOOR", "15")
        monkeypatch.setenv("AIT_IC_DELTA_MIN", "0.15")
        monkeypatch.setenv("AIT_IC_DELTA_MAX", "0.30")
        monkeypatch.setenv("AIT_IC_EM_MIN", "0.6")
        monkeypatch.setenv("AIT_IC_EM_MAX", "1.3")

    def _signals(self, budget):
        from ait.strategies.base import SignalDirection
        from ait.strategies.iron_condor import IronCondor
        ic = IronCondor()
        ic.risk_budget = budget
        return ic.generate_signals(
            "TST", _grid_chain(), SignalDirection.NEUTRAL,
            confidence=0.6, iv_rank=50.0,
        )

    def test_over_budget_structure_is_rejected_in_strategy(self):
        """A $90 budget caps the affordable width at ~$1.13, but the $5 grid
        snaps the wing to $5 — max_loss ~$375, 4x the budget the cap was
        derived from. Previously emitted and left to RiskManager gate 6b."""
        with structlog_events() as events:
            assert self._signals(budget=90.0) == []
        rejects = [e for e in events if e.get("event") == "condor_entry_quality"
                   and e.get("status") == "rejected"]
        assert rejects and rejects[-1]["reason"] == "max_loss_over_budget"

    def test_budget_respecting_structure_still_generates(self):
        sigs = self._signals(budget=100_000.0)
        assert len(sigs) == 1
        assert 0 < sigs[0].max_loss <= 100_000.0

    def test_gate_boundary_is_the_budget_itself(self):
        """max_loss exactly at budget passes; a dollar less of budget fails."""
        sig = self._signals(budget=100_000.0)[0]
        assert self._signals(budget=sig.max_loss) != []
        assert self._signals(budget=sig.max_loss - 1.0) == []

    def test_no_budget_means_no_new_gate(self):
        """risk_budget None (paper big-account behavior) must be unchanged."""
        assert len(self._signals(budget=None)) == 1

    def test_gate_matches_risk_manager_semantics(self):
        """The gate must compare (max_width - credit)*100 to risk_budget — the
        same quantity RiskManager 6b uses — not some looser proxy."""
        import ait.strategies.iron_condor as _ic
        src = inspect.getsource(_ic.IronCondor.generate_signals)
        assert "max_loss_over_budget" in src
        assert "max_loss > self.risk_budget" in src


# ---------------------------------------------------------------------------
# Options chain: liquidity authority + source telemetry
# ---------------------------------------------------------------------------

class TestLiquidityThresholdAuthority:

    def setup_method(self):
        from ait.data.options_chain import _liquidity_thresholds
        _liquidity_thresholds.cache_clear()

    def teardown_method(self):
        from ait.data.options_chain import _liquidity_thresholds
        _liquidity_thresholds.cache_clear()

    def test_config_is_the_base_when_env_is_unset(self, monkeypatch):
        from ait.config.settings import OptionsConfig
        from ait.data.options_chain import _resolve_liquidity
        for k in ("AIT_LIQ_MIN_VOLUME", "AIT_LIQ_MIN_OI", "AIT_LIQ_MAX_SPREAD"):
            monkeypatch.delenv(k, raising=False)
        cfg = OptionsConfig(min_volume=0, min_open_interest=10,
                            max_bid_ask_spread_pct=0.40)
        assert _resolve_liquidity(cfg) == (0, 10, 0.40)

    def test_env_overrides_config_per_key(self, monkeypatch):
        from ait.config.settings import OptionsConfig
        from ait.data.options_chain import _resolve_liquidity
        monkeypatch.setenv("AIT_LIQ_MIN_VOLUME", "7")
        monkeypatch.delenv("AIT_LIQ_MIN_OI", raising=False)
        monkeypatch.delenv("AIT_LIQ_MAX_SPREAD", raising=False)
        cfg = OptionsConfig(min_volume=0, min_open_interest=10,
                            max_bid_ask_spread_pct=0.40)
        assert _resolve_liquidity(cfg) == (7, 10, 0.40)

    def test_both_filters_agree(self, monkeypatch):
        """is_liquid (contract level) and filter_liquid (chain level) used to
        read two independent sources with different defaults (50/100/0.15 vs
        0/10/0.40) — a contract could pass one and fail the other."""
        from ait.config.settings import OptionsConfig
        from ait.data.options_chain import _liquidity_thresholds
        for k, v in (("AIT_LIQ_MIN_VOLUME", "10"), ("AIT_LIQ_MIN_OI", "20"),
                     ("AIT_LIQ_MAX_SPREAD", "0.30")):
            monkeypatch.setenv(k, v)
        _liquidity_thresholds.cache_clear()

        keep = _c(100.0, "C", 0.20, bid=1.00, ask=1.10)   # spread 9.5%, vol 500
        drop = _c(105.0, "C", 0.15, bid=1.00, ask=1.90)   # spread ~62%
        chain = OptionsChain(symbol="TST", underlying_price=100.0,
                             expiry=date.today() + timedelta(days=21),
                             calls=[keep, drop], puts=[])
        cfg = OptionsConfig(min_volume=0, min_open_interest=10,
                            max_bid_ask_spread_pct=0.40)
        filtered = chain.filter_liquid(cfg)
        assert [c.strike for c in filtered.calls] == [100.0]
        assert keep.is_liquid is True
        assert drop.is_liquid is False

    def test_env_only_process_still_resolves(self, monkeypatch):
        """No config passed -> loaded settings, never the entry-killing
        50/100/0.15 code defaults (which reject every contract on this feed)."""
        from ait.data.options_chain import _liquidity_thresholds
        for k in ("AIT_LIQ_MIN_VOLUME", "AIT_LIQ_MIN_OI", "AIT_LIQ_MAX_SPREAD"):
            monkeypatch.delenv(k, raising=False)
        _liquidity_thresholds.cache_clear()
        min_vol, min_oi, max_spread = _liquidity_thresholds()
        assert (min_vol, min_oi, max_spread) != (50, 100, 0.15)


class TestChainSourceTelemetry:

    def _svc(self):
        from ait.data.options_chain import OptionsChainService
        return OptionsChainService.__new__(OptionsChainService)

    def _wired_svc(self, *, ibkr_chains, yahoo_chains):
        """A real OptionsChainService with only the two FEEDS stubbed, so
        get_chain's own resolution/stamping/caching logic runs unmodified."""
        from ait.config.settings import OptionsConfig
        from ait.data.options_chain import OptionsChainService
        svc = OptionsChainService(ibkr_client=MagicMock(connected=False),
                                  market_data=MagicMock(),
                                  config=OptionsConfig())
        svc._market_data.get_current_price = AsyncMock(return_value=100.0)
        svc._get_ibkr_chain = AsyncMock(return_value=ibkr_chains)
        svc._get_yahoo_chain = AsyncMock(return_value=yahoo_chains)
        return svc

    def _chain(self):
        return OptionsChain(symbol="SPY", underlying_price=100.0,
                            expiry=date.today() + timedelta(days=21),
                            calls=[], puts=[])

    def test_yahoo_degradation_warns(self):
        svc = self._svc()
        with structlog_events() as events:
            svc._log_chain_source("SPY", "yahoo_delayed", [self._chain()])
        rec = [e for e in events
               if e.get("event") == "chain_source_degraded"]
        assert rec, "silent degradation to delayed Yahoo data"
        assert rec[0]["symbol"] == "SPY"
        assert rec[0]["source"] == "yahoo_delayed"
        assert rec[0]["log_level"] == "warning"

    def test_ibkr_path_does_not_warn(self):
        svc = self._svc()
        with structlog_events() as events:
            svc._log_chain_source("SPY", "ibkr", [self._chain()])
        assert not _has_event(events, "chain_source_degraded")
        assert _has_event(events, "chain_source")

    def test_no_chain_at_all_warns(self):
        svc = self._svc()
        with structlog_events() as events:
            svc._log_chain_source("SPY", "yahoo_delayed", [])
        assert _has_event(events, "chain_source_unavailable")

    async def test_get_chain_stamps_the_yahoo_fallback_it_used(self):
        """R17-bis: was `assert "chain.source = source" in getsource(get_chain)`
        — true of a line in a dead branch, of a line that assigns the wrong
        variable's value, and of a line whose loop no longer runs. get_chain
        was never executed by any test, so the stamp that every downstream
        feed-attribution check depends on had no executable coverage."""
        chain = self._chain()
        assert chain.source == ""          # unstamped until get_chain says so
        svc = self._wired_svc(ibkr_chains=[], yahoo_chains=[chain])

        with structlog_events() as events:
            out = await svc.get_chain("SPY")

        assert [c.source for c in out] == ["yahoo_delayed"]
        rec = [e for e in events if e.get("event") == "chain_source_degraded"]
        assert rec and rec[0]["log_level"] == "warning"

    async def test_get_chain_stamps_ibkr_when_ibkr_served_it(self):
        # Negative control: a hardcoded "yahoo_delayed" (or a stamp taken from
        # the wrong branch) must not pass the test above.
        chain = self._chain()
        svc = self._wired_svc(ibkr_chains=[chain], yahoo_chains=[])

        with structlog_events() as events:
            out = await svc.get_chain("SPY")

        assert [c.source for c in out] == ["ibkr"]
        svc._get_yahoo_chain.assert_not_awaited()   # no needless fallback
        assert not _has_event(events, "chain_source_degraded")
        assert _has_event(events, "chain_source")

    def test_source_survives_filtering(self):
        from ait.config.settings import OptionsConfig
        chain = self._chain()
        chain.source = "yahoo_delayed"
        assert chain.filter_by_delta(0.0, 1.0).source == "yahoo_delayed"
        assert chain.filter_liquid(OptionsConfig()).source == "yahoo_delayed"


# ---------------------------------------------------------------------------
# meta_label: loud quarantine, no accidental re-arming
# ---------------------------------------------------------------------------

class TestMetaLabelQuarantine:
    """The gate is DISABLED in config (meta_label.enabled=false) and stays that
    way — these tests assert observability and refusal, never re-arming."""

    def test_model_dir_is_repo_anchored(self):
        import ait.ml.meta_label as _ml
        src = inspect.getsource(_ml)
        assert 'MODEL_DIR = Path("models")' not in src
        assert "parents[3]" in src

    def test_missing_artifact_logs_at_error(self, tmp_path, monkeypatch):
        import ait.ml.meta_label as _ml
        monkeypatch.setattr(_ml, "MODEL_DIR", tmp_path)
        (tmp_path / "meta_label.pkl.clobbered_20260718").write_bytes(b"x")
        m = _ml.MetaLabeler()
        with structlog_events() as events:
            assert m.load_model() is False
        rec = [e for e in events
               if e.get("event") == "meta_label_artifact_missing"]
        assert rec, "the 07-18 quarantine has been silent for weeks"
        assert rec[0]["log_level"] == "error"
        assert rec[0]["quarantined"] == ["meta_label.pkl.clobbered_20260718"]
        assert m.is_trained is False

    def test_stale_artifact_refuses_to_arm(self, tmp_path, monkeypatch):
        import ait.ml.meta_label as _ml
        monkeypatch.setattr(_ml, "MODEL_DIR", tmp_path)
        art = tmp_path / "meta_label.pkl"
        art.write_bytes(b"not-a-real-pickle")
        old = (datetime.now()
               - timedelta(days=_ml.MAX_ARTIFACT_AGE_DAYS + 10)).timestamp()
        os.utime(art, (old, old))
        m = _ml.MetaLabeler()
        with structlog_events() as events:
            assert m.load_model() is False
        assert _has_event(events, "meta_label_artifact_stale")
        assert m.is_trained is False

    def test_fresh_artifact_is_not_flagged_stale(self, tmp_path, monkeypatch):
        """Guard the guard: a current artifact must not be refused (the load
        still fails here because the payload is junk, but for the RIGHT
        reason)."""
        import ait.ml.meta_label as _ml
        monkeypatch.setattr(_ml, "MODEL_DIR", tmp_path)
        (tmp_path / "meta_label.pkl").write_bytes(b"not-a-real-pickle")
        with structlog_events() as events:
            assert _ml.MetaLabeler().load_model() is False
        assert not _has_event(events, "meta_label_artifact_stale")
        assert not _has_event(events, "meta_label_artifact_missing")
        assert _has_event(events, "meta_label_load_failed")

    def test_degraded_feature_coverage_refuses_to_arm(self, tmp_path,
                                                     monkeypatch):
        """build_training_data supplies only 9 of the 20 META_FEATURES today
        (sentiment_score permanently 0 after R12-C) — retraining on that is the
        corrupted-input condition the artifact was quarantined for."""
        import ait.ml.meta_label as _ml
        monkeypatch.setattr(_ml, "MODEL_DIR", tmp_path)
        rng = np.random.RandomState(3)
        n = 60
        cols = {f: rng.normal(size=n) for f in _ml.META_FEATURES[:9]}
        cols["sentiment_score"] = np.zeros(n)  # retired feed: constant
        cols["profitable"] = rng.randint(0, 2, size=n)
        df = pd.DataFrame(cols)

        m = _ml.MetaLabeler()
        with structlog_events() as events:
            stats = m.train(df)
        assert stats == {}
        assert m.is_trained is False
        rec = [e for e in events if e.get("event") == "meta_label_arm_refused"]
        assert rec and rec[0]["reason"] == "insufficient_feature_coverage"
        assert "sentiment_score" in rec[0]["dead_or_constant"]
        assert not (tmp_path / "meta_label.pkl").exists()

    def test_full_coverage_is_allowed_to_train(self, tmp_path, monkeypatch):
        """The guard must not be a blanket refusal — with the full feature set
        present and informative, training proceeds normally."""
        import ait.ml.meta_label as _ml
        monkeypatch.setattr(_ml, "MODEL_DIR", tmp_path)
        rng = np.random.RandomState(11)
        n = 80
        cols = {f: rng.normal(size=n) for f in _ml.META_FEATURES}
        cols["profitable"] = rng.randint(0, 2, size=n)
        with structlog_events() as events:
            stats = _ml.MetaLabeler().train(pd.DataFrame(cols))
        assert not _has_event(events, "meta_label_arm_refused")
        # xgboost may be absent in a slim env; either way it must not be the
        # coverage guard that stopped it.
        assert stats != {} or _has_event(events,
                                         "xgboost_not_installed_for_meta_label")

    def test_regeneration_path_is_documented_in_source(self):
        import ait.ml.meta_label as _ml
        src = inspect.getsource(_ml)
        assert "REGENERATION PATH" in src
        assert "retrain_models" in src


# ---------------------------------------------------------------------------
# master.py: post-market restart window, mirror lag, log rotation
# ---------------------------------------------------------------------------

class TestPostMarketRestartWindow:

    def test_close_to_close_plus_window_defers(self):
        from ait.orchestration.master import _in_post_market_window
        from ait.utils.time import ET
        d = date(2026, 8, 10)  # a Monday trading day
        for hh, mm, expected in [
            (15, 59, False),   # still open — the market_open branch handles it
            (16, 0,  True),    # the exact minute the old check consumed the marker
            (16, 1,  True),    # observed duplicate EOD restarts landed here
            (16, 14, True),    # last minute of the bot's POST_MARKET phase
            (16, 20, True),    # inside the safety grace
            (16, 36, False),   # clear
            (19, 0,  False),
        ]:
            now = datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)
            assert _in_post_market_window(now) is expected, f"{hh}:{mm:02d}"

    def test_non_trading_day_never_defers(self):
        from ait.orchestration.master import _in_post_market_window
        from ait.utils.time import ET
        sunday = datetime(2026, 8, 9, 16, 5, tzinfo=ET)
        assert _in_post_market_window(sunday) is False

    def test_calendar_outage_falls_back_to_the_clock_not_forever(self,
                                                                monkeypatch):
        """Fail closed inside the window, but a calendar outage must never
        wedge the model-reload restart permanently."""
        import ait.utils.time as _t

        def _boom(*a, **kw):
            raise RuntimeError("calendar unavailable")

        monkeypatch.setattr(_t, "is_trading_day", _boom)
        from ait.orchestration.master import _in_post_market_window
        assert _in_post_market_window(datetime(2026, 8, 10, 16, 5)) is True
        assert _in_post_market_window(datetime(2026, 8, 10, 18, 0)) is False

    def test_health_check_consults_the_window(self):
        from ait.orchestration import master as _m
        src = inspect.getsource(_m.BotManager.health_check)
        assert "_in_post_market_window()" in src
        assert "fresh_models_restart_deferred_post_market" in src
        # The marker must survive the deferral (no unlink in that branch).
        deferred = src.split("fresh_models_restart_deferred_post_market")[0]
        assert deferred.count("marker.unlink()") == 0


class TestMirrorLag:

    def _db(self, path: Path, newest: str | None):
        import sqlite3
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE trades (entry_time TEXT, exit_time TEXT)")
        if newest:
            con.execute("INSERT INTO trades VALUES (?, NULL)", (newest,))
        con.commit()
        con.close()

    def test_equal_content_reads_current_not_stale(self, tmp_path, monkeypatch):
        """The bug: a quiet market made a perfectly fresh mirror report
        'mirror 69h stale' because the metric measured last-TRADE recency."""
        from ait.orchestration import master as _m
        home = tmp_path / "home"
        (home / "Documents" / "ait_backups").mkdir(parents=True)
        data = tmp_path / "data"
        data.mkdir()
        old_ts = (datetime.now() - timedelta(days=4)).isoformat()
        self._db(home / "Documents" / "ait_backups" / "ait_state.latest.db",
                 old_ts)
        self._db(data / "ait_state.db", old_ts)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(_m, "DATA_DIR", data)

        state, lag = _m._mirror_lag()
        assert state == "current"
        assert lag == 0.0
        # ...while the wall-clock content age is still ~96h.
        assert _m._mirror_content_age_hours() > 90

    def test_mirror_behind_source_reports_the_lag(self, tmp_path, monkeypatch):
        """This is the real 07-14 signature the digest has to catch."""
        from ait.orchestration import master as _m
        home = tmp_path / "home"
        (home / "Documents" / "ait_backups").mkdir(parents=True)
        data = tmp_path / "data"
        data.mkdir()
        self._db(home / "Documents" / "ait_backups" / "ait_state.latest.db",
                 (datetime.now() - timedelta(hours=30)).isoformat())
        self._db(data / "ait_state.db",
                 (datetime.now() - timedelta(hours=2)).isoformat())
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(_m, "DATA_DIR", data)

        state, lag = _m._mirror_lag()
        assert state == "behind"
        assert 27 < lag < 29

    def test_missing_mirror_still_reported(self, tmp_path, monkeypatch):
        from ait.orchestration import master as _m
        home = tmp_path / "home"
        (home / "Documents").mkdir(parents=True)
        data = tmp_path / "data"
        data.mkdir()
        self._db(data / "ait_state.db", datetime.now().isoformat())
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(_m, "DATA_DIR", data)
        assert _m._mirror_lag() == ("missing", None)

    def test_digest_prints_the_lag_state(self):
        from ait.orchestration import master as _m
        src = inspect.getsource(_m.daily_digest)
        assert "_mirror_lag()" in src
        assert "mirror current" in src
        assert "mirror BEHIND live" in src


class TestLogRotation:

    def test_rotates_above_cap_and_shifts_backups(self, tmp_path, monkeypatch):
        from ait.orchestration import master as _m
        monkeypatch.setattr(_m, "LOGS_DIR", tmp_path)
        target = tmp_path / "keeper.log"
        target.write_bytes(b"n" * (21 * 1024 * 1024))
        (tmp_path / "keeper.log.1").write_text("older")

        assert _m._rotate_if_oversized(target, 20 * 1024 * 1024) is True
        assert not target.exists()
        assert (tmp_path / "keeper.log.1").stat().st_size == 21 * 1024 * 1024
        assert (tmp_path / "keeper.log.2").read_text() == "older"

    def test_under_cap_is_untouched(self, tmp_path):
        from ait.orchestration import master as _m
        target = tmp_path / "dashboard.log"
        target.write_text("small")
        assert _m._rotate_if_oversized(target, 20 * 1024 * 1024) is False
        assert target.read_text() == "small"

    def test_missing_file_is_not_an_error(self, tmp_path):
        from ait.orchestration import master as _m
        assert _m._rotate_if_oversized(tmp_path / "nope.log", 1) is False

    def test_sweep_covers_the_unrotated_logs(self, tmp_path, monkeypatch):
        from ait.orchestration import master as _m
        monkeypatch.setattr(_m, "LOGS_DIR", tmp_path)
        for name in ("keeper.log", "dashboard.log", "weblog.log",
                     "orchestrator.log"):
            assert name in _m._ROTATE_TARGETS
            (tmp_path / name).write_bytes(b"x" * (21 * 1024 * 1024))
        _m.rotate_oversized_logs()
        for name in _m._ROTATE_TARGETS:
            assert not (tmp_path / name).exists()
            assert (tmp_path / f"{name}.1").exists()

    def test_cleanup_does_not_delete_the_active_logs(self, tmp_path,
                                                    monkeypatch):
        from ait.orchestration import master as _m
        logs = tmp_path / "logs"
        reports = tmp_path / "reports"
        logs.mkdir()
        reports.mkdir()
        monkeypatch.setattr(_m, "LOGS_DIR", logs)
        monkeypatch.setattr(_m, "REPORTS_DIR", reports)
        active = logs / "keeper.log"
        active.write_text("live")
        stale = logs / "something_old.log"
        stale.write_text("old")
        old = (datetime.now() - timedelta(days=90)).timestamp()
        os.utime(stale, (old, old))

        _m.cleanup_old_logs()
        assert active.exists()
        assert not stale.exists()


class TestKeeperBat:
    """The keeper is a .bat — assert the shape of the fixes textually."""

    def _text(self) -> str:
        root = Path(_engine.__file__).resolve().parents[3]
        return (root / "keeper_ait.bat").read_text()

    def test_no_bare_timeout_as_the_only_delay(self):
        text = self._text()
        assert "ping -n 91 127.0.0.1" in text, (
            "`timeout /t` fails instantly without a console input handle — the "
            "loop then spins at ~1-2 iterations/sec"
        )
        # timeout survives only as a guarded fallback.
        for line in text.splitlines():
            if line.strip().startswith("timeout /t"):
                pytest.fail("unguarded `timeout /t` is back as the primary delay")

    def test_fallbacks_use_no_shell_chaining_operators(self):
        """PS 5.1 / cmd parser safety: no && or || anywhere in the file."""
        text = self._text()
        assert "&&" not in text
        assert "||" not in text
        assert text.count("if errorlevel 1") >= 3  # process check + 2 fallbacks

    def test_keeper_log_is_size_capped(self):
        text = self._text()
        assert "%%~zA" in text
        assert "20971520" in text
        assert 'move /y "logs\\keeper.log" "logs\\keeper.log.1"' in text


# ---------------------------------------------------------------------------
# run_backtest.py parity manifest
# ---------------------------------------------------------------------------

class TestParityManifest:

    def _manifest(self):
        import sys
        root = Path(_engine.__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import run_backtest as rb
        old_argv = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            args = rb.parse_args()
        finally:
            sys.argv = old_argv
        return rb, args, rb.build_parity_manifest(args)

    def test_dte_band_comes_from_loaded_settings(self):
        from ait.config.settings import load_settings
        _rb, _args, manifest = self._manifest()
        assert manifest["live"]["dte_band"] == list(
            load_settings().options.dte_range)
        assert manifest["live"]["dte_band"] != [14, 45] or \
            list(load_settings().options.dte_range) == [14, 45]

    def test_blackout_comes_from_loaded_settings(self):
        from ait.config.settings import load_settings
        _rb, _args, manifest = self._manifest()
        assert (manifest["live"]["pre_event_blackout_days"]
                == load_settings().risk.pre_event_blackout_days)

    def test_live_env_values_come_from_the_runtime_contract(self):
        """A bare backtest process never applies the bot's env contract, so
        reading os.environ reported code defaults as live values."""
        _rb, _args, manifest = self._manifest()
        assert manifest["live"]["wing_k"] == 1.6
        assert manifest["live"]["ic_min_credit_width"] == 0.10
        assert manifest["live"]["macro_flatten_enabled"] is True

    def test_manifest_does_not_mutate_this_process_env(self):
        before = dict(os.environ)
        self._manifest()
        assert dict(os.environ) == before

    def test_stale_four_day_macro_comment_is_gone(self):
        root = Path(_engine.__file__).resolve().parents[3]
        text = (root / "run_backtest.py").read_text()
        assert "credit entries <=4d pre-event" not in text
        assert "blocks credit entries <=4 days" not in text
