"""W36: the leftovers assigned to this box after the R26 fix sets.

  * TEN values that manage trading were still raw ``os.environ.get("AIT_…",
    "<literal>")`` reads with no config home, against R19c ("config must be the
    ONLY place with values that manage trading"): the condor's IV-rank floor,
    its short-delta band and expected-move band, the risk manager's IV-rank cap
    and per-expiry cap, the entry ladder start and order timeout, and the
    credit touch-stop switch. None is in the env contract, so none is covered
    by ``test_every_trading_key_has_a_config_home``.
  * the IV-rank floor rejects SILENTLY. On 2026-09-18 it refused every condor
    all session (IWM 3.6, QQQ 12.3, SPY 12.7 against a floor of 15) and the
    only trace was ``signals_generated total_signals=0``.
  * the Black-Scholes risk-free rate was a bare ``r = 0.05`` in the Greeks math.
  * register ``time-calendar-03``: ``_trading_cycle`` returned for the
    no-new-entries window BEFORE its only outbox drain, so an orphan close
    landing in the last minutes of the session waited for the next one.

This change is about WHERE values live, not WHAT they are: every default below
is asserted equal to the literal it replaces.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import ait.config.runtime_env as RE

REPO = Path(__file__).resolve().parents[1]

# key -> (the literal it replaces, (config section, field))
HOMES = {
    "AIT_IRON_CONDOR_IV_FLOOR": ("15", ("backtest", "ic_iv_rank_floor")),
    "AIT_IC_DELTA_MIN": ("0.15", ("backtest", "ic_delta_min")),
    "AIT_IC_DELTA_MAX": ("0.30", ("backtest", "ic_delta_max")),
    "AIT_IC_EM_MIN": ("0.6", ("backtest", "ic_em_min")),
    "AIT_IC_EM_MAX": ("1.3", ("backtest", "ic_em_max")),
    "AIT_IV_RANK_CAP": ("85", ("risk", "iv_rank_cap")),
    "AIT_MAX_PER_EXPIRY": ("3", ("risk", "max_positions_per_expiry")),
    "AIT_ENTRY_LADDER_START": ("0.25", ("positions", "entry_ladder_start")),
    "AIT_ENTRY_ORDER_TIMEOUT": ("240", ("positions", "entry_order_timeout_seconds")),
    "AIT_CREDIT_TOUCH_STOP": ("1", ("exit", "credit_touch_stop")),
}


@pytest.fixture
def no_yaml(monkeypatch):
    for k in HOMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(RE, "_yaml_cache", {})


class TestEveryTradingKnobHasAHome:
    @pytest.mark.parametrize("key", sorted(HOMES))
    def test_it_is_in_the_contract_with_its_old_value(self, key, no_yaml):
        literal, home = HOMES[key]
        assert RE.CONTRACT_DEFAULTS.get(key) == literal, (
            f"{key} must default to the literal it replaces ({literal})")
        assert RE.CONFIG_BACKED.get(key) == home

    @pytest.mark.parametrize("key", sorted(HOMES))
    def test_config_yaml_is_honoured_and_env_still_wins(self, key, monkeypatch):
        section, field = HOMES[key][1]
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(RE, "_yaml_cache", {section: {field: 7}})
        assert RE.contract_float(key) == pytest.approx(7.0)
        monkeypatch.setenv(key, "9")
        assert RE.contract_float(key) == pytest.approx(9.0)

    def test_the_yaml_declares_exactly_what_live_runs_today(self):
        """.env on the live box sets none of these, so live ran the literals."""
        from ait.config.settings import load_settings
        s = load_settings(REPO / "config.yaml")
        for key, (literal, (section, field)) in HOMES.items():
            got = getattr(getattr(s, section), field)
            want = (literal == "1") if isinstance(got, bool) else float(literal)
            assert got == want, (key, got, want)

    def test_the_risk_free_rate_has_a_home(self):
        from ait.config.settings import load_settings
        assert load_settings(REPO / "config.yaml").options.risk_free_rate == pytest.approx(0.05)


class TestTheReadersUseTheContract:
    """Executed, not grepped: change the config value, watch the behaviour."""

    def _condor(self):
        from ait.strategies.iron_condor import IronCondor
        return IronCondor.__new__(IronCondor)

    def _signals(self, strat, iv_rank):
        import inspect
        chain = SimpleNamespace(calls=[], puts=[], symbol="SPY",
                                underlying_price=500.0, expiry=None,
                                atm_iv=0.2, expected_move=10.0,
                                iv_is_guessed=False)
        fn = strat.generate_signals
        kwargs = {}
        for name in inspect.signature(fn).parameters:
            kwargs[name] = {"symbol": "SPY", "direction": None, "confidence": 0.6,
                            "chain": chain, "iv_rank": iv_rank, "price": 500.0,
                            "current_price": 500.0}.get(name)
        return fn(**kwargs)

    def test_the_iv_floor_rejection_is_no_longer_silent(self, no_yaml):
        import ait.strategies.iron_condor as IC
        seen = []
        real = IC.log.info
        IC.log.info = lambda ev, **kw: seen.append((ev, kw))
        try:
            assert self._signals(self._condor(), iv_rank=12.3) == []
        finally:
            IC.log.info = real
        hits = [kw for ev, kw in seen if ev == "condor_rejected_iv_rank_floor"]
        assert hits and hits[0]["iv_rank"] == pytest.approx(12.3)
        assert hits[0]["floor"] == pytest.approx(15.0)

    def test_the_iv_floor_follows_config(self, monkeypatch):
        """Raise the floor in CONFIG to 50: an IV rank of 30 — which clears the
        old literal 15 — must now be refused, and say so with floor=50."""
        import ait.strategies.iron_condor as IC
        monkeypatch.delenv("AIT_IRON_CONDOR_IV_FLOOR", raising=False)
        monkeypatch.setattr(RE, "_yaml_cache",
                            {"backtest": {"ic_iv_rank_floor": 50}})
        seen = []
        real = IC.log.info
        IC.log.info = lambda ev, **kw: seen.append((ev, kw))
        try:
            assert self._signals(self._condor(), iv_rank=30.0) == []
        finally:
            IC.log.info = real
        hits = [kw for ev, kw in seen if ev == "condor_rejected_iv_rank_floor"]
        assert hits and hits[0]["floor"] == pytest.approx(50.0), seen

    def test_the_greeks_use_the_configured_rate(self, monkeypatch):
        """Same quotes, different r -> a different solved IV. HEAD: identical."""
        pytest.importorskip("py_vollib")
        from tests.test_w35_chain_iv_after_the_solve import _chain, _priced, _solve
        import ait.data.options_chain as OC

        def _atm_iv(rate):
            monkeypatch.setattr(OC, "_risk_free_rate", lambda: rate, raising=False)
            chain = _chain([_priced(k, "C") for k in (480.0, 500.0, 520.0)],
                           [_priced(k, "P") for k in (480.0, 500.0, 520.0)])
            _solve(chain)
            return chain.calls[1].implied_vol

        assert _atm_iv(0.05) == pytest.approx(0.22, abs=0.002)
        assert abs(_atm_iv(0.00) - _atm_iv(0.05)) > 0.002


class TestOrphanClosesAreDrainedInTheFinalMinutes:
    def test_the_drain_runs_inside_the_no_new_entries_window(self, tmp_path, monkeypatch):
        """time-calendar-03: the blackout `return` sat above the ONLY drain."""
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            o = rig.orch
            o._scheduler.should_avoid_new_trades = lambda: True
            o._drain_exit_outbox = AsyncMock()
            asyncio.run(o._trading_cycle())
            o._drain_exit_outbox.assert_awaited()
        finally:
            rig.restore()

    def test_no_entry_is_attempted_in_that_window(self, tmp_path, monkeypatch):
        """Control: moving the drain up must not move the blackout down."""
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            o = rig.orch
            o._scheduler.should_avoid_new_trades = lambda: True
            asyncio.run(o._trading_cycle())
            assert not rig.executed, rig.executed
        finally:
            rig.restore()
