"""HOT-PATH SMOKE — the entry pipeline is EXECUTED, not just read.

Why this file exists (incident, 2026-08): a one-line AttributeError in
``TradingOrchestrator._trading_cycle`` ran live for THREE DAYS. Every unit
test was green and both deploy-smoke checks (import walk, load_settings)
passed the whole time, because nothing anywhere ever *awaited* the entry
pipeline: every orchestrator test constructs via ``__new__`` and drives one
exit-side method, and the entry side was covered only by
``inspect.getsource`` string assertions — which cannot see an AttributeError.

This file closes that hole. It builds a REAL orchestrator through the REAL
``__init__`` (previously zero coverage) against fakes at the I/O boundary,
and awaits each hot-path function once:

    _trading_cycle()  _monitor_positions_fast()  _scan_symbol()
    _try_execute()    _post_market()

Two design rules make it able to FAIL on the bug class it exists to catch:

1. NO MagicMock for anything the orchestrator reads attributes off.
   ``settings`` is a REAL ``Settings`` loaded from the REAL config.yaml, so
   ``settings.options.strategies`` (the 3-day bug's exact shape) raises
   AttributeError when the attribute is wrong — a MagicMock would have
   happily returned a child mock and stayed green. Same reasoning as
   tests/fakes.py's SimpleNamespace-not-MagicMock rule for broker shapes.

2. EVERY hot-path function swallows its own exceptions (by design — a
   crashing monitor must not kill the loop), so "no exception propagated"
   proves nothing. A log spy records every structlog event the orchestrator
   emits and the test FAILS on any ``*_failed`` / ``*_error`` event that is
   not on a short, justified allow-list. That is what turns a swallowed
   AttributeError into a red test.

Hermetic: cwd is redirected to tmp_path before anything is constructed, so
every cwd-relative artefact (data/ait_state.db, data/historical.db, the
duckdb analytics store, thompson/counterfactual JSON, HALT flag files)
lands in the sandbox. No IBKR connection, no network, no notifications, no
writes to the repo's data/. Runtime is a few seconds.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.config.settings import load_settings
from ait.data.options_chain import OptionContract, OptionsChain
from ait.strategies.base import Signal, SignalDirection
from tests.fakes import FakeIB, FakeIBKRClient

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

SYMBOL = "SPY"
SPOT = 500.0
VIX = 16.0


# ---------------------------------------------------------------------------
# Log spy — the mechanism that makes a SWALLOWED exception fail the test
# ---------------------------------------------------------------------------

# Events whose name says "we caught an exception here". Any of these appearing
# during a hot-path run means a function silently degraded — exactly the state
# the 3-day outage lived in.
_FAILURE_TOKENS = ("failed", "_error", "error_", "broken", "exception")

# Justified exceptions to the rule. Keep this list SHORT and explain each one;
# a new entry here is a decision to let the smoke ignore a real degradation.
_ALLOWED_FAILURE_EVENTS = {
    # KNOWN LIVE BUG, found by this file's very first run (2026-08-14) —
    # NOT a sandbox artefact. TradingOrchestrator._persist_daily_iv does
    #     ivs = [c.atm_iv() for c in chains if hasattr(c, "atm_iv")]
    # but OptionsChain.atm_iv is a dataclass FIELD (float), not a method, so
    # every call raises "TypeError: 'float' object is not callable" into the
    # method's own `except Exception` and the R16 SELF-HEALING IV STORE has
    # never written a single row. That is why _estimate_iv_rank's freshness
    # gate keeps reporting a stale series and every iv_rank falls back to the
    # realized-vol proxy. Fix is `c.atm_iv` (drop the parens);
    # DELETE THIS ENTRY the moment it lands, so a regression goes red.
    "daily_iv_persist_failed",
}


class LogSpy:
    """Wraps the orchestrator's structlog logger and records every call.

    Forwards to the real logger so nothing about production logging changes.
    """

    LEVELS = ("debug", "info", "warning", "error", "critical")

    def __init__(self, inner) -> None:
        self._inner = inner
        self.events: list[tuple[str, str, dict]] = []
        for level in self.LEVELS:
            setattr(self, level, self._make(level))

    def _make(self, level: str):
        def _log(event: str = "", **kw):
            self.events.append((level, str(event), kw))
            return getattr(self._inner, level)(event, **kw)
        return _log

    def bind(self, **kw):  # structlog surface the orchestrator may use
        return self

    def names(self) -> list[str]:
        return [e for _, e, _ in self.events]

    def failures(self) -> list[tuple[str, str, dict]]:
        return [
            (lvl, ev, kw) for lvl, ev, kw in self.events
            if any(tok in ev for tok in _FAILURE_TOKENS)
            and ev not in _ALLOWED_FAILURE_EVENTS
        ]

    def assert_clean(self, what: str) -> None:
        bad = self.failures()
        assert not bad, (
            f"{what} swallowed at least one failure — the hot path degraded "
            f"silently (this is the 3-day-outage signature):\n  "
            + "\n  ".join(f"[{lvl}] {ev} {kw}" for lvl, ev, kw in bad)
        )

    def reset(self) -> None:
        self.events.clear()


# ---------------------------------------------------------------------------
# Boundary fakes (I/O only — everything inside the orchestrator stays real)
# ---------------------------------------------------------------------------

def _price_frame(rows: int = 320, start: float = SPOT) -> pd.DataFrame:
    """A deterministic OHLCV frame that survives DataQualityValidator."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    rng = np.random.default_rng(20260814)
    close = start * np.exp(np.cumsum(rng.normal(0.0002, 0.008, rows)))
    high = close * (1 + np.abs(rng.normal(0.0, 0.003, rows)))
    low = close * (1 - np.abs(rng.normal(0.0, 0.003, rows)))
    open_ = (high + low) / 2
    vol = rng.integers(60_000_000, 90_000_000, rows)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class FakeMarketData:
    """Stands in for MarketDataService — the whole network boundary.

    A plain class (not MagicMock): an attribute the orchestrator invents
    raises AttributeError here, which is the point.
    """

    def __init__(self) -> None:
        self.historical_calls: list[tuple[str, int]] = []
        self.vix_calls = 0
        self.price_calls: list[str] = []

    async def get_historical(self, symbol: str, days: int = 60, **kw):
        self.historical_calls.append((symbol, days))
        return _price_frame()

    async def get_intraday(self, symbol: str, interval: str = "5m", days: int = 7):
        return None

    async def get_intraday_since(self, symbol: str, since=None):
        return None

    async def get_vix(self) -> float:
        self.vix_calls += 1
        return VIX

    async def get_current_price(self, symbol: str) -> float:
        self.price_calls.append(symbol)
        return SPOT

    async def get_quote(self, symbol: str):
        return None


def _contract(right: str, strike: float, expiry: date, delta: float) -> OptionContract:
    mid = max(0.35, 6.0 - abs(strike - SPOT) * 0.10)
    return OptionContract(
        symbol=SYMBOL, expiry=expiry, strike=strike, right=right,
        bid=round(mid - 0.05, 2), ask=round(mid + 0.05, 2), last=round(mid, 2),
        volume=2_500, open_interest=9_000, implied_vol=0.18,
        delta=delta, gamma=0.01, theta=-0.05, vega=0.10,
        con_id=int(strike * 10) + (1 if right == "C" else 2),
        source="yahoo_delayed",
    )


def _chain(expiry: date | None = None) -> OptionsChain:
    """A REAL OptionsChain — filter_liquid / atm_iv / _persist_daily_iv run for real."""
    expiry = expiry or (date.today() + timedelta(days=21))
    strikes = [SPOT + i * 5.0 for i in range(-8, 9)]
    calls, puts = [], []
    for k in strikes:
        moneyness = (k - SPOT) / SPOT
        calls.append(_contract("C", k, expiry, delta=max(0.02, 0.50 - moneyness * 8)))
        puts.append(_contract("P", k, expiry, delta=min(-0.02, -0.50 - moneyness * 8)))
    return OptionsChain(symbol=SYMBOL, underlying_price=SPOT, expiry=expiry,
                        calls=calls, puts=puts, source="yahoo_delayed")


class FakeOptionsChainService:
    """Stands in for OptionsChainService (IBKR/Yahoo chain fetch)."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    async def get_chain(self, symbol: str, **kw) -> list[OptionsChain]:
        self.requested.append(symbol)
        return [_chain()]


class FakeMacroFetcher:
    """Stands in for MacroDataFetcher — _build_market_context otherwise makes a
    live FRED request (verified: the first run of this file fetched 4 series)."""

    def __init__(self) -> None:
        self.calls = 0

    async def fetch_all(self, lookback_days: int = 365) -> dict:
        self.calls += 1
        return {}


class FakePrediction:
    def __init__(self, direction: SignalDirection, confidence: float) -> None:
        self.direction = direction
        self.confidence = confidence
        self.probability_up = confidence
        self.model_version = "smoke-1"


class FakePredictor:
    """Stands in for the trained DirectionPredictor (models aren't in the sandbox)."""

    model_version = "smoke-1"
    is_trained = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, hist, symbol: str = "", **kw) -> FakePrediction:
        self.calls.append(symbol)
        # Above risk.min_confidence so the scan runs the FULL path rather than
        # the neutral-only short circuit.
        return FakePrediction(SignalDirection.NEUTRAL, 0.72)


class SmokeIBKRClient(FakeIBKRClient):
    """tests/fakes.FakeIBKRClient + the account/reconcile surface the hot path uses.

    Deliberately explicit: anything the orchestrator calls that is NOT defined
    here raises AttributeError instead of silently returning a mock.
    """

    async def get_account_values(self) -> dict[str, str]:
        return {
            "NetLiquidation": "100000", "BuyingPower": "200000",
            "AvailableFunds": "95000", "ExcessLiquidity": "95000",
            "MaintMarginReq": "5000", "InitMarginReq": "5000",
            "UnrealizedPnL": "0", "RealizedPnL": "0", "CashBalance": "100000",
        }

    async def verify_can_trade(self) -> bool:
        return True

    async def get_positions_fresh(self) -> list:
        return []

    async def get_all_open_trades(self) -> list:
        return []


def iron_condor_signal(confidence: float = 0.72) -> Signal:
    """A structurally realistic SPY iron condor (credit, defined risk).

    Public because scripts/smoke_deploy.py drives the same signal through
    _try_execute as its deploy gate.
    """
    expiry = date.today() + timedelta(days=21)
    short_put = _contract("P", SPOT - 20, expiry, -0.20)
    long_put = _contract("P", SPOT - 25, expiry, -0.12)
    short_call = _contract("C", SPOT + 20, expiry, 0.20)
    long_call = _contract("C", SPOT + 25, expiry, 0.12)
    return Signal(
        symbol=SYMBOL,
        strategy_name="iron_condor",
        direction=SignalDirection.NEUTRAL,
        confidence=confidence,
        contract=short_put,
        action="SELL",
        quantity=1,
        legs=[
            {"contract": long_put, "action": "BUY", "ratio": 1},
            {"contract": short_put, "action": "SELL", "ratio": 1},
            {"contract": short_call, "action": "SELL", "ratio": 1},
            {"contract": long_call, "action": "BUY", "ratio": 1},
        ],
        entry_price=1.20,          # credit collected
        max_loss=380.0,
        max_profit=120.0,
        iv_rank=45.0,
        underlying_price=SPOT,
        expiry=expiry,
    )


# ---------------------------------------------------------------------------
# The orchestrator under test
# ---------------------------------------------------------------------------

class SmokeRig:
    """Everything a caller needs to drive and inspect one real orchestrator."""

    def __init__(self, orch, spy, market, chains, predictor, ibkr, notes, executed):
        self.orch = orch
        self.log = spy
        self.market = market
        self.chains = chains
        self.predictor = predictor
        self.ibkr = ibkr
        self.notifications = notes
        self.executed = executed  # signals handed to executor.execute_signal
        self._real_log = spy._inner

    async def drain_notifications(self) -> None:
        """_send_notification is fire-and-forget; let its tasks finish."""
        for _ in range(3):
            await asyncio.sleep(0)
        tasks = getattr(self.orch, "_notify_tasks", None)
        if tasks:
            await asyncio.wait(set(tasks), timeout=5)

    def restore(self) -> None:
        """Undo the module-level log patch (the only global this rig touches)."""
        import ait.bot.orchestrator as orch_mod
        orch_mod.log = self._real_log


def build_smoke_orchestrator(config_path: Path | str = CONFIG_PATH) -> SmokeRig:
    """Construct a REAL TradingOrchestrator against fakes, in the CURRENT cwd.

    A plain function, not a fixture, so scripts/smoke_deploy.py's deploy gate
    can drive the identical rig — the gate and this test must never drift
    apart, because a gate built on different fakes proves something different
    from what the suite proves.

    CALLER CONTRACT: cwd must already be a throwaway sandbox directory. The
    real ``__init__`` creates data/ait_state.db, data/historical.db, the
    duckdb analytics store and the learning/thompson JSON, all cwd-relative;
    running this with cwd on the repo would write next to the live bot.
    Call ``rig.restore()`` when done.
    """
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 1. REAL settings from the REAL config.yaml. This is what gives the smoke
    #    its teeth: a wrong attribute path raises AttributeError instead of
    #    conjuring a mock. Only the scan universe is narrowed, for runtime.
    settings = load_settings(config_path)
    settings.trading.universe = [SYMBOL]

    ibkr = SmokeIBKRClient(FakeIB())

    # 2. THE REAL __init__ (zero coverage before this file).
    orch = TradingOrchestrator(settings, ibkr)

    # 3. Swap ONLY the I/O boundary. Everything between stays real: risk
    #    manager, circuit breaker, PDT guard, portfolio, executor, state,
    #    scheduler, learning adaptor, watchdog, thompson, counterfactual.
    market = FakeMarketData()
    chains = FakeOptionsChainService()
    predictor = FakePredictor()
    orch._market_data = market
    orch._options_chain = chains
    orch._predictor = predictor
    orch._portfolio._market_data = market
    orch._trainer._market_data = market
    # _build_market_context lazily builds this one; pre-seed it so no FRED
    # request is ever made.
    orch._macro_fetcher = FakeMacroFetcher()

    # Wall-clock / calendar / third-party-feed boundaries pinned so the smoke
    # is deterministic at any time of day and on any date.
    orch._scheduler.should_avoid_new_trades = lambda: False
    orch._earnings.is_near_earnings = lambda *a, **k: False
    orch._earnings.would_hold_through_earnings = lambda *a, **k: False
    orch._economic_cal.days_until_next_event = lambda: 30
    orch._economic_cal.should_skip_trading = lambda: False

    async def _no_filings(symbols):
        return []
    orch._edgar.check_for_material_events = _no_filings

    # The post-restart settling guard would otherwise short-circuit every entry.
    orch._started_at = datetime.now() - timedelta(hours=2)

    # Executor: real object, but the ORDER PLACEMENT call is captured. Nothing
    # may reach a broker from a smoke run.
    executed: list[tuple] = []

    async def _execute_signal(signal, contracts):
        executed.append((signal, contracts))
        return f"T-SMOKE-{len(executed)}"
    orch._executor.execute_signal = _execute_signal

    # Notifications: captured locally, never sent.
    notes: list[str] = []

    async def _notify(msg: str) -> bool:
        notes.append(msg)
        return True
    orch.set_notification_callback(_notify)

    # 4. The log spy (see module docstring, rule 2).
    import ait.bot.orchestrator as orch_mod
    spy = LogSpy(orch_mod.log)
    orch_mod.log = spy

    return SmokeRig(orch, spy, market, chains, predictor, ibkr, notes, executed)


@pytest.fixture
def rig(tmp_path, monkeypatch) -> SmokeRig:
    # Sandbox FIRST: nothing may be constructed while cwd is the repo.
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


# ---------------------------------------------------------------------------
# Construction — the real __init__ had ZERO coverage before this file
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_real_init_wires_every_hot_path_collaborator(self, rig):
        orch = rig.orch
        for attr in ("_settings", "_ibkr", "_account", "_market_data",
                     "_options_chain", "_historical", "_state",
                     "_circuit_breaker", "_risk_manager", "_executor",
                     "_portfolio", "_scheduler", "_reconciler", "_learning",
                     "_watchdog", "_thompson", "_counterfactual",
                     "_capital_tiers", "_strategy_selector", "_analytics"):
            assert getattr(orch, attr, None) is not None, f"__init__ left {attr} unset"
        # The alert routes __init__ wires up (both were silently unwired in
        # past incidents).
        assert orch._portfolio._notify_cb is not None
        assert orch._account._notify_cb is not None
        assert orch._account._circuit_breaker is not None

    def test_executor_receives_settings_spread_ceiling(self, rig):
        """PR#7 merge validation (2026-08-25): the ``settings=`` argument in
        orchestrator __init__'s TradeExecutor construction is the hunk that
        arms the executor's config-bound single-leg spread ceiling (0.40)
        over the 0.15 code default — the one risk-LOOSENING live change in
        the R19c-R21b arc, and the validation proved reverting just that
        wiring kept the entire suite green. This pins it: construct
        TradeExecutor without settings again and the ceiling silently falls
        back to DEFAULT_MAX_SPREAD_PCT, failing here."""
        from ait.execution.executor import DEFAULT_MAX_SPREAD_PCT
        expected = float(rig.orch._settings.options.max_bid_ask_spread_pct)
        assert rig.orch._executor._max_spread_pct == expected
        # Teeth check: config must differ from the code default, or this test
        # cannot tell wired from unwired. If you deliberately set
        # options.max_bid_ask_spread_pct to exactly 0.15, pick a nearby value
        # (e.g. 0.16) or rework this test — do not delete it.
        assert expected != DEFAULT_MAX_SPREAD_PCT

    def test_construction_is_sandboxed_away_from_the_live_databases(self, rig):
        """A LIVE BOT MAY BE RUNNING. Nothing here may touch the repo's data/."""
        state_db = Path(rig.orch._state._db_path).resolve()
        assert state_db.is_relative_to(Path.cwd()), state_db
        assert state_db != (REPO_ROOT / "data" / "ait_state.db").resolve()
        assert state_db.exists(), "the sandbox state DB was never created"
        # ...and the analytics store went with it (StateManager derives the
        # duck path from db_path — the fixture-pollution fix of 2026-07-11).
        duck = getattr(getattr(rig.orch._state, "_duck", None), "_db_path", None)
        if duck is not None:
            assert Path(duck).resolve().is_relative_to(Path.cwd()), duck


# ---------------------------------------------------------------------------
# The hot path, awaited for real
# ---------------------------------------------------------------------------

class TestTradingCycle:
    async def test_full_cycle_runs_and_reaches_the_scan_stage(self, rig):
        orch = rig.orch

        await orch._trading_cycle()
        await rig.drain_notifications()

        # (a) nothing swallowed
        rig.log.assert_clean("_trading_cycle")

        # (b) it did MEANINGFUL work — a function that returned early cannot
        #     satisfy all of these.
        events = rig.log.names()
        assert "capital_tier_active" in events, (
            "cycle never reached the capital-tier/universe stage — it returned "
            "early (circuit breaker? close-to-close? hour block?)")
        assert orch._state.get_state("last_net_liquidation", "") != "", (
            "cycle never published NLV — the account/tier stage did not run")
        assert rig.market.vix_calls >= 1, "cycle never fetched VIX"
        assert (SYMBOL, 504) in rig.market.historical_calls, (
            "_scan_symbol was never reached from _trading_cycle")
        assert rig.chains.requested == [SYMBOL], (
            "the options chain was never requested — the scan stopped before "
            f"strategy generation (chain requests: {rig.chains.requested})")
        assert "scan_symbol_timing" in events, "scan never completed a chain pass"

    async def test_cycle_halts_cleanly_when_the_breaker_is_tripped(self, rig):
        """The other real branch of the cycle: tripped => notify once, return."""
        orch = rig.orch
        orch._circuit_breaker._trip("smoke test trip", pause_seconds=600)
        assert orch._circuit_breaker.is_tripped

        await orch._trading_cycle()
        await rig.drain_notifications()

        rig.log.assert_clean("_trading_cycle (tripped)")
        assert "trading_halted" in rig.log.names()
        assert any("CIRCUIT BREAKER" in n for n in rig.notifications)
        assert not rig.chains.requested, "a tripped breaker must not scan"


class TestMonitorPositionsFast:
    async def test_fast_monitor_runs_the_whole_30s_tick(self, rig):
        orch = rig.orch
        # Force the every-10th-tick material-event branch to run too.
        orch._edgar_check_count = 9

        await orch._monitor_positions_fast()
        await rig.drain_notifications()

        rig.log.assert_clean("_monitor_positions_fast")

        # The mark-to-market daily-loss brake is wrapped in its own try/except
        # (it logs mtm_check_failed and keeps going), so the ONLY proof it ran
        # is its side effect: the start-of-day unrealized baseline.
        from ait.utils.time import now_et
        sod_key = f"mtm_sod_{now_et().date().isoformat()}"
        assert orch._state.get_state(sod_key, "") != "", (
            "the MTM daily-loss brake did not evaluate — it failed silently "
            "inside its own exception handler")
        assert orch._edgar_check_count == 0, (
            "the 8-K material-event branch never ran")
        # A completed tick feeds the watchdog; a swallowed error would not.
        assert orch._watchdog.get_health() is not None


class TestScanSymbol:
    async def test_scan_symbol_walks_the_full_entry_pipeline(self, rig):
        orch = rig.orch
        orch._risk_budget = 3000.0
        collected: list = []

        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)

        rig.log.assert_clean("_scan_symbol")
        events = rig.log.names()
        assert rig.predictor.calls == [SYMBOL], "the ML predictor was never called"
        assert "ml_prediction" in events
        assert rig.chains.requested == [SYMBOL], "options chain never fetched"
        assert "scan_symbol_timing" in events, (
            "the scan never reached the per-chain signal-generation stage")

    async def test_scan_symbol_collects_candidates_for_ranking(self, rig, monkeypatch):
        """The collect-mode contract _trading_cycle depends on.

        Signal GENERATION itself is the strategies' own test surface; here it
        is pinned so the orchestrator's ranking/execute handoff is exercised
        deterministically regardless of what a synthetic chain happens to
        produce.
        """
        orch = rig.orch
        sig = iron_condor_signal()
        monkeypatch.setattr(orch._strategy_selector, "generate_all_signals",
                            lambda **kw: [sig])
        collected: list = []

        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)

        rig.log.assert_clean("_scan_symbol (collect)")
        assert collected, "no candidate collected — the ranking pass gets nothing"
        score, out_sig, eff_conf, sentiment, regime = collected[0]
        assert out_sig is sig
        assert isinstance(score, float) and score > 0
        assert 0.0 <= eff_conf <= 1.0
        assert regime is not None, "regime must travel with the candidate"


class TestTryExecute:
    async def test_try_execute_reaches_risk_validation_and_the_executor(self, rig):
        orch = rig.orch
        validated: list = []
        real_validate = orch._risk_manager.validate_trade

        async def _spy_validate(request):
            validated.append(request)
            return await real_validate(request)
        orch._risk_manager.validate_trade = _spy_validate

        regime = orch._regime_detector.analyze(_price_frame(), VIX)
        handled = await orch._try_execute(iron_condor_signal(), 0.72, None, regime)
        await rig.drain_notifications()

        rig.log.assert_clean("_try_execute")
        assert isinstance(handled, bool)
        assert validated, (
            "_try_execute never reached risk validation — an entry gate "
            f"short-circuited it (events: {rig.log.names()})")
        assert validated[0].strategy == "iron_condor"
        assert validated[0].vix == VIX, "VIX never reached the risk request"

        # Risk approves this trade (1-lot defined-risk condor, $380 max loss,
        # $100k NLV, no open positions, VIX 16), so the executor and the whole
        # post-execute bookkeeping block must have run.
        assert rig.executed, (
            "risk approved but the executor was never called — the entry "
            f"pipeline broke after validation (reason: {rig.log.names()})")
        signal, contracts = rig.executed[0]
        assert contracts >= 1
        trade_id = "T-SMOKE-1"
        assert orch._state.get_state(f"trade_maxloss_{trade_id}", "") != "", (
            "capital-at-risk was never persisted for the new trade")
        assert orch._state.get_daily_stats().trades_taken == 1
        assert any("TRADE:" in n for n in rig.notifications)

    async def test_try_execute_halt_file_blocks_entries(self, rig, tmp_path):
        """The operator kill switch — a real branch, real file, real path."""
        (tmp_path / "data" / "HALT").write_text("smoke")
        handled = await rig.orch._try_execute(iron_condor_signal(), 0.72, None, None)
        assert handled is True, "HALT must report the symbol as handled"
        assert not rig.executed, "HALT file did not stop the entry"
        assert "entries_halted" in rig.log.names()


class TestPostMarket:
    async def test_post_market_runs_end_to_end(self, rig):
        orch = rig.orch

        await orch._post_market()
        await rig.drain_notifications()

        rig.log.assert_clean("_post_market")
        assert "post_market_complete" in rig.log.names(), (
            "_post_market never reached its final report")
        assert any("DAILY SUMMARY" in n for n in rig.notifications), (
            "the daily summary was never produced")
        assert any("EOD RECON" in n for n in rig.notifications), (
            "the EOD reconcile break report never ran")
        # The next session's MTM baseline must be pre-stamped at the close.
        from ait.utils.time import next_market_open
        key = f"mtm_sod_{next_market_open().date().isoformat()}"
        assert orch._state.get_state(key, "") != "", (
            "tomorrow's MTM baseline was not pre-stamped")


# ---------------------------------------------------------------------------
# The guard on the guard: prove the smoke actually fails on the bug class
# ---------------------------------------------------------------------------

class TestSmokeDetectsTheBugClass:
    """If this passes, an AttributeError in the hot path CANNOT ship green.

    Reproduces the 2026-08 incident shape: an attribute read in the middle of
    _trading_cycle that no longer resolves. The real cycle catches nothing
    around it, so the call raises; this test asserts it does.
    """

    async def test_attributeerror_in_trading_cycle_is_caught(self, rig, monkeypatch):
        orch = rig.orch

        class _NoStrategies:
            """settings.options with the `strategies` attribute removed."""

            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                if name == "strategies":
                    raise AttributeError(
                        "'OptionsConfig' object has no attribute 'strategies'")
                return getattr(self._real, name)

        monkeypatch.setattr(orch._settings, "options", _NoStrategies(orch._settings.options))

        with pytest.raises(AttributeError, match="strategies"):
            await orch._trading_cycle()

    async def test_swallowed_failure_is_reported_by_the_log_spy(self, rig, monkeypatch):
        """And a fault inside a try/except (the invisible kind) fails too."""
        orch = rig.orch

        def _boom(*a, **k):
            raise KeyError("mtm")
        monkeypatch.setattr(orch._state, "get_daily_stats", _boom)

        await orch._monitor_positions_fast()   # swallows the KeyError by design

        with pytest.raises(AssertionError, match="swallowed"):
            rig.log.assert_clean("_monitor_positions_fast")
