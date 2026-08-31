"""W2: orchestrator entry-gate FAIL DIRECTIONS plus the two relationship
defects that sit on the same gates.

Every test EXECUTES the real method under test (real TradingOrchestrator via
__new__ with only the attributes that method reads, real ModelTrainer, real
sqlite on a tmp file, real NYSE calendar). inspect.getsource is NOT used
anywhere in this file: a structural assertion cannot tell a gate that blocks
from a gate that logs and proceeds, which is exactly the defect class here.

Findings covered:
  fail-direction-05  _estimate_iv_rank fabricated 50.0 on ANY failure — the
                     one value that clears BOTH IV gates (iron_condor's >=15
                     floor and the risk manager's <=85 VRP cap) — and
                     _scan_symbol's gather laundered raised exceptions into
                     it with no log at all, journalling the fabrication as a
                     real entry_iv_rank measurement.
  fail-direction-02  with ml.entry_gates_enabled ON and no range prediction
                     available, iron_condor/short_strangle kept their
                     DIRECTIONAL confidence, so the payoff-matched gate
                     inverted by regime (silent drop on neutral days, entry
                     on directional evidence on trending days).
  fail-direction-01  the VIX-outage full entry stop was correctly fail-closed
                     but SILENT — 100% of entries refused with no page.
  numeric-pairs-02   the post-stop cooldown was a flat 30 wall-clock hours;
                     the R12-B4 spec is ONE TRADING day, so every Friday and
                     pre-holiday stop bought zero post-stop sessions.
  units-scale-05     the live range spec (0.05, 30) diverged from the research
                     authority (walkforward._range_label_horizon -> 9) that
                     calibrated the live 0.65 confidence floor.
  trade-life-entry-vix-refetch-defeats-lkg
                     save_trade_context re-fetched VIX independently and wrote
                     entry_vix=0 permanently on a transient failure, with the
                     validated value sitting unused in scope.

Pre-fix behaviour these would have caught: every "skipped"/"blocked" assertion
below was a scan that proceeded on fabricated data, every page assertion was
silence, every Friday-stop assertion was a re-entry, and entry_vix was 0.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

import ait.bot.orchestrator as orch_mod
from ait.bot.orchestrator import TradingOrchestrator
from ait.ml.range_spec import live_range_spec


# --------------------------------------------------------------------- tools
class LogCapture:
    """Records every structlog call on a module logger, still forwarding to
    the real one. Used instead of caplog because these modules bind their own
    structlog logger at import time."""

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

    def bind(self, **kw):
        return self

    def names(self) -> list[str]:
        return [e for _, e, _ in self.events]

    def kwargs_for(self, event: str) -> list[dict]:
        return [kw for _, e, kw in self.events if e == event]


@pytest.fixture
def spy(monkeypatch) -> LogCapture:
    cap = LogCapture(orch_mod.log)
    monkeypatch.setattr(orch_mod, "log", cap)
    return cap


def _bare_orch() -> TradingOrchestrator:
    """__new__ plus only the attributes the driven method actually reads — the
    established pattern (tests/test_w1_booking_integrity.py)."""
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._send_notification = AsyncMock()
    return o


def _price_frame(rows: int = 300, flat: bool = False) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp("2026-08-28"), periods=rows)
    if flat:
        close = np.full(rows, 100.0)
    else:
        rng = np.random.default_rng(20260831)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, rows)))
    return pd.DataFrame({"Close": close}, index=idx)


class _Market:
    """Market-data boundary stub. A plain class, not MagicMock: an attribute
    the orchestrator invents raises AttributeError here, which is the point."""

    def __init__(self, hist=None, raises: BaseException | None = None,
                 vix_values=None) -> None:
        self._hist = hist
        self._raises = raises
        self._vix_values = list(vix_values or [])
        self.hist_calls = 0
        self.vix_calls = 0

    async def get_historical(self, symbol: str, days: int = 60, **kw):
        self.hist_calls += 1
        if self._raises is not None:
            raise self._raises
        return self._hist

    async def get_vix(self):
        self.vix_calls += 1
        if not self._vix_values:
            return None
        v = self._vix_values.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v


# =====================================================================
# fail-direction-05 — unknown IV rank is None, and entries fail closed
# =====================================================================
class TestIvRankUnknownIsNone:
    """Pre-fix every one of these returned a fabricated 50.0."""

    async def test_proxy_history_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)          # no data/historical.db -> proxy path
        o = _bare_orch()
        o._market_data = _Market(hist=None)
        assert await o._estimate_iv_rank("QQQ") is None

    async def test_proxy_history_too_short_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        o = _bare_orch()
        o._market_data = _Market(hist=_price_frame(rows=20))
        assert await o._estimate_iv_rank("QQQ") is None

    async def test_proxy_exception_returns_none_not_fifty(self, tmp_path, monkeypatch,
                                                          spy):
        # The finder's P10 shape: the proxy fetch sat OUTSIDE any try, so a
        # raised sqlite/network error reached gather and became 50.0.
        monkeypatch.chdir(tmp_path)
        o = _bare_orch()
        o._market_data = _Market(raises=sqlite3.OperationalError("database is locked"))
        assert await o._estimate_iv_rank("QQQ") is None
        assert "iv_rank_unavailable" in spy.names()

    async def test_degenerate_flat_vol_series_returns_none(self, tmp_path, monkeypatch):
        # A frozen series says nothing about IV — it is not "average IV".
        monkeypatch.chdir(tmp_path)
        o = _bare_orch()
        o._market_data = _Market(hist=_price_frame(flat=True))
        assert await o._estimate_iv_rank("QQQ") is None

    async def test_healthy_data_still_returns_a_real_percentile(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.chdir(tmp_path)
        o = _bare_orch()
        o._market_data = _Market(hist=_price_frame())
        rank = await o._estimate_iv_rank("QQQ")
        assert isinstance(rank, float) and 0.0 <= rank <= 100.0


class TestScanFailsClosedOnUnknownIvRank:
    """_scan_symbol must SKIP the symbol for NEW entries. Pre-fix it carried
    the fabricated 50 into signal generation, both IV gates and the journal."""

    def _orch(self, iv_rank_result):
        o = _bare_orch()
        o._learning = SimpleNamespace(adaptor=SimpleNamespace(
            is_symbol_allowed=lambda s: True,
            is_strategy_enabled=lambda s: True,
            is_hour_allowed=lambda h: True,
        ))
        o._market_data = _Market(hist=_price_frame())
        # _historical/_predictor deliberately UNSET: reaching the next stage
        # would raise AttributeError rather than pass quietly.

        async def _iv(symbol):
            if isinstance(iv_rank_result, BaseException):
                raise iv_rank_result
            return iv_rank_result
        o._estimate_iv_rank = _iv
        return o

    async def test_none_iv_rank_skips_the_symbol_and_pages(self, spy):
        o = self._orch(None)
        collected: list = []
        await o._scan_symbol("QQQ", 16.0, market_context={}, collect=collected)
        assert collected == []
        assert "iv_rank_unavailable_entry_skipped" in spy.names()
        o._send_notification.assert_awaited_once()
        assert "QQQ" in o._send_notification.await_args.args[0]

    async def test_raised_iv_rank_skips_the_symbol(self, spy):
        # gather(return_exceptions=True) hands the Exception object back; the
        # old code turned that into 50.0 with no log line at all.
        o = self._orch(sqlite3.OperationalError("database is locked"))
        collected: list = []
        await o._scan_symbol("QQQ", 16.0, market_context={}, collect=collected)
        assert collected == []
        assert "iv_rank_unavailable" in spy.names()
        assert "iv_rank_unavailable_entry_skipped" in spy.names()

    async def test_page_is_throttled_per_symbol(self, spy):
        o = self._orch(None)
        for _ in range(3):
            await o._scan_symbol("QQQ", 16.0, market_context={}, collect=[])
        assert o._send_notification.await_count == 1, "1/hour gate did not hold"
        # ...but a DIFFERENT symbol still pages: the gate is keyed per symbol.
        await o._scan_symbol("SPY", 16.0, market_context={}, collect=[])
        assert o._send_notification.await_count == 2


# =====================================================================
# fail-direction-01 — the VIX outage stop must PAGE, not just log.error
# =====================================================================
class TestVixOutagePages:
    async def test_fail_closed_outage_pages_once_per_hour(self, spy):
        o = _bare_orch()
        o._market_data = _Market(vix_values=[None, None, None])
        assert await o._get_vix_lkg() is None
        o._send_notification.assert_awaited_once()
        msg = o._send_notification.await_args.args[0]
        assert "VIX" in msg and "FAIL-CLOSED" in msg
        # Repeat inside the hour: still fail-closed, but no alert storm.
        assert await o._get_vix_lkg() is None
        assert o._send_notification.await_count == 1
        assert o._vix_fail_streak == 2

    async def test_a_genuinely_high_vix_never_pages(self, spy):
        # Blocking on VIX 42 is a DECISION (the risk manager refuses credit
        # entries downstream), not a data outage — nothing is paged here.
        o = _bare_orch()
        o._market_data = _Market(vix_values=[42.0])
        assert await o._get_vix_lkg() == 42.0
        o._send_notification.assert_not_awaited()
        assert o._vix_fail_streak == 0

    async def test_fresh_last_known_good_does_not_page(self, spy):
        o = _bare_orch()
        o._market_data = _Market(vix_values=[None])
        o._vix_lkg = (18.5, time.time())            # inside the 45-min window
        assert await o._get_vix_lkg() == 18.5
        o._send_notification.assert_not_awaited()

    async def test_expired_last_known_good_pages(self, spy):
        o = _bare_orch()
        o._market_data = _Market(vix_values=[None])
        o._vix_lkg = (18.5, time.time() - 3600)     # older than the window
        assert await o._get_vix_lkg() is None
        o._send_notification.assert_awaited_once()

    async def test_recovery_resets_the_streak(self, spy):
        o = _bare_orch()
        o._market_data = _Market(vix_values=[None, 17.0])
        await o._get_vix_lkg()
        assert o._vix_fail_streak == 1
        assert await o._get_vix_lkg() == 17.0
        assert o._vix_fail_streak == 0


# =====================================================================
# numeric-pairs-02 — post-stop cooldown is ONE TRADING day, not 30 hours
# =====================================================================
FRI = datetime(2026, 8, 28, 10, 0)      # Friday
THU = datetime(2026, 8, 27, 10, 0)      # Thursday
MON = datetime(2026, 8, 31, 9, 35)      # Monday, just after the open
TOUCH = "short_strike_touch (spot 703.86 <= put 704.00)"


def _mk_db(tmp_path, reason: str, exit_time: datetime, symbol: str = "QQQ",
           status: str = "closed") -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    con = sqlite3.connect(tmp_path / "data" / "ait_state.db")
    con.execute("CREATE TABLE IF NOT EXISTS trades (symbol TEXT, status TEXT, "
                "exit_reason_detailed TEXT, exit_time TEXT)")
    con.execute("INSERT INTO trades VALUES (?,?,?,?)",
                (symbol, status, reason, exit_time.isoformat()))
    con.commit()
    con.close()


@pytest.fixture
def pin_clock(monkeypatch):
    """Pin the cooldown's reference instant so these run identically on any
    weekday, and clear the per-date cutoff memo around each pin."""
    def _pin(now: datetime) -> None:
        orch_mod._COOLDOWN_CUTOFF_CACHE.clear()
        monkeypatch.setattr(orch_mod, "_cooldown_now", lambda: now)
    yield _pin
    orch_mod._COOLDOWN_CUTOFF_CACHE.clear()


class TestPostStopCooldownIsOneTradingDay:
    def test_friday_stop_blocks_all_of_monday(self, tmp_path, monkeypatch, pin_clock):
        # THE numeric-pairs-02 case: the 30h window expired Saturday 16:00, so
        # the bot re-entered from Monday's open into the same move.
        _mk_db(tmp_path, TOUCH, FRI)
        monkeypatch.chdir(tmp_path)
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        for when in (MON, datetime(2026, 8, 31, 15, 55)):
            pin_clock(when)
            assert o._post_stop_cooldown_until("QQQ") is not None, when

    def test_thursday_stop_allows_monday(self, tmp_path, monkeypatch, pin_clock):
        # Monday's previous session is Friday; a Thursday stop has already
        # served its full trading day and must NOT block.
        _mk_db(tmp_path, TOUCH, THU)
        monkeypatch.chdir(tmp_path)
        pin_clock(MON)
        assert TradingOrchestrator.__new__(
            TradingOrchestrator)._post_stop_cooldown_until("QQQ") is None

    def test_same_day_stop_blocks(self, tmp_path, monkeypatch, pin_clock):
        # The live 08-24 QQQ shape: touch stop 09:47, re-entry attempt 10:00.
        _mk_db(tmp_path, TOUCH, datetime(2026, 8, 31, 9, 47))
        monkeypatch.chdir(tmp_path)
        pin_clock(datetime(2026, 8, 31, 10, 0))
        assert TradingOrchestrator.__new__(
            TradingOrchestrator)._post_stop_cooldown_until("QQQ") is not None

    def test_profit_exits_never_block(self, tmp_path, monkeypatch, pin_clock):
        # trailing/breakeven stops fire at or above breakeven, so re-entry
        # after them is not the autocorrelated-loss sequence R12-B4 targets.
        for reason in ("take_profit_short (P&L: 51.0%)",
                       "trailing_stop (P&L: 12.0%, peak: 30.0%, stop: 10.0%)",
                       "breakeven_stop (P&L: 0.5%, peak: 22.0%)"):
            _mk_db(tmp_path, reason, datetime(2026, 8, 31, 9, 47))
        monkeypatch.chdir(tmp_path)
        pin_clock(datetime(2026, 8, 31, 10, 0))
        assert TradingOrchestrator.__new__(
            TradingOrchestrator)._post_stop_cooldown_until("QQQ") is None

    def test_midweek_blocks_the_next_session_then_expires(self, tmp_path, monkeypatch,
                                                          pin_clock):
        _mk_db(tmp_path, "stop_loss (P&L: -38.0%)", datetime(2026, 9, 1, 10, 0))  # Tue
        monkeypatch.chdir(tmp_path)
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        pin_clock(datetime(2026, 9, 2, 15, 0))     # Wed — still one trading day
        assert o._post_stop_cooldown_until("QQQ") is not None
        pin_clock(datetime(2026, 9, 3, 9, 35))     # Thu — served
        assert o._post_stop_cooldown_until("QQQ") is None

    def test_market_holiday_extends_the_window(self, tmp_path, monkeypatch, pin_clock):
        # Labor Day Monday 2026-09-07 is not a session, so Tuesday's previous
        # session is Friday 09-04 and a Friday stop still blocks on Tuesday.
        _mk_db(tmp_path, TOUCH, datetime(2026, 9, 4, 10, 0))
        monkeypatch.chdir(tmp_path)
        pin_clock(datetime(2026, 9, 8, 10, 0))
        assert TradingOrchestrator.__new__(
            TradingOrchestrator)._post_stop_cooldown_until("QQQ") is not None

    def test_other_symbols_unaffected(self, tmp_path, monkeypatch, pin_clock):
        _mk_db(tmp_path, TOUCH, FRI)
        monkeypatch.chdir(tmp_path)
        pin_clock(MON)
        assert TradingOrchestrator.__new__(
            TradingOrchestrator)._post_stop_cooldown_until("SPY") is None

    def test_cutoff_is_the_previous_sessions_open(self):
        # The cutoff itself, executed: 09:30 on the latest session strictly
        # before the reference date, naive-local like the DB's exit_time rows.
        orch_mod._COOLDOWN_CUTOFF_CACHE.clear()
        try:
            assert orch_mod._post_stop_cooldown_cutoff(MON) == datetime(2026, 8, 28, 9, 30)
            assert orch_mod._post_stop_cooldown_cutoff(
                datetime(2026, 9, 8, 10, 0)) == datetime(2026, 9, 4, 9, 30)
        finally:
            orch_mod._COOLDOWN_CUTOFF_CACHE.clear()


# =====================================================================
# units-scale-05 — ONE authority for the live range spec
# =====================================================================
class TestLiveRangeSpecAuthority:
    def test_horizon_is_the_reachable_trade_horizon(self):
        from ait.config.settings import load_settings
        from ait.execution.exit_policy import EXPIRY_APPROACHING_DTE
        settings = load_settings()
        threshold, horizon = live_range_spec(settings)
        assert threshold == 0.05
        assert horizon == max(
            1, int(settings.options.dte_range[0]) - int(EXPIRY_APPROACHING_DTE))

    def test_live_horizon_equals_the_research_label_horizon(self):
        """The MIRROR units-scale-05 is about: the live gate and the
        walk-forward evidence that calibrated its 0.65 floor must ask the SAME
        question. Pre-fix: live 30 days vs research 9."""
        from ait.backtesting.walkforward import WalkForwardBacktester
        wf = WalkForwardBacktester(symbols=["SPY"], strategies=["iron_condor"])
        assert live_range_spec()[1] == wf._range_label_horizon()

    def test_spec_follows_the_configured_dte_band(self):
        # A future dte_range edit must move BOTH sides; no literal survives.
        from ait.execution.exit_policy import EXPIRY_APPROACHING_DTE
        fake = SimpleNamespace(options=SimpleNamespace(dte_range=[21, 45]))
        assert live_range_spec(fake)[1] == 21 - EXPIRY_APPROACHING_DTE
        tiny = SimpleNamespace(options=SimpleNamespace(dte_range=[3, 10]))
        assert live_range_spec(tiny)[1] == 1, "horizon must never go below 1 day"


class TestRangeModelSpecHonesty:
    """A LOADED model keeps its trained spec — its probabilities only answer
    that question — but a stale artifact must be VISIBLE at startup."""

    def test_mismatched_artifact_warns(self, spy):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._range_predictor = SimpleNamespace(
            is_trained=True, _threshold=0.05, _horizon=30,
            model_version="range-v-20260826-073033", trained_at="2026-08-26")
        assert o._check_range_model_spec(0.05, 9) is True
        assert "range_model_spec_mismatch" in spy.names()
        kw = spy.kwargs_for("range_model_spec_mismatch")[0]
        assert kw["live_horizon_days"] == 9 and kw["model_horizon_days"] == 30

    def test_matching_artifact_is_silent(self, spy):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._range_predictor = SimpleNamespace(
            is_trained=True, _threshold=0.05, _horizon=9,
            model_version="v", trained_at="2026-08-26")
        assert o._check_range_model_spec(0.05, 9) is False
        assert "range_model_spec_mismatch" not in spy.names()

    def test_untrained_predictor_is_not_a_spec_mismatch(self, spy):
        # Nothing is loaded, so there is no artifact to be honest about; the
        # untrained state is fail-direction-02's business, not this check's.
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._range_predictor = SimpleNamespace(
            is_trained=False, _threshold=0.05, _horizon=30)
        assert o._check_range_model_spec(0.05, 9) is False
        assert "range_model_spec_mismatch" not in spy.names()


class TestTrainerReadsTheSameAuthority:
    def _trainer(self, threshold, horizon):
        from ait.config.settings import MLConfig
        from ait.ml.trainer import ModelTrainer
        rp = SimpleNamespace(_spec_threshold=threshold, _spec_horizon=horizon)
        return ModelTrainer(MLConfig(), predictor=None, market_data=None,
                            historical_store=None, range_predictor=rp)

    def test_divergent_designed_spec_is_flagged(self, monkeypatch):
        # train() always rebuilds at the DESIGNED spec, so a predictor built
        # from anywhere but the authority re-establishes the divergence.
        import ait.ml.trainer as trainer_mod
        cap = LogCapture(trainer_mod.log)
        monkeypatch.setattr(trainer_mod, "log", cap)
        live_threshold, live_horizon = live_range_spec()
        trainer = self._trainer(live_threshold, live_horizon + 21)
        assert trainer.check_range_spec() is False
        assert "range_train_spec_mismatch" in cap.names()

    def test_authoritative_designed_spec_passes(self, monkeypatch):
        import ait.ml.trainer as trainer_mod
        cap = LogCapture(trainer_mod.log)
        monkeypatch.setattr(trainer_mod, "log", cap)
        assert self._trainer(*live_range_spec()).check_range_spec() is True
        assert "range_train_spec_mismatch" not in cap.names()


# =====================================================================
# The remaining two findings need the WHOLE entry pipeline, so they drive
# the hot-path smoke's REAL orchestrator (real __init__, real risk manager,
# real state; fakes only at the I/O boundary). build_smoke_orchestrator is
# public for exactly this reason — scripts/smoke_deploy.py drives it too.
# =====================================================================
@pytest.fixture
def rig(tmp_path, monkeypatch):
    from tests.test_hot_path_smoke import build_smoke_orchestrator
    monkeypatch.chdir(tmp_path)     # sandbox FIRST: __init__ writes data/*.db
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


class _RangeStub:
    """Stands in for RangePredictor at the model boundary (the smoke rig
    already fakes DirectionPredictor for the same reason: artifacts are not
    in the sandbox)."""

    def __init__(self, trained: bool = True, probability: float | None = None) -> None:
        self.is_trained = trained
        self._probability = probability
        self.calls = 0

    def predict(self, hist, **kw):
        from ait.ml.range_predictor import RangePrediction
        self.calls += 1
        if self._probability is None:
            return None          # the routine None paths: no-edge gate, empty
        return RangePrediction(                       # features, scaling failure
            probability_in_range=self._probability,
            threshold_pct=0.05, horizon_days=9,
            confidence=max(self._probability, 1 - self._probability),
            features_used=42, model_version="stub-1")


async def _scan_one(rig, monkeypatch, signal) -> list:
    """Drive the real _scan_symbol with signal generation pinned, and return
    the candidates it collected."""
    from tests.test_hot_path_smoke import SYMBOL, VIX
    monkeypatch.setattr(rig.orch._strategy_selector, "generate_all_signals",
                        lambda **kw: [signal])
    rig.orch._risk_budget = 3000.0
    collected: list = []
    await rig.orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=collected)
    await rig.drain_notifications()
    return collected


class TestRangeGateFailsClosed:
    """fail-direction-02: with the gates armed, a neutral-premium entry needs
    range evidence in BOTH regimes. Pre-fix, no prediction meant the signal
    kept the DIRECTIONAL model's confidence and was admitted on trending days
    (and silently dropped on neutral ones) — the gate inverted by regime."""

    async def test_untrained_model_blocks_the_condor_and_pages(self, rig, monkeypatch):
        from tests.test_hot_path_smoke import iron_condor_signal
        assert rig.orch._settings.ml.entry_gates_enabled, "gates must be ON here"
        rig.orch._range_predictor = _RangeStub(trained=False)

        collected = await _scan_one(rig, monkeypatch, iron_condor_signal())

        assert collected == [], "an unevidenced iron condor reached the ranking pass"
        assert "range_gate_unavailable_entries_blocked" in rig.log.names()
        assert any("RANGE GATE UNAVAILABLE" in n for n in rig.notifications)

    async def test_predict_returning_none_blocks_the_condor(self, rig, monkeypatch):
        # range_predictor.predict() has routine None paths (no-edge gate at
        # DEBUG, empty features, scaling failure) — trained is not enough.
        from tests.test_hot_path_smoke import iron_condor_signal
        stub = _RangeStub(trained=True, probability=None)
        rig.orch._range_predictor = stub

        collected = await _scan_one(rig, monkeypatch, iron_condor_signal())

        assert stub.calls == 1, "the model was never consulted"
        assert collected == []
        assert "range_gate_unavailable_entries_blocked" in rig.log.names()

    async def test_a_real_prediction_still_admits_the_condor(self, rig, monkeypatch):
        # The control: it is the MISSING evidence that blocks, not the gate.
        from tests.test_hot_path_smoke import iron_condor_signal
        rig.orch._range_predictor = _RangeStub(trained=True, probability=0.80)

        collected = await _scan_one(rig, monkeypatch, iron_condor_signal())

        assert collected, "a range-justified condor must still reach the ranking"
        _score, sig, eff_conf, _sent, _regime = collected[0]
        assert sig.strategy_name == "iron_condor"
        assert eff_conf == pytest.approx(0.80), (
            "the payoff-matched probability must be what risk sizing sees")
        assert "range_gate_unavailable_entries_blocked" not in rig.log.names()

    async def test_low_probability_still_fails_the_existing_floor(self, rig, monkeypatch):
        # Unchanged behaviour: evidence exists and says no (0.30 < 0.65).
        from tests.test_hot_path_smoke import iron_condor_signal
        rig.orch._range_predictor = _RangeStub(trained=True, probability=0.30)

        collected = await _scan_one(rig, monkeypatch, iron_condor_signal())

        assert collected == []
        assert "range_gate_unavailable_entries_blocked" not in rig.log.names(), (
            "a model that answered is not an unavailable model")

    async def test_gates_off_stays_observe_only(self, rig, monkeypatch):
        # R16 contract: with ml.entry_gates_enabled false the ML layer
        # observes and never vetoes — the fail-closed drop must not fire.
        from tests.test_hot_path_smoke import iron_condor_signal
        monkeypatch.setattr(rig.orch._settings.ml, "entry_gates_enabled", False)
        rig.orch._range_predictor = _RangeStub(trained=False)

        collected = await _scan_one(rig, monkeypatch, iron_condor_signal())

        assert collected, "gates-off must not veto — that is a different mode"
        assert "range_gate_unavailable_entries_blocked" not in rig.log.names()

    async def test_a_directional_strategy_is_not_range_gated(self, rig, monkeypatch):
        # Only the neutral-premium pair is paid for containment; a directional
        # signal has its own evidence and must survive a range outage.
        from tests.test_hot_path_smoke import iron_condor_signal
        sig = iron_condor_signal()
        sig.strategy_name = "bull_put_spread"
        rig.orch._range_predictor = _RangeStub(trained=False)

        collected = await _scan_one(rig, monkeypatch, sig)

        assert collected, "a non-range-gated strategy was collateral damage"


class TestEntryVixUsesTheValidatedValue:
    """trade-life-entry-vix-refetch-defeats-lkg: save_trade_context did a
    second, independent, uncached get_vix and `or 0`-ed the result, so one
    hiccup seconds after validation wrote entry_vix=0 permanently."""

    async def test_transient_refetch_failure_no_longer_writes_zero(self, rig):
        orch = rig.orch
        calls = {"n": 0}

        async def _flaky_vix():
            calls["n"] += 1
            if calls["n"] == 1:
                return 16.0
            raise RuntimeError("vix feed down between validation and save")
        orch._market_data.get_vix = _flaky_vix

        from tests.test_hot_path_smoke import iron_condor_signal
        await orch._try_execute(iron_condor_signal(), 0.72, None, None)
        await rig.drain_notifications()

        assert rig.executed, "the trade never executed — precondition failed"
        ctx = orch._state.get_trade_context("T-SMOKE-1")
        assert ctx is not None and ctx["entry_vix"] == pytest.approx(16.0), (
            "the validated VIX was not reused (pre-fix this row was 0.0)")
        assert calls["n"] == 1, "VIX was re-fetched at context-save time"

    async def test_last_known_good_is_reused(self, rig):
        # The R7 LKG is the whole point: the risk gates ran on 14.5, so the
        # journal must record 14.5 — not 0, and not a second network answer.
        orch = rig.orch

        async def _dead_vix():
            return None
        orch._market_data.get_vix = _dead_vix
        orch._vix_lkg = (14.5, time.time())

        from tests.test_hot_path_smoke import iron_condor_signal
        await orch._try_execute(iron_condor_signal(), 0.72, None, None)
        await rig.drain_notifications()

        assert rig.executed
        ctx = orch._state.get_trade_context("T-SMOKE-1")
        assert ctx["entry_vix"] == pytest.approx(14.5)

    async def test_genuinely_unknown_vix_is_written_as_null(self, rig):
        # Nothing fresh, nothing cached. The honest record is NULL: a 0 would
        # be read back as a real measurement and would disarm the vix_spike
        # thesis exit (`entry_vix > 0`) for the position's whole life.
        from ait.risk.manager import TradeValidation
        orch = rig.orch

        async def _dead_vix():
            return None
        orch._market_data.get_vix = _dead_vix

        async def _approve(_request):
            # The credit-entry VIX gate fails closed on vix=None by design
            # (R7); stub only the risk DECISION so the journalling path — the
            # thing under test — is reachable.
            return TradeValidation(approved=True, reason="stubbed",
                                   position_size=1, max_risk=380.0)
        orch._risk_manager.validate_trade = _approve

        from tests.test_hot_path_smoke import iron_condor_signal
        await orch._try_execute(iron_condor_signal(), 0.72, None, None)
        await rig.drain_notifications()

        assert rig.executed
        ctx = orch._state.get_trade_context("T-SMOKE-1")
        assert ctx["entry_vix"] is None, "unknown VIX must be NULL, never 0"
        assert "entry_vix_unknown_stored_null" in rig.log.names()
