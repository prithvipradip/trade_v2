"""W69 — live tunables externalized to config.yaml, two ignored config keys wired.

What this wave moved out of code (every default is the literal it replaced, so
the parity class below is the "no behavior change" contract):

  * position sizer curves            -> position_sizing.*
  * intraday trade budget + gate     -> trading.first_hour_* / midday_*
  * Friday weekend-gap rule          -> risk.weekend_gap_*  (and now_et(), not a
                                        naive datetime.now())
  * thesis-check thresholds          -> exit.thesis_*
  * regime classification            -> regime.*, ml.regime_override_confidence
  * drift detector                   -> ml.drift_*
  * learning analyzer cutoffs        -> learning.analyzer.*
  * capital-tier boundaries          -> account.capital_tier_*_min
  * roll short-delta target/tol      -> assignment_response.roll_short_delta_*

and two config keys that were silently IGNORED are now read:

  * dashboard.port / dashboard.enabled (master.py hardcoded 8501) + the new
    weblog_port / status_port
  * learning.max_adaptations_per_cycle / min_insight_confidence /
    min_confidence_floor / min_confidence_ceiling (AdaptationLimits() defaults
    ran instead)

House rules: every behavioral test drives the real code path (no
inspect.getsource); the class guards are AST walks that catch the NEXT
re-hardcoding; the clock test pins a real non-ET TZ.
"""
from __future__ import annotations

import ast
import time as _time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from ait.config.settings import (
    AccountConfig,
    AssignmentResponseConfig,
    DashboardConfig,
    LearningAnalyzerConfig,
    LearningConfig,
    MLConfig,
    PositionConfig,
    PositionSizingConfig,
    RegimeConfig,
    RiskConfig,
    Settings,
    TradingConfig,
    load_settings,
    resolve_dashboard_port,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"
CONFIG_PATH = ROOT / "config.yaml"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# 1. Parity — the "zero behavior change" contract
# ---------------------------------------------------------------------------

# dotted settings path -> the literal that lived in code before W69
LEGACY = {
    "trading.first_hour_end_et": "10:30",
    "trading.midday_end_et": "14:00",
    "trading.first_hour_trade_budget": 2,
    "trading.midday_trade_budget": 4,
    "trading.first_hour_min_confidence": 0.85,
    "account.capital_tier_small_min": 2_000,
    "account.capital_tier_medium_min": 5_000,
    "account.capital_tier_large_min": 25_000,
    "risk.weekend_gap_cutoff_et": "14:30",
    "risk.weekend_gap_min_confidence": 0.90,
    "risk.weekend_gap_strategies": ["short_strangle"],
    "exit.thesis_direction_flip_confidence": 0.70,
    "exit.thesis_regime_shift_confidence": 0.70,
    "exit.thesis_vix_spike_pct": 0.30,
    "ml.regime_override_confidence": 0.80,
    "ml.drift_window_size": 50,
    "ml.drift_accuracy_threshold": 0.40,
    "ml.drift_calibration_threshold": 0.25,
    "ml.drift_high_confidence": 0.80,
    "assignment_response.roll_short_delta_target": 0.20,
    "assignment_response.roll_short_delta_tolerance": 0.07,
    "dashboard.port": 8501,
    "dashboard.weblog_port": 8502,
    "dashboard.status_port": 8503,
    "dashboard.enabled": True,
    "regime.high_vol_vix": 30,
    "regime.high_vol_realized": 0.40,
    "regime.iv_crush_iv_rank_max": 0.25,
    "regime.iv_crush_vol_max": 0.20,
    "regime.iv_crush_trend_max": 0.20,
    "regime.mean_revert_hurst_max": 0.40,
    "regime.mean_revert_trend_max": 0.25,
    "regime.low_vol_vix": 15,
    "regime.low_vol_realized": 0.12,
    "regime.low_vol_trend_breakout": 0.30,
    "regime.strong_trend": 0.35,
    "position_sizing.confidence_size_floor": 0.30,
    "position_sizing.iv_full_size_max": 0.20,
    "position_sizing.iv_floor_size_min": 0.60,
    "position_sizing.iv_floor_size_factor": 0.50,
    "position_sizing.iv_rank_adj_min": 0.80,
    "position_sizing.iv_rank_adj_max": 1.20,
    "position_sizing.drawdown_tiers": [[3, 0.50], [5, 0.25]],
    "position_sizing.vix_size_tiers": [[25, 0.75], [30, 0.50]],
    "position_sizing.strategy_risk": {
        "long_call": 1.0, "long_put": 1.0, "covered_call": 0.5,
        "cash_secured_put": 0.5, "bull_call_spread": 0.6, "bear_put_spread": 0.6,
        "iron_condor": 0.4, "long_straddle": 1.2, "short_strangle": 0.8,
        "calendar_spread": 0.7, "event_straddle": 0.7,
    },
    "learning.analyzer.strategy_poor_win_rate": 0.35,
    "learning.analyzer.strategy_strong_win_rate": 0.65,
    "learning.analyzer.confidence_band_poor_win_rate": 0.40,
    "learning.analyzer.symbol_poor_win_rate": 0.30,
    "learning.analyzer.regime_poor_win_rate": 0.35,
    "learning.analyzer.hour_poor_win_rate": 0.30,
    "learning.analyzer.exit_giveback_threshold": 0.30,
    "learning.analyzer.exit_min_avg_peak": 0.20,
    "learning.analyzer.direction_peak_threshold": 0.05,
    "learning.analyzer.direction_good_accuracy": 0.60,
    "learning.analyzer.direction_exit_problem_rate": 0.40,
    "learning.analyzer.direction_poor_accuracy": 0.45,
}


def _dig(settings, dotted: str):
    obj = settings
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class TestParity:
    def test_code_defaults_are_the_former_literals(self):
        s = Settings()
        wrong = {k: (_dig(s, k), v) for k, v in LEGACY.items() if _dig(s, k) != v}
        assert not wrong, f"model default != pre-W69 literal: {wrong}"

    def test_shipped_config_yaml_carries_the_same_values(self):
        """config.yaml must not silently change a value while moving it — the
        whole wave is a relocation, so the yaml equals the legacy literal."""
        s = load_settings(str(CONFIG_PATH))
        wrong = {k: (_dig(s, k), v) for k, v in LEGACY.items() if _dig(s, k) != v}
        assert not wrong, f"config.yaml diverges from the pre-W69 literal: {wrong}"

    def test_every_new_field_is_present_in_config_yaml(self):
        """A key that exists only as a model default is invisible to the
        operator — the R19c argument for putting values in the file."""
        import yaml
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        missing = []
        for dotted in LEGACY:
            node = raw
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    missing.append(dotted)
                    break
                node = node[part]
        assert not missing, f"not in config.yaml: {missing}"

    def test_learning_limits_defaults_match_the_adaptor_dataclass(self):
        """The four learning.* keys now feed AdaptationLimits; their defaults
        must agree with the dataclass so a bare construction cannot drift."""
        from ait.learning.adaptor import AdaptationLimits
        lim, cfg = AdaptationLimits(), LearningConfig()
        assert cfg.min_confidence_floor == lim.min_confidence_floor
        assert cfg.min_confidence_ceiling == lim.min_confidence_ceiling
        assert cfg.min_insight_confidence == lim.min_insight_confidence
        assert cfg.max_adaptations_per_cycle == lim.max_adaptations_per_cycle


# ---------------------------------------------------------------------------
# 2. Validators — a bad value is a startup error, not a silently-wrong gate
# ---------------------------------------------------------------------------

class TestValidators:
    @pytest.mark.parametrize("ports", [
        {"port": 8502}, {"weblog_port": 8501}, {"status_port": 8501},
        {"weblog_port": 8503}])
    def test_duplicate_dashboard_ports_are_rejected(self, ports):
        with pytest.raises(ValidationError):
            DashboardConfig(**ports)

    @pytest.mark.parametrize("bad", ["10.30", "25:00", "9:30", "10:60", "", "noon"])
    def test_clock_strings_must_be_hh_mm(self, bad):
        with pytest.raises(ValidationError):
            TradingConfig(first_hour_end_et=bad)
        with pytest.raises(ValidationError):
            RiskConfig(weekend_gap_cutoff_et=bad)

    def test_first_hour_must_end_before_midday(self):
        with pytest.raises(ValidationError):
            TradingConfig(first_hour_end_et="15:00", midday_end_et="14:00")

    def test_capital_tiers_must_ascend_strictly(self):
        with pytest.raises(ValidationError):
            AccountConfig(capital_tier_small_min=6_000, capital_tier_medium_min=5_000)
        with pytest.raises(ValidationError):
            AccountConfig(capital_tier_medium_min=25_000, capital_tier_large_min=25_000)

    def test_sizing_tier_rows_are_validated(self):
        with pytest.raises(ValidationError):
            PositionSizingConfig(vix_size_tiers=[[30, 0.5], [25, 0.75]])   # unsorted
        with pytest.raises(ValidationError):
            PositionSizingConfig(vix_size_tiers=[[25, 0.75, 1.0]])         # arity
        with pytest.raises(ValidationError):
            PositionSizingConfig(drawdown_tiers=[[3, 1.5]])                # factor > 1
        with pytest.raises(ValidationError):
            PositionSizingConfig(iv_floor_size_min=0.10, iv_full_size_max=0.20)

    def test_regime_vol_bands_must_be_ordered(self):
        with pytest.raises(ValidationError):
            RegimeConfig(low_vol_realized=0.5, high_vol_realized=0.4)
        with pytest.raises(ValidationError):
            RegimeConfig(low_vol_vix=35, high_vol_vix=30)

    def test_unknown_key_in_a_new_section_is_a_startup_error(self):
        with pytest.raises(ValidationError):
            PositionSizingConfig(vix_sizing_tiers=[[25, 0.75]])    # typo'd key
        with pytest.raises(ValidationError):
            LearningAnalyzerConfig(strategy_poor_winrate=0.3)


# ---------------------------------------------------------------------------
# 3. Dashboard ports — the supervisor reads config; no literal ports left
# ---------------------------------------------------------------------------

@pytest.fixture
def master_mod(tmp_path, monkeypatch):
    import ait.orchestration.master as m
    monkeypatch.setattr(m, "LOGS_DIR", tmp_path)
    popen_calls: list[list[str]] = []

    class _FakePopen:
        def __init__(self, cmd, *a, **kw):
            popen_calls.append([str(c) for c in cmd])
            self.pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    m._popen_calls = popen_calls
    return m


class TestSupervisorReadsDashboardConfig:
    def test_streamlit_is_launched_on_the_configured_port(self, master_mod, monkeypatch):
        m = master_mod
        cleared: list[int] = []
        monkeypatch.setattr(m, "_clear_dashboard_port",
                            lambda port: cleared.append(port) or True)
        monkeypatch.setattr(m, "_dashboard_cfg", lambda: DashboardConfig(
            port=8611, weblog_port=8612, status_port=8613))
        wsm = m.WebServiceManager()
        wsm._start_dashboard()
        assert cleared == [8611], "the port-clearing step must target the configured port"
        cmd = m._popen_calls[-1]
        assert "--server.port=8611" in cmd
        assert not any("8501" in c for c in cmd)

    def test_log_viewer_and_status_page_receive_their_configured_ports(
            self, master_mod, monkeypatch):
        m = master_mod
        monkeypatch.setattr(m, "_dashboard_cfg", lambda: DashboardConfig(
            port=8611, weblog_port=8612, status_port=8613))
        wsm = m.WebServiceManager()
        wsm._start_weblog()
        wsm._start_status()
        weblog, status = m._popen_calls[-2], m._popen_calls[-1]
        assert weblog[-2:] == ["--port", "8612"] and weblog[1].endswith("web_logs.py")
        assert status[-2:] == ["--port", "8613"] and status[1].endswith("status_server.py")

    def test_dashboard_enabled_false_launches_nothing(self, master_mod, monkeypatch):
        m = master_mod
        cleared: list[int] = []
        monkeypatch.setattr(m, "_clear_dashboard_port",
                            lambda port: cleared.append(port) or True)
        monkeypatch.setattr(m, "_dashboard_cfg",
                            lambda: DashboardConfig(enabled=False))
        wsm = m.WebServiceManager()
        wsm._start_dashboard()
        assert m._popen_calls == [] and cleared == []
        assert wsm._dashboard_proc is None
        # ...and the health check must not "restart" a service that was never up.
        wsm.health_check()
        assert m._popen_calls == []

    def test_a_foreign_port_holder_alert_names_the_configured_port(
            self, master_mod, monkeypatch):
        m = master_mod
        alerts: list[str] = []
        monkeypatch.setattr(m, "_clear_dashboard_port", lambda port: False)
        monkeypatch.setattr(m, "_alert_gate", lambda *a, **k: True)
        monkeypatch.setattr(m, "_alert", lambda msg, *a, **k: alerts.append(msg))
        monkeypatch.setattr(m, "_dashboard_cfg", lambda: DashboardConfig(
            port=8611, weblog_port=8612, status_port=8613))
        m.WebServiceManager()._start_dashboard()
        assert m._popen_calls == []
        assert alerts and "8611" in alerts[0] and "8501" not in alerts[0]

    def test_settings_failure_falls_back_to_model_defaults_not_a_crash(self, monkeypatch):
        import ait.config.settings as settings_mod
        import ait.orchestration.master as m

        def _boom(*a, **k):
            raise RuntimeError("config.yaml unreadable")
        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        assert m._dashboard_cfg() == DashboardConfig()


class TestChildPortResolution:
    def test_explicit_cli_port_wins_without_touching_the_file(self, monkeypatch):
        import ait.config.settings as settings_mod

        def _boom(*a, **k):
            raise AssertionError("must not read config when --port is given")
        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        assert resolve_dashboard_port("weblog_port", 9100) == 9100

    def test_without_cli_port_the_config_value_is_used(self, monkeypatch):
        import ait.config.settings as settings_mod
        cfg = Settings(dashboard=DashboardConfig(weblog_port=9200, status_port=9300))
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: cfg)
        assert resolve_dashboard_port("weblog_port") == 9200
        assert resolve_dashboard_port("status_port") == 9300

    def test_unreadable_config_falls_back_to_the_model_default(self, monkeypatch):
        import ait.config.settings as settings_mod

        def _boom(*a, **k):
            raise RuntimeError("nope")
        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        assert resolve_dashboard_port("weblog_port") == DashboardConfig().weblog_port


# ---------------------------------------------------------------------------
# 4. Sizer — the curves move with config
# ---------------------------------------------------------------------------

def _sizer(**sizing):
    from ait.risk.position_sizer import PositionSizer
    pc = PositionConfig(max_position_pct=0.01, max_contracts_per_trade=100)
    rc = RiskConfig(min_confidence=0.5)
    return PositionSizer(pc, rc, PositionSizingConfig(**sizing))


# Uncapped regime: $1M account (scale cap 100, hard cap 100), a $10,000 budget
# (1%), a $200 contract, neutral IV / IV-rank, confidence 1.0 -> the base is
# 10,000 / 200 = 50 contracts, and each test multiplies in ONE factor.
def _size(sizer, **kw):
    args = dict(account_value=1_000_000, option_price=2.0, confidence=1.0,
                implied_vol=0.10, strategy="long_call", underlying_price=500.0,
                iv_rank=50.0, recent_losing_days=0, vix=0.0)
    args.update(kw)
    return sizer.calculate(**args)


class TestSizerMovesWithConfig:
    def test_vix_tiers(self):
        default = _size(_sizer(), vix=27.0)
        assert default.contracts == 37                        # 50 x 0.75
        custom = _size(_sizer(vix_size_tiers=[[20, 0.5]]), vix=27.0)
        assert custom.contracts == 25                         # 50 x 0.50
        assert _size(_sizer(), vix=0.0).contracts == 50       # below every tier

    def test_vix_tier_uses_the_highest_threshold_reached(self):
        s = _sizer(vix_size_tiers=[[20, 0.8], [30, 0.4]])
        assert _size(s, vix=25.0).contracts == 40             # 50 x 0.8
        assert _size(s, vix=31.0).contracts == 20             # 50 x 0.4

    def test_drawdown_tiers(self):
        assert _size(_sizer(), recent_losing_days=3).contracts == 25
        assert _size(_sizer(), recent_losing_days=5).contracts == 12   # int(12.5)
        custom = _sizer(drawdown_tiers=[[2, 0.2]])
        assert _size(custom, recent_losing_days=2).contracts == 10

    def test_strategy_risk_table(self):
        s = _sizer(strategy_risk={"long_call": 0.5})
        assert _size(s, strategy="long_call").contracts == 25
        # a strategy absent from the table keeps the sizer's 1.0 fallback
        assert _size(s, strategy="long_put").contracts == 50

    def test_volatility_curve(self):
        # iv=0.40: halfway along 0.20 -> 0.60, so 1.0 - 0.5 x 0.5 = 0.75
        assert _size(_sizer(), implied_vol=0.40).volatility_adjustment == pytest.approx(0.75)
        s = _sizer(iv_full_size_max=0.30, iv_floor_size_min=0.50, iv_floor_size_factor=0.2)
        assert _size(s, implied_vol=0.30).volatility_adjustment == pytest.approx(1.0)
        assert _size(s, implied_vol=0.50).volatility_adjustment == pytest.approx(0.2)
        assert _size(s, implied_vol=0.40).volatility_adjustment == pytest.approx(0.6)
        # Past the CONFIGURED floor (0.50) but still under the old 0.60 literal:
        # the factor must clamp at 0.2, not keep interpolating down to 0.0. The
        # boundary point above cannot tell the two apart (the line meets the
        # clamp exactly there) — a mutation to `>= 0.60` survived it.
        assert _size(s, implied_vol=0.55).volatility_adjustment == pytest.approx(0.2)

    def test_class_level_strategy_table_is_the_config_default(self):
        from ait.risk.position_sizer import PositionSizer
        assert PositionSizer.STRATEGY_RISK == PositionSizingConfig().strategy_risk

    def test_two_argument_construction_is_unchanged(self):
        """Every pre-W69 call site passes (positions, risk) only."""
        from ait.risk.position_sizer import PositionSizer
        s = PositionSizer(PositionConfig(), RiskConfig())
        assert s._sizing == PositionSizingConfig()


# ---------------------------------------------------------------------------
# 5. Weekend-gap rule — config-driven AND on the ET clock
# ---------------------------------------------------------------------------

def _risk_manager(risk_config: RiskConfig):
    from ait.risk.manager import RiskManager
    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=200_000.0)
    account.can_afford = AsyncMock(return_value=True)
    account.get_buying_power = AsyncMock(return_value=200_000.0)
    cb = MagicMock(is_tripped=False)
    pdt = MagicMock()
    pdt.can_open_position = MagicMock(return_value=(True, ""))
    sizer = MagicMock()
    sizer.calculate = MagicMock(return_value=MagicMock(contracts=1, max_risk_dollars=500))
    corr = MagicMock()
    corr.check_correlation = MagicMock(return_value=(True, ""))
    rm = RiskManager(PositionConfig(), risk_config, account, cb, pdt, sizer,
                     correlation_guard=corr, state=None)
    rm._open_positions = []
    return rm


def _strangle(confidence: float):
    from ait.risk.manager import TradeRequest
    return TradeRequest(symbol="SPY", strategy="short_strangle", direction="neutral",
                        contracts=1, entry_price=1.0, confidence=confidence,
                        max_loss=500.0, vix=18.0)


@pytest.fixture
def pinned_instant(monkeypatch):
    """Pin `ait.utils.time.datetime.now` to a fixed instant AND the process to a
    real non-ET TZ (Asia/Tokyo). A naive `datetime.now()` would then read the
    wrong weekday and hour; only an ET-aware read sees the Friday afternoon."""
    import ait.utils.time as ut

    def _pin(utc_instant: datetime):
        class _FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return utc_instant.astimezone(tz) if tz else utc_instant.astimezone(
                    ).replace(tzinfo=None)
        monkeypatch.setattr(ut, "datetime", _FakeDT)

    if hasattr(_time, "tzset"):
        import os
        old = os.environ.get("TZ")
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        _time.tzset()
        yield _pin
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        _time.tzset()
    else:  # pragma: no cover — POSIX-only pin, mirrors test_w28's guard
        pytest.skip("TZ + tzset is POSIX only")


class TestWeekendGapRule:
    async def test_fires_on_friday_after_the_et_cutoff_even_on_a_tokyo_host(
            self, pinned_instant):
        # 2026-10-09 14:31 ET (EDT) == 18:31 UTC == Sat 03:31 Tokyo
        pinned_instant(datetime(2026, 10, 9, 18, 31, tzinfo=UTC))
        v = await _risk_manager(RiskConfig(min_confidence=0.5)).validate_trade(
            _strangle(0.80))
        assert not v.approved and "weekend_gap_risk" in v.reason
        assert "90%" in v.reason

    async def test_silent_just_before_the_cutoff(self, pinned_instant):
        pinned_instant(datetime(2026, 10, 9, 18, 29, tzinfo=UTC))   # 14:29 ET Fri
        v = await _risk_manager(RiskConfig(min_confidence=0.5)).validate_trade(
            _strangle(0.80))
        assert "weekend_gap_risk" not in v.reason

    async def test_silent_on_a_non_friday(self, pinned_instant):
        pinned_instant(datetime(2026, 10, 8, 19, 30, tzinfo=UTC))   # Thu 15:30 ET
        v = await _risk_manager(RiskConfig(min_confidence=0.5)).validate_trade(
            _strangle(0.80))
        assert "weekend_gap_risk" not in v.reason

    async def test_cutoff_confidence_and_strategies_are_config(self, pinned_instant):
        pinned_instant(datetime(2026, 10, 9, 17, 30, tzinfo=UTC))   # Fri 13:30 ET
        # default cutoff 14:30 -> silent; moving it to 13:00 makes it fire
        quiet = await _risk_manager(RiskConfig(min_confidence=0.5)).validate_trade(
            _strangle(0.80))
        assert "weekend_gap_risk" not in quiet.reason
        early = RiskConfig(min_confidence=0.5, weekend_gap_cutoff_et="13:00")
        v = await _risk_manager(early).validate_trade(_strangle(0.80))
        assert not v.approved and "weekend_gap_risk" in v.reason
        # a stricter confidence floor is honoured and named in the reason
        strict = RiskConfig(min_confidence=0.5, weekend_gap_cutoff_et="13:00",
                            weekend_gap_min_confidence=0.95)
        v = await _risk_manager(strict).validate_trade(_strangle(0.92))
        assert not v.approved and "95%" in v.reason
        # an empty strategy list switches the rule off entirely
        off = RiskConfig(min_confidence=0.5, weekend_gap_cutoff_et="13:00",
                         weekend_gap_strategies=[])
        v = await _risk_manager(off).validate_trade(_strangle(0.80))
        assert "weekend_gap_risk" not in v.reason


# ---------------------------------------------------------------------------
# 6. Intraday budget, thesis check
# ---------------------------------------------------------------------------

def _orchestrator(tmp_path, monkeypatch, trading_updates=None, exit_updates=None):
    import ait.bot.orchestrator as om
    from ait.bot.state import StateManager

    monkeypatch.chdir(tmp_path)
    settings = load_settings(str(CONFIG_PATH))
    if trading_updates:
        settings.trading = settings.trading.model_copy(update=trading_updates)
    if exit_updates:
        settings.exit = settings.exit.model_copy(update=exit_updates)
    orch = om.TradingOrchestrator.__new__(om.TradingOrchestrator)
    orch._settings = settings
    orch._state = StateManager(tmp_path / "state.db")
    return orch


def _pin_et(monkeypatch, hour: int, minute: int = 0):
    import ait.utils.time as ut
    fixed = datetime(2026, 10, 7, hour, minute, tzinfo=ET)     # a Wednesday
    monkeypatch.setattr(ut, "now_et", lambda: fixed)


class TestIntradayBudgetIsConfig:
    def test_default_windows(self, tmp_path, monkeypatch):
        orch = _orchestrator(tmp_path, monkeypatch)
        _pin_et(monkeypatch, 10, 0)
        assert orch._get_trade_budget() == 2        # first hour
        _pin_et(monkeypatch, 11, 0)
        assert orch._get_trade_budget() == 4        # mid-day
        _pin_et(monkeypatch, 14, 0)
        assert orch._get_trade_budget() == 12       # power hour: max_daily_trades

    def test_budgets_and_edges_move_with_config(self, tmp_path, monkeypatch):
        orch = _orchestrator(tmp_path, monkeypatch, trading_updates={
            "first_hour_trade_budget": 1, "midday_trade_budget": 3,
            "first_hour_end_et": "09:45", "midday_end_et": "13:00"})
        _pin_et(monkeypatch, 9, 40)
        assert orch._get_trade_budget() == 1
        _pin_et(monkeypatch, 10, 0)                  # past the new first-hour edge
        assert orch._get_trade_budget() == 3
        _pin_et(monkeypatch, 13, 0)                  # at the new mid-day edge
        assert orch._get_trade_budget() == 12


def _thesis_orchestrator(tmp_path, monkeypatch, *, prediction=None, vix=18.0,
                         entry_vix=20.0, exit_updates=None):
    import ait.bot.orchestrator as om
    orch = _orchestrator(tmp_path, monkeypatch, exit_updates=exit_updates)
    orch._state = MagicMock()
    orch._state.get_trade_context.return_value = {
        "entry_direction": "bullish", "entry_regime": "range_bound",
        "entry_vix": entry_vix}
    orch._market_data = MagicMock()
    import pandas as pd
    hist = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    orch._market_data.get_historical = AsyncMock(return_value=hist)
    orch._market_data.get_vix = AsyncMock(return_value=vix)
    orch._predictor = MagicMock()
    orch._predictor.predict.return_value = prediction
    orch._regime_detector = MagicMock()
    orch._regime_detector.analyze.return_value = None
    return orch, om


class TestThesisThresholdsAreConfig:
    async def test_direction_flip_confidence(self, tmp_path, monkeypatch):
        import ait.bot.orchestrator as om
        pred = SimpleNamespace(confidence=0.75, direction=om.SignalDirection.BEARISH)
        pos = SimpleNamespace(trade_id="T1", symbol="SPY", strategy="long_call")

        orch, _ = _thesis_orchestrator(tmp_path, monkeypatch, prediction=pred)
        invalid, reason = await orch._check_thesis_valid(pos)
        assert invalid and "direction_flipped_to_bearish" in reason     # 0.75 > 0.70

        orch, _ = _thesis_orchestrator(
            tmp_path, monkeypatch, prediction=pred,
            exit_updates={"thesis_direction_flip_confidence": 0.80})
        assert await orch._check_thesis_valid(pos) == (False, "")        # 0.75 < 0.80

    async def test_vix_spike_pct(self, tmp_path, monkeypatch):
        pos = SimpleNamespace(trade_id="T1", symbol="SPY", strategy="long_call")
        # +50% since entry
        orch, _ = _thesis_orchestrator(tmp_path, monkeypatch, vix=30.0, entry_vix=20.0)
        invalid, reason = await orch._check_thesis_valid(pos)
        assert invalid and reason.startswith("vix_spike")

        orch, _ = _thesis_orchestrator(
            tmp_path, monkeypatch, vix=30.0, entry_vix=20.0,
            exit_updates={"thesis_vix_spike_pct": 0.60})
        assert await orch._check_thesis_valid(pos) == (False, "")

    async def test_condors_stay_exempt_whatever_the_thresholds(self, tmp_path, monkeypatch):
        pos = SimpleNamespace(trade_id="T1", symbol="SPY", strategy="iron_condor")
        orch, _ = _thesis_orchestrator(
            tmp_path, monkeypatch, vix=60.0, entry_vix=10.0,
            exit_updates={"thesis_vix_spike_pct": 0.05})
        assert await orch._check_thesis_valid(pos) == (False, "")


# ---------------------------------------------------------------------------
# 7. Regime, drift, capital tiers, roll delta, learning
# ---------------------------------------------------------------------------

def _classify(cfg: RegimeConfig | None, **kw):
    from ait.ml.regime import RegimeDetector
    args = dict(trend=0.0, vol=0.15, vix=18.0, iv_rank=0.5, hurst=0.5)
    args.update(kw)
    return RegimeDetector(cfg)._classify(**args)[0]


class TestRegimeThresholdsAreConfig:
    def test_defaults_classify_as_before(self):
        from ait.ml.regime import MarketRegime as R
        assert _classify(None, vix=31.0) is R.HIGH_VOLATILITY
        assert _classify(None, vol=0.41) is R.HIGH_VOLATILITY
        assert _classify(None, iv_rank=0.1, vol=0.15, trend=0.1) is R.IV_CRUSH
        assert _classify(None, hurst=0.35, trend=0.1) is R.MEAN_REVERTING
        assert _classify(None, trend=0.40) is R.TRENDING_UP
        assert _classify(None, trend=-0.40) is R.TRENDING_DOWN
        assert _classify(None, vix=14.0, trend=0.0) is R.LOW_VOLATILITY
        assert _classify(None) is R.RANGE_BOUND

    def test_vix_trigger_moves_with_config(self):
        from ait.ml.regime import MarketRegime as R
        assert _classify(None, vix=26.0) is not R.HIGH_VOLATILITY
        assert _classify(RegimeConfig(high_vol_vix=25), vix=26.0) is R.HIGH_VOLATILITY

    def test_trend_threshold_moves_with_config(self):
        from ait.ml.regime import MarketRegime as R
        assert _classify(None, trend=0.30) is R.RANGE_BOUND
        assert _classify(RegimeConfig(strong_trend=0.25), trend=0.30) is R.TRENDING_UP


class TestDriftIsConfig:
    def test_trainer_passes_the_ml_config_through(self):
        from ait.ml.trainer import ModelTrainer
        cfg = MLConfig(drift_window_size=80, drift_accuracy_threshold=0.55,
                       drift_calibration_threshold=0.15, drift_high_confidence=0.9)
        t = ModelTrainer(cfg, MagicMock(), MagicMock(), MagicMock())
        d = t.drift_detector
        assert d._window_size == 80 and d._accuracy_threshold == 0.55
        assert d._calibration_threshold == 0.15 and d._high_confidence == 0.9

    def test_bare_detector_uses_the_model_defaults(self):
        from ait.ml.drift import DriftDetector
        d = DriftDetector()
        assert (d._window_size, d._accuracy_threshold, d._calibration_threshold,
                d._high_confidence) == (50, 0.40, 0.25, 0.80)

    def test_accuracy_threshold_changes_the_verdict(self):
        from ait.ml.drift import DriftDetector

        def _run(threshold):
            d = DriftDetector(accuracy_threshold=threshold)
            for i in range(30):
                d.record_prediction(f"T{i}", "bullish", 0.6)
                d.record_outcome(f"T{i}", "bullish" if i % 2 == 0 else "bearish")  # 50%
            return d.check_drift().is_drifting
        assert _run(0.40) is False
        assert _run(0.60) is True


class TestCapitalTiersAreConfig:
    def test_default_boundaries(self):
        from ait.risk.capital_tiers import CapitalTier, CapitalTierManager
        m = CapitalTierManager()
        assert m.get_tier(1_999) is CapitalTier.MICRO
        assert m.get_tier(2_000) is CapitalTier.SMALL
        assert m.get_tier(5_000) is CapitalTier.MEDIUM
        assert m.get_tier(25_000) is CapitalTier.LARGE

    def test_boundaries_move_with_config(self):
        from ait.risk.capital_tiers import CapitalTier, CapitalTierManager
        m = CapitalTierManager(AccountConfig(
            capital_tier_small_min=1_000, capital_tier_medium_min=3_000,
            capital_tier_large_min=10_000))
        assert m.get_tier(1_000) is CapitalTier.SMALL
        assert m.get_tier(3_000) is CapitalTier.MEDIUM
        assert m.get_tier(10_000) is CapitalTier.LARGE


class TestRollDeltaIsConfig:
    async def test_target_and_tolerance_move_with_config(self):
        import json
        from dataclasses import dataclass
        from datetime import date, timedelta

        from ait.data.options_chain import OptionContract, OptionsChain
        from ait.execution.roll_selector import select_roll_target

        # Same fixture shape test_w53_roll_target_selection.py uses: a put
        # vertical 98/95 being rolled; target chain has |delta| 0.05/0.20/0.50.
        @dataclass
        class _Trade:
            trade_id: str = "T-W69"
            symbol: str = "SPY"
            legs: str = json.dumps([
                {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-09-24"},
                {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-09-24"},
                {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-09-24"},
                {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-09-24"},
            ])

        def _c(right, strike, expiry, delta):
            return OptionContract(
                symbol="SPY", expiry=expiry, strike=strike, right=right, bid=1.45,
                ask=1.55, last=1.50, volume=2_500, open_interest=9_000,
                implied_vol=0.18, delta=delta, con_id=int(strike * 10) + 2,
                source="yahoo_delayed")

        expiry = date.today() + timedelta(days=21)
        chain = OptionsChain(
            symbol="SPY", underlying_price=100.0, expiry=expiry, source="yahoo_delayed",
            puts=[_c("P", 90.0, expiry, -0.05), _c("P", 95.0, expiry, -0.20),
                  _c("P", 100.0, expiry, -0.50)],
            calls=[_c("C", 100.0, expiry, 0.50), _c("C", 105.0, expiry, 0.20),
                   _c("C", 110.0, expiry, 0.05)])

        class _Svc:
            async def get_chain(self, symbol, min_dte=None, max_dte=None):
                return [chain]

        svc, _FakeTrade = _Svc(), _Trade

        default = await select_roll_target(
            _FakeTrade(), "P", svc, AssignmentResponseConfig())
        assert default is not None and default.new_short.strike == 95.0   # |d| 0.20

        atm = await select_roll_target(
            _FakeTrade(), "P", svc,
            AssignmentResponseConfig(roll_short_delta_target=0.50))
        assert atm is not None and atm.new_short.strike == 100.0          # |d| 0.50

        # nothing within tolerance of 0.35: nearest strikes are 0.15 away
        none = await select_roll_target(
            _FakeTrade(), "P", svc,
            AssignmentResponseConfig(roll_short_delta_target=0.35,
                                     roll_short_delta_tolerance=0.07))
        assert none is None
        wide = await select_roll_target(
            _FakeTrade(), "P", svc,
            AssignmentResponseConfig(roll_short_delta_target=0.35,
                                     roll_short_delta_tolerance=0.20))
        assert wide is not None


class TestLearningConfigIsRead:
    def test_engine_limits_come_from_settings(self, tmp_path):
        from ait.bot.state import StateManager
        from ait.learning.adaptor import AdaptationLimits
        from ait.learning.engine import LearningEngine

        cfg = LearningConfig(max_adaptations_per_cycle=7, min_insight_confidence=0.8,
                             min_confidence_floor=0.45, min_confidence_ceiling=0.95)
        eng = LearningEngine(
            StateManager(tmp_path / "s.db"),
            limits=AdaptationLimits(
                min_confidence_floor=cfg.min_confidence_floor,
                min_confidence_ceiling=cfg.min_confidence_ceiling,
                min_insight_confidence=cfg.min_insight_confidence,
                max_adaptations_per_cycle=cfg.max_adaptations_per_cycle),
            analyzer_thresholds=cfg.analyzer)
        lim = eng.adaptor._limits
        assert (lim.max_adaptations_per_cycle, lim.min_insight_confidence,
                lim.min_confidence_floor, lim.min_confidence_ceiling) == (7, 0.8, 0.45, 0.95)

    def test_orchestrator_builds_the_engine_from_settings(self):
        """The wiring half: construct the REAL orchestrator's engine inputs.
        The orchestrator __init__ is too heavy to run here, so assert on what it
        passes: every LearningEngine( call in src/ supplies limits=."""
        offenders = []
        for path in SRC.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", getattr(node.func, "attr", ""))
                        == "LearningEngine"):
                    if "limits" not in {k.arg for k in node.keywords}:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        assert not offenders, f"LearningEngine built without limits=: {offenders}"

    def test_analyzer_cutoff_changes_the_insight(self, tmp_path):
        from ait.learning.analyzer import TradeAnalyzer

        # 5 closes of one strategy, 2 winners -> 40% win rate
        trades = [{"strategy": "iron_condor", "realized_pnl": 10.0 if i < 2 else -10.0,
                   "trade_id": f"T{i}"} for i in range(5)]
        db = tmp_path / "none.db"
        default = TradeAnalyzer(db_path=db)._analyze_strategies(trades)
        assert default == []                      # 40% is not < 35% and not > 65%
        strict = TradeAnalyzer(
            db_path=db, thresholds=LearningAnalyzerConfig(strategy_poor_win_rate=0.45)
        )._analyze_strategies(trades)
        assert [i.action for i in strict] == ["disable_iron_condor"]


# ---------------------------------------------------------------------------
# 8. Class guards — catch the NEXT re-hardcoding
# ---------------------------------------------------------------------------

def _numeric_constants(tree: ast.AST):
    for n in ast.walk(tree):
        if (isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)):
            yield n


class TestNoLiteralPortsRemain:
    @pytest.mark.parametrize("rel", [
        "src/ait/orchestration/master.py", "web_logs.py", "status_server.py"])
    def test_no_dashboard_port_literal_in_executable_code(self, rel):
        tree = ast.parse((ROOT / rel).read_text())
        bad = [n.lineno for n in _numeric_constants(tree)
               if n.value in (8501, 8502, 8503)]
        assert not bad, (
            f"{rel}: literal dashboard port at line(s) {bad} — read "
            "settings.dashboard.* (resolve_dashboard_port / _dashboard_cfg)")


class TestNoLiteralGatesRemainInTheOrchestrator:
    # The thresholds W69 moved to config. They only trip the guard as COMPARISON
    # operands, so unrelated uses of the same number (a sleep, a log width) are fine.
    MOVED = {10.5, 14.0, 0.85, 0.8, 0.7, 0.70, 0.3, 0.30, 0.15}

    def test_no_moved_threshold_is_compared_against_a_literal(self):
        path = SRC / "bot" / "orchestrator.py"
        tree = ast.parse(path.read_text())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for c in [node.left, *node.comparators]:
                if (isinstance(c, ast.Constant) and isinstance(c.value, float)
                        and c.value in self.MOVED):
                    offenders.append(f"line {node.lineno}: {ast.unparse(node)[:90]}")
        assert not offenders, (
            "a W69-externalized threshold is hardcoded again — use settings.*:\n  "
            + "\n  ".join(offenders))

    def test_trade_budget_has_no_numeric_tier_literals(self):
        tree = ast.parse((SRC / "bot" / "orchestrator.py").read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_get_trade_budget")
        bad = [(n.lineno, n.value) for n in _numeric_constants(fn)
               if n.value in (2, 4, 10.5, 14.0)]
        assert not bad, f"_get_trade_budget carries tier literals again: {bad}"


class TestNoAccountIdsInSourceOrTests:
    # Real-account shapes, assembled so this file does not itself contain them.
    IDS = ("DUN" + "603821", "U2195" + "9335", "DUO" + "442667")

    def test_src_and_tests_are_clean(self):
        offenders = []
        for base in (SRC, ROOT / "tests", ROOT / "scripts"):
            for path in base.rglob("*.py"):
                text = path.read_text(errors="ignore")
                offenders += [f"{path.relative_to(ROOT)}: {i}"
                              for i in self.IDS if i in text]
        assert not offenders, f"real account id in tracked source: {offenders}"
