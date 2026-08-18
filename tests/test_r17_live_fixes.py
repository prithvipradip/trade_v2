"""R17 live/active fixes — external review findings 1-7, each test fails
against the pre-fix code. Covers the bugs affecting the bot as currently
configured (iron_condor is the only enabled strategy):

  [1] portfolio.py: rule 3b (delta breach) claimed the exit elif-chain slot
      on strategy-membership alone, permanently blocking rule 3c
      (earnings pre-close) for iron_condor/short_strangle/etc once DTE>5
  [2] trainer.py: auto-rollback mean-accuracy calc always produced an empty
      list (per-symbol results are dicts, not scalars), so a degraded model
      was never rolled back
  [3] status.py get_status(): NULL exit_reason_detailed silently dropped a
      closed trade from pnl_today/pnl_life (missing COALESCE)
  [4] economic_calendar.py: the calendar-exhaustion alarm only logged,
      never reached a human
  [5] runtime_env.py: .env loaded after the protective setdefault() calls,
      so it could never override them (incl. the live/delayed data switch)
  [6] runtime_env.py: capital_base() had no enforcement if it silently fell
      back to the hardcoded default in live mode
  [7] earnings.py: used date.today() (server-local clock) instead of the
      codebase's now_et() convention
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.config.settings import ExitConfig
from ait.data.market_data import Quote


# --- [1] earnings pre-close must be reachable when delta is inert ----------

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


@dataclass
class _FakeTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-EARN"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, unrealized: float, earnings_in_days: int):
    from ait.execution.portfolio import PortfolioManager

    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()

    tick = {"n": 0}

    async def _quote(symbol):
        tick["n"] += 1
        return Quote(
            symbol=symbol, bid=round(spot - 0.01, 2), ask=round(spot + 0.01, 2),
            last=spot, volume=1_000_000,
            timestamp=datetime.now() + timedelta(seconds=tick["n"]),
        )
    mgr._market_data.get_quote = _quote

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price
    mgr._exit_config = ExitConfig()

    earnings = MagicMock()
    earnings.get_next_earnings.return_value = SimpleNamespace(
        next_earnings_date=date.today() + timedelta(days=earnings_in_days))
    mgr._earnings = earnings
    mgr._economic_cal = None

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks: unrealized
    # Today's reality (R16-confirmed): no greeks subscription in the exit
    # path, delta is always unavailable.
    mgr._get_position_delta = lambda trade: None
    return mgr


def _condor(dte: int) -> _FakeTrade:
    t = _FakeTrade(expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="neutral")
    return t


class TestEarningsExitReachableDespiteInertDelta:
    async def test_pre_earnings_exit_fires_two_days_out(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")  # isolate rule 3c
        # DTE > 5 so rules 3/3a don't pre-empt; modest P&L so take-profit
        # doesn't pre-empt either; spot inside the strikes.
        mgr = _manager(spot=100.0, unrealized=10.0, earnings_in_days=2)
        status = await mgr._evaluate_position(_condor(dte=20))
        # Pre-fix: rule 3b claims the elif slot on strategy-membership alone
        # (delta is always None), 3c never runs, should_exit stays False.
        assert status.should_exit is True
        assert status.exit_reason.startswith("pre_earnings_iv_crush")

    async def test_no_spurious_exit_far_from_earnings(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        mgr = _manager(spot=100.0, unrealized=10.0, earnings_in_days=30)
        status = await mgr._evaluate_position(_condor(dte=20))
        assert status.should_exit is False


# --- [2] auto-rollback must see the real mean, not an always-empty list ----

class TestTrainerAutoRollback:
    def _trainer(self, per_symbol_accuracy: float, prev_scores: dict):
        from ait.ml.trainer import ModelTrainer

        t = ModelTrainer.__new__(ModelTrainer)
        t._config = SimpleNamespace(lookback_days=60)
        t._predictor = MagicMock()
        t._predictor.model_version = "v1"
        t._predictor.cv_scores = dict(prev_scores)
        t._predictor.train.return_value = {
            "xgboost": per_symbol_accuracy, "lightgbm": per_symbol_accuracy,
        }
        t._market_data = MagicMock()

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

        async def _get_historical(symbol, days):
            # VIX/SPY market-context fetches inside _fetch_market_context
            # should look absent (len<=20 skips them); real symbols get data.
            if symbol in ("^VIX", "SPY"):
                return None
            return df
        t._market_data.get_historical = _get_historical
        t._store = MagicMock()
        t._last_train_date = None
        t._drift_detector = MagicMock()
        t._range_predictor = None
        t._vol_mag_predictor = None
        return t

    async def test_degraded_model_triggers_rollback(self):
        # prev mean 0.90, new per-model accuracy 0.55 across 2 symbols --
        # a clear, well-past-threshold regression.
        t = self._trainer(per_symbol_accuracy=0.55, prev_scores={"prev": 0.90})
        await t.train_all_symbols(["AAPL", "MSFT"])
        t._predictor.rollback.assert_called_once_with("v1")

    async def test_similar_model_does_not_rollback(self):
        t = self._trainer(per_symbol_accuracy=0.89, prev_scores={"prev": 0.90})
        await t.train_all_symbols(["AAPL", "MSFT"])
        t._predictor.rollback.assert_not_called()


# --- [3] dashboard P&L must include NULL-labeled closed trades -------------

class TestStatusDashboardNullSafe:
    def _seed_db(self, tmp_path):
        db_path = tmp_path / "ait_state.db"
        from ait.bot.state import StateManager
        StateManager(db_path=db_path)  # real schema, no other side effects

        con = sqlite3.connect(db_path)
        today_iso = datetime.now().isoformat()
        cols = ("trade_id, symbol, strategy, direction, status, entry_time, "
                "entry_price, quantity, contract_type, exit_time, "
                "realized_pnl, exit_reason_detailed")
        con.execute(
            f"INSERT INTO trades ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("T-1", "SPY", "iron_condor", "neutral", "closed", today_iso,
             1.0, 1, "spread", today_iso, 100.0, "take_profit"),
        )
        con.execute(
            f"INSERT INTO trades ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("T-2", "QQQ", "iron_condor", "neutral", "closed", today_iso,
             1.0, 1, "spread", today_iso, 50.0, None),
        )
        con.commit()
        con.close()
        return db_path

    def test_null_exit_reason_counted_in_pnl(self, tmp_path, monkeypatch):
        db_path = self._seed_db(tmp_path)
        import status as status_mod
        monkeypatch.setattr(status_mod, "DB", db_path)
        monkeypatch.setattr(status_mod, "_proc_running", lambda: (False, 0))

        out = status_mod.get_status()
        # Pre-fix: T-2 (NULL exit_reason_detailed) silently dropped.
        assert out["pnl_today_n"] == 2
        assert out["pnl_today"] == pytest.approx(150.0)
        assert out["pnl_life_n"] == 2
        assert out["pnl_life"] == pytest.approx(150.0)


# --- [4] calendar exhaustion must actually notify --------------------------

class TestEconCalendarExhaustionNotifies:
    async def test_notifies_once_when_exhausted(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._economic_cal = SimpleNamespace(exhausted_warned=True)
        orch._send_notification = AsyncMock()

        await orch._check_economic_calendar_exhaustion()
        await orch._check_economic_calendar_exhaustion()  # second tick: no repeat

        orch._send_notification.assert_awaited_once()
        assert "EXHAUSTED" in orch._send_notification.await_args.args[0]

    async def test_no_notification_when_not_exhausted(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._economic_cal = SimpleNamespace(exhausted_warned=False)
        orch._send_notification = AsyncMock()

        await orch._check_economic_calendar_exhaustion()
        orch._send_notification.assert_not_awaited()


# --- [5] .env must be able to override the protective defaults -------------

class TestDotenvPrecedence:
    def test_env_file_overrides_hardcoded_default(self, tmp_path, monkeypatch):
        import ait.config.runtime_env as mod

        (tmp_path / ".env").write_text("AIT_IC_WING_K=1.2\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.delenv("AIT_IC_WING_K", raising=False)

        mod.apply_runtime_env_defaults()
        # Pre-fix: setdefault("AIT_IC_WING_K", "1.6") ran before .env loaded,
        # so .env's 1.2 could never win.
        assert os.environ["AIT_IC_WING_K"] == "1.2"

    def test_true_shell_export_still_wins_over_dotenv(self, tmp_path, monkeypatch):
        import ait.config.runtime_env as mod

        (tmp_path / ".env").write_text("AIT_IC_WING_K=1.2\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setenv("AIT_IC_WING_K", "9.9")  # simulates a real export

        mod.apply_runtime_env_defaults()
        assert os.environ["AIT_IC_WING_K"] == "9.9"

    def test_bare_launch_still_gets_hardcoded_default(self, tmp_path, monkeypatch):
        import ait.config.runtime_env as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)  # no .env present
        monkeypatch.delenv("AIT_IC_WING_K", raising=False)

        mod.apply_runtime_env_defaults()
        assert os.environ["AIT_IC_WING_K"] == "1.6"


# --- [6] capital_base_source must report which tier actually fired ---------

class TestCapitalBaseSource:
    def test_env_tier(self, monkeypatch):
        import ait.config.runtime_env as mod
        monkeypatch.setenv("AIT_CAPITAL_BASE", "250000")
        assert mod.capital_base_source() == "env"
        assert mod.capital_base() == pytest.approx(250_000.0)

    def test_default_tier_when_nothing_set(self, tmp_path, monkeypatch):
        import ait.config.runtime_env as mod
        monkeypatch.delenv("AIT_CAPITAL_BASE", raising=False)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)  # no data/ait_state.db
        assert mod.capital_base_source() == "default"
        assert mod.capital_base() == pytest.approx(196_000.0)

    def test_live_nlv_tier(self, tmp_path, monkeypatch):
        import ait.config.runtime_env as mod
        monkeypatch.delenv("AIT_CAPITAL_BASE", raising=False)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

        db_dir = tmp_path / "data"
        db_dir.mkdir()
        from ait.bot.state import StateManager
        StateManager(db_path=db_dir / "ait_state.db")
        con = sqlite3.connect(db_dir / "ait_state.db")
        con.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES "
            "('last_net_liquidation', '212345.67', ?)", (datetime.now().isoformat(),))
        con.commit()
        con.close()

        assert mod.capital_base_source() == "live_nlv"
        assert mod.capital_base() == pytest.approx(212_345.67)


# --- [7] earnings date checks must use ET, not the server-local clock ------

class TestEarningsUsesEtClock:
    def test_is_near_earnings_follows_mocked_et_not_system_clock(self, monkeypatch):
        import ait.data.earnings as mod

        cal = mod.EarningsCalendar()
        cal._cache.set(
            "earnings_XYZ",
            mod.EarningsInfo(symbol="XYZ", next_earnings_date=date(2026, 3, 15)),
        )
        # 1 day before earnings, per the mocked ET clock -- deliberately far
        # from the real system date so a lingering date.today() call would
        # fail this assertion.
        monkeypatch.setattr(mod, "now_et", lambda: datetime(2026, 3, 14, 23, 30))

        assert cal.is_near_earnings("XYZ") is True

    def test_is_near_earnings_false_outside_window_per_et_clock(self, monkeypatch):
        import ait.data.earnings as mod

        cal = mod.EarningsCalendar()
        cal._cache.set(
            "earnings_XYZ",
            mod.EarningsInfo(symbol="XYZ", next_earnings_date=date(2026, 3, 15)),
        )
        monkeypatch.setattr(mod, "now_et", lambda: datetime(2026, 1, 1, 12, 0))

        assert cal.is_near_earnings("XYZ") is False
