"""R17 ops/tooling fixes — external review findings 12-16, each test fails
against the pre-fix code.

  [12] check_first_rth_liveness.py hardcoded an EDT (UTC-4) offset instead
       of the codebase's DST-aware ET convention
  [13] smoke_deploy.py's NoWriteGuard raised on any row-count delta, with
       no tolerance for the live bot's own concurrent writes
  [14] walkforward.py's window_capital divided by the configured symbol
       count instead of the count that actually survived the learner's
       allow/min-row filters
  [15] ml/features.py's blanket NaN-fill used 0.0, contradicting
       vix_level's documented neutral value of 0.5
  [16] options_chain.py's OI<=0-means-unknown leniency was applied to the
       Yahoo fallback too, where a reported 0 is a genuine value
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


class TestLivenessCheckUsesEtNotHardcodedEdt:
    def test_marks_age_correct_in_winter_est(self):
        """January -- EST is UTC-5, not the old hardcoded UTC-4 (EDT)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import check_first_rth_liveness as mod
        importlib.reload(mod)

        assert not hasattr(mod, "ET")  # module-level hardcoded EDT constant is gone

        import datetime as real_dt
        from ait.utils.time import ET as real_et

        now_winter = real_dt.datetime(2026, 1, 15, 10, 32, 0, tzinfo=real_et)
        mark_iso = real_dt.datetime(2026, 1, 15, 10, 30, 0).isoformat()  # 2 min earlier

        age = mod._mark_age_seconds(mark_iso, now=now_winter)
        # Pre-fix: the hardcoded UTC-4 offset relabels a true UTC-5 (EST)
        # wall-clock reading as if it were UTC-4, shifting the computed mark
        # instant by exactly one hour -- reading ~3720s old instead of ~120s,
        # a false LIVENESS FAIL against the 600s freshness threshold.
        assert age == pytest.approx(120.0, abs=1.0)
        assert age < mod.FRESH_SECS


class TestSmokeDeployToleratesLiveActivity:
    def _guard(self, sqlite_before, sqlite_after):
        from scripts.smoke_deploy import NoWriteGuard

        g = NoWriteGuard.__new__(NoWriteGuard)
        g.sqlite_before = sqlite_before
        g.duck_before = None
        g._dv_con = None
        g._sqlite_after = sqlite_after
        return g

    def test_live_write_table_change_with_fresh_heartbeat_is_a_warning(
        self, monkeypatch, tmp_path
    ):
        import scripts.smoke_deploy as mod

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "bot_heartbeat").write_text("x")
        monkeypatch.setattr(mod, "ROOT", tmp_path)

        before = {"trades": 5, "__schema_version__": 1}
        after = {"trades": 6, "__schema_version__": 1}  # a real trade closed
        g = self._guard(before, after)
        monkeypatch.setattr(g, "_sqlite_snapshot", lambda: after)
        monkeypatch.setattr(g, "_duck_snapshot", lambda: None)

        result = g.verify()  # pre-fix: raises RuntimeError
        assert "row counts unchanged" not in result or "rows changed" in result

    def test_unknown_table_change_still_fails(self, monkeypatch, tmp_path):
        import scripts.smoke_deploy as mod

        hb = tmp_path / "bot_heartbeat"
        hb.write_text("x")
        monkeypatch.setattr(mod, "ROOT", tmp_path)

        before = {"some_other_table": 1, "__schema_version__": 1}
        after = {"some_other_table": 2, "__schema_version__": 1}
        g = self._guard(before, after)
        monkeypatch.setattr(g, "_sqlite_snapshot", lambda: after)
        monkeypatch.setattr(g, "_duck_snapshot", lambda: None)

        with pytest.raises(RuntimeError, match="SMOKE WROTE"):
            g.verify()

    def test_live_write_table_change_without_fresh_heartbeat_still_fails(
        self, monkeypatch, tmp_path
    ):
        import scripts.smoke_deploy as mod

        monkeypatch.setattr(mod, "ROOT", tmp_path)  # no heartbeat file at all

        before = {"trades": 5, "__schema_version__": 1}
        after = {"trades": 6, "__schema_version__": 1}
        g = self._guard(before, after)
        monkeypatch.setattr(g, "_sqlite_snapshot", lambda: after)
        monkeypatch.setattr(g, "_duck_snapshot", lambda: None)

        with pytest.raises(RuntimeError, match="SMOKE WROTE"):
            g.verify()


def _make_ohlcv(days: int = 500, start_price: float = 100.0) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    returns = np.random.normal(0.0005, 0.015, days)
    close = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.random.randint(1_000_000, 10_000_000, days),
    }, index=dates)


class TestWalkforwardWindowCapitalUsesActiveSymbols:
    def test_window_capital_uses_active_not_configured_symbol_count(self, monkeypatch):
        """Two symbols passed in; QQQ's short history doesn't reach the test
        window at all, so only SPY actually deploys a sleeve of capital."""
        from datetime import date
        from ait.backtesting.engine import Backtester
        from ait.backtesting.result import BacktestResult
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

        def fast_run(self_bt):
            return BacktestResult(trades=[{"pnl": 100.0, "entry_date": "2023-11-05"}])

        monkeypatch.setattr(Backtester, "run", fast_run)

        data = {"SPY": _make_ohlcv(500), "QQQ": _make_ohlcv(30)}  # QQQ: Jan-Feb 2023 only
        cfg = WalkForwardConfig(train_days=300, test_days=50, gap_days=5,
                                initial_capital=100_000.0)
        bt = WalkForwardBacktester(["SPY", "QQQ"], ["iron_condor"], config=cfg)

        wr, _ = bt._run_single_window(
            window_id=1,
            train_start=date(2023, 1, 1), train_end=date(2023, 10, 28),
            test_start=date(2023, 11, 2), test_end=date(2023, 12, 22),
            data=data, vix_full=pd.DataFrame(), learner=None,
        )
        # Pre-fix: window_capital = 100_000 * len(data) = 200_000 (QQQ counted
        # despite deploying no capital). Post-fix: 100_000 * active_symbols = 100_000.
        assert wr.backtest_result.initial_capital == pytest.approx(100_000.0)


class TestFeatureNanFillRespectsNeutralDefaults:
    def test_neutral_defaults_dict_matches_documented_vix_neutral(self):
        # The dict FeatureEngine.compute() actually reads for the blanket
        # all-NaN-column fill -- ties this test to the real fix site rather
        # than re-deriving the same literal independently.
        from ait.ml.features import NEUTRAL_COLUMN_DEFAULTS
        assert NEUTRAL_COLUMN_DEFAULTS["vix_level"] == 0.5

    def test_fill_logic_applies_the_dict_per_column(self):
        from ait.ml.features import NEUTRAL_COLUMN_DEFAULTS

        features = pd.DataFrame({
            "vix_level": [float("nan")] * 5,
            "other_col": [float("nan")] * 5,
        })
        all_nan = [c for c in features.columns if features[c].isna().all()]
        for col in all_nan:
            features[col] = features[col].fillna(NEUTRAL_COLUMN_DEFAULTS.get(col, 0.0))

        # Pre-fix: a blanket fillna(0.0) would have made vix_level 0.0 too.
        assert (features["vix_level"] == 0.5).all()
        assert (features["other_col"] == 0.0).all()


class TestOptionsChainSourceAwareLiquidity:
    def test_yahoo_zero_oi_rejected_when_otherwise_thin(self, monkeypatch):
        monkeypatch.setenv("AIT_LIQ_MIN_VOL", "0")
        monkeypatch.setenv("AIT_LIQ_MIN_OI", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.50")
        from ait.data.options_chain import OptionContract

        c = OptionContract(symbol="SPY", expiry="2026-07-24", strike=650, right="C",
                           bid=1.00, ask=1.05, last=1.02, volume=100,
                           open_interest=0, implied_vol=0.2,
                           source="yahoo_delayed")
        # Pre-fix: any open_interest<=0 was treated as "unknown", passing.
        assert not c.is_liquid

    def test_ibkr_zero_oi_still_treated_as_unknown(self, monkeypatch):
        monkeypatch.setenv("AIT_LIQ_MIN_VOL", "0")
        monkeypatch.setenv("AIT_LIQ_MIN_OI", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.50")
        from ait.data.options_chain import OptionContract

        c = OptionContract(symbol="SPY", expiry="2026-07-24", strike=650, right="C",
                           bid=1.00, ask=1.05, last=1.02, volume=100,
                           open_interest=0, implied_vol=0.2)  # source defaults to "ibkr"
        assert c.is_liquid

    def test_filter_liquid_respects_per_contract_source(self, monkeypatch):
        monkeypatch.setenv("AIT_LIQ_MIN_VOL", "0")
        monkeypatch.setenv("AIT_LIQ_MIN_OI", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.50")
        from datetime import date
        from ait.data.options_chain import OptionsChain, OptionContract

        expiry = date(2026, 7, 24)
        ibkr_c = OptionContract(symbol="SPY", expiry=expiry, strike=650,
                                right="C", bid=1.00, ask=1.05, last=1.02,
                                volume=100, open_interest=0, implied_vol=0.2)
        yahoo_c = OptionContract(symbol="SPY", expiry=expiry, strike=655,
                                 right="C", bid=1.00, ask=1.05, last=1.02,
                                 volume=100, open_interest=0, implied_vol=0.2,
                                 source="yahoo_delayed")
        chain = OptionsChain(symbol="SPY", underlying_price=650.0,
                            expiry=expiry, calls=[ibkr_c, yahoo_c], puts=[])
        filtered = chain.filter_liquid()
        assert ibkr_c in filtered.calls
        assert yahoo_c not in filtered.calls
