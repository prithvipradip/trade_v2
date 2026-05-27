"""Full pipeline integration tests — require real IB data pre-populated by
scripts/run_integration_test.py.

Skip unless RUN_FULL_INTEGRATION=1 env var is set (same opt-in pattern as
test_integration.py).  These tests read from the test_intraday_prices table in
data/integration_test.db — the integration test script must have been run first.

Run:
    RUN_FULL_INTEGRATION=1 pytest tests/test_full_integration.py -v
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SKIP = not os.getenv("RUN_FULL_INTEGRATION")
_SKIP_REASON = "set RUN_FULL_INTEGRATION=1 to run (requires pre-populated integration_test.db)"

INTEGRATION_DB = Path("data/integration_test.db")
TABLE_PREFIX = "test_"
OUTPUT_DIR = Path("reports/integration_test")
SYMBOLS = ["SPY"]
START_DATE = "2023-01-01"

pytestmark = pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_store():
    from ait.data.historical import HistoricalDataStore
    assert INTEGRATION_DB.exists(), (
        f"Integration DB not found at {INTEGRATION_DB}. "
        "Run scripts/run_integration_test.py first."
    )
    return HistoricalDataStore(db_path=INTEGRATION_DB, table_prefix=TABLE_PREFIX)


# ---------------------------------------------------------------------------
# TestIntradayDataQuality
# ---------------------------------------------------------------------------

class TestIntradayDataQuality:
    """Verifies the test_intraday_prices table has sensible real-world data."""

    def test_db_exists(self) -> None:
        assert INTEGRATION_DB.exists(), f"DB not found: {INTEGRATION_DB}"

    def test_test_intraday_prices_table_exists(self) -> None:
        with sqlite3.connect(INTEGRATION_DB) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "test_intraday_prices" in tables, (
            "test_intraday_prices table missing — run run_integration_test.py first"
        )

    def test_production_intraday_prices_not_polluted(self) -> None:
        """Backfill must write to test_intraday_prices, NOT intraday_prices."""
        with sqlite3.connect(INTEGRATION_DB) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        # Production table should be absent (or empty) in the integration test DB
        if "intraday_prices" in tables:
            with sqlite3.connect(INTEGRATION_DB) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM intraday_prices"
                ).fetchone()[0]
            assert count == 0, (
                f"Production table intraday_prices has {count} rows in integration DB — "
                "test data must use test_intraday_prices only"
            )

    def test_row_count_at_least_70k(self) -> None:
        store = _get_store()
        count = store.row_count_intraday("SPY", interval="5m")
        assert count >= 70_000, (
            f"Expected ≥ 70,000 bars for 2-year SPY backfill, got {count}. "
            "Check IB connection and re-run backfill."
        )

    def test_date_range_spans_at_least_480_trading_days(self) -> None:
        store = _get_store()
        df = store.load_intraday("SPY", days=800, interval="5m")
        assert not df.empty, "No SPY intraday data found"
        trading_days = len(set(df.index.date))
        assert trading_days >= 480, (
            f"Expected ≥ 480 trading days (~2 years), got {trading_days}."
        )

    def test_coverage_at_least_80_pct(self) -> None:
        store = _get_store()
        df = store.load_intraday("SPY", days=800, interval="5m")
        assert not df.empty
        trading_days = len(set(df.index.date))
        expected_bars = trading_days * 78
        actual_bars = len(df)
        coverage = actual_bars / expected_bars
        assert coverage >= 0.80, (
            f"Intraday coverage {coverage:.1%} below 80% threshold "
            f"({actual_bars:,} / {expected_bars:,} expected bars)"
        )

    def test_no_negative_close_prices(self) -> None:
        store = _get_store()
        df = store.load_intraday("SPY", days=800, interval="5m")
        neg = (df["Close"] <= 0).sum()
        assert neg == 0, f"{neg} bars have Close ≤ 0 — data quality issue"

    def test_high_always_gte_low(self) -> None:
        store = _get_store()
        df = store.load_intraday("SPY", days=800, interval="5m")
        violations = (df["High"] < df["Low"]).sum()
        assert violations == 0, f"{violations} bars have High < Low — data quality issue"

    def test_index_is_utc(self) -> None:
        store = _get_store()
        df = store.load_intraday("SPY", days=30, interval="5m")
        assert df.index.tzinfo is not None, "DatetimeIndex must be timezone-aware (UTC)"


# ---------------------------------------------------------------------------
# TestFeatureHealth
# ---------------------------------------------------------------------------

class TestFeatureHealth:
    """Verifies all 18 fractal + VLMC features can be computed with low NaN rates."""

    @pytest.fixture(scope="class")
    def fractal_features(self):
        from ait.data.market_data import load_daily_ohlcv
        from ait.ml.features import FeatureEngine
        df = load_daily_ohlcv("SPY", days=730)
        assert len(df) >= 100, f"Insufficient daily data: {len(df)} rows"
        engine = FeatureEngine()
        return engine.compute(df)

    @pytest.fixture(scope="class")
    def vlmc_features(self):
        from ait.ml.features import FeatureEngine
        from ait.diagnostics.vlmc_report import VLMC_FEATURE_COLS
        store = _get_store()
        engine = FeatureEngine()
        df = store.load_intraday("SPY", days=800, interval="5m")
        assert not df.empty, "No intraday data for VLMC feature computation"
        rows = []
        for d in sorted(set(df.index.date)):
            session = df[df.index.date == d]
            if len(session) < 10:
                continue
            try:
                feats = engine.compute_intraday_features(session)
                rows.append({k: feats.get(k, np.nan) for k in VLMC_FEATURE_COLS})
            except Exception:
                pass
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=VLMC_FEATURE_COLS)

    def test_all_five_fractal_columns_present(self, fractal_features) -> None:
        from ait.diagnostics.fractal_report import FRACTAL_FEATURE_COLS
        missing = [c for c in FRACTAL_FEATURE_COLS if c not in fractal_features.columns]
        assert not missing, f"Missing fractal columns: {missing}"

    def test_all_thirteen_vlmc_columns_present(self, vlmc_features) -> None:
        from ait.diagnostics.vlmc_report import VLMC_FEATURE_COLS
        missing = [c for c in VLMC_FEATURE_COLS if c not in vlmc_features.columns]
        assert not missing, f"Missing VLMC columns: {missing}"

    def test_fractal_nan_rate_below_10pct(self, fractal_features) -> None:
        from ait.diagnostics.fractal_report import FRACTAL_FEATURE_COLS
        for col in FRACTAL_FEATURE_COLS:
            if col not in fractal_features.columns:
                continue
            nan_pct = 100 * fractal_features[col].isna().mean()
            assert nan_pct <= 10, (
                f"Fractal feature '{col}' has {nan_pct:.1f}% NaN — exceeds 10% threshold"
            )

    def test_vlmc_nan_rate_below_10pct(self, vlmc_features) -> None:
        from ait.diagnostics.vlmc_report import VLMC_FEATURE_COLS
        for col in VLMC_FEATURE_COLS:
            if col not in vlmc_features.columns:
                continue
            nan_pct = 100 * vlmc_features[col].isna().mean()
            assert nan_pct <= 10, (
                f"VLMC feature '{col}' has {nan_pct:.1f}% NaN — exceeds 10% threshold"
            )

    def test_hurst_wavelet_in_plausible_range(self, fractal_features) -> None:
        col = "hurst_wavelet"
        if col not in fractal_features.columns:
            pytest.skip(f"{col} not present")
        series = fractal_features[col].dropna()
        assert (series >= 0.0).all() and (series <= 1.0).all(), (
            f"hurst_wavelet values outside [0, 1]: min={series.min():.3f}, max={series.max():.3f}"
        )

    def test_session_volume_front_load_in_range(self, vlmc_features) -> None:
        col = "session_volume_front_load"
        if col not in vlmc_features.columns:
            pytest.skip(f"{col} not present")
        series = vlmc_features[col].dropna()
        assert (series >= 0.0).all() and (series <= 1.0).all(), (
            f"session_volume_front_load outside [0, 1]: min={series.min():.3f}, max={series.max():.3f}"
        )

    def test_vlmc_has_at_least_100_sessions(self, vlmc_features) -> None:
        assert len(vlmc_features) >= 100, (
            f"Only {len(vlmc_features)} VLMC sessions computed — expected ≥ 100 for 2-year window"
        )


# ---------------------------------------------------------------------------
# TestICAnalysis
# ---------------------------------------------------------------------------

class TestICAnalysis:
    """Validates that at least some features show predictive signal at 5-day horizon."""

    @pytest.fixture(scope="class")
    def ic_decay_csv(self):
        path = OUTPUT_DIR / "ic_decay.csv"
        if not path.exists():
            pytest.skip(f"ic_decay.csv not found at {path} — run integration test script first")
        return pd.read_csv(path)

    def test_ic_decay_csv_has_expected_columns(self, ic_decay_csv) -> None:
        required = {"symbol", "feature", "feature_type", "horizon_days", "ic", "p_value", "n_obs"}
        missing = required - set(ic_decay_csv.columns)
        assert not missing, f"ic_decay.csv missing columns: {missing}"

    def test_at_least_two_fractal_features_with_nonzero_ic_at_5d(self, ic_decay_csv) -> None:
        h5_fractal = ic_decay_csv[
            (ic_decay_csv["horizon_days"] == 5) &
            (ic_decay_csv["feature_type"] == "fractal")
        ]
        meaningful = (h5_fractal["ic"].abs() >= 0.02).sum()
        assert meaningful >= 2, (
            f"Only {meaningful}/5 fractal features show |IC| ≥ 0.02 at 5-day horizon. "
            "Feature engineering or data quality may need investigation."
        )

    def test_at_least_three_vlmc_features_with_p_below_0_10_at_5d(self, ic_decay_csv) -> None:
        h5_vlmc = ic_decay_csv[
            (ic_decay_csv["horizon_days"] == 5) &
            (ic_decay_csv["feature_type"] == "vlmc")
        ]
        sig = (h5_vlmc["p_value"] < 0.10).sum()
        assert sig >= 3, (
            f"Only {sig}/13 VLMC features have p < 0.10 at 5-day horizon. "
            "Consider checking session boundary logic or bar count per session."
        )

    def test_ic_values_are_bounded(self, ic_decay_csv) -> None:
        assert ic_decay_csv["ic"].between(-1.0, 1.0).all(), (
            "Some IC values are outside [-1, 1] — Spearman correlation must be bounded"
        )

    def test_all_five_horizons_present(self, ic_decay_csv) -> None:
        present = set(ic_decay_csv["horizon_days"].unique())
        expected = {1, 3, 5, 10, 20}
        missing = expected - present
        assert not missing, f"IC decay missing horizons: {missing}"


# ---------------------------------------------------------------------------
# TestWalkForwardResult
# ---------------------------------------------------------------------------

class TestWalkForwardResult:
    """Validates walk-forward completed without error and produced reasonable metrics."""

    @pytest.fixture(scope="class")
    def wf_summary(self):
        path = OUTPUT_DIR / "walkforward_summary.txt"
        if not path.exists():
            pytest.skip(f"walkforward_summary.txt not found — run integration test script first")
        return path.read_text()

    @pytest.fixture(scope="class")
    def equity_curve(self):
        path = OUTPUT_DIR / "equity_curve.csv"
        if not path.exists():
            pytest.skip("equity_curve.csv not found")
        return pd.read_csv(path)

    def test_walkforward_summary_file_exists(self, wf_summary) -> None:
        assert len(wf_summary) > 100, "walkforward_summary.txt appears empty"

    def test_equity_curve_has_required_columns(self, equity_curve) -> None:
        required = {"date", "equity", "pnl"}
        missing = required - set(equity_curve.columns)
        assert not missing, f"equity_curve.csv missing columns: {missing}"

    def test_at_least_one_trade_executed(self, equity_curve) -> None:
        assert len(equity_curve) > 0, (
            "No trades in equity curve — walk-forward produced no signals. "
            "Check min_confidence threshold and data coverage."
        )

    def test_equity_curve_equity_is_positive(self, equity_curve) -> None:
        assert (equity_curve["equity"] > 0).all(), (
            "Equity dropped to zero or below — catastrophic loss in walk-forward."
        )

    def test_walkforward_mentions_sharpe(self, wf_summary) -> None:
        assert "Sharpe" in wf_summary or "sharpe" in wf_summary.lower(), (
            "Walk-forward summary does not mention Sharpe ratio."
        )


# ---------------------------------------------------------------------------
# TestDiagnosticReports
# ---------------------------------------------------------------------------

class TestDiagnosticReports:
    """Validates that HTML diagnostic reports and IC CSV were produced."""

    def test_html_report_exists_for_spy(self) -> None:
        html = OUTPUT_DIR / "fractal_report_SPY.html"
        assert html.exists(), f"HTML report not found: {html}"

    def test_html_report_not_empty(self) -> None:
        html = OUTPUT_DIR / "fractal_report_SPY.html"
        if not html.exists():
            pytest.skip("HTML report not found")
        size_kb = html.stat().st_size / 1024
        assert size_kb >= 50, (
            f"fractal_report_SPY.html is only {size_kb:.0f} KB — expected ≥ 50 KB "
            "(suggests plots are missing)"
        )

    def test_html_report_contains_vlmc_section(self) -> None:
        html = OUTPUT_DIR / "fractal_report_SPY.html"
        if not html.exists():
            pytest.skip("HTML report not found")
        content = html.read_text()
        assert "VLMC" in content or "Session" in content, (
            "HTML report does not mention VLMC or Session — VLMC plots may be missing. "
            "Check that test_intraday_prices has data and db_path is correct."
        )

    def test_ic_summary_csv_exists(self) -> None:
        csv = OUTPUT_DIR / "ic_summary.csv"
        assert csv.exists(), f"ic_summary.csv not found: {csv}"

    def test_ic_summary_has_required_columns(self) -> None:
        csv = OUTPUT_DIR / "ic_summary.csv"
        if not csv.exists():
            pytest.skip("ic_summary.csv not found")
        df = pd.read_csv(csv)
        required = {"symbol", "feature", "feature_type", "ic", "p_value", "n_obs"}
        missing = required - set(df.columns)
        assert not missing, f"ic_summary.csv missing columns: {missing}"

    def test_ic_summary_has_at_least_ten_rows(self) -> None:
        csv = OUTPUT_DIR / "ic_summary.csv"
        if not csv.exists():
            pytest.skip("ic_summary.csv not found")
        df = pd.read_csv(csv)
        assert len(df) >= 10, (
            f"ic_summary.csv has only {len(df)} rows — expected ≥ 10 (5 fractal + VLMC features)"
        )

    def test_results_md_exists(self) -> None:
        md = OUTPUT_DIR / "RESULTS.md"
        assert md.exists(), f"RESULTS.md not found: {md}"

    def test_data_quality_txt_exists(self) -> None:
        txt = OUTPUT_DIR / "data_quality.txt"
        assert txt.exists(), f"data_quality.txt not found: {txt}"
