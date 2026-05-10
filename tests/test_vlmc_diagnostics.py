"""Smoke tests for the VLMC session structure diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestVLMCReportImport:

    def test_module_importable(self) -> None:
        from ait.diagnostics import vlmc_report  # noqa: F401

    def test_all_four_vlmc_plot_functions_exist(self) -> None:
        from ait.diagnostics import vlmc_report
        for fn in (
            "plot_session_vwap_trajectory",
            "plot_volume_profile_distribution",
            "plot_session_feature_ic_analysis",
            "plot_power_hour_patterns",
        ):
            assert callable(getattr(vlmc_report, fn, None)), (
                f"Function {fn!r} missing from ait.diagnostics.vlmc_report"
            )


class TestVLMCDiagnosticPlots:

    def _make_intraday_multi_session(self, n_sessions: int = 20) -> pd.DataFrame:
        """Build n_sessions × 78 bars of synthetic intraday data."""
        from datetime import datetime, timezone, timedelta
        bars_per_session = 78
        start = datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)
        idx = []
        day_offset = 0
        session_count = 0
        while session_count < n_sessions:
            candidate = start + timedelta(days=day_offset)
            if candidate.weekday() < 5:
                idx.extend([
                    candidate + timedelta(minutes=5 * bar)
                    for bar in range(bars_per_session)
                ])
                session_count += 1
            day_offset += 1
        idx = pd.DatetimeIndex(idx, tz="UTC")
        n = len(idx)
        price = 500.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.001, n)))
        return pd.DataFrame({
            "Open": price * 0.999, "High": price * 1.001,
            "Low": price * 0.999, "Close": price,
            "Volume": np.random.default_rng(1).integers(5000, 50000, n),
        }, index=idx)

    def _features_df_with_vlmc(self, rows: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=rows, freq="B")
        return pd.DataFrame({
            "session_vwap_position":     rng.normal(0.0, 0.01, rows),
            "session_vwap_q1":           rng.normal(0.0, 0.005, rows),
            "session_vwap_q2":           rng.normal(0.0, 0.005, rows),
            "session_vwap_q3":           rng.normal(0.0, 0.005, rows),
            "session_high_timing":       rng.uniform(0.0, 1.0, rows),
            "session_low_timing":        rng.uniform(0.0, 1.0, rows),
            "session_volume_front_load": rng.uniform(0.3, 0.6, rows),
            "session_volume_shape":      rng.uniform(-0.1, 0.1, rows),
            "power_hour_momentum":       rng.normal(0.0, 0.005, rows),
            "power_hour_vol_accel":      rng.normal(0.0, 0.1, rows),
            "power_hour_vwap_cross":     rng.choice([-1.0, 1.0], rows),
            "closing_imbalance":         rng.normal(0.0, 0.003, rows),
            "closing_range_position":    rng.uniform(0.2, 0.8, rows),
            "fwd_return_5d":             rng.normal(0.001, 0.02, rows),
        }, index=dates)

    def test_plot_session_vwap_trajectory_returns_object(self) -> None:
        from ait.diagnostics.vlmc_report import plot_session_vwap_trajectory
        fig = plot_session_vwap_trajectory("SPY", self._make_intraday_multi_session())
        assert fig is not None

    def test_plot_volume_profile_distribution_returns_object(self) -> None:
        from ait.diagnostics.vlmc_report import plot_volume_profile_distribution
        fig = plot_volume_profile_distribution("SPY", self._make_intraday_multi_session())
        assert fig is not None

    def test_plot_session_feature_ic_analysis_returns_object(self) -> None:
        from ait.diagnostics.vlmc_report import plot_session_feature_ic_analysis
        df = self._features_df_with_vlmc()
        fig = plot_session_feature_ic_analysis(
            df, pd.Series(df["fwd_return_5d"].values, index=df.index)
        )
        assert fig is not None

    def test_plot_power_hour_patterns_returns_object(self) -> None:
        from ait.diagnostics.vlmc_report import plot_power_hour_patterns
        fig = plot_power_hour_patterns("SPY", self._make_intraday_multi_session())
        assert fig is not None

    def test_generate_report_with_intraday_db_creates_html(self, tmp_path: Path) -> None:
        """generate_report() produces HTML even when intraday DB is missing."""
        from ait.diagnostics.report import generate_report
        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
            db_path=str(tmp_path / "nonexistent.db"),
        )
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1

    def test_generate_report_with_real_intraday_db_includes_vlmc(self, tmp_path: Path) -> None:
        """generate_report() includes VLMC plots when intraday data is in the DB."""
        from ait.data.historical import HistoricalDataStore
        from ait.diagnostics.report import generate_report

        store = HistoricalDataStore(db_path=tmp_path / "test.db")
        intraday = self._make_intraday_multi_session(30)
        store.save_intraday("SPY", intraday, interval="5m")

        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
            db_path=str(tmp_path / "test.db"),
        )
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1
        content = html_files[0].read_text()
        assert "VLMC" in content or "Session" in content, (
            "HTML report should contain VLMC section when intraday data is available"
        )
