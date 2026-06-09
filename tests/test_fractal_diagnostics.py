"""Smoke tests for the fractal diagnostic report module and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestFractalReportImport:

    def test_module_importable(self) -> None:
        from ait.diagnostics import fractal_report  # noqa: F401

    def test_all_seven_plot_functions_exist(self) -> None:
        from ait.diagnostics import fractal_report
        for fn in (
            "plot_hurst_timeseries",
            "plot_psd",
            "plot_multifractal_spectrum",
            "plot_scale_invariance_vs_vix",
            "plot_ic_analysis",
            "plot_shap_importance",
            "plot_gate_counterfactual",
        ):
            assert callable(getattr(fractal_report, fn, None)), (
                f"Function {fn!r} missing from ait.diagnostics.fractal_report"
            )

    def test_generate_report_callable(self) -> None:
        from ait.diagnostics.report import generate_report
        assert callable(generate_report)


class TestFractalReportSmoke:

    def _features_df(self, rows: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(77)
        dates = pd.date_range("2023-01-01", periods=rows, freq="B")
        return pd.DataFrame({
            "hurst_wavelet":       rng.uniform(0.4, 0.7, rows),
            "hurst_scale_spread":  rng.uniform(0.0, 0.2, rows),
            "psd_beta":            rng.uniform(1.5, 2.5, rows),
            "multifractal_width":  rng.uniform(0.1, 0.5, rows),
            "fwd_return_5d":       rng.normal(0.001, 0.02, rows),
        }, index=dates)

    def test_plot_hurst_timeseries_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_hurst_timeseries
        fig = plot_hurst_timeseries("SPY", self._features_df())
        assert fig is not None

    def test_plot_psd_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_psd
        fig = plot_psd(np.random.default_rng(88).normal(0, 1, 500))
        assert fig is not None

    def test_plot_ic_analysis_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_ic_analysis
        df = self._features_df()
        fig = plot_ic_analysis(df, pd.Series(df["fwd_return_5d"].values, index=df.index))
        assert fig is not None

    def test_generate_report_creates_html(self, tmp_path: Path) -> None:
        from ait.diagnostics.report import generate_report
        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
        )
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1, "generate_report must produce at least one .html file"

    def test_generate_report_creates_ic_csv(self, tmp_path: Path) -> None:
        from ait.diagnostics.report import generate_report
        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
        )
        csv_files = list(tmp_path.glob("ic_summary.csv"))
        assert len(csv_files) == 1, "generate_report must produce ic_summary.csv"


class TestCLIEntryPoint:

    def test_cli_script_exists(self) -> None:
        assert Path("scripts/run_fractal_diagnostics.py").exists(), (
            "scripts/run_fractal_diagnostics.py must exist (Phase 8)"
        )

    def test_cli_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_fractal_diagnostics.py", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"CLI --help returned non-zero:\n{result.stderr}"
        )

    def test_cli_requires_symbols_argument(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_fractal_diagnostics.py"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, (
            "CLI must fail when called without --symbols"
        )
