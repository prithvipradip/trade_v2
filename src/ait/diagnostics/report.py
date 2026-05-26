"""Report orchestrator — generates per-symbol HTML + IC-CSV diagnostics.

Combines fractal features (fractal_report) and VLMC session structure
features (vlmc_report) into a single HTML file per symbol plus ic_summary.csv.

Usage:
    from ait.diagnostics.report import generate_report
    generate_report(["SPY", "QQQ"], "2022-01-01", "2025-12-31", "reports/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    _PLOTLY = True
    import plotly.graph_objects as go  # noqa: F401
except ImportError:
    _PLOTLY = False

from ait.diagnostics.fractal_report import (
    FRACTAL_FEATURE_COLS,
    plot_hurst_timeseries,
    plot_ic_analysis,
    plot_multifractal_spectrum,
    plot_psd,
    plot_scale_invariance_vs_vix,
)
from ait.diagnostics.vlmc_report import (
    VLMC_FEATURE_COLS,
    plot_power_hour_patterns,
    plot_session_feature_ic_analysis,
    plot_session_vwap_trajectory,
    plot_volume_profile_distribution,
)
from ait.utils.logging import get_logger

log = get_logger("diagnostics.report")


def generate_report(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: str,
    fmt: str = "html",
    db_path: str = "data/historical.db",
    table_prefix: str = "",
) -> None:
    """Generate fractal + VLMC diagnostic report for each symbol.

    Fetches daily data via Yahoo Finance, computes fractal features,
    loads intraday data from SQLite for VLMC plots, and writes one HTML
    file per symbol plus ic_summary.csv.

    Args:
        symbols:    List of ticker symbols (e.g., ["SPY", "QQQ"]).
        start:      Start date string "YYYY-MM-DD".
        end:        End date string "YYYY-MM-DD".
        output_dir: Directory to write output files.
        fmt:          Output format — "html" (interactive) or "png" (static).
        db_path:      Path to the SQLite historical database.
        table_prefix: Table prefix passed to HistoricalDataStore, e.g. "test_".
    """
    from ait.data.market_data import load_daily_ohlcv
    from ait.ml.features import FeatureEngine

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    store_path = Path(db_path)
    intraday_store = None
    try:
        from ait.data.historical import HistoricalDataStore
        if store_path.exists():
            intraday_store = HistoricalDataStore(db_path=store_path, table_prefix=table_prefix)
    except Exception as exc:
        log.warning("intraday_store_unavailable", error=str(exc))

    ic_rows: list[dict] = []

    for symbol in symbols:
        log.info("generating_report", symbol=symbol)

        try:
            from datetime import date as _date
            # Fetch enough history to cover start date from today
            days = (_date.today() - _date.fromisoformat(start)).days + 30
            df = load_daily_ohlcv(symbol, days=days, db_path=store_path)
            if df is not None and not df.empty:
                df = df.loc[start:end]
                df.index.name = "Date"

            # If IB store doesn't cover the requested date range, fetch from Yahoo directly
            if df is None or len(df) < 50:
                import yfinance as _yf
                ydf = _yf.Ticker(symbol).history(start=start, end=end, interval="1d")
                if ydf is not None and not ydf.empty:
                    df = ydf[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index.name = "Date"

            if df is None or df.empty:
                log.warning("no_data_for_symbol", symbol=symbol)
                continue
        except Exception as exc:
            log.warning("fetch_failed", symbol=symbol, error=str(exc))
            continue

        if len(df) < 50:
            log.warning("insufficient_rows", symbol=symbol, rows=len(df))
            continue

        try:
            engine = FeatureEngine()
            features_df = engine.compute(df)
        except Exception as exc:
            log.warning("feature_compute_failed", symbol=symbol, error=str(exc))
            continue

        if features_df.empty:
            continue

        features_df["fwd_return_5d"] = (
            df["Close"].pct_change(5).shift(-5).reindex(features_df.index)
        )

        vix_df = None
        try:
            vix_df = yf.Ticker("^VIX").history(start=start, end=end, interval="1d")
            if vix_df is not None and not vix_df.empty:
                vix_df = vix_df[["Close"]].copy()
        except Exception:
            pass

        intraday_df: pd.DataFrame | None = None
        if intraday_store is not None:
            try:
                intraday_df = intraday_store.load_intraday(symbol, days=730)
                if intraday_df is not None and intraday_df.empty:
                    intraday_df = None
                if intraday_df is not None:
                    log.debug("intraday_loaded_for_vlmc", symbol=symbol, bars=len(intraday_df))
            except Exception as exc:
                log.warning("intraday_load_failed", symbol=symbol, error=str(exc))

        labels = features_df["fwd_return_5d"].dropna()
        for col in FRACTAL_FEATURE_COLS + VLMC_FEATURE_COLS:
            if col not in features_df.columns:
                continue
            try:
                from scipy.stats import spearmanr
                x = features_df[col].reindex(labels.index).dropna()
                y = labels.reindex(x.index)
                if len(x) >= 5:
                    corr, pval = spearmanr(x.values, y.values)
                    ic_rows.append({
                        "symbol": symbol,
                        "feature": col,
                        "feature_type": "fractal" if col in FRACTAL_FEATURE_COLS else "vlmc",
                        "ic": float(corr) if np.isfinite(corr) else 0.0,
                        "p_value": float(pval),
                        "n_obs": len(x),
                    })
            except Exception:
                pass

        if fmt == "html" and _PLOTLY:
            _write_html_report(symbol, features_df, df, vix_df, intraday_df, out)
        else:
            log.info("skipping_plot_output", symbol=symbol,
                     reason="plotly not available or fmt!=html")

    ic_df = pd.DataFrame(ic_rows)
    ic_path = out / "ic_summary.csv"
    ic_df.to_csv(ic_path, index=False)
    log.info("ic_summary_written", path=str(ic_path), rows=len(ic_rows))


def _write_html_report(
    symbol: str,
    features_df: pd.DataFrame,
    price_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    intraday_df: pd.DataFrame | None,
    out: Path,
) -> None:
    """Assemble fractal + VLMC plots into a single per-symbol HTML report."""
    from plotly.io import to_html

    returns = np.diff(np.log(price_df["Close"].values + 1e-12))
    labels = features_df.get("fwd_return_5d", pd.Series(dtype=float))

    fractal_plots = [
        plot_hurst_timeseries(symbol, features_df),
        plot_psd(returns),
        plot_multifractal_spectrum(returns),
        plot_scale_invariance_vs_vix(features_df, vix_df if vix_df is not None else pd.DataFrame()),
        plot_ic_analysis(features_df, labels),
    ]

    vlmc_plots: list[Any] = []
    if intraday_df is not None and not intraday_df.empty:
        vlmc_plots = [
            plot_session_vwap_trajectory(symbol, intraday_df),
            plot_volume_profile_distribution(symbol, intraday_df),
            plot_session_feature_ic_analysis(features_df, labels),
            plot_power_hour_patterns(symbol, intraday_df),
        ]

    def _fig_to_html(fig: Any) -> str:
        if fig is not None and hasattr(fig, "to_html"):
            return to_html(fig, full_html=False, include_plotlyjs="cdn")
        if isinstance(fig, dict) and "title" in fig:
            return f"<p><em>{fig['title']}</em></p>"
        return ""

    html_parts = [
        f"<html><head><title>Fractal + VLMC Report — {symbol}</title></head><body>",
        f"<h1>Fractal &amp; VLMC Diagnostic Report: {symbol}</h1>",
        "<h2>Fractal Features</h2>",
    ]
    for fig in fractal_plots:
        html_parts.append(_fig_to_html(fig))

    if vlmc_plots:
        html_parts.append("<h2>VLMC Session Structure Features</h2>")
        for fig in vlmc_plots:
            html_parts.append(_fig_to_html(fig))
    else:
        html_parts.append(
            "<p><em>VLMC plots unavailable — run backfill_intraday.py to populate "
            "the intraday_prices table.</em></p>"
        )

    html_parts.append("</body></html>")

    report_path = out / f"fractal_report_{symbol}.html"
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    log.info("html_report_written", symbol=symbol, path=str(report_path))
