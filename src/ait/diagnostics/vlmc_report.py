"""VLMC session structure diagnostic plots.

Validates the 13 intraday session features computed by FeatureEngine.compute_intraday_features().
Standalone — has no effect on the trading pipeline.

Requires intraday data to be populated via scripts/backfill_intraday.py before running.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
    go = None  # type: ignore[assignment]

from ait.utils.logging import get_logger

log = get_logger("diagnostics.vlmc_report")

VLMC_FEATURE_COLS = [
    "session_vwap_position",
    "session_vwap_q1", "session_vwap_q2", "session_vwap_q3",
    "session_high_timing", "session_low_timing",
    "session_volume_front_load", "session_volume_shape",
    "power_hour_momentum", "power_hour_vol_accel", "power_hour_vwap_cross",
    "closing_imbalance", "closing_range_position",
]


def _fallback_figure(title: str) -> dict:
    return {"title": title, "data": []}


def _session_dates(intraday_df: pd.DataFrame) -> list:
    """Return sorted unique trading dates from a multi-session intraday DataFrame."""
    return sorted(
        pd.Series(
            intraday_df.index.map(
                lambda x: x.date() if hasattr(x, "date") else pd.Timestamp(x).date()
            )
        ).unique()
    )


def _session_mask(intraday_df: pd.DataFrame, d) -> "pd.Series[bool]":
    return intraday_df.index.map(
        lambda x: (x.date() if hasattr(x, "date") else pd.Timestamp(x).date()) == d
    )


def plot_session_vwap_trajectory(symbol: str, intraday_df: pd.DataFrame) -> Any:
    """Rolling session VWAP position across all stored sessions.

    Aggregates session_vwap_position (close vs. session VWAP) per trading day.
    Trending markets show persistent above/below VWAP closing.
    Mean-reverting markets oscillate around zero.
    Systematic bias (e.g., always below VWAP) may indicate a session-boundary bug.
    """
    if not _PLOTLY:
        return _fallback_figure(f"Session VWAP Trajectory — {symbol}")
    if intraday_df is None or intraday_df.empty:
        return _fallback_figure(f"Session VWAP Trajectory — {symbol} (no data)")

    try:
        from ait.ml.features import FeatureEngine
        engine = FeatureEngine()

        sessions: list[dict] = []
        for d in _session_dates(intraday_df):
            session = intraday_df[_session_mask(intraday_df, d)]
            if len(session) < 10:
                continue
            feats = engine.compute_intraday_features(session)
            sessions.append({
                "date": pd.Timestamp(d),
                "vwap_pos": feats.get("session_vwap_position", 0.0),
            })

        if not sessions:
            return _fallback_figure(f"Session VWAP Trajectory — {symbol} (insufficient sessions)")

        sess_df = pd.DataFrame(sessions).set_index("date")
        rolling_mean = sess_df["vwap_pos"].rolling(20, min_periods=1).mean()
        rolling_std = sess_df["vwap_pos"].rolling(20, min_periods=1).std().fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sess_df.index, y=sess_df["vwap_pos"],
            mode="markers", marker={"size": 4, "color": "lightsteelblue"},
            name="Daily VWAP position",
        ))
        fig.add_trace(go.Scatter(
            x=sess_df.index, y=rolling_mean,
            mode="lines", line={"color": "royalblue", "width": 2},
            name="20-session rolling mean",
        ))
        fig.add_trace(go.Scatter(
            x=list(sess_df.index) + list(sess_df.index[::-1]),
            y=list(rolling_mean + rolling_std) + list((rolling_mean - rolling_std)[::-1]),
            fill="toself", fillcolor="rgba(65,105,225,0.1)",
            line={"color": "rgba(255,255,255,0)"}, name="±1σ band",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            title=f"Session VWAP Trajectory — {symbol}",
            xaxis_title="Date", yaxis_title="(Close − VWAP) / VWAP",
            template="plotly_white",
        )
        return fig

    except Exception as exc:
        log.warning("plot_session_vwap_trajectory_failed", error=str(exc))
        return _fallback_figure(f"Session VWAP Trajectory — {symbol} (error)")


def plot_volume_profile_distribution(symbol: str, intraday_df: pd.DataFrame) -> Any:
    """Histogram of session_volume_front_load and session_volume_shape across sessions.

    session_volume_front_load > 0.45 is typical for US equities.
    Values concentrated at 0.33 suggest the session boundary filter is broken
    (treating the full multi-day window as one session rather than per-day).
    Heavy back-loading > 0.6 may indicate ETF creation/redemption artifacts.
    """
    if not _PLOTLY:
        return _fallback_figure(f"Volume Profile Distribution — {symbol}")
    if intraday_df is None or intraday_df.empty:
        return _fallback_figure(f"Volume Profile Distribution — {symbol} (no data)")

    try:
        from ait.ml.features import FeatureEngine
        engine = FeatureEngine()

        front_loads, shapes = [], []
        for d in _session_dates(intraday_df):
            session = intraday_df[_session_mask(intraday_df, d)]
            if len(session) < 10:
                continue
            feats = engine.compute_intraday_features(session)
            front_loads.append(feats.get("session_volume_front_load", np.nan))
            shapes.append(feats.get("session_volume_shape", np.nan))

        if not front_loads:
            return _fallback_figure(f"Volume Profile Distribution — {symbol} (no sessions)")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Front-load ratio", "Shape (front − back)"))
        fig.add_trace(
            go.Histogram(x=[v for v in front_loads if np.isfinite(v)], nbinsx=20,
                         name="Front-load", marker_color="steelblue"),
            row=1, col=1,
        )
        fig.add_vline(x=0.45, line_dash="dot", line_color="firebrick",
                      annotation_text="0.45 typical", row=1, col=1)
        fig.add_trace(
            go.Histogram(x=[v for v in shapes if np.isfinite(v)], nbinsx=20,
                         name="Shape", marker_color="darkorange"),
            row=1, col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="grey", row=1, col=2)
        fig.update_layout(
            title=f"Volume Profile Distribution — {symbol} ({len(front_loads)} sessions)",
            template="plotly_white", showlegend=False,
        )
        return fig

    except Exception as exc:
        log.warning("plot_volume_profile_distribution_failed", error=str(exc))
        return _fallback_figure(f"Volume Profile Distribution — {symbol} (error)")


def plot_session_feature_ic_analysis(
    features_df: pd.DataFrame, labels: pd.Series
) -> Any:
    """IC (Spearman ρ) bar chart for all 13 VLMC session features vs. 5-day forward return.

    power_hour_momentum and closing_imbalance expected to show highest |IC| (0.04–0.10).
    Features with |IC| < 0.02 and p > 0.05 should be flagged for potential removal.
    Dark blue bars = statistically significant (p < 0.05); light = not significant.
    """
    if not _PLOTLY:
        return _fallback_figure("VLMC Session Feature IC Analysis")

    vlmc_cols = [c for c in VLMC_FEATURE_COLS if c in features_df.columns]
    if not vlmc_cols:
        return _fallback_figure("VLMC Session Feature IC Analysis (no VLMC columns found)")

    ics: dict[str, float] = {}
    pvals: dict[str, float] = {}
    for col in vlmc_cols:
        try:
            from scipy.stats import spearmanr
            aligned = features_df[col].align(labels, join="inner")
            x, y = aligned[0].dropna(), aligned[1].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 5:
                ics[col] = 0.0
                pvals[col] = 1.0
                continue
            corr, pval = spearmanr(x.loc[common].values, y.loc[common].values)
            ics[col] = float(corr) if np.isfinite(corr) else 0.0
            pvals[col] = float(pval)
        except Exception:
            ics[col] = 0.0
            pvals[col] = 1.0

    cols_sorted = sorted(ics, key=lambda k: abs(ics[k]), reverse=True)
    ic_vals = [ics[c] for c in cols_sorted]
    colors = ["steelblue" if pvals[c] < 0.05 else "lightsteelblue" for c in cols_sorted]

    fig = go.Figure(go.Bar(
        x=cols_sorted, y=ic_vals, marker_color=colors,
        text=[f"p={pvals[c]:.2f}" for c in cols_sorted],
        textposition="outside", name="IC",
    ))
    fig.add_hline(y=0.05, line_dash="dot", line_color="green",
                  annotation_text="|IC|=0.05 threshold")
    fig.add_hline(y=-0.05, line_dash="dot", line_color="green")
    fig.update_layout(
        title="Information Coefficient (Spearman) — VLMC Session Features<br>"
              "<sup>Dark blue = p<0.05 significant; light = not significant</sup>",
        xaxis_title="Feature", yaxis_title="IC",
        template="plotly_white",
    )
    return fig


def plot_power_hour_patterns(symbol: str, intraday_df: pd.DataFrame) -> Any:
    """Scatter: power_hour_momentum vs. next-session open return per trading day.

    Colour-coded by power_hour_vol_accel (red = accelerating, blue = decelerating).
    Should show a mild positive slope (continuation) in trending markets.
    A flat scatter with no slope means power_hour_momentum has no predictive content
    for this symbol — expected for highly mean-reverting assets.
    """
    if not _PLOTLY:
        return _fallback_figure(f"Power Hour Patterns — {symbol}")
    if intraday_df is None or intraday_df.empty:
        return _fallback_figure(f"Power Hour Patterns — {symbol} (no data)")

    try:
        from ait.ml.features import FeatureEngine
        engine = FeatureEngine()

        dates = _session_dates(intraday_df)
        sessions: list[dict] = []

        for i, d in enumerate(dates[:-1]):
            next_d = dates[i + 1]
            session = intraday_df[_session_mask(intraday_df, d)]
            if len(session) < 10:
                continue
            feats = engine.compute_intraday_features(session)

            next_session = intraday_df[_session_mask(intraday_df, next_d)]
            if next_session.empty:
                continue

            next_open = float(next_session["Open"].iloc[0])
            today_close = float(session["Close"].iloc[-1])
            next_open_ret = (next_open / today_close - 1.0) if today_close > 0 else 0.0

            sessions.append({
                "date": pd.Timestamp(d),
                "ph_momentum": feats.get("power_hour_momentum", 0.0),
                "ph_vol_accel": feats.get("power_hour_vol_accel", 0.0),
                "next_open_ret": next_open_ret,
            })

        if not sessions:
            return _fallback_figure(f"Power Hour Patterns — {symbol} (insufficient sessions)")

        sess_df = pd.DataFrame(sessions)
        accel = sess_df["ph_vol_accel"].values
        accel_norm = (accel - accel.min()) / (accel.ptp() + 1e-10)
        colors = [
            f"rgba({int(255*v)},{int(50*(1-v))},{int(255*(1-v))},0.7)"
            for v in accel_norm
        ]

        fig = go.Figure(go.Scatter(
            x=sess_df["ph_momentum"], y=sess_df["next_open_ret"],
            mode="markers",
            marker={"size": 7, "color": colors},
            text=sess_df["date"].astype(str),
            name="Sessions",
        ))

        x_vals = sess_df["ph_momentum"].values
        y_vals = sess_df["next_open_ret"].values
        if len(x_vals) > 3:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines", line={"color": "black", "dash": "dash"},
                name=f"Trend (slope={slope:.3f})",
            ))

        fig.update_layout(
            title=f"Power Hour Momentum vs Next-Session Open Return — {symbol}<br>"
                  "<sup>Color = vol acceleration (red=accelerating, blue=decelerating)</sup>",
            xaxis_title="Power Hour Momentum (log return)",
            yaxis_title="Next-Session Open Return",
            template="plotly_white",
        )
        return fig

    except Exception as exc:
        log.warning("plot_power_hour_patterns_failed", error=str(exc))
        return _fallback_figure(f"Power Hour Patterns — {symbol} (error)")
