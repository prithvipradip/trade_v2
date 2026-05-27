"""Fractal feature diagnostic plots.

Validates the 5 fractal/scaling features computed by FeatureEngine.compute().
Standalone — has no effect on the trading pipeline.
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

log = get_logger("diagnostics.fractal_report")

FRACTAL_FEATURE_COLS = [
    "hurst_wavelet",
    "hurst_scale_spread",
    "psd_beta",
    "multifractal_width",
    "multifractal_asymmetry",
]


def _fallback_figure(title: str) -> dict:
    return {"title": title, "data": []}


def plot_hurst_timeseries(symbol: str, features_df: pd.DataFrame) -> Any:
    """Plot Hurst exponent and scale-spread over time for a symbol.

    Confirms H stays in [0.3, 0.7] and that spikes align with known volatile periods.
    """
    if not _PLOTLY:
        return _fallback_figure(f"Hurst Timeseries — {symbol}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Wavelet Hurst (H)", "Hurst Scale Spread"))

    x = features_df.index
    if "hurst_wavelet" in features_df.columns:
        h = features_df["hurst_wavelet"]
        fig.add_trace(go.Scatter(x=x, y=h, name="H (wavelet)", line={"color": "royalblue"}),
                      row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey", row=1, col=1)
        fig.add_hrect(y0=0.3, y1=0.7, fillcolor="lightgreen", opacity=0.1,
                      line_width=0, row=1, col=1)

    if "hurst_scale_spread" in features_df.columns:
        spread = features_df["hurst_scale_spread"]
        fig.add_trace(go.Scatter(x=x, y=spread, name="Scale spread",
                                 line={"color": "tomato"}), row=2, col=1)

    fig.update_layout(title=f"Hurst Timeseries — {symbol}", height=500,
                      showlegend=True, template="plotly_white",
                      legend={"orientation": "h", "yanchor": "top", "y": -0.15,
                              "xanchor": "center", "x": 0.5})
    return fig


def plot_psd(returns: np.ndarray) -> Any:
    """Plot power spectral density of returns with fitted power-law slope.

    β ≈ 2 for Brownian motion; R² > 0.7 indicates a valid estimate.
    """
    if not _PLOTLY:
        return _fallback_figure("PSD")

    try:
        from scipy.signal import periodogram
        freqs, psd = periodogram(returns)
        mask = freqs > 0
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-20)

        slope, intercept = np.polyfit(log_f, log_p, 1)
        beta = -slope
        fit_line = slope * log_f + intercept

        ss_res = np.sum((log_p - fit_line) ** 2)
        ss_tot = np.sum((log_p - log_p.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_f, y=log_p, mode="markers",
                                 marker={"size": 3, "color": "steelblue"},
                                 name="PSD (log-log)"))
        fig.add_trace(go.Scatter(x=log_f, y=fit_line, mode="lines",
                                 line={"color": "firebrick", "dash": "dash"},
                                 name=f"Fit: β={beta:.2f}, R²={r2:.2f}"))
        fig.update_layout(title=f"Power Spectral Density (β={beta:.2f}, R²={r2:.2f})",
                          xaxis_title="log10(frequency)",
                          yaxis_title="log10(PSD)",
                          template="plotly_white",
                          legend={"orientation": "h", "yanchor": "top", "y": -0.20,
                                  "xanchor": "center", "x": 0.5})
        return fig

    except Exception as exc:
        log.warning("plot_psd_failed", error=str(exc))
        return _fallback_figure("PSD (error)")


def plot_multifractal_spectrum(returns: np.ndarray) -> Any:
    """Plot the multifractal f(α) spectrum.

    A bell-shaped curve confirms true multifractality; width widens during volatile windows.
    """
    if not _PLOTLY:
        return _fallback_figure("Multifractal Spectrum")

    try:
        q_vals = np.array([-5, -3, -1, 1, 3, 5], dtype=float)
        s_arr = []

        if len(returns) < 200:
            return _fallback_figure("Multifractal Spectrum (insufficient data)")

        scales = [10, 20, 40, 80]
        for q in q_vals:
            fluctuations = []
            for s in scales:
                segments = len(returns) // s
                if segments < 4:
                    fluctuations.append(1e-10)
                    continue
                f_q = []
                for seg in range(segments):
                    chunk = returns[seg * s:(seg + 1) * s]
                    trend = np.polyfit(np.arange(len(chunk)), chunk, 1)
                    residual = chunk - np.polyval(trend, np.arange(len(chunk)))
                    rms = np.sqrt(np.mean(residual ** 2)) + 1e-12
                    f_q.append(rms)
                fluctuations.append(np.mean(np.array(f_q) ** q) ** (1.0 / q) if q != 0 else
                                    np.exp(np.mean(np.log(f_q))))
            x = np.log2(scales)
            y = np.log2(np.clip(fluctuations, 1e-12, None))
            h_q, _ = np.polyfit(x, y, 1)
            s_arr.append(float(h_q))

        h_arr = np.array(s_arr)
        tau = q_vals * h_arr - 1
        alpha = np.gradient(tau, q_vals)
        f_alpha = q_vals * alpha - tau

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=alpha, y=f_alpha, mode="lines+markers",
                                 line={"color": "purple"}, name="f(α)"))
        fig.update_layout(title="Multifractal Spectrum f(α)",
                          xaxis_title="α (singularity strength)",
                          yaxis_title="f(α) (spectrum)",
                          template="plotly_white",
                          legend={"orientation": "h", "yanchor": "top", "y": -0.20,
                                  "xanchor": "center", "x": 0.5})
        return fig

    except Exception as exc:
        log.warning("plot_multifractal_spectrum_failed", error=str(exc))
        return _fallback_figure("Multifractal Spectrum (error)")


def plot_scale_invariance_vs_vix(features_df: pd.DataFrame, vix_df: pd.DataFrame) -> Any:
    """Plot hurst_scale_spread alongside VIX to verify leading indicator relationship.

    Scale spread should lead VIX spikes by 1–5 days.
    """
    if not _PLOTLY:
        return _fallback_figure("Scale Invariance vs VIX")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Hurst Scale Spread", "VIX"))

    if "hurst_scale_spread" in features_df.columns:
        fig.add_trace(go.Scatter(x=features_df.index, y=features_df["hurst_scale_spread"],
                                 name="Scale Spread", line={"color": "darkorange"}), row=1, col=1)

    if vix_df is not None and not vix_df.empty:
        vix_close = vix_df["Close"] if "Close" in vix_df.columns else vix_df.iloc[:, 0]
        fig.add_trace(go.Scatter(x=vix_df.index, y=vix_close,
                                 name="VIX", line={"color": "crimson"}), row=2, col=1)

    fig.update_layout(title="Scale Invariance vs VIX", height=500,
                      template="plotly_white",
                      legend={"orientation": "h", "yanchor": "top", "y": -0.15,
                              "xanchor": "center", "x": 0.5})
    return fig


def plot_ic_analysis(features_df: pd.DataFrame, labels: pd.Series) -> Any:
    """Bar chart of Information Coefficient (Spearman rank correlation) per fractal feature.

    Features with |IC| > 0.05 are considered predictively meaningful.
    """
    if not _PLOTLY:
        return _fallback_figure("IC Analysis")

    fractal_cols = [c for c in FRACTAL_FEATURE_COLS if c in features_df.columns]
    if not fractal_cols:
        fractal_cols = [c for c in features_df.columns if c != "fwd_return_5d"]

    ics = {}
    for col in fractal_cols:
        aligned = features_df[col].align(labels, join="inner")
        x, y = aligned[0].dropna(), aligned[1].dropna()
        common = x.index.intersection(y.index)
        if len(common) < 5:
            ics[col] = 0.0
            continue
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x.loc[common].values, y.loc[common].values)
        ics[col] = float(corr) if np.isfinite(corr) else 0.0

    cols_sorted = sorted(ics, key=lambda k: abs(ics[k]), reverse=True)
    ic_vals = [ics[c] for c in cols_sorted]
    colors = ["steelblue" if v >= 0 else "tomato" for v in ic_vals]

    fig = go.Figure(go.Bar(x=cols_sorted, y=ic_vals, marker_color=colors, name="IC"))
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="|IC|=0.05")
    fig.add_hline(y=-0.05, line_dash="dot", line_color="green")
    fig.update_layout(title="Information Coefficient (Spearman) — Fractal Features",
                      xaxis_title="Feature", yaxis_title="IC",
                      template="plotly_white",
                      legend={"orientation": "h", "yanchor": "top", "y": -0.20,
                              "xanchor": "center", "x": 0.5})
    return fig


def plot_shap_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> Any:
    """Bar chart of mean |SHAP| values for each feature.

    Fractal features in top 20 of 90+ features are genuinely contributing.
    """
    if not _PLOTLY:
        return _fallback_figure("SHAP Importance")

    try:
        import shap  # type: ignore[import]
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        order = np.argsort(mean_abs_shap)[::-1][:30]
        names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in order]
        vals = mean_abs_shap[order]
        colors = ["crimson" if any(k in n for k in ("hurst", "psd", "mfdfa", "multifractal", "wavelet"))
                  else "steelblue" for n in names]

        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                               marker_color=colors, name="Mean |SHAP|"))
        fig.update_layout(title="Top 30 Features by Mean |SHAP| (fractal features in red)",
                          xaxis_title="Mean |SHAP|",
                          yaxis={"autorange": "reversed"},
                          height=700, template="plotly_white",
                          legend={"orientation": "h", "yanchor": "top", "y": -0.10,
                                  "xanchor": "center", "x": 0.5})
        return fig

    except Exception as exc:
        log.warning("plot_shap_importance_failed", error=str(exc))
        return _fallback_figure(f"SHAP Importance (error: {exc})")


def plot_gate_counterfactual(trades: list[dict]) -> Any:
    """Compare win rate and P&L for gated vs. allowed trades.

    Shows whether the fractal regime gate improves iron condor outcomes.
    Each dict in trades must have: 'gated' (bool), 'pnl' (float).
    """
    if not _PLOTLY:
        return _fallback_figure("Gate Counterfactual")

    if not trades:
        return _fallback_figure("Gate Counterfactual (no trades)")

    gated = [t["pnl"] for t in trades if t.get("gated")]
    allowed = [t["pnl"] for t in trades if not t.get("gated")]

    def stats(pnls: list[float]) -> tuple[float, float, int]:
        if not pnls:
            return 0.0, 0.0, 0
        arr = np.array(pnls)
        win_rate = float((arr > 0).mean())
        mean_pnl = float(arr.mean())
        return win_rate, mean_pnl, len(arr)

    g_wr, g_pnl, g_n = stats(gated)
    a_wr, a_pnl, a_n = stats(allowed)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Win Rate", "Mean P&L"))
    for col, (label, wr, pnl, n) in enumerate(
        [("Gated", g_wr, g_pnl, g_n), ("Allowed", a_wr, a_pnl, a_n)], start=1
    ):
        fig.add_trace(go.Bar(x=[label], y=[wr], name=f"{label} WR (n={n})",
                             marker_color="tomato" if col == 1 else "steelblue"), row=1, col=1)
        fig.add_trace(go.Bar(x=[label], y=[pnl], name=f"{label} P&L",
                             marker_color="tomato" if col == 1 else "steelblue",
                             showlegend=False), row=1, col=2)

    fig.update_layout(title="Fractal Gate Counterfactual Analysis",
                      template="plotly_white", showlegend=True,
                      legend={"orientation": "h", "yanchor": "top", "y": -0.20,
                              "xanchor": "center", "x": 0.5})
    return fig
