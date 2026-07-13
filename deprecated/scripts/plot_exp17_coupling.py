"""Plot Exp 17 vs Exp 16: MS-GARCH OOS AUROC vs PnL delta per window.

Usage:
    python scripts/plot_exp17_coupling.py

Reads the most recent run from reports/runs/ (assumed to be Exp 17),
compares against the archived Exp 16 results, and produces:

  1. Scatter: MS-GARCH OOS AUROC (x) vs PnL delta E17-E16 (y) — the coupling plot
  2. Scatter: OU-Kou-GARCH OOS AUROC (x) vs PnL delta (y)
  3. Bar chart: per-window PnL comparison E16 vs E17
  4. Heatmap: all OOS probability metrics (Brier skill, log loss, AUROC) per model per window
  5. AEKF direction accuracy vs window (where available)

Output: reports/exp17_analysis.png  (multi-panel figure)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Locate run directories
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR  = REPO_ROOT / "reports" / "runs"

E16_RUN_ID = "QQQ_365d_iron_condor_20260603_0617"


def _latest_run() -> Path:
    """Return the most recently modified run directory (Exp 17)."""
    runs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        sys.exit("No run directories found in reports/runs/")
    return runs[0]


def _load_windows(run_dir: Path) -> dict[int, dict]:
    """Load all window JSON files from a run directory."""
    windows = {}
    for wf in sorted(run_dir.glob("window_*.json")):
        num = int(wf.stem.split("_")[1])
        with open(wf) as f:
            windows[num] = json.load(f)
    return windows


# ---------------------------------------------------------------------------
# Extract metrics
# ---------------------------------------------------------------------------

def _pnl(w: dict) -> float:
    return float(w.get("pnl", 0.0))


def _oos_auroc(w: dict, model: str) -> float | None:
    """Extract OOS AUROC for a given model from window JSON."""
    oos = (w.get("model_weights") or {}).get("range_predictor", {}).get("oos_scores") or {}
    stat = oos.get("statistical") or {}
    ml   = oos.get("ml") or {}
    entry = stat.get(model) or ml.get(model) or {}
    v = entry.get("auroc")
    return float(v) if v is not None else None


def _oos_brier_skill(w: dict, model: str) -> float | None:
    oos = (w.get("model_weights") or {}).get("range_predictor", {}).get("oos_scores") or {}
    stat = oos.get("statistical") or {}
    ml   = oos.get("ml") or {}
    entry = stat.get(model) or ml.get(model) or {}
    v = entry.get("brier_skill")
    return float(v) if v is not None else None


def _oos_rvol_bias(w: dict, model: str) -> float | None:
    oos = (w.get("model_weights") or {}).get("range_predictor", {}).get("oos_scores") or {}
    stat = oos.get("statistical") or {}
    entry = stat.get(model) or {}
    v = entry.get("rvol_bias")
    return float(v) if v is not None else None


def _aekf_direction_auroc(w: dict) -> float | None:
    oos = (w.get("model_weights") or {}).get("range_predictor", {}).get("oos_scores") or {}
    aekf = oos.get("aekf") or {}
    v = aekf.get("direction_auroc")
    return float(v) if v is not None else None


def _fitted_weight(w: dict, model: str) -> float:
    fw = (w.get("model_weights") or {}).get("range_predictor", {}).get("fitted_weights") or {}
    return float(fw.get(model, 0.0))


def _test_period(w: dict) -> str:
    ts = w.get("test_start", "")
    te = w.get("test_end", "")
    if ts and te:
        return f"{ts[2:7]}–{te[2:7]}"
    return f"W{w.get('window', '?')}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    e17_dir = _latest_run()
    e16_dir = RUNS_DIR / E16_RUN_ID

    print(f"Exp 17: {e17_dir.name}")
    print(f"Exp 16: {e16_dir.name}")

    e17 = _load_windows(e17_dir)
    e16 = _load_windows(e16_dir)

    windows = sorted(set(e17.keys()) & set(e16.keys()))
    n = len(windows)

    labels       = [f"W{w:02d}\n{_test_period(e17[w])}" for w in windows]
    pnl_e17      = np.array([_pnl(e17[w]) for w in windows])
    pnl_e16      = np.array([_pnl(e16[w]) for w in windows])
    pnl_delta    = pnl_e17 - pnl_e16

    auroc_msgarch = [_oos_auroc(e17[w], "msgarch")  for w in windows]
    auroc_oujump  = [_oos_auroc(e17[w], "oujump")   for w in windows]
    auroc_xgb     = [_oos_auroc(e17[w], "xgboost")  for w in windows]
    auroc_lgb     = [_oos_auroc(e17[w], "lightgbm") for w in windows]
    bss_msgarch   = [_oos_brier_skill(e17[w], "msgarch")  for w in windows]
    bss_oujump    = [_oos_brier_skill(e17[w], "oujump")   for w in windows]
    rvol_msgarch  = [_oos_rvol_bias(e17[w], "msgarch")    for w in windows]
    rvol_oujump   = [_oos_rvol_bias(e17[w], "oujump")     for w in windows]
    dir_auroc     = [_aekf_direction_auroc(e17[w])         for w in windows]
    w_msgarch     = [_fitted_weight(e17[w], "msgarch")     for w in windows]
    w_oujump      = [_fitted_weight(e17[w], "oujump")      for w in windows]

    # -----------------------------------------------------------------------
    # Figure layout: 3 rows × 2 cols + extras
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    DARK_BG   = "#1a1a1a"
    GRID_COL  = "#2a2a2a"
    TEXT_COL  = "#e0e0e0"
    POS_COL   = "#4caf50"
    NEG_COL   = "#f44336"
    BLUE      = "#2196f3"
    ORANGE    = "#ff9800"
    PURPLE    = "#9c27b0"
    TEAL      = "#009688"

    def _style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.spines[:].set_color(GRID_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        if title:
            ax.set_title(title, color=TEXT_COL, fontsize=10, pad=6)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)

    # -----------------------------------------------------------------------
    # Panel 1: MS-GARCH AUROC vs PnL delta (the coupling plot)
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "MS-GARCH OOS AUROC vs PnL Delta (E17−E16)")

    valid = [(auroc_msgarch[i], pnl_delta[i], windows[i], w_msgarch[i])
             for i in range(n) if auroc_msgarch[i] is not None]

    if valid:
        xs, ys, wids, wts = zip(*valid)
        colors = [POS_COL if y >= 0 else NEG_COL for y in ys]
        sizes  = [max(30, wt * 600) for wt in wts]  # bubble size = fitted weight
        ax1.scatter(xs, ys, c=colors, s=sizes, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        for x, y, wid in zip(xs, ys, wids):
            ax1.annotate(f"W{wid:02d}", (x, y), textcoords="offset points",
                         xytext=(6, 3), color=TEXT_COL, fontsize=7)
        ax1.axhline(0, color=GRID_COL, linewidth=1.2, linestyle="--")
        ax1.axvline(0.5, color=ORANGE, linewidth=1, linestyle=":", alpha=0.7)
        ax1.set_xlabel("OOS AUROC (MS-GARCH)")
        ax1.set_ylabel("PnL Delta E17−E16 ($)")
        ax1.text(0.02, 0.95, "Bubble size = fitted weight", transform=ax1.transAxes,
                 color=TEXT_COL, fontsize=7, va="top", alpha=0.7)
    else:
        ax1.text(0.5, 0.5, "No MS-GARCH OOS scores available", transform=ax1.transAxes,
                 ha="center", va="center", color=TEXT_COL)

    # -----------------------------------------------------------------------
    # Panel 2: OU-Kou-GARCH AUROC vs PnL delta
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "OU-Kou-GARCH OOS AUROC vs PnL Delta (E17−E16)")

    valid_ou = [(auroc_oujump[i], pnl_delta[i], windows[i], w_oujump[i])
                for i in range(n) if auroc_oujump[i] is not None]

    if valid_ou:
        xs, ys, wids, wts = zip(*valid_ou)
        colors = [POS_COL if y >= 0 else NEG_COL for y in ys]
        sizes  = [max(30, wt * 600) for wt in wts]
        ax2.scatter(xs, ys, c=colors, s=sizes, alpha=0.85, edgecolors="white", linewidths=0.5,
                    marker="D", zorder=3)
        for x, y, wid in zip(xs, ys, wids):
            ax2.annotate(f"W{wid:02d}", (x, y), textcoords="offset points",
                         xytext=(6, 3), color=TEXT_COL, fontsize=7)
        ax2.axhline(0, color=GRID_COL, linewidth=1.2, linestyle="--")
        ax2.axvline(0.5, color=ORANGE, linewidth=1, linestyle=":", alpha=0.7)
        ax2.set_xlabel("OOS AUROC (OU-Kou-GARCH)")
        ax2.set_ylabel("PnL Delta E17−E16 ($)")
    else:
        ax2.text(0.5, 0.5, "No OU-Kou-GARCH OOS scores available", transform=ax2.transAxes,
                 ha="center", va="center", color=TEXT_COL)

    # -----------------------------------------------------------------------
    # Panel 3: Per-window PnL bar chart E16 vs E17
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    _style_ax(ax3, "Per-Window PnL: Exp 16 vs Exp 17")

    x = np.arange(n)
    w_bar = 0.35
    bars16 = ax3.bar(x - w_bar/2, pnl_e16, w_bar, label="Exp 16", color=BLUE,   alpha=0.75)
    bars17 = ax3.bar(x + w_bar/2, pnl_e17, w_bar, label="Exp 17", color=ORANGE, alpha=0.75)
    ax3.axhline(0, color=GRID_COL, linewidth=1)

    short_labels = [f"W{w:02d}" for w in windows]
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, fontsize=8)
    ax3.set_ylabel("PnL ($)")
    ax3.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)

    total16 = pnl_e16.sum()
    total17 = pnl_e17.sum()
    ax3.set_title(
        f"Per-Window PnL: Exp 16 vs Exp 17   |   "
        f"E16 total: ${total16:+,.0f}   E17 total: ${total17:+,.0f}   "
        f"Delta: ${total17-total16:+,.0f}",
        color=TEXT_COL, fontsize=10, pad=6,
    )

    # -----------------------------------------------------------------------
    # Panel 4: OOS Brier Skill Score heatmap
    # -----------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    _style_ax(ax4, "OOS Brier Skill Score by Model & Window")

    models_bss = ["xgboost", "lightgbm", "msgarch", "oujump"]
    bss_matrix = np.full((len(models_bss), n), np.nan)
    for mi, model in enumerate(models_bss):
        for wi, w in enumerate(windows):
            v = _oos_brier_skill(e17[w], model)
            if v is not None:
                bss_matrix[mi, wi] = v

    im = ax4.imshow(bss_matrix, aspect="auto", cmap="RdYlGn",
                    vmin=-0.15, vmax=0.15, interpolation="nearest")
    ax4.set_xticks(range(n))
    ax4.set_xticklabels([f"W{w:02d}" for w in windows], fontsize=7, rotation=45)
    ax4.set_yticks(range(len(models_bss)))
    ax4.set_yticklabels(models_bss, fontsize=8)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04,
                 label="Brier Skill Score").ax.yaxis.label.set_color(TEXT_COL)

    # Annotate cells
    for mi in range(len(models_bss)):
        for wi in range(n):
            v = bss_matrix[mi, wi]
            if not np.isnan(v):
                ax4.text(wi, mi, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(v) < 0.08 else "white")

    # -----------------------------------------------------------------------
    # Panel 5: Realized vol bias heatmap (rvol_bias per statistical model)
    # -----------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    _style_ax(ax5, "OOS Realized Vol Bias (σ̂ − σ_realized, annualised)")

    stat_models = ["msgarch", "oujump"]
    rvol_matrix = np.full((len(stat_models), n), np.nan)
    for mi, model in enumerate(stat_models):
        for wi, w in enumerate(windows):
            v = _oos_rvol_bias(e17[w], model)
            if v is not None:
                rvol_matrix[mi, wi] = v

    im2 = ax5.imshow(rvol_matrix, aspect="auto", cmap="RdBu_r",
                     vmin=-0.05, vmax=0.05, interpolation="nearest")
    ax5.set_xticks(range(n))
    ax5.set_xticklabels([f"W{w:02d}" for w in windows], fontsize=7, rotation=45)
    ax5.set_yticks(range(len(stat_models)))
    ax5.set_yticklabels(stat_models, fontsize=8)
    fig.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04,
                 label="Vol Bias (annualised)").ax.yaxis.label.set_color(TEXT_COL)

    for mi in range(len(stat_models)):
        for wi in range(n):
            v = rvol_matrix[mi, wi]
            if not np.isnan(v):
                ax5.text(wi, mi, f"{v:+.3f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(v) < 0.03 else "white")

    # -----------------------------------------------------------------------
    # Panel 6: AEKF direction AUROC per window + fitted weights
    # -----------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[3, :])
    _style_ax(ax6, "AEKF Direction AUROC & Fitted Weights (OU-Kou-GARCH) per Window")

    x6 = np.arange(n)
    has_dir = [v for v in dir_auroc if v is not None]

    ax6b = ax6.twinx()
    ax6b.set_facecolor(DARK_BG)
    ax6b.tick_params(colors=TEXT_COL, labelsize=8)
    ax6b.spines[:].set_color(GRID_COL)
    ax6b.yaxis.label.set_color(TEXT_COL)

    bar_wou = ax6b.bar(x6, w_oujump, 0.6, label="OU-Kou weight",
                       color=PURPLE, alpha=0.35, zorder=1)
    bar_wms = ax6b.bar(x6, w_msgarch, 0.6, label="MS-GARCH weight",
                       color=TEAL, alpha=0.35, bottom=w_oujump, zorder=1)
    ax6b.set_ylabel("Fitted Weight")
    ax6b.set_ylim(0, 1.1)

    dir_vals = [v if v is not None else np.nan for v in dir_auroc]
    ax6.plot(x6, dir_vals, color=ORANGE, marker="o", linewidth=1.5,
             markersize=5, label="AEKF direction AUROC", zorder=3)
    ax6.axhline(0.5, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.6)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(short_labels, fontsize=8)
    ax6.set_ylabel("Direction AUROC")
    ax6.set_ylim(0.3, 0.85)

    lines1, labs1 = ax6.get_legend_handles_labels()
    lines2, labs2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labs1 + labs2, facecolor=DARK_BG,
               labelcolor=TEXT_COL, fontsize=7, loc="upper right")

    # -----------------------------------------------------------------------
    # Title & save
    # -----------------------------------------------------------------------
    fig.suptitle(
        f"Experiment 17 Analysis — {e17_dir.name}\n"
        f"MS-GARCH + OU-Kou-GARCH with RNG Isolation vs Exp 16 (ML-only baseline)",
        color=TEXT_COL, fontsize=12, y=0.995,
    )

    out_path = REPO_ROOT / "reports" / "exp17_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n" + "="*72)
    print(f"{'W':>4} {'E16 PnL':>10} {'E17 PnL':>10} {'Delta':>8} "
          f"{'MSGARCH AUROC':>14} {'OU AUROC':>10} {'OU-w':>6} {'MS-w':>6}")
    print("-"*72)
    for i, w in enumerate(windows):
        ms_a = f"{auroc_msgarch[i]:.3f}" if auroc_msgarch[i] is not None else "  N/A "
        ou_a = f"{auroc_oujump[i]:.3f}"  if auroc_oujump[i]  is not None else "  N/A "
        print(f"W{w:02d}  {pnl_e16[i]:>10.2f} {pnl_e17[i]:>10.2f} {pnl_delta[i]:>8.2f} "
              f"{ms_a:>14} {ou_a:>10} {w_msgarch[i]:>6.3f} {w_oujump[i]:>6.3f}")
    print("-"*72)
    print(f"TOT  {pnl_e16.sum():>10.2f} {pnl_e17.sum():>10.2f} {pnl_delta.sum():>8.2f}")
    print("="*72)


if __name__ == "__main__":
    main()
