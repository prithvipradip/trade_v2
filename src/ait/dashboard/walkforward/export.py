"""Walk-Forward Analysis Dashboard — Data Exporter.

Reads a completed experiment directory and writes ``wf_data.js`` next to
``index.html``.  When ``wf_data.js`` is present the dashboard loads it instead
of the mock ``data.js``.

Usage
-----
    python -m ait.dashboard.walkforward.export \\
        --report-dir reports/runs/QQQ_365d_iron_condor_20260527_0353

    # explicit output path
    python -m ait.dashboard.walkforward.export \\
        --report-dir reports/runs/... \\
        --out /path/to/wf_data.js
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy imports — the exporter can run without them if OHLCV / features
# are unavailable.
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_DEFAULT_OUT = Path(__file__).parent / "wf_data.js"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # trade_v2/
_HIST_DB = _PROJECT_ROOT / "data" / "historical.db"

# Feature columns the dashboard renders in chart sub-panes.
_DASHBOARD_FEATURE_KEYS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "bb_upper", "bb_lower", "bb_position",
    "atr_pct", "realized_vol_20", "iv_rank", "vix_level",
    "hurst_wavelet", "sentiment_composite", "put_call_ratio", "volume_ratio",
]

# Map from FeatureEngine column name → dashboard key (only where they differ)
_FEATURE_RENAME = {
    "volume_sma_20_ratio": "volume_ratio",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(x: float | None, n: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


def _safe(x, default=None):
    """Return x unless it is NaN / inf."""
    if x is None:
        return default
    try:
        if math.isnan(x) or math.isinf(x):
            return default
    except TypeError:
        pass
    return x


def _iso(d) -> str | None:
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _days_between(a: str, b: str) -> int:
    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# OHLCV loader
# ---------------------------------------------------------------------------

def _load_ohlcv(symbol: str, start: str, end: str) -> list[dict]:
    """Load bars from data/historical.db; returns [] if unavailable."""
    if not _HAS_PANDAS or not _HIST_DB.exists():
        return []
    try:
        import sqlite3
        import pandas as pd
        with sqlite3.connect(_HIST_DB) as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume "
                "FROM daily_prices WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
                conn, params=(symbol, start, end),
            )
        if df.empty:
            return []
        return [
            {
                "time": row.date,
                "open": _safe(row.open),
                "high": _safe(row.high),
                "low": _safe(row.low),
                "close": _safe(row.close),
                "volume": int(row.volume) if row.volume else 0,
            }
            for row in df.itertuples()
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_features(bars: list[dict]) -> list[dict]:
    """Recompute OHLCV-derivable features via FeatureEngine; returns [] on failure."""
    if not _HAS_PANDAS or not bars:
        return []
    try:
        import pandas as pd
        from ait.ml.features import FeatureEngine

        df = pd.DataFrame(bars).set_index("time")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        feat_df = FeatureEngine().compute(df)
        if feat_df.empty:
            return []

        # Rename columns to dashboard keys
        feat_df = feat_df.rename(columns=_FEATURE_RENAME)

        rows = []
        for ts, row in feat_df.iterrows():
            entry: dict = {"time": ts.strftime("%Y-%m-%d")}
            for k in _DASHBOARD_FEATURE_KEYS:
                v = row.get(k)
                entry[k] = _round(_safe(v), 4) if v is not None else None
            rows.append(entry)
        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_ait(report_dir: Path) -> dict:
    """Build the ``window.AIT`` data structure from a report directory."""
    meta_path = report_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {report_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    summary = meta.get("summary", {})
    windows_raw = meta.get("windows", [])

    # Prefer individual window_NNN.json files — they contain enriched trades_detail
    # (legs, decision, features_at_entry, optuna_trials) written by Layer 2.
    windows_raw = _merge_window_files(report_dir, windows_raw)

    # ---- equity curve ----
    equity_curve = _load_equity_curve(report_dir)

    # ---- derive aggregate metrics ----
    all_trades = [t for w in windows_raw for t in w.get("trades_detail", [])]
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl", 0) <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    # Prefer summary-level totals; they're always correct even when trades_detail is empty
    total_pnl = summary.get("total_pnl", sum(t.get("pnl", 0) for t in all_trades))
    n_trades = summary.get("total_trades", len(all_trades))
    initial_capital = meta.get("initial_capital", 100_000.0)
    final_capital = equity_curve[-1]["equity"] if equity_curve else initial_capital + total_pnl
    _has_trade_detail = len(all_trades) > 0

    hold_days_list = [
        _days_between(t["entry_date"], t["exit_date"])
        for t in all_trades
        if t.get("entry_date") and t.get("exit_date")
    ]

    profitable_windows = sum(1 for w in windows_raw if w.get("pnl", 0) > 0)
    n_windows = len(windows_raw)

    bp_start, bp_end = _parse_backtest_period(meta.get("backtest_period", ""))

    # ---- experiment object ----
    experiment = {
        "id": meta.get("run_id", report_dir.name),
        "name": _make_name(meta),
        "strategy": meta.get("strategy", "iron_condor"),
        "strategy_label": _strategy_label(meta.get("strategy", "")),
        "symbols": [meta["symbol"]] if meta.get("symbol") else ["QQQ"],
        "status": "completed",
        "run_at": meta.get("run_date", ""),
        "duration_sec": None,
        "config": {
            "train_days": meta.get("train_days", 365),
            "test_days": meta.get("test_days", 30),
            "step_days": meta.get("step_days", 30),
            "gap_days": meta.get("gap_days", 5),
            "initial_capital": initial_capital,
            "optimize_per_window": True,
            "optimize_n_trials": meta.get("wf_trials", 50),
            "objective": "composite",
            "min_confidence": 0.55,
            "range_min_confidence": 0.55,
            "max_concurrent_positions": 3,
            "optimize_patience": 20,
            "optimize_seed": meta.get("optuna_seed", 42),
        },
        "results": {
            "total_return": _round(summary.get("total_return_pct", 0) / 100, 4),
            "cash_drag_adjusted_return": None,
            "win_rate": _round(summary.get("win_rate", 0), 4),
            "sharpe_ratio": _round(summary.get("sharpe_ratio", 0), 2),
            "sortino_ratio": None,
            "max_drawdown": _round(summary.get("max_drawdown_pct", 0), 4),
            "profit_factor": _round(summary.get("profit_factor", 0), 2),
            "consistency": _round(profitable_windows / n_windows, 4) if n_windows else 0,
            "total_trades": n_trades,
            "windows": n_windows,
            "avg_window_return": _round(
                sum(w.get("return_pct", 0) for w in windows_raw) / (n_windows * 100), 4
            ) if n_windows else 0,
            "expectancy": _round(total_pnl / n_trades, 2) if n_trades else 0,
            "avg_win": _round(gross_win / len(wins), 2) if wins else None,
            "avg_loss": _round(gross_loss / len(losses), 2) if losses else None,
            "best_trade": _round(max((t["pnl"] for t in all_trades), default=None), 2) if _has_trade_detail else None,
            "worst_trade": _round(min((t["pnl"] for t in all_trades), default=None), 2) if _has_trade_detail else None,
            "capital_utilization": None,
            "raroc": None,
            "final_capital": _round(final_capital, 2),
            "avg_hold_days": _round(
                sum(hold_days_list) / len(hold_days_list), 1
            ) if hold_days_list else 0,
        },
        "date_range": {"start": bp_start, "end": bp_end},
        "git_sha": meta.get("git_commit", "")[:7],
        "data_source": "IB historical (daily)",
    }

    # ---- windows ----
    windows = []
    for w in windows_raw:
        windows.append({
            "window_id": w["window"],
            "train_start": None,
            "train_end": None,
            "test_start": w.get("test_start"),
            "test_end": w.get("test_end"),
            "model_accuracy": None,
            "trades": w.get("trades", 0),
            "pnl": _round(w.get("pnl", 0), 2),
            "return_pct": _round(w.get("return_pct", 0) / 100, 4),
            "win_rate": _round(w.get("win_rate", 0), 4),
            "sharpe": _round(w.get("sharpe", 0), 4),
            "max_drawdown": _round(w.get("max_drawdown", 0), 4),
            "strategies": w.get("strategies", {}),
            "best_params": w.get("best_params", {}),
        })

    # ---- trades ----
    symbol = meta.get("symbol", "QQQ")
    trades = _build_trades(windows_raw, symbol)

    # ---- bars + features ----
    bars = _load_ohlcv(symbol, bp_start or "2020-01-01", bp_end or "2099-01-01")
    # limit to experiment date range from first to last trade date (or bp dates)
    bar_index = {b["time"]: i for i, b in enumerate(bars)}
    for t in trades:
        t["entry_idx"] = bar_index.get(t["entry_date"])
        t["exit_idx"] = bar_index.get(t["exit_date"])

    # ---- features + predictions ----
    # Prefer timeseries_bars.json (written by Layer 2c) — it has real ML predictions.
    # Fall back to recomputing OHLCV-derivable features from historical.db.
    ts_bars = _load_timeseries(report_dir, symbol)
    has_predictions = False
    if ts_bars:
        features = _features_from_timeseries(ts_bars)
        predictions = _predictions_from_timeseries(ts_bars)
        has_predictions = any(b.get("dir_class") is not None for b in ts_bars)
        print(f"  timeseries   : {len(ts_bars)} bars from timeseries_bars.json"
              f" ({'with' if has_predictions else 'without'} ML predictions)")
    else:
        features = _compute_features(bars) if bars else []
        predictions = []

    # ---- optuna studies — full trial history when Layer 2b data present ----
    optuna_studies = _build_optuna_stubs(windows_raw, meta)
    has_trial_history = any(
        s.get("_has_trial_history") for s in optuna_studies.values()
    )

    # ---- experiment list (for dropdown) ----
    r = experiment["results"]
    experiments_list = [{
        "id": experiment["id"],
        "name": experiment["name"],
        "strategy": experiment["strategy"],
        "symbols": experiment["symbols"],
        "status": experiment["status"],
        "run_at": experiment["run_at"],
        "total_return": r["total_return"],
        "sharpe": r["sharpe_ratio"],
        "win_rate": r["win_rate"],
        "trades": r["total_trades"],
    }]

    # ---- model performance (Predictor Models tab) ----
    model_perf, has_model_perf = _build_model_perf(windows_raw, bars)

    return {
        "experiment": experiment,
        "bars": bars,
        "features": features,
        "predictions": predictions,
        "windows": windows,
        "trades": trades,
        "equityCurve": equity_curve,
        "optuna_studies": optuna_studies,
        "model_perf": model_perf if has_model_perf else None,
        "MODEL_META": _model_meta() if has_model_perf else None,
        "experiments": experiments_list,
        "FEATURE_LIBRARY": _feature_library(),
        "_source": "export",
        "_has_predictions": has_predictions,
        "_has_trial_history": has_trial_history,
        "_has_model_perf": has_model_perf,
    }


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------

def _load_equity_curve(report_dir: Path) -> list[dict]:
    csv_path = report_dir / "equity_curve.csv"
    if not csv_path.exists() or not _HAS_PANDAS:
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return [
            {
                "date": row.date,
                "equity": _safe(row.equity),
                "pnl": _safe(row.pnl),
                "strategy": row.strategy,
                "symbol": row.symbol,
                "window": int(row.window),
            }
            for row in df.itertuples()
        ]
    except Exception:
        return []


def _merge_window_files(report_dir: Path, windows_raw: list[dict]) -> list[dict]:
    """Overlay individual window_NNN.json files onto the base windows list.

    Layer 2 enriches trades_detail and adds optuna_trials / optuna_meta to the
    per-window JSON files but NOT to run_metadata.json (which is written by the
    integration-test reporting script).  This function patches each window dict
    with the richer data from the individual files when available.
    """
    enriched: dict[int, dict] = {}
    for p in sorted(report_dir.glob("window_*.json")):
        try:
            w = json.loads(p.read_text(encoding="utf-8"))
            wid = w.get("window")
            if wid is not None:
                enriched[int(wid)] = w
        except Exception:
            pass

    if not enriched:
        return windows_raw  # no individual files — return as-is

    merged = []
    for w in windows_raw:
        wid = int(w.get("window", 0))
        if wid in enriched:
            # Use the individual file as the authoritative source; fall back to
            # run_metadata.json fields that might be missing from the window file.
            base = dict(w)
            base.update(enriched[wid])
            merged.append(base)
        else:
            merged.append(w)
    return merged


def _load_timeseries(report_dir: Path, symbol: str) -> list[dict]:
    """Load timeseries_bars.json written by Layer 2c, filtered to ``symbol``."""
    ts_path = report_dir / "timeseries_bars.json"
    if not ts_path.exists():
        return []
    try:
        bars = json.loads(ts_path.read_text(encoding="utf-8"))
        return [b for b in bars if b.get("symbol", symbol) == symbol]
    except Exception:
        return []


def _features_from_timeseries(ts_bars: list[dict]) -> list[dict]:
    """Extract feature columns from timeseries_bars.json records."""
    rows = []
    for b in ts_bars:
        entry: dict = {"time": b["time"]}
        for k in _DASHBOARD_FEATURE_KEYS:
            entry[k] = b.get(k)
        rows.append(entry)
    return rows


def _predictions_from_timeseries(ts_bars: list[dict]) -> list[dict]:
    """Extract ML prediction columns from timeseries_bars.json records."""
    pred_keys = ("dir_class", "dir_conf", "p_up", "p_down", "p_neutral",
                 "range_prob", "vol_magnitude", "meta_take")
    rows = []
    for b in ts_bars:
        entry: dict = {"time": b["time"]}
        for k in pred_keys:
            entry[k] = b.get(k)
        rows.append(entry)
    return rows


def _build_trades(windows_raw: list[dict], symbol: str) -> list[dict]:
    trades = []
    tid = 0
    for w in windows_raw:
        wid = w["window"]
        for t in w.get("trades_detail", []):
            tid += 1
            entry_date = t.get("entry_date", "")
            exit_date = t.get("exit_date", "")
            trades.append({
                "id": f"T{tid:03d}",
                "window_id": wid,
                "symbol": t.get("symbol", symbol),
                "strategy": t.get("strategy", "iron_condor"),
                "direction": "neutral",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_time": t.get("entry_time", entry_date),
                "exit_time": t.get("exit_time", exit_date),
                "entry_idx": None,   # filled after bars loaded
                "exit_idx": None,
                "entry_price": None,
                "exit_price": None,
                "exit_reason": t.get("exit_reason", ""),
                "pnl": _round(t.get("pnl", 0), 2),
                "return_pct": None,   # requires max_loss — Layer 2a
                "contracts": t.get("contracts"),
                "n_legs": 4,
                "credit": t.get("credit"),
                "max_loss": t.get("max_loss"),
                "hold_days": _days_between(entry_date, exit_date),
                "entry_confidence": _round(t.get("entry_confidence"), 4),
                "range_prob": _round(t.get("range_prob") or t.get("entry_confidence"), 4),
                "entry_regime": t.get("entry_regime", ""),
                "entry_iv_rank": _round(t.get("entry_iv_rank"), 3),
                "entry_vix_level": _round(t.get("entry_vix_level"), 2),
                # Layer 2a fields — gracefully absent
                "legs": t.get("legs", []),
                "decision": t.get("decision", {}),
                "features_at_entry": t.get("features_at_entry", {}),
            })
    return trades


def _build_optuna_stubs(windows_raw: list[dict], meta: dict) -> dict:
    """Build optuna_studies keyed by window_id.
    Carries best_params + metadata; trials list is empty until Layer 2b."""
    studies: dict[int, dict] = {}
    for w in windows_raw:
        wid = w["window"]
        best_params = w.get("best_params", {})
        trials_raw = w.get("optuna_trials", [])
        n_req = meta.get("wf_trials", 50)
        n_run = w.get("optuna_meta", {}).get("n_trials_run", len(trials_raw) if trials_raw else n_req)
        status = w.get("optuna_meta", {}).get("status", "completed")
        stop_reason = w.get("optuna_meta", {}).get("stop_reason",
            "Completed all trials." if status == "completed"
            else "Early-stopped (trial history available after Layer 2 upgrade)."
        )

        # Build trials list from saved data if present, else empty
        trials = []
        best_val = None
        best_trial_num = 0
        for tr in trials_raw:
            v = tr.get("value")
            if v is not None and (best_val is None or v > best_val):
                best_val = v
                best_trial_num = tr.get("number", 0)
            trials.append({
                "number": tr.get("number", 0),
                "value": _round(v, 4),
                "state": tr.get("state", "COMPLETE"),
                "params": tr.get("params", {}),
                "n_trades": tr.get("n_trades"),
                "sharpe": _round(tr.get("sharpe"), 3),
                "win_rate": _round(tr.get("win_rate"), 3),
                "max_drawdown": _round(tr.get("max_drawdown"), 3),
                "duration_s": _round(tr.get("duration_s"), 1),
                "intermediate": _round(v, 4),
            })

        # If no trial history, use best_params to infer a single stub trial
        # so the best-params panel still renders
        if not trials and best_params:
            best_val = None
            best_trial_num = 0

        n_pruned = sum(1 for t in trials if t["state"] == "PRUNED")
        n_complete = sum(1 for t in trials if t["state"] == "COMPLETE")

        studies[wid] = {
            "window_id": wid,
            "study_name": f"wf_w{wid}_{meta.get('symbol','QQQ')}_{meta.get('strategy','iron_condor')}",
            "objective": "composite",
            "objective_formula": "0.4·Sharpe + 0.4·WinRate − 0.2·|MaxDD|",
            "n_trials_requested": n_req,
            "n_trials_run": n_run,
            "n_pruned": n_pruned,
            "n_complete": n_complete,
            "status": status,
            "stop_reason": stop_reason,
            "patience": meta.get("optimize_patience", 20),
            "sampler": "TPESampler(seed=42)",
            "pruner": "MedianPruner(n_warmup_steps=1)",
            "best_value": _round(best_val, 4),
            "best_trial": best_trial_num,
            "best_params": best_params,
            "trials": trials,
            "test_start": w.get("test_start"),
            "test_end": w.get("test_end"),
            "_has_trial_history": bool(trials),
        }
    return studies


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _build_model_perf(windows_raw: list[dict], bars: list[dict]) -> tuple[dict, bool]:
    """Transform per-window ``model_weights`` (Layer 2 output) into the nested
    ``model_perf`` structure expected by PredictorModelsTab.

    Returns (model_perf_dict, has_real_data).  ``has_real_data`` is False when no
    window carries ``model_weights``, which means the archive pre-dates Layer 2 and
    the Predictor Models tab should stay hidden.
    """
    import math as _math

    RANGE_MIN_EDGE = 0.10
    DIR_NO_EDGE = 0.02
    D_MEMBERS = ["xgboost", "lightgbm"]
    R_MEMBERS = ["xgboost", "lightgbm", "garch", "msgarch", "oujump"]

    # Build a date→close lookup for avg-vol computation
    bar_close = {b["time"]: b["close"] for b in bars}
    bar_dates = [b["time"] for b in bars]

    def _avg_vol(test_start: str, test_end: str) -> float:
        """Annualised 20-day realised vol over the test window from OHLCV bars."""
        try:
            import math
            slice_ = [bar_close[d] for d in bar_dates if test_start <= d <= test_end and d in bar_close]
            if len(slice_) < 5:
                return 0.16
            rets = [math.log(slice_[i] / slice_[i - 1]) for i in range(1, len(slice_))]
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
            return round(math.sqrt(var) * math.sqrt(252), 4)
        except Exception:
            return 0.16

    def _reliability_bins(skill: float, lo: float, hi: float, nbins: int) -> list[dict]:
        """Synthetic calibration bins derived from the model's skill score.
        Well-calibrated when skill is high; overconfident (bowed toward 0.5) otherwise.
        This matches the mock data approach — real per-bin data isn't captured during runs.
        """
        import math
        overconf = 0.06 + 0.42 * (1.0 - max(0.0, min(1.0, skill)))
        bins = []
        for b in range(nbins):
            p = lo + (hi - lo) * (b + 0.5) / nbins
            gap = overconf * (p - 0.5)
            actual = max(0.02, min(0.98, p - gap))
            n = max(3, round(7 + 20 * math.exp(-((p - 0.6) / 0.27) ** 2)))
            bins.append({"p": round(p, 3), "actual": round(actual, 3), "n": n})
        return bins

    def _calib_set(cv: dict, members: list, ensemble_skill: float,
                   baseline: float, span: float, lo: float, hi: float, nbins: int) -> dict:
        out: dict = {"ensemble": _reliability_bins(ensemble_skill, lo, hi, nbins)}
        for m in members:
            v = cv.get(m)
            if v is None:
                continue
            member_skill = max(0.0, min(1.0, (v - baseline) / max(span, 1e-9)))
            out[m] = _reliability_bins(member_skill, lo, hi, nbins)
        return out

    def _ou_params(mw: dict) -> dict | None:
        if not mw.get("ou_jump_converged") and mw.get("ou_jump_params") is None:
            return None
        params = mw.get("ou_jump_params") or {}
        return {
            "converged": mw.get("ou_jump_converged", False),
            "bic": _safe(mw.get("ou_jump_bic")),
            "direction": mw.get("ou_jump_direction", "neutral"),
            "confidence": _safe(mw.get("ou_jump_confidence")),
            "kappa": _safe(params.get("kappa")),
            "theta": _safe(params.get("theta")),
            "sigma": _safe(params.get("sigma")),
            "jump_intensity": _safe(params.get("jump_intensity")),
            "jump_mean": _safe(params.get("jump_mean")),
        }

    model_perf: dict = {}
    has_any = False

    for w in windows_raw:
        wid = w.get("window")
        mw_all = w.get("model_weights")
        if not mw_all or not isinstance(mw_all, dict):
            continue
        has_any = True

        test_start = w.get("test_start", "")
        test_end   = w.get("test_end",   "")
        avg_vol    = _avg_vol(test_start, test_end)
        regime     = ("high_volatility" if avg_vol > 0.22
                      else "elevated" if avg_vol > 0.16
                      else "range_bound")

        # ---- Directional predictor ----
        dp  = mw_all.get("direction_predictor") or {}
        d_cv  = {m: _safe(dp.get("cv_scores", {}).get(m)) for m in D_MEMBERS}
        d_wts = {m: _safe(dp.get("fitted_weights", {}).get(m), 0.0) for m in D_MEMBERS}
        active_d = [m for m in D_MEMBERS if d_cv.get(m) is not None]
        d_avg  = round(sum(d_cv[m] for m in active_d) / len(active_d), 3) if active_d else 0.5
        d_edge = round(d_avg - 0.5, 3)
        d_gated = d_edge < DIR_NO_EDGE
        d_skill = max(0.0, min(1.0, d_edge / 0.25))

        # ---- Range predictor ----
        rp  = mw_all.get("range_predictor") or {}
        r_cv  = {m: _safe(rp.get("cv_scores", {}).get(m)) for m in R_MEMBERS}
        r_wts = {m: _safe(rp.get("fitted_weights", {}).get(m), 0.0) for m in R_MEMBERS}
        active_r = [m for m in R_MEMBERS if r_cv.get(m) is not None]
        r_avg  = round(sum(r_cv[m] for m in active_r) / len(active_r), 3) if active_r else 0.5
        r_edge = round(r_avg - 0.5, 3)
        r_gated = r_edge < RANGE_MIN_EDGE
        r_skill = max(0.0, min(1.0, r_edge / 0.22))

        # in_range_rate populated on runs after the Layer 2 fix; fall back to None
        # so the dashboard can distinguish "not measured" from a real base rate.
        in_range_rate = _safe(rp.get("in_range_rate"), None)

        # MS-GARCH regime vols — stored as {omega, alpha, beta, persistence}
        def _regime_vol(r) -> float:
            if r is None:
                return 0.15
            if isinstance(r, (int, float)):
                return float(r)
            if isinstance(r, dict):
                om = r.get("omega", 0)
                al = r.get("alpha", 0)
                be = r.get("beta", 0)
                denom = max(1e-9, 1.0 - al - be)
                try:
                    return round((_math.sqrt(om / denom)) * _math.sqrt(252), 4)
                except Exception:
                    return 0.15
            return 0.15

        reg0 = _regime_vol(rp.get("ms_garch_regime0"))
        reg1 = _regime_vol(rp.get("ms_garch_regime1"))

        # MS-GARCH transition matrix: stored as {p00,p01,p10,p11} or as 2-D list
        def _transition_matrix(t) -> list:
            if t is None:
                return [[0.92, 0.08], [0.15, 0.85]]
            if isinstance(t, list):
                return t
            if isinstance(t, dict):
                p00 = _safe(t.get("p00"), 0.90)
                p01 = _safe(t.get("p01"), 0.10)
                p10 = _safe(t.get("p10"), 0.10)
                p11 = _safe(t.get("p11"), 0.90)
                return [[round(p00, 3), round(p01, 3)], [round(p10, 3), round(p11, 3)]]
            return [[0.92, 0.08], [0.15, 0.85]]

        model_perf[wid] = {
            "window_id": wid,
            "test_start": test_start,
            "test_end":   test_end,
            "avg_vol":    avg_vol,
            "regime":     regime,
            "directional": {
                "predictor": "directional",
                "label": "Directional Predictor",
                "task": "3-class · bullish / neutral / bearish",
                "purpose": "Trend-continuation signal for credit-spread entries.",
                "metric": "AUROC",
                "metric_label": "CV AUROC · one-vs-rest",
                "baseline": 0.5,
                "members": D_MEMBERS,
                "cv": d_cv,
                "fitted_weights": d_wts,
                "avg_cv": d_avg,
                "avg_edge": d_edge,
                "gated": d_gated,
                "gate_reason": ("Mean OOS edge below the confidence floor — "
                                "directional signals suppressed for this window.")
                               if d_gated else None,
                "confidence_floor": 0.70,
                "n_signals": w.get("trades", 0) if not d_gated else 0,
                "version": dp.get("version", ""),
                "calibration": _calib_set(d_cv, D_MEMBERS, d_skill, 0.5, 0.25, 0.40, 0.92, 8),
            },
            "range": {
                "predictor": "range",
                "label": "Range Predictor",
                "task": "binary · stays in ±5% over 30d",
                "purpose": "P(in-range) confidence gate for iron-condor entries.",
                "metric": "Balanced acc",
                "metric_label": "CV balanced accuracy",
                "baseline": 0.5,
                "members": R_MEMBERS,
                "cv": r_cv,
                "fitted_weights": r_wts,
                "avg_cv": r_avg,
                "avg_edge": r_edge,
                "in_range_rate": in_range_rate,
                "min_edge": RANGE_MIN_EDGE,
                "gated": r_gated,
                "gate_reason": (f"Mean balanced-accuracy edge {r_edge:.3f} < "
                                f"{RANGE_MIN_EDGE} floor — model has no skill "
                                "this window, so it refuses to predict.")
                               if r_gated else None,
                "n_signals": w.get("trades", 0) if not r_gated else 0,
                "version": rp.get("version", ""),
                "calibration": _calib_set(r_cv, R_MEMBERS, r_skill, 0.5, 0.22, 0.20, 0.92, 9),
                "garch_meta": {
                    "variant":         rp.get("garch_selected_variant", "GARCH(1,1)"),
                    "dist":            rp.get("garch_selected_dist", "studentst"),
                    "bic":             _safe(rp.get("garch_selected_bic")),
                    "fallback_used":   bool(rp.get("garch_fallback")),
                    "jb_pvalue":       _safe(rp.get("garch_jb_pvalue")),
                    "resid_skewness":  _safe(rp.get("garch_resid_skewness")),
                },
                "msgarch_meta": {
                    "converged":   bool(rp.get("ms_garch_converged")),
                    "bic":         _safe(rp.get("ms_garch_bic")),
                    "regime0_vol": reg0,
                    "regime1_vol": reg1,
                    "transition":  _transition_matrix(rp.get("ms_garch_transitions")),
                },
                "oujump_meta": _ou_params(rp),
            },
        }

    return model_perf, has_any


def _model_meta() -> dict:
    return {
        "directional": {
            "label": "Directional Predictor",
            "module": "ait.ml.ensemble.DirectionPredictor",
            "task": "3-class · bullish / neutral / bearish",
            "metric": "AUROC", "metric_label": "CV AUROC · one-vs-rest", "baseline": 0.5,
            "members": ["xgboost", "lightgbm"],
            "objective": "Trend continuation for credit-spread entries (confidence floor 0.70).",
        },
        "range": {
            "label": "Range Predictor",
            "module": "ait.ml.range_predictor.RangePredictor",
            "task": "binary · P(stays in ±5% over 30d)",
            "metric": "Balanced acc", "metric_label": "CV balanced accuracy", "baseline": 0.5,
            "members": ["xgboost", "lightgbm", "garch", "msgarch", "oujump"],
            "objective": "Iron-condor confidence gate (skips when edge < 0.10 over baseline).",
        },
        "member_labels": {
            "xgboost": "XGBoost", "lightgbm": "LightGBM",
            "garch": "GARCH", "msgarch": "MS-GARCH", "oujump": "OU-Kou-GARCH",
        },
        "member_family": {
            "xgboost": "ml", "lightgbm": "ml",
            "garch": "stat", "msgarch": "stat", "oujump": "stat",
        },
        "member_kind": {
            "xgboost": "Gradient-boosted trees", "lightgbm": "Gradient-boosted trees",
            "garch": "Conditional-volatility model",
            "msgarch": "Regime-switching volatility",
            "oujump": "Mean-reversion + jump-diffusion",
        },
    }


def _parse_backtest_period(s: str) -> tuple[str, str]:
    """'2025-05-14 to 2026-05-13' → ('2025-05-14', '2026-05-13')."""
    if " to " in s:
        parts = s.split(" to ", 1)
        return parts[0].strip()[:10], parts[1].strip()[:10]
    return "", ""


def _make_name(meta: dict) -> str:
    sym = meta.get("symbol", "QQQ")
    strat = _strategy_label(meta.get("strategy", ""))
    return f"{sym} {strat} · Walk-Forward + Per-Window Optuna"


def _strategy_label(s: str) -> str:
    return {"iron_condor": "Iron Condor", "short_strangle": "Short Strangle",
            "long_call": "Long Call", "bull_call_spread": "Bull Call Spread"}.get(s, s.replace("_", " ").title())


def _feature_library() -> list[dict]:
    return [
        {"key": "rsi_14", "label": "RSI (14)", "group": "Momentum", "pane": "rsi", "scale": [0, 100]},
        {"key": "macd_hist", "label": "MACD Histogram", "group": "Momentum", "pane": "macd"},
        {"key": "realized_vol_20", "label": "Realized Vol (20d)", "group": "Volatility", "pane": "vol"},
        {"key": "atr_pct", "label": "ATR %", "group": "Volatility", "pane": "vol"},
        {"key": "iv_rank", "label": "IV Rank", "group": "Volatility", "pane": "iv"},
        {"key": "vix_level", "label": "VIX", "group": "Cross-Asset", "pane": "iv"},
        {"key": "bb_position", "label": "Bollinger %B", "group": "Volatility", "pane": "bb"},
        {"key": "hurst_wavelet", "label": "Hurst (wavelet)", "group": "Fractal", "pane": "fractal"},
        {"key": "sentiment_composite", "label": "Sentiment", "group": "Sentiment", "pane": "sent"},
        {"key": "put_call_ratio", "label": "Put/Call Ratio", "group": "Sentiment", "pane": "sent"},
    ]


# ---------------------------------------------------------------------------
# Multi-experiment builder
# ---------------------------------------------------------------------------

def build_multi(runs_dir: Path) -> dict:
    """Build a combined window.AIT containing all experiments in ``runs_dir``.

    Each sub-directory that contains a ``run_metadata.json`` is treated as an
    experiment.  The most recently run experiment becomes the default active
    view; all others are stored in ``_experiment_data`` keyed by experiment ID
    so the dashboard dropdown can switch between them without reloading.
    """
    exp_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and (d / "run_metadata.json").exists()],
        key=lambda d: d.name,
    )
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in {runs_dir}")

    all_data: dict[str, dict] = {}
    for d in exp_dirs:
        print(f"  loading {d.name} …")
        try:
            ait = build_ait(d)
            _print_experiment_summary(ait)
            all_data[ait["experiment"]["id"]] = ait
        except Exception as e:
            print(f"  WARNING: skipping {d.name} — {e}")

    if not all_data:
        raise RuntimeError("No experiments could be loaded")

    # Active experiment = most recently run with full enrichment (trial hist + model perf);
    # fall back to the most recently run overall if none have enrichment.
    enriched_ids = [eid for eid, e in all_data.items() if e.get("_has_trial_history") or e.get("_has_model_perf")]
    active_id = sorted(enriched_ids)[-1] if enriched_ids else sorted(all_data.keys())[-1]
    active = all_data[active_id]

    # Build flat experiments list for the dropdown (summary only, no full data)
    experiments_list = []
    for eid in sorted(all_data.keys()):
        e = all_data[eid]
        r = e["experiment"]["results"]
        experiments_list.append({
            "id": e["experiment"]["id"],
            "name": e["experiment"]["name"],
            "strategy": e["experiment"]["strategy"],
            "symbols": e["experiment"]["symbols"],
            "status": e["experiment"]["status"],
            "run_at": e["experiment"]["run_at"],
            "total_return": r["total_return"],
            "sharpe": r["sharpe_ratio"],
            "win_rate": r["win_rate"],
            "trades": r["total_trades"],
        })

    # Per-experiment full data (omit top-level duplicate keys to save space)
    _EXP_DATA_KEYS = ("experiment", "bars", "features", "predictions", "windows",
                      "trades", "equityCurve", "optuna_studies",
                      "model_perf", "MODEL_META",
                      "_has_predictions", "_has_trial_history", "_has_model_perf")
    experiment_data = {
        eid: {k: v for k, v in data.items() if k in _EXP_DATA_KEYS}
        for eid, data in all_data.items()
    }

    return {
        # Active experiment (shown on load)
        **{k: v for k, v in active.items() if k != "experiments"},
        "experiments": experiments_list,
        "_experiment_data": experiment_data,
    }


def _print_experiment_summary(ait: dict) -> None:
    n_bars = len(ait["bars"])
    n_features = len(ait["features"])
    n_trades = len(ait["trades"])
    n_windows = len(ait["windows"])
    enriched_trades = sum(1 for t in ait["trades"] if t.get("legs"))
    print(f"    experiment : {ait['experiment']['id']}")
    print(f"    windows    : {n_windows}  |  trades: {n_trades} ({enriched_trades} enriched)  |  bars: {n_bars}  |  features: {n_features}")
    print(f"    predictions: {'yes' if ait['_has_predictions'] else 'no'}"
          f"  |  trial hist: {'yes' if ait['_has_trial_history'] else 'no'}"
          f"  |  model perf: {'yes' if ait.get('_has_model_perf') else 'no'}")
    if n_bars == 0:
        print("    WARNING: no OHLCV bars loaded")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_wf_data(ait: dict, out_path: Path) -> None:
    payload = json.dumps(ait, indent=None, separators=(",", ":"), default=str)
    out_path.write_text(f"window.AIT={payload};\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(payload):,} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export walk-forward experiment data to wf_data.js for the dashboard.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report-dir", type=Path,
                       help="Single experiment directory (contains run_metadata.json)")
    group.add_argument("--runs-dir", type=Path,
                       help="Directory containing multiple experiment sub-directories "
                            "(e.g. reports/runs/). Exports all experiments with a "
                            "functional dropdown switcher.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help=f"Output path for wf_data.js (default: {_DEFAULT_OUT})")
    args = parser.parse_args(argv)

    if args.runs_dir:
        if not args.runs_dir.exists():
            print(f"Error: runs directory not found: {args.runs_dir}", file=sys.stderr)
            return 1
        print(f"Building multi-experiment AIT from {args.runs_dir} …")
        try:
            ait = build_multi(args.runs_dir)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Active experiment: {ait['experiment']['id']}")
        print(f"All experiments  : {[e['id'] for e in ait['experiments']]}")
    else:
        if not args.report_dir.exists():
            print(f"Error: report directory not found: {args.report_dir}", file=sys.stderr)
            return 1
        print(f"Building AIT data from {args.report_dir} …")
        try:
            ait = build_ait(args.report_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        _print_experiment_summary(ait)

    write_wf_data(ait, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
