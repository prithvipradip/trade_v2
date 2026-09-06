"""Streamlit dashboard for monitoring the trading bot.

Run separately: streamlit run src/ait/dashboard/app.py
Tabs: Portfolio Overview, Trade History, Analytics, Self-Learning, System Health.

Uses DuckDB for analytics-heavy queries (trade history, strategy breakdown,
regime analysis) and SQLite for live operational state (open positions, KV store).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DB_PATH = Path("data/ait_state.db")

# Non-real closes (reconciler bookkeeping rows: never_filled / pending /
# migrated) must not count as closes in PF / win-rate / drawdown — mirrors
# the filter in status.py.
from ait.reporting.go_live import (
    NOT_REAL_CLOSE_PATTERNS as _NOT_REAL_PATTERNS,
)

# W3/string-contracts-4: membership now comes from the ONE authority
# (src/ait/reporting/go_live.py) — this local copy omitted the
# reconciler $0 sentinels (reconciler_unknown / needs_review), which
# therefore counted as REAL closes in PF / win-rate / drawdown.
_REAL_CLOSE_SQL = " ".join(
    f"COALESCE(exit_reason_detailed, '') NOT LIKE '{_p}'"
    for _p in _NOT_REAL_PATTERNS
).replace("' COALESCE", "' AND COALESCE")


def _capital_base() -> float:
    """Account equity base. Defaults to 196000 (paper NLV).

    Go-live MUST set AIT_CAPITAL_BASE to the funded amount, otherwise
    return/drawdown percentages are computed off the wrong base.
    """
    try:
        from ait.config.runtime_env import capital_base as _cb
        return _cb()
    except Exception:  # noqa: BLE001 - R16: single authority, safe fallback
        try:
            return float(os.environ.get("AIT_CAPITAL_BASE", "196000"))
        except (TypeError, ValueError):
            return 196000.0


def _annualization_factor(trades: list[dict]) -> float:
    """sqrt(trades-per-year) — replicates ait.backtesting.result.annualization_factor.

    sqrt(252) treated every TRADE as one trading DAY, overstating
    Sharpe/Sortino ~4x at typical trade frequency (see result.py BT-H4 note).
    Span runs first→last exit (COALESCE to entry when exit missing); capped
    at daily sqrt(252).
    """
    try:
        dates = sorted(
            datetime.fromisoformat(str(t.get("exit_time") or t.get("entry_time"))[:19]).date()
            for t in trades
            if (t.get("exit_time") or t.get("entry_time"))
        )
        if len(dates) < 2:
            return 1.0
        span_days = max((dates[-1] - dates[0]).days, 1)
        trades_per_year = len(dates) / (span_days / 365.25)
        return math.sqrt(min(252.0, max(trades_per_year, 1.0)))
    except Exception:
        return 1.0


def _get_duck():
    """Get DuckDB analytics instance (cached, returns None if unavailable)."""
    try:
        from ait.monitoring.duckdb_analytics import DuckDBAnalytics
        return DuckDBAnalytics()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SQL query, returning an empty DataFrame on error."""
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _safe_fetchall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Run a SQL query, returning a list of dicts (empty on error)."""
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _safe_fetchone(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    """Run a SQL query, returning a single dict or None."""
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _get_state_value(conn: sqlite3.Connection, key: str) -> str | None:
    """Read a value from the state table."""
    row = _safe_fetchone(conn, "SELECT value FROM bot_state WHERE key = ?", (key,))
    return row["value"] if row else None


def _get_state_json(conn: sqlite3.Connection, key: str) -> dict | list | None:
    """Read and parse a JSON value from the state table."""
    raw = _get_state_value(conn, key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Panel models (W6: string-contracts-5, db-contracts-6, dead-surface-3/-4)
#
# The System Health tab read FIVE bot_state keys that nothing in the codebase
# ever wrote, so every panel deterministically took its fallback branch — and
# the error panel's fallback was an affirmative green st.success('No errors
# logged'), which would have shown all-clear through the documented full-day
# outage.
#
# The DECISION for each panel now lives in a pure function here (testable
# without Streamlit, and one place per decision); the _tab_* renderers only
# turn the result into widgets.  The invariant every one of them obeys:
#
#     an absent channel renders as NOT WIRED, naming the missing producer.
#     It never renders as green, and never as a bare zero.
# ---------------------------------------------------------------------------

#: Real data — render it.
PANEL_OK = "ok"
#: Channel is wired and current, and genuinely has nothing to show.
PANEL_EMPTY = "empty"
#: Channel is wired but its last publish is old: values are historical.
PANEL_STALE = "stale"
#: No producer exists for this channel. NEVER green.
PANEL_NOT_WIRED = "not_wired"

#: Options contract multiplier — mirrors execution/portfolio.py:401, which is
#: what produced the peak_pnl_pct fractions we divide into.
_OPTION_MULTIPLIER = 100

_STATUS_ICONS = {
    "healthy": "🟢", "ok": "🟢", "running": "🟢",
    "degraded": "🟡",
    "down": "🔴",
    "unknown": "⚪",
}


def _ops_health():
    """ait.monitoring.ops_health, or None when src/ isn't importable."""
    try:
        from ait.monitoring import ops_health
        return ops_health
    except Exception:  # noqa: BLE001
        return None


def health_channel(conn: sqlite3.Connection, *, now: datetime | None = None):
    """Wiring/freshness beacon for the watchdog's bot_state health channel."""
    oh = _ops_health()
    if oh is None:
        return None
    return oh.read_channel_state(conn, now=now)


def component_status_panel(conn: sqlite3.Connection, *,
                           now: datetime | None = None) -> dict:
    """Component Status panel model (db-contracts-6).

    Reads the same ``bot_state LIKE 'watchdog_%'`` contract the panel always
    read — which ait.monitoring.watchdog now actually writes — but excludes the
    two sibling keys that are not component rows, and refuses to show a green
    dot from a channel that stopped publishing.
    """
    oh = _ops_health()
    skip = set(oh.NON_COMPONENT_KEYS) if oh else {"watchdog_channel", "watchdog_errors"}
    rows = _safe_fetchall(
        conn,
        "SELECT key, value, updated_at FROM bot_state "
        "WHERE key LIKE 'watchdog_%' ORDER BY key",
    )
    rows = [r for r in rows if r.get("key") not in skip]
    channel = health_channel(conn, now=now)
    stale = bool(channel is not None and channel.wired and not channel.fresh)

    components = []
    for row in rows:
        raw = row.get("value")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            parsed = raw
        name = str(row.get("key", "")).replace("watchdog_", "").replace("_", " ").title()
        if isinstance(parsed, dict):
            status = str(parsed.get("status", "unknown")).lower()
            components.append({
                "name": name,
                "status": status,
                # A stale channel can only report history, so no dot may read
                # as "currently green".
                "icon": "⚪" if stale else _STATUS_ICONS.get(status, "🔴"),
                "last_seen": parsed.get("last_heartbeat", parsed.get("last_seen", "")),
                "error_count": parsed.get("error_count", 0),
                "last_error": parsed.get("last_error", ""),
                "raw": parsed,
            })
        else:
            components.append({"name": name, "status": "unknown", "icon": "⚪",
                               "last_seen": "", "error_count": 0,
                               "last_error": "", "raw": parsed})

    if not components:
        return {
            "state": PANEL_NOT_WIRED,
            "components": [],
            "channel": channel,
            "message": (
                "Component Status NOT WIRED — no bot_state 'watchdog_*' key has "
                "ever been written. The writer is ait.monitoring.watchdog's "
                "HealthStatePublisher; if this persists while the bot is up, the "
                "publisher is disabled or cannot reach data/ait_state.db. "
                "This is NOT an all-clear."
            ),
        }
    if stale:
        return {"state": PANEL_STALE, "components": components, "channel": channel,
                "message": channel.detail}
    return {"state": PANEL_OK, "components": components, "channel": channel,
            "message": channel.detail if channel else ""}


def memory_panel(conn: sqlite3.Connection, *, now: datetime | None = None) -> dict:
    """Memory Usage panel model (db-contracts-6)."""
    raw = _get_state_value(conn, "system_memory_usage")
    channel = health_channel(conn, now=now)
    if raw is None:
        return {
            "state": PANEL_NOT_WIRED,
            "memory": None,
            "channel": channel,
            "message": ("Memory Usage NOT WIRED — nothing has written "
                        "bot_state['system_memory_usage']. Writer: "
                        "ait.monitoring.watchdog."),
        }
    try:
        mem = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        mem = raw
    stale = bool(channel is not None and channel.wired and not channel.fresh)
    return {
        "state": PANEL_STALE if stale else PANEL_OK,
        "memory": mem,
        "channel": channel,
        "message": channel.detail if channel else "",
    }


def error_panel(conn: sqlite3.Connection, *, now: datetime | None = None) -> dict:
    """Recent Errors panel model (string-contracts-5) — the worst offender.

    The old fallback was ``st.success('No errors logged')``: a GREEN all-clear
    whose real meaning was "no error channel exists".  Four outcomes now, and
    only one of them is green:

      ok        — errors recorded, show them
      empty     — the channel is wired AND current and holds no errors
      stale     — the channel is wired but stopped publishing (NOT an all-clear)
      not_wired — nothing writes here at all       (NOT an all-clear)
    """
    channel = health_channel(conn, now=now)
    errors = None
    source = None
    # bot_state['error_log'] has no producer in src/ (the orchestrator's
    # _note_loop_error path goes to logs + Telegram + in-memory streaks only),
    # but it stays as the primary read so a future writer needs no dashboard
    # change.  watchdog_errors is the channel that is actually wired today.
    for key in ("error_log", "watchdog_errors"):
        raw = _get_state_value(conn, key)
        if raw is None:
            continue
        source = key
        try:
            errors = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            errors = raw
        break

    if source is None:
        return {
            "state": PANEL_NOT_WIRED,
            "errors": [],
            "source": None,
            "channel": channel,
            "message": (
                "Error channel NOT WIRED — neither bot_state['error_log'] nor "
                "bot_state['watchdog_errors'] has ever been written, so this "
                "panel cannot show an error even during a total outage. "
                "This is NOT an all-clear: read logs/ait.log."
            ),
        }
    if isinstance(errors, list) and errors:
        return {"state": PANEL_OK, "errors": errors, "source": source,
                "channel": channel,
                "message": f"{len(errors)} recorded via bot_state['{source}']"}
    if not isinstance(errors, list):
        return {"state": PANEL_OK, "errors": errors, "source": source,
                "channel": channel,
                "message": f"raw value from bot_state['{source}']"}

    stale = bool(channel is not None and channel.wired and not channel.fresh)
    if stale or channel is None or not channel.wired:
        detail = channel.detail if channel is not None else "health channel unreadable"
        return {
            "state": PANEL_STALE,
            "errors": [],
            "source": source,
            "channel": channel,
            "message": (f"No errors in bot_state['{source}'], but the {detail} "
                        "An empty list from a channel that stopped publishing "
                        "is NOT an all-clear."),
        }
    return {
        "state": PANEL_EMPTY,
        "errors": [],
        "source": source,
        "channel": channel,
        "message": (f"No errors recorded — the watchdog error channel is live "
                    f"(last publish {channel.updated_at})."),
    }


def model_info_panel(conn: sqlite3.Connection) -> dict:
    """Model Info panel model (string-contracts-5).

    bot_state['model_version'] has no writer anywhere in src/, so the panel was
    permanently blank.  ``trade_context.model_version`` IS written on every
    entry (state.py:721-732), so the honest reading is "the model version
    recorded on the most recent trade" — real data, correctly labelled, instead
    of a dead key.
    """
    raw = _get_state_value(conn, "model_version")
    if raw is not None:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = raw
        return {"state": PANEL_OK, "source": "bot_state['model_version']",
                "model": parsed, "message": ""}

    row = _safe_fetchone(
        conn,
        "SELECT tc.model_version AS model_version, t.entry_time AS entry_time, "
        "       tc.trade_id AS trade_id "
        "FROM trade_context tc JOIN trades t ON t.trade_id = tc.trade_id "
        "WHERE COALESCE(tc.model_version, '') != '' "
        "ORDER BY t.entry_time DESC LIMIT 1",
    )
    if row:
        return {
            "state": PANEL_OK,
            "source": "trade_context.model_version",
            "model": {
                "model_version": row["model_version"],
                "recorded_on_trade": row["trade_id"],
                "entry_time": (row["entry_time"] or "")[:19],
            },
            "message": ("bot_state['model_version'] has no writer in src/; this "
                        "is the version stamped on the most recent trade's "
                        "context row."),
        }
    return {
        "state": PANEL_NOT_WIRED,
        "source": None,
        "model": None,
        "message": ("Model version NOT WIRED — nothing writes "
                    "bot_state['model_version'], and no trade_context row "
                    "carries a version either."),
    }


def meta_label_panel(conn: sqlite3.Connection) -> dict:
    """Meta-Label Filter panel model (string-contracts-5).

    The old fallback asserted a FACT the dashboard cannot know — "Meta-labeler
    not yet trained (needs 30+ closed trades with context)" — when the truth is
    that nothing writes the key, trained or not.
    """
    stats = _get_state_json(conn, "meta_label_stats")
    if isinstance(stats, dict) and stats:
        return {"state": PANEL_OK, "stats": stats, "message": ""}
    if stats is not None:
        return {
            "state": PANEL_NOT_WIRED,
            "stats": None,
            "message": ("bot_state['meta_label_stats'] holds a value this panel "
                        "cannot read (expected a JSON object)."),
        }
    return {
        "state": PANEL_NOT_WIRED,
        "stats": None,
        "message": ("Meta-label stats NOT WIRED — nothing in src/ writes "
                    "bot_state['meta_label_stats'], so this panel stays blank "
                    "whether or not the meta-labeler has been trained. It is "
                    "not evidence about training state."),
    }


def crash_panel(fatal_log=None, legacy_logs=None) -> dict:
    """Native-crash panel model (log-contracts-3).

    status.py still counts 'Windows fatal exception' in bot_stdout.log, where
    faulthandler stopped writing when src/ait/main.py:43-48 moved the sink to
    logs/fatal.log — so it prints "native crashes: 0" forever.  This panel
    counts the real sink, and reports NOT WIRED when the sink is absent rather
    than a reassuring zero.
    """
    oh = _ops_health()
    if oh is None:
        return {"state": PANEL_NOT_WIRED, "report": None,
                "message": "ait.monitoring.ops_health unavailable."}
    kw = {}
    if fatal_log is not None:
        kw["fatal_log"] = fatal_log
    if legacy_logs is not None:
        kw["legacy_logs"] = legacy_logs
    report = oh.native_crash_report(**kw)
    if not report["channel_wired"]:
        return {"state": PANEL_NOT_WIRED, "report": report,
                "message": report["detail"]}
    return {
        "state": PANEL_OK if report["count"] else PANEL_EMPTY,
        "report": report,
        "message": report["detail"],
    }


def liveness_panel() -> dict:
    """Bot-liveness panel model (bot-day-02) — evidence, not process existence."""
    oh = _ops_health()
    if oh is None:
        return {"state": PANEL_NOT_WIRED, "verdict": None,
                "message": "ait.monitoring.ops_health unavailable."}
    verdict = oh.bot_liveness()
    return {
        "state": PANEL_OK if verdict.ok else PANEL_STALE,
        "verdict": verdict,
        "message": verdict.detail,
    }


def add_capture_efficiency(exit_data: pd.DataFrame) -> pd.DataFrame:
    """Add realized_pnl_pct / capture_pct in a CONSISTENT basis (dead-surface-4).

    The panel showed "Avg Capture Efficiency -1653%", computed as
    ``realized_pnl / (peak_pnl_pct * 100) * 100`` — absolute DOLLARS divided by
    a FRACTION.  It is dimensionally meaningless (doubling position size
    doubles "efficiency"), and a $127 loss on a tiny-peak condor dominated the
    average at -16376%.

    peak_pnl_pct is a fraction of cost basis (portfolio.py:458-459
    ``pnl_pct = unrealized_pnl / cost_basis``), so realized P&L must be put in
    the same basis before the ratio means anything:

        cost_basis   = abs(entry_price) * quantity * multiplier
        realized_pct = realized_pnl / cost_basis
        capture_pct  = realized_pct / peak_pnl_pct * 100

    Rows without a usable cost basis get NaN rather than a fabricated number.
    """
    df = exit_data.copy()
    multiplier = _OPTION_MULTIPLIER
    if "contract_type" in df.columns:
        mult = df["contract_type"].apply(
            lambda c: 1 if str(c).lower() == "stock" else _OPTION_MULTIPLIER)
    else:
        mult = pd.Series([multiplier] * len(df), index=df.index)

    entry = pd.to_numeric(df.get("entry_price"), errors="coerce").abs()
    qty = pd.to_numeric(df.get("quantity"), errors="coerce")
    realized = pd.to_numeric(df.get("realized_pnl"), errors="coerce")
    peak = pd.to_numeric(df.get("peak_pnl_pct"), errors="coerce")

    cost_basis = entry * qty * mult
    cost_basis = cost_basis.where(cost_basis > 0)
    df["cost_basis"] = cost_basis
    df["realized_pnl_pct"] = realized / cost_basis
    df["capture_pct"] = (df["realized_pnl_pct"] / peak.where(peak > 0)) * 100
    return df


def capture_efficiency_panel(exit_data: pd.DataFrame) -> dict:
    """Profit-capture panel model (dead-surface-4)."""
    if exit_data is None or exit_data.empty:
        return {"state": PANEL_EMPTY, "rows": None, "avg_capture": None,
                "avg_peak": None, "dropped": 0,
                "message": "No closed trades with exit journalling yet."}
    peak = pd.to_numeric(exit_data.get("peak_pnl_pct"), errors="coerce")
    has_peak = exit_data[peak > 0]
    if has_peak.empty:
        return {"state": PANEL_EMPTY, "rows": None, "avg_capture": None,
                "avg_peak": None, "dropped": 0,
                "message": "No trades with peak P&L data yet."}
    enriched = add_capture_efficiency(has_peak)
    usable = enriched[enriched["capture_pct"].notna()]
    dropped = int(len(enriched) - len(usable))
    if usable.empty:
        return {
            "state": PANEL_NOT_WIRED, "rows": enriched, "avg_capture": None,
            "avg_peak": float(pd.to_numeric(enriched["peak_pnl_pct"],
                                            errors="coerce").mean()),
            "dropped": dropped,
            "message": (f"Capture efficiency not computable for any of "
                        f"{len(enriched)} trades — entry_price/quantity give no "
                        "cost basis, so realized P&L cannot be put in the same "
                        "unit as peak_pnl_pct."),
        }
    # The MEDIAN is the headline: capture is a ratio with peak_pnl_pct in the
    # denominator, so a trade that barely went positive before reversing
    # (NVDA: peak +0.78%, closed -45%) legitimately scores -5828% and drags any
    # mean far outside the range of every other trade. The mean is still
    # reported, labelled, next to it.
    notes = []
    if dropped:
        notes.append(f"{dropped} trade(s) excluded: no usable cost basis.")
    outliers = int((usable["capture_pct"].abs() > 500).sum())
    if outliers:
        notes.append(f"{outliers} trade(s) beyond +/-500% (near-zero peak in the "
                     "denominator) skew the mean; the median is the headline.")
    return {
        "state": PANEL_OK,
        "rows": enriched,
        "avg_capture": float(usable["capture_pct"].mean()),
        "median_capture": float(usable["capture_pct"].median()),
        "outliers": outliers,
        "avg_peak": float(pd.to_numeric(enriched["peak_pnl_pct"],
                                        errors="coerce").mean()),
        "dropped": dropped,
        "message": " ".join(notes),
    }


def direction_accuracy_panel(exit_data: pd.DataFrame) -> dict:
    """ML Direction Accuracy panel model (dead-surface-3).

    ``trades.direction_correct`` sits at its DDL default -1 on every row:
    close_trade's UPDATE (state.py:386-394) never writes it, despite
    record_trade's docstring promising it is "populated on close".  The old
    fallback, ``st.info('No direction accuracy data yet (needs trade context)')``,
    reads as "keep trading and it will fill in" — it never will.  State it.
    """
    if exit_data is None or exit_data.empty or "direction_correct" not in exit_data:
        return {"state": PANEL_NOT_WIRED, "correct": 0, "total": 0,
                "right_but_lost": 0, "unset": 0,
                "message": ("Direction accuracy NOT RECORDED — no closed trades "
                            "carry a direction_correct value.")}
    col = pd.to_numeric(exit_data["direction_correct"], errors="coerce")
    known = exit_data[col.isin([0, 1])]
    unset = int((col == -1).sum())
    if known.empty:
        return {
            "state": PANEL_NOT_WIRED,
            "correct": 0, "total": 0, "right_but_lost": 0, "unset": unset,
            "message": (
                f"Direction accuracy NOT RECORDED — all {unset} closed trade(s) "
                "sit at the DDL default direction_correct = -1. "
                "StateManager.close_trade (state.py:386-394) does not write the "
                "column, so this panel can never populate no matter how many "
                "trades close. It is not 'no data yet'."
            ),
        }
    correct = int(pd.to_numeric(known["direction_correct"], errors="coerce").sum())
    total = int(len(known))
    right_but_lost = int(len(known[
        (pd.to_numeric(known["direction_correct"], errors="coerce") == 1)
        & (pd.to_numeric(known["realized_pnl"], errors="coerce") <= 0)
    ]))
    return {
        "state": PANEL_OK,
        "correct": correct,
        "total": total,
        "accuracy": correct / total * 100 if total else 0.0,
        "right_but_lost": right_but_lost,
        "unset": unset,
        "message": (f"{unset} further closed trade(s) still unwritten "
                    "(direction_correct = -1)." if unset else ""),
    }


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _tab_portfolio_overview(conn: sqlite3.Connection) -> None:
    import streamlit as st

    col1, col2, col3, col4 = st.columns(4)

    today_str = date.today().isoformat()
    stats = _safe_query(
        conn, "SELECT * FROM daily_stats WHERE date = ?", (today_str,)
    )

    if not stats.empty:
        s = stats.iloc[0]
        col1.metric("Today's P&L", f"${s.get('total_pnl', 0):.2f}")
        col2.metric("Trades Today", int(s.get("trades_taken", 0)))
        taken = int(s.get("trades_taken", 0))
        won = int(s.get("trades_won", 0))
        wr = f"{won / taken * 100:.0f}%" if taken > 0 else "N/A"
        col3.metric("Win Rate", wr)
        col4.metric("Day Trades Used", f"{int(s.get('day_trades_count', 0))}/3")
    else:
        col1.metric("Today's P&L", "$0.00")
        col2.metric("Trades Today", 0)
        col3.metric("Win Rate", "N/A")
        col4.metric("Day Trades Used", "0/3")

    st.divider()

    # Open Positions (with HWM from open_positions table)
    st.subheader("Open Positions")
    open_positions = _safe_query(
        conn,
        "SELECT t.symbol, t.strategy, t.direction, t.entry_price, t.quantity, "
        "t.entry_time, t.ml_confidence, "
        "COALESCE(op.high_water_mark, 0) as peak_pnl_pct "
        "FROM trades t "
        "LEFT JOIN open_positions op ON t.trade_id = op.trade_id "
        "WHERE t.status IN ('filled', 'partial') "
        "ORDER BY t.entry_time DESC",
    )
    if not open_positions.empty:
        st.dataframe(open_positions, use_container_width=True)
    else:
        st.info("No open positions")

    st.divider()

    # Daily P&L chart
    st.subheader("Daily P&L")
    daily = _safe_query(
        conn,
        "SELECT date, total_pnl, trades_taken FROM daily_stats ORDER BY date",
    )
    if not daily.empty:
        daily["cumulative_pnl"] = daily["total_pnl"].cumsum()

        fig = go.Figure()
        colors = ["green" if v >= 0 else "red" for v in daily["total_pnl"]]
        fig.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["total_pnl"],
                name="Daily P&L",
                marker_color=colors,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["cumulative_pnl"],
                name="Cumulative",
                yaxis="y2",
                line=dict(color="royalblue", width=2),
            )
        )
        fig.update_layout(
            yaxis=dict(title="Daily P&L ($)"),
            yaxis2=dict(overlaying="y", side="right", title="Cumulative P&L ($)"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily stats yet")


def _tab_trade_history(
    conn: sqlite3.Connection, start_date: date, end_date: date
) -> None:
    import streamlit as st

    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()
    duck = _get_duck()

    # Recent trades (still from SQLite — includes pending/open trades)
    st.subheader("Recent Trades")
    recent = _safe_query(
        conn,
        # exit-time window (COALESCE→entry for open trades): a trade opened
        # before the window but closed inside it must appear in the range
        "SELECT trade_id, symbol, strategy, direction, status, "
        "entry_price, exit_price, realized_pnl, entry_time, exit_time "
        "FROM trades WHERE date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ? "
        "ORDER BY entry_time DESC LIMIT 100",
        (start_iso, end_iso),
    )
    if not recent.empty:
        st.dataframe(recent, use_container_width=True)
    else:
        st.info("No trades in the selected date range")

    st.divider()

    # Strategy performance — use DuckDB when available
    st.subheader("Strategy Performance")
    strategy_data = None
    if duck:
        try:
            lookback = (end_date - start_date).days or 60
            strategy_data = duck.get_strategy_breakdown(lookback)
        except Exception:
            strategy_data = None

    if strategy_data:
        strategy_perf = pd.DataFrame(strategy_data)
        st.dataframe(strategy_perf, use_container_width=True)
    else:
        strategy_perf = _safe_query(
            conn,
            # exit-time window + real-close filter (bookkeeping closes must
            # not count in win-rate/P&L aggregates)
            f"SELECT strategy, COUNT(*) as trades, "
            f"SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
            f"ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct, "
            f"ROUND(SUM(realized_pnl), 2) as total_pnl, "
            f"ROUND(AVG(realized_pnl), 2) as avg_pnl "
            f"FROM trades WHERE status = 'closed' "
            f"AND date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ? "
            f"AND {_REAL_CLOSE_SQL} "
            f"GROUP BY strategy ORDER BY total_pnl DESC",
            (start_iso, end_iso),
        )
        if not strategy_perf.empty:
            st.dataframe(strategy_perf, use_container_width=True)
        else:
            st.info("No closed trades for strategy breakdown")

    st.divider()

    # Symbol performance — use DuckDB when available
    st.subheader("Symbol Performance")
    symbol_data = None
    if duck:
        try:
            lookback = (end_date - start_date).days or 60
            symbol_data = duck.get_symbol_breakdown(lookback)
        except Exception:
            symbol_data = None

    if symbol_data:
        symbol_perf = pd.DataFrame(symbol_data)
        st.dataframe(symbol_perf, use_container_width=True)
    else:
        symbol_perf = _safe_query(
            conn,
            # exit-time window + real-close filter (see strategy query above)
            f"SELECT symbol, COUNT(*) as trades, "
            f"SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
            f"ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct, "
            f"ROUND(SUM(realized_pnl), 2) as total_pnl, "
            f"ROUND(AVG(realized_pnl), 2) as avg_pnl "
            f"FROM trades WHERE status = 'closed' "
            f"AND date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ? "
            f"AND {_REAL_CLOSE_SQL} "
            f"GROUP BY symbol ORDER BY total_pnl DESC",
            (start_iso, end_iso),
        )
        if not symbol_perf.empty:
            st.dataframe(symbol_perf, use_container_width=True)
        else:
            st.info("No closed trades for symbol breakdown")

    st.divider()

    # NEW: Regime breakdown (DuckDB only — new analytics)
    if duck:
        st.subheader("Regime Performance")
        try:
            lookback = (end_date - start_date).days or 60
            regime_data = duck.get_regime_breakdown(lookback)
            if regime_data:
                regime_df = pd.DataFrame(regime_data)
                st.dataframe(regime_df, use_container_width=True)
            else:
                st.info("No regime data yet")
        except Exception:
            pass

    st.divider()

    # ML Confidence calibration — does the model's confidence predict outcomes?
    st.subheader("ML Confidence Calibration")
    st.caption(
        "If model is well-calibrated, win rate should rise with confidence bucket. "
        "If high-confidence trades don't win more, the threshold is meaningless."
    )
    conf_perf = _safe_query(
        conn,
        # exit-time window + real-close filter (see strategy query above)
        f"""
        SELECT
            CASE
                WHEN ml_confidence < 0.55 THEN '1. low (<0.55)'
                WHEN ml_confidence < 0.70 THEN '2. mid (0.55-0.70)'
                WHEN ml_confidence < 0.85 THEN '3. high (0.70-0.85)'
                ELSE '4. very_high (>=0.85)'
            END as confidence_bucket,
            COUNT(*) as trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct,
            ROUND(SUM(realized_pnl), 2) as total_pnl,
            ROUND(AVG(realized_pnl), 2) as avg_pnl
        FROM trades
        WHERE status = 'closed'
          AND date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ?
          AND {_REAL_CLOSE_SQL}
          AND ml_confidence > 0
        GROUP BY confidence_bucket
        ORDER BY confidence_bucket
        """,
        (start_iso, end_iso),
    )
    if not conf_perf.empty:
        st.dataframe(conf_perf, use_container_width=True)
    else:
        st.info("Not enough data for confidence calibration yet")

    st.divider()

    # Day of week breakdown
    st.subheader("Day-of-Week Performance")
    dow_perf = _safe_query(
        conn,
        # exit-time window + real-close filter; grouping stays on ENTRY
        # weekday — the question is entry timing
        f"""
        SELECT
            CASE strftime('%w', entry_time)
                WHEN '1' THEN '1. Monday'
                WHEN '2' THEN '2. Tuesday'
                WHEN '3' THEN '3. Wednesday'
                WHEN '4' THEN '4. Thursday'
                WHEN '5' THEN '5. Friday'
            END as day_of_week,
            COUNT(*) as trades,
            ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct,
            ROUND(SUM(realized_pnl), 2) as total_pnl,
            ROUND(AVG(realized_pnl), 2) as avg_pnl
        FROM trades
        WHERE status = 'closed'
          AND date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ?
          AND {_REAL_CLOSE_SQL}
        GROUP BY day_of_week
        ORDER BY day_of_week
        """,
        (start_iso, end_iso),
    )
    if not dow_perf.empty:
        st.dataframe(dow_perf, use_container_width=True)
    else:
        st.info("No day-of-week data yet")


def _bt_period_years(bt: dict) -> tuple[float, bool]:
    """Backtest span in years derived from the report JSON's date range.

    Returns (years, assumed). assumed=True means no date range was found and
    the 4y fallback is a fabrication — the UI must label it so the reader
    knows the annualized figure rests on an assumption.
    """
    def _parse_date(s):
        try:
            return date.fromisoformat(str(s)[:10])
        except (ValueError, TypeError):
            return None

    start = _parse_date(bt.get("start_date") or bt.get("period_start"))
    end = _parse_date(bt.get("end_date") or bt.get("period_end"))
    if not (start and end):
        period = bt.get("backtest_period") or bt.get("period") or ""
        if " to " in str(period):
            a, b = str(period).split(" to ", 1)
            start, end = _parse_date(a.strip()), _parse_date(b.strip())
    if start and end and end > start:
        return max((end - start).days / 365.25, 1 / 365.25), False
    return 4.0, True


def _live_vs_backtest_panel(conn: sqlite3.Connection) -> None:
    """Compare live realized P&L vs the most recent backtest expectations.

    Without this, every "the backtest says X" is a leap of faith.
    """
    import streamlit as st
    import json
    import pathlib
    from datetime import date, timedelta

    st.subheader("📊 Live vs Backtest")
    st.caption(
        "Annualized comparison of live performance against the most recent "
        "backtest run. A negative gap means live is underperforming "
        "simulation — investigate slippage, fills, or data quality."
    )

    # Find the latest backtest report
    reports_dir = pathlib.Path(__file__).resolve().parents[3] / "reports"
    backtest_files = sorted(reports_dir.glob("backtest_*.json"), reverse=True)
    if not backtest_files:
        st.info("No backtest reports yet. Run `python run_backtest.py` to generate one.")
        return

    latest = backtest_files[0]
    try:
        with open(latest) as f:
            bt = json.load(f)
    except Exception:
        st.warning(f"Could not read {latest.name}")
        return

    # Parse backtest metrics (stored as strings like "+5.84%")
    def _parse_pct(s):
        if not s or s == "?":
            return None
        try:
            return float(str(s).replace("%", "").replace("+", "").strip()) / 100
        except (ValueError, AttributeError):
            return None

    bt_return = _parse_pct(bt.get("total_return"))
    bt_sharpe = bt.get("sharpe", "?")
    bt_win_rate = _parse_pct(bt.get("win_rate"))
    bt_max_dd = _parse_pct(bt.get("max_drawdown"))

    # Get live stats — last 30 days of closed trades.
    # Window on EXIT time: a trade opened 35 days ago but closed yesterday at
    # a big loss must count. Exclude bookkeeping closes (never_filled etc.).
    end = date.today()
    start = end - timedelta(days=30)
    live_trades = _safe_fetchall(
        conn,
        f"SELECT realized_pnl, entry_time, exit_time FROM trades "
        f"WHERE status = 'closed' "
        f"AND date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ? "
        f"AND {_REAL_CLOSE_SQL}",
        (start.isoformat(), end.isoformat()),
    )
    # Same population in numerator and denominator: the old truthiness check
    # dropped $0 scratch trades from wins/P&L while the denominator counted
    # all rows. A $0 trade is a non-win, not a dropped row.
    live_pnls = [t["realized_pnl"] for t in live_trades if t.get("realized_pnl") is not None]
    live_total_pnl = sum(live_pnls)
    live_trade_count = len(live_pnls)
    live_wins = sum(1 for p in live_pnls if p > 0)
    live_win_rate = live_wins / live_trade_count if live_trade_count > 0 else None

    # Get account value for return calc — fallback is AIT_CAPITAL_BASE
    # (paper NLV default), not a hardcoded 250k that misstates returns
    nlv_row = _safe_fetchall(
        conn,
        "SELECT value FROM bot_state WHERE key = 'account_value' LIMIT 1",
    )
    nlv = float(nlv_row[0]["value"]) if nlv_row else _capital_base()
    live_return_30d = (live_total_pnl / nlv) if nlv > 0 else 0

    # Annualize: derive the backtest span from the report's date range —
    # only fall back to the 4y guess when absent, and say so in the UI
    bt_years, bt_years_assumed = _bt_period_years(bt)
    bt_annualized = (1 + bt_return) ** (1 / bt_years) - 1 if bt_return is not None else None
    live_annualized = ((1 + live_return_30d) ** 12) - 1 if live_return_30d else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Backtest Return (annualized)"
        + (" (assumed 4y)" if bt_years_assumed else f" ({bt_years:.1f}y)"),
        f"{bt_annualized:.1%}" if bt_annualized is not None else "n/a",
    )
    col2.metric(
        "Live Return (annualized, last 30d)",
        f"{live_annualized:.1%}",
    )
    if bt_annualized is not None and bt_annualized != 0:
        gap = live_annualized - bt_annualized
        col3.metric(
            "Gap",
            f"{gap:+.1%}",
            delta=f"{(gap / abs(bt_annualized)):+.0%} vs backtest",
        )
    else:
        col3.metric("Gap", "n/a")

    st.write("**Detail:**")
    detail = pd.DataFrame({
        "metric": [
            "Total Return", "Sharpe Ratio", "Win Rate", "Max Drawdown",
            "Trade Count (30d)", "Total P&L (30d)",
        ],
        "Backtest": [
            f"{bt_return:.1%}" if bt_return is not None else "n/a",
            str(bt_sharpe),
            f"{bt_win_rate:.1%}" if bt_win_rate is not None else "n/a",
            f"{bt_max_dd:.1%}" if bt_max_dd is not None else "n/a",
            "n/a",
            "n/a",
        ],
        "Live (30d)": [
            f"{live_return_30d:.2%}",
            "n/a (need 100+ trades)",
            f"{live_win_rate:.1%}" if live_win_rate is not None else "n/a",
            "n/a",
            str(live_trade_count),
            f"${live_total_pnl:,.2f}",
        ],
    })
    st.dataframe(detail, hide_index=True, use_container_width=True)

    st.caption(
        f"Backtest source: {latest.name}  ·  "
        f"Live window: {start} → {end}  ·  "
        f"Need 50+ live trades for statistical confidence."
    )


def _tab_analytics(conn: sqlite3.Connection) -> None:
    import streamlit as st

    _live_vs_backtest_panel(conn)
    st.divider()

    # Gather closed trades for analytics — exclude bookkeeping closes
    # (never_filled/pending/migrated) and order by CLOSE time so the equity
    # curve, drawdown and streaks follow realization order
    trades = _safe_fetchall(
        conn,
        f"SELECT realized_pnl, entry_time, exit_time FROM trades "
        f"WHERE status = 'closed' AND {_REAL_CLOSE_SQL} "
        f"ORDER BY COALESCE(exit_time, entry_time)",
    )

    pnls = [t["realized_pnl"] for t in trades if t.get("realized_pnl") is not None]

    # Compute metrics
    sharpe = sortino = max_dd = profit_factor = avg_hold = 0.0
    current_streak = max_win_streak = max_loss_streak = 0

    if len(pnls) > 1:
        import statistics

        # Annualize by ACTUAL trade frequency, not sqrt(252) — sqrt(252)
        # treated each trade as one trading day (see
        # ait.backtesting.result.annualization_factor)
        ann = _annualization_factor(trades)
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)

        if std_pnl > 0:
            sharpe = (mean_pnl / std_pnl) * ann

        # Sortino — target-0 downside deviation over ALL returns:
        # sqrt(mean(min(r,0)^2)), matching result.py. The old stdev-of-losses
        # definition gave a consistent loser downside-dev≈0.
        downside_dev = math.sqrt(sum(min(p, 0.0) ** 2 for p in pnls) / len(pnls))
        if downside_dev > 0:
            sortino = (mean_pnl / downside_dev) * ann
        elif mean_pnl > 0:
            sortino = float("inf")

    # Max drawdown
    if pnls:
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

    # Profit factor
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses

    # Streaks
    streak = 0
    for p in pnls:
        if p > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))
    current_streak = streak

    # Average hold time
    hold_hours = []
    for t in trades:
        if t.get("entry_time") and t.get("exit_time"):
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hold_hours.append((exit_ - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
    avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    # Display metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Sortino Ratio", f"{sortino:.2f}")
    c3.metric("Max Drawdown", f"${max_dd:.2f}")
    c4.metric("Profit Factor", f"{profit_factor:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades", len(pnls))
    c6.metric("Win Rate", f"{len(wins) / len(pnls) * 100:.1f}%" if pnls else "N/A")
    c7.metric("Avg Hold Time", f"{avg_hold:.1f}h")
    streak_label = f"+{current_streak}W" if current_streak > 0 else f"{current_streak}L" if current_streak < 0 else "0"
    c8.metric("Current Streak", streak_label)

    st.divider()

    # Equity curve
    st.subheader("Equity Curve")
    if pnls:
        cum = []
        running = 0.0
        dates = []
        for t in trades:
            if t.get("realized_pnl") is not None:
                running += t["realized_pnl"]
                cum.append(running)
                dates.append(t.get("exit_time") or t.get("entry_time") or "")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum,
                mode="lines",
                name="Cumulative P&L",
                fill="tozeroy",
                line=dict(color="royalblue", width=2),
            )
        )
        fig.update_layout(
            yaxis=dict(title="Cumulative P&L ($)"),
            xaxis=dict(title="Time"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades to chart")

    st.divider()

    # Streak detail
    st.subheader("Win/Loss Streaks")
    sc1, sc2 = st.columns(2)
    sc1.metric("Best Win Streak", f"{max_win_streak}")
    sc2.metric("Worst Loss Streak", f"{max_loss_streak}")

    # --- DuckDB-powered advanced analytics ---
    duck = _get_duck()
    if duck:
        st.divider()

        # Rolling Sharpe
        st.subheader("Rolling Sharpe Ratio (20-day)")
        try:
            rolling = duck.get_rolling_sharpe(window_days=20, lookback_days=90)
            if rolling:
                roll_df = pd.DataFrame(rolling)
                fig_rs = go.Figure()
                fig_rs.add_trace(go.Scatter(
                    x=roll_df["date"], y=roll_df["rolling_sharpe"],
                    mode="lines", name="Rolling Sharpe",
                    line=dict(color="orange", width=2),
                ))
                fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_rs.update_layout(yaxis=dict(title="Sharpe Ratio"), height=300)
                st.plotly_chart(fig_rs, use_container_width=True)
            else:
                st.info("Not enough daily data for rolling Sharpe")
        except Exception:
            pass

        st.divider()

        # Confidence band analysis
        st.subheader("Win Rate by ML Confidence Band")
        try:
            bands = duck.get_confidence_band_analysis(lookback_days=90)
            if bands:
                band_df = pd.DataFrame(bands)
                st.dataframe(band_df, use_container_width=True)
            else:
                st.info("No confidence band data")
        except Exception:
            pass

        st.divider()

        # Hourly performance
        st.subheader("Performance by Hour of Day")
        try:
            hourly = duck.get_hourly_performance(lookback_days=90)
            if hourly:
                hourly_df = pd.DataFrame(hourly)
                fig_h = go.Figure()
                colors = ["green" if r["total_pnl"] >= 0 else "red" for r in hourly]
                fig_h.add_trace(go.Bar(
                    x=hourly_df["hour"], y=hourly_df["total_pnl"],
                    name="P&L by Hour", marker_color=colors,
                ))
                fig_h.update_layout(
                    xaxis=dict(title="Hour (ET)", dtick=1),
                    yaxis=dict(title="Total P&L ($)"),
                    height=300,
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("No hourly data")
        except Exception:
            pass

        st.divider()

        # IV Rank analysis
        st.subheader("Strategy Performance by IV Rank")
        try:
            iv_data = duck.get_iv_rank_analysis(lookback_days=90)
            if iv_data:
                iv_df = pd.DataFrame(iv_data)
                st.dataframe(iv_df, use_container_width=True)
            else:
                st.info("No IV rank data (needs trade context)")
        except Exception:
            pass


def _tab_self_learning(conn: sqlite3.Connection) -> None:
    import streamlit as st

    # Learning adaptations from state
    adaptations = _get_state_json(conn, "learning_adaptations")

    st.subheader("Current Adaptations")
    if adaptations and isinstance(adaptations, dict):
        # Strategy multipliers
        multipliers = adaptations.get("strategy_multipliers", {})
        if multipliers:
            st.write("**Strategy Multipliers**")
            mult_df = pd.DataFrame(
                [{"Strategy": k, "Multiplier": f"{v:.2f}"} for k, v in multipliers.items()]
            )
            st.dataframe(mult_df, use_container_width=True, hide_index=True)
        else:
            st.info("No strategy multiplier overrides")

        st.divider()

        # Confidence override
        conf = adaptations.get("confidence_override")
        if conf is not None:
            st.metric("Confidence Override", f"{conf:.2f}")
        else:
            st.info("No confidence override (using default)")

        # Stop loss override
        sl = adaptations.get("stop_loss_override")
        if sl is not None:
            st.metric("Stop Loss Override", f"{sl:.2f}")
        else:
            st.info("No stop loss override (using default)")
    else:
        st.info("No learning adaptations recorded yet")

    st.divider()

    # Disabled strategies
    st.subheader("Disabled Strategies")
    disabled_raw = _get_state_value(conn, "learning_disabled_strategies")
    if disabled_raw:
        try:
            disabled = json.loads(disabled_raw)
        except (json.JSONDecodeError, TypeError):
            disabled = []
    elif adaptations and isinstance(adaptations, dict):
        disabled = adaptations.get("disabled_strategies", [])
    else:
        disabled = []

    if disabled:
        for s in disabled:
            st.warning(f"Disabled: **{s}**")
    else:
        st.success("All strategies enabled")

    st.divider()

    # Removed symbols
    st.subheader("Removed Symbols")
    removed_raw = _get_state_value(conn, "learning_removed_symbols")
    if removed_raw:
        try:
            removed = json.loads(removed_raw)
        except (json.JSONDecodeError, TypeError):
            removed = []
    elif adaptations and isinstance(adaptations, dict):
        removed = adaptations.get("removed_symbols", [])
    else:
        removed = []

    if removed:
        for s in removed:
            st.error(f"Removed: **{s}**")
    else:
        st.success("No symbols removed")

    st.divider()

    # Learning history
    st.subheader("Learning History")
    history = _get_state_json(conn, "learning_history")
    if history and isinstance(history, list):
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No learning history available")

    st.divider()

    # All learning_ keys for transparency
    st.subheader("All Learning State Keys")
    learning_keys = _safe_query(
        conn,
        "SELECT key, value FROM bot_state WHERE key LIKE 'learning_%' ORDER BY key",
    )
    if not learning_keys.empty:
        st.dataframe(learning_keys, use_container_width=True, hide_index=True)
    else:
        st.info("No learning state entries found")


def _tab_trade_intelligence(conn: sqlite3.Connection) -> None:
    """New tab: exit management, meta-label, thesis invalidation insights."""
    import streamlit as st

    # --- Exit Management Overview ---
    st.subheader("Dynamic Exit Management")

    # Trades with journaling data — exclude bookkeeping closes so
    # stale_pending_never_filled etc. don't pollute the exit-reason and
    # direction-accuracy aggregates
    # dead-surface-4: entry_price/quantity/contract_type are selected so
    # realized P&L can be expressed in the SAME basis as peak_pnl_pct (a
    # fraction of cost basis). Without them the panel divided dollars by a
    # percentage.
    exit_data = _safe_query(
        conn,
        f"SELECT symbol, strategy, exit_reason_detailed, peak_pnl_pct, "
        f"realized_pnl, direction_correct, entry_price, quantity, contract_type "
        f"FROM trades WHERE status = 'closed' AND exit_reason_detailed != '' "
        f"AND {_REAL_CLOSE_SQL} "
        f"ORDER BY exit_time DESC LIMIT 50",
    )

    if not exit_data.empty:
        # Exit reason breakdown
        st.write("**Exit Reason Distribution**")
        reason_counts = exit_data["exit_reason_detailed"].apply(
            lambda x: x.split(":")[0] if ":" in str(x) else str(x)
        ).value_counts()
        reason_df = pd.DataFrame({
            "Exit Reason": reason_counts.index,
            "Count": reason_counts.values,
        })
        st.dataframe(reason_df, use_container_width=True, hide_index=True)

        st.divider()

        # Peak vs Realized P&L (profit giveback analysis)
        st.write("**Peak vs Realized P&L (Profit Capture Efficiency)**")
        cap = capture_efficiency_panel(exit_data)
        if cap["state"] == PANEL_OK:
            rows = cap["rows"]
            c1, c2 = st.columns(2)
            c1.metric("Avg Peak P&L %", f"{cap['avg_peak']:.1%}")
            c2.metric("Median Capture Efficiency",
                      f"{cap['median_capture']:.0f}%")
            st.caption(
                "capture = realized P&L / peak P&L, both as a fraction of the "
                "same cost basis (abs(entry_price) x quantity x multiplier). "
                "Before W6 this divided dollars by a percentage and read -1653%. "
                f"Mean across {len(cap['rows'])} trades: "
                f"{cap['avg_capture']:.0f}%."
            )
            if cap["message"]:
                st.caption(cap["message"])

            # dead-surface-4: both series are now PERCENTAGES, so one y-axis is
            # meaningful. It previously mixed peak-% with realized-dollars.
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rows["symbol"],
                y=rows["peak_pnl_pct"] * 100,
                name="Peak P&L %",
                marker_color="lightblue",
            ))
            fig.add_trace(go.Bar(
                x=rows["symbol"],
                y=rows["realized_pnl_pct"] * 100,
                name="Realized P&L %",
                marker_color=["green" if p > 0 else "red"
                              for p in rows["realized_pnl_pct"].fillna(0)],
            ))
            fig.update_layout(barmode="group", height=350,
                              yaxis_title="% of cost basis")
            st.plotly_chart(fig, use_container_width=True)
        elif cap["state"] == PANEL_NOT_WIRED:
            st.warning(cap["message"])
        else:
            st.info(cap["message"])

        st.divider()

        # Direction accuracy
        st.write("**ML Direction Accuracy**")
        dir_panel = direction_accuracy_panel(exit_data)
        if dir_panel["state"] == PANEL_OK:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Direction Correct",
                       f"{dir_panel['correct']}/{dir_panel['total']}")
            dc2.metric("Direction Accuracy", f"{dir_panel['accuracy']:.1f}%")
            dc3.metric("Right Direction, Lost $",
                       f"{dir_panel['right_but_lost']}/{dir_panel['correct']}")
            if dir_panel["right_but_lost"] > 0:
                st.warning(
                    f"{dir_panel['right_but_lost']} trades had correct direction "
                    "but lost money — exit management is the bottleneck, not ML "
                    "predictions."
                )
            if dir_panel["message"]:
                st.caption(dir_panel["message"])
        else:
            # dead-surface-3: NOT "no data yet" — the column is never written.
            st.warning(dir_panel["message"])
    else:
        st.info("No exit intelligence data yet — trades need exit_reason_detailed")

    st.divider()

    # --- High Water Marks on Open Positions ---
    st.subheader("Open Position Health")
    open_hwm = _safe_query(
        conn,
        "SELECT op.trade_id, t.symbol, t.strategy, op.high_water_mark, "
        "op.quantity, op.entry_price "
        "FROM open_positions op "
        "JOIN trades t ON op.trade_id = t.trade_id "
        "WHERE t.status IN ('filled', 'partial') "
        "ORDER BY op.high_water_mark DESC",
    )
    if not open_hwm.empty:
        open_hwm["high_water_mark"] = open_hwm["high_water_mark"].apply(
            lambda x: f"{x:.1%}" if x else "0%"
        )
        st.dataframe(open_hwm, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions with HWM data")

    st.divider()

    # --- Partial Exit History ---
    st.subheader("Partial Exits")
    partial_data = _safe_query(
        conn,
        "SELECT op.trade_id, t.symbol, op.partial_exits "
        "FROM open_positions op "
        "JOIN trades t ON op.trade_id = t.trade_id "
        "WHERE op.partial_exits != '[]'",
    )
    if not partial_data.empty:
        for _, row in partial_data.iterrows():
            try:
                exits = json.loads(row["partial_exits"])
                if exits:
                    st.write(f"**{row['symbol']}** ({row['trade_id']})")
                    exits_df = pd.DataFrame(exits)
                    st.dataframe(exits_df, use_container_width=True, hide_index=True)
            except (json.JSONDecodeError, TypeError):
                pass
    else:
        st.info("No partial exits recorded yet")

    st.divider()

    # --- Meta-Label Stats ---
    st.subheader("Meta-Label Filter")
    meta_panel = meta_label_panel(conn)
    meta_stats = meta_panel["stats"]
    if meta_panel["state"] == PANEL_OK and isinstance(meta_stats, dict):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{meta_stats.get('accuracy', 0):.1%}")
        m2.metric("Precision", f"{meta_stats.get('precision', 0):.1%}")
        m3.metric("Trades Used", meta_stats.get("trades_used", 0))

        top_features = meta_stats.get("top_features", {})
        if top_features:
            st.write("**Top Predictive Features**")
            feat_df = pd.DataFrame([
                {"Feature": k, "Importance": f"{v:.3f}"}
                for k, v in top_features.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
    else:
        # string-contracts-5: the old copy asserted "not yet trained (needs 30+
        # closed trades)" — a claim about training state the dashboard has no
        # way to know, when the truth is that nothing writes the key at all.
        st.warning(meta_panel["message"])


def _tab_system_health(conn: sqlite3.Connection) -> None:
    """System Health tab.

    W6 string-contracts-5 / db-contracts-6: every panel here used to read a
    bot_state key with no writer, and the error panel's fallback was a green
    st.success('No errors logged').  All panel decisions now come from the pure
    models above, whose invariant is that an absent channel says NOT WIRED and
    never renders green.
    """
    import streamlit as st

    # --- Bot liveness (bot-day-02: evidence, not process existence) ---
    st.subheader("Bot Liveness")
    live = liveness_panel()
    verdict = live.get("verdict")
    if verdict is None:
        st.warning(live["message"])
    elif verdict.ok:
        st.success(f"{verdict.state}: {verdict.detail}")
    else:
        st.error(f"{verdict.state}: {verdict.detail}")
    if verdict is not None and verdict.heartbeat_age_s is not None:
        st.caption(f"data/bot_heartbeat age: {verdict.heartbeat_age_s / 60:.1f} min")

    st.divider()

    # --- Watchdog / component status ---
    st.subheader("Component Status")
    panel = component_status_panel(conn)
    if panel["state"] == PANEL_NOT_WIRED:
        st.warning(panel["message"])
    else:
        if panel["state"] == PANEL_STALE:
            st.warning(panel["message"])
        elif panel["message"]:
            st.caption(panel["message"])
        for comp in panel["components"]:
            st.write(f"{comp['icon']} **{comp['name']}** — {comp['status']}")
            bits = []
            if comp["last_seen"]:
                bits.append(f"Last seen: {comp['last_seen']}")
            if comp["error_count"]:
                bits.append(f"errors: {comp['error_count']}")
            if comp["last_error"]:
                bits.append(f"last error: {comp['last_error']}")
            if bits:
                st.caption(" | ".join(str(b) for b in bits))

    st.divider()

    # --- Memory usage ---
    st.subheader("Memory Usage")
    mem_panel = memory_panel(conn)
    if mem_panel["state"] == PANEL_NOT_WIRED:
        st.warning(mem_panel["message"])
    else:
        mem = mem_panel["memory"]
        if isinstance(mem, dict):
            mc1, mc2 = st.columns(2)
            mc1.metric("RSS (MB)", f"{mem.get('rss_mb', 'N/A')}")
            mc2.metric("VMS (MB)", f"{mem.get('vms_mb', 'N/A')}")
        else:
            st.write(f"Memory: {mem}")
        if mem_panel["state"] == PANEL_STALE:
            st.warning(mem_panel["message"])

    st.divider()

    # --- Error log ---
    st.subheader("Recent Errors")
    errs = error_panel(conn)
    if errs["state"] == PANEL_OK:
        payload = errs["errors"]
        if isinstance(payload, list):
            last_20 = list(payload)[-20:]
            last_20.reverse()
            st.dataframe(pd.DataFrame(last_20), use_container_width=True,
                         hide_index=True)
        else:
            st.code(str(payload))
        st.caption(errs["message"])
    elif errs["state"] == PANEL_EMPTY:
        # The ONLY green branch: the channel is wired AND currently publishing.
        st.success(errs["message"])
    else:
        st.warning(errs["message"])

    st.divider()

    # --- Native crashes (log-contracts-3) ---
    st.subheader("Native Crashes")
    crashes = crash_panel()
    if crashes["state"] == PANEL_NOT_WIRED:
        st.warning(crashes["message"])
    else:
        report = crashes["report"]
        cc1, cc2 = st.columns(2)
        cc1.metric("faulthandler dumps", report["count"])
        cc2.metric("in logs/fatal.log", report["fatal_count"])
        st.caption(crashes["message"])
        if report.get("last_write"):
            st.caption(f"fatal.log last written: {report['last_write']}")

    st.divider()

    # --- Model version info ---
    st.subheader("Model Info")
    model = model_info_panel(conn)
    if model["state"] == PANEL_NOT_WIRED:
        st.warning(model["message"])
    else:
        payload = model["model"]
        if isinstance(payload, dict):
            for k, v in payload.items():
                st.write(f"**{k}**: {v}")
        else:
            st.write(f"Model version: {payload}")
        st.caption(f"source: {model['source']}")
        if model["message"]:
            st.caption(model["message"])


# ---------------------------------------------------------------------------
# Backtest tab (walk-forward results)
# ---------------------------------------------------------------------------

_BACKTEST_RESULTS_FILE = Path("backtest_results.json")
_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"]
_STRATEGIES = ["long_call", "long_put", "bull_call_spread", "bear_put_spread", "iron_condor"]


def _load_backtest_results() -> dict | None:
    if _BACKTEST_RESULTS_FILE.exists():
        try:
            with open(_BACKTEST_RESULTS_FILE) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _run_walkforward(symbols, strategies, capital, min_conf) -> dict:
    import asyncio as _asyncio
    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

    cfg = WalkForwardConfig(
        train_days=365,
        test_days=63,
        # step == test window: step_days=21 with 63-day tests re-counted each
        # period ~3x (window-overlap triple-counting fixed in the engine)
        step_days=63,
        gap_days=5,
        initial_capital=capital,
        min_confidence=min_conf,
        trailing_stop_enabled=True,
    )
    bt = WalkForwardBacktester(symbols, strategies, config=cfg)
    result = _asyncio.run(bt.run())

    trades = []
    for w in result.windows:
        for t in w.backtest_result.trades:
            trades.append(t)

    return {
        "total_return": result.total_return,
        "win_rate": result.win_rate,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "consistency": result.consistency,
        "total_trades": result.total_trades,
        "windows": len(result.windows),
        "initial_capital": capital,
        "final_capital": capital * (1 + result.total_return),
        "trades": trades,
        "strategy_results": result.strategy_results,
        "symbol_results": {
            sym: {
                "total_return": r.total_return,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "sharpe_ratio": r.sharpe_ratio,
            }
            for sym, r in result.symbol_results.items()
        },
        "equity_curve": result.equity_curve().to_dict(orient="records"),
        "run_at": datetime.now().isoformat(),
        "symbols": symbols,
        "strategies": strategies,
    }


def _tab_feature_importance() -> None:
    """Show which ML features are pulling weight vs dead-weight.

    Loads saved models and reads model.feature_importances_ per symbol.
    Reveals dead-weight features that could be cut.
    """
    import streamlit as st
    import pickle
    import pathlib

    st.subheader("🧮 Feature Importance — what's helping the model?")
    st.caption(
        "Aggregated across all symbols and both ML models (XGBoost + LightGBM). "
        "Higher = more predictive. Features with importance ~0 are dead weight "
        "and could be removed without hurting accuracy."
    )

    models_dir = pathlib.Path(__file__).resolve().parents[3] / "models"
    model_files = {
        "Direction (3-class)": "ensemble.pkl",
        "Range (iron condor)": "range.pkl",
        "Vol Magnitude (straddle)": "vol_magnitude.pkl",
    }

    model_choice = st.selectbox(
        "Which model?", list(model_files.keys()), index=1,
    )
    pkl_path = models_dir / model_files[model_choice]

    if not pkl_path.exists():
        st.warning(f"Model file not found: {pkl_path.name}. Run a retrain first.")
        return

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    sym_models = data.get("symbol_models", {})
    if not sym_models:
        st.info("No per-symbol models found. Retrain to populate.")
        return

    # Aggregate feature importance across symbols and sub-models (xgb + lgbm)
    feature_totals = {}
    feature_counts = {}
    per_symbol_data = []
    for symbol, sym_data in sym_models.items():
        importances = sym_data.get("feature_importances", {})
        if not importances:
            continue
        # Average across xgb + lgbm for this symbol
        symbol_avg = {}
        for model_name, imps in importances.items():
            for feat, val in imps.items():
                symbol_avg.setdefault(feat, []).append(val)
        for feat, vals in symbol_avg.items():
            avg = sum(vals) / len(vals) if vals else 0
            feature_totals[feat] = feature_totals.get(feat, 0) + avg
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
            per_symbol_data.append({"symbol": symbol, "feature": feat, "importance": avg})

    if not feature_totals:
        st.info(
            "No feature_importances saved on the loaded models yet. "
            "Re-run training (`python run_orchestrator.py --retrain`) — "
            "the schema was added recently."
        )
        return

    # Mean importance across symbols
    feature_avg = {
        feat: total / feature_counts[feat]
        for feat, total in feature_totals.items()
    }

    importance_df = pd.DataFrame(
        sorted(feature_avg.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "avg_importance"],
    )
    importance_df["avg_importance"] = importance_df["avg_importance"].round(4)

    col_l, col_r = st.columns(2)

    with col_l:
        st.write(f"**Top 15 features ({model_choice})**")
        st.dataframe(importance_df.head(15), hide_index=True, use_container_width=True)

    with col_r:
        st.write(f"**Bottom 15 features (candidates to drop)**")
        st.dataframe(
            importance_df.tail(15).iloc[::-1],
            hide_index=True, use_container_width=True,
        )

    st.divider()

    # Per-symbol heatmap of top 20 features
    st.write(f"**Per-symbol importance — top 20 features**")
    if per_symbol_data:
        per_df = pd.DataFrame(per_symbol_data)
        top_20 = importance_df.head(20)["feature"].tolist()
        per_df = per_df[per_df["feature"].isin(top_20)]
        pivoted = per_df.pivot(index="feature", columns="symbol", values="importance")
        pivoted = pivoted.reindex(top_20)
        st.dataframe(pivoted.round(3), use_container_width=True)

    st.divider()

    # Action recommendations
    dead_weight = importance_df[importance_df["avg_importance"] < 0.005]
    st.write(f"**Action items**")
    if len(dead_weight) > 0:
        st.warning(
            f"⚠️ {len(dead_weight)} features have <0.5% average importance "
            f"across all symbols. Consider removing to reduce overfitting risk:"
        )
        st.write(", ".join(dead_weight["feature"].tolist()))
    else:
        st.success("✅ All features pulling weight (no dead weight detected).")

    # Summary stats
    st.caption(
        f"Total features: {len(importance_df)}  ·  "
        f"Symbols: {len(sym_models)}  ·  "
        f"Model version: {data.get('version', 'unknown')}"
    )


def _tab_backtest() -> None:
    import streamlit as st

    st.header("Walk-Forward Backtest")
    st.caption("Train ML on 1yr history, test on next 3 months, slide forward — real Black-Scholes pricing.")

    # Config panel
    with st.expander("Configure Backtest", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            symbols = st.multiselect("Symbols", _UNIVERSE, default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
            capital = st.number_input("Capital ($)", min_value=5000, max_value=500_000, value=50_000, step=5000)
        with col2:
            strategies = st.multiselect("Strategies", _STRATEGIES, default=["bull_call_spread", "iron_condor"])
            min_conf = st.slider("Min ML Confidence", 0.50, 0.90, 0.65, 0.05)

        run = st.button("Run Backtest", type="primary")

    if run:
        if not symbols or not strategies:
            st.error("Select at least 1 symbol and 1 strategy.")
        else:
            with st.spinner(f"Running walk-forward backtest on {', '.join(symbols)}... (~2-5 min)"):
                try:
                    data = _run_walkforward(symbols, strategies, capital, min_conf)
                    with open(_BACKTEST_RESULTS_FILE, "w") as f:
                        json.dump(data, f, default=str)
                    st.success("Backtest complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    results = _load_backtest_results()
    if not results:
        st.info("No backtest results yet. Run a backtest above.")
        return

    st.caption(f"Last run: {results.get('run_at', 'unknown')} | "
               f"Symbols: {', '.join(results.get('symbols', []))} | "
               f"Strategies: {', '.join(results.get('strategies', []))}")

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{results['total_return']:.1%}",
              delta=f"${results['final_capital'] - results['initial_capital']:,.0f}")
    c2.metric("Win Rate", f"{results['win_rate']:.1%}")
    c3.metric("Sharpe", f"{results['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
    c5.metric("Profit Factor", f"{results['profit_factor']:.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", results["total_trades"])
    col2.metric("Windows", results["windows"])
    col3.metric("Consistency", f"{results['consistency']:.0%} windows profitable")

    st.divider()

    # Equity curve
    if results.get("equity_curve"):
        st.subheader("Equity Curve")
        curve_df = pd.DataFrame(results["equity_curve"])
        if not curve_df.empty and "date" in curve_df.columns and "equity" in curve_df.columns:
            curve_df["date"] = pd.to_datetime(curve_df["date"], errors="coerce")
            curve_df = curve_df.dropna(subset=["date"]).sort_values("date")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=curve_df["date"], y=curve_df["equity"],
                mode="lines", fill="tozeroy",
                line=dict(color="royalblue", width=2), name="Equity",
            ))
            fig_eq.add_hline(y=results["initial_capital"], line_dash="dash", line_color="gray",
                             annotation_text="Starting Capital")
            fig_eq.update_layout(yaxis=dict(title="Portfolio Value ($)"), height=350)
            st.plotly_chart(fig_eq, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Strategy breakdown
    strat_data = results.get("strategy_results", {})
    if strat_data:
        with col_l:
            st.subheader("Strategy Breakdown")
            rows = []
            for strat, d in strat_data.items():
                rows.append({
                    "Strategy": strat,
                    "Trades": d["trades"],
                    "Win Rate": f"{d['win_rate']:.0%}",
                    "Total P&L": d["total_pnl"],
                    "Avg P&L": round(d["avg_pnl"], 0),
                })
            strat_df = pd.DataFrame(rows).sort_values("Total P&L", ascending=False)

            def _color_pnl(v):
                return "color: green" if v > 0 else "color: red"

            st.dataframe(
                strat_df.style.applymap(_color_pnl, subset=["Total P&L", "Avg P&L"]),
                use_container_width=True, hide_index=True,
            )

    # Symbol breakdown
    sym_data = results.get("symbol_results", {})
    if sym_data:
        with col_r:
            st.subheader("Symbol Breakdown")
            rows = []
            for sym, d in sym_data.items():
                rows.append({
                    "Symbol": sym,
                    "Return": d["total_return"],
                    "Trades": d["total_trades"],
                    "Win Rate": f"{d['win_rate']:.0%}",
                    "Sharpe": round(d["sharpe_ratio"], 2),
                })
            sym_df = pd.DataFrame(rows).sort_values("Return", ascending=False)

            def _color_ret(v):
                return "color: green" if v > 0 else "color: red"

            st.dataframe(
                sym_df.style.applymap(_color_ret, subset=["Return"]).format({"Return": "{:.1%}"}),
                use_container_width=True, hide_index=True,
            )

    # Trade log
    if results.get("trades"):
        st.divider()
        st.subheader("Trade Log")
        trades_df = pd.DataFrame(results["trades"])
        cols_order = ["symbol", "strategy", "trade_type", "direction", "entry_date",
                      "exit_date", "exit_reason", "pnl", "contracts"]
        cols_order = [c for c in cols_order if c in trades_df.columns]
        trades_df = trades_df[cols_order].copy()
        if "pnl" in trades_df.columns:
            trades_df["pnl"] = trades_df["pnl"].round(2)

        def _row_color(row):
            color = "background-color: #d4edda" if row.get("pnl", 0) > 0 else "background-color: #f8d7da"
            return [color if col == "pnl" else "" for col in row.index]

        st.dataframe(
            trades_df.style.apply(_row_color, axis=1),
            use_container_width=True, height=400, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AIT Trading Dashboard", layout="wide")
    st.title("AIT - Autonomous Intelligent Trading")

    if not DB_PATH.exists():
        st.warning("No trading data found. Start the bot first.")
        return

    conn = _get_conn()

    # --- Sidebar ---
    st.sidebar.header("Controls")

    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

    st.sidebar.divider()
    st.sidebar.header("Date Range Filter")
    default_start = date.today() - timedelta(days=30)
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=date.today())

    # --- Tabs ---
    tabs = st.tabs(
        ["Portfolio Overview", "Trade History", "Analytics",
         "Trade Intelligence", "Self-Learning", "System Health",
         "Feature Importance", "Backtest"]
    )
    tab1, tab2, tab3, tab4, tab5, tab6, tab_fi, tab7 = tabs

    with tab1:
        _tab_portfolio_overview(conn)

    with tab2:
        _tab_trade_history(conn, start_date, end_date)

    with tab3:
        _tab_analytics(conn)

    with tab4:
        _tab_trade_intelligence(conn)

    with tab5:
        _tab_self_learning(conn)

    with tab6:
        _tab_system_health(conn)

    with tab_fi:
        _tab_feature_importance()

    with tab7:
        _tab_backtest()

    conn.close()

    # Auto-refresh: sleep first, then rerun (avoids infinite rerun loop)
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
