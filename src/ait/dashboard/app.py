"""Streamlit dashboard for monitoring the trading bot.

Run separately: streamlit run src/ait/dashboard/app.py
Tabs: Portfolio Overview, Trade History, Analytics, Self-Learning, System Health.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DB_PATH = Path("data/ait_state.db")


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
    row = _safe_fetchone(conn, "SELECT value FROM state WHERE key = ?", (key,))
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

    # Open Positions
    st.subheader("Open Positions")
    open_positions = _safe_query(
        conn,
        "SELECT symbol, strategy, direction, entry_price, quantity, entry_time, "
        "ml_confidence FROM trades WHERE status IN ('filled', 'partial') "
        "ORDER BY entry_time DESC",
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

    # Recent trades
    st.subheader("Recent Trades")
    recent = _safe_query(
        conn,
        "SELECT trade_id, symbol, strategy, direction, status, "
        "entry_price, exit_price, realized_pnl, entry_time, exit_time "
        "FROM trades WHERE date(entry_time) BETWEEN ? AND ? "
        "ORDER BY entry_time DESC LIMIT 100",
        (start_iso, end_iso),
    )
    if not recent.empty:
        st.dataframe(recent, use_container_width=True)
    else:
        st.info("No trades in the selected date range")

    st.divider()

    # Strategy performance
    st.subheader("Strategy Performance")
    strategy_perf = _safe_query(
        conn,
        "SELECT strategy, COUNT(*) as trades, "
        "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct, "
        "ROUND(SUM(realized_pnl), 2) as total_pnl, "
        "ROUND(AVG(realized_pnl), 2) as avg_pnl "
        "FROM trades WHERE status = 'closed' AND date(entry_time) BETWEEN ? AND ? "
        "GROUP BY strategy ORDER BY total_pnl DESC",
        (start_iso, end_iso),
    )
    if not strategy_perf.empty:
        st.dataframe(strategy_perf, use_container_width=True)
    else:
        st.info("No closed trades for strategy breakdown")

    st.divider()

    # Symbol performance
    st.subheader("Symbol Performance")
    symbol_perf = _safe_query(
        conn,
        "SELECT symbol, COUNT(*) as trades, "
        "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate_pct, "
        "ROUND(SUM(realized_pnl), 2) as total_pnl, "
        "ROUND(AVG(realized_pnl), 2) as avg_pnl "
        "FROM trades WHERE status = 'closed' AND date(entry_time) BETWEEN ? AND ? "
        "GROUP BY symbol ORDER BY total_pnl DESC",
        (start_iso, end_iso),
    )
    if not symbol_perf.empty:
        st.dataframe(symbol_perf, use_container_width=True)
    else:
        st.info("No closed trades for symbol breakdown")


def _tab_analytics(conn: sqlite3.Connection) -> None:
    import streamlit as st

    # Gather closed trades for analytics
    trades = _safe_fetchall(
        conn,
        "SELECT realized_pnl, entry_time, exit_time FROM trades "
        "WHERE status = 'closed' ORDER BY entry_time",
    )

    pnls = [t["realized_pnl"] for t in trades if t.get("realized_pnl") is not None]

    # Compute metrics
    sharpe = sortino = max_dd = profit_factor = avg_hold = 0.0
    current_streak = max_win_streak = max_loss_streak = 0

    if len(pnls) > 1:
        import statistics

        import math

        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)

        if std_pnl > 0:
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)

        downside = [p for p in pnls if p < 0]
        if len(downside) > 1:
            ds_std = statistics.stdev(downside)
            if ds_std > 0:
                sortino = (mean_pnl / ds_std) * math.sqrt(252)
        elif len(downside) == 1:
            ds_std = abs(downside[0])
            if ds_std > 0:
                sortino = (mean_pnl / ds_std) * math.sqrt(252)

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
        "SELECT key, value FROM state WHERE key LIKE 'learning_%' ORDER BY key",
    )
    if not learning_keys.empty:
        st.dataframe(learning_keys, use_container_width=True, hide_index=True)
    else:
        st.info("No learning state entries found")


def _tab_system_health(conn: sqlite3.Connection) -> None:
    import streamlit as st

    # Watchdog / component status
    st.subheader("Component Status")
    watchdog_keys = _safe_query(
        conn,
        "SELECT key, value FROM state WHERE key LIKE 'watchdog_%' ORDER BY key",
    )
    if not watchdog_keys.empty:
        for _, row in watchdog_keys.iterrows():
            key_name = row["key"].replace("watchdog_", "").replace("_", " ").title()
            val = row["value"]
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
            except (json.JSONDecodeError, TypeError):
                parsed = val

            if isinstance(parsed, dict):
                status = parsed.get("status", "unknown")
                icon = "🟢" if status in ("ok", "running", "healthy") else "🔴"
                last_seen = parsed.get("last_heartbeat", parsed.get("last_seen", ""))
                st.write(f"{icon} **{key_name}** — {status}")
                if last_seen:
                    st.caption(f"Last seen: {last_seen}")
            else:
                st.write(f"**{key_name}**: {parsed}")
    else:
        st.info("No watchdog data available")

    st.divider()

    # Memory usage
    st.subheader("Memory Usage")
    mem_val = _get_state_value(conn, "system_memory_usage")
    if mem_val:
        try:
            mem = json.loads(mem_val)
            if isinstance(mem, dict):
                mc1, mc2 = st.columns(2)
                mc1.metric("RSS (MB)", f"{mem.get('rss_mb', 'N/A')}")
                mc2.metric("VMS (MB)", f"{mem.get('vms_mb', 'N/A')}")
            else:
                st.write(f"Memory: {mem}")
        except (json.JSONDecodeError, TypeError):
            st.write(f"Memory: {mem_val}")
    else:
        st.info("No memory usage data available")

    st.divider()

    # Error log
    st.subheader("Recent Errors")
    errors_raw = _get_state_value(conn, "error_log")
    if errors_raw:
        try:
            errors = json.loads(errors_raw)
            if isinstance(errors, list):
                last_20 = errors[-20:]
                last_20.reverse()
                err_df = pd.DataFrame(last_20)
                st.dataframe(err_df, use_container_width=True, hide_index=True)
            else:
                st.code(str(errors))
        except (json.JSONDecodeError, TypeError):
            st.code(errors_raw)
    else:
        # Try watchdog_errors key as fallback
        errors_raw2 = _get_state_value(conn, "watchdog_errors")
        if errors_raw2:
            try:
                errors = json.loads(errors_raw2)
                if isinstance(errors, list):
                    last_20 = errors[-20:]
                    last_20.reverse()
                    err_df = pd.DataFrame(last_20)
                    st.dataframe(err_df, use_container_width=True, hide_index=True)
                else:
                    st.code(str(errors))
            except (json.JSONDecodeError, TypeError):
                st.code(errors_raw2)
        else:
            st.success("No errors logged")

    st.divider()

    # Model version info
    st.subheader("Model Info")
    model_val = _get_state_value(conn, "model_version")
    if model_val:
        try:
            model = json.loads(model_val)
            if isinstance(model, dict):
                for k, v in model.items():
                    st.write(f"**{k}**: {v}")
            else:
                st.write(f"Model version: {model}")
        except (json.JSONDecodeError, TypeError):
            st.write(f"Model version: {model_val}")
    else:
        st.info("No model version info available")


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Portfolio Overview", "Trade History", "Analytics", "Self-Learning", "System Health"]
    )

    with tab1:
        _tab_portfolio_overview(conn)

    with tab2:
        _tab_trade_history(conn, start_date, end_date)

    with tab3:
        _tab_analytics(conn)

    with tab4:
        _tab_self_learning(conn)

    with tab5:
        _tab_system_health(conn)

    conn.close()

    # Auto-refresh: sleep first, then rerun (avoids infinite rerun loop)
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
