"""Streamlit dashboard for monitoring the trading bot.

Run separately: streamlit run src/ait/dashboard/app.py
Shows: positions, P&L, signals, risk status, trade history.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DB_PATH = Path("data/ait_state.db")


def main():
    import streamlit as st

    st.set_page_config(page_title="AIT Trading Dashboard", layout="wide")
    st.title("AIT - Autonomous Intelligent Trading")

    if not DB_PATH.exists():
        st.warning("No trading data found. Start the bot first.")
        return

    conn = sqlite3.connect(DB_PATH)

    # --- Sidebar ---
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.rerun()  # Streamlit will rerun on interval

    # --- Portfolio Summary ---
    col1, col2, col3, col4 = st.columns(4)

    # Today's stats
    today = date.today().isoformat()
    stats = pd.read_sql_query(
        "SELECT * FROM daily_stats WHERE date = ?", conn, params=(today,)
    )

    if not stats.empty:
        s = stats.iloc[0]
        col1.metric("Today's P&L", f"${s.get('total_pnl', 0):.2f}")
        col2.metric("Trades Today", int(s.get("trades_taken", 0)))
        col3.metric("Win Rate", f"{s.get('trades_won', 0)}/{s.get('trades_taken', 1)}")
        col4.metric("Day Trades Used", f"{int(s.get('day_trades_count', 0))}/3")
    else:
        col1.metric("Today's P&L", "$0.00")
        col2.metric("Trades Today", 0)
        col3.metric("Win Rate", "N/A")
        col4.metric("Day Trades Used", "0/3")

    st.divider()

    # --- Open Positions ---
    st.subheader("Open Positions")
    open_positions = pd.read_sql_query(
        "SELECT symbol, strategy, entry_price, quantity, entry_time, ml_confidence "
        "FROM trades WHERE status IN ('filled', 'partial') "
        "ORDER BY entry_time DESC",
        conn,
    )

    if not open_positions.empty:
        st.dataframe(open_positions, use_container_width=True)
    else:
        st.info("No open positions")

    # --- Trade History ---
    st.subheader("Recent Trades")
    recent = pd.read_sql_query(
        "SELECT trade_id, symbol, strategy, direction, status, "
        "entry_price, exit_price, realized_pnl, entry_time, exit_time "
        "FROM trades ORDER BY entry_time DESC LIMIT 50",
        conn,
    )

    if not recent.empty:
        st.dataframe(recent, use_container_width=True)

    # --- P&L Chart ---
    st.subheader("Daily P&L")
    daily = pd.read_sql_query(
        "SELECT date, total_pnl, trades_taken FROM daily_stats ORDER BY date",
        conn,
    )

    if not daily.empty:
        daily["cumulative_pnl"] = daily["total_pnl"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["total_pnl"], name="Daily P&L"))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cumulative_pnl"],
            name="Cumulative", yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Cumulative P&L"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Strategy Performance ---
    st.subheader("Strategy Performance")
    strategy_perf = pd.read_sql_query(
        "SELECT strategy, COUNT(*) as trades, "
        "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "SUM(realized_pnl) as total_pnl, "
        "AVG(realized_pnl) as avg_pnl "
        "FROM trades WHERE status = 'closed' "
        "GROUP BY strategy ORDER BY total_pnl DESC",
        conn,
    )

    if not strategy_perf.empty:
        st.dataframe(strategy_perf, use_container_width=True)

    conn.close()


if __name__ == "__main__":
    main()
