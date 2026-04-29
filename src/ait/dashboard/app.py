"""Streamlit dashboard for monitoring the trading bot.

Run separately: streamlit run src/ait/dashboard/app.py
Tabs: Portfolio Overview, Trade History, Analytics, Self-Learning, System Health.

Uses DuckDB for analytics-heavy queries (trade history, strategy breakdown,
regime analysis) and SQLite for live operational state (open positions, KV store).
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
        """
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
          AND date(entry_time) BETWEEN ? AND ?
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
        """
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
          AND date(entry_time) BETWEEN ? AND ?
        GROUP BY day_of_week
        ORDER BY day_of_week
        """,
        (start_iso, end_iso),
    )
    if not dow_perf.empty:
        st.dataframe(dow_perf, use_container_width=True)
    else:
        st.info("No day-of-week data yet")


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

    # Get live stats — last 30 days of closed trades
    end = date.today()
    start = end - timedelta(days=30)
    live_trades = _safe_fetchall(
        conn,
        "SELECT realized_pnl, entry_time FROM trades "
        "WHERE status = 'closed' AND date(entry_time) BETWEEN ? AND ?",
        (start.isoformat(), end.isoformat()),
    )
    live_pnls = [t["realized_pnl"] for t in live_trades if t.get("realized_pnl")]
    live_total_pnl = sum(live_pnls)
    live_trade_count = len(live_trades)
    live_wins = sum(1 for p in live_pnls if p > 0)
    live_win_rate = live_wins / live_trade_count if live_trade_count > 0 else None

    # Get account value for return calc
    nlv_row = _safe_fetchall(
        conn,
        "SELECT value FROM bot_state WHERE key = 'account_value' LIMIT 1",
    )
    nlv = float(nlv_row[0]["value"]) if nlv_row else 250000.0
    live_return_30d = (live_total_pnl / nlv) if nlv > 0 else 0

    # Annualize: backtest is multi-year, live is 30 days
    bt_period_years = 4.0  # rough — backtest covers ~4 years of test data
    bt_annualized = (1 + bt_return) ** (1 / bt_period_years) - 1 if bt_return is not None else None
    live_annualized = ((1 + live_return_30d) ** 12) - 1 if live_return_30d else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Backtest Return (annualized)",
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
        "SELECT key, value FROM state WHERE key LIKE 'learning_%' ORDER BY key",
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

    # Trades with journaling data
    exit_data = _safe_query(
        conn,
        "SELECT symbol, strategy, exit_reason_detailed, peak_pnl_pct, "
        "realized_pnl, direction_correct "
        "FROM trades WHERE status = 'closed' AND exit_reason_detailed != '' "
        "ORDER BY exit_time DESC LIMIT 50",
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
        has_peak = exit_data[exit_data["peak_pnl_pct"] > 0].copy()
        if not has_peak.empty:
            has_peak["capture_pct"] = has_peak.apply(
                lambda r: (r["realized_pnl"] / (r["peak_pnl_pct"] * 100)) * 100
                if r["peak_pnl_pct"] > 0 else 0, axis=1
            )
            avg_capture = has_peak["capture_pct"].mean()
            avg_peak = has_peak["peak_pnl_pct"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Avg Peak P&L %", f"{avg_peak:.1%}")
            c2.metric("Avg Capture Efficiency", f"{avg_capture:.0f}%")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=has_peak["symbol"],
                y=has_peak["peak_pnl_pct"] * 100,
                name="Peak P&L %",
                marker_color="lightblue",
            ))
            fig.add_trace(go.Bar(
                x=has_peak["symbol"],
                y=has_peak["realized_pnl"],
                name="Realized P&L $",
                marker_color=["green" if p > 0 else "red" for p in has_peak["realized_pnl"]],
            ))
            fig.update_layout(barmode="group", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades with peak P&L data yet")

        st.divider()

        # Direction accuracy
        st.write("**ML Direction Accuracy**")
        known = exit_data[exit_data["direction_correct"].isin([0, 1])]
        if not known.empty:
            correct = known["direction_correct"].sum()
            total = len(known)
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Direction Correct", f"{correct}/{total}")
            dc2.metric("Direction Accuracy", f"{correct / total * 100:.1f}%")

            # Direction right but lost money = exit problem
            right_but_lost = known[(known["direction_correct"] == 1) & (known["realized_pnl"] <= 0)]
            dc3.metric("Right Direction, Lost $", f"{len(right_but_lost)}/{correct}")
            if len(right_but_lost) > 0:
                st.warning(
                    f"{len(right_but_lost)} trades had correct direction but lost money — "
                    "exit management is the bottleneck, not ML predictions."
                )
        else:
            st.info("No direction accuracy data yet (needs trade context)")
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
    meta_stats = _get_state_json(conn, "meta_label_stats")
    if meta_stats and isinstance(meta_stats, dict):
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
        st.info("Meta-labeler not yet trained (needs 30+ closed trades with context)")


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
        step_days=21,
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Portfolio Overview", "Trade History", "Analytics",
         "Trade Intelligence", "Self-Learning", "System Health", "Backtest"]
    )

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

    with tab7:
        _tab_backtest()

    conn.close()

    # Auto-refresh: sleep first, then rerun (avoids infinite rerun loop)
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
