"""Tests for DuckDB analytics engine."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from ait.monitoring.duckdb_analytics import DuckDBAnalytics, PerformanceSnapshot


@pytest.fixture
def duck_db(tmp_path: Path) -> DuckDBAnalytics:
    """Create a DuckDB analytics instance in a temp directory."""
    return DuckDBAnalytics(db_path=tmp_path / "test_analytics.duckdb")


@pytest.fixture
def populated_duck(duck_db: DuckDBAnalytics) -> DuckDBAnalytics:
    """DuckDB with sample trades for analytics testing."""
    base_time = datetime.now() - timedelta(days=10)

    trades = [
        {
            "trade_id": "t1", "symbol": "SPY", "strategy": "bull_call_spread",
            "direction": "long", "status": "closed",
            "entry_time": (base_time - timedelta(days=9)).isoformat(),
            "entry_price": 2.50, "quantity": 2, "contract_type": "spread",
            "strike": 450.0, "expiry": "2025-04-15",
            "exit_time": (base_time - timedelta(days=7)).isoformat(),
            "exit_price": 3.50, "realized_pnl": 200.0, "commission": 2.60,
            "ml_confidence": 0.85, "sentiment_score": 0.3,
            "market_regime": "trending_up", "notes": "", "legs": "[]",
            "exit_reason_detailed": "take_profit", "peak_pnl_pct": 0.45,
            "time_to_peak_hours": 12.0, "direction_correct": 1,
        },
        {
            "trade_id": "t2", "symbol": "QQQ", "strategy": "long_call",
            "direction": "long", "status": "closed",
            "entry_time": (base_time - timedelta(days=8)).isoformat(),
            "entry_price": 3.00, "quantity": 1, "contract_type": "call",
            "strike": 380.0, "expiry": "2025-04-15",
            "exit_time": (base_time - timedelta(days=6)).isoformat(),
            "exit_price": 1.50, "realized_pnl": -150.0, "commission": 1.30,
            "ml_confidence": 0.72, "sentiment_score": -0.1,
            "market_regime": "range_bound", "notes": "", "legs": "[]",
            "exit_reason_detailed": "stop_loss", "peak_pnl_pct": 0.10,
            "time_to_peak_hours": 4.0, "direction_correct": 0,
        },
        {
            "trade_id": "t3", "symbol": "AAPL", "strategy": "iron_condor",
            "direction": "short", "status": "closed",
            "entry_time": (base_time - timedelta(days=5)).isoformat(),
            "entry_price": 1.80, "quantity": 3, "contract_type": "iron_condor",
            "strike": None, "expiry": "2025-04-15",
            "exit_time": (base_time - timedelta(days=2)).isoformat(),
            "exit_price": 0.60, "realized_pnl": 360.0, "commission": 5.20,
            "ml_confidence": 0.90, "sentiment_score": 0.0,
            "market_regime": "range_bound", "notes": "", "legs": "[]",
            "exit_reason_detailed": "take_profit", "peak_pnl_pct": 0.70,
            "time_to_peak_hours": 48.0, "direction_correct": 1,
        },
        {
            "trade_id": "t4", "symbol": "SPY", "strategy": "bull_call_spread",
            "direction": "long", "status": "closed",
            "entry_time": (base_time - timedelta(days=3)).isoformat(),
            "entry_price": 2.00, "quantity": 2, "contract_type": "spread",
            "strike": 455.0, "expiry": "2025-04-22",
            "exit_time": (base_time - timedelta(days=1)).isoformat(),
            "exit_price": 2.80, "realized_pnl": 160.0, "commission": 2.60,
            "ml_confidence": 0.78, "sentiment_score": 0.2,
            "market_regime": "trending_up", "notes": "", "legs": "[]",
            "exit_reason_detailed": "take_profit", "peak_pnl_pct": 0.50,
            "time_to_peak_hours": 20.0, "direction_correct": 1,
        },
    ]

    for t in trades:
        duck_db.ingest_trade(t)

    # Add daily stats
    for i in range(10):
        d = (date.today() - timedelta(days=10 - i)).isoformat()
        duck_db.ingest_daily_stats({
            "date": d,
            "trades_taken": 1 if i % 2 == 0 else 0,
            "trades_won": 1 if i % 3 == 0 else 0,
            "trades_lost": 1 if i % 3 == 1 else 0,
            "total_pnl": 50.0 if i % 2 == 0 else -20.0,
            "max_drawdown": 0.01,
            "day_trades_count": 0,
            "circuit_breaker_triggered": False,
        })

    # Add trade contexts
    contexts = [
        {"trade_id": "t1", "entry_direction": "bullish", "entry_confidence": 0.85,
         "entry_regime": "trending_up", "entry_vix": 18.0, "entry_iv_rank": 0.35,
         "entry_sentiment_score": 0.3, "entry_signals": "{}"},
        {"trade_id": "t2", "entry_direction": "bullish", "entry_confidence": 0.72,
         "entry_regime": "range_bound", "entry_vix": 22.0, "entry_iv_rank": 0.55,
         "entry_sentiment_score": -0.1, "entry_signals": "{}"},
        {"trade_id": "t3", "entry_direction": "neutral", "entry_confidence": 0.90,
         "entry_regime": "range_bound", "entry_vix": 20.0, "entry_iv_rank": 0.65,
         "entry_sentiment_score": 0.0, "entry_signals": "{}"},
        {"trade_id": "t4", "entry_direction": "bullish", "entry_confidence": 0.78,
         "entry_regime": "trending_up", "entry_vix": 17.5, "entry_iv_rank": 0.30,
         "entry_sentiment_score": 0.2, "entry_signals": "{}"},
    ]
    for c in contexts:
        duck_db.ingest_trade_context(c)

    return duck_db


class TestDuckDBSchema:
    """Test schema creation and basic operations."""

    def test_init_creates_db(self, duck_db: DuckDBAnalytics) -> None:
        assert duck_db._db_path.exists()

    def test_empty_performance(self, duck_db: DuckDBAnalytics) -> None:
        snap = duck_db.get_performance()
        assert snap.total_trades == 0
        assert snap.total_pnl == 0.0

    def test_trade_count_empty(self, duck_db: DuckDBAnalytics) -> None:
        assert duck_db.get_trade_count() == 0


class TestTradeIngestion:
    """Test writing trades to DuckDB."""

    def test_ingest_and_count(self, duck_db: DuckDBAnalytics) -> None:
        duck_db.ingest_trade({
            "trade_id": "test1", "symbol": "SPY", "strategy": "long_call",
            "direction": "long", "status": "closed",
            "entry_time": datetime.now().isoformat(),
            "entry_price": 2.50, "quantity": 1, "contract_type": "call",
            "strike": 450.0, "expiry": "2025-04-15",
            "exit_time": datetime.now().isoformat(),
            "exit_price": 3.50, "realized_pnl": 100.0, "commission": 1.30,
            "ml_confidence": 0.85, "sentiment_score": 0.2,
            "market_regime": "trending_up", "notes": "", "legs": "[]",
            "exit_reason_detailed": "take_profit", "peak_pnl_pct": 0.40,
            "time_to_peak_hours": 8.0, "direction_correct": 1,
        })
        assert duck_db.get_trade_count() == 1

    def test_upsert_trade(self, duck_db: DuckDBAnalytics) -> None:
        """Inserting same trade_id twice should update, not duplicate."""
        trade = {
            "trade_id": "upsert1", "symbol": "SPY", "strategy": "long_call",
            "direction": "long", "status": "filled",
            "entry_time": datetime.now().isoformat(),
            "entry_price": 2.50, "quantity": 1, "contract_type": "call",
            "strike": 450.0, "expiry": "2025-04-15",
            "exit_time": None, "exit_price": None,
            "realized_pnl": 0.0, "commission": 0.0,
            "ml_confidence": 0.80, "sentiment_score": 0.1,
            "market_regime": "trending_up", "notes": "", "legs": "[]",
            "exit_reason_detailed": "", "peak_pnl_pct": 0.0,
            "time_to_peak_hours": 0.0, "direction_correct": -1,
        }
        duck_db.ingest_trade(trade)

        trade["status"] = "closed"
        trade["realized_pnl"] = 150.0
        duck_db.ingest_trade(trade)

        assert duck_db.get_trade_count() == 1

    def test_ingest_feature_snapshot(self, duck_db: DuckDBAnalytics) -> None:
        features = {"rsi_14": 55.0, "macd": 0.5, "bb_position": 0.7}
        duck_db.ingest_feature_snapshot("t1", features)
        # Verify by querying directly
        with duck_db._get_conn() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM feature_snapshots WHERE trade_id = 't1'"
            ).fetchone()
        assert rows[0] == 3


class TestPerformanceQueries:
    """Test analytics queries on populated data."""

    def test_performance_metrics(self, populated_duck: DuckDBAnalytics) -> None:
        snap = populated_duck.get_performance(lookback_days=30)
        assert snap.total_trades == 4
        assert snap.total_pnl == pytest.approx(570.0)  # 200 - 150 + 360 + 160
        assert snap.win_rate == pytest.approx(0.75)  # 3 wins / 4 trades
        assert snap.largest_win == pytest.approx(360.0)
        assert snap.largest_loss == pytest.approx(-150.0)
        assert snap.profit_factor > 1.0

    def test_daily_pnl(self, populated_duck: DuckDBAnalytics) -> None:
        daily = populated_duck.get_daily_pnl(lookback_days=30)
        assert len(daily) == 10
        # Check cumulative is running sum
        assert daily[0]["cumulative_pnl"] == daily[0]["daily_pnl"]
        running = 0.0
        for d in daily:
            running += d["daily_pnl"]
            assert d["cumulative_pnl"] == pytest.approx(running)

    def test_strategy_breakdown(self, populated_duck: DuckDBAnalytics) -> None:
        breakdown = populated_duck.get_strategy_breakdown(lookback_days=30)
        assert len(breakdown) == 3
        strategies = {b["strategy"] for b in breakdown}
        assert strategies == {"bull_call_spread", "long_call", "iron_condor"}

    def test_symbol_breakdown(self, populated_duck: DuckDBAnalytics) -> None:
        breakdown = populated_duck.get_symbol_breakdown(lookback_days=30)
        assert len(breakdown) == 3
        symbols = {b["symbol"] for b in breakdown}
        assert symbols == {"SPY", "QQQ", "AAPL"}

    def test_regime_breakdown(self, populated_duck: DuckDBAnalytics) -> None:
        breakdown = populated_duck.get_regime_breakdown(lookback_days=30)
        assert len(breakdown) >= 1
        regimes = {b["regime"] for b in breakdown}
        assert "trending_up" in regimes

    def test_strategy_regime_matrix(self, populated_duck: DuckDBAnalytics) -> None:
        matrix = populated_duck.get_strategy_regime_matrix(lookback_days=30)
        # Should have at least bull_call_spread x trending_up (2 trades)
        assert len(matrix) >= 1

    def test_confidence_band_analysis(self, populated_duck: DuckDBAnalytics) -> None:
        bands = populated_duck.get_confidence_band_analysis(lookback_days=30)
        assert len(bands) >= 1
        # All trades have confidence in 0.70-0.90 range
        for b in bands:
            assert b["trades"] >= 1

    def test_exit_efficiency(self, populated_duck: DuckDBAnalytics) -> None:
        eff = populated_duck.get_exit_efficiency(lookback_days=30)
        # 3 trades have peak_pnl_pct > 0
        assert len(eff) >= 1


class TestSQLiteSync:
    """Test importing data from SQLite into DuckDB."""

    def test_sync_from_nonexistent_sqlite(self, duck_db: DuckDBAnalytics) -> None:
        count = duck_db.sync_from_sqlite(Path("nonexistent.db"))
        assert count == 0

    def test_sync_from_sqlite(self, duck_db: DuckDBAnalytics, tmp_path: Path) -> None:
        """Create a temp SQLite DB with trades and sync to DuckDB."""
        import sqlite3

        sq_path = tmp_path / "state.db"
        with sqlite3.connect(sq_path) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT, strategy TEXT, direction TEXT, status TEXT,
                    entry_time TEXT, entry_price REAL, quantity INTEGER,
                    contract_type TEXT, strike REAL, expiry TEXT,
                    exit_time TEXT, exit_price REAL,
                    realized_pnl REAL DEFAULT 0, commission REAL DEFAULT 0,
                    ml_confidence REAL DEFAULT 0, sentiment_score REAL DEFAULT 0,
                    market_regime TEXT DEFAULT '', notes TEXT DEFAULT '',
                    legs TEXT DEFAULT '[]',
                    exit_reason_detailed TEXT DEFAULT '',
                    peak_pnl_pct REAL DEFAULT 0,
                    time_to_peak_hours REAL DEFAULT 0,
                    direction_correct INTEGER DEFAULT -1
                )
            """)
            conn.execute("""
                CREATE TABLE daily_stats (
                    date TEXT PRIMARY KEY,
                    trades_taken INTEGER DEFAULT 0, trades_won INTEGER DEFAULT 0,
                    trades_lost INTEGER DEFAULT 0, total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0, day_trades_count INTEGER DEFAULT 0,
                    circuit_breaker_triggered BOOLEAN DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE trade_context (
                    trade_id TEXT PRIMARY KEY,
                    entry_direction TEXT, entry_confidence REAL DEFAULT 0,
                    entry_regime TEXT DEFAULT '', entry_vix REAL DEFAULT 0,
                    entry_iv_rank REAL DEFAULT 0,
                    entry_sentiment_score REAL DEFAULT 0,
                    entry_signals TEXT DEFAULT '{}'
                )
            """)
            conn.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("sync1", "SPY", "long_call", "long", "closed",
                 datetime.now().isoformat(), 2.5, 1, "call", 450.0, "2025-04-15",
                 datetime.now().isoformat(), 3.5, 100.0, 1.3, 0.85, 0.2,
                 "trending_up", "", "[]", "take_profit", 0.4, 8.0, 1),
            )
            conn.execute(
                "INSERT INTO daily_stats VALUES (?,?,?,?,?,?,?,?)",
                (date.today().isoformat(), 1, 1, 0, 100.0, 0.01, 0, 0),
            )

        count = duck_db.sync_from_sqlite(sq_path)
        assert count == 1
        assert duck_db.get_trade_count() == 1
