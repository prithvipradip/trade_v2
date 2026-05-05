"""SQLite store for IB-sourced news and analyst recommendations.

Two tables:
- news:                    IB news headlines with pre-computed sentiment scores.
- analyst_recommendations: Structured Briefing.com analyst rating actions.

Both tables use INSERT OR IGNORE — records are immutable facts and are never
overwritten after first insertion.  Re-fetching the same date window on a later
trigger will silently skip rows that already exist (idempotent by design).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("data.fundamentals_db")

DEFAULT_DB_PATH = Path("data/fundamentals.db")


class FundamentalsStore:
    """SQLite CRUD layer for news and analyst recommendations."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    article_id   TEXT PRIMARY KEY,
                    symbol       TEXT NOT NULL,
                    provider     TEXT NOT NULL,
                    headline     TEXT NOT NULL,
                    url          TEXT DEFAULT '',
                    published_at TEXT NOT NULL,
                    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
                    sentiment    REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_symbol_date
                    ON news(symbol, published_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_recommendations (
                    id           TEXT PRIMARY KEY,
                    symbol       TEXT NOT NULL,
                    issued_at    TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
                    firm         TEXT DEFAULT '',
                    action       TEXT DEFAULT '',
                    rating       TEXT DEFAULT '',
                    price_target REAL DEFAULT 0.0,
                    prior_target REAL DEFAULT 0.0,
                    raw_text     TEXT DEFAULT '',
                    UNIQUE(symbol, issued_at, firm)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyst_symbol_date
                    ON analyst_recommendations(symbol, issued_at DESC)
            """)
        log.info("fundamentals_db_initialized", path=str(self._db_path))

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def insert_news(self, articles: list[dict]) -> int:
        """INSERT OR IGNORE news rows.  Returns count of newly inserted rows."""
        if not articles:
            return 0
        inserted = 0
        with self._connect() as conn:
            for a in articles:
                cur = conn.execute("""
                    INSERT OR IGNORE INTO news
                        (article_id, symbol, provider, headline, url, published_at, sentiment)
                    VALUES
                        (:article_id, :symbol, :provider, :headline, :url, :published_at, :sentiment)
                """, {
                    "article_id": a["article_id"],
                    "symbol":     a["symbol"],
                    "provider":   a.get("provider", ""),
                    "headline":   a["headline"],
                    "url":        a.get("url", ""),
                    "published_at": a["published_at"],
                    "sentiment":  a.get("sentiment", 0.0),
                })
                inserted += cur.rowcount
        log.debug("news_inserted", count=inserted, total=len(articles))
        return inserted

    def get_recent_news(self, symbol: str, hours: int = 24) -> list[dict]:
        """Return news articles for *symbol* published in the last *hours* hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT article_id, symbol, provider, headline, url,
                       published_at, fetched_at, sentiment
                FROM news
                WHERE symbol = ? AND published_at >= ?
                ORDER BY published_at DESC
            """, [symbol, cutoff]).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Analyst recommendations
    # ------------------------------------------------------------------

    def insert_analyst_rec(self, recs: list[dict]) -> int:
        """INSERT OR IGNORE analyst rows.  Returns count of newly inserted rows."""
        if not recs:
            return 0
        inserted = 0
        with self._connect() as conn:
            for r in recs:
                cur = conn.execute("""
                    INSERT OR IGNORE INTO analyst_recommendations
                        (id, symbol, issued_at, published_at, firm, action,
                         rating, price_target, prior_target, raw_text)
                    VALUES
                        (:id, :symbol, :issued_at, :published_at, :firm, :action,
                         :rating, :price_target, :prior_target, :raw_text)
                """, {
                    "id":           r["id"],
                    "symbol":       r["symbol"],
                    "issued_at":    r["issued_at"],
                    "published_at": r["published_at"],
                    "firm":         r.get("firm", ""),
                    "action":       r.get("action", ""),
                    "rating":       r.get("rating", ""),
                    "price_target": r.get("price_target", 0.0),
                    "prior_target": r.get("prior_target", 0.0),
                    "raw_text":     r.get("raw_text", ""),
                })
                inserted += cur.rowcount
        log.debug("analyst_recs_inserted", count=inserted, total=len(recs))
        return inserted

    def get_analyst_recs(self, symbol: str, days: int = 30) -> list[dict]:
        """Return analyst actions for *symbol* issued in the last *days* days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, symbol, issued_at, published_at, fetched_at,
                       firm, action, rating, price_target, prior_target, raw_text
                FROM analyst_recommendations
                WHERE symbol = ? AND issued_at >= ?
                ORDER BY issued_at DESC
            """, [symbol, cutoff]).fetchall()
        return [dict(r) for r in rows]
