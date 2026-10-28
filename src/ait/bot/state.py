"""Persistent state management using SQLite + DuckDB analytics.

SQLite: operational state (open positions, bot KV store, pending trades).
DuckDB: analytics store (closed trades, daily stats, trade context).
Dual-write on close_trade() and update_daily_stats().
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from ait.utils.logging import get_logger
from ait.utils.time import now_et

log = get_logger("bot.state")


class PartialExitsCorruptError(ValueError):
    """W104: ``open_positions.partial_exits`` is not a JSON list of slices."""

    def __init__(self, trade_id: str, raw: str):
        super().__init__(f"partial_exits unreadable for {trade_id}")
        self.trade_id = trade_id
        self.raw = raw

DB_PATH = Path("data/ait_state.db")


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSING = "closing"
    CLOSED = "closed"


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecord:
    """A complete trade record from entry to exit."""

    trade_id: str
    symbol: str
    strategy: str
    direction: TradeDirection
    status: TradeStatus

    # Entry
    entry_time: str  # ISO format
    entry_price: float
    quantity: int
    contract_type: str  # "stock", "call", "put", "spread", "iron_condor"
    strike: float | None = None
    expiry: str | None = None

    # Exit
    exit_time: str | None = None
    exit_price: float | None = None

    # P&L
    realized_pnl: float = 0.0
    commission: float = 0.0

    # Context
    ml_confidence: float = 0.0
    sentiment_score: float = 0.0
    market_regime: str = ""
    notes: str = ""

    # Multi-leg details (JSON for spreads/condors)
    legs: str = "[]"

    # Journaling (populated on close)
    exit_reason_detailed: str = ""
    peak_pnl_pct: float = 0.0
    time_to_peak_hours: float = 0.0
    direction_correct: int = -1  # -1 = unknown, 0 = wrong, 1 = correct
    capital_at_risk: float = 0.0  # R7: retained max_loss at entry (verdict math)
    # W50 (money-flow-05 / trade-life-partial-slice-pnl-erased): archived
    # open_positions.partial_exits JSON, folded into realized_pnl and copied
    # here by close_trade before the open_positions row is deleted — the
    # round-trip's slice-by-slice history otherwise vanished with the DELETE.
    partial_exits: str = "[]"
    # W51 (model-vs-reality-margin-model-unimplemented): the broker's own
    # reason an order died, stamped on a CANCELLED entry when IBKR's
    # errorEvent captured one (see ibkr_client.order_error_codes) — without
    # this a margin reject (201) was indistinguishable from a routine
    # timeout/cancel. 0 / "" means no broker error was captured for this
    # trade's order, not "no rejection happened" (status still says that).
    reject_code: int = 0
    reject_text: str = ""
    # W53 (roll-mechanics groundwork): links a roll's new trade row back to
    # the trade it replaced. Nullable, no default -- unset for every trade
    # that isn't the product of a roll (i.e. every trade today). Not yet
    # written by any code path; the fill handler that would set it is a
    # follow-up wave, gated on the atomicity probe.
    rolled_from_trade_id: str | None = None


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    day_trades_count: int = 0  # For PDT tracking
    circuit_breaker_triggered: bool = False


class StateManager:
    """Manages persistent bot state in SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()
        self._duck = self._init_duckdb()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    contract_type TEXT NOT NULL,
                    strike REAL,
                    expiry TEXT,
                    exit_time TEXT,
                    exit_price REAL,
                    realized_pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    ml_confidence REAL DEFAULT 0,
                    sentiment_score REAL DEFAULT 0,
                    market_regime TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    legs TEXT DEFAULT '[]',
                    exit_reason_detailed TEXT DEFAULT '',
                    peak_pnl_pct REAL DEFAULT 0,
                    time_to_peak_hours REAL DEFAULT 0,
                    direction_correct INTEGER DEFAULT -1
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trades_taken INTEGER DEFAULT 0,
                    trades_won INTEGER DEFAULT 0,
                    trades_lost INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    day_trades_count INTEGER DEFAULT 0,
                    circuit_breaker_triggered BOOLEAN DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    position_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    contract_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    current_price REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    legs TEXT DEFAULT '[]',
                    high_water_mark REAL DEFAULT 0,
                    partial_exits TEXT DEFAULT '[]',
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                CREATE TABLE IF NOT EXISTS trade_context (
                    trade_id TEXT PRIMARY KEY,
                    entry_direction TEXT NOT NULL,
                    entry_confidence REAL DEFAULT 0,
                    entry_regime TEXT DEFAULT '',
                    entry_vix REAL DEFAULT 0,
                    entry_iv_rank REAL DEFAULT 0,
                    entry_sentiment_score REAL DEFAULT 0,
                    entry_signals TEXT DEFAULT '{}',
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_date
                ON trades(entry_time);

                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON trades(status);

                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol);
            """)
            # Lightweight migrations for pre-existing DBs (SQLite has no
            # ADD COLUMN IF NOT EXISTS; duplicate-add raises OperationalError).
            for ddl in (
                "ALTER TABLE open_positions ADD COLUMN pnl_pct REAL DEFAULT 0",
                "ALTER TABLE open_positions ADD COLUMN mark_time TEXT DEFAULT ''",
                # GOV-2 (governance audit): model lineage — without this you
                # cannot attribute a live trade to the model version that
                # produced its prediction.
                "ALTER TABLE trade_context ADD COLUMN model_version TEXT DEFAULT ''",
                # R7 (gap audit): capital-at-risk retained on the trade row —
                # it used to live only in bot_state KV and was deleted on
                # close, making PF-per-unit-risk and DD-vs-deployed-risk
                # unreconstructable for the go-live verdict.
                "ALTER TABLE trades ADD COLUMN capital_at_risk REAL DEFAULT 0",
                # W10 (R24: the pre-registered observe-mode verdict is
                # unscoreable). The entry-time RANGE PROBABILITY had no home:
                # it was written into entry_confidence when the gates were
                # armed and silently replaced by the observe-mode neutral
                # baseline (0.60) on 2026-09-01, so the column means different
                # things in different epochs and the rule "compare closes
                # where p_in_range was above vs below 0.65" cannot be run.
                # These give it a durable home of its own, plus the spec it
                # was produced under (a probability is meaningless without its
                # horizon/threshold) and which gate mode was in force.
                "ALTER TABLE trade_context ADD COLUMN range_probability REAL",
                "ALTER TABLE trade_context ADD COLUMN range_threshold_pct REAL",
                "ALTER TABLE trade_context ADD COLUMN range_horizon_days INTEGER",
                "ALTER TABLE trade_context ADD COLUMN range_model_version TEXT",
                "ALTER TABLE trade_context ADD COLUMN entry_gates_mode TEXT",
                "ALTER TABLE trade_context ADD COLUMN directional_confidence REAL",
                # R7: per-fill execution ledger (real commissions + fill
                # quality). trades.commission was written by ZERO callers —
                # the PF verdict was being computed on guessed costs.
                """CREATE TABLE IF NOT EXISTS executions (
                    exec_id TEXT PRIMARY KEY,
                    order_id INTEGER,
                    perm_id INTEGER,
                    trade_id TEXT,
                    symbol TEXT,
                    con_id INTEGER,
                    side TEXT,
                    shares REAL,
                    price REAL,
                    exec_time TEXT,
                    commission REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    signal_price REAL DEFAULT 0,
                    live_mid REAL DEFAULT 0,
                    nbbo_spread REAL DEFAULT 0
                )""",
                # W105: who placed the order. The operator also trades by hand in
                # this account, and the probes use their own client ids; their fills
                # legitimately have no bot trade. Without these a trade-less row could
                # not be told from a bot fill we failed to attribute.
                "ALTER TABLE executions ADD COLUMN client_id INTEGER",
                "ALTER TABLE executions ADD COLUMN order_ref TEXT",
                "ALTER TABLE executions ADD COLUMN origin TEXT",
                # W1 (R23 concurrency-2 / trade-life-close-booking-two-owners):
                # transactional exit outbox. close_trade enqueues a row in the
                # SAME transaction as the close; every booking path CLAIMS the
                # row (single DELETE, sqlite-serialized => exactly-once) before
                # touching daily stats / breaker / PDT / Thompson. Closes booked
                # by the reconciler while the bot was down are drained by the
                # orchestrator's cycle instead of silently bypassing the breaker.
                """CREATE TABLE IF NOT EXISTS exit_outbox (
                    trade_id TEXT PRIMARY KEY,
                    realized_pnl REAL NOT NULL,
                    exit_reason TEXT DEFAULT '',
                    exit_time TEXT NOT NULL
                )""",
                # W14 (R24 concurrency-01): MUST come after the CREATE above —
                # this tuple executes in order and a duplicate/missing-table
                # ALTER is swallowed, so an ALTER placed earlier failed
                # silently and left the column absent on fresh databases.
                # The outbox was AT-MOST-once: the claim was a committed
                # DELETE performed BEFORE any booking write, so a crash in
                # that window destroyed the only record of the obligation
                # while nothing had been booked. Claiming now MARKS the row;
                # it is deleted only once the booking lands.
                "ALTER TABLE exit_outbox ADD COLUMN claimed_at TEXT",
                # W32 (R26 two-phase-claim-double-books): W14's claim-as-mark made
                # the booking AT-LEAST-once — a fault after some of the booking
                # writes left the claim to go stale and the drain re-applied the
                # whole close (one -272.29 stop read as -544.58 / 2 losses).
                # core_booked marks that the MONEY writes (daily_stats + the
                # breaker blob) landed, in the same transaction that wrote them;
                # steps_done lists the other side effects as each one lands.
                "ALTER TABLE exit_outbox ADD COLUMN core_booked INTEGER DEFAULT 0",
                "ALTER TABLE exit_outbox ADD COLUMN steps_done TEXT DEFAULT ''",
                # W50 (money-flow-05 / trade-life-partial-slice-pnl-erased):
                # archive of open_positions.partial_exits, copied in by
                # close_trade before that row is DELETEd. See TradeRecord's
                # own field comment for why this exists.
                "ALTER TABLE trades ADD COLUMN partial_exits TEXT DEFAULT '[]'",
                # W51 (model-vs-reality-margin-model-unimplemented): the
                # broker's own reason an entry order was cancelled, when
                # errorEvent captured one. See TradeRecord's own field
                # comment.
                "ALTER TABLE trades ADD COLUMN reject_code INTEGER DEFAULT 0",
                "ALTER TABLE trades ADD COLUMN reject_text TEXT DEFAULT ''",
                # W53 (roll-mechanics groundwork): see TradeRecord's own
                # field comment. Nullable, no FK (this codebase never
                # enforces FKs outside _init_db's own connection; existing
                # link columns like exit_outbox.trade_id are plain TEXT too).
                "ALTER TABLE trades ADD COLUMN rolled_from_trade_id TEXT",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists

    def _init_duckdb(self):
        """Initialize DuckDB analytics store (lazy — returns None if unavailable).

        R11 (smoke-script find): the duck path was cwd-relative regardless of
        this manager's db_path — every pytest run with a tmp_path SQLite
        fixture dual-wrote its FIXTURE trades into the PRODUCTION analytics
        store (105 synthetic trades found; dashboard 7d P&L read +$1,613 vs
        true +$383). The mirror now lives next to whatever SQLite file this
        manager was given, so test isolation is automatic.
        """
        try:
            from pathlib import Path as _P
            from ait.monitoring.duckdb_analytics import DuckDBAnalytics
            duck = DuckDBAnalytics(
                db_path=_P(self._db_path).parent / "ait_analytics.duckdb")
            log.info("duckdb_analytics_enabled")
            return duck
        except Exception as e:
            log.warning("duckdb_unavailable", error=str(e))
            return None

    # --- Trade Management ---

    def record_trade(self, trade: TradeRecord) -> None:
        """Insert or update a trade record.

        Uses INSERT OR IGNORE for new trades to avoid overwriting journaling
        columns (exit_reason_detailed, peak_pnl_pct, time_to_peak_hours,
        direction_correct) that are populated on close.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO trades
                   (trade_id, symbol, strategy, direction, status,
                    entry_time, entry_price, quantity, contract_type,
                    strike, expiry, exit_time, exit_price,
                    realized_pnl, commission, ml_confidence,
                    sentiment_score, market_regime, notes, legs,
                    exit_reason_detailed, peak_pnl_pct,
                    time_to_peak_hours, direction_correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.trade_id, trade.symbol, trade.strategy,
                    trade.direction.value, trade.status.value,
                    trade.entry_time, trade.entry_price, trade.quantity,
                    trade.contract_type, trade.strike, trade.expiry,
                    trade.exit_time, trade.exit_price, trade.realized_pnl,
                    trade.commission, trade.ml_confidence,
                    trade.sentiment_score, trade.market_regime,
                    trade.notes, trade.legs,
                    "", 0.0, 0.0, -1,
                ),
            )

    def update_trade_status(
        self,
        trade_id: str,
        status: TradeStatus,
        from_statuses=None,
    ) -> bool:
        """Update the status of an existing trade without touching journaling columns.

        R12 Tier-A #1: when ``from_statuses`` is given this delegates to the
        compare-and-swap :meth:`transition` (the write only lands if the row
        is currently in one of those statuses). The unconditional form remains
        ONLY for legacy callers outside the executor/reconciler; every status
        writer in the execution layer must pass a from-set or call
        transition() directly.
        """
        if from_statuses is not None:
            return self.transition(trade_id, from_statuses, status)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE trades SET status = ? WHERE trade_id = ?",
                (status.value, trade_id),
            )
        return True

    def transition(self, trade_id: str, from_statuses, to_status: TradeStatus) -> bool:
        """R12 Tier-A #1 (concurrency audit, NAIVE state-machine verdict):
        compare-and-swap status writer. Six call sites used to do blind
        UPDATEs, making illegal edges reachable: CLOSING->FILLED after a
        restart (duplicate close -> position REVERSAL), CLOSED resurrection,
        PARTIAL->FILLED quantity loss. The UPDATE only lands when the row is
        currently in one of ``from_statuses``; a blocked write is logged and
        reported so the caller can react instead of silently corrupting state.

        Returns True iff exactly one row transitioned.
        """
        froms = [
            s.value if isinstance(s, TradeStatus) else str(s)
            for s in (from_statuses if not isinstance(from_statuses, (str, TradeStatus))
                      else [from_statuses])
        ]
        to_val = to_status.value if isinstance(to_status, TradeStatus) else str(to_status)
        placeholders = ", ".join("?" for _ in froms)
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                f"UPDATE trades SET status = ? "
                f"WHERE trade_id = ? AND status IN ({placeholders})",
                (to_val, trade_id, *froms),
            )
            rowcount = cur.rowcount
        if rowcount == 1:
            return True
        log.warning(
            "illegal_transition_blocked",
            trade_id=trade_id,
            from_statuses=froms,
            to_status=to_val,
        )
        return False

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        commission: float = 0.0,
        exit_reason_detailed: str = "",
        from_statuses: tuple[TradeStatus, ...] | None = None,
        exit_time: str | None = None,
    ) -> bool:
        """Mark a trade as closed with exit details and journaling data.

        W101: ``exit_time`` (ISO, timezone-aware) books the close to the
        session it belongs to instead of now: the reconciler passes the expiry
        date for a structure it finds expired, so the loss is credited to that
        day's stats and not to today's circuit breaker (the exit outbox row
        carries the same time, and the drain decides prior-session booking
        from it). Default: now.

        R12 Tier-A #1: the close is a compare-and-swap from an OPEN status
        (PENDING/FILLED/PARTIAL/CLOSING) only. A trade that is already CLOSED
        (or CANCELLED/REJECTED) is refused — a second close used to overwrite
        the real exit data AND dual-write a duplicate row into the DuckDB
        analytics store (double-counted P&L). Returns True iff the close
        landed.

        R16: `from_statuses` narrows the CAS at the DB layer — the stale-
        pending sweep must pass (PENDING,) so a row promoted to FILLED by a
        concurrent process (second bot, operator script, broker-lag race)
        can never be booked as "never filled"/$0. Default keeps all four
        open statuses.

        W50 (money-flow-05 / trade-life-partial-slice-pnl-erased): `realized_pnl`
        here is the REMAINDER slice's own P&L (unchanged contract — callers do
        not need to change). Before this closes, any prior partial-exit
        history in open_positions.partial_exits is folded in so the trade's
        FULL round trip lands in trades.realized_pnl — the go-live verdict
        (reporting/go_live.py) reads that column directly, and the old
        behaviour silently discarded every partial slice's P&L when the
        DELETE below ran.
        """
        now = exit_time or now_et().isoformat()

        # Compute journaling fields
        peak_pnl_pct = self.get_high_water_mark(trade_id)

        # Calculate time to peak (from entry to when HWM was set)
        time_to_peak_hours = 0.0

        if from_statuses:
            _open = tuple(s.value for s in from_statuses)
            _open = (_open * 4)[:4]  # pad (by repetition) to the 4 IN placeholders
        else:
            _open = (TradeStatus.PENDING.value, TradeStatus.FILLED.value,
                     TradeStatus.PARTIAL.value, TradeStatus.CLOSING.value)
        with sqlite3.connect(self._db_path) as conn:
            # W50: BEGIN IMMEDIATE (same idiom as book_exit_core) so the read
            # of partial_exits below and the UPDATE that folds it in are one
            # atomic unit — a concurrent record_partial_exit for this SAME
            # trade_id cannot land between the read and the DELETE and be
            # silently lost.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT partial_exits FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
            partial_exits_json = row[0] if row and row[0] else "[]"
            slices_pnl = 0.0
            if row and row[0]:
                try:
                    slices_pnl = sum(
                        float(entry.get("pnl", 0.0))
                        for entry in json.loads(row[0]))
                except (json.JSONDecodeError, TypeError, AttributeError):
                    # Fail closed on the NUMBER (don't invent P&L from
                    # unparseable data) but keep the raw string archived
                    # below for forensics rather than discarding it too.
                    log.warning("partial_exits_archive_unparseable",
                                trade_id=trade_id, raw=row[0])
            folded_realized_pnl = round(realized_pnl + slices_pnl, 2)

            cur = conn.execute(
                """UPDATE trades
                   SET status = ?, exit_time = ?, exit_price = ?,
                       realized_pnl = ?, commission = ?,
                       exit_reason_detailed = ?, peak_pnl_pct = ?,
                       time_to_peak_hours = ?, partial_exits = ?
                   WHERE trade_id = ? AND status IN (?, ?, ?, ?)""",
                (TradeStatus.CLOSED.value, now, exit_price, folded_realized_pnl,
                 commission, exit_reason_detailed, peak_pnl_pct,
                 time_to_peak_hours, partial_exits_json, trade_id, *_open),
            )
            if cur.rowcount != 1:
                # R12 Tier-A #1: double-close / close-of-terminal blocked.
                log.warning(
                    "illegal_transition_blocked",
                    trade_id=trade_id,
                    from_statuses=list(_open),
                    to_status=TradeStatus.CLOSED.value,
                    context="close_trade",
                )
                return False

            # Clean up open_positions row now that trade is closed
            conn.execute(
                "DELETE FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            )

            # W1: enqueue the booking obligation in the SAME transaction as
            # the close — whoever books stats/breaker must claim this row
            # first (claim_exit_booking), so a close can be booked exactly
            # once no matter which path (executor callback, reconciler,
            # sweep, crash-recovery drain) discovers it.
            # W50: folded_realized_pnl (the WHOLE round trip), not the bare
            # `realized_pnl` argument (the remainder slice alone) — the
            # outbox drain (pending_exit_bookings) reads this column as if it
            # were the trade's total P&L, same as the in-process callback
            # path reads completed_exits[...]["realized_pnl"]. Both paths
            # must agree on what "the trade's P&L" means.
            conn.execute(
                "INSERT OR REPLACE INTO exit_outbox "
                "(trade_id, realized_pnl, exit_reason, exit_time) "
                "VALUES (?, ?, ?, ?)",
                (trade_id, folded_realized_pnl, exit_reason_detailed, now),
            )

            # Dual-write: sync closed trade to DuckDB analytics
            if self._duck:
                try:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
                    ).fetchone()
                    if row:
                        self._duck.ingest_trade(dict(row))
                except Exception as e:
                    log.warning("duckdb_trade_sync_failed", trade_id=trade_id, error=str(e))
        return True  # R12 Tier-A #1: close landed (CAS matched an open status)

    def insert_open_position(
        self,
        trade_id: str,
        symbol: str,
        contract_type: str,
        quantity: int,
        entry_price: float,
        legs: str = "[]",
    ) -> None:
        """Insert a row into open_positions so HWM / partial-exit tracking works.

        Uses trade_id as position_id.  Called when an entry order fills.
        """
        now = now_et().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO open_positions
                   (position_id, trade_id, symbol, contract_type, quantity,
                    entry_price, entry_time, current_price, unrealized_pnl,
                    stop_loss, take_profit, legs, high_water_mark, partial_exits)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id, trade_id, symbol, contract_type, quantity,
                    entry_price, now, entry_price, 0.0,
                    None, None, legs, 0.0, "[]",
                ),
            )
        log.info("open_position_inserted", trade_id=trade_id, symbol=symbol)

    def remove_open_position(self, trade_id: str) -> None:
        """Remove a position from open_positions when it is fully closed."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            )

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all trades that are not closed/cancelled."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status IN (?, ?, ?, ?)",
                (TradeStatus.PENDING.value, TradeStatus.FILLED.value,
                 TradeStatus.PARTIAL.value, TradeStatus.CLOSING.value),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trade_by_id(self, trade_id: str) -> TradeRecord | None:
        """Get a single trade record by ID (any status)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return self._row_to_trade(row) if row else None

    def get_trades_for_date(self, d: date) -> list[TradeRecord]:
        """Get all trades entered on a specific date."""
        date_str = d.isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE entry_time LIKE ?",
                (f"{date_str}%",),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_recent_trades(self, n: int = 20) -> list[TradeRecord]:
        """Get the N most recent trades."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # --- Daily Stats ---

    def update_daily_stats(self, stats: DailyStats) -> None:
        """Update daily statistics (SQLite + DuckDB)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_stats
                   (date, trades_taken, trades_won, trades_lost,
                    total_pnl, max_drawdown, day_trades_count,
                    circuit_breaker_triggered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stats.date, stats.trades_taken, stats.trades_won,
                    stats.trades_lost, stats.total_pnl, stats.max_drawdown,
                    stats.day_trades_count, stats.circuit_breaker_triggered,
                ),
            )

        # Dual-write to DuckDB
        if self._duck:
            try:
                self._duck.ingest_daily_stats({
                    "date": stats.date,
                    "trades_taken": stats.trades_taken,
                    "trades_won": stats.trades_won,
                    "trades_lost": stats.trades_lost,
                    "total_pnl": stats.total_pnl,
                    "max_drawdown": stats.max_drawdown,
                    "day_trades_count": stats.day_trades_count,
                    "circuit_breaker_triggered": stats.circuit_breaker_triggered,
                })
            except Exception as e:
                log.warning("duckdb_daily_sync_failed", error=str(e))

    def get_daily_stats(self, d: date | None = None) -> DailyStats:
        """Get daily stats for a specific date (default: today)."""
        from ait.utils.time import now_et
        d = d or now_et().date()  # ET-pinned (A3)
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM daily_stats WHERE date = ?",
                (d.isoformat(),),
            ).fetchone()

        if row:
            return DailyStats(**{
            # R8: same schema-ahead-of-dataclass armor as _row_to_trade —
            # this sits in the fast monitor + exit booking; an unguarded
            # daily_stats migration would repeat the 07-10 incident.
            k: v for k, v in dict(row).items()
            if k in {f.name for f in __import__("dataclasses").fields(DailyStats)}
        })
        return DailyStats(date=d.isoformat())

    # --- High Water Mark & Partial Exits ---

    def update_high_water_mark(self, trade_id: str, hwm: float) -> None:
        """Update the high water mark for a trade's P&L percentage."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE open_positions SET high_water_mark = MAX(high_water_mark, ?) WHERE trade_id = ?",
                (hwm, trade_id),
            )

    def update_position_mark(
        self, trade_id: str, unrealized_pnl: float, pnl_pct: float
    ) -> None:
        """Persist the live mark for an open position.

        Called from PortfolioManager._evaluate_position on every tick where
        real IBKR leg marks are present. Before this existed, unrealized_pnl
        sat at its insert-time 0.0 forever and the dashboard had no live
        per-position P&L (audit 2026-07-07 item 1.4).
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE open_positions SET unrealized_pnl = ?, pnl_pct = ?, "
                "mark_time = ? WHERE trade_id = ?",
                (unrealized_pnl, pnl_pct, datetime.now().isoformat(), trade_id),
            )

    def get_position_mark(self, trade_id: str) -> dict | None:
        """Return the last persisted live mark for an open position.

        R16: added for the exit-pricing mark anchor — the wing-width sanity
        cap alone stopped binding once wings widened to $30-35 (a garbage
        quote of 10.00 on a structure whose fair close is ~$1 passes a
        35.00 cap). Consumers must check mark_time freshness themselves.
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT unrealized_pnl, pnl_pct, mark_time FROM open_positions "
                "WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        if row is None or row[0] is None or row[2] is None:
            return None
        return {"unrealized_pnl": row[0], "pnl_pct": row[1], "mark_time": row[2]}

    def get_high_water_mark(self, trade_id: str) -> float:
        """Get the high water mark P&L pct for a trade."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT high_water_mark FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return row[0] if row else 0.0

    def record_partial_exit(
        self,
        trade_id: str,
        quantity: int,
        price: float,
        pnl: float,
        pnl_level: float | None = None,
    ) -> None:
        """Record a partial exit for a trade.

        Args:
            pnl_level: The milestone P&L level that triggered this partial exit.
                       Stored so the same level is not re-triggered.
        """
        try:
            raw = self.get_partial_exits(trade_id)
        except PartialExitsCorruptError as e:
            # Keep the unreadable history for a human instead of overwriting it
            # without a trace; the slice P&L inside it cannot be recovered here.
            self.set_state(f"partial_exits_corrupt_{trade_id}", e.raw[:2000])
            raw = []
        entry: dict = {
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "time": datetime.now().isoformat(),
        }
        if pnl_level is not None:
            entry["pnl_level"] = pnl_level
        raw.append(entry)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE open_positions SET partial_exits = ? WHERE trade_id = ?",
                (json.dumps(raw), trade_id),
            )

    def get_partial_exits(self, trade_id: str) -> list[dict]:
        """Get partial exit history for a trade.

        W104: raises ``PartialExitsCorruptError`` when the stored history cannot
        be read as a list of slices. It used to return ``[]`` for bad JSON, which
        told the partial-exit check that no level had been taken, so a level that
        already sold contracts fired again; and it passed ``null``/``{}`` through,
        which raised inside ``check_positions`` and cost the whole pass its exit
        evaluation. An empty history is still ``[]``.
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT partial_exits FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        if row and row[0]:
            try:
                val = json.loads(row[0])
            except json.JSONDecodeError:
                val = None
            if not isinstance(val, list) or not all(isinstance(p, dict) for p in val):
                log.error("partial_exits_corrupt", trade_id=trade_id,
                          raw=str(row[0])[:200])
                raise PartialExitsCorruptError(trade_id, str(row[0]))
            return val
        return []

    def update_trade_entry_price(self, trade_id: str, entry_price: float) -> None:
        """Overwrite trades.entry_price with the REAL fill price.

        The signal's target price is recorded at order placement; once IBKR
        reports the actual fill, this replaces it so every downstream P&L
        computation (exits, analytics, learning) uses reality, not the
        optimistic target (audit R2/C1 — entry slippage was invisible).
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE trades SET entry_price = ? WHERE trade_id = ?",
                (entry_price, trade_id),
            )

    def update_trade_quantity(self, trade_id: str, new_quantity: int) -> None:
        """Update remaining quantity after a partial exit."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE trades SET quantity = ? WHERE trade_id = ?",
                (new_quantity, trade_id),
            )
            conn.execute(
                "UPDATE open_positions SET quantity = ? WHERE trade_id = ?",
                (new_quantity, trade_id),
            )

    # --- Trade Context ---

    def save_trade_context(
        self,
        trade_id: str,
        direction: str,
        confidence: float,
        regime: str,
        vix: float,
        iv_rank: float,
        sentiment_score: float,
        signals: str = "{}",
        model_version: str = "",
        range_probability: float | None = None,
        range_threshold_pct: float | None = None,
        range_horizon_days: int | None = None,
        range_model_version: str | None = None,
        entry_gates_mode: str | None = None,
        directional_confidence: float | None = None,
    ) -> None:
        """Save the market context at time of trade entry (SQLite + DuckDB).

        W10: the range-model fields are separate from `confidence` on purpose.
        `confidence` is whatever number actually drove risk validation for
        this trade, which changes with gate mode; the range fields record what
        the model SAID, under which spec, so the observe-mode verdict stays
        scoreable across a mode change. None means "not produced" — which is
        itself the answer for a symbol whose model failed its edge gate — and
        must never be written as 0.0."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trade_context
                   (trade_id, entry_direction, entry_confidence, entry_regime,
                    entry_vix, entry_iv_rank, entry_sentiment_score, entry_signals,
                    model_version, range_probability, range_threshold_pct,
                    range_horizon_days, range_model_version, entry_gates_mode,
                    directional_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, direction, confidence, regime, vix, iv_rank,
                 sentiment_score, signals, model_version, range_probability,
                 range_threshold_pct, range_horizon_days, range_model_version,
                 entry_gates_mode, directional_confidence),
            )

        # Dual-write to DuckDB
        if self._duck:
            try:
                self._duck.ingest_trade_context({
                    "trade_id": trade_id,
                    "entry_direction": direction,
                    "entry_confidence": confidence,
                    "entry_regime": regime,
                    "entry_vix": vix,
                    "entry_iv_rank": iv_rank,
                    "entry_sentiment_score": sentiment_score,
                    "entry_signals": signals,
                })
            except Exception as e:
                log.warning("duckdb_context_sync_failed", trade_id=trade_id, error=str(e))

    def get_trade_context(self, trade_id: str) -> dict | None:
        """Get the entry context for a trade."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_context WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return dict(row) if row else None

    # --- Bot State (Key-Value) ---

    #: W14: a claim older than this is assumed to belong to a booker that
    #: died mid-flight, and the obligation is re-offered. Long enough that a
    #: slow-but-live booker is never overtaken.
    CLAIM_STALE_SECONDS = 300.0

    def claim_exit_booking(self, trade_id: str) -> bool:
        """Claim the right to book a close's stats/breaker effects.

        W14 (R24 concurrency-01): this used to DELETE the row and return
        rowcount — a committed destruction performed BEFORE any booking write.
        A process kill or sqlite error in that window lost the obligation
        entirely: nothing had been booked, and the drain had nothing left to
        find, so a real stop-out could vanish from the breaker and daily stats
        for good. The claim now MARKS the row (UPDATE ... WHERE claimed_at IS
        NULL, still serialized by sqlite, so exactly one claimer wins) and
        :meth:`complete_exit_booking` deletes it once the booking has landed.
        A claim left stale by a crash is re-offered by
        :meth:`pending_exit_bookings`.
        """
        from datetime import timedelta as _td
        stale = (now_et() - _td(seconds=self.CLAIM_STALE_SECONDS)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE exit_outbox SET claimed_at = ? "
                "WHERE trade_id = ? AND (claimed_at IS NULL OR claimed_at < ?)",
                (now_et().isoformat(), trade_id, stale))
            return cur.rowcount > 0

    def complete_exit_booking(self, trade_id: str) -> None:
        """W14: the booking landed — retire the obligation.

        Called ONLY after stats/breaker/PDT/Thompson have been updated. Until
        this runs the row survives, so a crash mid-booking leaves work the
        drain will retry rather than silently dropping a loss.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM exit_outbox WHERE trade_id = ?", (trade_id,))

    def exit_core_booked(self, trade_id: str) -> bool | None:
        """W32: have this close's money writes landed? None = no outbox row
        (already retired, or never enqueued)."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT core_booked FROM exit_outbox WHERE trade_id = ?",
                (trade_id,)).fetchone()
        return None if row is None else bool(row[0])

    def book_exit_core(self, trade_id: str, pnl: float,
                       breaker_key: str | None = None,
                       breaker_blob: str | None = None,
                       stats_date: str | None = None,
                       outcome_pnl: float | None = None) -> bool:
        """W32: apply a close's MONEY effects exactly once.

        One sqlite transaction writes today's daily_stats delta, the circuit
        breaker's persisted blob (when the caller supplies one) and the
        core_booked marker. Either all three land or none does, so a retry —
        by the drain in this process, or by the next process after a kill —
        can tell "already counted" from "never counted" instead of guessing.
        Returns True iff THIS call booked it.

        W37: ``stats_date`` (ISO) credits a PRIOR session's daily_stats row
        instead of today's — for a close the drain finds a day late. The
        caller passes no breaker blob for those: the breaker is today's.

        W50 (money-flow-05): ``pnl`` is the total_pnl DELTA — for a trade
        with partial-exit history this is the remainder only, since the
        slices already fed daily_stats/breaker in real time when they closed
        (see orchestrator._archived_partial_pnl). ``outcome_pnl`` (defaults
        to ``pnl``) is the trade's FULL round-trip P&L and decides the
        trades_won/trades_lost classification — a trade that nets positive
        overall must count as a win even when its remainder slice alone was
        a loss.
        """
        _outcome = pnl if outcome_pnl is None else outcome_pnl
        day = stats_date or now_et().date().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT core_booked FROM exit_outbox WHERE trade_id = ?",
                (trade_id,)).fetchone()
            if row is None or row[0]:
                return False
            conn.execute("INSERT OR IGNORE INTO daily_stats (date) VALUES (?)", (day,))
            conn.execute(
                "UPDATE daily_stats SET total_pnl = COALESCE(total_pnl, 0) + ?, "
                "trades_won = COALESCE(trades_won, 0) + ?, "
                "trades_lost = COALESCE(trades_lost, 0) + ? WHERE date = ?",
                (pnl, 1 if _outcome > 0 else 0, 0 if _outcome > 0 else 1, day))
            if breaker_key and breaker_blob is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO bot_state (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (breaker_key, breaker_blob, datetime.now().isoformat()))
            conn.execute("UPDATE exit_outbox SET core_booked = 1 WHERE trade_id = ?",
                         (trade_id,))
        if self._duck:  # best-effort mirror, never part of the transaction
            try:
                st = self.get_daily_stats(date.fromisoformat(day))
                self._duck.ingest_daily_stats({
                    "date": st.date, "trades_taken": st.trades_taken,
                    "trades_won": st.trades_won, "trades_lost": st.trades_lost,
                    "total_pnl": st.total_pnl, "max_drawdown": st.max_drawdown,
                    "day_trades_count": st.day_trades_count,
                    "circuit_breaker_triggered": st.circuit_breaker_triggered,
                })
            except Exception as e:  # noqa: BLE001
                log.warning("duckdb_daily_sync_failed", error=str(e))
        return True

    def exit_steps_done(self, trade_id: str) -> set[str]:
        """W32: the non-money booking steps that have already landed."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT steps_done FROM exit_outbox WHERE trade_id = ?",
                (trade_id,)).fetchone()
        return {x for x in ((row[0] or "") if row else "").split(",") if x}

    def mark_exit_step(self, trade_id: str, step: str) -> None:
        """W32: record that one side effect of a close has landed, so a retry
        skips it instead of repeating it."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT steps_done FROM exit_outbox WHERE trade_id = ?",
                (trade_id,)).fetchone()
            if row is None:
                return
            done = {x for x in (row[0] or "").split(",") if x}
            done.add(step)
            conn.execute("UPDATE exit_outbox SET steps_done = ? WHERE trade_id = ?",
                         (",".join(sorted(done)), trade_id))

    def pending_exit_bookings(self, older_than_seconds: float = 120.0) -> list[dict]:
        """W1: closes whose booking nobody has claimed yet. The grace period
        keeps the drain from racing an in-flight executor booking callback —
        the callback claims within seconds; anything older is an orphan
        (reconciler/sweep close, or a booking lost to a crash)."""
        from datetime import timedelta as _td
        cutoff = (now_et() - _td(seconds=older_than_seconds)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            # W14: unclaimed rows, PLUS rows whose claim went stale because
            # the booker died between claiming and writing.
            stale = (now_et() - _td(seconds=self.CLAIM_STALE_SECONDS)).isoformat()
            rows = conn.execute(
                "SELECT trade_id, realized_pnl, exit_reason, exit_time "
                "FROM exit_outbox WHERE (exit_time < ? "
                # W104: a value that is not date-shaped sorts after every date
                # ("N/A" > "2026"), so `exit_time < ?` alone never selected it.
                "OR exit_time NOT GLOB '[0-9][0-9][0-9][0-9]-*') "
                "AND (claimed_at IS NULL OR claimed_at < ?) ORDER BY exit_time",
                (cutoff, stale)).fetchall()
            return [dict(r) for r in rows]

    def set_state(self, key: str, value: str) -> None:
        """Set a key-value pair in bot state."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now().isoformat()),
            )

    def get_state(self, key: str, default: str = "") -> str:
        """Get a value from bot state."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def _connect(self):
        """R8 CRITICAL fix: the four R7 ledger methods were written against a
        _connect() helper that never existed — every call raised
        AttributeError (swallowed at their call sites), leaving the entire
        real-cost ledger dead-on-arrival: executions empty, commissions $0,
        capital_at_risk 0 on every row. sqlite3.Connection is a context
        manager, matching the `with self._connect() as conn:` usage."""
        import sqlite3 as _sq
        return _sq.connect(self._db_path)

    def record_execution(self, exec_id: str, order_id: int, perm_id: int,
                         trade_id: str, symbol: str, con_id: int, side: str,
                         shares: float, price: float, exec_time: str,
                         commission: float | None = None, realized_pnl: float = 0.0,
                         signal_price: float = 0.0, live_mid: float = 0.0,
                         nbbo_spread: float = 0.0, client_id: int | None = None,
                         order_ref: str = "", origin: str | None = None) -> None:
        """R7: upsert one broker execution (commission may arrive later —
        re-upserting the same exec_id refreshes it).

        W48 (trade-life-commission-trueup-gate-measures-wrong-quantity):
        `commission=None` means "not yet reported", NOT "reported as zero" —
        the column already allows NULL (no NOT NULL constraint), so this
        needed no migration. Before this, every fill was written with a
        literal 0.0 the instant it was seen, indistinguishable from a broker
        report that genuinely said $0 — a completeness check built on row
        presence (or even on `commission IS NOT NULL` without this change)
        could never tell "reported" from "not yet reported".

        W105: ``client_id`` / ``order_ref`` say who placed the order and ``origin``
        classes the row: ``bot`` (a bot order, attributed or not), ``manual`` (the
        operator in TWS, clientId 0) or ``other`` (a probe or another tool). NULL is a
        row written before this column existed. Manual and other rows have no bot
        trade by design; an unattributed ``bot`` row is the real orphan.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO executions
                   (exec_id, order_id, perm_id, trade_id, symbol, con_id,
                    side, shares, price, exec_time, commission, realized_pnl,
                    signal_price, live_mid, nbbo_spread, client_id, order_ref, origin)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(exec_id) DO UPDATE SET
                     client_id=COALESCE(client_id, excluded.client_id),
                     order_ref=CASE WHEN COALESCE(order_ref, '') = ''
                                    THEN excluded.order_ref ELSE order_ref END,
                     -- a bot row never becomes manual/other; a row first seen as
                     -- manual/other becomes bot if a later pass attributes it
                     origin=CASE WHEN excluded.origin = 'bot' THEN 'bot'
                                 WHEN origin = 'bot' THEN 'bot'
                                 WHEN excluded.origin IS NOT NULL THEN excluded.origin
                                 ELSE origin END,
                     -- W48: never regress a known commission back to unknown
                     -- (a later sweep pass re-seeing the same fill without a
                     -- fresh commissionReport attached must not erase one
                     -- already recorded) -- the same "never zero a known
                     -- value" principle the three fields below already had,
                     -- which commission alone was missing until now.
                     commission=COALESCE(excluded.commission, commission),
                     realized_pnl=excluded.realized_pnl,
                     trade_id=CASE WHEN excluded.trade_id != ''
                                   THEN excluded.trade_id ELSE trade_id END,
                     -- R13: fill-quality context is only known at placement;
                     -- a later sweep may carry it when the first insert did
                     -- not (event-fill races). Never zero a known value.
                     signal_price=CASE WHEN excluded.signal_price > 0
                                       THEN excluded.signal_price ELSE signal_price END,
                     live_mid=CASE WHEN excluded.live_mid > 0
                                   THEN excluded.live_mid ELSE live_mid END,
                     nbbo_spread=CASE WHEN excluded.nbbo_spread > 0
                                      THEN excluded.nbbo_spread ELSE nbbo_spread END""",
                (exec_id, order_id, perm_id, trade_id, symbol, con_id, side,
                 shares, price, exec_time, commission, realized_pnl,
                 signal_price, live_mid, nbbo_spread, client_id, order_ref or "", origin),
            )

    def total_commission(self, trade_id: str) -> float:
        """R7: sum of real broker commissions recorded for a trade."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(commission),0) FROM executions "
                "WHERE trade_id = ?", (trade_id,)).fetchone()
            return float(row[0] or 0.0)

    def count_executions(self, trade_id: str) -> int:
        """R16: execution rows recorded for a trade.

        W48: this is raw row PRESENCE, not commission presence — it has a
        second, different caller (executor.py's reconnect fill-reconstruction
        liveness check) that genuinely wants "did any fill get recorded at
        all", regardless of commission. The commission true-up's own
        completeness check is `count_executions_with_commission` below; do
        not repurpose this method for that again.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM executions WHERE trade_id = ?",
                (trade_id,)).fetchone()
            return int(row[0] or 0)

    def count_executions_with_commission(
        self, trade_id: str, exclude_con_id: int | None = None,
    ) -> int:
        """W48: execution rows for a trade whose commission is actually
        KNOWN (reported by the broker), for the commission true-up's real
        completeness check — see `record_execution`'s docstring for why
        `commission IS NOT NULL` is the right test now that NULL means
        "not yet reported" rather than a placeholder zero.

        `exclude_con_id` matters: a combo order's synthetic BAG summary row
        gets its own commissionReport too (real production case study,
        `tests/test_r16_broker_fixes.py`'s 2026-08-04 SPY condor fixture —
        reported as exactly 0.0, a real report, not a missing one). Counting
        it toward completeness lets a handful of quickly-reported BAG rows
        substitute for still-pending LEG rows and satisfy the gate early —
        this wave's own version of the original defect. Callers pass
        `BAG_CON_ID` (execution.executor) to exclude it; `None` counts every
        row (this trade has no combo leg, or the caller has already excluded
        BAG rows some other way).
        """
        with self._connect() as conn:
            if exclude_con_id is None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM executions "
                    "WHERE trade_id = ? AND commission IS NOT NULL",
                    (trade_id,)).fetchone()
            else:
                # SQL three-valued logic: `con_id != ?` is NULL (not TRUE)
                # for a row whose con_id is itself NULL, so a bare != would
                # silently EXCLUDE it -- the opposite of "count every row
                # that isn't the BAG one". Production rows always populate
                # con_id (executor.py's `getattr(..., 0) or 0`), but this
                # must not depend on that; treat NULL as "not the BAG row".
                row = conn.execute(
                    "SELECT COUNT(*) FROM executions WHERE trade_id = ? "
                    "AND commission IS NOT NULL "
                    "AND (con_id IS NULL OR con_id != ?)",
                    (trade_id, exclude_con_id)).fetchone()
            return int(row[0] or 0)

    def executed_shares_by_con(self, trade_id: str) -> dict[int, float]:
        """W112: contracts executed per ``con_id`` for a trade (the BAG summary row included under its
        own id), from the on-disk ledger. Used to recover how many lots a fill that predates a reconnect
        covered, from the one source that survives it."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT COALESCE(con_id, 0), COALESCE(SUM(shares), 0) FROM executions "
                "WHERE trade_id = ? GROUP BY COALESCE(con_id, 0)", (trade_id,)).fetchall()
        return {int(c): float(q or 0.0) for c, q in rows}

    def sum_commissioned_shares(self, trade_id: str, exclude_con_id: int | None = None) -> float:
        """W112: the CONTRACTS (not rows) of a trade's executions whose commission is known.

        The commission true-up's completeness is "every leg's entry and exit contracts have a reported
        commission". At one lot a leg fills in one execution so a row count says that; at N lots a leg can
        fill in several executions, and the rows of the entry alone can reach the row count before any exit
        commission exists. Same filters as `count_executions_with_commission` (known commission, BAG row
        excluded, a NULL con_id is not the BAG row). A row with no recorded size counts as one contract,
        which is what the row count assumed.
        """
        with self._connect() as conn:
            if exclude_con_id is None:
                row = conn.execute(
                    "SELECT COALESCE(SUM(CASE WHEN shares IS NULL OR shares <= 0 THEN 1 ELSE shares END), 0) "
                    "FROM executions WHERE trade_id = ? AND commission IS NOT NULL",
                    (trade_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(CASE WHEN shares IS NULL OR shares <= 0 THEN 1 ELSE shares END), 0) "
                    "FROM executions WHERE trade_id = ? "
                    "AND commission IS NOT NULL AND (con_id IS NULL OR con_id != ?)",
                    (trade_id, exclude_con_id)).fetchone()
            return float(row[0] or 0.0)

    def state_keys_like(self, pattern: str) -> list[str]:
        """R16: bot_state keys matching a SQL LIKE pattern.

        Needed to sweep orphaned trade_maxloss_* keys (they leak whenever a
        trade leaves the book by any path other than a completed exit, and
        each orphan permanently inflates the aggregate risk denominator).
        """
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM bot_state WHERE key LIKE ?", (pattern,)
            ).fetchall()
        return [r[0] for r in rows]

    def pending_trueup_trade_ids(self) -> list[str]:
        """R16: trades whose commission true-up was deferred on a partial
        executions ledger (bot_state keys trueup_pending_<trade_id>)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM bot_state WHERE key LIKE 'trueup_pending_%'"
            ).fetchall()
        return [r[0][len("trueup_pending_"):] for r in rows]

    def update_trade_realized_pnl(self, trade_id: str, realized_pnl: float) -> None:
        """R15 #8: commission true-up rewrite of a just-closed trade's P&L
        (estimate swapped for the executions-ledger truth at booking time)."""
        with self._connect() as conn:
            conn.execute("UPDATE trades SET realized_pnl = ? WHERE trade_id = ?",
                         (realized_pnl, trade_id))

    def update_trade_commission(self, trade_id: str, commission: float) -> None:
        """R7: record the REAL round-trip commission on the trade row."""
        with self._connect() as conn:
            conn.execute("UPDATE trades SET commission = ? WHERE trade_id = ?",
                         (commission, trade_id))
            # R13 (shadow-referee break #9): commissions arrive AFTER
            # close_trade's dual-write, so the DuckDB mirror kept
            # commission=0 forever (sqlite $22.72 vs duckdb $0.00 measured
            # 2026-07-14). Re-ingest the row — ingest_trade is an
            # INSERT OR REPLACE upsert, so this is idempotent.
            if self._duck:
                try:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
                    ).fetchone()
                    if row:
                        self._duck.ingest_trade(dict(row))
                except Exception as e:
                    log.warning("duckdb_commission_sync_failed",
                                trade_id=trade_id, error=str(e))

    def update_trade_reject_reason(self, trade_id: str, code: int, text: str) -> None:
        """W51 (model-vs-reality-margin-model-unimplemented): stamp the
        broker's own reason a CANCELLED entry died, when errorEvent captured
        one. No DuckDB re-ingest — a cancelled entry never reaches
        close_trade's dual-write, so there is no DuckDB row to keep in sync."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET reject_code = ?, reject_text = ? WHERE trade_id = ?",
                (code, text, trade_id))

    def set_trade_capital_at_risk(self, trade_id: str, car: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET capital_at_risk = ? WHERE trade_id = ?",
                (car, trade_id))

    def count_roll_ancestors(self, trade_id: str, max_depth: int = 20) -> int:
        """W54a: how many times has `trade_id` already been rolled -- a
        single recursive-CTE walk BACKWARD from `trade_id` via
        rolled_from_trade_id to its origin, counting hops. A never-rolled
        trade returns 0.

        Deliberately NOT go_live.py's fold_roll_chains, confirmed by direct
        read this session that function can't be reused here: it only walks
        CLOSED rows FORWARD for historical accounting and drops any chain
        that ends at a still-open trade entirely -- exactly the live,
        open-position case this exists for.

        `max_depth` bounds the recursion (belt-and-suspenders -- rolled_
        from_trade_id is written exactly once, at seed time, by roll_
        booking.seed_roll_opening_trade, so a cycle should never exist, but
        the bound costs nothing and this must never be the query that hangs
        a live decision).
        """
        with self._connect() as conn:
            row = conn.execute(
                """WITH RECURSIVE anc(id, parent, depth) AS (
                       SELECT trade_id, rolled_from_trade_id, 0 FROM trades
                       WHERE trade_id = ?
                       UNION ALL
                       SELECT t.trade_id, t.rolled_from_trade_id, anc.depth + 1
                       FROM trades t JOIN anc ON t.trade_id = anc.parent
                       WHERE anc.depth < ?
                   )
                   SELECT MAX(depth) FROM anc""",
                (trade_id, max_depth),
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def delete_state(self, key: str) -> None:
        """Delete a key from bot state (e.g. per-trade risk keys on close)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM bot_state WHERE key = ?", (key,))

    # --- Helpers ---

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        d = dict(row)
        d["direction"] = TradeDirection(d["direction"])
        d["status"] = TradeStatus(d["status"])
        # R7 incident (2026-07-10): a migration added a column the dataclass
        # didn't have — TradeRecord(**d) raised on EVERY row read, which
        # silently killed the fast monitor and trading cycle for a full
        # session (stops/TPs off all day). Filter to known fields so a
        # schema-ahead-of-dataclass mismatch can never take the loop down;
        # log once if we drop anything so the drift is visible.
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(TradeRecord)}
        extra = set(d) - known
        if extra:
            import structlog as _sl
            _sl.get_logger("bot.state").warning(
                "trade_row_extra_columns_dropped", columns=sorted(extra))
            d = {k: v for k, v in d.items() if k in known}
        return TradeRecord(**d)
