#!/usr/bin/env python3
"""Post-deploy smoke test — run after EVERY deploy, before walking away.

    python scripts/smoke_deploy.py             # local verdict only
    python scripts/smoke_deploy.py --telegram  # also send the verdict via Telegram

Why this exists: on 2026-07-10 a deploy with a fully green unit-test suite
shipped a runtime-only fault (schema vs dataclass mismatch in the state
layer) that killed the trading loop for an entire session — stops and
take-profits were unmanaged — and it was discovered a day later. Unit tests
run against fixtures; this script executes the SAME paths the live bot
executes at runtime, against the REAL config and the REAL databases:

  1.  import every module under src/ait           (import-time faults)
  2.  load_settings('config.yaml')                (config parse + validation)
  2b. ONE REAL TRADING CYCLE, dry-run             (the entry pipeline is
      EXECUTED against fakes in a throwaway sandbox — see
      check_trading_cycle_dryrun; added after the 2026-08 outage below)
  3.  StateManager on the real data/ait_state.db  (row->dataclass mapping of
      every open + recent trade, daily stats, trade context, commissions)
  4.  TradeAnalytics.get_performance(7)           (DuckDB path AND the forced
      SQLite fallback path)
  5.  orchestrator daily_digest/weekly_scorecard  (real-DB read paths; _alert
      is monkeypatched to a local collector — nothing is sent to Telegram)
  6.  CounterfactualTracker / ThompsonSampler     (learning-state loads)
  7.  IronCondor._vol_scaled_width               (wing sizing with and
      without a risk budget)

2026-08 (why check 2b exists): a one-line AttributeError in
TradingOrchestrator._trading_cycle ran live for THREE DAYS. Checks 1 and 2
were green the entire time — importing a module and parsing a config prove
nothing about whether the loop can complete a pass. The gate now runs one
real cycle and one real fast-monitor tick.

Safety: read-only by design. It never inserts/updates/deletes rows; a
no-write guard snapshots per-table row counts + schema versions of BOTH DBs
before and after and FAILS the smoke if row data changed. (StateManager's
own idempotent DDL migrations may bump the SQLite schema version on a
migration-bearing deploy — that is reported as a warning, not a failure,
because bot startup would apply the identical DDL.) No IBKR connection is
opened. Refuses to create a missing state DB.

Exit 0  => prints "SMOKE PASS (n checks)".
Exit 1  => compact failure list.
Exit 2  => the harness itself crashed (treat as FAIL).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pkgutil
import sqlite3
import sys
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "data" / "ait_state.db"
DUCK_PATH = ROOT / "data" / "ait_analytics.duckdb"

# R17: tables the LIVE BOT itself legitimately writes during normal
# operation (trade open/close, daily stats, feature snapshots). A row-count
# delta confined to these, while the bot's own heartbeat is fresh, is
# positive evidence of a concurrent legitimate writer -- not the smoke
# script writing to a database, the actual failure mode this guard exists
# to catch (which most plausibly happens with the bot mid-restart/stopped).
LIVE_WRITE_TABLES = {
    "trades", "daily_stats", "open_positions", "trade_context",
    "bot_state", "executions", "feature_snapshots", "equity_stats",
}
HEARTBEAT_FRESH_SECS = 600


def _bot_heartbeat_fresh() -> bool:
    hb = ROOT / "data" / "bot_heartbeat"
    if not hb.exists():
        return False
    return (time.time() - hb.stat().st_mtime) < HEARTBEAT_FRESH_SECS

RESULTS: list[tuple[str, str | None]] = []  # (check name, None | error text)
WARNINGS: list[str] = []
_SETTINGS = None  # cached by check_load_settings for --telegram reuse


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Run from repo root regardless of cwd; make stdout unicode-safe."""
    os.chdir(ROOT)  # config.yaml, data/, logs/ are all cwd-relative in src
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - non-tty streams may lack reconfigure
            pass
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # The repo root itself, so check_trading_cycle_dryrun can import the
    # shared hot-path rig from tests/ (running as `python scripts/...` puts
    # scripts/ on sys.path, not the root).
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _fmt_exc(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    loc = ""
    if tb:
        frame = tb[-1]
        loc = f" (at {Path(frame.filename).name}:{frame.lineno})"
    return f"{type(e).__name__}: {e}{loc}"


def _run(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, None))
        print(f"[ok]   {name}" + (f" -- {detail}" if detail else ""))
    except KeyboardInterrupt:
        raise
    except BaseException as e:  # noqa: BLE001 - a smoke must survive anything
        msg = _fmt_exc(e)
        RESULTS.append((name, msg))
        print(f"[FAIL] {name}: {msg}")


# ---------------------------------------------------------------------------
# No-write guard — verify the smoke committed nothing to either DB
# ---------------------------------------------------------------------------

class NoWriteGuard:
    """Row counts + schema versions of both DBs, before vs after.

    - SQLite: every table's COUNT(*) plus PRAGMA schema_version via a
      read-only URI connection. data_version is tracked on a held-open ro
      connection purely as information (the live bot may legitimately commit
      while the smoke runs; row-count parity is the assertion).
    - DuckDB: every base table's COUNT(*) via read_only connect, opened and
      closed immediately (DuckDB dislikes mixed ro/rw connections).
    - Row-count change => FAIL. schema_version change with identical row
      counts => WARN (StateManager's idempotent startup migrations).
    """

    def __init__(self) -> None:
        self.sqlite_before = self._sqlite_snapshot()
        self.duck_before = self._duck_snapshot()
        self._dv_con = None
        self._dv_before = None
        if SQLITE_PATH.exists():
            try:
                self._dv_con = sqlite3.connect(self._ro_uri(), uri=True)
                self._dv_before = self._dv_con.execute(
                    "PRAGMA data_version").fetchone()[0]
            except sqlite3.Error:
                self._dv_con = None

    @staticmethod
    def _ro_uri() -> str:
        return f"file:{SQLITE_PATH.as_posix()}?mode=ro"

    def _sqlite_snapshot(self) -> dict | None:
        if not SQLITE_PATH.exists():
            return None
        try:
            con = sqlite3.connect(self._ro_uri(), uri=True)
        except sqlite3.Error:
            # ro open can fail on a WAL db with no -shm; fall back to a
            # normal connection that only ever SELECTs.
            con = sqlite3.connect(SQLITE_PATH)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            snap = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    for t in tables}
            snap["__schema_version__"] = con.execute(
                "PRAGMA schema_version").fetchone()[0]
            return snap
        finally:
            con.close()

    @staticmethod
    def _duck_snapshot() -> dict | None:
        if not DUCK_PATH.exists():
            return None
        try:
            import duckdb
            con = duckdb.connect(str(DUCK_PATH), read_only=True)
        except Exception as e:  # noqa: BLE001 - locked by live bot, etc.
            WARNINGS.append(f"duckdb snapshot unavailable ({e}); "
                            "duckdb no-write compare skipped")
            return None
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type='BASE TABLE' ORDER BY table_name").fetchall()]
            return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    for t in tables}
        finally:
            con.close()

    @staticmethod
    def _diff(before: dict, after: dict) -> str:
        keys = sorted(set(before) | set(after))
        return ", ".join(
            f"{k}: {before.get(k, '<absent>')} -> {after.get(k, '<absent>')}"
            for k in keys if before.get(k) != after.get(k))

    def verify(self) -> str:
        problems: list[str] = []
        notes: list[str] = []

        after_sql = self._sqlite_snapshot()
        if self.sqlite_before is not None and after_sql is not None:
            b = dict(self.sqlite_before)
            a = dict(after_sql)
            sv_b = b.pop("__schema_version__", None)
            sv_a = a.pop("__schema_version__", None)
            if b != a:
                changed = {k for k in set(b) | set(a) if b.get(k) != a.get(k)}
                if changed <= LIVE_WRITE_TABLES and _bot_heartbeat_fresh():
                    notes.append(
                        f"sqlite rows changed in live-write tables while the "
                        f"bot's heartbeat is fresh (concurrent live activity, "
                        f"not the smoke) [{self._diff(b, a)}]")
                else:
                    problems.append(f"sqlite rows changed [{self._diff(b, a)}]")
            elif sv_b != sv_a:
                notes.append(
                    f"sqlite schema_version {sv_b} -> {sv_a}: StateManager "
                    "applied its idempotent startup DDL (expected on a "
                    "migration-bearing deploy); row data unchanged")
        elif SQLITE_PATH.exists() and self.sqlite_before is None:
            problems.append("could not baseline sqlite — no-write unverified")

        after_duck = self._duck_snapshot()
        if self.duck_before is not None and after_duck is not None:
            if self.duck_before != after_duck:
                changed = {k for k in set(self.duck_before) | set(after_duck)
                           if self.duck_before.get(k) != after_duck.get(k)}
                if changed <= LIVE_WRITE_TABLES and _bot_heartbeat_fresh():
                    notes.append(
                        f"duckdb rows changed in live-write tables while the "
                        f"bot's heartbeat is fresh (concurrent live activity, "
                        f"not the smoke) "
                        f"[{self._diff(self.duck_before, after_duck)}]")
                else:
                    problems.append(
                        f"duckdb rows changed "
                        f"[{self._diff(self.duck_before, after_duck)}]")

        if self._dv_con is not None:
            try:
                dv = self._dv_con.execute("PRAGMA data_version").fetchone()[0]
                if dv != self._dv_before and not problems:
                    notes.append("sqlite data_version moved (another process "
                                 "committed while the smoke ran; counts unchanged)")
            finally:
                self._dv_con.close()

        WARNINGS.extend(notes)
        if problems:
            raise RuntimeError("SMOKE WROTE TO A DATABASE — " + "; ".join(problems))
        return "row counts unchanged in both DBs" + (
            f" [{'; '.join(notes)}]" if notes else "")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_import_walk() -> str:
    """(a) Import every module under src/ait; collect failures."""
    import ait
    failures: list[tuple[str, str]] = []
    seen = 0

    def _onerror(name: str) -> None:
        e = sys.exc_info()[1]
        failures.append((name, f"{type(e).__name__}: {e}"))

    for info in pkgutil.walk_packages(ait.__path__, prefix="ait.", onerror=_onerror):
        seen += 1
        try:
            importlib.import_module(info.name)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 - collect, don't die
            failures.append((info.name, f"{type(e).__name__}: {e}"))
    if failures:
        lines = "\n".join(f"         - {n}: {err}" for n, err in failures)
        raise RuntimeError(
            f"{len(failures)}/{seen} module(s) failed to import:\n{lines}")
    return f"{seen} modules imported"


def check_load_settings() -> str:
    """(b) The exact call the bot makes at startup, against the real yaml."""
    global _SETTINGS
    from ait.config.settings import load_settings
    s = load_settings("config.yaml")
    for attr in ("api_keys", "ibkr", "notifications", "learning", "logging"):
        getattr(s, attr)
    # R13 (human-factors): this check used to prove only that the yaml
    # PARSED — it asserted no values, so a typo'd key running a silent code
    # default sailed through the deploy smoke. Strict extra="forbid" now
    # rejects unknown keys at load; these sentinels additionally pin the
    # values a silent default would poison worst.
    assert s.positions.max_contracts_per_trade == 1, (
        f"max_contracts_per_trade={s.positions.max_contracts_per_trade} — "
        "expected 1 (sample-building sizing); a default leak here trades 10x")
    assert s.learning.paper_trading_mode is True, (
        "learning.paper_trading_mode is not True — live-only learning "
        "overlays would contaminate the paper sample (R5 class)")
    assert s.trading.mode == "paper", (
        f"trading.mode={s.trading.mode!r} — this box is the PAPER stage")
    _SETTINGS = s
    return "config.yaml parsed + validated + sentinels (1-lot, paper-mode)"


def check_trading_cycle_dryrun() -> str:
    """(b2) EXECUTE the entry pipeline: one _trading_cycle + one fast monitor.

    THE gate. On 2026-08 a one-line AttributeError in _trading_cycle ran live
    for three days while check_import_walk and check_load_settings stayed
    green — because importing a module and parsing a config never execute the
    loop. This check does: it builds a REAL TradingOrchestrator through the
    REAL __init__ (real settings, real risk manager, real circuit breaker,
    real state layer, real executor, real portfolio) with fakes only at the
    I/O boundary, and awaits one full scan cycle, one 30-second monitor tick,
    and one entry decision (_try_execute on a canonical iron condor — the
    live models legitimately veto the synthetic candidate inside the cycle,
    so the entry path needs driving explicitly to be covered here).

    Two things are asserted, and both are needed:

      - NOTHING RAISED. Any AttributeError/KeyError/TypeError anywhere in the
        cycle fails the deploy.
      - NOTHING WAS SWALLOWED, and real work happened. Every hot-path
        function catches its own exceptions by design, so "it returned" is
        not evidence: a cycle that dies in its first try/except returns
        normally. The rig's log spy fails on any *_failed / *_error event,
        and the stage markers below prove the cycle reached the scan.

    Safety: runs entirely inside a throwaway temp directory (cwd is switched
    for the duration and restored in a finally), so every cwd-relative
    artefact the orchestrator creates — state DB, historical DB, duckdb
    store, learning JSON — lands there and NEVER next to the live bot's
    data/. No IBKR connection, no network, no notifications, no order.
    """
    import asyncio
    import shutil
    import tempfile

    try:
        # Shared with tests/test_hot_path_smoke.py on purpose: a gate built on
        # its own private fakes proves something different from what the suite
        # proves, and the two would drift.
        from tests.test_hot_path_smoke import (
            build_smoke_orchestrator,
            iron_condor_signal,
        )
    except ImportError as e:
        raise RuntimeError(
            f"cannot import the hot-path rig from tests/ ({e}) — the deploy "
            "gate needs the tests tree and pytest installed "
            "(pip install -e .[dev])"
        ) from e

    async def _drive(rig) -> bool:
        await rig.orch._trading_cycle()
        await rig.orch._monitor_positions_fast()
        # Entry decision path: every risk gate, sizing, and the post-execute
        # bookkeeping block. execute_signal is stubbed by the rig, so an
        # approval builds an order object and places NOTHING.
        handled = await rig.orch._try_execute(iron_condor_signal(), 0.72, None, None)
        await rig.drain_notifications()
        return handled

    cwd = os.getcwd()
    # mkdtemp + best-effort rmtree, NOT TemporaryDirectory: the orchestrator's
    # SQLite/DuckDB handles stay open, and on Windows that makes the context
    # manager's cleanup raise PermissionError — failing the deploy on a
    # janitorial detail after the cycle itself passed.
    sandbox = tempfile.mkdtemp(prefix="ait_smoke_cycle_")
    rig = None
    try:
        os.chdir(sandbox)
        try:
            rig = build_smoke_orchestrator(config_path=ROOT / "config.yaml")
            handled = asyncio.run(_drive(rig))
            if not isinstance(handled, bool):
                raise RuntimeError(
                    f"_try_execute returned {handled!r}, not a bool — its "
                    "handled/rejected contract drives the fall-through to the "
                    "next-ranked strategy")

            swallowed = rig.log.failures()
            if swallowed:
                raise RuntimeError(
                    "the trading cycle degraded SILENTLY (caught its own "
                    "exception and kept going): "
                    + "; ".join(f"{ev} {kw}" for _lvl, ev, kw in swallowed))

            events = rig.log.names()
            # Stage markers — a cycle that returned early at the circuit
            # breaker, the close-to-close guard, or the learning hour block
            # would otherwise pass this gate looking healthy.
            for marker, meaning in (
                ("capital_tier_active", "reached the account/universe stage"),
                ("ml_prediction", "reached ML scoring inside _scan_symbol"),
                ("scan_symbol_timing", "completed a full chain/strategy pass"),
            ):
                if marker not in events:
                    raise RuntimeError(
                        f"cycle never {meaning} (no {marker!r} event) — it "
                        f"returned early; saw: {sorted(set(events))}")
            if not rig.chains.requested:
                raise RuntimeError("no options chain was ever requested — the "
                                   "scan stopped before strategy generation")
            # _try_execute must have reached RISK VALIDATION: it either
            # approved (order built) or the risk manager rejected it. Anything
            # else means a pre-gate short-circuited the entry path.
            if not rig.executed and "trade_rejected" not in events:
                raise RuntimeError(
                    "_try_execute never reached risk validation — an entry "
                    f"gate short-circuited it; orchestrator saw: "
                    f"{sorted(set(events))}")
            from ait.utils.time import now_et
            sod_key = f"mtm_sod_{now_et().date().isoformat()}"
            if not rig.orch._state.get_state(sod_key, ""):
                raise RuntimeError(
                    "the fast monitor's mark-to-market daily-loss brake never "
                    "evaluated — it failed inside its own exception handler")

            return (f"cycle+monitor executed clean: {len(events)} orchestrator "
                    f"events, {len(rig.chains.requested)} chain fetch(es), "
                    f"{len(rig.executed)} order(s) built (none placed)")
        finally:
            if rig is not None:
                rig.restore()
    finally:
        os.chdir(cwd)
        shutil.rmtree(sandbox, ignore_errors=True)


def check_state_manager() -> str:
    """(c) StateManager against the REAL DB — the 07-10 failure class."""
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"{SQLITE_PATH} missing — refusing to create it (smoke is "
            "read-only; a missing state DB post-deploy is itself an incident)")
    from ait.bot.state import StateManager
    sm = StateManager(SQLITE_PATH)

    open_trades = sm.get_open_trades()          # maps every non-closed row
    for t in open_trades:                        # touch what the loop touches
        json.loads(t.legs or "[]")
        float(t.entry_price)
        int(t.quantity)
        str(t.status)

    stats = sm.get_daily_stats(date.today())
    assert stats.date, "get_daily_stats returned a record with no date"

    recent = sm.get_recent_trades(20)            # maps 20 most recent rows
    for t in recent:
        json.loads(t.legs or "[]")

    comm = sm.total_commission("x")              # unknown id => clean 0.0
    assert isinstance(comm, float), f"total_commission returned {type(comm)}"

    ctx_note = "no trades in DB"
    if recent:
        newest = recent[0]
        ctx = sm.get_trade_context(newest.trade_id)
        ctx_note = (f"context of newest trade "
                    f"{'present' if ctx else 'absent (ok)'}")
    return (f"{len(open_trades)} open + {len(recent)} recent rows mapped, "
            f"daily_stats {stats.date}, {ctx_note}")


def check_analytics_duckdb() -> str:
    """(d1) Primary DuckDB analytics path."""
    from ait.monitoring.analytics import TradeAnalytics
    a = TradeAnalytics(SQLITE_PATH)
    if a._duck is None:
        raise RuntimeError("DuckDBAnalytics failed to initialize — the "
                           "primary analytics path is dead (check duckdb "
                           "install / data/ait_analytics.duckdb)")
    snap = a._duck.get_performance(lookback_days=7)   # duckdb engine directly
    a.get_performance(lookback_days=7)                # full wrapper path
    return f"duckdb 7d: {snap.total_trades} trades, pnl ${snap.total_pnl:+,.0f}"


def check_analytics_sqlite_fallback() -> str:
    """(d2) The fallback the bot silently drops to when DuckDB breaks."""
    from ait.monitoring.analytics import TradeAnalytics
    a = TradeAnalytics(SQLITE_PATH)
    a._duck = None  # force the SQLite branch
    p = a.get_performance(lookback_days=7)
    assert p.total_trades >= 0
    return f"sqlite 7d: {p.total_trades} trades, pnl ${p.total_pnl:+,.0f}"


def _run_with_alert_captured(fn_name: str) -> list[str]:
    """Call an orchestrator digest with _alert redirected to a collector.

    These functions swallow their own exceptions (by design — a broken digest
    must not kill the supervisor), so 'no alert collected' is the failure
    signal here, not an exception.
    """
    from ait.orchestration import master
    sent: list[str] = []
    orig = master._alert
    master._alert = lambda msg: sent.append(str(msg))
    try:
        getattr(master, fn_name)()
    finally:
        master._alert = orig
    if not sent:
        raise RuntimeError(
            f"{fn_name} produced no alert — it swallowed an internal error "
            "(see logs/orchestrator.log for the *_failed warning)")
    return sent


def check_daily_digest() -> str:
    """(e1) daily_digest reads the real DB; alert captured locally."""
    sent = _run_with_alert_captured("daily_digest")
    return sent[0][:100]


def check_weekly_scorecard() -> str:
    """(e2) weekly_scorecard reads the real DB; alert captured locally."""
    sent = _run_with_alert_captured("weekly_scorecard")
    return sent[0].splitlines()[0][:100]


def check_counterfactual_state() -> str:
    """(f1) Deserialize the real data/counterfactual_log.json."""
    from ait.learning.counterfactual import CounterfactualTracker
    t = CounterfactualTracker()
    assert isinstance(t._skipped, list)
    return f"{len(t._skipped)} skipped-trade records loaded"


def check_thompson_state() -> str:
    """(f2) Deserialize the real data/thompson_state.json + sample."""
    from ait.strategies.thompson import ThompsonSampler
    s = ThompsonSampler()
    assert isinstance(s._arms, dict)
    ranked = s.rank_strategies(list(s._arms) or ["iron_condor"])
    assert ranked, "rank_strategies returned nothing"
    return f"{len(s._arms)} arms loaded, best={ranked[0]}"


def check_analytics_store_consistency() -> str:
    """(d3) Cross-check the dual-write stores: DuckDB mirrors SQLite closes.

    Found live on 2026-07-11: tests/test_dynamic_exits.py isolates the
    SQLite side in tmp_path but StateManager._init_duckdb() still opens the
    cwd-relative PRODUCTION data/ait_analytics.duckdb, so every local pytest
    run dual-writes fixture trades (T-ML-TRAIN-*, ...) into the real
    analytics store, inflating every DuckDB-path metric. Divergence is
    reported as a loud warning (pre-existing data damage, and this smoke
    never writes/repairs); flip WARN to raise once the store is purged.
    """
    con = sqlite3.connect(f"file:{SQLITE_PATH.as_posix()}?mode=ro", uri=True)
    try:
        sq_ids = {r[0] for r in con.execute(
            "SELECT trade_id FROM trades WHERE status='closed'").fetchall()}
    finally:
        con.close()
    import duckdb
    dcon = duckdb.connect(str(DUCK_PATH), read_only=True)
    try:
        duck_ids = {r[0] for r in dcon.execute(
            "SELECT trade_id FROM trades WHERE status='closed'").fetchall()}
    finally:
        dcon.close()
    extra = sorted(duck_ids - sq_ids)
    missing = sorted(sq_ids - duck_ids)
    if extra or missing:
        # R11: HARD FAIL — the fixture-pollution root cause is fixed
        # (StateManager derives the duck path from db_path) and the store was
        # purged/aligned on 2026-07-11; any divergence now is a real sync bug.
        sample = ", ".join(extra[:5] + missing[:2])
        raise AssertionError(
            f"ANALYTICS STORES DIVERGED: duckdb has {len(duck_ids)} closed "
            f"trades vs sqlite {len(sq_ids)} ({len(extra)} extra in duckdb, "
            f"{len(missing)} missing; e.g. {sample})")
    return (f"duckdb {len(duck_ids)} vs sqlite {len(sq_ids)} closed trades"
            + (" (DIVERGED — see warning)" if extra or missing else " (in sync)"))


def check_iron_condor_width() -> str:
    """(g) Wing sizing with and without the per-trade risk budget."""
    from ait.strategies.iron_condor import IronCondor
    ic = IronCondor()

    ic.risk_budget = None
    w_free = ic._vol_scaled_width(100.0, None, None, None)
    assert math.isfinite(w_free) and w_free >= 2.0, \
        f"unbudgeted width {w_free!r} violates the $2 floor"

    ic.risk_budget = 147.0  # launch-scale per-trade budget
    w_capped = ic._vol_scaled_width(100.0, None, None, None)
    assert math.isfinite(w_capped) and 1.0 <= w_capped <= w_free, \
        f"budget cap broken: capped {w_capped!r} vs free {w_free!r}"
    return f"width ${w_free:.2f} free -> ${w_capped:.2f} under $147 budget"


# ---------------------------------------------------------------------------
# Optional Telegram verdict
# ---------------------------------------------------------------------------

def _send_telegram(verdict: str) -> None:
    try:
        import asyncio
        from ait.config.settings import load_settings
        from ait.notifications.telegram import TelegramNotifier
        s = _SETTINGS or load_settings("config.yaml")
        notifier = TelegramNotifier(
            s.api_keys.telegram_bot_token, s.api_keys.telegram_chat_id)
        ok = asyncio.run(notifier.send(f"DEPLOY SMOKE: {verdict}"))
        print(f"[info] telegram verdict {'sent' if ok else 'NOT sent (unconfigured or send failed)'}")
    except Exception as e:  # noqa: BLE001 - never let notify break the verdict
        print(f"[warn] telegram send failed: {_fmt_exc(e)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-deploy smoke: executes the bot's runtime paths "
                    "read-only against the real config and DBs.")
    parser.add_argument("--telegram", action="store_true",
                        help="send the verdict via ait.notifications (default off)")
    args = parser.parse_args(argv)

    _bootstrap()
    print(f"smoke_deploy: repo={ROOT}")

    guard = NoWriteGuard()

    _run("import_walk(src/ait)", check_import_walk)
    _run("load_settings(config.yaml)", check_load_settings)
    _run("orchestrator.trading_cycle(dryrun)", check_trading_cycle_dryrun)
    _run("state_manager(data/ait_state.db)", check_state_manager)
    _run("analytics.get_performance(duckdb)", check_analytics_duckdb)
    _run("analytics.get_performance(sqlite_fallback)", check_analytics_sqlite_fallback)
    _run("analytics.store_consistency(duckdb_vs_sqlite)", check_analytics_store_consistency)
    _run("master.daily_digest(alert_captured)", check_daily_digest)
    _run("master.weekly_scorecard(alert_captured)", check_weekly_scorecard)
    _run("counterfactual_tracker.state_load", check_counterfactual_state)
    _run("thompson_sampler.state_load", check_thompson_state)
    _run("iron_condor.vol_scaled_width", check_iron_condor_width)
    _run("no_writes(both DBs)", guard.verify)

    for w in WARNINGS:
        print(f"[warn] {w}")

    failures = [(n, e) for n, e in RESULTS if e]
    total = len(RESULTS)
    if not failures:
        verdict = f"SMOKE PASS ({total} checks)"
        print(verdict)
        if args.telegram:
            _send_telegram(verdict)
        return 0

    print()
    print(f"SMOKE FAIL ({len(failures)}/{total} checks failed):")
    for name, err in failures:
        print(f"  * {name}: {err}")
    if args.telegram:
        _send_telegram(f"FAIL ({len(failures)}/{total}): "
                       + "; ".join(n for n, _ in failures))
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:  # argparse --help / explicit exits pass through
        raise
    except KeyboardInterrupt:
        rc = 130
    except BaseException:  # noqa: BLE001 - harness crash must not exit 0
        traceback.print_exc()
        print("SMOKE FAIL (harness crashed)")
        rc = 2
    sys.exit(rc)
