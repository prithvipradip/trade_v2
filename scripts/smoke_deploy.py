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
  3.  StateManager on the real data/ait_state.db  (row->dataclass mapping of
      every open + recent trade, daily stats, trade context, commissions)
  4.  TradeAnalytics.get_performance(7)           (DuckDB path AND the forced
      SQLite fallback path)
  5.  orchestrator daily_digest/weekly_scorecard  (real-DB read paths; _alert
      is monkeypatched to a local collector — nothing is sent to Telegram)
  6.  CounterfactualTracker / ThompsonSampler     (learning-state loads)
  7.  IronCondor._vol_scaled_width               (wing sizing with and
      without a risk budget)

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
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "data" / "ait_state.db"
DUCK_PATH = ROOT / "data" / "ait_analytics.duckdb"

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
                problems.append(
                    f"duckdb rows changed [{self._diff(self.duck_before, after_duck)}]")

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
    _SETTINGS = s
    return "config.yaml parsed + validated"


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
