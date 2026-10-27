"""W105: the executions ledger tells the truth; one default IB port; the digest marker uses the ET date.

Real code throughout: the real `_sweep_executions` over a real `StateManager`, the real referee
script in a subprocess, the real restatement script on a copy of an old-schema database, the real
supervisor functions. Fakes only at the broker edge (the fills ib_insync would hand back).

The operator also trades by hand in the same account. Those fills (TWS: clientId 0, no order
reference, order id 0) and the probes' fills have no bot trade by design; they must stay
unattributed, must never reach a bot trade's commission, and must not raise a false alarm.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ait.bot.state import StateManager
from ait.execution.executor import BAG_CON_ID, TradeExecutor
from ait.orchestration import master as master_mod
from ait.utils.time import ET, now_et

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# rigs
# --------------------------------------------------------------------------
def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed_trade(st, trade_id="T-1", symbol="QQQ", status="filled"):
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, entry_time, "
            "entry_price, quantity, contract_type) VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, symbol, "iron_condor", "short", status, now_et().isoformat(), 1.2))


def _fill(exec_id, *, order_id=0, perm_id=0, client_id=1, order_ref="", con_id=111,
          sec_type="OPT", symbol="QQQ", price=1.0, side="SLD", commission=0.65,
          exec_time=None, fill_time=None):
    exec_time = exec_time or datetime(2026, 9, 28, 17, 54, 15, tzinfo=timezone.utc)
    return SimpleNamespace(
        execution=SimpleNamespace(execId=exec_id, orderId=order_id, permId=perm_id,
                                  clientId=client_id, orderRef=order_ref, side=side,
                                  shares=1.0, price=price, time=exec_time),
        commissionReport=(None if commission is None else
                          SimpleNamespace(commission=commission, realizedPNL=0.0)),
        contract=SimpleNamespace(conId=con_id, secType=sec_type, symbol=symbol),
        time=fill_time if fill_time is not None else exec_time,
    )


def _executor(st, *, fills, trades=(), order_trade_map=None):
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = st
    ex._ibkr = SimpleNamespace(ib=SimpleNamespace(trades=lambda: list(trades),
                                                  fills=lambda: list(fills)))
    ex._order_trade_map = dict(order_trade_map or {})
    ex._pending_orders = {}
    ex._pending_exit_orders = {}
    ex._order_ctx_map = {}
    ex._order_perm_map = {}
    ex._perm_trade_map = {}
    return ex


def _rows(st) -> dict[str, dict]:
    with sqlite3.connect(st._db_path) as conn:
        conn.row_factory = sqlite3.Row
        return {r["exec_id"]: dict(r) for r in conn.execute("SELECT * FROM executions")}


@pytest.fixture
def berlin_host():
    """The host zone the 2026-09-28 fill was swept in (CEST, UTC+2)."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Berlin"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


# ==========================================================================
# A. the fill time
# ==========================================================================
class TestFillTime:
    def test_a_live_fill_stores_fill_time_not_the_mis_zoned_execution_time(self, tmp_path, berlin_host):
        """ib_insync converts the gateway's zone-less time with the HOST zone, so
        Execution.time of a live fill was 2 h early; Fill.time is the arrival time."""
        st = _state(tmp_path)
        mis_zoned = datetime(2026, 9, 28, 15, 54, 15, tzinfo=timezone.utc)
        true_time = datetime(2026, 9, 28, 17, 54, 16, tzinfo=timezone.utc)
        f = _fill("E1", order_id=7, perm_id=70, exec_time=mis_zoned, fill_time=true_time)
        _executor(st, fills=[f], order_trade_map={7: "T-1"})._sweep_executions()
        assert _rows(st)["E1"]["exec_time"] == "2026-09-28T17:54:16+00:00"

    def test_a_replayed_fill_is_unchanged(self, tmp_path, berlin_host):
        st = _state(tmp_path)
        t = datetime(2026, 9, 29, 14, 27, 39, tzinfo=timezone.utc)
        _executor(st, fills=[_fill("E2", client_id=0, exec_time=t, fill_time=t)])._sweep_executions()
        assert _rows(st)["E2"]["exec_time"] == "2026-09-29T14:27:39+00:00"

    def test_a_fill_without_a_fill_time_falls_back_to_the_execution_time(self, tmp_path):
        st = _state(tmp_path)
        f = _fill("E3", client_id=0)
        f.time = None
        _executor(st, fills=[f])._sweep_executions()
        assert _rows(st)["E3"]["exec_time"] == "2026-09-28T17:54:15+00:00"


# ==========================================================================
# B/C. what reaches the table, and who placed it
# ==========================================================================
class TestEveryAccountFillIsRecorded:
    def test_a_fill_with_no_trade_object_is_written(self, tmp_path):
        """The 11 manual fills of 09-28 and 09-30 were replayed without a Trade and never seen."""
        st = _state(tmp_path)
        f = _fill("M1", order_id=0, perm_id=999, client_id=0, symbol="QQQ")
        _executor(st, fills=[f], trades=[])._sweep_executions()
        row = _rows(st)["M1"]
        assert row["symbol"] == "QQQ" and row["trade_id"] == ""

    def test_a_late_commission_report_updates_the_row(self, tmp_path):
        st = _state(tmp_path)
        f = _fill("M2", client_id=0, commission=None)
        ex = _executor(st, fills=[f])
        ex._sweep_executions()
        assert _rows(st)["M2"]["commission"] is None            # not yet reported
        f.commissionReport = SimpleNamespace(commission=1.3, realizedPNL=0.0)
        ex._sweep_executions()
        assert _rows(st)["M2"]["commission"] == pytest.approx(1.3)

    def test_the_trade_still_supplies_ids_symbol_and_combo_context(self, tmp_path):
        """A fill that has a Trade keeps the Trade's order id (the pre-W105 behaviour)."""
        st = _state(tmp_path)
        bag = _fill("B1", order_id=0, perm_id=0, con_id=BAG_CON_ID, sec_type="BAG", price=-1.2)
        trade = SimpleNamespace(order=SimpleNamespace(orderId=555, permId=99123),
                                contract=SimpleNamespace(symbol="SPY", secType="BAG"),
                                fills=[bag])
        ex = _executor(st, fills=[bag], trades=[trade], order_trade_map={555: "T-1"})
        ex._order_ctx_map = {555: (1.2, 1.2, 0.1)}
        ex._sweep_executions()
        row = _rows(st)["B1"]
        assert (row["order_id"], row["trade_id"], row["symbol"]) == (555, "T-1", "SPY")
        assert row["price"] == pytest.approx(1.2) and row["live_mid"] == pytest.approx(1.2)


    def test_the_trades_own_order_id_wins_over_the_executions(self, tmp_path):
        """Pre-W105 precedence, pinned: a fill that has a Trade is keyed by the Trade's ids."""
        st = _state(tmp_path)
        _seed_trade(st, "T-1")
        f = _fill("K1", order_id=4242, perm_id=424200, client_id=1)
        trade = SimpleNamespace(order=SimpleNamespace(orderId=555, permId=55500),
                                contract=SimpleNamespace(symbol="QQQ", secType="OPT"), fills=[f])
        _executor(st, fills=[f], trades=[trade], order_trade_map={555: "T-1"})._sweep_executions()
        row = _rows(st)["K1"]
        assert (row["order_id"], row["perm_id"], row["trade_id"]) == (555, 55500, "T-1")


class TestManualAndProbeFillsAreNotTheBots:
    def test_a_manual_fill_stays_unattributed_even_with_a_bot_trade_open_on_the_symbol(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-1", "QQQ")
        f = _fill("M3", order_id=0, perm_id=555, client_id=0, symbol="QQQ", commission=2.08)
        _executor(st, fills=[f])._sweep_executions()
        row = _rows(st)["M3"]
        assert (row["trade_id"], row["origin"], row["client_id"]) == ("", "manual", 0)
        assert st.total_commission("T-1") == 0.0                 # never the bot trade's cost

    def test_a_probe_fill_is_other(self, tmp_path):
        st = _state(tmp_path)
        f = _fill("P1", order_id=2276, perm_id=1779141650, client_id=97,
                  order_ref="W53PROBE-open-1790698798")
        _executor(st, fills=[f])._sweep_executions()
        row = _rows(st)["P1"]
        assert (row["trade_id"], row["origin"], row["order_ref"]) == ("", "other", "W53PROBE-open-1790698798")

    def test_an_unattributed_bot_fill_is_still_a_real_orphan(self, tmp_path):
        st = _state(tmp_path)
        f = _fill("O1", order_id=31, perm_id=310, client_id=1, order_ref="")
        _executor(st, fills=[f])._sweep_executions()
        row = _rows(st)["O1"]
        assert (row["trade_id"], row["origin"]) == ("", "bot")

    def test_origin_never_downgrades_from_bot_and_upgrades_to_it(self, tmp_path):
        st = _state(tmp_path)
        kw = dict(order_id=1, perm_id=1, symbol="SPY", con_id=1, side="BOT", shares=1.0,
                  price=1.0, exec_time="2026-10-01T00:00:00+00:00")
        st.record_execution("X1", trade_id="T-1", origin="bot", **kw)
        st.record_execution("X1", trade_id="", origin="manual", **kw)
        assert _rows(st)["X1"]["origin"] == "bot"
        st.record_execution("X2", trade_id="", origin="manual", **kw)
        st.record_execution("X2", trade_id="T-1", origin="bot", **kw)
        assert _rows(st)["X2"]["origin"] == "bot"

    def test_the_first_client_id_and_order_ref_are_kept(self, tmp_path):
        st = _state(tmp_path)
        kw = dict(order_id=1, perm_id=1, trade_id="", symbol="SPY", con_id=1, side="BOT",
                  shares=1.0, price=1.0, exec_time="2026-10-01T00:00:00+00:00")
        st.record_execution("X3", client_id=97, order_ref="W53PROBE-x", origin="other", **kw)
        st.record_execution("X3", client_id=5, order_ref="other-ref", origin="other", **kw)
        r = _rows(st)["X3"]
        assert (r["client_id"], r["order_ref"]) == (97, "W53PROBE-x")


class TestAttributionSurvivesARestart:
    """The in-memory order maps are empty after a restart; orderRef == trade_id is not."""

    def test_a_bot_fill_resolves_through_its_order_ref(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-20260928-195224-adfe01", "QQQ")
        f = _fill("R1", order_id=0, perm_id=0, client_id=1, order_ref="T-20260928-195224-adfe01")
        _executor(st, fills=[f])._sweep_executions()
        row = _rows(st)["R1"]
        assert (row["trade_id"], row["origin"]) == ("T-20260928-195224-adfe01", "bot")
        assert st.total_commission("T-20260928-195224-adfe01") == pytest.approx(0.65)

    def test_a_roll_ref_books_to_the_closing_trade_like_the_in_memory_path(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-A", "SPY")
        f = _fill("R2", client_id=1, order_ref="ROLL:T-A:T-B")
        _executor(st, fills=[f])._sweep_executions()
        assert _rows(st)["R2"]["trade_id"] == "T-A"

    def test_an_unknown_ref_attributes_to_nothing(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-1")
        for i, ref in enumerate(("T-NOPE", "W53PROBE-open-1", "ROLL:T-X:T-Y")):
            _executor(st, fills=[_fill(f"U{i}", client_id=97, order_ref=ref)])._sweep_executions()
        assert {r["trade_id"] for r in _rows(st).values()} == {""}

    def test_the_in_memory_maps_still_win_when_they_know_the_order(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-1")
        _seed_trade(st, "T-2")
        f = _fill("W1", order_id=9, perm_id=90, client_id=1, order_ref="T-2")
        _executor(st, fills=[f], order_trade_map={9: "T-1"})._sweep_executions()
        assert _rows(st)["W1"]["trade_id"] == "T-1"


class TestOriginClassification:
    @pytest.mark.parametrize("trade_id,client_id,ref,expected", [
        ("T-1", 1, "T-1", "bot"),
        ("T-1", 0, "", "bot"),              # an attributed fill is the bot's whatever the client
        ("", 0, "", "manual"),
        ("", 0, "SOMETHING", "manual"),
        ("", 97, "W53PROBE-1", "other"),
        ("", 1, "", "bot"),
        ("", None, "", "bot"),
    ])
    def test_the_rule(self, trade_id, client_id, ref, expected):
        assert TradeExecutor._execution_origin(trade_id, client_id, ref) == expected


# ==========================================================================
# the old-schema database
# ==========================================================================
OLD_EXECUTIONS = """CREATE TABLE executions (
    exec_id TEXT PRIMARY KEY, order_id INTEGER, perm_id INTEGER, trade_id TEXT, symbol TEXT,
    con_id INTEGER, side TEXT, shares REAL, price REAL, exec_time TEXT,
    commission REAL DEFAULT 0, realized_pnl REAL DEFAULT 0, signal_price REAL DEFAULT 0,
    live_mid REAL DEFAULT 0, nbbo_spread REAL DEFAULT 0)"""


def _columns(path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return [r[1] for r in conn.execute("PRAGMA table_info(executions)")]


class TestMigration:
    def test_an_old_database_gains_the_columns_with_its_data_intact(self, tmp_path):
        path = tmp_path / "ait_state.db"
        with sqlite3.connect(path) as conn:
            conn.execute(OLD_EXECUTIONS)
            conn.execute("INSERT INTO executions (exec_id, trade_id, symbol, price, exec_time) "
                         "VALUES ('OLD1','T-1','SPY',1.5,'2026-09-14T19:00:29+00:00')")
        assert "origin" not in _columns(path)
        StateManager(db_path=path)
        assert {"client_id", "order_ref", "origin"} <= set(_columns(path))
        StateManager(db_path=path)                                  # idempotent
        row = _rows(StateManager(db_path=path))["OLD1"]
        assert row["price"] == 1.5 and row["origin"] is None


# ==========================================================================
# the shadow referee and the digest
# ==========================================================================
def _referee(db: Path) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "shadow_referee.py"),
                          "--db", str(db), "--mirror", str(db.parent / "none.duckdb"), "--json"],
                         capture_output=True, text=True, cwd=ROOT)
    return {c["check"]: c for c in json.loads(out.stdout)["checks"]}


class TestRefereeOrphanCheck:
    def _add(self, st, exec_id, origin, trade_id=""):
        st.record_execution(exec_id, order_id=0, perm_id=1, trade_id=trade_id, symbol="QQQ",
                            con_id=1, side="BOT", shares=1.0, price=1.0,
                            exec_time="2026-09-29T14:27:39+00:00", origin=origin)

    def test_manual_and_probe_fills_are_not_a_break(self, tmp_path):
        st = _state(tmp_path)
        self._add(st, "M", "manual")
        self._add(st, "P", "other")
        assert _referee(st._db_path)["orphan_fills"]["status"] == "PASS"

    def test_an_unattributed_bot_fill_is_a_break(self, tmp_path):
        st = _state(tmp_path)
        self._add(st, "B", "bot")
        assert _referee(st._db_path)["orphan_fills"]["status"] == "BREAK"

    def test_a_legacy_row_with_no_origin_is_still_counted(self, tmp_path):
        st = _state(tmp_path)
        self._add(st, "L", None)
        assert _referee(st._db_path)["orphan_fills"]["status"] == "BREAK"

    def test_an_old_database_without_the_column_is_still_checked(self, tmp_path):
        path = tmp_path / "old.db"
        with sqlite3.connect(path) as conn:
            conn.execute(OLD_EXECUTIONS)
            conn.execute("CREATE TABLE trades (trade_id TEXT, status TEXT, symbol TEXT, strategy TEXT, "
                         "direction TEXT, entry_time TEXT, exit_time TEXT, entry_price REAL, "
                         "exit_price REAL, quantity INTEGER, realized_pnl REAL, commission REAL, "
                         "contract_type TEXT, exit_reason_detailed TEXT, capital_at_risk REAL)")
            conn.execute("CREATE TABLE open_positions (trade_id TEXT)")
            conn.execute("CREATE TABLE bot_state (key TEXT, value TEXT)")
            conn.execute("INSERT INTO executions (exec_id, trade_id) VALUES ('Z','')")
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "shadow_referee.py"), "--db",
                              str(path), "--mirror", str(tmp_path / "n.duckdb"), "--json"],
                             capture_output=True, text=True, cwd=ROOT)
        checks = {c["check"]: c for c in json.loads(res.stdout)["checks"]}
        assert checks["orphan_fills"]["status"] == "BREAK"


class TestScorecardSplitsBotAndOtherCommissions:
    def test_the_executions_line(self, tmp_path, monkeypatch):
        st = _state(tmp_path)
        _seed_trade(st, "T-1")
        kw = dict(order_id=1, perm_id=1, symbol="SPY", con_id=1, side="BOT", shares=1.0,
                  price=1.0, exec_time="2026-10-01T00:00:00+00:00")
        st.record_execution("D1", trade_id="T-1", commission=2.50, origin="bot", **kw)
        st.record_execution("D2", trade_id="", commission=12.45, origin="other", **kw)
        st.record_execution("D3", trade_id="", commission=5.23, origin="manual", **kw)
        monkeypatch.setattr(master_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(master_mod, "_mirror_content_age_hours", lambda: 1.0)
        sent: list[str] = []
        monkeypatch.setattr(master_mod, "_alert", lambda m: sent.append(m) or True)
        master_mod.weekly_scorecard()
        line = next(p for m in sent for p in m.split("\n") if p.startswith("executions:"))
        assert "$2.50 commissions (bot)" in line and "$17.68 manual/other" in line


# ==========================================================================
# D. one default IB port
# ==========================================================================
class TestOneDefaultPort:
    def test_a_missing_or_empty_ibkr_port_falls_back_to_the_bots_own_default(self, monkeypatch):
        from ait.config.settings import IBKREnvConfig
        bot_default = int(IBKREnvConfig.model_fields["ibkr_port"].default)
        monkeypatch.delenv("IBKR_PORT", raising=False)
        assert master_mod._configured_ib_port() == bot_default
        monkeypatch.setenv("IBKR_PORT", "")
        assert master_mod._configured_ib_port() == bot_default

    def test_a_set_value_wins_and_an_explicit_default_is_honoured(self, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "4002")
        assert master_mod._configured_ib_port() == 4002
        monkeypatch.delenv("IBKR_PORT", raising=False)
        assert master_mod._configured_ib_port(7496) == 7496

    def test_the_supervisor_and_the_bot_agree_with_no_env_value(self, monkeypatch):
        from ait.config.settings import IBKREnvConfig
        monkeypatch.delenv("IBKR_PORT", raising=False)
        monkeypatch.setattr(master_mod, "_gateway_listening", lambda port, host="127.0.0.1": True)
        monkeypatch.setattr(master_mod, "_note_ib_endpoint", lambda *a, **k: None)
        assert master_mod._ib_port() == int(IBKREnvConfig.model_fields["ibkr_port"].default)


# ==========================================================================
# E. the digest marker date
# ==========================================================================
class TestDigestMarkerUsesTheEtDate:
    def test_a_digest_sent_after_19_et_is_stamped_with_the_et_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(master_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(master_mod, "_mirror_content_age_hours", lambda: 1.0)
        monkeypatch.setattr(master_mod, "_alert", lambda m: True)
        # A past date, so the host's real "today" can never equal it: 20:30 ET is 02:30 CET the
        # next day (01:30 in the weeks the EU is back on winter time and the US is not).
        late = datetime(2026, 3, 3, 20, 30, tzinfo=ET)
        monkeypatch.setattr(master_mod, "now_et", lambda: late)
        master_mod.daily_digest()
        assert (tmp_path / "digest_sent_20260303").exists()
        assert not (tmp_path / "digest_sent_20260304").exists()
        # the next morning's catch-up sees that no AM digest has gone out for 03-04
        assert master_mod._digest_catchup_due(datetime(2026, 3, 4, 10, 0, tzinfo=ET), tmp_path) is True


# ==========================================================================
# the restatement script
# ==========================================================================
def _load_restate():
    spec = importlib.util.spec_from_file_location("restate_w105", ROOT / "scripts" / "restate_w105_executions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["restate_w105"] = mod
    spec.loader.exec_module(mod)
    return mod


def _audited_db(tmp_path):
    """An old-schema database holding exactly the rows the audit found."""
    r = _load_restate()
    path = tmp_path / "ait_state.db"
    with sqlite3.connect(path) as conn:
        conn.execute(OLD_EXECUTIONS)
        n = 0
        for (oid, perm), (_o, _c, _ref, count) in r.AUDITED.items():
            for _ in range(count):
                n += 1
                conn.execute("INSERT INTO executions (exec_id, order_id, perm_id, trade_id, exec_time) "
                             "VALUES (?,?,?,?,?)", (f"X{n}", oid, perm, "", "2026-09-29T16:00:00+00:00"))
        for eid in r.TIME_FIX_IDS:
            conn.execute("INSERT INTO executions (exec_id, order_id, perm_id, trade_id, exec_time) "
                         "VALUES (?,?,?,?,?)", (eid, 555, 5550, "T-20260928-195224-adfe01", r.WRONG_TIME))
    return r, path


def _dump(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT * FROM executions ORDER BY exec_id").fetchall()


class TestRestatementScript:
    def test_a_dry_run_changes_nothing_and_cannot_write(self, tmp_path):
        r, path = _audited_db(tmp_path)
        before = _dump(path)
        assert r.main([], db_path=path) == 0
        assert _dump(path) == before and "origin" not in _columns(path)
        opened: list[str] = []
        real_connect = sqlite3.connect

        def _spy(target, *a, **k):
            opened.append(str(target))
            return real_connect(target, *a, **k)

        r.sqlite3.connect = _spy
        try:
            assert r.main([], db_path=path) == 0
        finally:
            r.sqlite3.connect = real_connect
        assert opened and all("mode=ro" in o for o in opened)     # a dry run cannot write

    def test_apply_snapshots_tags_the_31_rows_and_fixes_the_5_times(self, tmp_path):
        r, path = _audited_db(tmp_path)
        assert r.main(["--apply"], db_path=path) == 0
        assert list((tmp_path / "backups").glob("ait_state.pre_w105_executions.*.db"))
        with sqlite3.connect(path) as conn:
            tagged = dict(conn.execute("SELECT origin, COUNT(*) FROM executions "
                                       "WHERE trade_id = '' GROUP BY 1").fetchall())
            assert tagged == {"manual": 8, "other": 23}
            assert conn.execute("SELECT COUNT(DISTINCT order_ref) FROM executions "
                                "WHERE origin = 'other'").fetchone()[0] == 7
            assert conn.execute("SELECT COUNT(*) FROM executions WHERE exec_time = ?",
                                (r.RIGHT_TIME,)).fetchone()[0] == 5
            assert conn.execute("SELECT COUNT(*) FROM executions WHERE trade_id = '' "
                                "AND origin = 'manual' AND client_id = 0").fetchone()[0] == 8
            assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 36   # nothing deleted
            assert conn.execute("SELECT COUNT(*) FROM executions WHERE trade_id != '' "
                                "AND origin IS NOT NULL").fetchone()[0] == 0              # bot rows untouched

    def test_a_second_run_is_a_no_op(self, tmp_path):
        r, path = _audited_db(tmp_path)
        r.main(["--apply"], db_path=path)
        after = _dump(path)
        assert r.main(["--apply"], db_path=path) == 0
        assert _dump(path) == after
        assert len(list((tmp_path / "backups").glob("*.db"))) == 1       # no second snapshot

    def test_it_refuses_when_the_database_differs_from_the_audit(self, tmp_path):
        r, path = _audited_db(tmp_path)
        with sqlite3.connect(path) as conn:
            conn.execute("INSERT INTO executions (exec_id, order_id, perm_id, trade_id) "
                         "VALUES ('STRAY', 0, 424242, '')")
        before = _dump(path)
        assert r.main(["--apply"], db_path=path) == 1
        assert _dump(path) == before and not (tmp_path / "backups").exists()

    def test_it_refuses_when_an_audited_order_has_a_different_row_count(self, tmp_path):
        r, path = _audited_db(tmp_path)
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM executions WHERE rowid = (SELECT MIN(rowid) FROM executions "
                         "WHERE order_id = 5910)")
        before = _dump(path)
        assert r.main(["--apply"], db_path=path) == 1
        assert _dump(path) == before

    def test_it_refuses_a_per_order_mismatch_even_when_the_total_still_adds_up(self, tmp_path):
        r, path = _audited_db(tmp_path)
        with sqlite3.connect(path) as conn:
            conn.execute("DELETE FROM executions WHERE rowid = (SELECT MIN(rowid) FROM executions "
                         "WHERE perm_id = 1178063164)")                     # 3 -> 2
            conn.execute("INSERT INTO executions (exec_id, order_id, perm_id, trade_id) "
                         "VALUES ('EXTRA', 0, 1779141601, '')")             # 5 -> 6: still 31 rows
        before = _dump(path)
        assert r.main(["--apply"], db_path=path) == 1
        assert _dump(path) == before

    def test_it_refuses_a_fill_time_it_does_not_recognise(self, tmp_path):
        r, path = _audited_db(tmp_path)
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE executions SET exec_time = '2026-09-28T09:00:00+00:00' WHERE exec_id = ?",
                         (r.TIME_FIX_IDS[0],))
        assert r.main(["--apply"], db_path=path) == 1

    def test_a_database_that_the_new_code_already_migrated_is_handled(self, tmp_path):
        r, path = _audited_db(tmp_path)
        StateManager(db_path=path)                                   # adds the columns
        assert r.main(["--apply"], db_path=path) == 0
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM executions WHERE trade_id = '' "
                                "AND origin IS NULL").fetchone()[0] == 0
