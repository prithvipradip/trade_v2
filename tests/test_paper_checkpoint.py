"""W110: scripts/paper_checkpoint.py - the read-only comparison of paper closes with the walk-forward model.

The rule it implements is pre-registered in PLAN.md ("W110 AS PLANNED"); these tests pin that the script uses
the go-live population definition, the real state schema, the live exit-reason strings, the schedule and the
decision classes exactly as written there. A tmp database is created with the REAL ``StateManager`` schema so
the column contract is checked, not assumed.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("paper_checkpoint", ROOT / "scripts" / "paper_checkpoint.py")
pc = importlib.util.module_from_spec(_spec)
sys.modules["paper_checkpoint"] = pc
_spec.loader.exec_module(pc)

EPOCH = "2026-10-29"
NOW = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)


def _legs(width=3.0, sp=700.0, sc=760.0):
    return json.dumps([
        {"strike": sp - width, "right": "P", "action": "BUY"}, {"strike": sp, "right": "P", "action": "SELL"},
        {"strike": sc, "right": "C", "action": "SELL"}, {"strike": sc + width, "right": "C", "action": "BUY"}])


def make_db(tmp_path, trades):
    """A state DB with the real schema; ``trades`` = list of dict overrides."""
    from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
    db = tmp_path / "ait_state.db"
    st = StateManager(db_path=db)
    for i, t in enumerate(trades):
        tid = t.get("trade_id", f"T-{i}")
        status = t.get("status", "closed")
        st.record_trade(TradeRecord(
            trade_id=tid, symbol=t.get("symbol", "QQQ"), strategy=t.get("strategy", "iron_condor"),
            direction=TradeDirection.SHORT, status=TradeStatus(status if status != "closed" else "filled"),
            entry_time=t.get("entry_time", "2026-11-02T10:00:00-05:00"), entry_price=t.get("entry_price", 0.9),
            quantity=t.get("quantity", 1), contract_type="iron_condor", expiry="2026-11-20",
            legs=t.get("legs", _legs())))
        if status == "closed":
            st.close_trade(tid, exit_price=0.4, realized_pnl=t.get("pnl", 40.0), commission=t.get("commission", 5.2),
                           exit_reason_detailed=t.get("reason", "take_profit_short (P&L: 50.0%, target: 50.0%)"),
                           exit_time=t.get("exit_time", "2026-11-09T11:00:00-05:00"))
        with sqlite3.connect(db) as con:
            con.execute("UPDATE trades SET capital_at_risk=?, rolled_from_trade_id=? WHERE trade_id=?",
                        (t.get("car", 210.0), t.get("rolled_from"), tid))
            if t.get("exec"):
                price, mid = t["exec"]
                con.execute(
                    "INSERT INTO executions (exec_id, order_id, perm_id, trade_id, symbol, con_id, side, shares, "
                    "price, exec_time, live_mid) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (f"E-{tid}", 1, 1, tid, "QQQ", 0, "SLD", 1, price, t.get("entry_time", "2026-11-02T10:00:00-05:00"),
                     mid))
            if status in ("cancelled", "rejected", "pending"):
                con.execute("UPDATE trades SET status=? WHERE trade_id=?", (status, tid))
    return db


class TestTheSchemaContract:
    def test_the_real_schema_has_every_column_the_script_reads(self, tmp_path):
        db = make_db(tmp_path, [])
        with sqlite3.connect(db) as con:
            trades = {r[1] for r in con.execute("pragma table_info(trades)")}
            execs = {r[1] for r in con.execute("pragma table_info(executions)")}
        assert set(pc.TRADE_COLUMNS) <= trades
        assert set(pc.EXECUTION_COLUMNS) <= execs

    def test_the_connection_is_read_only(self, tmp_path):
        db = make_db(tmp_path, [{}])
        con = pc.open_ro(db)
        with pytest.raises(sqlite3.OperationalError):
            con.execute("UPDATE trades SET realized_pnl = 1")
        con.close()

    def test_a_full_run_leaves_the_database_untouched(self, tmp_path):
        db = make_db(tmp_path, [{}, {"trade_id": "T-x", "pnl": -80.0}])
        before = db.read_bytes()
        pc.run(db, "C1", EPOCH, NOW)
        assert db.read_bytes() == before


class TestThePopulation:
    def test_only_closes_entered_in_the_epoch_count_and_adfe01_is_outside(self, tmp_path):
        db = make_db(tmp_path, [
            {"trade_id": "T-20260928-195224-adfe01", "entry_time": "2026-09-28T13:52:24-04:00"},
            {"trade_id": "in-1", "entry_time": "2026-11-02T10:00:00-05:00"},
            {"trade_id": "in-2", "entry_time": "2026-10-29T09:40:00-04:00"}])
        _, info = pc.run(db, "C1", EPOCH, NOW)
        assert info["metrics"]["n"] == 2

    def test_not_real_closes_and_other_strategies_are_excluded(self, tmp_path):
        db = make_db(tmp_path, [
            {"trade_id": "real"},
            {"trade_id": "ghost", "reason": "never_filled"},
            {"trade_id": "unknown", "reason": "reconciler_unknown_exit_expired_needs_review"},
            {"trade_id": "strangle", "strategy": "short_strangle"}])
        _, info = pc.run(db, "C1", EPOCH, NOW)
        assert info["metrics"]["n"] == 1

    def test_epoch_closes_filters_the_strategy_itself(self):
        """The SQL already restricts to iron_condor; the function must not rely on that."""
        rows = [{"trade_id": "a", "rolled_from_trade_id": None, "reason": "", "strategy": "iron_condor",
                 "entry_time": "2026-11-02T10:00:00-05:00"},
                {"trade_id": "b", "rolled_from_trade_id": None, "reason": "", "strategy": "short_strangle",
                 "entry_time": "2026-11-02T10:00:00-05:00"}]
        kept, _ = pc.epoch_closes(rows, EPOCH)
        assert [r["trade_id"] for r in kept] == ["a"]

    def test_a_rolled_chain_is_one_outcome(self, tmp_path):
        db = make_db(tmp_path, [
            {"trade_id": "A", "reason": "rolled_to_B", "pnl": -30.0},
            {"trade_id": "B", "rolled_from": "A", "pnl": 50.0}])
        _, info = pc.run(db, "C1", EPOCH, NOW)
        assert info["metrics"]["n"] == 1 and info["metrics"]["net"] == 20.0

    def test_the_population_equals_the_go_live_real_close_count(self, tmp_path):
        from ait.reporting.go_live import real_close_count
        db = make_db(tmp_path, [{"trade_id": "a"}, {"trade_id": "b", "reason": "never_filled"}, {"trade_id": "c"}])
        _, info = pc.run(db, "C1", "2026-01-01", NOW)
        assert info["metrics"]["n"] == real_close_count(db) == 2


class TestExitMapping:
    @pytest.mark.parametrize("reason, expected", [
        ("short_strike_touch (spot 705.10 <= short put 707.00) [fresh quote, 2 agreeing ticks]", "touch_stop"),
        ("short_strike_touch (spot 770.00 >= short call 769.00)", "touch_stop"),
        ("take_profit_short (P&L: 52.0%, target: 50.0%)", "take_profit"),
        ("take_profit (P&L: 80.0%, target: 75.0%)", "take_profit"),
        ("expiry_approaching (DTE: 1)", "expiry_approach"),
        ("reconciler_expired_intrinsic", "expiry_settled"),
        ("stop_loss (P&L: -50.0%)", "other"),
        ("condor_defense_close", "other"),
        ("", "other"),
        (None, "other"),
    ])
    def test_the_live_reason_strings(self, reason, expected):
        assert pc.classify_exit(reason) == expected

    def test_the_mapping_covers_the_strings_the_live_code_emits(self):
        src = (ROOT / "src" / "ait" / "execution" / "portfolio.py").read_text()
        assert "short_strike_touch" in src and "take_profit_short" in src and "expiry_approaching" in src


class TestMetrics:
    def closes(self, pnls):
        return [{"realized_pnl": p, "entry_price": 0.9, "legs": _legs(), "entry_time": "2026-11-02T10:00:00-05:00",
                 "exit_time": "2026-11-10T10:00:00-05:00", "capital_at_risk": 210.0, "reason": "take_profit_short",
                 "trade_id": f"t{i}", "quantity": 1, "commission": 5.2} for i, p in enumerate(pnls)]

    def test_fixed_numbers(self):
        m = pc.metrics(self.closes([40.0, 40.0, -60.0, -20.0]), {"t0": 1.0, "t1": 1.0})
        assert m["n"] == 4 and m["wins"] == 2 and m["win_rate_pct"] == 50.0 and m["net"] == 0.0
        assert m["pf"] == pytest.approx(1.0) and m["avg_win"] == 40.0 and m["avg_loss"] == -40.0
        assert m["mean_wing"] == 3.0 and m["mean_credit"] == 0.9 and m["credit_over_width"] == 0.3
        assert m["mean_hold_days"] == 8.0 and m["mean_capital_at_risk"] == 210.0
        # friction: two trades with a live mid of 1.00 against a 0.90 fill, 5.2 commission each
        assert m["friction_share_pct"] == pytest.approx(100.0 * (2 * (10.0 + 5.2)) / 200.0, abs=0.1)

    def test_an_undefined_pf_is_none_and_the_interval_is_capped(self):
        m = pc.metrics(self.closes([10.0, 20.0, 30.0]))
        assert m["pf"] is None and m["pf_interval"][1] <= pc.PF_CAP

    def test_the_bootstrap_is_reproducible_and_widens_with_fewer_closes(self):
        a = pc.bootstrap_pf_interval([40.0, -50.0] * 10)
        assert a == pc.bootstrap_pf_interval([40.0, -50.0] * 10)
        b = pc.bootstrap_pf_interval([40.0, -50.0, 40.0, -50.0])
        assert (b[1] - b[0]) > (a[1] - a[0])
        assert pc.bootstrap_pf_interval([5.0]) is None

    def test_width_parsing(self):
        assert pc.parse_width(_legs(3.0)) == 3.0 and pc.parse_width(_legs(11.0)) == 11.0
        assert pc.parse_width("[]") is None and pc.parse_width("not json") is None
        assert pc.parse_width(json.dumps([{"strike": 1, "right": "C", "action": "SELL"}])) is None


class TestFillsPaceSlip:
    def test_the_fill_record(self, tmp_path):
        db = make_db(tmp_path, [{"trade_id": "f1"}, {"trade_id": "f2", "status": "filled"},
                                {"trade_id": "c1", "status": "cancelled"}, {"trade_id": "c2", "status": "cancelled"},
                                {"trade_id": "r1", "status": "rejected"}, {"trade_id": "p1", "status": "pending"},
                                {"trade_id": "g1", "reason": "never_filled"}])
        _, info = pc.run(db, "C1", EPOCH, NOW)
        f = info["fills"]
        assert (f["placed"], f["filled"], f["cancelled"], f["rejected"], f["pending"]) == (7, 2, 3, 1, 1)
        assert f["fill_rate_pct"] == pytest.approx(100.0 * 2 / 6, abs=0.1)

    def test_the_slip_statistic_is_the_go_live_one(self):
        rows = [{"price": 0.92, "live_mid": 1.00, "entry_price": 0.92},      # 8.7%
                {"price": 0.95, "live_mid": 1.00, "entry_price": 0.95},      # 5.3%
                {"price": 1.00, "live_mid": 0.0, "entry_price": 1.00}]       # no mid: dropped
        s = pc.slip_stats(rows)
        assert s["n"] == 2 and s["median_pct"] == pytest.approx((8.6957 + 5.2632) / 2, abs=0.1)

    def test_weeks(self):
        assert pc.weeks_between("2026-10-29", datetime(2026, 11, 12, tzinfo=timezone.utc)) == pytest.approx(2.0, abs=0.01)
        assert pc.weeks_between("2026-10-29", datetime(2026, 10, 1, tzinfo=timezone.utc)) == 0.0


class TestTheSchedule:
    @pytest.mark.parametrize("which, n, weeks, expected", [
        ("C1", 9, 11.9, "NOT DUE"), ("C1", 10, 1.0, "DUE"), ("C1", 0, 12.0, "DUE"),
        ("C2", 24, 25.9, "NOT DUE"), ("C2", 25, 3.0, "DUE"), ("C2", 15, 26.0, "DUE"),
        ("C2", 14, 26.0, "EXTENDED"), ("C2", 14, 51.9, "EXTENDED"), ("C2", 14, 52.0, "PACE-LIMITED"),
        ("C2", 0, 60.0, "PACE-LIMITED"), ("C2", 15, 52.0, "DUE"),
    ])
    def test_boundaries(self, which, n, weeks, expected):
        assert pc.checkpoint_status(which, n, weeks) == expected

    @pytest.mark.parametrize("n, net, pf, expected", [
        (15, 120.0, 1.0, "MODEL TOO PESSIMISTIC AT 3-WIDE"), (15, 120.0, None, "MODEL TOO PESSIMISTIC AT 3-WIDE"),
        (15, 120.0, 0.99, "INDETERMINATE"), (15, -300.0, 0.6, "MODEL CONSISTENT"), (15, -300.0, 0.61, "INDETERMINATE"),
        (15, 0.0, 1.0, "INDETERMINATE"), (14, 500.0, 2.0, "INDETERMINATE (fewer than 15 real closes)"),
        (25, -10.0, 0.9, "INDETERMINATE"),
    ])
    def test_the_decision_classes(self, n, net, pf, expected):
        assert pc.decide(n, net, pf) == expected

    def test_a_c1_report_never_carries_a_decision_class(self, tmp_path):
        db = make_db(tmp_path, [{"trade_id": f"t{i}"} for i in range(30)])
        text, info = pc.run(db, "C1", EPOCH, NOW)
        assert info["decision"] is None and "Decision (C2)" not in text and "Data-capture audit" in text

    def test_c2_decides_only_when_due(self, tmp_path):
        db = make_db(tmp_path, [{"trade_id": f"t{i}"} for i in range(3)])
        _, info = pc.run(db, "C2", EPOCH, datetime(2026, 11, 20, tzinfo=timezone.utc))
        assert info["status"] == "NOT DUE" and "no decision" in info["decision"]
        db2 = make_db(tmp_path / "b", [{"trade_id": f"t{i}"} for i in range(25)])
        _, info2 = pc.run(db2, "C2", EPOCH, NOW)
        assert info2["status"] == "DUE" and info2["decision"] == "MODEL TOO PESSIMISTIC AT 3-WIDE"


class TestTheAudit:
    def test_missing_fields_are_counted_and_unmapped_reasons_listed(self, tmp_path):
        db = make_db(tmp_path, [{"trade_id": "ok", "exec": (0.92, 1.0)},
                                {"trade_id": "bare", "reason": "stop_loss (P&L: -50%)", "car": 0.0}])
        text, _ = pc.run(db, "C1", EPOCH, NOW)
        assert "missing_capital_at_risk: 1" in text and "missing_entry_live_mid: 1" in text
        assert "stop_loss (P&L: -50%)" in text


class TestTheModelFigures:
    def test_they_equal_the_w110_entry_in_plan_md(self):
        plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        entry = plan[plan.index("W110 AS PLANNED"):]
        for text in ("win rate 39.4% / 37.5%", "PF 0.41 / 0.34", "$0.855 / $0.838", "mean 3.0 wing", "mean max loss $215",
                     "8.5 / 11.2 days", "14.4% / 15.3%", "touch stop 45.7% / 51.5%", "take profit 49.4% / 43.9%",
                     "expiry approach 4.9% / 4.5%", "10 real closes or 12 weeks", "25 real closes or 26 weeks",
                     "15 closes or 52 weeks", "real PF >= 1.0", "real PF <= 0.6", "fewer than 15"):
            assert text in entry, text
        m = pc.MODEL
        assert (m["win_rate_pct"], m["pf"], m["mean_credit"], m["mean_hold_days"], m["friction_share_pct"],
                m["touch_stop_pct"], m["take_profit_pct"], m["expiry_approach_pct"]) == (
            (39.4, 37.5), (0.41, 0.34), (0.855, 0.838), (8.5, 11.2), (14.4, 15.3), (45.7, 51.5), (49.4, 43.9), (4.9, 4.5))
        assert (pc.C1_CLOSES, pc.C1_WEEKS, pc.C2_CLOSES, pc.C2_WEEKS, pc.C2_EXT_WEEKS, pc.MIN_DECISION_CLOSES,
                pc.PF_PESSIMISTIC, pc.PF_CONSISTENT) == (10, 12, 25, 26, 52, 15, 1.0, 0.6)

    def test_the_default_epoch_is_the_paper_restart(self):
        assert pc.EPOCH_START_DEFAULT == "2026-10-29"
