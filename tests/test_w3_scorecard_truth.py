"""W3 — scorecard truth. Every test EXECUTES the real reporting code against
a REAL temp sqlite state DB; nothing is asserted by reading source.

The cluster being pinned (audit 2026-08-25):

* money-flow-04 / policy-vs-impl-1: the SCHEDULED Friday Telegram
  "GO-LIVE SCORECARD" (ait.orchestration.master.weekly_scorecard) graded the
  RETIRED all-strategy metric while the pinned R19d verdict metric
  (iron-condor closes only) lived solely in status.py — on the live book the
  two surfaces sat on opposite sides of the PF>1.3 gate, and the Telegram one
  is the number the operator receives.
* policy-vs-impl-2: the scorecard rendered a LIFETIME DOLLAR MEAN of
  |fill - mid| under the label "gate: median <=8% of credit".
* policy-vs-impl-3: 50 is the PRELIMINARY denominator; 100 is required before
  funding, and no surface said so.
* policy-vs-impl-5: two of gate 1's five criteria were simply absent, and an
  exception silently deleted the WHOLE gate readout.
* string-contracts-1 / -4: "not a real close" had different membership at
  every consumer; meta_label.build_training_data had NONE of it, and the
  reconciler's three $0 needs-review sentinels passed every filter.
* db-contracts-4: "open position" was `status NOT IN ('closed')`, which
  admitted terminal CANCELLED rows.

Pre-fix failure mode by construction: ait.reporting.go_live did not exist, so
every test here raised ImportError/AttributeError before the fix.
"""

from __future__ import annotations

import sqlite3

import pytest

from ait.bot.state import StateManager, TradeStatus
from ait.reporting.go_live import (
    GO_LIVE_FUNDING_CLOSES,
    GO_LIVE_PRELIM_CLOSES,
    NOT_REAL_CLOSE_PATTERNS,
    OPEN_TRADE_STATUSES,
    compute_go_live_verdict,
    format_verdict_lines,
    is_real_close,
    not_real_close_sql,
    open_trade_status_sql,
)

# --------------------------------------------------------------------------
# One seeded book, used by (almost) every test below.
#
#   3 REAL iron-condor closes   +100, +200, -50   -> n=3, 2W-1L, PF 6.00
#   1 never-filled $0 phantom   (string-contracts-1)
#   1 reconciler $0 sentinel    (string-contracts-4)
#   1 CANCELLED iron condor     (db-contracts-4)
#   1 real short_strangle close -400  (retired experiment, R19d-excluded)
#   1 genuinely OPEN iron condor
#
# The IC verdict (PF 6.00 PASS) and the all-strategy book (PF 0.67 FAIL) sit
# on opposite sides of the gate ON PURPOSE — that is the live defect in
# miniature, so a surface grading the wrong population cannot pass these
# tests by coincidence.
# --------------------------------------------------------------------------

_TRADES = [
    # (trade_id, strategy, status, entry, exit, pnl, reason, car, entry_price)
    ("T-IC-W1", "iron_condor", "closed", "2026-08-03T10:00:00",
     "2026-08-05T15:00:00", 100.0, "take_profit_short (P&L: 40.0%)", 1000.0, 5.00),
    ("T-IC-W2", "iron_condor", "closed", "2026-08-04T10:00:00",
     "2026-08-06T15:00:00", 200.0, "take_profit_short (P&L: 41.0%)", 1000.0, 6.00),
    ("T-IC-L1", "iron_condor", "closed", "2026-08-10T10:00:00",
     "2026-08-12T15:00:00", -50.0, "short_strike_touch (spot 703 <= put 704)",
     1000.0, 4.00),
    # never filled: the reconciler books $0 so the row stops occupying the book
    ("T-IC-PHANTOM", "iron_condor", "closed", "2026-08-07T10:00:00",
     "2026-08-07T16:00:00", 0.0, "stale_pending_never_filled", 0.0, 3.00),
    # position vanished at the broker, true P&L unrecoverable -> $0 + review
    ("T-IC-SENTINEL", "iron_condor", "closed", "2026-08-08T10:00:00",
     "2026-08-09T16:00:00", 0.0, "reconciler_unknown_exit_needs_review",
     1000.0, 3.50),
    # cancelled a month ago: terminal, NOT open
    ("T-IC-CANCELLED", "iron_condor", "cancelled", "2026-07-15T11:00:00",
     None, 0.0, "", 0.0, 2.00),
    # retired experiment: real close, but NOT the verdict metric
    ("T-SS-L1", "short_strangle", "closed", "2026-08-11T10:00:00",
     "2026-08-13T15:00:00", -400.0, "macro_event_flatten (days_to_event=1)",
     2000.0, 7.00),
    # genuinely open
    ("T-IC-OPEN", "iron_condor", "filled", "2026-08-20T10:00:00",
     None, 0.0, "", 500.0, 6.35),
]

_EXECUTIONS = [
    # (exec_id, trade_id, price, live_mid, exec_time)  -> slip % of credit
    ("X1", "T-IC-W1", 5.10, 5.00, "2026-08-03T10:00:01"),   # 2.0%
    ("X2", "T-IC-W2", 6.12, 6.00, "2026-08-04T10:00:01"),   # 2.0%
    ("X3", "T-IC-L1", 4.16, 4.00, "2026-08-10T10:00:01"),   # 4.0%
    ("X4", "T-SS-L1", 7.28, 7.00, "2026-08-11T10:00:01"),   # 4.0%
]

IC_N = 3
IC_WINS = 2
IC_NET = 250.0
IC_PF = 6.0            # 300 gross profit / 50 gross loss
IC_MAX_DD = 50.0
IC_DD_BASE = 2000.0    # W1 and W2 overlap; the $0-sentinel car must NOT count
BOOK_N = 4
BOOK_NET = -150.0


def _seed(tmp_path, *, trades=_TRADES, executions=_EXECUTIONS,
          with_context=False):
    """Build a REAL state DB through StateManager, then insert rows directly."""
    st = StateManager(db_path=tmp_path / "ait_state.db")
    with sqlite3.connect(st._db_path) as con:
        for (tid, strat, status, t_in, t_out, pnl, reason, car,
             px) in trades:
            con.execute(
                "INSERT INTO trades (trade_id, symbol, strategy, direction, "
                "status, entry_time, entry_price, quantity, contract_type, "
                "exit_time, realized_pnl, exit_reason_detailed, "
                "capital_at_risk) VALUES (?,?,?,?,?,?,?,1,'iron_condor',?,?,?,?)",
                (tid, "SPY", strat, "neutral", status, t_in, px, t_out, pnl,
                 reason, car))
            if with_context:
                con.execute(
                    "INSERT INTO trade_context (trade_id, entry_direction, "
                    "entry_confidence, entry_regime, entry_vix, entry_iv_rank, "
                    "entry_sentiment_score, entry_signals) "
                    "VALUES (?,'neutral',0.7,'range_bound',16.0,50.0,0.1,'{}')",
                    (tid,))
        for exec_id, tid, price, mid, when in executions:
            con.execute(
                "INSERT INTO executions (exec_id, trade_id, symbol, side, "
                "shares, price, exec_time, live_mid) "
                "VALUES (?,?,'SPY','BOT',1,?,?,?)",
                (exec_id, tid, price, when, mid))
        # the open IC also has an open_positions row, like the live book
        con.execute(
            "INSERT INTO open_positions (position_id, trade_id, symbol, "
            "contract_type, quantity, entry_price, entry_time, unrealized_pnl) "
            "VALUES ('T-IC-OPEN','T-IC-OPEN','SPY','iron_condor',1,6.35,"
            "'2026-08-20T10:00:00', 343.18)")
    return st


# ==========================================================================
# 1. the shared "not a real close" authority (string-contracts-1 / -4)
# ==========================================================================
class TestNotRealCloseAuthority:
    def test_reconciler_sentinels_are_excluded(self):
        """The three $0 review sentinels the reconciler books passed every
        pre-W3 filter (which knew only never_filled/pending/migrated) and
        would have counted as real LOSING closes."""
        from ait.execution import reconciler

        for reason in reconciler.NOT_REAL_CLOSE_EXIT_REASONS:
            assert not is_real_close(reason), reason

    def test_reconciler_still_writes_the_same_strings(self):
        """W3 changed who COUNTS these rows, never what the reconciler
        WRITES — other consumers match on the exact strings."""
        from ait.execution import reconciler

        assert reconciler.UNKNOWN_EXIT_REASON == "reconciler_unknown_exit"
        assert (reconciler.UNKNOWN_EXPIRED_EXIT_REASON
                == "reconciler_unknown_exit_expired_needs_review")
        assert (reconciler.NON_FINITE_EXIT_REASON
                == "reconciler_unknown_exit_needs_review")
        assert reconciler.NEVER_FILLED_EXIT_REASON == "stale_pending_never_filled"

    def test_real_exit_reasons_survive(self):
        for reason in ("take_profit_short (P&L: 40.0%)", "short_strike_touch",
                       "stop_loss (P&L: -35.3%)", "reconciler_expired_intrinsic",
                       "macro_event_flatten (days_to_event=1)", "", None):
            assert is_real_close(reason), reason

    def test_sql_and_python_twins_agree_on_a_real_db(self, tmp_path):
        st = _seed(tmp_path)
        with sqlite3.connect(st._db_path) as con:
            con.row_factory = sqlite3.Row
            kept = {r["trade_id"] for r in con.execute(
                f"SELECT trade_id FROM trades WHERE status='closed' "
                f"{not_real_close_sql()}")}
            everything = [(r["trade_id"], r["exit_reason_detailed"])
                          for r in con.execute(
                              "SELECT trade_id, exit_reason_detailed FROM "
                              "trades WHERE status='closed'")]
        assert kept == {t for t, reason in everything if is_real_close(reason)}
        assert kept == {"T-IC-W1", "T-IC-W2", "T-IC-L1", "T-SS-L1"}


# ==========================================================================
# 2. the verdict itself (policy-vs-impl-1/-3, money-flow-04)
# ==========================================================================
class TestVerdictMetric:
    def test_counts_only_real_iron_condor_closes(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.verdict.n == IC_N
        assert (v.verdict.wins, v.verdict.losses) == (IC_WINS, IC_N - IC_WINS)
        assert v.verdict.net == pytest.approx(IC_NET)
        assert v.verdict.pf == pytest.approx(IC_PF)
        assert v.verdict.max_dd == pytest.approx(IC_MAX_DD)

    def test_phantom_and_sentinel_are_not_in_pf_or_win_rate(self, tmp_path):
        """Both $0 rows are non-positive: counting either would drop the win
        rate and (as extra closes) move the go-live clock."""
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.verdict.n == 3, "a $0 phantom/sentinel entered the sample"
        assert v.verdict.wins / v.verdict.n == pytest.approx(2 / 3)
        assert v.excluded == {"never_filled": 1, "reconciler_needs_review": 1}

    def test_retired_experiments_excluded_from_the_verdict_but_kept_as_book(
            self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.book.n == BOOK_N
        assert v.book.net == pytest.approx(BOOK_NET)
        # the whole point: the two populations disagree about the gate
        assert v.verdict.pf > 1.3 and v.book.pf < 1.3

    def test_denominator_is_50_prelim_and_100_funding(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        line = v.criterion("sample").render()
        assert f"{IC_N}/{GO_LIVE_PRELIM_CLOSES} prelim" in line
        assert f"{IC_N}/{GO_LIVE_FUNDING_CLOSES} funding" in line
        assert (GO_LIVE_PRELIM_CLOSES, GO_LIVE_FUNDING_CLOSES) == (50, 100)

    def test_dd_base_uses_only_the_graded_population(self, tmp_path):
        """The $0 sentinel carries capital_at_risk=1000; letting it into the
        D2 base would inflate the denominator and flatter DD%."""
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.dd_base == pytest.approx(IC_DD_BASE)
        assert v.dd_pct == pytest.approx(IC_MAX_DD / IC_DD_BASE * 100)
        assert v.criterion("drawdown").available
        assert v.criterion("drawdown").passed is True


# ==========================================================================
# 3. slippage rendered as the PINNED statistic (policy-vs-impl-2)
# ==========================================================================
class TestSlippageCriterion:
    def test_is_a_percent_of_credit_median_not_a_dollar_mean(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        # |fill-mid| dollars are 0.10/0.12/0.16/0.28 -> a DOLLAR MEAN of 0.165
        # (what the old scorecard printed under this gate's percent label);
        # the PINNED statistic is the median of 2/2/4/4 % of credit = 3.0%.
        assert v.slip_median_pct == pytest.approx(3.0)
        assert v.slip_n == 4
        text = v.criterion("slippage").render()
        assert "median 3.0% of credit" in text
        assert "$0.16" not in text and "$0.17" not in text
        assert v.criterion("slippage").passed is True

    def test_trend_clause_is_reported(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.slip_trend is not None
        assert "trend" in v.criterion("slippage").render()

    def test_unavailable_when_no_fill_carries_a_live_mid(self, tmp_path):
        st = _seed(tmp_path, executions=[])
        v = compute_go_live_verdict(st._db_path)
        c = v.criterion("slippage")
        assert c.available is False
        assert "UNAVAILABLE" in c.render()
        assert "unmeasurable" in c.render()
        # and NO number wearing the gate's label
        assert "median" not in c.render().split("UNAVAILABLE")[0]


# ==========================================================================
# 4. incompletely-evaluated gates print UNAVAILABLE (policy-vs-impl-5)
# ==========================================================================
class TestUnavailableCriteria:
    def test_all_five_criteria_are_always_rendered(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert [c.key for c in v.criteria] == [
            "sample", "pf", "drawdown", "slippage", "incidents"]

    def test_unmanaged_incidents_is_explicitly_unavailable(self, tmp_path):
        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        c = v.criterion("incidents")
        assert c.available is False
        assert "UNAVAILABLE" in c.render()
        assert "shadow_referee" in c.render()

    def test_drawdown_unavailable_when_no_capital_at_risk(self, tmp_path):
        """A missing D2 base must read UNAVAILABLE, never a percentage
        computed against an invented $1,000 floor."""
        trades = [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], 0.0, t[8])
                  for t in _TRADES]
        st = _seed(tmp_path, trades=trades)
        v = compute_go_live_verdict(st._db_path)
        c = v.criterion("drawdown")
        assert v.dd_base is None and v.dd_pct is None
        assert c.available is False
        assert "UNAVAILABLE" in c.render() and "capital_at_risk" in c.render()
        assert "%" not in c.render().split("cannot be evaluated")[0].replace(
            "<8% of deployed risk", "")

    def test_pf_unavailable_with_no_verdict_closes(self, tmp_path):
        trades = [t for t in _TRADES if t[1] != "iron_condor"]
        st = _seed(tmp_path, trades=trades, executions=[])
        v = compute_go_live_verdict(st._db_path)
        assert v.verdict.n == 0
        assert v.criterion("pf").available is False
        assert "UNAVAILABLE" in v.criterion("pf").render()


# ==========================================================================
# 5. db-contracts-4 — cancelled rows are not open
# ==========================================================================
class TestOpenPositionAuthority:
    def test_open_statuses_come_from_TradeStatus(self):
        assert OPEN_TRADE_STATUSES == (
            TradeStatus.PENDING.value, TradeStatus.FILLED.value,
            TradeStatus.PARTIAL.value, TradeStatus.CLOSING.value)
        assert TradeStatus.CANCELLED.value not in OPEN_TRADE_STATUSES
        assert TradeStatus.REJECTED.value not in OPEN_TRADE_STATUSES

    def test_status_get_status_excludes_cancelled(self, tmp_path, monkeypatch):
        st = _seed(tmp_path)
        import status as status_mod
        monkeypatch.setattr(status_mod, "DB", st._db_path)
        monkeypatch.setattr(status_mod, "_proc_running", lambda: (False, 0))
        monkeypatch.setattr(status_mod, "_tail_lines", lambda *a, **k: [])

        out = status_mod.get_status()
        ids = {p["status"] for p in out["open_positions"]}
        # pre-fix: 2 rows (T-IC-OPEN + T-IC-CANCELLED) on a 1-position book
        assert len(out["open_positions"]) == 1, out["open_positions"]
        assert ids == {"filled"}
        assert out["unrealized_total"] == pytest.approx(343.18)

    def test_cancelled_row_is_present_but_not_open_in_sql(self, tmp_path):
        st = _seed(tmp_path)
        with sqlite3.connect(st._db_path) as con:
            n_open = con.execute(
                f"SELECT COUNT(*) FROM trades WHERE "
                f"{open_trade_status_sql()}").fetchone()[0]
            n_loose = con.execute(
                "SELECT COUNT(*) FROM trades WHERE status NOT IN "
                "('closed')").fetchone()[0]
        assert n_open == 1
        assert n_loose == 2, "fixture must contain the cancelled row"


# ==========================================================================
# 6. the two operator surfaces agree (money-flow-04)
# ==========================================================================
def _run_status_gate_block(db_path, monkeypatch, capsys) -> str:
    import status as status_mod
    monkeypatch.setattr(status_mod, "DB", db_path)
    monkeypatch.setattr(status_mod, "_proc_running", lambda: (False, 0))
    monkeypatch.setattr(status_mod, "_tail_lines", lambda *a, **k: [])
    status_mod.main()
    return capsys.readouterr().out


def _run_master_scorecard(db_path, monkeypatch) -> str:
    from ait.orchestration import master
    sent: list[str] = []
    monkeypatch.setattr(master, "DATA_DIR", db_path.parent)
    monkeypatch.setattr(master, "_alert", lambda msg: sent.append(msg) or True)
    master.weekly_scorecard()
    assert sent, "the scheduled scorecard sent NOTHING"
    return sent[-1]


class TestBothSurfacesAgree:
    def test_master_scorecard_reports_the_iron_condor_verdict(
            self, tmp_path, monkeypatch):
        st = _seed(tmp_path)
        msg = _run_master_scorecard(st._db_path, monkeypatch)
        # pre-fix this said "closes: 5/50 | PF: 0.67" — the RETIRED metric
        assert "3/50 prelim" in msg
        assert "PF 6.00" in msg
        assert "2W-1L" in msg
        assert "net $+250.00" in msg

    def test_master_keeps_the_all_strategy_line_labelled_as_book_honesty(
            self, tmp_path, monkeypatch):
        st = _seed(tmp_path)
        msg = _run_master_scorecard(st._db_path, monkeypatch)
        book = [ln for ln in msg.splitlines() if "book-level" in ln]
        assert len(book) == 1, msg
        assert "ALL strategies" in book[0] and "NOT the verdict" in book[0]
        assert "4 closes" in book[0] and "PF 0.67" in book[0]

    def test_master_no_longer_labels_a_dollar_mean_as_the_percent_gate(
            self, tmp_path, monkeypatch):
        st = _seed(tmp_path)
        msg = _run_master_scorecard(st._db_path, monkeypatch)
        assert "avg |fill-mid|" not in msg
        assert "median 3.0% of credit" in msg

    def test_master_pace_is_not_satisfied_by_a_single_close(
            self, tmp_path, monkeypatch):
        """'on track' used to be printed for any n >= 1."""
        st = _seed(tmp_path)
        msg = _run_master_scorecard(st._db_path, monkeypatch)
        assert "on track" not in msg
        assert "verdict closes/wk" in msg

    def test_the_two_surfaces_render_identical_gate_lines(
            self, tmp_path, monkeypatch, capsys):
        st = _seed(tmp_path)
        cli = _run_status_gate_block(st._db_path, monkeypatch, capsys)
        telegram = _run_master_scorecard(st._db_path, monkeypatch)

        v = compute_go_live_verdict(st._db_path)
        gate_lines = format_verdict_lines(v)
        assert len(gate_lines) >= 7
        for line in gate_lines:
            assert line in telegram, f"missing from Telegram: {line}"
            assert line in cli, f"missing from the CLI: {line}"

    def test_neither_surface_grades_the_retired_population(
            self, tmp_path, monkeypatch, capsys):
        st = _seed(tmp_path)
        cli = _run_status_gate_block(st._db_path, monkeypatch, capsys)
        telegram = _run_master_scorecard(st._db_path, monkeypatch)
        for surface in (cli, telegram):
            verdict_line = [ln for ln in surface.splitlines()
                            if "[2] profit fac" in ln][0]
            assert "PF 6.00" in verdict_line
            assert "0.67" not in verdict_line


# ==========================================================================
# 7. a failed gate readout must be LOUD (policy-vs-impl-5)
# ==========================================================================
class TestGateReadoutFailsLoudly:
    def test_status_prints_gate_readout_failed(self, tmp_path, monkeypatch,
                                               capsys):
        st = _seed(tmp_path)
        import status as status_mod
        import ait.reporting.go_live as gl

        def _boom(*a, **k):
            raise sqlite3.OperationalError("no such column: capital_at_risk")

        monkeypatch.setattr(status_mod, "DB", st._db_path)
        monkeypatch.setattr(status_mod, "_proc_running", lambda: (False, 0))
        monkeypatch.setattr(status_mod, "_tail_lines", lambda *a, **k: [])
        monkeypatch.setattr(status_mod, "compute_go_live_verdict", _boom)
        monkeypatch.setattr(gl, "compute_go_live_verdict", _boom)

        status_mod.main()
        out = capsys.readouterr().out
        # pre-fix: `except Exception: pass` -> the gates VANISHED silently
        assert "GATE READOUT FAILED" in out
        assert "no such column" in out
        assert "GO-LIVE GATES" not in out

    def test_master_alerts_gate_readout_failed(self, tmp_path, monkeypatch):
        st = _seed(tmp_path)
        from ait.orchestration import master
        import ait.reporting.go_live as gl

        def _boom(*a, **k):
            raise RuntimeError("schema drift")

        sent: list[str] = []
        monkeypatch.setattr(master, "DATA_DIR", st._db_path.parent)
        monkeypatch.setattr(master, "_alert", lambda m: sent.append(m) or True)
        monkeypatch.setattr(gl, "compute_go_live_verdict", _boom)

        master.weekly_scorecard()
        assert sent, "a broken gate readout produced a SILENT Friday"
        assert "GATE READOUT FAILED" in sent[-1]
        assert "schema drift" in sent[-1]
        # the rest of the message still went out, so the failure is visible
        # in context rather than replacing the whole scorecard
        assert "GO-LIVE SCORECARD" in sent[-1]


# ==========================================================================
# 8. string-contracts-1 — the meta-label trainer
# ==========================================================================
class TestMetaLabelTrainingData:
    def test_phantoms_and_sentinels_are_not_training_rows(self, tmp_path):
        from ait.ml.meta_label import MetaLabeler

        st = _seed(tmp_path, with_context=True)
        df = MetaLabeler().build_training_data(st)
        # pre-fix: 6 rows (every closed trade incl. both $0 rows), of which
        # 2 were fabricated all-loss labels = 33% of the set.
        assert len(df) == 4, df
        assert int(df["profitable"].sum()) == 2
        assert int((df["profitable"] == 0).sum()) == 2

    def test_no_zero_pnl_phantom_pollutes_the_negative_class(self, tmp_path):
        from ait.ml.meta_label import MetaLabeler

        st = _seed(tmp_path, with_context=True)
        n_rows = len(MetaLabeler().build_training_data(st))
        with sqlite3.connect(st._db_path) as con:
            n_closed = con.execute(
                "SELECT COUNT(*) FROM trades t JOIN trade_context c "
                "ON c.trade_id = t.trade_id WHERE t.status='closed'"
            ).fetchone()[0]
        assert n_closed == 6, "fixture must contain the $0 rows"
        assert n_rows == n_closed - 2


# ==========================================================================
# 9. the authority is not re-declared anywhere it is consumed
# ==========================================================================
class TestSingleAuthority:
    def test_verdict_strategies_come_from_base(self, tmp_path):
        from ait.strategies.base import GO_LIVE_VERDICT_STRATEGIES

        st = _seed(tmp_path)
        v = compute_go_live_verdict(st._db_path)
        assert v.strategies == GO_LIVE_VERDICT_STRATEGIES

    def test_patterns_cover_every_family(self):
        assert set(NOT_REAL_CLOSE_PATTERNS) == {
            "%migrated%", "%pending%", "%never_filled%",
            "%reconciler_unknown%", "%needs_review%"}

    def test_extending_the_verdict_set_changes_both_surfaces(
            self, tmp_path, monkeypatch):
        """A second promoted strategy must flow through the single set, not
        through a string literal at each call site."""
        st = _seed(tmp_path)
        v = compute_go_live_verdict(
            st._db_path, strategies=frozenset({"iron_condor", "short_strangle"}))
        assert v.verdict.n == 4
        assert v.verdict.net == pytest.approx(BOOK_NET)
