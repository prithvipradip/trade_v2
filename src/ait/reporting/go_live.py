"""W3 — the SINGLE authority for go-live scorecard truth.

Why this module exists (audit W3, 2026-08-25):

* money-flow-04 / policy-vs-impl-1: two operator surfaces rendered the
  "go-live verdict" from the same database and disagreed.  ``status.py``
  implemented the pinned R19d metric (IRON CONDOR closes only), while the
  SCHEDULED Friday Telegram "GO-LIVE SCORECARD"
  (``ait.orchestration.master.weekly_scorecard``) graded ALL strategies —
  including the retired long_straddle / long_call / short_strangle
  experiments R19d explicitly excluded.  On the live book the two sat on
  opposite sides of the PF>1.3 gate, and the Telegram one — the number the
  operator actually receives — was the retired metric.  Both surfaces now
  call :func:`compute_go_live_verdict` and render :func:`format_verdict_lines`,
  so a divergence is no longer expressible.

* string-contracts-1 / string-contracts-4: "rows that are not real closes"
  was open-coded at ~8 call sites with DIFFERENT membership.
  ``meta_label.build_training_data`` had NONE of it (it trained on $0
  never-filled phantoms), and the reconciler's three $0 review sentinels
  (``reconciler_unknown_exit``, ``..._needs_review``,
  ``..._expired_needs_review``) passed every existing filter and would have
  counted as real losing closes in PF / win-rate / drawdown.
  :data:`NOT_REAL_CLOSE_PATTERNS` + :func:`not_real_close_sql` +
  :func:`is_real_close` are now the one authority.  NOTE: this changes only
  who COUNTS those rows, never what the reconciler WRITES — other consumers
  depend on those exact strings.

* db-contracts-4: "open position" was spelled ``status NOT IN ('closed')``,
  which let terminal CANCELLED/REJECTED rows through (the live book reported
  5 open when it held 2).  :data:`OPEN_TRADE_STATUSES` derives the answer
  from the real authority, ``ait.bot.state.TradeStatus``.

* policy-vs-impl-2 / -5: a criterion is either computed AS THE GATE DEFINES
  IT or printed as UNAVAILABLE with a reason.  Never a differently-defined
  number wearing the gate's label (the old scorecard printed a LIFETIME
  DOLLAR MEAN under the label "gate: median <=8% of credit").

Import-light on purpose — ``status.py`` imports this.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from ait.bot.state import TradeStatus
from ait.strategies.base import GO_LIVE_VERDICT_STRATEGIES

__all__ = [
    "NOT_REAL_CLOSE_PATTERNS",
    "not_real_close_sql",
    "is_real_close",
    "not_real_close_bucket",
    "OPEN_TRADE_STATUSES",
    "TERMINAL_TRADE_STATUSES",
    "open_trade_status_sql",
    "GO_LIVE_PRELIM_CLOSES",
    "GO_LIVE_FUNDING_CLOSES",
    "GATE_PF",
    "GATE_DD_PCT",
    "GATE_SLIP_PCT",
    "SLIP_WINDOW",
    "BookStats",
    "Criterion",
    "GoLiveVerdict",
    "compute_go_live_verdict",
    "format_verdict_lines",
    "format_pace_line",
    "max_concurrent_car",
]

# ---------------------------------------------------------------------------
# "not a real close" — ONE authority (string-contracts-1 + string-contracts-4)
# ---------------------------------------------------------------------------
# Rows booked with status='closed' that are NOT trading outcomes and must
# never enter PF / win-rate / drawdown / P&L totals / ML training sets:
#
#   never_filled  -> reconciler.py stale-pending sweep: the signal never
#                    filled, $0 booked so the row stops occupying the open
#                    book.
#   pending       -> same family, older vocabulary.
#   migrated      -> 2026-07-06 reset bookkeeping rows.
#   reconciler_unknown / needs_review
#                 -> reconciler.py books $0 with "P&L NOT booked, needs
#                    manual review" when the true exit price is
#                    unrecoverable.  A $0 placeholder for an UNKNOWN outcome
#                    is not a scratch trade; counting it dilutes every
#                    aggregate and (being non-positive) reads as a LOSS.
#
# Patterns are matched against exit_reason_detailed; SQL LIKE is already
# case-insensitive for ASCII and the Python twin lowercases.
NOT_REAL_CLOSE_PATTERNS: tuple[str, ...] = (
    "%migrated%",
    "%pending%",
    "%never_filled%",
    "%reconciler_unknown%",
    "%needs_review%",
)


def not_real_close_sql(column: str = "exit_reason_detailed") -> str:
    """SQL fragment excluding non-real closes, to append to a WHERE clause.

    Returns a string that STARTS with ``AND`` (every call site appends it to
    an existing predicate such as ``status='closed'``).  COALESCE is
    mandatory: ``NULL NOT LIKE x`` is NULL, which a WHERE clause drops — that
    silently deleted every NULL-reason close from status.py's P&L before R17.
    """
    return " ".join(
        f"AND COALESCE({column},'') NOT LIKE '{p}'" for p in NOT_REAL_CLOSE_PATTERNS
    )


def is_real_close(exit_reason_detailed: str | None) -> bool:
    """Python twin of :func:`not_real_close_sql` (same membership)."""
    r = (exit_reason_detailed or "").lower()
    return not any(p.strip("%") in r for p in NOT_REAL_CLOSE_PATTERNS)


def not_real_close_bucket(exit_reason_detailed: str | None) -> str | None:
    """Which excluded family a row belongs to, or None if it is a real close."""
    r = (exit_reason_detailed or "").lower()
    if "reconciler_unknown" in r or "needs_review" in r:
        return "reconciler_needs_review"
    if "never_filled" in r or "pending" in r:
        return "never_filled"
    if "migrated" in r:
        return "migrated"
    return None


# ---------------------------------------------------------------------------
# "open position" — derived from the real authority (db-contracts-4)
# ---------------------------------------------------------------------------
OPEN_TRADE_STATUSES: tuple[str, ...] = tuple(
    s.value
    for s in (
        TradeStatus.PENDING,
        TradeStatus.FILLED,
        TradeStatus.PARTIAL,
        TradeStatus.CLOSING,
    )
)
# The terminal statuses that ``status NOT IN ('closed')`` wrongly admitted.
TERMINAL_TRADE_STATUSES: tuple[str, ...] = tuple(
    s.value for s in (TradeStatus.CLOSED, TradeStatus.CANCELLED, TradeStatus.REJECTED)
)


def open_trade_status_sql(column: str = "status") -> str:
    """SQL predicate (no leading AND) selecting genuinely-open trades."""
    vals = ", ".join(f"'{s}'" for s in OPEN_TRADE_STATUSES)
    return f"{column} IN ({vals})"


# ---------------------------------------------------------------------------
# Pinned gate constants (PLAN.md "Go-live gates" #1; R19d 2026-08-20)
# ---------------------------------------------------------------------------
GO_LIVE_PRELIM_CLOSES = 50      # PRELIMINARY read
GO_LIVE_FUNDING_CLOSES = 100    # before funding (R9: 50 cannot settle PF>1.3)
GATE_PF = 1.3
GATE_DD_PCT = 8.0               # of DEPLOYED RISK, not of paper NLV (D2)
GATE_SLIP_PCT = 8.0             # median entry slip as % of credit
SLIP_WINDOW = 20                # trailing fills
PACE_WINDOW_DAYS = 28
PACE_TARGET_PER_WEEK = (3.0, 4.0)


@dataclass(frozen=True)
class BookStats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    net: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    pf: float | None = None      # None => undefined (no losing closes)
    pf_str: str = "n/a"
    max_dd: float = 0.0

    @classmethod
    def from_pnls(cls, pnls) -> "BookStats":
        vals = [float(p or 0.0) for p in pnls]
        n = len(vals)
        wins = sum(1 for p in vals if p > 0)
        gp = sum(p for p in vals if p > 0)
        gl = abs(sum(p for p in vals if p < 0))
        pf = (gp / gl) if gl > 0 else None
        pf_str = "n/a" if n == 0 else ("inf" if pf is None else f"{pf:.2f}")
        peak = dd = cum = 0.0
        for p in vals:
            cum += p
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        # losses = n - wins keeps the historical W-L rendering (a $0 real
        # close is not a win); the $0 PHANTOMS that made that convention
        # dangerous are excluded upstream by is_real_close().
        return cls(n, wins, n - wins, sum(vals), gp, gl, pf, pf_str, dd)


@dataclass(frozen=True)
class Criterion:
    """One of gate 1's five criteria, rendered honestly."""

    key: str
    label: str
    text: str                   # the value line, or the UNAVAILABLE reason
    available: bool
    passed: bool | None = None  # None => not decidable (unavailable / no data)
    suffix: str = ""            # rendered AFTER the PASS/FAIL verdict

    def render(self) -> str:
        if not self.available:
            return f"{self.label}: UNAVAILABLE — {self.text}"
        verdict = "" if self.passed is None else (" PASS" if self.passed else " FAIL")
        return f"{self.label}: {self.text}{verdict}{self.suffix}"


@dataclass
class GoLiveVerdict:
    strategies: frozenset = field(default_factory=lambda: GO_LIVE_VERDICT_STRATEGIES)
    verdict: BookStats = field(default_factory=BookStats)   # IC-only (the gate)
    book: BookStats = field(default_factory=BookStats)      # all strategies
    dd_base: float | None = None
    dd_pct: float | None = None
    dd_missing_car: int = 0
    slip_median_pct: float | None = None
    slip_n: int = 0
    slip_trend: str | None = None
    excluded: dict = field(default_factory=dict)  # bucket -> row count
    pace_per_week: float | None = None
    criteria: list = field(default_factory=list)

    def criterion(self, key: str) -> Criterion | None:
        for c in self.criteria:
            if c.key == key:
                return c
        return None


def max_concurrent_car(rows) -> float:
    """D2 (pinned 2026-07-16): max CONCURRENT sum of capital_at_risk over the
    trades' [entry_time, exit_time) windows — the economically meaningful
    drawdown denominator (what was actually at risk when the loss happened).

    ``rows`` is any iterable of ``(entry_time, exit_time, capital_at_risk)``.
    Rows without a positive car contribute nothing.
    """
    ev: list[tuple[str, float]] = []
    for entry_time, exit_time, car in rows:
        car = float(car or 0.0)
        if car <= 0:
            continue
        ev.append((entry_time or "", +car))
        ev.append((exit_time or "9999", -car))
    peak = cur = 0.0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def _median(vals) -> float:
    return float(statistics.median(vals))


def compute_go_live_verdict(
    db_path,
    *,
    strategies: frozenset = GO_LIVE_VERDICT_STRATEGIES,
    now: datetime | None = None,
) -> GoLiveVerdict:
    """Compute the R19d go-live verdict — the ONE implementation.

    Both the ``status.py`` CLI/web surface and the scheduled Friday Telegram
    scorecard call this.  Any criterion whose data cannot support the PINNED
    definition is returned UNAVAILABLE with a reason rather than substituted
    by a differently-defined number.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        closed = con.execute(
            "SELECT realized_pnl, strategy, entry_time, exit_time, "
            "COALESCE(capital_at_risk, 0) car, "
            "COALESCE(exit_reason_detailed,'') reason "
            f"FROM trades WHERE status='closed' {not_real_close_sql()} "
            "ORDER BY COALESCE(exit_time, entry_time)"
        ).fetchall()
        # Surface (rather than silently drop) what the filter removed —
        # string-contracts-4 asked for a needs-review count on the surface.
        all_reasons = con.execute(
            "SELECT COALESCE(exit_reason_detailed,'') reason, COUNT(*) n "
            "FROM trades WHERE status='closed' GROUP BY 1"
        ).fetchall()
        open_rows = con.execute(
            "SELECT strategy, entry_time, exit_time, COALESCE(capital_at_risk,0) car "
            f"FROM trades WHERE {open_trade_status_sql()}"
        ).fetchall()
        try:
            slips = con.execute(
                "SELECT e.price, e.live_mid, e.exec_time, t.entry_price "
                "FROM executions e JOIN trades t ON t.trade_id = e.trade_id "
                "WHERE e.live_mid > 0 AND COALESCE(t.entry_price,0) > 0 "
                "ORDER BY e.exec_time DESC LIMIT ?",
                (SLIP_WINDOW,),
            ).fetchall()
            slip_err = None
        except sqlite3.Error as e:  # pre-migration DB with no executions table
            slips, slip_err = [], str(e)
    finally:
        con.close()

    v = GoLiveVerdict(strategies=frozenset(strategies))

    excluded: dict[str, int] = {}
    for r in all_reasons:
        bucket = not_real_close_bucket(r["reason"])
        if bucket:
            excluded[bucket] = excluded.get(bucket, 0) + int(r["n"])
    v.excluded = excluded

    ic = [r for r in closed if r["strategy"] in v.strategies]
    v.verdict = BookStats.from_pnls([r["realized_pnl"] for r in ic])
    v.book = BookStats.from_pnls([r["realized_pnl"] for r in closed])

    # ---- [3] drawdown vs DEPLOYED RISK -------------------------------------
    # D2 base restricted to the SAME population being graded (money-flow-04):
    # grading IC drawdown against a base inflated by retired-experiment risk
    # would flatter the percentage.
    ic_open = [r for r in open_rows if r["strategy"] in v.strategies]
    base = max_concurrent_car(
        [(r["entry_time"], r["exit_time"], r["car"]) for r in ic]
        + [(r["entry_time"], r["exit_time"], r["car"]) for r in ic_open]
    )
    v.dd_missing_car = sum(1 for r in ic if (r["car"] or 0) <= 0)
    if base > 0:
        v.dd_base = base
        v.dd_pct = v.verdict.max_dd / base * 100.0

    # ---- [4] slippage: median |fill-mid| as % of CREDIT, trailing N --------
    # Replicates scripts/shadow_referee.py check [7] verbatim — the pinned
    # statistic (PLAN.md gate 1).  The old scorecard printed
    # AVG(ABS(price-live_mid)) in DOLLARS under this gate's percent label
    # (policy-vs-impl-2).  Population is all-strategy, exactly as the referee
    # computes it; the rendered line says so.
    slip_vals = [
        abs(float(s["price"]) - float(s["live_mid"])) / float(s["entry_price"]) * 100.0
        for s in slips
    ]
    if slip_vals:
        v.slip_median_pct = _median(slip_vals)
        v.slip_n = len(slip_vals)
        # "no worsening trend": fills come back newest-first; compare the
        # newer half of the trailing window against the older half.
        if len(slip_vals) >= 4:
            newer = slip_vals[: len(slip_vals) // 2]
            older = slip_vals[len(slip_vals) // 2:]
            mn, mo = _median(newer), _median(older)
            v.slip_trend = (
                f"trend {'WORSENING' if mn > mo else 'stable'} "
                f"(newer {mn:.1f}% vs older {mo:.1f}%)"
            )

    # ---- pace (policy-vs-impl-1: 'on track' used to mean n >= 1) -----------
    ref = now or datetime.now()
    cutoff = (ref - timedelta(days=PACE_WINDOW_DAYS)).isoformat()
    recent = sum(1 for r in ic if (r["exit_time"] or r["entry_time"] or "") >= cutoff)
    v.pace_per_week = recent / (PACE_WINDOW_DAYS / 7.0)

    v.criteria = _build_criteria(v, slip_err)
    return v


def _build_criteria(v: GoLiveVerdict, slip_err: str | None) -> list:
    s = v.verdict
    crit: list[Criterion] = [
        Criterion(
            key="sample",
            label="[1] sample    ",
            text=(
                f"closes {s.n}/{GO_LIVE_PRELIM_CLOSES} prelim, "
                f"{s.n}/{GO_LIVE_FUNDING_CLOSES} funding | "
                f"{s.wins}W-{s.losses}L | net ${s.net:+,.2f}"
            ),
            available=True,
            passed=(s.n >= GO_LIVE_PRELIM_CLOSES),
        ),
    ]
    if s.n == 0:
        crit.append(Criterion(
            key="pf", label="[2] profit fac",
            text="no real iron-condor closes yet", available=False))
    else:
        crit.append(Criterion(
            key="pf", label="[2] profit fac",
            text=f"PF {s.pf_str} (gate >{GATE_PF:g})",
            available=True,
            passed=(s.pf is None or s.pf > GATE_PF)))

    # [3] drawdown vs deployed risk (policy-vs-impl-5: was a bare dollar
    # number with no denominator and no gate annotation on status.py).
    if v.dd_base is None:
        crit.append(Criterion(
            key="drawdown", label="[3] drawdown  ",
            text=(f"maxDD ${s.max_dd:,.0f} but NO deployed-risk base — zero "
                  f"graded rows carry capital_at_risk, so the "
                  f"'<{GATE_DD_PCT:g}% of deployed risk' gate cannot be evaluated"),
            available=False))
    else:
        hole = (f" | COVERAGE HOLE: {v.dd_missing_car} close(s) missing "
                f"capital_at_risk" if v.dd_missing_car else "")
        crit.append(Criterion(
            key="drawdown", label="[3] drawdown  ",
            text=(f"maxDD ${s.max_dd:,.0f} = {v.dd_pct:.1f}% of deployed risk "
                  f"~${v.dd_base:,.0f} (gate <{GATE_DD_PCT:g}%){hole}"),
            available=True, passed=(v.dd_pct < GATE_DD_PCT)))

    # [4] slippage (policy-vs-impl-2)
    if v.slip_median_pct is None:
        why = (f"executions table unreadable ({slip_err})" if slip_err else
               "no execution carries live_mid on a trade with an entry price — "
               "median |fill-mid| as % of credit is unmeasurable")
        crit.append(Criterion(
            key="slippage", label="[4] slippage  ", text=why, available=False))
    else:
        trend = (f" | {v.slip_trend}" if v.slip_trend
                 else " | trend UNAVAILABLE (<4 fills in window)")
        crit.append(Criterion(
            key="slippage", label="[4] slippage  ",
            text=(f"median {v.slip_median_pct:.1f}% of credit over trailing "
                  f"{v.slip_n} fills, all strategies (gate <={GATE_SLIP_PCT:g}%)"),
            available=True, passed=(v.slip_median_pct <= GATE_SLIP_PCT),
            suffix=trend))

    # [5] unmanaged-position incidents — genuinely not answerable from the
    # state DB alone (HALT markers + duplicate-position detection live in
    # scripts/shadow_referee.py check [8]).  PRINTED, not omitted, so the
    # operator can SEE the gate is incompletely evaluated (policy-vs-impl-5).
    crit.append(Criterion(
        key="incidents", label="[5] incidents ",
        text=("zero-unmanaged-position-incidents is not evaluable from the "
              "state DB; run scripts/shadow_referee.py (check [8])"),
        available=False))
    return crit


def format_verdict_lines(v: GoLiveVerdict, indent: str = "") -> list[str]:
    """The gate readout — IDENTICAL text on every surface that renders it."""
    names = "/".join(sorted(v.strategies)).upper().replace("_", " ")
    lines = [f"{indent}GO-LIVE GATES ({names} — the R19d verdict metric)"]
    lines += [f"{indent}  {c.render()}" for c in v.criteria]
    b = v.book
    lines.append(
        f"{indent}  book-level (ALL strategies incl. retired experiments — "
        f"NOT the verdict): {b.n} closes | {b.wins}W-{b.losses}L | "
        f"net ${b.net:+,.2f} | PF {b.pf_str}"
    )
    if v.excluded:
        exc = ", ".join(f"{k} {n}" for k, n in sorted(v.excluded.items()))
        lines.append(f"{indent}  excluded as not-real closes: {exc}")
    return lines


def format_pace_line(v: GoLiveVerdict, indent: str = "") -> str:
    lo, hi = PACE_TARGET_PER_WEEK
    rate = v.pace_per_week or 0.0
    remaining = max(GO_LIVE_PRELIM_CLOSES - v.verdict.n, 0)
    if remaining == 0:
        eta = f"prelim {GO_LIVE_PRELIM_CLOSES}-close sample complete"
    elif rate <= 0:
        eta = "ETA unknown (no verdict closes in the window)"
    else:
        eta = f"~{remaining / rate:.0f} wk to {GO_LIVE_PRELIM_CLOSES}"
    return (f"{indent}pace: {rate:.1f} verdict closes/wk over the trailing "
            f"{PACE_WINDOW_DAYS}d (target {lo:.0f}-{hi:.0f}/wk) — {eta}")


def db_path_default() -> Path:
    """``data/ait_state.db`` relative to the repo root."""
    return Path(__file__).resolve().parents[3] / "data" / "ait_state.db"
