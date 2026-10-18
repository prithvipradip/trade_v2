"""Fixed limits: values the system deliberately does NOT let anyone configure (W70).

WHY THIS MODULE EXISTS
    The operator's rule (memory ``feedback-visible-vs-tunable``): *visible* is not the
    same as *tunable*. Everything an operator may change lives in ``config.yaml`` with
    bounds and a stated consequence. The values below are different in kind: each is a
    backstop that exists to catch the system (or its configuration) being wrong, a rule
    imposed from outside, or the line that a config value is checked AGAINST. A key in
    ``config.yaml`` is tunable regardless of any comment, and that file is shared
    between machines through git and edited by people and AI sessions, so putting these
    there would let one edit loosen both a gate and its check.

    They are therefore code, in ONE place, named, documented, logged at startup
    (``fixed_limits_in_effect``) and listed in docs/RUNBOOK.md ("Fixed limits").

HOW TO CHANGE ONE
    Edit this file, update the RUNBOOK row (a drift test compares them), and record the
    change in PLAN.md. That is a reviewed code change on purpose.

THIS MODULE MUST STAY A LEAF: stdlib only, no imports from ``ait``. Settings, the
executor, the PDT guard and the orchestrator all import it; none may import them back.

The fast-monitor tick (30 s) is NOT duplicated here: its single source is
``ait.utils.cadence.FAST_MONITOR_CADENCE_SECONDS`` and the timeline documentation
references that.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Order-rate backstop ---------------------------------------------------------
ORDER_RATE_LIMIT_MAX = 8
ORDER_RATE_LIMIT_WINDOW_S = 60

# --- NBBO sanity gates (combo entry quote, checked just before placement) --------
NBBO_DISORDERLY_SPREAD_RATIO = 1.0
NBBO_STALE_SIGNAL_DEVIATION = 0.35
NBBO_MID_FLOOR = 0.05

# --- Live-profile floors (startup assertion when trading.mode != paper) ----------
LIVE_MAX_SPREAD_PCT = 0.15
LIVE_MIN_VOLUME = 10

# --- Iron-condor wing: the smallest wing a budget-capped condor may shrink to ----
# The CONFIGURABLE minimum wing is backtest.wing_floor_dollars (env AIT_IC_WING_FLOOR,
# default $2.00); it applies when the per-trade budget does not bind. When it does,
# the wing is cut to what the budget can carry and may drop below that floor, but
# never below this: $1 is the narrowest wing the listed-strike grid offers.
IC_BUDGET_CAPPED_MIN_WING = 1.0

# --- Pattern-day-trader rule (regulatory) ----------------------------------------
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_DAYS = 5

# --- Broker open-order snapshot cache (API-call pacing, not a trading preference) -
BROKER_SNAPSHOT_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class Invariant:
    """One fixed limit: the value plus the explanation the RUNBOOK and logs reuse."""

    name: str
    value: float
    unit: str
    what: str
    why_fixed: str
    enforced_in: str


INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        "ORDER_RATE_LIMIT_MAX", ORDER_RATE_LIMIT_MAX, "orders",
        "Hard ceiling on orders placed within ORDER_RATE_LIMIT_WINDOW_S (entry and roll "
        "placement share one bucket); the next order is refused and logged CRITICAL.",
        "A backstop against a MALFUNCTION (a logic bug spamming placements). The daily "
        "budget bounds intents; this bounds a bug. If config could raise it, the bug it "
        "exists to stop would not be stopped.",
        "execution/executor.py (entry placement and roll placement)"),
    Invariant(
        "ORDER_RATE_LIMIT_WINDOW_S", ORDER_RATE_LIMIT_WINDOW_S, "seconds",
        "The rolling window ORDER_RATE_LIMIT_MAX is counted over.",
        "Part of the same backstop as ORDER_RATE_LIMIT_MAX.",
        "execution/executor.py"),
    Invariant(
        "NBBO_DISORDERLY_SPREAD_RATIO", NBBO_DISORDERLY_SPREAD_RATIO, "ratio",
        "Reject a combo entry whose (ask - bid) is wider than this multiple of its own mid.",
        "A backstop against a bad quote (delayed data, crossed or locked market). Distinct "
        "from the configurable per-contract options.max_bid_ask_spread_pct filter.",
        "execution/executor.py (combo entry NBBO check)"),
    Invariant(
        "NBBO_STALE_SIGNAL_DEVIATION", NBBO_STALE_SIGNAL_DEVIATION, "fraction",
        "Reject a combo entry whose signal price differs from the live mid by more than "
        "this fraction.",
        "A backstop against acting on a signal priced on a market that has since moved; "
        "loosening it lets an order go out at a price unrelated to the market.",
        "execution/executor.py (combo entry NBBO check)"),
    Invariant(
        "NBBO_MID_FLOOR", NBBO_MID_FLOOR, "dollars",
        "The two NBBO ratio tests apply only when |mid| exceeds this.",
        "Below it the ratios blow up and mean nothing; keeps the gates from firing on "
        "near-zero mids.",
        "execution/executor.py (combo entry NBBO check)"),
    Invariant(
        "LIVE_MAX_SPREAD_PCT", LIVE_MAX_SPREAD_PCT, "fraction",
        "Maximum bid-ask spread (AIT_LIQ_MAX_SPREAD and options.max_bid_ask_spread_pct) "
        "allowed when the bot is started in a non-paper mode.",
        "It is the line config values are checked AGAINST. The paper config is "
        "deliberately relaxed (0.40); a floor in the same file would let one edit loosen "
        "both the setting and its check.",
        "bot/orchestrator.py (GOV-5 live-profile assertion at startup)"),
    Invariant(
        "LIVE_MIN_VOLUME", LIVE_MIN_VOLUME, "contracts",
        "Minimum option volume (AIT_LIQ_MIN_VOLUME) allowed when started in a non-paper "
        "mode.",
        "Same reason as LIVE_MAX_SPREAD_PCT; docs/RUNBOOK.md's go-live checklist "
        "documents both numbers.",
        "bot/orchestrator.py (GOV-5 live-profile assertion at startup)"),
    Invariant(
        "IC_BUDGET_CAPPED_MIN_WING", IC_BUDGET_CAPPED_MIN_WING, "dollars",
        "Narrowest wing a budget-capped iron condor may shrink to. Below the configurable "
        "wing floor (AIT_IC_WING_FLOOR / backtest.wing_floor_dollars) only when the "
        "per-trade budget cannot carry the floor.",
        "A physical limit, not a preference: $1 is the narrowest wing the listed-strike "
        "grid offers. Lowering it asks for strikes that do not exist; raising it defeats "
        "the budget cap on small accounts. Live and the research engine share it.",
        "strategies/iron_condor.py (_vol_scaled_width) and backtesting/engine.py"),
    Invariant(
        "PDT_MAX_DAY_TRADES", PDT_MAX_DAY_TRADES, "trades",
        "Day trades the bot allows itself inside PDT_WINDOW_DAYS trading days.",
        "A regulatory rule (FINRA pattern-day-trader, margin accounts under $25,000 are "
        "flagged at 4 in 5 business days), not a preference. The SWITCHES are config: "
        "account.pdt_protection and account.pdt_account_under_25k.",
        "risk/pdt_guard.py"),
    Invariant(
        "PDT_WINDOW_DAYS", PDT_WINDOW_DAYS, "trading days",
        "The rolling window PDT_MAX_DAY_TRADES is counted over.",
        "Part of the same regulatory rule.",
        "risk/pdt_guard.py"),
    Invariant(
        "BROKER_SNAPSHOT_TTL_SECONDS", BROKER_SNAPSHOT_TTL_SECONDS, "seconds",
        "How long the all-clients open-order snapshot is re-used inside and across "
        "check_fills passes.",
        "A cache lifetime that avoids repeated broker round trips within one pass; an "
        "implementation detail of API-call pacing, not a trading preference. Passes are "
        "~30 s apart so it never hides a change.",
        "execution/executor.py (_broker_open_order_ids)"),
)


def invariant_values() -> dict[str, float]:
    """name -> value for every fixed limit (the startup-log payload)."""
    return {i.name: i.value for i in INVARIANTS}
