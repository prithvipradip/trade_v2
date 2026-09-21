"""Fast-monitor cadence and quote-TTL authority (W47, `numeric-pairs-07`).

Four numbers across four files have always had to agree with each other, and
until this module existed they agreed only by coincidence — each was a bare
literal, re-derived by hand wherever it was needed:

  - `FAST_MONITOR_CADENCE_SECONDS` — how often the position monitor evaluates
    stops/take-profits/exits (orchestrator.py `_trading_loop`).
  - `QUOTE_CACHE_TTL_SECONDS` — how long a quote is served from cache before
    a fresh one is fetched (data/market_data.py). Must stay STRICTLY BELOW
    the cadence: if the TTL ever reaches or exceeds the cadence, two
    consecutive monitor passes read the exact same cached print, and
    `ExitConfig.touch_confirm_ticks=2` ("two consecutive agreeing
    evaluations") becomes one stale print counted twice — the touch stop
    (the only credit-loss exit; see execution/portfolio.py's R16/R19
    incident notes) fires on, or is delayed by, phantom corroboration.
  - `MARKS_MISSING_ALERT_SECONDS` — the outage duration
    (`MARKS_MISSING_ALERT_TICKS` consecutive marks-missing ticks, in
    execution/portfolio.py) after which stop/take-profit protection being
    OFF gets a page instead of staying silent. Expressed in seconds, not
    ticks, so it survives a future cadence change unchanged.
  - `QUOTE_STALENESS_BUDGET_SECONDS` — `ExitConfig.max_quote_staleness_seconds`,
    the age at which an underlying quote is DEGRADED for the touch stop.
    Sized generously (6x cadence) precisely because the monitor's own cadence
    and cache TTL are what it is being generous ABOUT.

R16/R19 (execution/portfolio.py:122-134) already lived through the
consequence of this coupling drifting apart once — a cache/cadence
interaction misclassified a live feed as frozen and delayed a real
short-strike touch exit. `tests/test_w47_cadence_invariants.py` asserts the
relationships below hold against these constants directly (not against
re-typed copies), so a future edit to any one number fails loudly instead of
silently re-opening that incident.

Pure stdlib, zero internal imports — importable from config/settings.py
(which nothing here may import back from) with no cycle risk, the same
constraint execution/exit_policy.py documents for itself.
"""

from __future__ import annotations

#: The position monitor's evaluation cadence, in seconds. Every other
#: constant in this module is sized relative to this one.
FAST_MONITOR_CADENCE_SECONDS: float = 30.0

#: Quote cache TTL, in seconds. Must stay strictly below the cadence above.
QUOTE_CACHE_TTL_SECONDS: int = 15

#: Consecutive-marks-missing outage duration, in seconds, before the
#: "stop/take-profit protection is NOT running" page fires. Divided by the
#: cadence to get MARKS_MISSING_ALERT_TICKS (portfolio.py) — a tick count,
#: not a duration, is what the monitor loop can actually count.
MARKS_MISSING_ALERT_SECONDS: float = 300.0

#: Underlying-quote staleness budget for the touch stop, in seconds
#: (ExitConfig.max_quote_staleness_seconds' default). 6x the cadence: generous
#: enough that ordinary cadence jitter never falsely degrades a quote, tight
#: enough to still catch a genuinely stuck feed.
QUOTE_STALENESS_BUDGET_SECONDS: float = FAST_MONITOR_CADENCE_SECONDS * 6
