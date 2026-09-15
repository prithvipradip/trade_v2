# W17 batch 3 — fail-open census (2026-09-16)

**STATUS: SINGLE-LENS. These are candidates, not findings.** Every entry below survived a
code-reading pass with its consumer traced, but **none has an execution probe yet**. Batch 3 is the
work of probing the shortlist. R24 refuted 28 of 137 candidates at exactly this stage; R25 refuted
6 of its own, three of them mine. Do not cite anything here as a defect until it is probed.

## Why this document exists

The census was produced during W17 batch-2 planning and is the entire input to batch 3. Planning
artefacts do not survive a session — this repo's ledger and `reports/` are the memory — so it is
written down before any probing starts.

## Scale — R25 undercounted this substantially

R25 §5 registered *"54 silent-swallow sites on the live/ops path (of 419 `except Exception`)"*.
An AST scan over the seven live/ops packages found:

| | |
|---|---|
| `except` clauses in scope | **355** |
| Silent-swallow candidates (body is only `pass`/`continue`/`break`, or returns a default with no log/raise/notify) | **177** |
| Traced to their consumer | **120** |
| — **DECISION-AFFECTING** (swallowing changes what the bot *does*) | **38** |
| — TELEMETRY-ONLY (a log line, counter, page or panel is lost; decisions unchanged) | **51** |
| — CLEANUP (teardown where failure genuinely does not matter) | **31** |
| Left deliberately unverdicted (numeric-coercion shims, ~57) | not guessed |

Scope: `bot/`, `execution/`, `risk/`, `broker/`, `monitoring/`, `orchestration/`, `data/`.

The ~57 residual sites are int/float/ISO-timestamp coercion shims with conservative defaults. They
*look* telemetry-or-low-grade, but each consumer was not traced, so **no verdict is asserted** on
them rather than inventing one.

## Ranked shortlist — the 12 to probe

Ordered by claimed consequence. Each needs an execution probe before it may be cited.

| # | Where | What goes wrong when it fires |
|---|---|---|
| 1 | `broker/account.py:228` `get_snapshot` | A malformed `AIT_SIMULATED_CAPITAL` (or one non-numeric balance string) silently disables the simulated-capital governor — NLV reverts to the **full real account**, and every %-based risk cap (3% per-trade, 20% portfolio, 20% concentration) and the position sizer scale up with it |
| 2 | `execution/reconciler.py:1083` `reconcile` | The `data/HALT_UNTRACKED` freeze file is never written, so entries are **not** frozen beside an unmanaged live position — while the Telegram page sent three lines later asserts "NEW ENTRIES FROZEN" |
| 3 | `bot/orchestrator.py:701` `_drain_exit_outbox` | One unparseable `exit_time` ⇒ `exit_d=None ≠ today` ⇒ realized P&L is **claimed and discarded**: never added to daily stats, never fed to `record_trade_result`. The daily-loss breaker goes blind to that loss |
| 4 | `execution/portfolio.py:779` `_evaluate_position` 3c | An earnings-lookup fault means the pre-earnings IV-crush exit never fires; a short-premium condor rides **through** the event at max gamma/vega. This is the `elif` slot that actually runs today (3b is documented inert) |
| 5 | `execution/portfolio.py:811` `_evaluate_position` 3d | Same shape for the FOMC/CPI/NFP flatten. `AIT_SKIP_MACRO_EVENTS` defaults to `"1"` in `CONTRACT_DEFAULTS`, so this path is **live**, not dormant |
| 6 | `bot/state.py:705` `get_partial_exits` | Corrupt `partial_exits` JSON ⇒ `[]` ⇒ `completed_levels` empty ⇒ an already-taken partial take-profit **re-fires**, over-closing the position |
| 7 | `bot/orchestrator.py:2878` `_execute_partial_exit` | The unfilled partial-exit order is **not** cancelled, against the code's own contract ("cancel so a re-trigger next cycle can't double-sell"). Next cycle re-sends; two sells can fill |
| 8 | `execution/reconciler.py:299` (twin `:719`) | A malformed broker position row drops a leg from `fresh_keys` ⇒ verdict `"gone"` for a **live** position ⇒ `book_vanished_trade` closes a real open trade in the DB and stops managing it |
| 9 | `bot/orchestrator.py:2934` `_execute_partial_exit` | `record_partial_pnl` lost ⇒ the breaker's daily P&L never sees partial-exit money — the exact defect the comment above it says was fixed |
| 10 | `bot/orchestrator.py:1074` `_trading_loop` | `watchdog.check_and_recover()` fails ⇒ component-down **recovery actions and their pages never run**, with zero log evidence. It is the only periodic caller of the recovery path |
| 11 | `orchestration/master.py:712` `health_check` | Calendar fault ⇒ `market_open=False` ⇒ `_in_post_market_window()` False at e.g. 11:00 ⇒ the post-retrain branch runs `stop(); start()` — **restarting the live bot mid-RTH with positions open** |
| 12 | `bot/orchestrator.py:4057` `_sessions_until` | Calendar fault ⇒ silent revert to the weekday heuristic the docstring itself proves makes the **pre-event blackout fail OPEN** (holiday miscount ⇒ entry admitted one session before the event) |

### The load-bearing code

```python
# account.py:217-229 — simulated-capital governor
if _sim:
    try:
        _cap = float(_sim); _real_nlv = float(values.get("NetLiquidation", 0)) or _cap
        _scale = min(1.0, _cap / _real_nlv)
        for _k in ("NetLiquidation", "BuyingPower", ...): values[_k] = str(float(values[_k]) * _scale)
    except (TypeError, ValueError):
        pass                      # <- cap silently gone; snapshot = real account

# reconciler.py:1078-1088
try:
    _P("data/HALT_UNTRACKED").write_text(...)
except Exception:  # noqa: BLE001
    pass
msg = (f"UNTRACKED OPTION POSITION ... — NEW ENTRIES FROZEN (data/HALT_UNTRACKED)")

# orchestrator.py:699-710
try:    exit_d = datetime.fromisoformat(row["exit_time"]).date()
except Exception:  # noqa: BLE001
    exit_d = None
if exit_d != today:               # claimed, never booked to stats/breaker
    ... ; continue

# state.py:703-707
if row and row[0]:
    try:    return json.loads(row[0])
    except json.JSONDecodeError:
        return []                 # "no partials taken" -> level re-fires

# master.py:709-731
try:    market_open = is_market_open()
except Exception:
    market_open = False           # -> else-branch: marker.unlink(); self.stop(); self.start()
```

## Notable non-findings — the census cut both ways

Recording these so batch 3 does not waste probes on them, and so the count is honest:

- **`orchestrator.py:2118` `_check_trading_enabled`** *looks* decision-affecting (`return True` on
  error) but **both callers (`:513`, `:900`) discard the result**. Only the read-only page is lost
  → TELEMETRY.
- **`ops_health.py:613, 643`** (`_default_market_open`, `bot_liveness`) fail toward
  `not_rth`/`heartbeat_missing` — the **non-green** side. Classified TELEMETRY here. Note this is
  the *same code* as batch 1's w17-02, which was VERIFIED: the distinction is that w17-02's defect
  is the *dashboard's* rendering of `ok=True`, not the swallow itself.
- **`orchestrator.py:3266` `_cooldown_blocks_entry`** fails **closed** (`is_credit=True` blocks the
  entry) — W15's fix, working as designed.
- **`master.py:218/225`** `_acquire_singleton_lock` — correct fail-**closed** after the W16 A4 fix.
- **`orchestrator.py:721, 804, 831, 1181, 2561`** `delete_state`/`set_state` housekeeping →
  CLEANUP: orphan `trade_maxloss_*` keys are swept at `orchestrator.py:3690`, and the risk cap only
  reads keys of *open* trades.
- **`executor.py:735`** is not a swallow at all — it is classified into the NBBO gate correctly.

## Pattern worth noting

Four of the twelve (#11, #12, and portfolio's #4/#5 by a different route) are **calendar faults**.
A single NYSE-calendar failure would concurrently: restart the live bot mid-session, open the
pre-event blackout, and disable both event-driven exits. That is a common-mode dependency no
single-site fix addresses, and it is the same dependency batch 1's w17-02 hit from the liveness
side. Worth probing as a *class*, not five instances.

## Probe hygiene for batch 3

- Set `PYTEST_CURRENT_TEST` before any supervisor call so `_log`'s file sink stays suppressed;
  confirm 0 probe lines reach `logs/orchestrator.log` afterwards.
- `data/HALT_UNTRACKED` and `data/*.db` are live state — probe against copies, never in place.
- #11 restarts the bot if driven carelessly. Drive `_in_post_market_window` and the branch
  predicate directly; do **not** call `health_check()` against the live `BotManager`.
