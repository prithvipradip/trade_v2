# W17 batch 2 — probe results (2026-09-16)

Six probes, executing real code. Mocks only at the external edge (broker, the authority import);
never the logic under test. Fixture DBs throughout — `data/ait_state.db` was never written.

## Counts

| | |
|---|---|
| Probed | 6 |
| **VERIFIED** | **6** (one with its consequence corrected downward) |
| REFUTED | 0 |

Zero refutations again. As in batch 1 this is a **small, pre-selected sample, not a hit rate** —
these leads had already survived planning-phase code reading, during which **1(b) was refuted and
three register claims corrected**. The refutations happened before this table, not in it.

---

## w17-06 · A solved per-contract sigma selects a DIFFERENT STRIKE — VERIFIED

**Probe:** real QQQ chain from `data/historical.db` (`option_spread_samples`, 117 puts,
S≈682, dte=33, chain-median IV 0.2361). Two identical contract sets — deltas from the chain median
vs each contract's own real IV — through the **real** `Strategy._find_strike_by_delta`.

```
 target      median-sigma pick      solved-sigma pick   verdict
   0.10          625 (d=0.092)          605 (d=0.097)   DIFFERENT STRIKE (20 pts)
   0.15          640 (d=0.160)          625 (d=0.147)   DIFFERENT STRIKE (15 pts)
   0.20          645 (d=0.188)          640 (d=0.200)   DIFFERENT STRIKE (5 pts)
   0.25          655 (d=0.252)          650 (d=0.247)   DIFFERENT STRIKE (5 pts)
   0.30          660 (d=0.287)          660 (d=0.302)   same
```

**4 of 5 targets select a different strike**, including **0.20 — the value `iron_condor.py:169,177`
actually uses**. No trade/no-trade flip occurred in this slice; the `tolerance=0.07` guard did not
fire, so that half of the claim is **unproven here** and should not be cited.

**Consequence:** `df704d7` changed which strikes the live bot sells. That is a strategy change,
shipped under a data-correctness heading.

## w17-07 · A below-intrinsic quote is laundered into a healthy-looking contract — VERIFIED

**Probe:** a deep-ITM call priced below intrinsic (K=60, S=100, intrinsic 40, mid 5.0) through the
real `_calculate_greeks`. py_vollib raises `BelowIntrinsicException` — *"this price is impossible"*.

```
implied_vol : 0.24          <- chain median, silently substituted
delta       : 0.9999999...
gamma       : 3.08e-14
log lines naming the failure : NONE
has_two_sided_quote=True  delta!=0=True  iv>0=True
```

An impossible quote emerges indistinguishable from a healthy deep-ITM contract. `iv_solved_from_price`
does not increment (it counts successes only), and nothing logs.

**Honest scoping:** a ~1.0 delta will not win a 0.20-delta argmin, so this specific contract is
unlikely to be *selected*. The real damage is that it inflates apparent chain health — most
importantly, `degraded_greeks` (`iron_condor.py:161`) cannot fire once backfill has made every
delta non-zero, so the gate designed to catch a degraded chain is disarmed by the degradation.

## w17-08 · `verify_can_trade` fails open — VERIFIED, **consequence corrected**

**Probe:** patched `qualifyContractsAsync` to raise (nothing reached a broker).

```
ConnectionError  -> returns True   placeOrder called: False
TimeoutError     -> returns True   placeOrder called: False
RuntimeError     -> returns True   placeOrder called: False
```

It answers *"yes, this session can place orders"* without having placed the probe.

**CORRECTION to R25's framing and to my own re-scoping.** I called this "a pre-trade safety probe
that fails open", implying trading proceeds unguarded. Tracing the consumer:
`orchestrator.py:2117` is the sole caller, inside `_check_trading_enabled` — and **both of its
callers (`:513`, `:900`) discard the return value**. Nothing gates trading on it.

What `can_trade=False` actually does is send a Telegram alert. So the fail-open's real consequence
is that **the READ-ONLY alarm never fires** — returning the operator to the exact silent failure
the probe exists to prevent ("nothing fills, and it's invisible until end of day"). Severity is
`ops_reporting`, **not** `live_money`. It also logs `verify_can_trade_error`, so it is not fully
silent.

## w17-09 · The go-live slippage gate measures the wrong population — VERIFIED (3 claims)

**Probe:** fixture DB, real `compute_go_live_verdict`.

Fixture: one trade, entry credit 1.00; an **entry** fill slipping 2%; an **exit** fill slipping 30%
(`|0.40−0.10|/1.00`); one row with `live_mid=0`.

```
executions rows written : 3
slip_n reported         : 2      <- the live_mid=0 row vanished, uncounted
slip_median_pct         : 16.0   <- the exit fill is in the pool
```
The exit fill drags the median to **16%**, which **fails the 8% gate** — on the strength of a fill
the criterion claims not to measure, divided by the *entry* credit.

At n == 1:
```
slip_n=1  median=0.50%   available=True  passed=True
rendered: "median 0.5% of credit over trailing 1 fills, all strategies (gate <=8%)"
```
**A single fill authorises the gate.** UNAVAILABLE fires only at n == 0.

All three sub-claims verified: exit fills pooled, drops invisible, no sample floor.

## w17-10 · The restate script would zero data the current writer produces — VERIFIED

**Probe:** the **real** `scripts/restate_r16_executions.py` `main()`, dry-run, with `DB` repointed
at a fixture holding one legitimate single-leg fill (`con_id=711234567`, `live_mid=3.52`) and one
BAG row.

```
leg rows carrying combo context : 1
sample rows currently feeding the gate:
  T-LONGCALL con=711234567 price=3.55 live_mid=3.52 signal=3.5
DRY RUN: would clear context on 1 leg rows and abs() price on 1 BAG rows.
```

The script prints the row under **"sample rows currently feeding the gate"** and then declares it
would clear it. Its predicate is `con_id != BAG_CON_ID`, true of every single-leg order, while the
writer branches on `is_combo` (`executor.py:1530-1537`) and stores `live_mid` deliberately on that
shape.

**Compounding:** `state.py:913-918`'s upsert never zeroes a known value, so a zeroed row cannot
self-heal; `go_live.py:327`'s `WHERE live_mid > 0` then hides it while `slip_n` still reports only
survivors (w17-09). Destruction is invisible at the gate by construction. This already happened
once — three single-leg fills zeroed 2026-08-10.

## w17-11 · `_capital_base()` falls back to a 3.7× wrong denominator, silently — VERIFIED

**Probe:** injected a failing `ait.config.runtime_env` and called all three copies.

```
authority runtime_env.capital_base() = 723,071.22  (source=live_nlv)
distinct implementations among the 3 copies: 1  -> IDENTICAL

healthy:                    all three -> 723,071.22
authority import FAILS:     all three -> 196,000.00   (73% off, 3.7x understated)
```

Confirms both halves: the copies are **identical to each other** (so R25's register, which never
claimed otherwise, was right), and all three lack the authority's live-NLV tier, reverting to the
exact pre-R16 hardcoded literal behind a bare unlogged `except`. `app.py:1004` then raises this to
the 12th power for the annualised-return metric.

---

## Proposed fix order

| Wave | Content |
|---|---|
| **W18** | w17-09 + w17-10 together — they form one loop: the gate authorising real capital measures the wrong population, and a script with no tests can silently destroy its inputs. Add a side filter, a sample floor, a drop count; give the script a date cutoff and tests |
| **W18** | w17-11 — give the three copies the authority's live-NLV tier, or delete the fallback and fail loud |
| **W19** | w17-06 + w17-07 — decide whether per-contract solved IV is *wanted* for strike selection (it may be an improvement), then make the choice explicit and instrument the solver's failures |
| **W19** | w17-08 — count the probe's own failures rather than reporting success |

## Still unproven, do not cite

- The `tolerance=0.07` trade/no-trade flip (w17-06) — not observed in the probed slice.
- The residual from 1(b): nothing bounds the solver in a chain-wide `modelGreeks` outage, and
  `scan_symbol_timing` has no `greeks_ms` stage to attribute it. Mechanism only; unmeasured.
