# W17 batch 2 — pre-probe findings (2026-09-16)

Written **before** batch 2's probing run, because these results were produced during planning and
planning artefacts do not survive a session. Several already change the batch, and four of them
**correct claims in R25's own register**.

## Lens status — read this before citing anything

The two-lens standard requires a code-reading lens **and** an execution probe, both conceding.
This document mixes both, and says which is which per finding:

- **[BOTH]** — code read *and* executed. Citable as a verdict.
- **[CODE]** — code-reading only. A candidate. Batch 2 owes it a probe.

---

## Lead 1 — `options_chain.py`, live since `df704d7` (bot restarted 22:33:24)

### 1(a) Strike selection — VERIFIED **[BOTH]**, with a sharper mechanism than registered

`delta` is recomputed only for contracts meeting **all three**: `delta == 0`, IV missing or
`< MIN_PHYSICAL_IV`, and `has_two_sided_quote`. That looked like a narrow tail — it is not.
`has_two_sided_quote` is *also* the precondition for `is_liquid` (`options_chain.py:212-214`), so
**every contract whose delta changed is by construction a strike-selection candidate.**

Selection is a pure argmin over delta, `strategies/base.py:204-205`:
```python
best = min(contracts, key=lambda c: abs(abs(c.delta) - target_delta))
if tolerance is not None and abs(abs(best.delta) - target_delta) > tolerance:
    return None
```
Change `delta`, change the argmin. Line `:205` additionally converts a delta change into a
**trade/no-trade** decision. Callers: `iron_condor.py:169,177` (0.20 put/call), `spreads.py:51,58,
132,139`, `straddles.py:140-141`, `long_options.py:47,107`. Further delta-reading gates:
`degraded_greeks` (`iron_condor.py:161-162`), the `[0.15, 0.30]` `delta_band` reject
(`:191-200`), and `filter_by_delta` (`options_chain.py:367,371`).

**Measured magnitude (executed):** at S=600, 30 DTE, an IV move of 0.20→0.28 shifts delta at
K=555 by **0.0719**, against a delta gap between adjacent $5 strikes of **0.0193** — the shift is
**3.7× the strike spacing**. A solved IV differing from the chain median by a realistic skew
amount moves the selected strike by several strikes.

**Mitigating, and load-bearing:** `atm_iv` and `expected_move` are frozen in
`OptionsChain.__post_init__` (`:325-330`), which runs at construction (`:770-778`) **before**
`_calculate_greeks` (`:500`). `filter_by_delta` (`:375`) and `filter_liquid` (`:434`) explicitly
propagate `atm_iv`. So the EM sanity gate (`iron_condor.py:204-215`) and `_persist_daily_iv` →
`iv_rank` read **pre-backfill** IV. Fragile: any future path that rebuilds a chain without passing
`atm_iv` flips this silently.

### 1(b) Performance — **REFUTED [BOTH]**, and my registered phrasing was mechanically wrong

R25 said: *"removing the `continue` early-exit means every contract in a ~1,500-contract chain may
now run a py_vollib Newton solve."* **That is wrong twice over.**

1. **The `continue` never guarded a solver call.** The solver did not exist in the parent commit.
   The removed `if c.delta != 0: continue` guarded the four *analytic* BS calls. The
   `implied_volatility()` call is new code with **its own** guard (`:969-970`), gated on missing-IV
   + two-sided-quote — not on `delta`.
2. **It does not fire on 1,500 contracts.** Live logs (`iv_solved_from_price`, 2026-09-14) show
   **2, 55, 16** (SPY per expiry), **8, 31, 12** (QQQ), **6, 7, 35** (IWM). Max observed: **55**,
   roughly 5% — IBKR supplies `impliedVol` for the rest.

**Measured cost (executed):** `implied_volatility()` averages **56.4 µs/contract** over a realistic
strike ladder. At the observed worst case (55 contracts) that is **~3 ms**; extrapolated to a
hypothetical 1,500 it is **84.5 ms**. Against a `chain` stage of **100,130 ms (IWM)** to
**213,614 ms (SPY)**, that is **0.04–0.08%**.

The chain stage is dominated by network waits, not math: `md_batch_size = 25`, `md_wait_s = 3.0`
with `await asyncio.sleep(md_wait_s)` per batch ⇒ **~156 s of pure sleep** out of SPY's 213 s.

**Residual worth registering [CODE]:** nothing *bounds* it. There is no cap on `target_strikes`,
and this file's own comments (`:672-682`, `:936-939`) document recurring chain-wide `modelGreeks`
loss. In that regime every quoted contract (~500/expiry) enters the solver at once — and
`scan_symbol_timing` has only `pre_chain`/`chain`/`total` stages, so it would show `chain` growing
with **no way to attribute solver from network**. A `greeks_ms` stage does not exist.

### 1(c) Exception masking — VERIFIED **[CODE]**, plus two findings I had not registered

```python
try:
    c.implied_vol = implied_volatility(c.mid, S, c.strike, t, r, flag)
    iv_solved_from_price += 1
except Exception:
    c.implied_vol = _chain_iv
```
- Bare `except Exception` catches `BelowIntrinsicException`/`AboveMaximumException` — py_vollib's
  signal that the observed price is outside arbitrage bounds, i.e. a stale, crossed or genuinely
  mispriced quote — identically to a convergence failure. Neither exception name appears anywhere
  in `src/` or `tests/`.
- No log in the handler.
- **NEW: the counter cannot see failures.** `iv_solved_from_price` is incremented **inside the
  `try`, after** the assignment succeeds. It counts successes only, and there is no attempt
  counter, so failures cannot even be derived by subtraction.
- **NEW: the compounding case.** `_chain_iv` falls back to a hardcoded **`0.30`** when the chain
  has no valid IV at all (`:930`). In a full `modelGreeks` outage every backfilled delta derives
  from a flat 30% guess — **and `degraded_greeks` (`iron_condor.py:161`) will not fire, because the
  backfill made every delta non-zero.** The gate designed to catch this is disarmed by the thing it
  should catch.
- **NEW: model mismatch.** The solve uses `py_vollib.black_scholes` — no dividend yield — with
  `r = 0.05` hardcoded (`:919`). SPY/QQQ/IWM all pay dividends, and IBKR's `impliedVol` uses a
  dividend schedule. A solved IV and an IBKR IV in the same chain are on **different models**, and
  `_chain_iv` takes a median over that mixed population.

**Test coverage:** five tests call `_calculate_greeks`, and **none asserts a market-realistic IV** —
`:267` and `:293` assert literally `> 0` (passing for 0.0001 or 40.0); `:307`'s `approx(0.20)` is
the *chain-median fallback* branch, not a solved IV. The `except` fallback, the counter, and the
newly-reachable gamma/theta/vega branches have **no coverage at all**.

---

## Lead 2 — `ibkr_client.py:350` — **RE-SCOPED [CODE]**

R25 flagged *"places a **real** SPY BUY LMT with hardcoded symbol/qty/price on every startup."*
Factually true, misleading in emphasis. It is a deliberately **non-fillable** `BUY 1 SPY @ $1.00`
against a ~$600 underlying, cancelled immediately, and it exists to catch a documented silent
killer: a Gateway session that logs in READ-ONLY, rejecting every order with Error 321 while the
bot still logs `trade_executed`. The design is sound and the docstring reasons about all of it.

**The real defect is the last line:** `except Exception: ... return True` — *"probe failed for
another reason; don't block"*. A **pre-trade safety probe that fails open**: any error in the probe
reads as "trading works". Secondary: the cancel is wrapped in a bare `except Exception: pass`
(`:358`), so a failed cancel leaves the probe order working, against the code's own comment
"never leave the probe live".

This belongs to the fail-open class, so it doubles as calibration for batch 3.

---

## Lead 3 — `go_live.py` slippage — **STRONGER than registered [CODE]**

- **No side filter, confirmed**, against **three** independent entry-only declarations:
  `go_live.py:186` (`GATE_SLIP_PCT = 8.0  # median entry slip`), `shadow_referee.py:226`, and the
  gate comment at `:364-369`.
- **Worse than unfiltered:** every row is divided by `t.entry_price` (`:371`), so an **exit** fill's
  `|price − live_mid|` is normalised against the **entry credit**. `trades.exit_price` exists
  (`state.py:123`) and is not used.
- **A `side` column exists** (`state.py:232`), written from the broker at `executor.py:1551`, and
  read in exactly one non-reporting place (`executor.py:1929`). **Zero reporting queries use it.**
  Honest caveat: `side` is direction (BOT/SLD), not lifecycle — a short-condor entry is SLD, a
  long-call entry is BOT — so it is not a complete discriminator alone. But "no clean column
  exists" is not the current state either; the query uses **no** discriminator at all.
- **W12 caused this.** Before W12, exit fills carried no `live_mid`, so `WHERE live_mid > 0` made
  the query *accidentally* entry-only. W12 (`executor.py:2793`) deliberately populated exit
  `live_mid` — and silently mixed the pool a gate labelled "entry slip" consumes. A fix created
  the defect.
- **`slip_n` reports survivors, not sample size.** Three uninstrumented shrinkages (`live_mid > 0`,
  `entry_price > 0`, the INNER JOIN dropping orphans) plus a `LIMIT 20` cap — so the gate renders
  **identically at 20 fills and at 2,000**.
- **n == 1 passes the gate.** UNAVAILABLE fires only at n == 0. The trend sub-clause degrades at
  n < 4; the criterion itself does not degrade at all.
- The same file already implements the right pattern for criteria [1] and [3] — `v.excluded`
  (`:313-318`, `:339-344`) and `dd_missing_car` (`:359`, rendered `:434-435`). Criterion [4] is the
  one that does not declare its coverage hole, violating the module docstring's own promise at
  `:296-301`.

## Lead 3b — `restate_r16_executions.py` — **WORSE than registered; has already destroyed data [CODE]**

- Predicate is `con_id != BAG_CON_ID` — true of **every single-leg order** — while the writer it
  claims to align with branches on `is_combo` (`executor.py:1530-1537`). They disagree exactly
  where the writer legitimately stores `live_mid`.
- **It already destroyed real data**: three valid single-leg `long_call` entry fills zeroed on
  2026-08-10 (documented in `reports/blindspot_composition_hunt_20260825.md:228`; R24 re-confirmed
  the code unchanged).
- Dry-run **is** the default (good). But: no backup, no date cutoff, no confirmation, no row cap,
  no reversibility. Its provenance record uses `INSERT OR REPLACE`, so a second apply **overwrites
  the first run's record** of what it destroyed.
- Idempotent narrowly; **unsafe to re-run later** — the predicate is time-invariant, so it re-arms
  the moment a single-leg strategy is enabled.
- **Zero tests** — while its sibling `restate_r16_dailystats` *is* tested
  (`test_r17_dormant_strategy_fixes.py:106-149`). The harness pattern exists and was not applied.
- **Compounding:** `state.py:913-918`'s upsert never zeroes a known value, so a zeroed row
  **cannot self-heal** from a later broker sweep; `go_live.py:327`'s `WHERE live_mid > 0` then
  hides it without changing `slip_n`. Destruction is invisible at the gate by construction.

---

## Lead 5 — `_capital_base()` — **PARTLY REFUTED [BOTH]**, and my citation was wrong

- **Citation error:** it is at `app.py:42-55`, not `:79-90` (`:81-87` is `_get_duck()`).
- **Correction to my own planning note, not to the register.** While planning I asserted the
  register claimed these three copies "diverge" from each other and that this was false. Re-reading
  it, **the register makes no such claim** — it says they each wrap the declared single authority
  in `except Exception`, which is accurate. The copies *are* byte-identical to each other; that is
  a finding about the code, not a correction of R25. Recorded so the correction count stays honest:
  it is **three** register corrections, not four.
- **The real divergence is copies-vs-authority.** `runtime_env.capital_base()` resolves
  env → **live NLV from `bot_state['last_net_liquidation']`** → default. The three local fallbacks
  skip the live-NLV tier entirely and go straight to a hardcoded `196000` — restoring the exact
  pre-R16 bug the authority was created to fix, behind a **bare unlogged `except`**.
- R17's enforcement does not protect them: `capital_base_source()` is checked only at
  `orchestrator.py:500`, a **different process** from the dashboard.

**NEW finding, verified live [BOTH]:** `bot_state['account_value']` — which `app.py:995` queries —
is **never written anywhere in `src/`**. The live DB holds only `last_net_liquidation`:
```
bot_state account_value    : None          <- app.py:995 reads THIS
runtime_env.capital_base() : 723,071.22    (source=live_nlv)
```
The dashboard lands on the right number **only** because the authority has the live-NLV tier its
own fallback lacks. If that import ever fails, the displayed base becomes **196,000 against an
actual 723,071 — a 3.7× error**, silent, and compounded twelvefold by the `** 12` annualisation at
`app.py:1004`.

---

## Register corrections to apply

**Three** of R25's own claims, corrected in `reports/full_audit_r25_20260914.md` (I initially
counted four — see the Lead 5 note; the "copies diverge" item was a misreading of my own register,
not an error in it):

1. `_capital_base()` is at `app.py:42-55`, not `:79-90`.
2. The `options_chain` performance claim is mechanically wrong — the removed `continue` guarded the
   analytic BS calls, not a solver that did not yet exist — and the magnitude is refuted.
3. `ibkr_client.py:350` should read as a **fail-open probe**, not "it places a real order".
