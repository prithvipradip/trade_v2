# Studies and findings

The register of what research studies and measurements have shown, and whether each result may be cited. For anyone deciding something and asking "has this been tested, and can I rely on it?". Not for: how to run a study or read a run
([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 9), or the decisions themselves (`PLAN.md`). Terms: [GLOSSARY.md](GLOSSARY.md).

**This is the only document that holds study results.** Every other document explains how things work and points here. A result that is not recorded here, or is recorded as not citable, is not evidence.

## Rules

1. **One entry per study or measurement**, added when it is run, using the template below. Never edit an entry's result afterwards; if a later run changes the picture, add a new entry and mark the old one `superseded`.
2. **Status** is one of:
   - `pre-registered`: the pass/fail rule was written in `PLAN.md` *before* the run (its `rule:` names that heading). Citable.
   - `exploratory`: looking around; no rule written first. **Not citable** as evidence for a decision; a decision that leans on one cites the file and commit in its own `PLAN.md` entry.
   - `migrated, unreviewed`: found in another document or a code comment when the documentation was restructured. **Not citable** until someone reviews it, re-runs it on the current engine, or retires it.
   - `superseded` / `invalidated`: kept for the record, with the entry that replaces it or the reason it is wrong.
3. **Comparability.** A result is only comparable with another made under the same research defaults and engine. Record the epoch (for example the value of `backtest.max_concurrent_positions`, `engine_model`, `gates_mode`, the history span).
4. **Statistics only.** Never put per-trade prices, per-trade P&L or any account identifier here or in `PLAN.md` (they are shared through git with the other box). Report counts, rates and errors by band.
5. **No decision here.** What was decided because of a finding is in `PLAN.md`; link its heading.

## Entry template

```
### F-NNN  <short title>
- Status: pre-registered | exploratory | migrated, unreviewed | superseded | invalidated        Citable: yes | no
- Question: <one sentence>
- Rule: <PLAN.md heading, or "exploratory">
- Study file / script: studies/<name>.yaml | scripts/<script>.py <arguments>
- Run record: reports/runs/<run_id>/   Commit: <hash>   Data span: <from> to <to>   Epoch: <engine / defaults that matter>
- Result: <statistics only>
- Caveats: <what it does not show>
- Fed decision: <PLAN.md heading, or "none">      Superseded by: <F-NNN or "-">
```

## Index

| ID | Title | Status | Citable |
|---|---|---|---|
| F-001 | Gate parity A/B at entry DTE 14 (volatility cap, trending-down veto, VIX halt) | migrated, unreviewed | no |
| F-002 | Share of sessions the ex-dividend entry block refuses | migrated, unreviewed | no |
| F-003 | Whether the expiry-day flatten moves outcomes while the take-profit ladder is on | migrated, unreviewed | no |
| F-004 | Credit stop-loss variants versus the short-strike touch close (17-month walk-forward) | migrated, unreviewed | no |
| F-005 | Fast-monitor tick latency | migrated, unreviewed | no |
| F-006 | Order acknowledgement and fill times (entries) | migrated, unreviewed | no |
| F-007 | Range-model confidence floor 0.65 versus 0.55 | migrated, unreviewed | no |

## Migrated, not yet reviewed

Text copied verbatim from where it was found. Nothing below has been re-checked against the current engine, data or defaults.

### F-001  Gate parity A/B at entry DTE 14
- Status: migrated, unreviewed        Citable: no
- Source: former RUNBOOK section "One-sided entry gates", script `scripts/gate_parity_ab.py`; first result dated 2026-10-13.
- Text: "at entry DTE 14 the veto reached 0 / 0 / 3 days (QQQ / SPY / IWM), the vol cap 0, and no trade entered on a day whose previous VIX close was >= 28 (18 such sessions); the same at DTE 1, 3 and 7 on QQQ and IWM."
- Note: the "veto" is the research trending-down regime veto; the "vol cap" is `backtest.max_entry_vol_annual`; the VIX halt is applied after the fact (no slot re-use).

### F-002  Share of sessions the ex-dividend entry block refuses
- Status: migrated, unreviewed        Citable: no
- Source: former RUNBOOK section "The ex-dividend entry block", live rule with the projection in `ait.data.dividends.project_next_ex_div`.
- Text: "It blocks 14-17% of sessions at entry DTE 14."

### F-003  Expiry-day flatten while the take-profit ladder is on
- Status: migrated, unreviewed        Citable: no
- Source: former RUNBOOK section "Searching the entry DTE", script `scripts/dte_dimension_flatness.py`.
- Text: the flatten "is not searched: `scripts/dte_dimension_flatness.py` showed it moves nothing while the take-profit ladder is on."

### F-004  Credit stop-loss variants versus the short-strike touch close
- Status: migrated, unreviewed        Citable: no
- Source: comment on exit rule 1 in `src/ait/execution/portfolio.py` (credit structures); not a document.
- Text: "(evidence backtest, 17mo walk-forward, 26 matched trades): flat credit stops fired THROUGH their trigger on gaps ... and every flat level underperformed both no-stop and touch-close (touch +$756/PF 1.38 > none -$24 > 1.25x -$134 > 2.0x -$1,067)."
- Note: this is the evidence behind the default of no flat credit stop; the dollar figures are the study's aggregate over its matched trades.

### F-005  Fast-monitor tick latency
- Status: migrated, unreviewed        Citable: no
- Source: former RUNBOOK section "Order lifecycle timeline" (an operational measurement, not a study).
- Text: "measured 2026-09: median 30.9 s between ticks, 98% under 35 s, worst 100 s."

### F-006  Order acknowledgement and fill times (entries)
- Status: migrated, unreviewed        Citable: no
- Source: former RUNBOOK section "Order lifecycle timeline", "What the floors rest on, honestly" (operational measurements).
- Text: one entry order (2026-09-28) "was acknowledged by IBKR in under 1 s; two more (QQQ and SPY, Prithvi's box) filled in 11.4 s and 14.4 s; no exit order has been timed and no mid-session disconnect occurred in the retained logs." The `order_ack_latency` and `order_absent_judged` log events collect the real distribution.

### F-007  Range-model confidence floor 0.65 versus 0.55
- Status: migrated, unreviewed        Citable: no
- Source: description of the setting `ml.range_min_confidence` in `src/ait/config/settings.py`; not a document.
- Text: "0.65 beat 0.55 across every backtest metric."
