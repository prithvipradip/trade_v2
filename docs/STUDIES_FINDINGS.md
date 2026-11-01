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
| F-008 | IV-surface features versus the hold-to-expiry result of a QQQ iron condor, by DTE bucket | pre-registered | yes |
| F-009 | The F-008 condors scored with a 50%-of-credit take-profit (exploratory) | exploratory | no |

## Study results

### F-008  IV-surface features versus the hold-to-expiry result of a QQQ iron condor, by DTE bucket
- Status: pre-registered        Citable: yes
- Question: Do the variance risk premium at the condor's horizon (`vrp_h`), the 7-to-30-day ATM term slope (`term_slope`) and the normalised 25-delta risk reversal (`rr25_norm`) predict the hold-to-expiry result of a QQQ iron condor after friction, beyond IV rank and the ATM IV level, in each DTE bucket?
- Rule: PLAN.md "W118 AS PLANNED" as amended by "W118 ADDENDUM 1" (both written before any outcome was computed)
- Study file / script: studies/surface_feature_study.yaml / scripts/surface_feature_study.py --run
- Run record: reports/runs/study_surface_feature_study_surface_feature_study_20261101_171235/   Commit: 38e82e3   Data span: 2024-10-01 to 2026-10-20 (warm-up to 2025-03-31, six quarterly test folds 2025-04-01 to 2026-10-20)   Epoch: surface IV fitted from the stored QQQ option bars; per-leg friction bands (the 0-2 DTE band extrapolated); hold-to-expiry intrinsic settlement; the QQQ option-bar backfill was still adding expiries after the window end during the run
- Result: one synthetic condor per (session, target DTE), short strikes at |delta| 0.20, wing 0.35 x expected move, scored from real option-bar closes. Verdicts: bucket A (1-2 DTE) NOT SHOWN, bucket B (3-10 DTE) NOT SHOWN, bucket C (14-30 DTE) INSUFFICIENT. No feature passes. Spearman rho with the result per unit of risk (90% block-bootstrap interval; Holm family of 6 tests):

  | bucket (n_eff) | feature | rho | 90% interval | p raw | p Holm | partial rho (p) | folds with expected sign |
  |---|---|---|---|---|---|---|---|
  | A (274.2) | `vrp_h` | +0.090 | [+0.015, +0.165] | 0.022 | 0.135 | +0.045 (0.178) | 6 / 6 |
  | A | `term_slope` | +0.045 | [-0.031, +0.122] | 0.330 | 1.000 | +0.018 (0.738) | 4 / 6 |
  | A | `rr25_norm` | -0.078 | [-0.155, +0.002] | 0.107 | 0.535 | -0.021 (0.622) | 5 / 6 |
  | B (77.3) | `vrp_h` | -0.086 | [-0.161, -0.000] | 0.950 | 1.000 | -0.043 (0.768) | 2 / 6 |
  | B | `term_slope` | -0.046 | [-0.122, +0.042] | 0.417 | 1.000 | -0.037 (0.653) | 5 / 6 |
  | B | `rr25_norm` | -0.024 | [-0.150, +0.086] | 0.673 | 1.000 | +0.025 (0.757) | 4 / 6 |
  | C (24.8, descriptive only) | `vrp_h` | -0.089 | [-0.231, +0.059] | 0.856 | - | +0.049 (0.490) | 3 / 6 |
  | C | `term_slope` | -0.205 | [-0.303, -0.092] | 0.002 | - | -0.183 (0.026) | 5 / 6 |
  | C | `rr25_norm` | -0.117 | [-0.251, +0.020] | 0.174 | - | -0.040 (0.554) | 6 / 6 |

  `vrp_h` is one-sided (pre-stated direction positive), the others two-sided. Mean result per unit of risk at 1x friction: A -0.084, B -0.003, C +0.048. Wiring checks all passed (as-of mutation check; leaked canary significant in A and B; shuffled canary 3 of 20 in each; data-defect share 3.0% / 2.1% / 3.1%).
- Caveats: About 18 months of test sessions, one underlying, one regime. Rows of adjacent sessions overlap (hence the block bootstrap, and `n_eff` counted in distinct sessions); bucket C is below the floor of 30 independent observations, so its figures, including the `term_slope` interval that excludes zero, are descriptive and are not a finding. Bucket A friction is extrapolated for 86.7% of entries. Settlement is intrinsic, with no assignment or pin modelling. The 0.35 wing, the 0.03 smile-fit limit and the 25% void limits are chosen values. The option-bar backfill was writing during the run: it did not change any entry or outcome, and an exploratory re-computation with the surface cut at the window end moved no verdict and no figure by more than 0.02 in rho (bucket C `term_slope` -0.211 against -0.205). A near miss, not a pass: `vrp_h` at 1-2 DTE has the expected sign in every fold, but the partial correlation controlling for IV rank and ATM IV level is +0.045.
- Fed decision: none (a rerun of the same rule is registered for when bucket C reaches `n_eff` 30)      Superseded by: -

### F-009  The F-008 condors scored with a 50%-of-credit take-profit (exploratory)
- Status: exploratory        Citable: no
- Question: Do the F-008 conclusions change if each condor is closed early once it can be bought back for half its credit, instead of held to expiry?
- Rule: exploratory (written and run after F-008; it is not part of the F-008 rule and cannot change its verdicts)
- Study file / script: scripts/surface_feature_study.py --exploratory-tp (no study file, no run record)
- Run record: none   Commit: the W118 follow-up entry in PLAN.md   Data span: as F-008 (test rows 2025-04-01 to 2026-10-20)   Epoch: as F-008, with the QQQ option-bar backfill still running
- Result: the same entries as F-008, closed on the first later session on which all four legs have a traded bar and the condor's mark is at or below 50% of the credit, otherwise held to expiry; exit friction and commissions charged at that session. Marks are real bar closes; a session with a leg missing is skipped and nothing is modelled. Test rows:

  | bucket | target reached | rows with no priced session | mean result per unit of risk, hold | managed, 1x friction | managed, 2x friction |
  |---|---|---|---|---|---|
  | A (1-2 DTE) | 6.9% | 62.9% | -0.084 | -0.071 | -0.127 |
  | B (3-10 DTE) | 58.8% | 0.0% | -0.003 | -0.0004 | -0.034 |
  | C (14-30 DTE) | 74.4% | 0.1% | +0.048 | +0.022 | -0.0004 |

  Spearman rho of each F-008 feature with the managed result (90% block-bootstrap interval): A `vrp_h` +0.084 [+0.008, +0.159], `term_slope` +0.031, `rr25_norm` -0.085 [-0.162, -0.008]; B `vrp_h` -0.068, `term_slope` -0.026, `rr25_norm` +0.004 (every interval spans zero); C `vrp_h` -0.033, `term_slope` -0.101 [-0.203, -0.010], `rr25_norm` -0.072. Bucket C `term_slope` is -0.205 with hold-to-expiry (F-008).
- Caveats: Exploratory and post-hoc, chosen after the F-008 result was known, with no multiplicity control; not a finding. A leg that did not trade is unpriced, so the target can be missed (the reached rate is a lower bound), and for 1-2 DTE there is at most one session between entry and expiry, which is why most bucket A rows have no priced session and the managed result there is mostly the hold result. Taking profit at 50% lowers the mean result in bucket C in this sample. Bucket C remains below the floor of 30 independent observations (F-008), so its figures are descriptive.
- Fed decision: none      Superseded by: -

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
