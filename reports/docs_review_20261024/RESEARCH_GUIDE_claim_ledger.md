# Claim ledger: docs/RESEARCH_GUIDE.md (after fixes)

- Provenance table source tags (IB_TRADES_RTH, IB_OPTION_IV, IB_INDEX_TRADES, IB_BID_ASK_BARS, YAHOO_DIVIDENDS, INVESTING_CSV, EDGAR/FED/BLS tags) | data/historical.py:37-49 | OK
- Default windows 365 / 5 / 63 / 63 | settings.py:757-766 | OK
- show_research_config flags | scripts/show_research_config.py:28-31 | OK
- Layers (config, overlay, study) and what a study may not set | CLAUDE.md, study harness (earlier reading) | OK
- Engine: realistic, pricing sources, friction model, slippage only for debit structures | config_research.yaml, engine.py:2463 | OK
- Range models per horizon (horizon_for, RANGE_HORIZON_BUCKETS) | ml/range_spec.py:65 | OK
- Learner numbers (reduce_min_change 0.03 stalls a protected condor at 0.3277, boost/reenable minimums inert, caps) | config_research.yaml learner block | OK
- Objective weights 0.4 / 0.4 / 0.2; hard minimum 3 trades; graded floor | config_research.yaml:130-132,196-200 | OK
- DTE search: flags exist; top-decile in two thirds of windows | dte_report.py:8-12, flag-existence check | OK
- Study rules and run record contents | run_record.py (feature_list present), study files | OK
- Option-bar history span | historical.db read-only: 2024-09-30 to 2026-10-20 | OK, reworded as volatile
- `run_optimizer.py --apply` | run_optimizer.py:228-245, optimization/results.py:81-123 | FIXED (also needs AIT_ALLOW_APPLY=1; four parameters; rewrites config as plain YAML, dropping comments)
- Short-DTE collection (flags, ACTIVE bands, status file, supervisor job) | calibrate_option_spreads.py flags, short_dte_job.py:34, master.py:1767-1792 | OK
- Parity replay inputs/outputs | scripts/parity_replay.py (earlier reading) | OK
- One-sided gates, gate_parity registry | backtesting/gate_parity.py, config_research.yaml:64-107 | OK
- IV-rank rise veto off in research | config_research.yaml:64 | OK
- Ex-dividend block | dividends.py:55 project_next_ex_div, config_research.yaml:107 | OK
- min_windows semantics | settings.py:801, run_backtest.py:550 | FIXED
- A flatness-study result in prose | n/a | FIXED (moved to STUDIES_FINDINGS pointer only)
- "Default run looks poor" phrasing | n/a | FIXED (removed a results claim)

Second pass: 10.1 IB pacing 60 requests per 10 minutes | calibrate_option_spreads.py:157 | OK; the "40-55 min per symbol" figure could not be confirmed | FIXED (reworded to "tens of minutes"). 10.2 outputs under data/replay/out/<YYYYMMDD>, summary.md statistics only, trades.csv/legs.csv local, refusal to write outside data/replay | parity_replay.py:10-12,52-72,159-160 | OK.
