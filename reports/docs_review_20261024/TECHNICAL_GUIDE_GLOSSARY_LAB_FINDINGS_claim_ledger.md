# Claim ledger: TECHNICAL_GUIDE, GLOSSARY, WALKFORWARD_DASHBOARD, STUDIES_FINDINGS (after fixes)

## TECHNICAL_GUIDE
- Package ownership and single-source definitions exist where named | named-file test; strategies/base.py, go_live.py, runtime_env.py, ml/regime_gate.py, data/iv_rank.py, backtesting/friction.py, config/feature_names.py | OK
- resolve_config_value in settings.py:3146; contract_str/float/flag in runtime_env.py:191-215; ET via utils.tz re-export | grep | OK
- Generator refuses an undocumented job (JOB_TEXT) | gen_docs.py | OK
- Test fails on a setting with no reader | tests/test_w70_wing_floor_and_research_parity.py:441 | OK
- CI lanes and cron 06:00 UTC | .github/workflows/ci.yml:29-33,86,111 | OK
- Feature-spec hash statement | feature_spec.py:65-78,145 | FIXED
- Literal census workflow | scripts/literal_census.py docstring | FIXED (it classifies new entries; check the category)
- Confirmation list | CLAUDE.md (moved verbatim in substance) | OK
- ruff config E,F,I,N,W,UP | pyproject.toml:87-90 | OK

## GLOSSARY (project-specific entries only)
- IV rank | data/iv_rank.py | FIXED (percentile of 252 readings, min 60)
- Gap stress | orchestrator.py:5510-5525 | FIXED (-10% at intrinsic)
- Direction ensemble, vol-magnitude, regime (seven regimes), drift (live track record, off by default) | MODELS sources above | FIXED
- Heartbeat / watchdog | orchestrator.py:1560-1585, 473-516 | FIXED (split into two entries)
- Friction (half-spread plus commission and fees) | RESEARCH_GUIDE section 5 | FIXED
- Circuit breaker, PDT, HALT files, outbox, reconciler | RUNBOOK/GUIDE sources | OK
- Touch stop | portfolio.py:306-354 | OK (reworded to "reaches a short strike")
- VLMC | feature_names.py:18-31 | OK (the code does not expand the acronym; entry says so by describing the block)

## WALKFORWARD_DASHBOARD
- Input files and producer (run_integration_test archive) | export.py:237-260,435-500; run_integration_test.py:1030-1050 | OK
- build_multi, CLI flags, default view | export.py:991-1134 | OK
- Trades table columns, filters | components.jsx:196-262 | FIXED
- KPI strips (run and window) | components.jsx:28-66 | FIXED
- Drawer sections; missing gate renders as a green pass | panels.jsx:36-189 | FIXED
- Markers, strike lines, zoom, controls, Hurst 0.5 line | charts.js:254-258,280-281,326-346; app.jsx | OK
- Optuna tab elements | optuna.jsx | OK
- Predictor tiles; gating flag computed by the exporter at fixed 0.10; fixed model captions | predictor-models.jsx:566-573, export.py:670,803,919-952 | FIXED
- Header: objective not recorded by the archive; confidence floors never displayed | components.jsx:78, run_integration_test.py:1030-1048 | FIXED (code and text)
- Brier metrics not shown by the page | grep | FIXED (row removed)
- Sentiment pane offered only when data has values | components.jsx, app.jsx | FIXED (code)

## STUDIES_FINDINGS
- F-001..F-003 quotes appear verbatim in the previous RUNBOOK | git show HEAD:docs/RUNBOOK.md | OK
- F-004 quotes the comment at portfolio.py:600-605 | read | OK
- F-005, F-006 quotes appear (split across lines) in the previous RUNBOOK | git show HEAD:docs/RUNBOOK.md lines 300-306, 328 | OK
- F-007 quotes the description of ml.range_min_confidence | settings.py:1916-1918 | OK
- None re-run or re-validated: all stay "migrated, unreviewed, not citable"
