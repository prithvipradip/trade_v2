# Claim ledger: docs/MODELS.md (after fixes)

Generated tables (catalogue, windows, labels, hyperparameters, meta-label, regime, drift, gates) are built from the schema and config.yaml by scripts/gen_docs.py. The prose was checked as below.

- Tree models only, no deep-learning framework | pyproject extras, no torch import in src | OK
- Model table (direction, range, vol-magnitude, meta-labeler, regime, drift) and live/research use | ensemble.py, range_predictor.py, vol_magnitude_predictor.py, meta_label.py, regime.py, drift.py | OK
- Direction and range models are per-symbol with a fallback | ensemble.py:97-147, 395-408; range_predictor.py:141-150,306 | OK
- Range weights fitted from CV edge | range_predictor.py (earlier reading) | OK
- Switches, windows (27 window-bearing columns), constraints | feature_names.py, settings.py validators, feature_spec.py | OK
- Spec hash covers windows and non-default switches only | feature_spec.py:65-78 | OK
- Saved model files, versions, spec record, mismatch table | ml/trainer.py, model identifiers checked by tests | OK
- Retrain: own IB session, IB bars only, aborts before write, restart marker | retrain_ib.py:14, master.py:1546,1567,1655 | OK
- Retrain due conditions (untrained, spec mismatch, interval, drift) | trainer.py needs_training:60-90 | OK
- Meta-labeler 20 inputs, guards | meta_label settings text, min_feature_coverage description | OK
- Regime: seven regimes, rule-based, entry override / thesis check / record | regime.py:27-34, orchestrator | OK
- Regime IV-rank proxy | regime.py:175-197 | FIXED (it is a min-max rank over the window, not a percentile)
- Drift: outcomes recorded only for directional trades | orchestrator.py:1262-1285 | OK
- Observe vs enforce table | orchestrator.py:2470-2500 | OK
- Changing a model setting table | settings/feature_spec behaviour | OK
- Setting descriptions in generated tables free of history | provenance test | FIXED (cleaned by gen_docs overrides)
- Range label band and horizon | ml/range_spec.py:1-40, settings ml.range_threshold_pct | FIXED (names the setting: a fixed 5% by default; horizon is the reachable hold dte_range[0] minus the days before expiry at which a credit structure is closed)
