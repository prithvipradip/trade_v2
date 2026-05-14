# Integration Test Results

**Date**: 2026-05-14 02:21 UTC
**Symbols**: QQQ
**Intraday history**: 2 years of 5-min bars from IB
**Database**: `data/integration_test.db` (tables: `test_intraday_prices`, `test_daily_prices`)
**Total run time**: 553.5 minutes

---

## A. Data Quality

- **QQQ** ✅ PASS
  - Bars: 38,376 | Sessions: 495
  - Date range: 2024-05-02 → 2026-05-08
  - Coverage: 99.4% (threshold: ≥ 80%)
  - Sparse sessions (< 60 bars): 6
  - Price anomalies: neg_close=0, high<low=0

---

## B. Feature Health (18 features: 5 fractal + 13 VLMC)

### QQQ
- Missing features: None ✅
- High NaN (> 5%): None ✅

---

## C. IC Decay Curve

**1-day horizon**: 1/5 features significant (p<0.05) | best IC: multifractal_asymmetry = +0.1234
**3-day horizon**: 1/5 features significant (p<0.05) | best IC: multifractal_asymmetry = +0.2211
**5-day horizon**: 2/5 features significant (p<0.05) | best IC: multifractal_asymmetry = +0.2706
**10-day horizon**: 2/5 features significant (p<0.05) | best IC: multifractal_asymmetry = +0.2320
**20-day horizon**: 1/5 features significant (p<0.05) | best IC: multifractal_width = +0.2998

---

## D. Walk-Forward (multi-strategy, optimize_per_window=True)

✅ PASS
- Windows: 2
- Total trades: 8
- Total return: +5.5%
- Sharpe ratio: 24.169 (threshold: > -1.0)
- Max drawdown: 0.1%
- Win rate: 75.0%
- Consistency: 50.0% profitable windows

---

## E. Ablation (baseline: no optimization)

- Sharpe (optimized): 24.169 vs. baseline: 5.961 → delta: +18.208
- Return (optimized): +5.5% vs. baseline: +26.8%
- Win rate (optimized): 75.0% vs. baseline: 64.9%
- ✅ Per-window optimization improves Sharpe over baseline.

---

## F. Output Files

| File | Description |
|------|-------------|
| `data_quality.txt` | Coverage, gaps, price sanity per symbol |
| `feature_health.txt` | NaN rates and ranges for all 18 features |
| `ic_decay.csv` | Spearman IC at 1/3/5/10/20-day horizons |
| `walkforward_summary.txt` | Walk-forward metrics and per-window breakdown |
| `equity_curve.csv` | Trade-level equity curve from walk-forward |
| `ablation_summary.txt` | Baseline walk-forward (no optimization) |
| `fractal_report_QQQ.html` | Interactive fractal + VLMC diagnostic plots |
| `ic_summary.csv` | Aggregated IC from diagnostic report pipeline |
| `RESULTS.md` | This file |

---

## G. Pass/Fail Summary

| Check | Criterion | Status |
|-------|-----------|--------|
| Data coverage | ≥ 80% | ✅ PASS |
| No negative prices | 0 anomalies | ✅ PASS |
| All 18 features present | 0 missing | ✅ PASS |
| Feature NaN rate | < 10% | ✅ PASS |
| IC significance at 5d | ≥ 3 features p<0.10 | ⚠️ WARN |
| Walk-forward trades | > 0 | ✅ PASS |
| Walk-forward Sharpe | > -1.0 | ✅ PASS |