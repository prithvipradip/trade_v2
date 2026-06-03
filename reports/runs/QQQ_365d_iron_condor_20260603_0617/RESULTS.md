# Integration Test Results

**Date**: 2026-06-03 06:17 UTC
**Symbols**: QQQ
**Intraday history**: 3 years of 5-min bars from IB
**Database**: `data/integration_test.db` (tables: `test_intraday_prices`, `test_daily_prices`)
**Total run time**: 335.2 minutes

---

## A. Data Quality

- **QQQ** ✅ PASS
  - Bars: 58,602 | Sessions: 755
  - Date range: 2023-05-22 → 2026-05-29
  - Coverage: 99.5% (threshold: ≥ 80%)
  - Sparse sessions (< 60 bars): 8
  - Price anomalies: neg_close=0, high<low=0

---

## B. Feature Health (18 features: 5 fractal + 13 VLMC)

### QQQ
- Missing features: None ✅
- High NaN (> 5%): None ✅

---

## C. IC Decay Curve

**1-day horizon**: 0/18 features significant (p<0.05) | best IC: psd_beta = +0.0608
**3-day horizon**: 0/18 features significant (p<0.05) | best IC: psd_beta = +0.0920
**5-day horizon**: 0/18 features significant (p<0.05) | best IC: psd_beta = +0.1000
**10-day horizon**: 5/18 features significant (p<0.05) | best IC: hurst_wavelet = +0.1518
**20-day horizon**: 3/18 features significant (p<0.05) | best IC: hurst_wavelet = +0.3107

---

## D. Walk-Forward (multi-strategy, optimize_per_window=True)

⚠️ WARN
- Windows: 10
- Total trades: 25
- Total return: -1.4%
- Sharpe ratio: -2.264 (threshold: > -1.0)
- Max drawdown: 2.7%
- Win rate: 56.0%
- Consistency: 40.0% profitable windows

---

## E. Ablation (baseline: no optimization)

- Sharpe (optimized): -2.264 vs. baseline: 0.623 → delta: -2.886
- Return (optimized): -1.4% vs. baseline: +0.7%
- Win rate (optimized): 56.0% vs. baseline: 44.4%
- ℹ️ Per-window optimization did not improve Sharpe in this test run.

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
| IC significance at 5d | ≥ 3 features p<0.10 | ✅ PASS |
| Walk-forward trades | > 0 | ✅ PASS |
| Walk-forward Sharpe | > -1.0 | ⚠️ WARN |