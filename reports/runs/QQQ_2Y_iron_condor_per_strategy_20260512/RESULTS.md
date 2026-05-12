# Integration Test Results

**Date**: 2026-05-11 08:29 UTC
**Symbols**: QQQ
**Intraday history**: 2 years of 5-min bars from IB
**Database**: `data/integration_test.db` (tables: `test_intraday_prices`, `test_daily_prices`)
**Total run time**: 354.4 minutes

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

**1-day horizon**: 0/5 features significant (p<0.05) | best IC: multifractal_width = -0.0176
**3-day horizon**: 1/5 features significant (p<0.05) | best IC: hurst_wavelet = -0.1015
**5-day horizon**: 1/5 features significant (p<0.05) | best IC: hurst_wavelet = -0.1376
**10-day horizon**: 2/5 features significant (p<0.05) | best IC: hurst_wavelet = -0.1776
**20-day horizon**: 2/5 features significant (p<0.05) | best IC: multifractal_asymmetry = +0.2901

---

## D. Walk-Forward (multi-strategy, optimize_per_window=True)

✅ PASS
- Windows: 7
- Total trades: 33
- Total return: +29.1%
- Sharpe ratio: 43.035 (threshold: > -1.0)
- Max drawdown: 0.0%
- Win rate: 97.0%
- Consistency: 100.0% profitable windows

---

## E. Ablation (baseline: no optimization)

- Sharpe (optimized): 43.035 vs. baseline: 8.726 → delta: +34.309
- Return (optimized): +29.1% vs. baseline: +14.6%
- Win rate (optimized): 97.0% vs. baseline: 66.7%
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