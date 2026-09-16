| variant | n | win% | net $ | PF | avg $ | max DD $ | DD % | hold d | net $/pos-day | touch | TP | backtest_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live_ladder | 57 | 52.6 | -4,020 | 0.583 | -70.53 | 4,993 | 4.96 | 5.67 | -12.409 | 22 | 26 | 6 |
| live_ladder_explicit | 57 | 52.6 | -4,020 | 0.583 | -70.53 | 4,993 | 4.96 | 5.67 | -12.409 | 22 | 26 | 6 |
| flat_40 | 56 | 50.0 | -2,203 | 0.765 | -39.34 | 3,866 | 3.81 | 6.38 | -6.154 | 23 | 24 | 6 |
| flat_50 | 56 | 48.2 | -2,261 | 0.769 | -40.37 | 3,615 | 3.58 | 6.77 | -5.95 | 24 | 18 | 6 |
| flat_60 | 56 | 48.2 | -1,682 | 0.828 | -30.04 | 3,473 | 3.43 | 6.91 | -4.336 | 24 | 15 | 6 |
| ladder_plus_10 | 56 | 50.0 | -2,203 | 0.765 | -39.34 | 3,866 | 3.81 | 6.38 | -6.154 | 23 | 24 | 6 |
| ladder_14dte_50 | 57 | 52.6 | -4,020 | 0.583 | -70.53 | 4,993 | 4.96 | 5.67 | -12.409 | 22 | 26 | 6 |
| no_take_profit | 56 | 48.2 | -1,492 | 0.848 | -26.64 | 3,470 | 3.42 | 6.98 | -3.806 | 24 | 0 | 6 |

**Verdict:** CANDIDATE no_take_profit (PF 0.848 vs 0.583); live change needs operator approval and starts a new sample epoch

- sanity: explicit ladder reproduces the default run exactly
- flat_40: QUALIFIES
- flat_50: QUALIFIES
- flat_60: QUALIFIES
- ladder_plus_10: QUALIFIES
- ladder_14dte_50: fails pf>base+0.10
- no_take_profit: QUALIFIES
