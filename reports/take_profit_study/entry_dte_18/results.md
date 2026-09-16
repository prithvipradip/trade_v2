| variant | n | win% | net $ | PF | avg $ | max DD $ | DD % | hold d | net $/pos-day | touch | TP | backtest_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live_ladder | 51 | 47.1 | -5,442 | 0.445 | -106.7 | 5,995 | 5.96 | 7.65 | -13.918 | 24 | 20 | 6 |
| live_ladder_explicit | 51 | 47.1 | -5,442 | 0.445 | -106.7 | 5,995 | 5.96 | 7.65 | -13.918 | 24 | 20 | 6 |
| flat_40 | 50 | 46.0 | -4,552 | 0.536 | -91.04 | 5,172 | 5.14 | 7.94 | -11.438 | 24 | 19 | 6 |
| flat_50 | 47 | 44.7 | -3,595 | 0.629 | -76.48 | 4,272 | 4.24 | 8.64 | -8.832 | 23 | 15 | 5 |
| flat_60 | 46 | 45.7 | -2,189 | 0.76 | -47.59 | 2,866 | 2.85 | 9.02 | -5.262 | 22 | 13 | 6 |
| ladder_plus_10 | 50 | 46.0 | -4,506 | 0.541 | -90.12 | 5,126 | 5.09 | 7.98 | -11.265 | 24 | 19 | 6 |
| ladder_14dte_50 | 51 | 47.1 | -5,396 | 0.45 | -105.8 | 5,949 | 5.92 | 7.69 | -13.73 | 24 | 20 | 6 |
| no_take_profit | 46 | 45.7 | -2,221 | 0.757 | -48.28 | 2,898 | 2.88 | 9.09 | -5.301 | 22 | 0 | 6 |

**Verdict:** CANDIDATE flat_60 (PF 0.76 vs 0.445); live change needs operator approval and starts a new sample epoch

- sanity: explicit ladder reproduces the default run exactly
- flat_40: fails pf>base+0.10
- flat_50: QUALIFIES
- flat_60: QUALIFIES
- ladder_plus_10: fails pf>base+0.10
- ladder_14dte_50: fails pf>base+0.10
- no_take_profit: QUALIFIES
