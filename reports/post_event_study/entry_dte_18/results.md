| variant | n | win% | net $ | PF | avg $ | max DD $ | DD % | hold d | net $/pos-day | touch | TP | removed (P&L) | added (P&L) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_buffer | 57 | 50.9 | -3,521 | 0.628 | -61.77 | 4,671 | 4.32 | 6.96 | -8.802 | 25 | 25 | - | - |
| no_buffer_dates_only | 57 | 50.9 | -3,521 | 0.628 | -61.77 | 4,671 | 4.32 | 6.96 | -8.802 | 25 | 25 | - | - |
| post_fomc_1d | 54 | 51.9 | -3,424 | 0.627 | -63.41 | 4,574 | 4.22 | 7.19 | -8.78 | 24 | 25 | 3 (-96.67) | 0 (0) |
| post_fomc_2d | 53 | 50.9 | -2,980 | 0.659 | -56.23 | 4,068 | 3.75 | 7.23 | -7.741 | 23 | 25 | 6 (-748.21) | 2 (-207.67) |
| post_fomc_3d | 53 | 50.9 | -2,980 | 0.659 | -56.23 | 4,068 | 3.75 | 7.23 | -7.741 | 23 | 25 | 6 (-748.21) | 2 (-207.67) |

**Verdict:** NO BUFFER (no variant clears every condition)

- sanity: dates without a window reproduce the baseline exactly
- take-profit study live_ladder (advisory, different data window): n 51 vs 57 here, pf 0.445 vs 0.628 here, net -5441.86 vs -3520.99 here, max_dd_dollars 5995.38 vs 4670.72 here
- post_fomc_1d: fails pf>base+0.10
- post_fomc_2d: fails pf>base+0.10
- post_fomc_3d: fails pf>base+0.10
