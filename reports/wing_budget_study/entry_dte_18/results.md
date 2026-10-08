| variant | cap $ | n | win% | net $ | PF | max DD $ | avg win | avg loss | worst | touch | TP | wing $ | max loss ~$ | credit | cap bound % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_1lot | - | 55 | 54.5 | -776 | 0.836 | 1,678 | 131.37 | -188.67 | -505.06 | 25 | 27 | 25.34 | 2195.0 | 3.39 | 0.0 |
| cap_huge_1lot | 10000000.0 | 55 | 54.5 | -776 | 0.836 | 1,678 | 131.37 | -188.67 | -505.06 | 25 | 27 | 25.34 | 2195.0 | 3.39 | 0.0 |
| live_3pct | 5962.0 | 55 | 54.5 | -776 | 0.836 | 1,678 | 131.37 | -188.67 | -505.06 | 25 | 27 | 25.34 | 2195.0 | 3.39 | 0.0 |
| ahmed_1_33pct | 2643.0 | 57 | 54.4 | -1,236 | 0.751 | 1,625 | 120.06 | -190.69 | -465.36 | 26 | 27 | 22.41 | 1913.0 | 3.282 | 33.3 |
| user_1pct | 1990.0 | 57 | 54.4 | -1,494 | 0.687 | 1,612 | 105.75 | -183.53 | -408.84 | 26 | 27 | 19.32 | 1630.0 | 3.025 | 50.9 |

**Verdict:** ahmed_1_33pct ACCEPTABLE; user_1pct COSTLY

- sanity: a non-binding cap reproduces the uncapped 1-lot run exactly
- ahmed_1_33pct: ACCEPTABLE
- user_1pct: COSTLY (fails pf>=base-0.1)
