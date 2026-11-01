# Q1 results (test-day pools; settings tuned on train-day pools)

Regret = expected clicks lost vs always showing the best arm, summed over the run. ± is a 95% t-interval over runs; Regret − Thompson is paired within runs.
### K2: K=2, T=100,000, drift=none

Best arm CTR 0.0437; personalized oracle CTR 0.0474 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.02` | 39 ± 29 | 0.0433 | 20 ± 24 | 93% | 93% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 95 ± 59 | 0.0428 | 76 ± 58 | 93% | 93% |
| ε-greedy | `egreedy_0.01` | 101 ± 73 | 0.0427 | 82 ± 72 | 89% |  |
| UCB1 | `ucb1_0.1` | 10 ± 4 | 0.0436 | -9 ± 7 | 97% |  |
| Thompson sampling | `thompson` | 19 ± 9 | 0.0435 | 0 ± 0 | 99% |  |
| Discounted Thompson | `dthompson_0.99999` | 21 ± 5 | 0.0435 | 2 ± 7 | 97% |  |
| Uniform (never-ending A/B) | `uniform` | 1,218 ± 451 | 0.0315 | 1,199 ± 453 | 50% |  |

### K5: K=5, T=100,000, drift=none

Best arm CTR 0.0509; personalized oracle CTR 0.0623 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.02` | 104 ± 64 | 0.0499 | 38 ± 63 | 83% | 83% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 341 ± 54 | 0.0475 | 276 ± 60 | 87% | 87% |
| ε-greedy | `egreedy_decay_100.0` | 147 ± 76 | 0.0494 | 82 ± 78 | 80% |  |
| UCB1 | `ucb1_0.1` | 43 ± 12 | 0.0505 | -22 ± 12 | 92% |  |
| Thompson sampling | `thompson` | 66 ± 9 | 0.0503 | 0 ± 0 | 92% |  |
| Discounted Thompson | `dthompson_0.99999` | 94 ± 11 | 0.0500 | 29 ± 9 | 91% |  |
| Uniform (never-ending A/B) | `uniform` | 2,372 ± 743 | 0.0272 | 2,306 ± 747 | 20% |  |

### K10: K=10, T=100,000, drift=none

Best arm CTR 0.0862; personalized oracle CTR 0.1059 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.05` | 379 ± 121 | 0.0824 | 251 ± 124 | 80% | 80% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 1,203 ± 273 | 0.0742 | 1,075 ± 281 | 93% | 93% |
| ε-greedy | `egreedy_decay_10.0` | 426 ± 227 | 0.0819 | 299 ± 228 | 73% |  |
| UCB1 | `ucb1_0.1` | 66 ± 23 | 0.0855 | -61 ± 22 | 96% |  |
| Thompson sampling | `thompson` | 128 ± 12 | 0.0849 | 0 ± 0 | 96% |  |
| Discounted Thompson | `dthompson_0.99999` | 187 ± 15 | 0.0843 | 59 ± 14 | 91% |  |
| Uniform (never-ending A/B) | `uniform` | 5,480 ± 2,001 | 0.0314 | 5,353 ± 2,008 | 10% |  |

### K20: K=20, T=100,000, drift=none

Best arm CTR 0.0936; personalized oracle CTR 0.1213 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.05` | 739 ± 370 | 0.0862 | 474 ± 373 | 73% | 73% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 3,327 ± 547 | 0.0603 | 3,061 ± 559 | 93% | 93% |
| ε-greedy | `egreedy_decay_30.0` | 333 ± 94 | 0.0903 | 67 ± 103 | 89% |  |
| UCB1 | `ucb1_0.1` | 132 ± 84 | 0.0923 | -133 ± 88 | 95% |  |
| Thompson sampling | `thompson` | 265 ± 24 | 0.0910 | 0 ± 0 | 93% |  |
| Discounted Thompson | `dthompson_0.99999` | 371 ± 18 | 0.0899 | 106 ± 23 | 93% |  |
| Uniform (never-ending A/B) | `uniform` | 6,515 ± 1,513 | 0.0285 | 6,250 ± 1,524 | 5% |  |

### K50: K=50, T=100,000, drift=none

Best arm CTR 0.1304; personalized oracle CTR 0.1692 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.05` | 1,728 ± 785 | 0.1132 | 1,128 ± 788 | 60% | 60% |
| A/B test (power analysis) | `ab_power_mde_0.2` | 10,151 ± 1,548 | 0.0289 | 9,551 ± 1,571 | 2% | 0% (only 0% finished) |
| ε-greedy | `egreedy_decay_30.0` | 995 ± 245 | 0.1205 | 395 ± 255 | 85% |  |
| UCB1 | `ucb1_0.1` | 138 ± 28 | 0.1291 | -461 ± 20 | 97% |  |
| Thompson sampling | `thompson` | 600 ± 27 | 0.1244 | 0 ± 0 | 94% |  |
| Discounted Thompson | `dthompson_0.99999` | 871 ± 23 | 0.1217 | 271 ± 17 | 89% |  |
| Uniform (never-ending A/B) | `uniform` | 10,149 ± 1,547 | 0.0290 | 9,550 ± 1,571 | 2% |  |

### K10_T1k: K=10, T=1,000, drift=none

Best arm CTR 0.0862; personalized oracle CTR 0.1059 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.1` | 29 ± 14 | 0.0570 | 1 ± 12 | 30% | 30% |
| A/B test (power analysis) | `ab_power_mde_0.2` | 55 ± 20 | 0.0313 | 27 ± 17 | 10% | 0% (only 0% finished) |
| ε-greedy | `egreedy_0.1` | 26 ± 9 | 0.0606 | -3 ± 8 | 40% |  |
| UCB1 | `ucb1_0.1` | 17 ± 4 | 0.0687 | -11 ± 5 | 61% |  |
| Thompson sampling | `thompson` | 28 ± 5 | 0.0578 | 0 ± 0 | 41% |  |
| Discounted Thompson | `dthompson_0.9999` | 29 ± 5 | 0.0574 | 0 ± 2 | 47% |  |
| Uniform (never-ending A/B) | `uniform` | 55 ± 20 | 0.0315 | 26 ± 17 | 10% |  |

### K10_T10k: K=10, T=10,000, drift=none

Best arm CTR 0.0862; personalized oracle CTR 0.1059 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.2` | 179 ± 56 | 0.0683 | 99 ± 56 | 53% | 53% |
| A/B test (power analysis) | `ab_power_mde_0.2` | 548 ± 200 | 0.0313 | 469 ± 201 | 10% | 0% (only 0% finished) |
| ε-greedy | `egreedy_decay_30.0` | 90 ± 29 | 0.0771 | 11 ± 30 | 78% |  |
| UCB1 | `ucb1_0.1` | 47 ± 16 | 0.0815 | -33 ± 16 | 86% |  |
| Thompson sampling | `thompson` | 80 ± 6 | 0.0782 | 0 ± 0 | 77% |  |
| Discounted Thompson | `dthompson_0.99999` | 80 ± 5 | 0.0782 | 1 ± 6 | 81% |  |
| Uniform (never-ending A/B) | `uniform` | 548 ± 200 | 0.0314 | 468 ± 201 | 10% |  |

### K10_abrupt: K=10, T=100,000, drift=abrupt

Best arm CTR 0.0854; personalized oracle CTR 0.1020 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.02` | 3,410 ± 1,404 | 0.0513 | 2,050 ± 1,112 | 3% | 63% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 4,429 ± 1,664 | 0.0411 | 3,069 ± 1,369 | 3% | 93% |
| ε-greedy | `egreedy_0.05` | 2,995 ± 1,370 | 0.0555 | 1,635 ± 1,083 | 24% |  |
| UCB1 | `ucb1_0.3` | 629 ± 202 | 0.0791 | -731 ± 261 | 85% |  |
| Thompson sampling | `thompson` | 1,360 ± 367 | 0.0718 | 0 ± 0 | 72% |  |
| Discounted Thompson | `dthompson_0.9999` | 770 ± 59 | 0.0777 | -590 ± 322 | 72% |  |
| Uniform (never-ending A/B) | `uniform` | 5,679 ± 2,427 | 0.0286 | 4,319 ± 2,130 | 10% |  |

### K10_gradual: K=10, T=100,000, drift=gradual

Best arm CTR 0.0748; personalized oracle CTR 0.0962 (contextual headroom).

| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | Plays on best arm, last window | A/B picked the best arm |
|---|---|---|---|---|---|---|
| A/B test (tuned length) | `ab_frac_0.05` | 1,601 ± 397 | 0.0477 | 958 ± 315 | 3% | 63% |
| A/B test (power analysis) | `ab_power_mde_0.5` | 1,943 ± 536 | 0.0442 | 1,301 ± 403 | 20% | 73% |
| ε-greedy | `egreedy_decay_100.0` | 1,177 ± 424 | 0.0519 | 534 ± 297 | 20% |  |
| UCB1 | `ucb1_0.1` | 649 ± 270 | 0.0572 | 6 ± 188 | 64% |  |
| Thompson sampling | `thompson` | 642 ± 192 | 0.0572 | 0 ± 0 | 53% |  |
| Discounted Thompson | `dthompson_0.99999` | 444 ± 67 | 0.0592 | -198 ± 153 | 74% |  |
| Uniform (never-ending A/B) | `uniform` | 3,326 ± 982 | 0.0304 | 2,684 ± 851 | 10% |  |
