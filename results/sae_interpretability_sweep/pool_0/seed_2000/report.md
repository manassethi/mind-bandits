# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01333 |
| top_features | 25.3% | -0.00028 | 0.01361 |
| random_control_1 | 7.4% | -0.00007 | 0.01340 |
| random_control_2 | 11.8% | 0.00035 | 0.01299 |
| random_control_3 | 42.5% | -0.00521 | 0.01854 |
| random_control_4 | 18.9% | 0.00033 | 0.01300 |
| random_control_5 | 13.4% | 0.00051 | 0.01282 |
| random_control_6 | 23.5% | 0.00065 | 0.01268 |
| random_control_7 | 16.8% | 0.00064 | 0.01269 |
| random_control_8 | 10.8% | 0.00018 | 0.01315 |
| random_control_9 | 4.7% | 0.00001 | 0.01332 |
| random_control_10 | 9.1% | 0.00010 | 0.01323 |

Removed top feature IDs: 439, 470, 372

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
