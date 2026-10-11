# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00336 |
| top_features | 16.7% | -0.00070 | 0.00406 |
| random_control_1 | 18.3% | -0.00078 | 0.00414 |
| random_control_2 | 9.7% | -0.00100 | 0.00436 |
| random_control_3 | 5.5% | 0.00008 | 0.00328 |
| random_control_4 | 24.2% | -0.00372 | 0.00708 |
| random_control_5 | 31.4% | -0.00391 | 0.00727 |
| random_control_6 | 10.8% | -0.00021 | 0.00357 |
| random_control_7 | 16.1% | -0.00227 | 0.00562 |
| random_control_8 | 11.8% | -0.00004 | 0.00340 |
| random_control_9 | 12.8% | 0.00088 | 0.00248 |
| random_control_10 | 14.9% | -0.00007 | 0.00343 |

Removed top feature IDs: 470, 38, 489

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
