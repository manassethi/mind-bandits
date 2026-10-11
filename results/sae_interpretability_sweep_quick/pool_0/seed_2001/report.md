# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01509 |
| top_features | 15.6% | -0.00117 | 0.01625 |
| random_control_1 | 12.8% | 0.00079 | 0.01430 |
| random_control_2 | 8.0% | -0.00015 | 0.01523 |
| random_control_3 | 4.2% | 0.00037 | 0.01471 |

Removed top feature IDs: 439, 470, 372

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
