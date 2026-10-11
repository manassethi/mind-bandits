# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00817 |
| top_features | 26.8% | -0.00486 | 0.01304 |
| random_control_1 | 31.0% | -0.00274 | 0.01092 |
| random_control_2 | 13.4% | -0.00128 | 0.00945 |
| random_control_3 | 18.0% | -0.00319 | 0.01136 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
