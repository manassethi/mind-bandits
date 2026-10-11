# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.02631 |
| top_features | 23.4% | -0.00467 | 0.03099 |
| random_control_1 | 1.2% | -0.00031 | 0.02663 |
| random_control_2 | 13.6% | -0.00026 | 0.02657 |
| random_control_3 | 5.6% | 0.00058 | 0.02573 |

Removed top feature IDs: 439, 470, 735

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
