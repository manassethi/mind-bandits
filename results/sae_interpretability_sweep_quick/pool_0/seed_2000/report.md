# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01827 |
| top_features | 36.8% | 0.00312 | 0.01514 |
| random_control_1 | 29.2% | 0.00296 | 0.01531 |
| random_control_2 | 10.8% | 0.00019 | 0.01808 |
| random_control_3 | 24.6% | 0.00427 | 0.01400 |

Removed top feature IDs: 439, 470, 915

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
