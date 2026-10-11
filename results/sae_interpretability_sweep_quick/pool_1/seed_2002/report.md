# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00515 |
| top_features | 37.6% | -0.00912 | 0.01427 |
| random_control_1 | 0.4% | -0.00002 | 0.00517 |
| random_control_2 | 4.8% | -0.00014 | 0.00530 |
| random_control_3 | 2.2% | -0.00030 | 0.00545 |

Removed top feature IDs: 439, 470, 735

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
