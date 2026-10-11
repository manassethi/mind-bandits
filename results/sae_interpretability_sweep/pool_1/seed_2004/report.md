# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00374 |
| top_features | 41.8% | -0.01285 | 0.01659 |
| random_control_1 | 4.9% | -0.00045 | 0.00419 |
| random_control_2 | 13.4% | 0.00022 | 0.00352 |
| random_control_3 | 44.4% | -0.01481 | 0.01855 |
| random_control_4 | 16.4% | -0.00279 | 0.00653 |
| random_control_5 | 49.0% | -0.01668 | 0.02043 |
| random_control_6 | 2.9% | -0.00035 | 0.00409 |
| random_control_7 | 12.7% | -0.00126 | 0.00500 |
| random_control_8 | 23.6% | -0.00265 | 0.00639 |
| random_control_9 | 7.3% | 0.00026 | 0.00348 |
| random_control_10 | 31.1% | -0.00440 | 0.00815 |

Removed top feature IDs: 439, 470, 489

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
