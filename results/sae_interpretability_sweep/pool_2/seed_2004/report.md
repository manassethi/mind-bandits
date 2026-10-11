# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00683 |
| top_features | 17.2% | 0.00023 | 0.00660 |
| random_control_1 | 19.0% | -0.00275 | 0.00957 |
| random_control_2 | 10.8% | 0.00052 | 0.00631 |
| random_control_3 | 12.7% | -0.00037 | 0.00720 |
| random_control_4 | 14.5% | -0.00189 | 0.00872 |
| random_control_5 | 7.0% | 0.00038 | 0.00645 |
| random_control_6 | 5.1% | 0.00065 | 0.00618 |
| random_control_7 | 21.9% | -0.00310 | 0.00993 |
| random_control_8 | 14.4% | -0.00105 | 0.00788 |
| random_control_9 | 15.8% | -0.00144 | 0.00827 |
| random_control_10 | 6.0% | 0.00080 | 0.00602 |

Removed top feature IDs: 470, 38, 489

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
