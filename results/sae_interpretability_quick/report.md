# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01011 |
| top_features | 29.0% | -0.00969 | 0.01980 |
| random_control_1 | 23.8% | -0.00193 | 0.01205 |
| random_control_2 | 13.6% | 0.00091 | 0.00920 |
| random_control_3 | 20.0% | -0.00245 | 0.01256 |
| random_control_4 | 28.0% | -0.00348 | 0.01359 |
| random_control_5 | 14.2% | 0.00056 | 0.00955 |
| random_control_6 | 32.6% | -0.00723 | 0.01734 |
| random_control_7 | 15.6% | 0.00031 | 0.00980 |
| random_control_8 | 13.8% | 0.00025 | 0.00986 |
| random_control_9 | 12.4% | 0.00019 | 0.00992 |
| random_control_10 | 15.6% | 0.00033 | 0.00978 |

Removed top feature IDs: 762, 451, 58

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
