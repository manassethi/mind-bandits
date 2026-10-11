# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00471 |
| top_features | 31.1% | -0.00990 | 0.01462 |
| random_control_1 | 22.6% | -0.00184 | 0.00655 |
| random_control_2 | 17.1% | -0.00012 | 0.00483 |
| random_control_3 | 6.2% | -0.00039 | 0.00511 |
| random_control_4 | 18.9% | -0.00013 | 0.00484 |
| random_control_5 | 15.8% | -0.00045 | 0.00517 |
| random_control_6 | 9.8% | 0.00001 | 0.00471 |
| random_control_7 | 19.6% | -0.00024 | 0.00496 |
| random_control_8 | 11.9% | -0.00115 | 0.00586 |
| random_control_9 | 10.1% | -0.00062 | 0.00534 |
| random_control_10 | 12.6% | -0.00168 | 0.00639 |

Removed top feature IDs: 741, 496, 372

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
