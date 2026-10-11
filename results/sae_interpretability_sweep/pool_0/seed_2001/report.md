# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00803 |
| top_features | 47.7% | -0.01473 | 0.02277 |
| random_control_1 | 19.7% | -0.00142 | 0.00945 |
| random_control_2 | 22.9% | -0.00413 | 0.01216 |
| random_control_3 | 11.7% | -0.00113 | 0.00916 |
| random_control_4 | 22.1% | -0.00155 | 0.00958 |
| random_control_5 | 20.2% | -0.00092 | 0.00895 |
| random_control_6 | 14.1% | -0.00119 | 0.00922 |
| random_control_7 | 13.4% | -0.00097 | 0.00900 |
| random_control_8 | 22.1% | -0.00410 | 0.01213 |
| random_control_9 | 10.1% | -0.00108 | 0.00911 |
| random_control_10 | 5.1% | -0.00061 | 0.00864 |

Removed top feature IDs: 439, 470, 915

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
