# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00350 |
| top_features | 44.3% | -0.01834 | 0.02184 |
| random_control_1 | 19.6% | -0.00381 | 0.00731 |
| random_control_2 | 19.0% | -0.00081 | 0.00431 |
| random_control_3 | 47.2% | -0.01970 | 0.02320 |
| random_control_4 | 27.7% | -0.00801 | 0.01151 |
| random_control_5 | 18.1% | -0.00341 | 0.00691 |
| random_control_6 | 40.8% | -0.00462 | 0.00812 |
| random_control_7 | 32.5% | -0.00318 | 0.00668 |
| random_control_8 | 21.2% | -0.00103 | 0.00453 |
| random_control_9 | 14.8% | -0.00034 | 0.00384 |
| random_control_10 | 8.9% | 0.00009 | 0.00341 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
