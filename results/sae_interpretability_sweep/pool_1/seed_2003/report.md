# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01552 |
| top_features | 39.2% | -0.00038 | 0.01589 |
| random_control_1 | 38.8% | -0.00589 | 0.02141 |
| random_control_2 | 22.4% | -0.00029 | 0.01581 |
| random_control_3 | 42.1% | -0.00782 | 0.02334 |
| random_control_4 | 14.5% | -0.00035 | 0.01586 |
| random_control_5 | 50.5% | -0.00995 | 0.02546 |
| random_control_6 | 11.2% | -0.00021 | 0.01573 |
| random_control_7 | 27.4% | -0.00423 | 0.01975 |
| random_control_8 | 23.7% | -0.00060 | 0.01612 |
| random_control_9 | 42.3% | -0.00786 | 0.02338 |
| random_control_10 | 22.0% | -0.00137 | 0.01688 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
