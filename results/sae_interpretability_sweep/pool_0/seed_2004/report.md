# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01163 |
| top_features | 45.1% | -0.01220 | 0.02383 |
| random_control_1 | 18.4% | -0.00022 | 0.01185 |
| random_control_2 | 20.8% | 0.00037 | 0.01127 |
| random_control_3 | 39.8% | -0.00715 | 0.01879 |
| random_control_4 | 24.3% | -0.00116 | 0.01279 |
| random_control_5 | 28.4% | 0.00036 | 0.01127 |
| random_control_6 | 4.7% | -0.00058 | 0.01222 |
| random_control_7 | 24.6% | -0.00080 | 0.01244 |
| random_control_8 | 21.0% | 0.00010 | 0.01154 |
| random_control_9 | 6.8% | -0.00007 | 0.01171 |
| random_control_10 | 25.1% | 0.00097 | 0.01066 |

Removed top feature IDs: 439, 470, 735

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
