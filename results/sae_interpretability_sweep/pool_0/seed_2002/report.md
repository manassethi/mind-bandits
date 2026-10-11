# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01123 |
| top_features | 26.9% | -0.00476 | 0.01600 |
| random_control_1 | 0.9% | -0.00038 | 0.01161 |
| random_control_2 | 13.0% | -0.00057 | 0.01180 |
| random_control_3 | 6.2% | -0.00038 | 0.01161 |
| random_control_4 | 9.1% | 0.00024 | 0.01100 |
| random_control_5 | 13.3% | -0.00088 | 0.01212 |
| random_control_6 | 17.7% | -0.00074 | 0.01197 |
| random_control_7 | 7.4% | -0.00062 | 0.01186 |
| random_control_8 | 22.7% | -0.00222 | 0.01345 |
| random_control_9 | 3.1% | 0.00010 | 0.01113 |
| random_control_10 | 14.2% | -0.00061 | 0.01185 |

Removed top feature IDs: 439, 470, 38

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
