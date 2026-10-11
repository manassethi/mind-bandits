# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00535 |
| top_features | 42.6% | -0.00857 | 0.01391 |
| random_control_1 | 47.7% | -0.00929 | 0.01463 |
| random_control_2 | 16.6% | -0.00201 | 0.00736 |
| random_control_3 | 14.6% | -0.00054 | 0.00588 |
| random_control_4 | 20.2% | -0.00297 | 0.00832 |
| random_control_5 | 29.6% | -0.00487 | 0.01021 |
| random_control_6 | 13.4% | -0.00031 | 0.00566 |
| random_control_7 | 10.4% | 0.00011 | 0.00524 |
| random_control_8 | 10.9% | -0.00175 | 0.00709 |
| random_control_9 | 14.8% | -0.00180 | 0.00715 |
| random_control_10 | 33.9% | -0.00765 | 0.01300 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
