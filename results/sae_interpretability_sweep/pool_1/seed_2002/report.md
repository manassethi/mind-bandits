# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00384 |
| top_features | 43.0% | -0.01346 | 0.01729 |
| random_control_1 | 5.2% | 0.00009 | 0.00374 |
| random_control_2 | 8.3% | -0.00036 | 0.00419 |
| random_control_3 | 7.3% | -0.00035 | 0.00418 |
| random_control_4 | 12.4% | -0.00136 | 0.00520 |
| random_control_5 | 11.6% | -0.00003 | 0.00386 |
| random_control_6 | 14.6% | -0.00143 | 0.00526 |
| random_control_7 | 11.6% | 0.00000 | 0.00383 |
| random_control_8 | 25.3% | -0.00387 | 0.00771 |
| random_control_9 | 4.7% | 0.00065 | 0.00319 |
| random_control_10 | 21.8% | -0.00162 | 0.00545 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
