# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00489 |
| top_features | 26.7% | -0.00545 | 0.01033 |
| random_control_1 | 11.6% | -0.00100 | 0.00589 |
| random_control_2 | 11.2% | 0.00008 | 0.00481 |
| random_control_3 | 13.7% | -0.00006 | 0.00494 |
| random_control_4 | 11.6% | 0.00016 | 0.00473 |
| random_control_5 | 10.3% | 0.00089 | 0.00400 |
| random_control_6 | 13.1% | -0.00092 | 0.00581 |
| random_control_7 | 9.9% | 0.00068 | 0.00421 |
| random_control_8 | 14.9% | 0.00009 | 0.00480 |
| random_control_9 | 16.4% | -0.00063 | 0.00552 |
| random_control_10 | 12.2% | 0.00037 | 0.00452 |

Removed top feature IDs: 1015, 992, 750

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
