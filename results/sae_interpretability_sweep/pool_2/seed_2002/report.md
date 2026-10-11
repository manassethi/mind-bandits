# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00374 |
| top_features | 31.2% | -0.00517 | 0.00890 |
| random_control_1 | 8.3% | 0.00008 | 0.00365 |
| random_control_2 | 10.9% | -0.00037 | 0.00410 |
| random_control_3 | 11.8% | 0.00060 | 0.00313 |
| random_control_4 | 9.2% | 0.00057 | 0.00317 |
| random_control_5 | 13.5% | 0.00025 | 0.00349 |
| random_control_6 | 24.6% | -0.00146 | 0.00519 |
| random_control_7 | 3.3% | 0.00010 | 0.00364 |
| random_control_8 | 21.6% | -0.00082 | 0.00455 |
| random_control_9 | 9.4% | -0.00019 | 0.00393 |
| random_control_10 | 6.2% | -0.00080 | 0.00453 |

Removed top feature IDs: 872, 526, 1015

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
