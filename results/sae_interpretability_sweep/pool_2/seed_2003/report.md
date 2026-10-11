# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00321 |
| top_features | 7.7% | 0.00040 | 0.00281 |
| random_control_1 | 10.0% | 0.00008 | 0.00312 |
| random_control_2 | 8.1% | 0.00025 | 0.00296 |
| random_control_3 | 25.9% | -0.00425 | 0.00746 |
| random_control_4 | 7.3% | 0.00028 | 0.00293 |
| random_control_5 | 25.1% | -0.00294 | 0.00615 |
| random_control_6 | 11.0% | -0.00039 | 0.00360 |
| random_control_7 | 9.2% | 0.00012 | 0.00309 |
| random_control_8 | 5.1% | -0.00016 | 0.00337 |
| random_control_9 | 4.5% | 0.00011 | 0.00310 |
| random_control_10 | 18.0% | -0.00236 | 0.00557 |

Removed top feature IDs: 735, 549, 205

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
