# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.01003 |
| top_features | 36.2% | -0.00129 | 0.01132 |
| random_control_1 | 19.2% | -0.00271 | 0.01274 |
| random_control_2 | 13.6% | -0.00304 | 0.01307 |
| random_control_3 | 33.2% | -0.00146 | 0.01149 |

Removed top feature IDs: 439, 470, 724

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
