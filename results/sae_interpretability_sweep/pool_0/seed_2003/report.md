# SAE policy sensitivity

One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.

| Ablation | Choices changed | Expected CTR change | Expected regret per round |
|---|---:|---:|---:|
| baseline | 0.0% | 0.00000 | 0.00921 |
| top_features | 30.6% | -0.00753 | 0.01674 |
| random_control_1 | 22.9% | -0.00367 | 0.01288 |
| random_control_2 | 8.5% | -0.00086 | 0.01006 |
| random_control_3 | 13.2% | -0.00196 | 0.01117 |
| random_control_4 | 6.7% | -0.00029 | 0.00949 |
| random_control_5 | 9.2% | 0.00031 | 0.00889 |
| random_control_6 | 6.9% | 0.00023 | 0.00897 |
| random_control_7 | 15.7% | -0.00226 | 0.01146 |
| random_control_8 | 5.6% | -0.00087 | 0.01008 |
| random_control_9 | 21.5% | -0.00270 | 0.01191 |
| random_control_10 | 4.9% | -0.00045 | 0.00966 |

Removed top feature IDs: 439, 470, 205

Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.
