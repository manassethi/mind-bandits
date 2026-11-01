# Q2 results (test-day pools; settings and layers chosen on train-day pools)

## Main comparison (article lifetime 2000, 20 live articles, 20,000 rounds, d = 32)

Regret is vs the per-user oracle. Cold-start regret: rounds where the user's best live article is in its first 10% of life. ± = 95% CI over 30 runs; Δ columns are paired.

| Representation | Policy | Regret | Cold-start regret / round | CTR | Δ vs no features |
|---|---|---|---|---|---|
| No features (per-article TS) | `beta_ts` | 1,338 ± 28 | 0.0562 ± 0.0016 | 0.0577 | 0 ± 0 |
| Category one-hot | `greedy` | 1,845 ± 114 | 0.0922 ± 0.0055 | 0.0323 | 507 ± 120 |
| Category one-hot | `linucb_0.3` **(main)** | 1,490 ± 100 | 0.0769 ± 0.0051 | 0.0501 | 152 ± 106 |
| Category one-hot | `lints_0.1` | 1,551 ± 105 | 0.0813 ± 0.0051 | 0.0470 | 213 ± 111 |
| bge-m3 (XLM-R family) | `greedy` | 1,080 ± 103 | 0.0615 ± 0.0046 | 0.0706 | -257 ± 117 |
| bge-m3 (XLM-R family) | `linucb_0.3` **(main)** | 816 ± 58 | 0.0429 ± 0.0030 | 0.0838 | -522 ± 68 |
| bge-m3 (XLM-R family) | `lints_0.1` | 841 ± 66 | 0.0484 ± 0.0032 | 0.0825 | -497 ± 76 |
| Qwen3 embeddings (simulator's encoder: ceiling) | `greedy` | 1,040 ± 95 | 0.0584 ± 0.0045 | 0.0726 | -298 ± 101 |
| Qwen3 embeddings (simulator's encoder: ceiling) | `linucb_0.3` **(main)** | 760 ± 48 | 0.0407 ± 0.0026 | 0.0866 | -578 ± 48 |
| Qwen3 embeddings (simulator's encoder: ceiling) | `lints_0.1` | 791 ± 42 | 0.0460 ± 0.0029 | 0.0850 | -547 ± 42 |
| Gemma-3-1B dense (layer 23) | `greedy` | 1,032 ± 91 | 0.0588 ± 0.0047 | 0.0730 | -306 ± 91 |
| Gemma-3-1B dense (layer 23) | `linucb_0.3` **(main)** | 753 ± 36 | 0.0408 ± 0.0020 | 0.0869 | -585 ± 39 |
| Gemma-3-1B dense (layer 23) | `lints_0.1` | 810 ± 62 | 0.0481 ± 0.0037 | 0.0841 | -528 ± 65 |
| Gemma-3-1B SAE (layer 17) | `greedy` | 1,299 ± 114 | 0.0706 ± 0.0062 | 0.0596 | -39 ± 120 |
| Gemma-3-1B SAE (layer 17) | `linucb_0.1` **(main)** | 920 ± 55 | 0.0515 ± 0.0029 | 0.0786 | -418 ± 47 |
| Gemma-3-1B SAE (layer 17) | `lints_0.1` | 945 ± 65 | 0.0545 ± 0.0032 | 0.0773 | -393 ± 69 |
| Gemma-3-12B dense (layer 43) | `greedy` | 1,073 ± 97 | 0.0607 ± 0.0049 | 0.0709 | -265 ± 98 |
| Gemma-3-12B dense (layer 43) | `linucb_0.3` **(main)** | 754 ± 44 | 0.0400 ± 0.0022 | 0.0869 | -584 ± 42 |
| Gemma-3-12B dense (layer 43) | `lints_0.1` | 801 ± 54 | 0.0468 ± 0.0028 | 0.0845 | -537 ± 53 |
| Gemma-3-12B SAE (layer 12) | `greedy` | 1,253 ± 111 | 0.0685 ± 0.0048 | 0.0619 | -85 ± 117 |
| Gemma-3-12B SAE (layer 12) | `linucb_0.3` | 927 ± 84 | 0.0477 ± 0.0044 | 0.0782 | -411 ± 81 |
| Gemma-3-12B SAE (layer 12) | `lints_0.1` **(main)** | 921 ± 64 | 0.0527 ± 0.0036 | 0.0785 | -417 ± 72 |

## Cost of interpretability: SAE − dense regret (same model, paired)

| Model | d | Dense regret | SAE regret | SAE − dense |
|---|---|---|---|---|
| g1b | 16 | 858 | 1,093 | 236 ± 102 |
| g1b | 32 | 753 | 920 | 168 ± 55 |
| g1b | 64 | 673 | 892 | 219 ± 55 |
| g1b | 128 | 669 | 930 | 261 ± 94 |
| g12b | 16 | 861 | 1,034 | 172 ± 79 |
| g12b | 32 | 754 | 921 | 168 ± 67 |
| g12b | 64 | 689 | 893 | 204 ± 86 |
| g12b | 128 | 669 | 881 | 212 ± 57 |

## Article churn: regret by article lifetime (d = 32, main policy)

| Representation | life 500 | life 2000 | life 10000 |
|---|---|---|---|
| No features (per-article TS) | 1,716 ± 32 | 1,338 ± 28 | 955 ± 49 |
| Category one-hot | 1,532 ± 51 | 1,490 ± 100 | 1,307 ± 190 |
| bge-m3 (XLM-R family) | 920 ± 26 | 816 ± 58 | 563 ± 44 |
| Qwen3 embeddings (simulator's encoder: ceiling) | 863 ± 23 | 760 ± 48 | 522 ± 38 |
| Gemma-3-1B dense (layer 23) | 894 ± 33 | 753 ± 36 | 529 ± 33 |
| Gemma-3-1B SAE (layer 17) | 1,037 ± 48 | 920 ± 55 | 845 ± 134 |
| Gemma-3-12B dense (layer 43) | 889 ± 30 | 754 ± 44 | 550 ± 45 |
| Gemma-3-12B SAE (layer 12) | 1,098 ± 42 | 921 ± 64 | 686 ± 66 |
